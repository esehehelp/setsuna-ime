//! Reranker: a causal LM's log-probability of each conversion candidate
//! after the left context, for the IME to add to the converter's score
//! (`score = converter + lambda * lm`).
//!
//! The model loads only through an LM scorer manifest
//! (`setsuna_artifact::LmScorerManifest`); its tokenizer is the manifest's
//! SentencePiece model when it names one, else the one in the weights file.
//! A candidate is scored as `[BOS] + tokens(context + candidate)` from where
//! the candidates part from `[BOS] + tokens(context)`, summing its tokens'
//! log-probabilities.
//!
//!   - trie: the candidates as one token trie
//!   - scorer: the trie decoded with llama.cpp, the cache kept across requests
//!   - spm: a SentencePiece Unigram tokenizer (the LM's own segmentation)
//!   - protocol: the JSONL requests and responses of `daemon` and `serve`
//!   - server: `serve`, one reranker for every app of a login over a named
//!     pipe (Windows)

pub mod protocol;
pub mod scorer;
#[cfg(windows)]
pub mod server;
pub mod spm;
pub mod trie;

use std::path::Path;

use anyhow::{Context, Result};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use setsuna_artifact::LmScorerManifest;

pub use scorer::{Stats, MAX_SEQ};

/// Settings a caller may give; unset ones come from the manifest's
/// `score_defaults`, then the defaults here.
#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    pub threads: Option<usize>,
    pub max_context_chars: Option<usize>,
    pub max_candidates: Option<usize>,
    pub lambda: Option<f32>,
    /// Tokens llama.cpp computes in one pass. Default: `scorer::N_UBATCH`.
    pub n_ubatch: Option<u32>,
}

/// The resolved settings of a reranker.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RerankConfig {
    pub threads: usize,
    pub max_context_chars: usize,
    pub max_candidates: usize,
    /// Weight of the LM score for the caller; the reranker itself only scores.
    pub lambda: f32,
}

pub struct Reranker {
    model: &'static LlamaModel,
    backend: &'static LlamaBackend,
    /// The llama.cpp context (KV cache and compute buffers); `None` once
    /// released, made again by the next request.
    scorer: Option<scorer::TreeScorer>,
    n_ubatch: u32,
    bos: i32,
    /// the manifest's SentencePiece tokenizer, else llama.cpp's vocabulary
    spm: Option<spm::SentencePiece>,
    pub config: RerankConfig,
}

/// Load llama.cpp's CPU backends: next to the executable when they were
/// installed there, else from where the build put them.
fn load_backends() {
    let installed = std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent()?.to_path_buf();
        let has = std::fs::read_dir(&dir).ok()?.flatten().any(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.starts_with("ggml-cpu") || name.starts_with("libggml-cpu")
        });
        has.then_some(dir)
    });
    match installed {
        Some(dir) => llama_cpp_2::llama_backend::load_backends_from_path(&dir),
        None => llama_cpp_2::llama_backend::load_backends(),
    }
}

impl Reranker {
    pub fn load(manifest: &LmScorerManifest, options: &RerankOptions) -> Result<Self> {
        let d = &manifest.score_defaults;
        let config = RerankConfig {
            // one: an IME's rerank runs on one core, not on every core to
            // make it faster
            threads: options.threads.unwrap_or(1),
            max_context_chars: options
                .max_context_chars
                .or(d.max_context_chars)
                .unwrap_or(40),
            max_candidates: options
                .max_candidates
                .or(d.max_candidates)
                .unwrap_or(20)
                .clamp(1, MAX_SEQ),
            lambda: options.lambda.or(d.lambda).unwrap_or(0.3),
        };
        load_backends();
        let backend: &'static LlamaBackend =
            Box::leak(Box::new(LlamaBackend::init().context("init llama.cpp")?));
        let model = LlamaModel::load_from_file(
            backend,
            &manifest.weights_path,
            &LlamaModelParams::default(),
        )
        .with_context(|| format!("load {}", manifest.weights_path.display()))?;
        // the daemon keeps one model for its whole life
        let model: &'static LlamaModel = Box::leak(Box::new(model));
        let bos = manifest.special_ids.bos as i32;
        let spm = match &manifest.tokenizer_path {
            Some(p) => Some(spm::SentencePiece::load(p)?),
            None => None,
        };
        let n_ubatch = options.n_ubatch.unwrap_or(scorer::N_UBATCH);
        let scorer = scorer::TreeScorer::new(model, backend, config.threads, n_ubatch)?;
        Ok(Self {
            model,
            backend,
            scorer: Some(scorer),
            n_ubatch,
            bos,
            spm,
            config,
        })
    }

    /// Free the llama.cpp context, its KV cache and compute buffers, keeping
    /// the model; the next request makes a new one (and decodes its context
    /// again). False when there was none.
    pub fn release(&mut self) -> bool {
        self.scorer.take().is_some()
    }

    pub fn load_path(path: &Path, options: &RerankOptions) -> Result<Self> {
        let manifest = LmScorerManifest::load(path)?;
        Self::load(&manifest, options)
    }

    /// `[BOS] + tokens(text)`, no other special token added or parsed.
    pub fn tokens(&self, text: &str) -> Vec<i32> {
        let mut ids = vec![self.bos];
        match &self.spm {
            Some(sp) => ids.extend(sp.encode(text).into_iter().map(|t| t as i32)),
            None => ids.extend(
                self.model
                    .vocab()
                    .tokenize(text.as_bytes(), false, false)
                    .into_iter()
                    .map(|t| t.0),
            ),
        }
        ids
    }

    /// The LM log-probability of each candidate after `context` (its last
    /// `max_context_chars` chars), at most `max_candidates` of them. A
    /// request for the candidates of the one before plus more, after the
    /// same context, decodes only the new ones.
    pub fn score(&mut self, context: &str, candidates: &[String]) -> Result<(Vec<f64>, Stats)> {
        anyhow::ensure!(
            candidates.len() <= self.config.max_candidates,
            "at most {} candidates, got {}",
            self.config.max_candidates,
            candidates.len()
        );
        if candidates.is_empty() {
            return Ok((Vec::new(), Stats::default()));
        }
        let n = context.chars().count();
        let context: String = context
            .chars()
            .skip(n.saturating_sub(self.config.max_context_chars))
            .collect();
        let seqs: Vec<Vec<i32>> = candidates
            .iter()
            .map(|c| self.tokens(&format!("{context}{c}")))
            .collect();
        let start = trie::scoring_start(&seqs, &self.tokens(&context));
        let scorer = match &mut self.scorer {
            Some(s) => s,
            None => self.scorer.insert(scorer::TreeScorer::new(
                self.model,
                self.backend,
                self.config.threads,
                self.n_ubatch,
            )?),
        };
        scorer.score(&seqs, start)
    }
}
