//! Engine session: a CTC-NAT artifact loaded through its manifest, its
//! resolved decode settings, and the entry point that runs the conversion
//! pipeline (`crate::pipeline`) for one request.
//!
//! The process environment is read in exactly one place, `RuntimeEnv::read`,
//! when a session is created.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{bail, Result};
use setsuna_artifact::{CtcNatManifest, DecodeDefaults};

use crate::candidate::Candidate;
use crate::pipeline::proposal::{FrameReduce, ProposalRunner};
use crate::pipeline::{
    artifact_load, chunk, decode, digitsplit, pool, proposal, rescore, symsplit, symtoken,
};
use crate::tokenizer::SharedCharTokenizer;

#[derive(Debug, Clone, Copy, Default)]
pub struct ConvertPerf {
    pub input_len: usize,
    pub proposal_ms: f64,
    pub beam_ms: f64,
    pub post_ms: f64,
    /// The rescoring of the candidates (`pipeline::rescore`).
    pub rescore_ms: f64,
    pub total_ms: f64,
    pub hyp_count: usize,
    pub candidate_count: usize,
}

/// One conversion: a hiragana reading and its left context.
#[derive(Debug, Clone, Copy)]
pub struct ConversionRequest<'a> {
    pub context: &'a str,
    pub reading: &'a str,
    /// 0 for a conversion; n > 0 asks for more candidates of the same
    /// request: the beam and the candidate limit widened by
    /// `CONTINUE_STEP * n`, over the proposal already computed for it.
    pub continue_beam: usize,
}

/// How much each `continue_beam` step widens the beam and the candidates.
pub const CONTINUE_STEP: usize = 10;

/// Candidates after the display limit, best first, with the timings of the
/// conversion that produced them.
#[derive(Debug, Clone)]
pub struct CandidateSet {
    pub candidates: Vec<Candidate>,
    pub perf: ConvertPerf,
    /// The model's output frame by frame, when [`EngineSession::inspect`]
    /// asks for it (and the reading was converted in one pass, not split
    /// at symbols).
    pub frames: Option<Vec<FrameView>>,
}

/// One output frame of the model, for inspection: the input token it sits
/// on, the blank's probability and the most probable other tokens.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FrameView {
    /// The input token (`[CLS]`, a context char, `[SEP]`, a reading char;
    /// a byte-fallback token as `<0xE3>`).
    pub input: String,
    /// The reading char the frame holds; `None` for `[CLS]`, the context
    /// and `[SEP]`.
    pub reading_index: Option<u32>,
    /// Probability of emitting nothing here.
    pub blank: f32,
    /// The best non-blank tokens and their probabilities, best first.
    pub top: Vec<(String, f32)>,
}

/// The environment variables the engine honours. Everything else comes from
/// the caller and the manifest.
#[derive(Debug, Clone)]
struct RuntimeEnv {
    /// `SETSUNA_ORT_INTRA_THREADS` (default 1: an IME's conversion runs on
    /// one core; more cores are for other work, not to make one faster).
    ort_intra_threads: usize,
    /// `SETSUNA_TOP_K_PER_STEP` (default 16): beam expansion breadth when
    /// neither the caller nor the manifest sets one.
    top_k_per_step: usize,
}

impl RuntimeEnv {
    fn read() -> Self {
        Self {
            ort_intra_threads: env_usize("SETSUNA_ORT_INTRA_THREADS", 1),
            top_k_per_step: env_usize("SETSUNA_TOP_K_PER_STEP", 16),
        }
    }
}

/// Resolved decode settings of a session.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Prefix-beam width; 1 means true CTC greedy collapse.
    pub beam_width: usize,
    pub top_k_per_step: usize,
    /// Maximum returned candidates.
    pub max_candidates: usize,
    /// Maximum left-context tokens fed to the model (a char the vocabulary
    /// lacks takes one a byte).
    pub max_context: usize,
    /// Token ids whose log-prob is forced to -inf before decoding.
    pub vocab_blacklist: Vec<u32>,
    /// Convert bracket / annotation symbols' kana spans separately and splice
    /// the symbols back (see `pipeline::symsplit`).
    pub symbol_split: bool,
    /// Read each run of symbols in the reading as one `◇` token, in one pass,
    /// and put the runs back into the candidates (see `pipeline::symtoken`).
    /// Wins over `symbol_split`: the reading is not split.
    pub symbol_token: bool,
    /// Cut the runs of ASCII digits out of the reading and splice them back
    /// verbatim; the other segments are converted after them (see
    /// `pipeline::digitsplit`). Applies before symbol token / symbol split,
    /// which then handle each other segment.
    pub digit_split: bool,
    /// Map the candidates' punctuation and full-width ASCII the IME way
    /// (`normalize_candidate`); off, a candidate is the model's output as
    /// decoded (the research default: what training and the bench compare).
    pub normalize_candidates: bool,
    /// Weights of the rescoring's two-character and word terms
    /// (`pipeline::rescore`); 0 leaves a term out (and its table unloaded).
    pub pair_weight: f32,
    pub word_weight: f32,
}

impl DecodeConfig {
    /// Every knob: `explicit` value > the manifest's `decode_defaults` > the
    /// engine default.
    fn resolve(explicit: &DecodeDefaults, d: &DecodeDefaults, env: &RuntimeEnv) -> Self {
        Self {
            beam_width: explicit.beam_width.or(d.beam_width).unwrap_or(1).max(1),
            // 0 would underflow the per-step top-K selection in prefix beam.
            top_k_per_step: explicit
                .top_k_per_step
                .or(d.top_k_per_step)
                .map(|tk| tk.max(1))
                .unwrap_or(env.top_k_per_step),
            max_candidates: explicit.top_k.or(d.top_k).unwrap_or(5).max(1),
            max_context: explicit.max_context.or(d.max_context).unwrap_or(64),
            vocab_blacklist: explicit
                .vocab_blacklist
                .clone()
                .or_else(|| d.vocab_blacklist.clone())
                .unwrap_or_default(),
            symbol_split: explicit.symbol_split.or(d.symbol_split).unwrap_or(false),
            symbol_token: explicit.symbol_token.or(d.symbol_token).unwrap_or(false),
            digit_split: explicit.digit_split.or(d.digit_split).unwrap_or(false),
            normalize_candidates: explicit
                .normalize_candidates
                .or(d.normalize_candidates)
                .unwrap_or(false),
            pair_weight: explicit.pair_weight.or(d.pair_weight).unwrap_or(0.0),
            word_weight: explicit.word_weight.or(d.word_weight).unwrap_or(0.0),
        }
    }

    /// Candidates kept per span before the display cut: the whole beam, so a
    /// symbol-split re-splice sees every hypothesis.
    fn pool_limit(&self) -> usize {
        self.beam_width.max(self.max_candidates)
    }
}

pub struct EngineSession {
    pub tokenizer: SharedCharTokenizer,
    proposal: ProposalRunner,
    /// Positional-table size of the model (manifest `max_seq_len`).
    pub seq_len: usize,
    pub config: DecodeConfig,
    /// How the proposal reduces a frame.
    reduce: FrameReduce,
    /// Tokens per frame [`CandidateSet::frames`] lists; 0 (the default)
    /// leaves the frames out.
    pub inspect: usize,
    /// Candidates of the chunks of long readings, by (context, reading,
    /// beam width).
    chunk_cache: HashMap<(String, String, usize), Vec<Candidate>>,
    /// The last span's proposal: a `continue_beam` request (or the same
    /// request again) decodes it again without running the model.
    last_proposal: Option<LastProposal>,
    /// The model's CTC-CRF transitions (manifest `crf`), added by the beam.
    crf: Option<crate::pipeline::crf::CrfTables>,
    /// The candidates' rescoring (manifest `rescore`), when a weight is set.
    rescorer: Option<rescore::Rescorer>,
}

/// A proposal with what it was computed for.
struct LastProposal {
    context: String,
    reading: String,
    greedy: bool,
    inspect: usize,
    proposal: proposal::ProposalOutput,
    frame_reading: Vec<u32>,
    inputs: Vec<i64>,
}

/// Chunk results [`EngineSession`] keeps before starting over.
const CHUNK_CACHE: usize = 64;

impl EngineSession {
    /// Load the artifact a CTC-NAT manifest names and resolve every decode
    /// knob as: `explicit` value > the manifest's `decode_defaults` > engine
    /// default.
    pub fn from_manifest(manifest: &CtcNatManifest, explicit: &DecodeDefaults) -> Result<Self> {
        let env = RuntimeEnv::read();
        let (tokenizer, proposal) = artifact_load::load(manifest, env.ort_intra_threads)?;
        let config = DecodeConfig::resolve(explicit, &manifest.decode_defaults, &env);
        if config.symbol_token {
            let mut buf = [0u8; 4];
            let token: &str = symtoken::SYMBOL_TOKEN.encode_utf8(&mut buf);
            // Without it the token reaches the model as [UNK] (or as bytes it
            // never saw as one); blacklisted, no run of symbols comes back.
            let Some(&id) = tokenizer.token_to_id.get(token) else {
                bail!(
                    "symbol_token is on but the vocabulary of {} has no {token} token",
                    manifest.manifest_path.display()
                );
            };
            if config.vocab_blacklist.contains(&id) {
                bail!(
                    "symbol_token is on but vocab_blacklist forbids the {token} token (id {id}); \
                     drop it from the blacklist"
                );
            }
        }
        let crf = manifest
            .crf
            .as_ref()
            .map(|(path, rank)| {
                crate::pipeline::crf::CrfTables::load(path, *rank, tokenizer.vocab_size())
            })
            .transpose()?;
        let rescorer = rescore::Rescorer::load(
            manifest.rescore_pairs.as_deref(),
            config.pair_weight,
            manifest.rescore_words.as_deref(),
            config.word_weight,
        )?;
        Ok(Self {
            tokenizer,
            proposal,
            seq_len: manifest.max_seq_len,
            config,
            reduce: FrameReduce::default(),
            inspect: 0,
            chunk_cache: HashMap::new(),
            last_proposal: None,
            crf,
            rescorer,
        })
    }

    /// Convert one request: the pipeline's candidates, cut to the display
    /// limit. A reading longer than a chunk (`chunk::MAX_CHUNK`, or what the
    /// model's sequence holds) is converted chunk by chunk, a sentence after
    /// another. A reading
    /// carrying bracket / annotation symbols, or chars the vocabulary has no
    /// token for, goes through symbol split when it is enabled; with symbol
    /// token, every run of symbols is read as one token instead. With digit
    /// split, the runs of digits are cut out first and spliced back. A
    /// `continue_beam` request decodes with a wider beam and returns more
    /// candidates; which of them are new is the caller's to tell.
    pub fn convert(&mut self, request: ConversionRequest<'_>) -> Result<CandidateSet> {
        if request.continue_beam == 0 {
            return self.convert_at(request);
        }
        let saved = self.config.clone();
        let extra = CONTINUE_STEP * request.continue_beam;
        self.config.beam_width += extra;
        self.config.max_candidates += extra;
        let out = self.convert_at(request);
        self.config = saved;
        out
    }

    fn convert_at(&mut self, request: ConversionRequest<'_>) -> Result<CandidateSet> {
        let max_chunk = chunk::MAX_CHUNK.min(self.seq_len.saturating_sub(2));
        let chunks = chunk::split_with_tokens(
            request.reading,
            max_chunk,
            self.seq_len.saturating_sub(2),
            self.config.symbol_token,
            self.config.digit_split,
            |c| {
                let in_vocab = self.tokenizer.has_char(c);
                if self.config.symbol_split
                    && !self.config.symbol_token
                    && (symsplit::is_strip_symbol(c) || !in_vocab)
                {
                    0
                } else if in_vocab {
                    1
                } else {
                    // as the input encodes it: one [UNK] or one token a byte
                    let mut buf = [0u8; 4];
                    self.tokenizer
                        .encode_with_byte_fallback(c.encode_utf8(&mut buf))
                        .len()
                }
            },
        )?;
        let (mut candidates, mut perf, frames) = if chunks.len() > 1 {
            let total_t0 = Instant::now();
            let pool_limit = self.config.pool_limit();
            let candidates =
                chunk::convert_chunks(request.context, &chunks, pool_limit, |c, r| {
                    self.convert_chunk(c, r)
                })?;
            let perf = ConvertPerf {
                total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
                candidate_count: candidates.len(),
                hyp_count: candidates.len(),
                ..ConvertPerf::default()
            };
            (candidates, perf, None)
        } else {
            self.convert_piece(request.context, request.reading)?
        };
        if let Some(r) = &self.rescorer {
            let t0 = Instant::now();
            r.rescore(request.context, request.reading, &mut candidates);
            perf.rescore_ms = t0.elapsed().as_secs_f64() * 1000.0;
            perf.total_ms += perf.rescore_ms;
        }
        candidates.truncate(self.config.max_candidates);
        perf.candidate_count = candidates.len();
        Ok(CandidateSet {
            candidates,
            perf,
            frames,
        })
    }

    /// A chunk's candidates. Remembered: while a long reading is typed, the
    /// chunks before the last keep their reading and context.
    fn convert_chunk(&mut self, context: &str, reading: &str) -> Result<Vec<Candidate>> {
        let key = (
            context.to_string(),
            reading.to_string(),
            self.config.beam_width,
        );
        if let Some(hit) = self.chunk_cache.get(&key) {
            return Ok(hit.clone());
        }
        let (candidates, _, _) = self.convert_piece(context, reading)?;
        if self.chunk_cache.len() >= CHUNK_CACHE {
            self.chunk_cache.clear();
        }
        self.chunk_cache.insert(key, candidates.clone());
        Ok(candidates)
    }

    /// A reading the model takes in one sequence. With digit split and a
    /// digit in it: its runs of digits spliced back verbatim and every other
    /// segment through [`Self::convert_segment`], after the context and the
    /// output so far (so the digits reach the model only as context; training
    /// readings never hold them). Else the reading through
    /// [`Self::convert_segment`]. Candidates up to the pool limit.
    fn convert_piece(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        if !(self.config.digit_split && digitsplit::has_digit(reading)) {
            return self.convert_segment(context, reading);
        }
        let total_t0 = Instant::now();
        let pool_limit = self.config.pool_limit();
        let candidates = digitsplit::convert(context, reading, pool_limit, |c, r| {
            self.convert_segment(c, r)
                .map(|(candidates, _, _)| candidates)
        })?;
        let perf = ConvertPerf {
            total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            candidate_count: candidates.len(),
            hyp_count: candidates.len(),
            ..ConvertPerf::default()
        };
        Ok((candidates, perf, None))
    }

    /// A digit-free segment (or a reading, without digit split): with symbol
    /// token, one pass over it with its runs of symbols as tokens; else
    /// through symbol split when it carries split chars, else one pass of
    /// the pipeline. Candidates up to the pool limit.
    fn convert_segment(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        if self.config.symbol_token {
            return self.convert_tokenized(context, reading);
        }
        let out_of_vocab: HashSet<char> = reading
            .chars()
            .filter(|&c| !self.tokenizer.has_char(c))
            .collect();
        // Only [UNK] chars leave the context: one the vocabulary spells with
        // byte-fallback tokens is still read (Aster-v1.2-small spells ~ that
        // way, and dropping it turned 東京~大阪間 into 東京~大阪館).
        let dropped_from_context: HashSet<char> = out_of_vocab
            .iter()
            .copied()
            .filter(|&c| self.tokenizer.is_unknown(c))
            .collect();
        let is_split = move |c: char| symsplit::is_strip_symbol(c) || out_of_vocab.contains(&c);
        if self.config.symbol_split && reading.chars().any(&is_split) {
            let total_t0 = Instant::now();
            let pool_limit = self.config.pool_limit();
            let candidates =
                symsplit::convert_split(context, reading, pool_limit, is_split, |c, r| {
                    // A spliced char the vocabulary lacks would reach the
                    // model as [UNK] through the running context: with
                    // Aster-v1.2-small minus `.`, 2ぱーせんと after `6.`
                    // gave 2ーセン%, after `6` 2パーセント.
                    let c: String = c
                        .chars()
                        .filter(|ch| !dropped_from_context.contains(ch))
                        .collect();
                    self.convert_span(&c, r)
                        .map(|(candidates, _, _)| candidates)
                })?;
            let perf = ConvertPerf {
                total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
                candidate_count: candidates.len(),
                hyp_count: candidates.len(),
                ..ConvertPerf::default()
            };
            Ok((candidates, perf, None))
        } else {
            self.convert_span(context, reading)
        }
    }

    /// Symbol token: one pass of the pipeline over the reading with each run
    /// of symbols as one token (the context as typed), and the runs put back
    /// into every candidate (`symtoken::Tokenized::restore_candidates`: a
    /// token past the runs dropped, the reading positions moved to the
    /// reading, duplicates after restoring dropped). The frames' reading
    /// indices are moved to the reading too; their inputs stay the tokens
    /// the model read. Run even on a reading without symbols, so a stray
    /// token in an output is dropped.
    fn convert_tokenized(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        let tokenized = symtoken::tokenize(reading);
        let (candidates, mut perf, frames) = self.convert_span(context, &tokenized.text)?;
        let candidates = tokenized.restore_candidates(&candidates);
        perf.candidate_count = candidates.len();
        let frames = frames.map(|frames| {
            frames
                .into_iter()
                .map(|mut f| {
                    f.reading_index = f.reading_index.and_then(|i| tokenized.origin(i));
                    f
                })
                .collect()
        });
        Ok((candidates, perf, frames))
    }

    /// One pass of the pipeline over a symbol-free reading: proposal, decode,
    /// candidates.
    fn convert_span(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        let total_t0 = Instant::now();
        if reading.is_empty() {
            return Ok((Vec::new(), ConvertPerf::default(), None));
        }
        let cfg = &self.config;
        let greedy = cfg.beam_width <= 1;
        // Inspection needs probabilities even when decoding is greedy.
        let greedy_only = greedy && self.inspect == 0;

        let proposal_t0 = Instant::now();
        let hit = self.last_proposal.as_ref().is_some_and(|p| {
            p.greedy == greedy_only
                && p.inspect == self.inspect
                && p.context == context
                && p.reading == reading
        });
        if !hit {
            let encoded = proposal::encode(
                &self.tokenizer,
                self.seq_len,
                cfg.max_context,
                context,
                reading,
            )?;
            let frame_reading =
                proposal::frame_reading(&self.tokenizer, &encoded.0, encoded.2, reading);
            let inputs: Vec<i64> = encoded.0[..encoded.2].to_vec();
            let proposal = self.proposal.run(
                &self.tokenizer,
                encoded,
                &cfg.vocab_blacklist,
                cfg.top_k_per_step.max(self.inspect.saturating_add(1)),
                self.reduce,
                greedy_only,
            )?;
            self.last_proposal = Some(LastProposal {
                context: context.to_string(),
                reading: reading.to_string(),
                greedy: greedy_only,
                inspect: self.inspect,
                proposal,
                frame_reading,
                inputs,
            });
        }
        let last = self.last_proposal.as_ref().expect("stored above");
        let (proposal, frame_reading, inputs) = (&last.proposal, &last.frame_reading, &last.inputs);
        let proposal_ms = proposal_t0.elapsed().as_secs_f64() * 1000.0;

        let beam_t0 = Instant::now();
        let hyps = decode::hypotheses(
            proposal,
            self.tokenizer.blank_id,
            cfg.beam_width,
            cfg.top_k_per_step,
            self.crf.as_ref(),
        );
        let beam_ms = beam_t0.elapsed().as_secs_f64() * 1000.0;
        let hyp_count = hyps.len();

        let post_t0 = Instant::now();
        let mut candidates = pool::candidates(
            hyps,
            greedy,
            proposal,
            reading,
            &self.tokenizer,
            frame_reading,
            cfg.normalize_candidates,
        );
        candidates.truncate(cfg.pool_limit());
        let perf = ConvertPerf {
            input_len: proposal.input_len,
            proposal_ms,
            beam_ms,
            post_ms: post_t0.elapsed().as_secs_f64() * 1000.0,
            rescore_ms: 0.0,
            total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            hyp_count,
            candidate_count: candidates.len(),
        };
        let frames =
            (self.inspect > 0).then(|| self.frame_views(&proposal.steps, inputs, frame_reading));
        Ok((candidates, perf, frames))
    }

    /// The proposal's frames as [`FrameView`]s; `inputs` are the encoded
    /// ids, one per frame.
    fn frame_views(
        &self,
        steps: &[proposal::FrameLogProbs],
        inputs: &[i64],
        frame_reading: &[u32],
    ) -> Vec<FrameView> {
        let tok = &self.tokenizer;
        // reading frames start after the first [SEP]
        let reading_from = inputs
            .iter()
            .skip(1)
            .position(|&id| id == tok.sep_id as i64)
            .map_or(inputs.len(), |i| i + 2);
        let name = |id: u32| {
            tok.id_to_token
                .get(id as usize)
                .cloned()
                .unwrap_or_default()
        };
        steps
            .iter()
            .zip(inputs)
            .enumerate()
            .map(|(f, (step, &input))| FrameView {
                input: name(input as u32),
                reading_index: (f >= reading_from)
                    .then(|| frame_reading.get(f).copied())
                    .flatten(),
                blank: step.blank.exp(),
                top: step
                    .top
                    .iter()
                    .filter(|(id, _)| *id != tok.blank_id)
                    .take(self.inspect)
                    .map(|&(id, lp)| (name(id), lp.exp()))
                    .collect(),
            })
            .collect()
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> RuntimeEnv {
        RuntimeEnv {
            ort_intra_threads: 1,
            top_k_per_step: 16,
        }
    }

    #[test]
    fn explicit_beats_manifest_beats_default() {
        let explicit = DecodeDefaults {
            beam_width: Some(8),
            ..DecodeDefaults::default()
        };
        let manifest = DecodeDefaults {
            beam_width: Some(20),
            top_k: Some(10),
            symbol_split: Some(true),
            ..DecodeDefaults::default()
        };
        let cfg = DecodeConfig::resolve(&explicit, &manifest, &env());
        assert_eq!(cfg.beam_width, 8);
        assert_eq!(cfg.max_candidates, 10);
        assert!(cfg.symbol_split);
        assert!(!cfg.symbol_token);
        assert_eq!(cfg.max_context, 64);
        assert_eq!(cfg.top_k_per_step, 16);
        assert_eq!(cfg.pool_limit(), 10);
        assert_eq!((cfg.pair_weight, cfg.word_weight), (0.0, 0.0));
    }

    #[test]
    fn symbol_token_resolves_like_symbol_split() {
        let on = DecodeDefaults {
            symbol_token: Some(true),
            ..DecodeDefaults::default()
        };
        let off = DecodeDefaults {
            symbol_token: Some(false),
            ..DecodeDefaults::default()
        };
        let none = DecodeDefaults::default();
        assert!(DecodeConfig::resolve(&none, &on, &env()).symbol_token);
        assert!(!DecodeConfig::resolve(&off, &on, &env()).symbol_token);
        assert!(DecodeConfig::resolve(&on, &none, &env()).symbol_token);
        assert!(!DecodeConfig::resolve(&none, &none, &env()).symbol_token);
    }

    #[test]
    fn digit_split_resolves_like_symbol_split() {
        let on = DecodeDefaults {
            digit_split: Some(true),
            ..DecodeDefaults::default()
        };
        let off = DecodeDefaults {
            digit_split: Some(false),
            ..DecodeDefaults::default()
        };
        let none = DecodeDefaults::default();
        assert!(DecodeConfig::resolve(&none, &on, &env()).digit_split);
        assert!(!DecodeConfig::resolve(&off, &on, &env()).digit_split);
        assert!(DecodeConfig::resolve(&on, &none, &env()).digit_split);
        assert!(!DecodeConfig::resolve(&none, &none, &env()).digit_split);
    }

    #[test]
    fn explicit_empty_blacklist_clears_the_manifest_one() {
        let explicit = DecodeDefaults {
            vocab_blacklist: Some(Vec::new()),
            ..DecodeDefaults::default()
        };
        let manifest = DecodeDefaults {
            vocab_blacklist: Some(vec![724]),
            ..DecodeDefaults::default()
        };
        assert!(DecodeConfig::resolve(&explicit, &manifest, &env())
            .vocab_blacklist
            .is_empty());
    }

    #[test]
    fn zero_widths_are_clamped() {
        let explicit = DecodeDefaults {
            beam_width: Some(0),
            top_k: Some(0),
            top_k_per_step: Some(0),
            ..DecodeDefaults::default()
        };
        let cfg = DecodeConfig::resolve(&explicit, &DecodeDefaults::default(), &env());
        assert_eq!(
            (cfg.beam_width, cfg.max_candidates, cfg.top_k_per_step),
            (1, 1, 1)
        );
        assert_eq!(cfg.pool_limit(), 1);
    }
}
