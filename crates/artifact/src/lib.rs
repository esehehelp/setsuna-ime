//! Model artifact manifests.
//!
//! Every manifest is a JSON object whose `schema` and `model_family` fields
//! form a thin envelope; the rest of the object is family-specific. The
//! runtime dispatches on `schema` and never infers artifact paths from file
//! names: every file an artifact needs is listed in its manifest, relative to
//! the directory that holds the manifest.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

pub const CTC_NAT_SCHEMA: &str = "neo-ime-ctc-nat-export-v1";
pub const CTC_NAT_FAMILY: &str = "ctc_nat";

/// Fields shared by every manifest schema.
#[derive(Debug, Clone, Deserialize)]
pub struct Envelope {
    pub schema: String,
    pub model_family: String,
}

/// Read a manifest file, returning its envelope and the full JSON value for
/// family-specific parsing.
/// The file `rel` a manifest at `manifest` names, beside it or below: a
/// relative path of plain names (no `..`, no drive, no root, no share:
/// `\\host\share` would send the user's credentials to the host), which
/// stays within the manifest's directory when links are followed too.
/// `what` names it in the errors.
pub fn file_within(manifest: &Path, rel: &Path, what: &str) -> Result<PathBuf> {
    use std::path::Component;
    anyhow::ensure!(
        !rel.as_os_str().is_empty()
            && rel
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "manifest {}: {what} {} is not a path within the manifest's directory",
        manifest.display(),
        rel.display()
    );
    let base = manifest.parent().unwrap_or_else(|| Path::new("."));
    // no part of it a link to a share, looked at before anything opens it:
    // opening one would send the user's credentials to the host before
    // the check below could refuse it
    let mut part = base.to_path_buf();
    for c in rel.components() {
        part.push(c);
        anyhow::ensure!(
            !leads_to_share(&part),
            "manifest {}: {what} {} leads to a network share",
            manifest.display(),
            rel.display()
        );
    }
    let full = base.join(rel);
    anyhow::ensure!(
        full.is_file(),
        "manifest {}: {what} {} not found",
        manifest.display(),
        full.display()
    );
    let canonical = std::fs::canonicalize(&full)
        .with_context(|| format!("manifest {}: {what} {}", manifest.display(), full.display()))?;
    let root = std::fs::canonicalize(if base.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base
    })
    .with_context(|| format!("manifest {}: its directory", manifest.display()))?;
    anyhow::ensure!(
        canonical.starts_with(&root),
        "manifest {}: {what} {} leads out of the manifest's directory",
        manifest.display(),
        rel.display()
    );
    Ok(full)
}

/// Whether `path` is a link that leads, itself or through the links it
/// leads to (a few at most), to a network path (`\\host\share`); read
/// without following any. A link that cannot be read is taken for one.
fn leads_to_share(path: &Path) -> bool {
    use std::path::{Component, Prefix};
    let network = |p: &Path| {
        matches!(p.components().next(),
            Some(Component::Prefix(x)) if matches!(x.kind(), Prefix::UNC(..) | Prefix::VerbatimUNC(..)))
            || p.to_string_lossy().starts_with("//")
    };
    let mut p = path.to_path_buf();
    for _ in 0..8 {
        let Ok(m) = std::fs::symlink_metadata(&p) else {
            return false;
        };
        if !m.file_type().is_symlink() {
            return false;
        }
        let Ok(target) = std::fs::read_link(&p) else {
            return true;
        };
        if network(&target) {
            return true;
        }
        p = if target.is_absolute() {
            target
        } else {
            p.parent().unwrap_or_else(|| Path::new(".")).join(target)
        };
    }
    true
}

pub fn read_manifest(path: &Path) -> Result<(Envelope, serde_json::Value)> {
    let bytes = std::fs::read(path).with_context(|| format!("read manifest {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse manifest {}", path.display()))?;
    let envelope: Envelope = serde_json::from_value(value.clone())
        .with_context(|| format!("manifest {} has no schema/model_family", path.display()))?;
    Ok((envelope, value))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalArtifact {
    /// The ONNX graph. Absent when `native` holds the weights for the engine's own implementation (the published
    /// packages without their graphs); one of the two is required.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// One of fp32 / fp16 / int8 / int4. One manifest names exactly one
    /// proposal graph; another precision is another manifest.
    pub precision: String,
    /// Weight files the graph stores outside itself (`*.onnx.data`), listed
    /// so the artifact's file set is complete (the engine reads the native
    /// weights, not the graph).
    #[serde(default)]
    pub external_data: Vec<PathBuf>,
    /// The vocabulary id of each column of the graph's logits (ascending), when it writes only those (the
    /// research export's: a reading corrector's head cut to the tokens a reading has);
    /// absent, a column a vocabulary id.
    #[serde(default)]
    pub logit_ids: Option<Vec<u32>>,
    /// The graph's weights for the engine's own implementation of it (the research export's, a
    /// safetensors file; the engine's `pipeline::native_encoder`), on an int8 manifest.
    #[serde(default)]
    pub native: Option<NativeEncoderArtifact>,
}

/// The encoder's weights for the engine's own implementation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEncoderArtifact {
    pub path: PathBuf,
    /// int8 (the int8head graph's weights)
    pub precision: String,
}

/// A CTC-CRF's transition table (the research export's): `a` ((V+1) x
/// rank, the last row the start symbol) then `b` (V x rank), f32 little
/// endian; ψ(c, d) = mean_j exp(a_cj + b_dj).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrfArtifact {
    pub path: PathBuf,
    pub rank: usize,
}

/// Outside statistics the engine rescores a conversion's candidates with
/// (each weighted by `decode_defaults`' `pair_weight` / `word_weight`):
/// `pairs`, two-character counts (the research export's: magic
/// `NIPAIRS1`, then the chars, row starts, columns and counts of a sparse
/// count matrix); `words`, a (reading, word, count) TSV, the words NFKC.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RescoreArtifact {
    pub pairs: Option<PathBuf>,
    pub words: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialIds {
    pub pad: u32,
    pub unk: u32,
    pub cls: u32,
    pub sep: u32,
    pub blank: u32,
    pub mask: u32,
}

/// Decode settings the artifact recommends (the recipe it was verified
/// with). Every field is optional; a value set by the caller wins, and an
/// unset one falls back to the engine default.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeDefaults {
    pub beam_width: Option<usize>,
    pub top_k: Option<usize>,
    pub top_k_per_step: Option<usize>,
    pub max_context: Option<usize>,
    pub vocab_blacklist: Option<Vec<u32>>,
    pub symbol_split: Option<bool>,
    /// Each run of symbols in the reading read as one `◇` token (models
    /// trained on rows with symbol tokens); wins over `symbol_split`.
    pub symbol_token: Option<bool>,
    /// Runs of ASCII digits cut out of the reading and spliced back verbatim
    /// (training readings spell numbers in kana).
    pub digit_split: Option<bool>,
    /// Map the candidates' punctuation and full-width ASCII the IME way
    /// (`normalize_candidate`); unset or false, a candidate is the model's
    /// output as decoded (the research default).
    pub normalize_candidates: Option<bool>,
    /// Weight of the two-character term of the rescoring (`rescore.pairs`);
    /// unset or 0, off.
    pub pair_weight: Option<f32>,
    /// Weight of the word term of the rescoring (`rescore.words`); unset or
    /// 0, off.
    pub word_weight: Option<f32>,
    /// A reading corrector's own manifest (an engine's corrector named by its
    /// manifest, not bundled): the margin log P(corrected) - log P(as typed)
    /// above which the corrected reading is converted first, as a bundled
    /// corrector's `threshold`. Unset: never first, only offered.
    pub corrector_threshold: Option<f32>,
}

/// Keys of the dictionary / n-gram fusion recipe that engine v1 read and v2
/// dropped. A manifest carrying them was verified with that fusion, so it is
/// refused with a pointer instead of being silently served without it.
const REMOVED_DECODE_KEYS: &[&str] = &[
    "lm_alpha",
    "lm_beta",
    "lm_backoff_penalty",
    "dict_gamma",
    "neural_floor",
    "dict_topk",
    "dict_min_cost",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CtcNatManifestFile {
    #[allow(dead_code)] // checked through the envelope
    schema: String,
    #[allow(dead_code)] // checked through the envelope
    model_family: String,
    proposal: ProposalArtifact,
    vocab: PathBuf,
    /// The vocabulary the training rows were written in, when the model's
    /// keeps part of it (the shard vocabulary remapped to `[UNK]`): a char of
    /// it the model's vocabulary lacks is one `[UNK]`, a char outside both
    /// one `[UNK]` per UTF-8 byte, as the training rows encode it.
    #[serde(default)]
    input_vocab: Option<PathBuf>,
    special_ids: SpecialIds,
    max_seq_len: usize,
    #[serde(default)]
    decode_defaults: DecodeDefaults,
    /// The model's CTC-CRF transitions, when it has them: the beam search
    /// adds them (decoding without them is another model).
    #[serde(default)]
    crf: Option<CrfArtifact>,
    /// The candidates' rescoring tables.
    #[serde(default)]
    rescore: RescoreArtifact,
    /// A model decoded by its shallow AR decoder (the research export's AR
    /// form); the proposal graph is then its encoder graph.
    #[serde(default)]
    ar_decoder: Option<ArDecoderArtifact>,
    /// A reading corrector the engine runs before each conversion.
    #[serde(default)]
    corrector: Option<CorrectorArtifact>,
}

/// A reading corrector carried by a CTC-NAT artifact (the research export's; the engine's
/// `pipeline::correct`): its own CTC graph over the converter's vocabulary and special ids, read greedily before each
/// conversion with its own decode defaults (its context length, symbol token); `threshold`, the margin log
/// P(corrected) - log P(as typed) above which the corrected reading is converted first.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectorArtifact {
    pub proposal: ProposalArtifact,
    pub max_seq_len: usize,
    #[serde(default)]
    pub decode_defaults: DecodeDefaults,
    pub threshold: f32,
}

/// A manifest's reading corrector, resolved: a CTC-NAT manifest of its own (the converter's vocabulary, input
/// vocabulary and special ids; no CRF, rescoring, AR decoder or corrector) and its threshold.
#[derive(Debug, Clone)]
pub struct BundledCorrector {
    pub manifest: CtcNatManifest,
    pub threshold: f32,
}

/// Logit columns (`ProposalArtifact::logit_ids`) ascending, distinct and not empty.
fn check_logit_ids(ids: &[u32], path: &Path, what: &str) -> Result<()> {
    anyhow::ensure!(
        !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]),
        "manifest {}: {what}.logit_ids must be ascending, distinct and not empty",
        path.display()
    );
    Ok(())
}

/// The shallow AR decoder of a model (the research export's AR form): the
/// proposal graph is its encoder graph, (input_ids,
/// attention_mask) -> `cross_k` / `cross_v` f32 (layers, B, heads, T,
/// head_dim); `step` is one output position of N hypotheses, (tokens,
/// positions i64 (N); past_k / past_v f32 (layers, N, heads, S, head_dim);
/// cross_k / cross_v (layers, 1, heads, T, head_dim); memory_bias f32 (1, T))
/// -> `log_probs` f32 (N, V), `new_k` / `new_v` (layers, N, heads, 1,
/// head_dim). Position 0 reads `bos_id`; `eos_id` ends a hypothesis.
/// `native`: the step's int8 weights for the engine's own implementation of
/// it (the research export's native step, a safetensors file; the engine
/// `pipeline::ar_native`), on an int8 manifest.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArDecoderArtifact {
    /// The step graph; absent when `native` holds the step's weights (one of the two is required).
    #[serde(default)]
    pub step: Option<ProposalArtifact>,
    #[serde(default)]
    pub native: Option<NativeStepArtifact>,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    /// The best next tokens the step graph gives a hypothesis.
    pub top_k: usize,
    /// The script latent at choice points (the research export's script scores): the step graph takes
    /// `script_bias` (one number per class, in `classes`' order) added to the prior's logits of the classes.
    #[serde(default)]
    pub script: Option<ScriptLatentArtifact>,
}

/// The longest input a manifest may say its model takes (tokens), and the
/// highest CTC-CRF rank: past them a manifest is no model's (and the
/// buffers sized by them would not fit).
pub const MAX_SEQ_LEN: usize = 1 << 14;
pub const MAX_CRF_RANK: usize = 4096;

/// The classes of [`ScriptLatentArtifact`], in the order of its `script_bias`.
pub const SCRIPT_CLASSES: [&str; 4] = ["other", "kanji", "hiragana", "katakana"];

/// A model's script latent (`ar_decoder.script`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptLatentArtifact {
    /// "choice": the latent at choice points, class 0 reading every token, 1-3 their own class's
    pub mode: String,
    /// [`SCRIPT_CLASSES`]
    pub classes: Vec<String>,
}

/// The step's weights for the engine's own implementation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeStepArtifact {
    pub path: PathBuf,
    /// int8
    pub precision: String,
}

/// [`ArDecoderArtifact`] with the step graph's path (and the native step's)
/// resolved.
#[derive(Debug, Clone)]
pub struct ArDecoder {
    /// The step graph, when the manifest names one (else `native_path`).
    pub step_path: Option<PathBuf>,
    pub native_path: Option<PathBuf>,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    pub top_k: usize,
    /// Whether the model has the script latent at choice points (`ar_decoder.script`, checked).
    pub script: bool,
}

/// A CTC-NAT manifest with every artifact path resolved against the
/// manifest's directory and checked to exist.
#[derive(Debug, Clone)]
pub struct CtcNatManifest {
    pub manifest_path: PathBuf,
    /// The proposal graph, when the manifest names one (else `proposal_native_path`).
    pub proposal_path: Option<PathBuf>,
    pub precision: String,
    /// See `ProposalArtifact::logit_ids` (checked ascending and distinct).
    pub logit_ids: Option<Vec<u32>>,
    /// See `ProposalArtifact::native` (resolved; int8 only).
    pub proposal_native_path: Option<PathBuf>,
    pub vocab_path: PathBuf,
    /// See `CtcNatManifestFile::input_vocab`.
    pub input_vocab_path: Option<PathBuf>,
    pub special_ids: SpecialIds,
    pub max_seq_len: usize,
    pub decode_defaults: DecodeDefaults,
    /// The CTC-CRF table (path resolved) and its rank.
    pub crf: Option<(PathBuf, usize)>,
    /// The rescoring tables (`RescoreArtifact`), paths resolved.
    pub rescore_pairs: Option<PathBuf>,
    pub rescore_words: Option<PathBuf>,
    /// The AR decoder (`ArDecoderArtifact`), its step graph resolved.
    pub ar_decoder: Option<ArDecoder>,
    /// The reading corrector (`CorrectorArtifact`), its graph resolved.
    pub corrector: Option<Box<BundledCorrector>>,
}

impl CtcNatManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let (envelope, value) = read_manifest(path)?;
        anyhow::ensure!(
            envelope.schema == CTC_NAT_SCHEMA && envelope.model_family == CTC_NAT_FAMILY,
            "manifest {} is {}/{}, expected {CTC_NAT_SCHEMA}/{CTC_NAT_FAMILY}",
            path.display(),
            envelope.schema,
            envelope.model_family
        );
        let removed: Vec<&str> = REMOVED_DECODE_KEYS
            .iter()
            .copied()
            .filter(|k| value["decode_defaults"].get(k).is_some())
            .chain(value.get("ngram").map(|_| "ngram"))
            .collect();
        anyhow::ensure!(
            removed.is_empty(),
            "manifest {} sets {removed:?}: dictionary / n-gram fusion is not part of the \
             engine any more; re-export the model (or run it with the engine at tag \
             engine-v1-final)",
            path.display()
        );
        let file: CtcNatManifestFile = serde_json::from_value(value)
            .with_context(|| format!("parse CTC-NAT manifest {}", path.display()))?;
        anyhow::ensure!(
            matches!(
                file.proposal.precision.as_str(),
                "fp32" | "fp16" | "int8" | "int4"
            ),
            "manifest {}: unknown proposal precision {:?}",
            path.display(),
            file.proposal.precision
        );
        anyhow::ensure!(
            (1..=MAX_SEQ_LEN).contains(&file.max_seq_len),
            "manifest {}: max_seq_len {} out of 1..={MAX_SEQ_LEN}",
            path.display(),
            file.max_seq_len
        );
        if let Some(c) = &file.crf {
            anyhow::ensure!(
                (1..=MAX_CRF_RANK).contains(&c.rank),
                "manifest {}: CRF rank {} out of 1..={MAX_CRF_RANK}",
                path.display(),
                c.rank
            );
        }
        let resolve = |p: &Path, what: &str| -> Result<PathBuf> {
            let full = file_within(path, p, what)?;
            // absolute, without `..`: on Windows the `\\?\` form, which ONNX
            // Runtime opens past MAX_PATH (an export path of 274 chars, or a
            // shorter one through `../..`, was "File doesn't exist" to it)
            std::fs::canonicalize(&full)
                .with_context(|| format!("manifest {}: {what} {}", path.display(), full.display()))
        };
        for extra in &file.proposal.external_data {
            resolve(extra, "external data")?;
        }
        // a graph or the weights for the engine's own implementation: the proposal, a corrector's, the AR step
        anyhow::ensure!(
            file.proposal.path.is_some() || file.proposal.native.is_some(),
            "manifest {}: proposal needs its graph (path) or its native weights (native)",
            path.display()
        );
        anyhow::ensure!(
            file.proposal.path.is_some() || file.proposal.external_data.is_empty(),
            "manifest {}: proposal.external_data without the graph (path)",
            path.display()
        );
        if let Some(c) = &file.corrector {
            anyhow::ensure!(
                c.proposal.path.is_some() || c.proposal.native.is_some(),
                "manifest {}: corrector.proposal needs its graph (path) or its native weights (native)",
                path.display()
            );
        }
        let ar_decoder = match &file.ar_decoder {
            None => None,
            Some(a) => {
                anyhow::ensure!(
                    a.step.as_ref().is_some_and(|s| s.path.is_some()) || a.native.is_some(),
                    "manifest {}: ar_decoder needs its step graph (step.path) or its native step (native)",
                    path.display()
                );
                for extra in a.step.iter().flat_map(|s| &s.external_data) {
                    resolve(extra, "AR step external data")?;
                }
                anyhow::ensure!(
                    a.layers > 0
                        && a.heads > 0
                        && a.head_dim > 0
                        && a.max_positions > 1
                        && a.top_k > 0,
                    "manifest {}: ar_decoder sizes must be positive",
                    path.display()
                );
                if let Some(n) = &a.native {
                    anyhow::ensure!(
                        n.precision == "int8",
                        "manifest {}: native AR step precision {:?}, expected int8",
                        path.display(),
                        n.precision
                    );
                }
                if let Some(s) = &a.step {
                    anyhow::ensure!(
                        matches!(s.precision.as_str(), "fp32" | "fp16" | "int8" | "int4"),
                        "manifest {}: unknown AR step precision {:?}",
                        path.display(),
                        s.precision
                    );
                }
                if let Some(s) = &a.script {
                    anyhow::ensure!(
                        s.mode == "choice" && s.classes == SCRIPT_CLASSES,
                        "manifest {}: ar_decoder.script {s:?}, expected mode \"choice\" and classes {SCRIPT_CLASSES:?}",
                        path.display()
                    );
                }
                Some(ArDecoder {
                    step_path: a
                        .step
                        .as_ref()
                        .and_then(|s| s.path.as_ref())
                        .map(|p| resolve(p, "AR step graph"))
                        .transpose()?,
                    native_path: a
                        .native
                        .as_ref()
                        .map(|n| resolve(&n.path, "AR native step"))
                        .transpose()?,
                    layers: a.layers,
                    heads: a.heads,
                    head_dim: a.head_dim,
                    max_positions: a.max_positions,
                    bos_id: a.bos_id,
                    eos_id: a.eos_id,
                    top_k: a.top_k,
                    script: a.script.is_some(),
                })
            }
        };
        if let Some(ids) = &file.proposal.logit_ids {
            check_logit_ids(ids, path, "proposal")?;
            anyhow::ensure!(
                file.ar_decoder.is_none(),
                "manifest {}: proposal.logit_ids is for a CTC graph, not an AR model's encoder",
                path.display()
            );
        }
        // the native weights of a graph (int8 on int8); with logit_ids, its head holds those columns alone
        let native = |p: &ProposalArtifact, what: &str| -> Result<Option<PathBuf>> {
            let Some(n) = &p.native else {
                return Ok(None);
            };
            anyhow::ensure!(
                n.precision == "int8" && p.precision == "int8",
                "manifest {}: {what} precision {:?} on a {:?} graph, expected int8 on int8",
                path.display(),
                n.precision,
                p.precision
            );
            Ok(Some(resolve(&n.path, what)?))
        };
        let proposal_native_path = native(&file.proposal, "native encoder")?;
        let vocab_path = resolve(&file.vocab, "vocab")?;
        let input_vocab_path = file
            .input_vocab
            .as_deref()
            .map(|p| resolve(p, "input vocab"))
            .transpose()?;
        let corrector = match &file.corrector {
            None => None,
            Some(c) => {
                anyhow::ensure!(
                    matches!(
                        c.proposal.precision.as_str(),
                        "fp32" | "fp16" | "int8" | "int4"
                    ),
                    "manifest {}: unknown corrector precision {:?}",
                    path.display(),
                    c.proposal.precision
                );
                anyhow::ensure!(
                    c.proposal.external_data.is_empty(),
                    "manifest {}: the corrector's graph must hold its weights itself",
                    path.display()
                );
                anyhow::ensure!(
                    !c.threshold.is_nan(),
                    "manifest {}: corrector.threshold is NaN",
                    path.display()
                );
                anyhow::ensure!(
                    (1..=MAX_SEQ_LEN).contains(&c.max_seq_len),
                    "manifest {}: corrector.max_seq_len {} out of 1..={MAX_SEQ_LEN}",
                    path.display(),
                    c.max_seq_len
                );
                if let Some(ids) = &c.proposal.logit_ids {
                    check_logit_ids(ids, path, "corrector.proposal")?;
                }
                Some(Box::new(BundledCorrector {
                    manifest: CtcNatManifest {
                        manifest_path: path.to_path_buf(),
                        proposal_path: c
                            .proposal
                            .path
                            .as_ref()
                            .map(|p| resolve(p, "corrector proposal"))
                            .transpose()?,
                        precision: c.proposal.precision.clone(),
                        logit_ids: c.proposal.logit_ids.clone(),
                        proposal_native_path: native(&c.proposal, "corrector's native encoder")?,
                        vocab_path: vocab_path.clone(),
                        input_vocab_path: input_vocab_path.clone(),
                        special_ids: file.special_ids,
                        max_seq_len: c.max_seq_len,
                        decode_defaults: c.decode_defaults.clone(),
                        crf: None,
                        rescore_pairs: None,
                        rescore_words: None,
                        ar_decoder: None,
                        corrector: None,
                    },
                    threshold: c.threshold,
                }))
            }
        };
        Ok(Self {
            manifest_path: path.to_path_buf(),
            proposal_path: file
                .proposal
                .path
                .as_ref()
                .map(|p| resolve(p, "proposal"))
                .transpose()?,
            precision: file.proposal.precision,
            logit_ids: file.proposal.logit_ids,
            proposal_native_path,
            vocab_path,
            input_vocab_path,
            special_ids: file.special_ids,
            max_seq_len: file.max_seq_len,
            decode_defaults: file.decode_defaults,
            crf: file
                .crf
                .as_ref()
                .map(|c| resolve(&c.path, "CRF table").map(|p| (p, c.rank)))
                .transpose()?,
            rescore_pairs: file
                .rescore
                .pairs
                .as_deref()
                .map(|p| resolve(p, "rescore pairs"))
                .transpose()?,
            rescore_words: file
                .rescore
                .words
                .as_deref()
                .map(|p| resolve(p, "rescore words"))
                .transpose()?,
            ar_decoder,
            corrector,
        })
    }
}

pub const LM_SCORER_SCHEMA: &str = "neo-ime-lm-scorer-export-v1";
pub const LM_SCORER_FAMILY: &str = "causal_lm";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsArtifact {
    pub path: PathBuf,
    /// The file format; `gguf` (read with llama.cpp) is the only one.
    pub format: String,
    /// One of f32 / f16 / q8_0.
    pub precision: String,
}

/// The LM's own tokenizer, when the weights file's vocabulary would segment
/// otherwise (a Unigram model, which llama.cpp reads as a
/// score-merge vocabulary).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmTokenizerArtifact {
    pub path: PathBuf,
    /// `sentencepiece` (a Unigram `tokenizer.model`) is the only one.
    pub format: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmSpecialIds {
    pub bos: u32,
    pub eos: u32,
}

/// Rerank settings the scorer was verified with. Every field is optional;
/// a value set by the caller wins, and an unset one falls back to the
/// runtime default.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreDefaults {
    /// Left context chars the scorer reads before a candidate.
    pub max_context_chars: Option<usize>,
    /// Weight of the scorer's log-probability against the converter's score.
    pub lambda: Option<f32>,
    /// Candidates reranked at most.
    pub max_candidates: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LmScorerManifestFile {
    #[allow(dead_code)] // checked through the envelope
    schema: String,
    #[allow(dead_code)] // checked through the envelope
    model_family: String,
    weights: WeightsArtifact,
    special_ids: LmSpecialIds,
    #[serde(default)]
    score_defaults: ScoreDefaults,
    #[serde(default)]
    tokenizer: Option<LmTokenizerArtifact>,
}

/// A causal LM that scores conversion candidates (the reranker), its
/// weights (and tokenizer) path resolved against the manifest's directory
/// and checked to exist. Without a tokenizer, the one in the weights file.
#[derive(Debug, Clone)]
pub struct LmScorerManifest {
    pub manifest_path: PathBuf,
    pub weights_path: PathBuf,
    pub format: String,
    pub precision: String,
    pub special_ids: LmSpecialIds,
    pub score_defaults: ScoreDefaults,
    /// a SentencePiece `tokenizer.model`
    pub tokenizer_path: Option<PathBuf>,
}

impl LmScorerManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let (envelope, value) = read_manifest(path)?;
        anyhow::ensure!(
            envelope.schema == LM_SCORER_SCHEMA && envelope.model_family == LM_SCORER_FAMILY,
            "manifest {} is {}/{}, expected {LM_SCORER_SCHEMA}/{LM_SCORER_FAMILY}",
            path.display(),
            envelope.schema,
            envelope.model_family
        );
        let file: LmScorerManifestFile = serde_json::from_value(value)
            .with_context(|| format!("parse LM scorer manifest {}", path.display()))?;
        anyhow::ensure!(
            file.weights.format == "gguf",
            "manifest {}: unknown weights format {:?}",
            path.display(),
            file.weights.format
        );
        anyhow::ensure!(
            matches!(file.weights.precision.as_str(), "f32" | "f16" | "q8_0"),
            "manifest {}: unknown weights precision {:?}",
            path.display(),
            file.weights.precision
        );
        let weights_path = file_within(path, &file.weights.path, "weights")?;
        let tokenizer_path = match &file.tokenizer {
            Some(t) => {
                anyhow::ensure!(
                    t.format == "sentencepiece",
                    "manifest {}: unknown tokenizer format {:?}",
                    path.display(),
                    t.format
                );
                Some(file_within(path, &t.path, "tokenizer")?)
            }
            None => None,
        };
        Ok(Self {
            manifest_path: path.to_path_buf(),
            weights_path,
            format: file.weights.format,
            precision: file.weights.precision,
            special_ids: file.special_ids,
            score_defaults: file.score_defaults,
            tokenizer_path,
        })
    }
}

pub const ANALYZER_SCHEMA: &str = "neo-ime-analyzer-export-v1";
pub const ANALYZER_FAMILY: &str = "lattice_crf";

/// The analyzer's compiled dictionary.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzerDictionaryArtifact {
    pub path: PathBuf,
    /// [`DICTIONARY_VIBRATO`] or [`DICTIONARY_PACKED`].
    pub format: String,
}

/// A vibrato system dictionary, zstd-compressed; its features are columns
/// the manifest's `features` name.
pub const DICTIONARY_VIBRATO: &str = "vibrato-zstd";
/// One in vibrato-packed's layout (`vendor/vibrato-packed`), zstd-compressed;
/// its features are codes the manifest's `packed` table reads.
pub const DICTIONARY_PACKED: &str = "vibrato-packed-zstd";

/// What a packed dictionary's codes stand for. A word's features are one
/// code char for its part of speech, then its reading: `=` the surface in
/// hiragana, `*` none, `~` and the reading as it is, else one code char a
/// kana. The code chars are `setsuna_analyzer::packed::CODE_CHARS`, in
/// order: the `n`th stands for `pos[n]`, or for the `n`th char of `kana`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedCodes {
    /// Each code's (pos1, pos2).
    pub pos: Vec<[String; 2]>,
    pub kana: String,
}

/// Analysis settings the analyzer was verified with. Every field is
/// optional; an unset one falls back to the runtime default.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeDefaults {
    /// Longest unknown word grouped from one character class (0: no limit).
    pub max_grouping_len: Option<usize>,
    /// Other readings of a token that cost the path at most this much more.
    pub alt_max_cost: Option<i32>,
    /// Other readings per token at most.
    pub max_alts: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalyzerManifestFile {
    #[allow(dead_code)] // checked through the envelope
    schema: String,
    #[allow(dead_code)] // checked through the envelope
    model_family: String,
    dictionary: AnalyzerDictionaryArtifact,
    /// The other readings' table (each surface's entries and the connection
    /// costs), when the analyzer gives them.
    #[serde(default)]
    alternates: Option<PathBuf>,
    /// A `vibrato-zstd` dictionary's feature columns by name, in order;
    /// `reading` (hiragana, `*` when unknown) is required.
    #[serde(default)]
    features: Option<Vec<String>>,
    /// A `vibrato-packed-zstd` dictionary's codes.
    #[serde(default)]
    packed: Option<PackedCodes>,
    #[serde(default)]
    analyze_defaults: AnalyzeDefaults,
}

/// A morphological analyzer (a lattice of dictionary words and a CRF's
/// costs), its files resolved against the manifest's directory and checked
/// to exist.
#[derive(Debug, Clone)]
pub struct AnalyzerManifest {
    pub manifest_path: PathBuf,
    pub dictionary_path: PathBuf,
    /// [`DICTIONARY_VIBRATO`] or [`DICTIONARY_PACKED`].
    pub dictionary_format: String,
    pub alternates_path: Option<PathBuf>,
    /// A `vibrato-zstd` dictionary's feature columns (empty for a packed one).
    pub features: Vec<String>,
    /// A `vibrato-packed-zstd` dictionary's codes.
    pub packed: Option<PackedCodes>,
    pub analyze_defaults: AnalyzeDefaults,
}

impl AnalyzerManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let (envelope, value) = read_manifest(path)?;
        anyhow::ensure!(
            envelope.schema == ANALYZER_SCHEMA && envelope.model_family == ANALYZER_FAMILY,
            "manifest {} is {}/{}, expected {ANALYZER_SCHEMA}/{ANALYZER_FAMILY}",
            path.display(),
            envelope.schema,
            envelope.model_family
        );
        let file: AnalyzerManifestFile = serde_json::from_value(value)
            .with_context(|| format!("parse analyzer manifest {}", path.display()))?;
        match file.dictionary.format.as_str() {
            DICTIONARY_VIBRATO => {
                anyhow::ensure!(
                    file.packed.is_none(),
                    "manifest {}: codes for a {DICTIONARY_VIBRATO} dictionary",
                    path.display()
                );
                anyhow::ensure!(
                    file.features
                        .as_ref()
                        .is_some_and(|f| f.iter().any(|f| f == "reading")),
                    "manifest {}: features name no reading",
                    path.display()
                );
            }
            DICTIONARY_PACKED => {
                anyhow::ensure!(
                    file.features.is_none() && file.alternates.is_none(),
                    "manifest {}: features or alternates for a {DICTIONARY_PACKED} dictionary",
                    path.display()
                );
                anyhow::ensure!(
                    file.packed.is_some(),
                    "manifest {}: a {DICTIONARY_PACKED} dictionary without its codes",
                    path.display()
                );
            }
            other => anyhow::bail!(
                "manifest {}: unknown dictionary format {other:?}",
                path.display()
            ),
        }
        let resolve = |p: &Path, what: &str| file_within(path, p, what);
        Ok(Self {
            manifest_path: path.to_path_buf(),
            dictionary_path: resolve(&file.dictionary.path, "dictionary")?,
            dictionary_format: file.dictionary.format,
            alternates_path: file
                .alternates
                .as_deref()
                .map(|p| resolve(p, "alternates"))
                .transpose()?,
            features: file.features.unwrap_or_default(),
            packed: file.packed,
            analyze_defaults: file.analyze_defaults,
        })
    }

    /// The column of feature `name`.
    pub fn feature(&self, name: &str) -> Option<usize> {
        self.features.iter().position(|f| f == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("setsuna-artifact-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manifest_body(proposal: &str, vocab: &str, extra: &str) -> String {
        format!(
            r#"{{"schema":"{CTC_NAT_SCHEMA}","model_family":"ctc_nat",
            "proposal":{{"path":"{proposal}","precision":"fp32"}},
            "vocab":"{vocab}",
            "special_ids":{{"pad":0,"unk":1,"cls":2,"sep":3,"blank":4,"mask":5}},
            "max_seq_len":128{extra}}}"#
        )
    }

    #[test]
    fn loads_and_resolves_relative_paths() {
        let dir = temp_dir("ok");
        write(&dir, "m.fp32.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        let manifest = write(
            &dir,
            "m.fp32.manifest.json",
            &manifest_body(
                "m.fp32.onnx",
                "m.vocab.hex.tsv",
                r#","decode_defaults":{"beam_width":20,"symbol_token":true,"digit_split":true}"#,
            ),
        );
        let m = CtcNatManifest::load(&manifest).unwrap();
        let canon = |p: PathBuf| std::fs::canonicalize(p).unwrap();
        assert_eq!(m.proposal_path, Some(canon(dir.join("m.fp32.onnx"))));
        assert!(m.proposal_path.as_ref().unwrap().is_absolute());
        assert_eq!(m.special_ids.blank, 4);
        assert_eq!(m.decode_defaults.beam_width, Some(20));
        assert_eq!(m.decode_defaults.top_k, None);
        assert_eq!(m.decode_defaults.symbol_token, Some(true));
        assert_eq!(m.decode_defaults.digit_split, Some(true));
        assert_eq!(m.logit_ids, None);
    }

    /// A proposal graph's logit columns: read when ascending and distinct, refused otherwise.
    #[test]
    fn loads_the_logit_ids() {
        let dir = temp_dir("cols");
        write(&dir, "m.int8.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        let with = |cols: &str| {
            let body = manifest_body("m.int8.onnx", "m.vocab.hex.tsv", "").replace(
                r#""precision":"fp32"}"#,
                &format!(r#""precision":"int8","logit_ids":{cols}}}"#),
            );
            CtcNatManifest::load(&write(&dir, "m.int8.manifest.json", &body))
        };
        assert_eq!(with("[4, 6, 7]").unwrap().logit_ids, Some(vec![4, 6, 7]));
        assert!(with("[6, 4]").is_err());
        assert!(with("[4, 4]").is_err());
        assert!(with("[]").is_err());
    }

    /// The encoder's weights for the engine's own implementation: resolved on an int8 proposal; refused on another
    /// precision, missing, or beside cut logit columns.
    #[test]
    fn loads_the_native_encoder() {
        let dir = temp_dir("native-enc");
        write(&dir, "m.int8.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        write(&dir, "m.encoder.native.safetensors", "");
        let with = |precision: &str, native: &str| {
            let body = manifest_body("m.int8.onnx", "m.vocab.hex.tsv", "").replace(
                r#""precision":"fp32"}"#,
                &format!(r#""precision":"{precision}",{native}}}"#),
            );
            CtcNatManifest::load(&write(&dir, "m.int8.manifest.json", &body))
        };
        let native =
            |file: &str, p: &str| format!(r#""native":{{"path":"{file}","precision":"{p}"}}"#);
        let m = with("int8", &native("m.encoder.native.safetensors", "int8")).unwrap();
        assert_eq!(
            m.proposal_native_path,
            Some(std::fs::canonicalize(dir.join("m.encoder.native.safetensors")).unwrap())
        );
        assert!(with("fp32", &native("m.encoder.native.safetensors", "int8")).is_err());
        assert!(with("int8", &native("m.encoder.native.safetensors", "fp32")).is_err());
        assert!(with("int8", &native("missing.safetensors", "int8")).is_err());
        // a cut head (a reading corrector's): its native weights hold those columns
        assert_eq!(
            with(
                "int8",
                &format!(
                    r#"{},"logit_ids":[4,6]"#,
                    native("m.encoder.native.safetensors", "int8")
                )
            )
            .unwrap()
            .logit_ids,
            Some(vec![4, 6])
        );
        assert_eq!(
            with("int8", r#""logit_ids":[4,6]"#)
                .unwrap()
                .proposal_native_path,
            None
        );
    }

    /// A package without its ONNX graphs: the proposal by its native weights alone, the AR step by its native step
    /// alone; neither a graph nor native weights refused.
    #[test]
    fn loads_without_the_graphs() {
        let dir = temp_dir("no-graphs");
        write(&dir, "m.vocab.hex.tsv", "");
        write(&dir, "m.encoder.safetensors", "");
        write(&dir, "m.ar_step.safetensors", "");
        let with = |proposal: &str, extra: &str| {
            let body = manifest_body("unused.onnx", "m.vocab.hex.tsv", extra).replace(
                r#""proposal":{"path":"unused.onnx","precision":"fp32"}"#,
                proposal,
            );
            CtcNatManifest::load(&write(&dir, "m.int8.manifest.json", &body))
        };
        let native = r#""proposal":{"precision":"int8","native":{"path":"m.encoder.safetensors","precision":"int8"}}"#;
        let m = with(native, "").unwrap();
        assert_eq!(
            (m.proposal_path.is_none(), m.proposal_native_path.is_some()),
            (true, true)
        );
        assert!(with(r#""proposal":{"precision":"int8"}"#, "").is_err());
        let ar = |step: &str, native: &str| {
            format!(
                r#","ar_decoder":{{{step}{native}"layers":1,"heads":6,"head_dim":32,"max_positions":128,"bos_id":3,
                "eos_id":2,"top_k":32}}"#
            )
        };
        let a = with(
            native,
            &ar(
                "",
                r#""native":{"path":"m.ar_step.safetensors","precision":"int8"},"#,
            ),
        )
        .unwrap()
        .ar_decoder
        .unwrap();
        assert_eq!(
            (a.step_path.is_none(), a.native_path.is_some()),
            (true, true)
        );
        assert!(with(native, &ar("", "")).is_err());
        assert!(with(native, &ar(r#""step":{"precision":"int8"},"#, "")).is_err());
    }

    /// A reading corrector's own manifest: its threshold in decode_defaults.corrector_threshold, absent by default.
    #[test]
    fn loads_the_corrector_threshold() {
        let dir = temp_dir("corrector-threshold");
        write(&dir, "m.int8.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        let with = |defaults: &str| {
            let body = manifest_body(
                "m.int8.onnx",
                "m.vocab.hex.tsv",
                &format!(r#","decode_defaults":{defaults}"#),
            );
            CtcNatManifest::load(&write(&dir, "m.int8.manifest.json", &body))
        };
        assert_eq!(
            with(r#"{"max_context":0,"corrector_threshold":11.5}"#)
                .unwrap()
                .decode_defaults
                .corrector_threshold,
            Some(11.5)
        );
        assert_eq!(
            with(r#"{"max_context":0}"#)
                .unwrap()
                .decode_defaults
                .corrector_threshold,
            None
        );
    }

    /// A reading corrector: its graph resolved, the converter's vocabulary and special ids, its own decode defaults
    /// and threshold; a missing graph, external data or bad columns refused.
    #[test]
    fn loads_the_corrector_section() {
        let dir = temp_dir("corrector");
        write(&dir, "m.int8.onnx", "");
        write(&dir, "m.corrector.int8head.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        let with = |section: &str| {
            let body = manifest_body(
                "m.int8.onnx",
                "m.vocab.hex.tsv",
                &format!(r#","corrector":{section}"#),
            );
            CtcNatManifest::load(&write(&dir, "m.int8.manifest.json", &body))
        };
        let m = with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8","logit_ids":[4,6]},
            "max_seq_len":128,"decode_defaults":{"max_context":0,"symbol_token":true},"threshold":11.5}"#,
        )
        .unwrap();
        let c = m.corrector.as_ref().unwrap();
        let canon = |p: PathBuf| std::fs::canonicalize(p).unwrap();
        assert_eq!(
            c.manifest.proposal_path,
            Some(canon(dir.join("m.corrector.int8head.onnx")))
        );
        assert_eq!(c.manifest.vocab_path, m.vocab_path);
        assert_eq!(c.manifest.special_ids, m.special_ids);
        assert_eq!(c.manifest.logit_ids, Some(vec![4, 6]));
        assert_eq!(c.manifest.decode_defaults.max_context, Some(0));
        assert!(c.manifest.corrector.is_none() && c.manifest.ar_decoder.is_none());
        assert_eq!(c.threshold, 11.5);
        assert_eq!(c.manifest.proposal_native_path, None);
        // its native weights, as a model's
        write(&dir, "m.corrector.native.safetensors", "");
        let m = with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8","logit_ids":[4,6],
            "native":{"path":"m.corrector.native.safetensors","precision":"int8"}},"max_seq_len":128,"threshold":1}"#,
        )
        .unwrap();
        assert_eq!(
            m.corrector.unwrap().manifest.proposal_native_path,
            Some(canon(dir.join("m.corrector.native.safetensors")))
        );
        assert!(with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8","logit_ids":[4,6],
            "native":{"path":"none.safetensors","precision":"int8"}},"max_seq_len":128,"threshold":1}"#,
        )
        .is_err());
        assert!(with(r#"{"proposal":{"path":"none.onnx","precision":"int8"},"max_seq_len":128,"threshold":1}"#)
            .is_err());
        assert!(with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8","logit_ids":[6,4]},
            "max_seq_len":128,"threshold":1}"#
        )
        .is_err());
        assert!(with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8","external_data":["x"]},
            "max_seq_len":128,"threshold":1}"#
        )
        .is_err());
        assert!(with(
            r#"{"proposal":{"path":"m.corrector.int8head.onnx","precision":"int8"},"threshold":1}"#
        )
        .is_err());
    }

    /// An AR decoder section: its step graph resolved with its external data;
    /// none on a CTC manifest; a missing step graph or a zero size refused.
    #[test]
    fn loads_the_ar_decoder_section() {
        let dir = temp_dir("ar");
        write(&dir, "m.fp32.onnx", "");
        write(&dir, "m.vocab.hex.tsv", "");
        write(&dir, "m.ar_step.fp32.onnx", "");
        write(&dir, "m.ar_step.fp32.onnx.data", "");
        let section = |step: &str, layers: usize| {
            format!(
                r#","ar_decoder":{{"step":{{"path":"{step}","precision":"fp32","external_data":["m.ar_step.fp32.onnx.data"]}},
                "layers":{layers},"heads":6,"head_dim":32,"max_positions":128,"bos_id":3,"eos_id":2,"top_k":32}}"#
            )
        };
        let ok = write(
            &dir,
            "ok.json",
            &manifest_body(
                "m.fp32.onnx",
                "m.vocab.hex.tsv",
                &section("m.ar_step.fp32.onnx", 2),
            ),
        );
        let m = CtcNatManifest::load(&ok).unwrap();
        let ar = m.ar_decoder.expect("ar_decoder");
        assert_eq!(
            ar.step_path,
            Some(std::fs::canonicalize(dir.join("m.ar_step.fp32.onnx")).unwrap())
        );
        assert_eq!(
            (
                ar.layers,
                ar.heads,
                ar.head_dim,
                ar.max_positions,
                ar.bos_id,
                ar.eos_id,
                ar.top_k
            ),
            (2, 6, 32, 128, 3, 2, 32)
        );
        assert!(ar.native_path.is_none());
        // the native step's weights: resolved; int8 only; a missing file refused
        write(&dir, "m.ar_step.native.safetensors", "");
        let native = |file: &str, precision: &str| {
            section("m.ar_step.fp32.onnx", 1).replacen(
                r#""layers""#,
                &format!(r#""native":{{"path":"{file}","precision":"{precision}"}},"layers""#),
                1,
            )
        };
        let with = write(
            &dir,
            "native.json",
            &manifest_body(
                "m.fp32.onnx",
                "m.vocab.hex.tsv",
                &native("m.ar_step.native.safetensors", "int8"),
            ),
        );
        assert_eq!(
            CtcNatManifest::load(&with)
                .unwrap()
                .ar_decoder
                .unwrap()
                .native_path,
            Some(std::fs::canonicalize(dir.join("m.ar_step.native.safetensors")).unwrap())
        );
        for (name, file, precision) in [
            ("fp.json", "m.ar_step.native.safetensors", "fp32"),
            ("gone.json", "none", "int8"),
        ] {
            let bad = write(
                &dir,
                name,
                &manifest_body("m.fp32.onnx", "m.vocab.hex.tsv", &native(file, precision)),
            );
            assert!(CtcNatManifest::load(&bad).is_err(), "{name}");
        }
        // the script latent: read as the research export writes it; another mode or class order refused; none without it
        assert!(!ar.script);
        let script = |mode: &str, classes: &str| {
            section("m.ar_step.fp32.onnx", 1).replacen(
                r#""layers""#,
                &format!(r#""script":{{"mode":"{mode}","classes":{classes}}},"layers""#),
                1,
            )
        };
        let order = r#"["other","kanji","hiragana","katakana"]"#;
        let with = write(
            &dir,
            "script.json",
            &manifest_body("m.fp32.onnx", "m.vocab.hex.tsv", &script("choice", order)),
        );
        assert!(
            CtcNatManifest::load(&with)
                .unwrap()
                .ar_decoder
                .unwrap()
                .script
        );
        for (name, mode, classes) in [
            ("within.json", "within", order),
            (
                "order.json",
                "choice",
                r#"["other","hiragana","kanji","katakana"]"#,
            ),
        ] {
            let bad = write(
                &dir,
                name,
                &manifest_body("m.fp32.onnx", "m.vocab.hex.tsv", &script(mode, classes)),
            );
            assert!(CtcNatManifest::load(&bad).is_err(), "{name}");
        }
        let plain = write(
            &dir,
            "plain.json",
            &manifest_body("m.fp32.onnx", "m.vocab.hex.tsv", ""),
        );
        assert!(CtcNatManifest::load(&plain).unwrap().ar_decoder.is_none());
        let missing = write(
            &dir,
            "missing.json",
            &manifest_body("m.fp32.onnx", "m.vocab.hex.tsv", &section("none.onnx", 1)),
        );
        assert!(CtcNatManifest::load(&missing).is_err());
        let zero = write(
            &dir,
            "zero.json",
            &manifest_body(
                "m.fp32.onnx",
                "m.vocab.hex.tsv",
                &section("m.ar_step.fp32.onnx", 0),
            ),
        );
        assert!(CtcNatManifest::load(&zero).is_err());
    }

    /// The fixture the Python writer test reproduces byte for byte.
    #[test]
    fn loads_shared_fixture() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctc_nat_manifest");
        let m = CtcNatManifest::load(&dir.join("model.int4.manifest.json")).unwrap();
        let canon = |p: PathBuf| std::fs::canonicalize(p).unwrap();
        assert_eq!(m.proposal_path, Some(canon(dir.join("model.int4.onnx"))));
        assert_eq!(m.precision, "int4");
        assert_eq!(
            m.vocab_path,
            canon(dir.join("model.fp32.tokenizer.json.vocab.hex.tsv"))
        );
        assert_eq!(
            m.special_ids,
            SpecialIds {
                pad: 0,
                unk: 1,
                cls: 3,
                sep: 2,
                blank: 4,
                mask: 5
            }
        );
        assert_eq!(m.max_seq_len, 128);
        let d = &m.decode_defaults;
        assert_eq!(d.beam_width, Some(20));
        assert_eq!(d.vocab_blacklist, Some(vec![724, 730]));
        assert_eq!(d.symbol_split, Some(true));
        assert_eq!(d.symbol_token, None);
        assert_eq!(d.digit_split, None);
        assert_eq!(d.top_k, None);
    }

    #[test]
    fn rejects_other_schema_missing_files_and_unknown_fields() {
        let dir = temp_dir("bad");
        let other = write(
            &dir,
            "other.json",
            r#"{"schema":"neo-ime-ctc-lm-export-v1","model_family":"ctc_lm"}"#,
        );
        assert!(CtcNatManifest::load(&other).is_err());
        let missing = write(
            &dir,
            "missing.json",
            &manifest_body("absent.onnx", "absent.tsv", ""),
        );
        assert!(CtcNatManifest::load(&missing).is_err());
        write(&dir, "p.onnx", "");
        write(&dir, "v.tsv", "");
        let unknown = write(
            &dir,
            "unknown.json",
            &manifest_body("p.onnx", "v.tsv", r#","refine":"x""#),
        );
        assert!(CtcNatManifest::load(&unknown).is_err());
    }

    /// The LM scorer fixture the Python writer test reproduces byte for byte.
    #[test]
    fn loads_shared_lm_scorer_fixture() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/lm_scorer_manifest");
        let m = LmScorerManifest::load(&dir.join("model.q8_0.manifest.json")).unwrap();
        assert_eq!(m.weights_path, dir.join("model.q8_0.gguf"));
        assert_eq!((m.format.as_str(), m.precision.as_str()), ("gguf", "q8_0"));
        assert_eq!(m.special_ids, LmSpecialIds { bos: 1, eos: 2 });
        assert_eq!(
            m.score_defaults,
            ScoreDefaults {
                max_context_chars: Some(40),
                lambda: Some(0.3),
                max_candidates: Some(20),
            }
        );
    }

    #[test]
    fn lm_scorer_refuses_other_schemas_formats_and_unknown_fields() {
        let dir = temp_dir("lm");
        write(&dir, "w.gguf", "");
        let body = |weights: &str, extra: &str| {
            format!(
                r#"{{"schema":"{LM_SCORER_SCHEMA}","model_family":"causal_lm",
                "weights":{weights},"special_ids":{{"bos":1,"eos":2}}{extra}}}"#
            )
        };
        let ok = write(
            &dir,
            "ok.json",
            &body(r#"{"path":"w.gguf","format":"gguf","precision":"f16"}"#, ""),
        );
        let m = LmScorerManifest::load(&ok).unwrap();
        assert_eq!(m.score_defaults, ScoreDefaults::default());
        assert_eq!(m.tokenizer_path, None);
        // the LM's own SentencePiece tokenizer, beside the weights
        write(&dir, "tokenizer.model", "");
        let w = r#"{"path":"w.gguf","format":"gguf","precision":"f16"}"#;
        let sp = write(
            &dir,
            "sp.json",
            &body(
                w,
                r#","tokenizer":{"path":"tokenizer.model","format":"sentencepiece"}"#,
            ),
        );
        assert_eq!(
            LmScorerManifest::load(&sp).unwrap().tokenizer_path,
            Some(dir.join("tokenizer.model"))
        );
        for (name, extra) in [
            (
                "tok-absent.json",
                r#","tokenizer":{"path":"x.model","format":"sentencepiece"}"#,
            ),
            (
                "tok-format.json",
                r#","tokenizer":{"path":"tokenizer.model","format":"hf"}"#,
            ),
        ] {
            let m = write(&dir, name, &body(w, extra));
            assert!(LmScorerManifest::load(&m).is_err(), "{name}");
        }
        for (name, weights, extra) in [
            (
                "onnx.json",
                r#"{"path":"w.gguf","format":"onnx","precision":"f16"}"#,
                "",
            ),
            (
                "q4.json",
                r#"{"path":"w.gguf","format":"gguf","precision":"q4_0"}"#,
                "",
            ),
            (
                "absent.json",
                r#"{"path":"x.gguf","format":"gguf","precision":"f16"}"#,
                "",
            ),
            (
                "unknown.json",
                r#"{"path":"w.gguf","format":"gguf","precision":"f16"}"#,
                r#","score_defaults":{"beam":3}"#,
            ),
        ] {
            let m = write(&dir, name, &body(weights, extra));
            assert!(LmScorerManifest::load(&m).is_err(), "{name}");
        }
        // a CTC-NAT manifest is not a scorer, nor the other way round
        let ctc = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ctc_nat_manifest/model.int4.manifest.json");
        assert!(LmScorerManifest::load(&ctc).is_err());
        assert!(CtcNatManifest::load(&ok).is_err());
        // a file out of the manifest's directory, however it is named
        let inner = dir.join("inner");
        std::fs::create_dir_all(&inner).unwrap();
        let outside = write(&dir, "outside.gguf", "");
        let outside = outside.to_string_lossy().replace('\\', "/");
        for path in [
            "../outside.gguf",
            "./../outside.gguf",
            outside.as_str(),
            "//host/share/w.gguf",
            "",
        ] {
            let weights = format!(r#"{{"path":"{path}","format":"gguf","precision":"f16"}}"#);
            let m = write(&inner, "out.json", &body(&weights, ""));
            assert!(LmScorerManifest::load(&m).is_err(), "{path}");
        }
        // one below it is fine
        write(&inner, "sub.gguf", "");
        let below = write(
            &dir,
            "below.json",
            &body(
                r#"{"path":"inner/sub.gguf","format":"gguf","precision":"f16"}"#,
                "",
            ),
        );
        assert!(LmScorerManifest::load(&below).is_ok());
        // a link to a share is refused before it is opened (where links can
        // be made: Windows needs the privilege or developer mode)
        let share = if cfg!(windows) {
            r"\\share-host.invalid\models\w.gguf"
        } else {
            "//share-host.invalid/models/w.gguf"
        };
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(share, dir.join("linked.gguf"));
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(share, dir.join("linked.gguf"));
        if linked.is_ok() {
            let m = write(
                &dir,
                "linked.json",
                &body(
                    r#"{"path":"linked.gguf","format":"gguf","precision":"f16"}"#,
                    "",
                ),
            );
            let e = LmScorerManifest::load(&m).unwrap_err();
            assert!(format!("{e:#}").contains("network share"), "{e:#}");
        }
    }

    #[test]
    fn refuses_manifests_verified_with_dictionary_or_ngram_fusion() {
        let dir = temp_dir("removed");
        write(&dir, "p.onnx", "");
        write(&dir, "v.tsv", "");
        for extra in [
            r#","decode_defaults":{"beam_width":20,"dict_gamma":0.5}"#,
            r#","ngram":"lm.bin""#,
        ] {
            let m = write(&dir, "m.json", &manifest_body("p.onnx", "v.tsv", extra));
            let err = CtcNatManifest::load(&m).unwrap_err().to_string();
            assert!(err.contains("re-export"), "{err}");
        }
    }

    #[test]
    fn analyzer_loads_and_refuses_bad_manifests() {
        let dir = temp_dir("analyzer");
        write(&dir, "system.dic.zst", "");
        write(&dir, "alternates.bin", "");
        let body = |extra: &str| {
            format!(
                r#"{{"schema":"{ANALYZER_SCHEMA}","model_family":"lattice_crf",
                "dictionary":{{"path":"system.dic.zst","format":"vibrato-zstd"}},
                "features":["pos1","pos2","cType","cForm","reading","orth"]{extra}}}"#
            )
        };
        let m = write(
            &dir,
            "ok.json",
            &body(r#","alternates":"alternates.bin","analyze_defaults":{"max_alts":8}"#),
        );
        let a = AnalyzerManifest::load(&m).unwrap();
        assert_eq!(a.dictionary_path, dir.join("system.dic.zst"));
        assert_eq!(a.alternates_path, Some(dir.join("alternates.bin")));
        assert_eq!(a.feature("reading"), Some(4));
        assert_eq!(a.analyze_defaults.max_alts, Some(8));
        let bare = write(&dir, "bare.json", &body(""));
        assert_eq!(AnalyzerManifest::load(&bare).unwrap().alternates_path, None);
        for (name, text) in [
            ("absent.json", body(r#","alternates":"x.bin""#)),
            ("format.json", body("").replace("vibrato-zstd", "mecab")),
            ("noreading.json", body("").replace("\"reading\",", "")),
            ("unknown.json", body(r#","extra":1"#)),
            ("schema.json", body("").replace("lattice_crf", "ctc_nat")),
            (
                "codes.json",
                body(r#","packed":{"pos":[["名詞","普通名詞"]],"kana":"あ"}"#),
            ),
        ] {
            let m = write(&dir, name, &text);
            assert!(AnalyzerManifest::load(&m).is_err(), "{name}");
        }
    }

    #[test]
    fn a_packed_analyzer_has_codes_and_no_features() {
        let dir = temp_dir("packed");
        write(&dir, "system.dic.zst", "");
        write(&dir, "alternates.bin", "");
        let body = |extra: &str| {
            format!(
                r#"{{"schema":"{ANALYZER_SCHEMA}","model_family":"lattice_crf",
                "dictionary":{{"path":"system.dic.zst","format":"{DICTIONARY_PACKED}"}}{extra}}}"#
            )
        };
        let codes = r#","packed":{"pos":[["名詞","普通名詞"],["助詞","係助詞"]],"kana":"あい"}"#;
        let m = write(&dir, "ok.json", &body(codes));
        let a = AnalyzerManifest::load(&m).unwrap();
        assert_eq!(a.dictionary_format, DICTIONARY_PACKED);
        assert!(a.features.is_empty());
        let p = a.packed.unwrap();
        assert_eq!(p.pos[1], ["助詞".to_string(), "係助詞".to_string()]);
        assert_eq!(p.kana, "あい");
        for (name, text) in [
            ("nocodes.json", body("")),
            (
                "features.json",
                body(&format!(r#"{codes},"features":["pos1","pos2","reading"]"#)),
            ),
            (
                "alternates.json",
                body(&format!(r#"{codes},"alternates":"alternates.bin""#)),
            ),
        ] {
            let m = write(&dir, name, &text);
            assert!(AnalyzerManifest::load(&m).is_err(), "{name}");
        }
    }
}
