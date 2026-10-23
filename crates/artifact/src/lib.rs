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
    pub path: PathBuf,
    /// One of fp32 / fp16 / int8 / int4. One manifest names exactly one
    /// proposal graph; another precision is another manifest.
    pub precision: String,
    /// Weight files the graph stores outside itself (`*.onnx.data`).
    /// onnxruntime resolves them from the graph; they are listed so the
    /// artifact's file set is complete.
    #[serde(default)]
    pub external_data: Vec<PathBuf>,
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
}

/// A CTC-NAT manifest with every artifact path resolved against the
/// manifest's directory and checked to exist.
#[derive(Debug, Clone)]
pub struct CtcNatManifest {
    pub manifest_path: PathBuf,
    pub proposal_path: PathBuf,
    pub precision: String,
    pub vocab_path: PathBuf,
    /// See `CtcNatManifestFile::input_vocab`.
    pub input_vocab_path: Option<PathBuf>,
    pub special_ids: SpecialIds,
    pub max_seq_len: usize,
    pub decode_defaults: DecodeDefaults,
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
            "manifest {} sets {removed:?}: dictionary / n-gram fusion is not part of the              engine any more; re-export the model (or run it with the engine at tag              engine-v1-final)",
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
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let resolve = |p: &Path, what: &str| -> Result<PathBuf> {
            let full = base.join(p);
            anyhow::ensure!(
                full.is_file(),
                "manifest {}: {what} {} not found",
                path.display(),
                full.display()
            );
            Ok(full)
        };
        for extra in &file.proposal.external_data {
            resolve(extra, "external data")?;
        }
        Ok(Self {
            manifest_path: path.to_path_buf(),
            proposal_path: resolve(&file.proposal.path, "proposal")?,
            precision: file.proposal.precision,
            vocab_path: resolve(&file.vocab, "vocab")?,
            input_vocab_path: file
                .input_vocab
                .as_deref()
                .map(|p| resolve(p, "input vocab"))
                .transpose()?,
            special_ids: file.special_ids,
            max_seq_len: file.max_seq_len,
            decode_defaults: file.decode_defaults,
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
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let weights_path = base.join(&file.weights.path);
        anyhow::ensure!(
            weights_path.is_file(),
            "manifest {}: weights {} not found",
            path.display(),
            weights_path.display()
        );
        let tokenizer_path = match &file.tokenizer {
            Some(t) => {
                anyhow::ensure!(
                    t.format == "sentencepiece",
                    "manifest {}: unknown tokenizer format {:?}",
                    path.display(),
                    t.format
                );
                let p = base.join(&t.path);
                anyhow::ensure!(
                    p.is_file(),
                    "manifest {}: tokenizer {} not found",
                    path.display(),
                    p.display()
                );
                Some(p)
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
        assert_eq!(m.proposal_path, dir.join("m.fp32.onnx"));
        assert_eq!(m.special_ids.blank, 4);
        assert_eq!(m.decode_defaults.beam_width, Some(20));
        assert_eq!(m.decode_defaults.top_k, None);
        assert_eq!(m.decode_defaults.symbol_token, Some(true));
        assert_eq!(m.decode_defaults.digit_split, Some(true));
    }

    /// The fixture the Python writer test reproduces byte for byte.
    #[test]
    fn loads_shared_fixture() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctc_nat_manifest");
        let m = CtcNatManifest::load(&dir.join("model.int4.manifest.json")).unwrap();
        assert_eq!(m.proposal_path, dir.join("model.int4.onnx"));
        assert_eq!(m.precision, "int4");
        assert_eq!(
            m.vocab_path,
            dir.join("model.fp32.tokenizer.json.vocab.hex.tsv")
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
}
