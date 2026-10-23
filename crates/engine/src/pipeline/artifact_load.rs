//! Artifact load: the vocab and the proposal graph a CTC-NAT manifest names.

use anyhow::{Context, Result};
use ort::session::builder::SessionBuilder;
use setsuna_artifact::CtcNatManifest;

use crate::pipeline::proposal::ProposalRunner;
use crate::tokenizer::SharedCharTokenizer;

/// Load the tokenizer and the proposal graph, and check that the manifest's
/// special token ids match the vocab.
pub fn load(
    manifest: &CtcNatManifest,
    ort_intra_threads: usize,
) -> Result<(SharedCharTokenizer, ProposalRunner)> {
    crate::runtime::init(None)?;
    let tsv_path = &manifest.vocab_path;
    let mut tokenizer = SharedCharTokenizer::load_vocab_hex_tsv(tsv_path)
        .with_context(|| format!("load tokenizer vocab {}", tsv_path.display()))?;
    if let Some(input) = &manifest.input_vocab_path {
        tokenizer = tokenizer
            .with_input_vocab(input)
            .with_context(|| format!("load input vocab {}", input.display()))?;
    }

    // the fused MoE kernel spreads a layer's experts over as many threads as ONNX Runtime's own ops use, or
    // SETSUNA_MOE_THREADS (1: one after another)
    let moe_threads = std::env::var("SETSUNA_MOE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ort_intra_threads);
    crate::moe_op::set_threads(moe_threads);
    let mut builder = SessionBuilder::new()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(ort_intra_threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        // the research export's fused ops (neo_ime.TokenMoE); a graph without them never calls them
        .with_operators(crate::moe_op::domain().map_err(|e| anyhow::anyhow!("{e}"))?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // SETSUNA_ORT_SPIN=0: ONNX Runtime's intra-op threads sleep between ops instead of spinning (they take cores the
    // MoE kernel's pool would use); research timing
    if std::env::var("SETSUNA_ORT_SPIN").is_ok_and(|v| v == "0") {
        builder = builder
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    // SETSUNA_ORT_PROFILE=<prefix>: ONNX Runtime's per-node timing, written as <prefix>_<date>.json when the session
    // ends (a research measurement; off by default)
    if let Some(prefix) = std::env::var_os("SETSUNA_ORT_PROFILE") {
        builder = builder
            .with_profiling(prefix)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let session = builder
        .commit_from_file(&manifest.proposal_path)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let t = &tokenizer;
    let ids = manifest.special_ids;
    anyhow::ensure!(
        (t.pad_id, t.unk_id, t.cls_id, t.sep_id, t.blank_id, t.mask_id)
            == (ids.pad, ids.unk, ids.cls, ids.sep, ids.blank, ids.mask),
        "manifest {} special_ids {:?} do not match vocab {}",
        manifest.manifest_path.display(),
        ids,
        manifest.vocab_path.display()
    );
    Ok((tokenizer, ProposalRunner::new(session)))
}
