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

    let session = SessionBuilder::new()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(ort_intra_threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?
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
