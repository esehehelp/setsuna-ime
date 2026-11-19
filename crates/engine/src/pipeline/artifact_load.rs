//! Artifact load: the vocab and the weights a CTC-NAT manifest names for the engine's own implementation: the
//! encoder's (`proposal.native`), and for a model with an AR decoder (manifest `ar_decoder`) its step's
//! (`ar_decoder.native`). The graphs the manifest also names (ONNX) are not read: a package without those weights
//! is refused.

use anyhow::{Context, Result};
use setsuna_artifact::CtcNatManifest;

use crate::pipeline::ar::{ArRunner, ArSpec};
use crate::pipeline::ar_native::NativeDecoder;
use crate::pipeline::native_encoder::NativeEncoder;
use crate::pipeline::proposal::ProposalRunner;
use crate::tokenizer::SharedCharTokenizer;

/// The model a manifest names: CTC frames (the proposal graph) or an AR decoder (`pipeline::ar`).
pub enum Model {
    Ctc(ProposalRunner),
    Ar(ArRunner),
}

/// Load the tokenizer and the model's weights, and check that the manifest's
/// special token ids match the vocab. A conversion runs on the calling
/// thread alone.
pub fn load(manifest: &CtcNatManifest) -> Result<(SharedCharTokenizer, Model)> {
    let tsv_path = &manifest.vocab_path;
    let mut tokenizer = SharedCharTokenizer::load_vocab_hex_tsv(tsv_path)
        .with_context(|| format!("load tokenizer vocab {}", tsv_path.display()))?;
    if let Some(input) = &manifest.input_vocab_path {
        tokenizer = tokenizer
            .with_input_vocab(input)
            .with_context(|| format!("load input vocab {}", input.display()))?;
    }

    let path = manifest.proposal_native_path.as_ref().with_context(|| {
        format!(
            "manifest {} names no weights for the engine's own encoder (proposal.native): this engine runs no ONNX \
             graph, the model's package must carry them",
            manifest.manifest_path.display()
        )
    })?;
    let encoder = NativeEncoder::load(path)?;
    let c = &encoder.config;
    let ar = manifest
        .ar_decoder
        .as_ref()
        .map(|a| (a.layers, a.heads, a.head_dim));
    anyhow::ensure!(
        c.vocab == tokenizer.vocab_size() && c.sep_id == tokenizer.sep_id && encoder.ar_shape() == ar,
        "native encoder {} has {} tokens, [SEP] {} and AR layers {:?}; the vocabulary {} tokens, [SEP] {}, \
         the manifest's AR layers {ar:?}",
        path.display(),
        c.vocab,
        c.sep_id,
        encoder.ar_shape(),
        tokenizer.vocab_size(),
        tokenizer.sep_id
    );
    // the logits: every token's, or the manifest's columns (a reading corrector's cut head)
    let columns = manifest
        .logit_ids
        .as_ref()
        .map_or(tokenizer.vocab_size(), Vec::len);
    anyhow::ensure!(
        encoder.ar_shape().is_some() || encoder.logit_width() == columns,
        "native encoder {} gives {} logits a frame, the manifest {columns} (its proposal.logit_ids, or the \
         vocabulary)",
        path.display(),
        encoder.logit_width()
    );
    let encoder = Box::new(encoder);

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
    if let Some(cols) = &manifest.logit_ids {
        anyhow::ensure!(
            cols.last().is_some_and(|&id| (id as usize) < t.vocab_size()) && cols.binary_search(&t.blank_id).is_ok(),
            "manifest {}: proposal.logit_ids must be ids of the vocabulary ({} tokens) and hold the blank ({})",
            manifest.manifest_path.display(),
            t.vocab_size(),
            t.blank_id
        );
    }
    let model = match &manifest.ar_decoder {
        None => Model::Ctc(ProposalRunner::new(encoder).with_logit_ids(manifest.logit_ids.clone())),
        Some(a) => {
            let path = a.native_path.as_ref().with_context(|| {
                format!(
                    "manifest {} names no weights for the engine's own AR step (ar_decoder.native)",
                    manifest.manifest_path.display()
                )
            })?;
            let step = NativeDecoder::load(path)?;
            let c = &step.config;
            anyhow::ensure!(
                (
                    c.layers,
                    c.heads,
                    c.head_dim,
                    c.max_positions,
                    c.bos_id,
                    c.eos_id,
                    c.vocab,
                    step.script()
                ) == (
                    a.layers,
                    a.heads,
                    a.head_dim,
                    a.max_positions,
                    a.bos_id,
                    a.eos_id,
                    tokenizer.vocab_size(),
                    a.script
                ),
                "native AR step {} is {c:?}, the manifest's decoder {a:?} with {} tokens",
                path.display(),
                tokenizer.vocab_size()
            );
            let spec = ArSpec {
                layers: a.layers,
                heads: a.heads,
                head_dim: a.head_dim,
                max_positions: a.max_positions,
                bos: a.bos_id,
                eos: a.eos_id,
                top_k: a.top_k,
                script: a.script,
            };
            Model::Ar(ArRunner::new(encoder, step, spec))
        }
    };
    Ok((tokenizer, model))
}
