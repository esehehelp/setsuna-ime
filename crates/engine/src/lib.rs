//! Setsuna engine: CTC-NAT inference via ONNX Runtime, and nothing else.
//!
//! A manifest-described proposal graph turns `[CLS] context [SEP] reading`
//! into per-frame log-probabilities; CTC greedy / prefix beam decoding turns
//! those into ranked candidates. Dictionaries, language-model fusion and
//! other IME policy belong to the host, not here.
//!
//! Modules:
//!   - tokenizer: SharedCharTokenizer (`.vocab.hex.tsv` sidecar)
//!   - pipeline: conversion stages (proposal, decode, pool, symsplit)
//!   - normalize: full-width ASCII / punctuation normalization
//!   - cpu: the process pinned to one L3 domain on a Zen CPU with several
//!   - moe_op: the `neo_ime.TokenMoE` fused kernel registered with every session
//!   - norm_op: `neo_ime.AddLayerNorm`, a residual add and the LayerNorm after it in one kernel
//!   - runtime: loading the official ONNX Runtime from an explicit path
//!   - session: EngineSession (manifest load, decode settings, `convert`)

pub mod candidate;
pub mod cpu;
pub mod moe_op;
pub mod norm_op;
pub mod normalize;
pub mod pipeline;
pub mod runtime;
pub mod session;
pub mod tokenizer;

pub use candidate::{Candidate, CandidateSource};
pub use normalize::normalize_candidate;
pub use pipeline::decode::{collapse_frame_ids, prefix_beam_search, BeamHypothesis};
pub use pipeline::proposal::{FrameLogProbs, ProposalFrame, ProposalOutput};
pub use session::{CandidateSet, ConversionRequest, DecodeConfig, EngineSession};
pub use tokenizer::SharedCharTokenizer;
