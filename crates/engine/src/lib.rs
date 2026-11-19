//! Setsuna engine: CTC-NAT inference in its own implementation (no ONNX
//! Runtime), and nothing else.
//!
//! A manifest-described model (its weights for the engine's own encoder, and
//! an AR decoder's step when it has one) turns `[CLS] context [SEP] reading`
//! into per-frame log-probabilities; CTC greedy / prefix beam decoding turns
//! those into ranked candidates (or an AR decoder searches its own, which a
//! CTC model may score too: the session's fusion). Dictionaries,
//! language-model fusion and other IME policy belong to the host, not here.
//!
//! Modules:
//!   - tokenizer: SharedCharTokenizer (`.vocab.hex.tsv` sidecar)
//!   - pipeline: conversion stages (proposal, decode, pool, symsplit)
//!   - normalize: full-width ASCII / punctuation normalization
//!   - cpu: the vector instructions the kernels use, as the running CPU has them
//!   - affinity: the process pinned to one L3 domain on a Zen CPU with several (off by default)
//!   - moe_op: the int8 GEMM (u8 x i8) and the MoE helpers the encoder's kernels share
//!   - norm_op: a residual add and the LayerNorm after it in one kernel
//!   - group_q: Linears and token-routed experts on low-bit codes with a scale and a zero point per group of inputs
//!   - session: EngineSession (manifest load, decode settings, `convert`)

pub mod affinity;
pub mod candidate;
pub mod cpu;
pub mod group_q;
pub mod moe_op;
pub mod norm_op;
pub mod normalize;
pub mod pipeline;
pub mod session;
pub mod tokenizer;

pub use candidate::{Candidate, CandidateSource};
pub use normalize::normalize_candidate;
pub use pipeline::decode::{collapse_frame_ids, prefix_beam_search, BeamHypothesis};
pub use pipeline::proposal::{FrameLogProbs, ProposalFrame, ProposalOutput};
pub use session::{CandidateSet, ConversionRequest, DecodeConfig, EngineSession};
pub use tokenizer::SharedCharTokenizer;
