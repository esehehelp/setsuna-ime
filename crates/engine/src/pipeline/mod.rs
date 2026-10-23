//! The conversion pipeline, one module per stage:
//!
//!   proposal (encode + ONNX forward + log-softmax)
//!   -> decode (greedy / prefix beam, with a CTC-CRF's transitions when the
//!      artifact has them)
//!   -> pool (detokenize, normalize, dedup, fallback)
//!
//! with symbol split wrapping the whole pipeline per kana span (or symbol
//! token turning each run of symbols into one token and back), digit split
//! taking the runs of digits out of the reading around that, and chunk
//! wrapping that per chunk of a long reading.

pub mod artifact_load;
pub mod chunk;
pub mod crf;
pub mod decode;
pub mod digitsplit;
pub mod lse;
pub mod pool;
pub mod proposal;
pub mod symsplit;
pub mod symtoken;
