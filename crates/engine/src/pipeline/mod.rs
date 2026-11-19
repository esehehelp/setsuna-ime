//! The conversion pipeline, one module per stage:
//!
//!   proposal (encode + ONNX forward + log-softmax)
//!   -> decode (greedy / prefix beam, with a CTC-CRF's transitions when the
//!      artifact has them)
//!   -> pool (detokenize, normalize, dedup, fallback)
//!   -> rescore (outside statistics, when the artifact has them)
//!
//! An AR model (`ar`, `ar_native`) takes the proposal's and the beam's place
//! with its encoder and its search; its candidates may be scored by a CTC
//! model too (`ctc_score`, the session's fusion). A reading corrector
//! (`correct`) may rewrite the reading before any of it.
//!
//! with symbol split wrapping the whole pipeline per kana span (or symbol
//! token turning each run of symbols into one token and back), digit split
//! taking the runs of digits out of the reading around that, and chunk
//! wrapping that per chunk of a long reading.

pub mod ar;
pub mod ar_native;
pub mod artifact_load;
pub mod chunk;
pub mod correct;
pub mod crf;
pub mod ctc_score;
pub mod decode;
pub mod digitsplit;
pub mod lse;
pub mod native_encoder;
pub mod pool;
pub mod proposal;
pub mod rescore;
pub mod symsplit;
pub mod symtoken;
