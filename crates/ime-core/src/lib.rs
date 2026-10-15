//! Setsuna core — the IME logic every platform frontend shares.
//!
//! Romaji composition, the conversion state machine (`engine_bridge`), the
//! candidate post-processing and learning, the config and the clients of the
//! `setsuna-engine daemon`, `setsuna-rerank daemon` and `setsuna-rewrite
//! daemon` child processes. A frontend (the Windows TSF DLL in
//! `crates/tsf`, the fcitx5 addon in `crates/linux`) turns its key events
//! into `engine_bridge::KeyKind`s and renders the `Action`s that come back.

pub mod char_forms;
pub mod child_daemon;
pub mod composing_text;
pub mod config;
pub mod date_time;
pub mod diagnostics;
pub mod engine_bridge;
pub mod engine_client;
pub mod kana_candidates;
pub mod learning;
pub mod mixed;
pub mod postprocess;
pub mod privacy;
mod reading_align;
pub mod rerank_client;
pub mod rewrite_client;
mod segmented;
pub mod segments;
pub mod symbol_dict;
