//! Setsuna on Linux: the Rust half of the fcitx5 addon.
//!
//! fcitx5 input methods are C++ shared-library addons, so the addon is a
//! thin C++ shim (`fcitx5/`) around this crate, linked in as a static
//! library through the C ABI of `ffi` (`include/setsuna.h`). The shim turns
//! fcitx5 key events into `keys::Key`s and renders the `session::View`
//! (commit text, preedit, candidate list) with fcitx5's own panel.
//!
//! The IME logic is `setsuna-core`, shared with the TSF DLL; conversion runs
//! in a `setsuna-engine daemon` child process.

pub mod ffi;
pub mod keys;
pub mod session;
