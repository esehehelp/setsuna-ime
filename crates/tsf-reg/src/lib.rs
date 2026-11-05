//! The IME's identity in TSF and its registration (adapted from karukan-tsf,
//! MIT/Apache-2.0), shared by the IME DLL (`crates/tsf`) and the shim
//! (`crates/tsf-shim`), which is registered in the DLL's place under the
//! same CLSID and profile.
//!
//! The category list is karukan's (IMMERSIVESUPPORT / SYSTRAYSUPPORT /
//! SECUREMODE / UIELEMENTENABLED) — omitting any of these caused key events
//! to never reach us in modern apps — and CorvusSKK's COMLESS and
//! INPUTMODECOMPARTMENT, which the Start menu's search and other immersive
//! hosts look for. Registration grants AppContainers read and execute on
//! the DLL so that UWP hosts can load it (install.ps1 grants them the whole
//! install directory and config.toml).

#[cfg(windows)]
mod reg;
#[cfg(windows)]
pub use reg::*;
