//! Candidate-window abstraction for the TSF IME.
//!
//! Why a trait
//! -----------
//! The implementation in `tsf/candidate_window.rs` is a hand-rolled GDI
//! popup. It sits behind `CandidateUi` so a future replacement
//! (Direct2D / WinUI / Tauri-side renderer) can land without touching the
//! TIP, key-event-sink, or engine-bridge code paths. Today only one impl
//! ships (`GdiCandidateUi`), plus a headless `NoopUi` for tests.
//!
//! Call sites that need an instance go through `default_ui(&Config)` so
//! a swap is a one-line change.

#[cfg(target_os = "windows")]
pub mod gdi_window;

#[cfg(test)]
pub mod noop;

/// What the popup is placed by: the caret's (the composition's) rect on
/// screen. The window goes below it, or above it near the bottom of the
/// screen (`crate::placement`); `None` keeps the previous position.
pub type Anchor = Option<crate::placement::Rect>;

/// Minimal candidate-window contract. The trait is intentionally narrow
/// so the call sites in `key_event_sink::apply_actions` (`Show...` /
/// `Hide...` / focus query) stay obvious. Anything richer (themeing,
/// orientation, sub-region selection) is left to the impl.
pub trait CandidateUi {
    /// Show the candidate list. `selected` is the 0-based index of the
    /// highlighted item; the impl is responsible for capping it to the
    /// list length. `anchor` repositions the window — pass `None` to keep
    /// the current position.
    fn show(&mut self, candidates: &[String], selected: usize, anchor: Anchor);

    /// Hide the window without destroying state. A subsequent `show` is
    /// expected to be fast (no resource re-init).
    fn hide(&mut self);

    /// True when the window is currently visible. Used in tests; some
    /// impls may always return `false` if visibility is opaque to them.
    fn is_visible(&self) -> bool;

    /// Called with a candidate's index in the list when the user clicks it.
    fn set_on_select(&mut self, _on_select: Box<dyn Fn(usize)>) {}

    /// A line about the list beside its position (empty: none). `hide`
    /// clears it.
    fn set_status(&mut self, _status: &str) {}

    /// Show suggestions while typing (live conversion off: the conversions
    /// of the kana shown): no candidate highlighted.
    fn show_suggestions(&mut self, _list: &[String], _anchor: Anchor) {}

    /// Whether what is shown is suggestions (`show_suggestions`).
    fn is_suggesting(&self) -> bool {
        false
    }
}

/// Construct the default impl for the running platform. Today this is
/// the GDI popup on Windows; on non-Windows it falls back to `NoopUi`
/// so cross-platform compile checks pass.
#[cfg(target_os = "windows")]
pub fn default_ui(config: &setsuna_core::config::Config) -> Box<dyn CandidateUi> {
    Box::new(gdi_window::GdiCandidateUi::new(config))
}

#[cfg(all(not(target_os = "windows"), test))]
pub fn default_ui(_config: &setsuna_core::config::Config) -> Box<dyn CandidateUi> {
    Box::new(noop::NoopUi::default())
}
