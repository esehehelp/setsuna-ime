//! `CandidateUi` adapter wrapping the in-process GDI `CandidateWindow`
//! (`tsf/candidate_window.rs`), drawn as `[ui.candidate_window]` says.

use super::{CandidateUi, ScreenPos};
use crate::tsf::candidate_window::CandidateWindow;

/// In-process GDI candidate popup. Owns a `CandidateWindow` and tracks
/// visibility for `is_visible`.
pub struct GdiCandidateUi {
    inner: CandidateWindow,
    visible: bool,
}

impl GdiCandidateUi {
    pub fn new(cfg: &setsuna_core::config::Config) -> Self {
        Self {
            inner: CandidateWindow::new(
                &cfg.ui.candidate_window,
                crate::tsf::candidate_window::page_size(cfg),
            ),
            visible: false,
        }
    }
}

impl CandidateUi for GdiCandidateUi {
    fn show(&mut self, candidates: &[String], selected: usize, anchor: ScreenPos) {
        self.inner.show(candidates, selected);
        if anchor != ScreenPos::default() {
            self.inner.move_to(anchor.x, anchor.y);
        }
        self.visible = true;
    }

    fn hide(&mut self) {
        self.inner.hide();
        self.inner.set_status("");
        self.visible = false;
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn set_on_select(&mut self, on_select: Box<dyn Fn(usize)>) {
        self.inner.set_on_select(on_select);
    }

    fn set_status(&mut self, status: &str) {
        self.inner.set_status(status);
    }
}
