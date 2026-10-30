//! `CandidateUi` adapter wrapping the in-process GDI `CandidateWindow`
//! (`tsf/candidate_window.rs`), drawn as `[ui.candidate_window]` says.

use super::{Anchor, CandidateUi};
use crate::tsf::candidate_window::CandidateWindow;

/// In-process GDI candidate popup. Owns a `CandidateWindow` and tracks
/// visibility for `is_visible`.
pub struct GdiCandidateUi {
    inner: CandidateWindow,
    visible: bool,
    suggesting: bool,
}

impl GdiCandidateUi {
    pub fn new(cfg: &setsuna_core::config::Config) -> Self {
        Self {
            inner: CandidateWindow::new(
                &cfg.ui.candidate_window,
                crate::tsf::candidate_window::page_size(cfg),
            ),
            visible: false,
            suggesting: false,
        }
    }
}

impl CandidateUi for GdiCandidateUi {
    fn show(&mut self, candidates: &[String], selected: usize, anchor: Anchor) {
        // placed by its new size before it is shown: no flash where it was
        if let Some(caret) = anchor {
            self.inner.set_anchor(caret);
        }
        self.inner.show(candidates, selected);
        self.visible = true;
        self.suggesting = false;
    }

    fn hide(&mut self) {
        self.inner.hide();
        self.inner.set_status("");
        self.visible = false;
        self.suggesting = false;
    }

    fn show_suggestions(&mut self, list: &[String], anchor: Anchor) {
        if let Some(caret) = anchor {
            self.inner.set_anchor(caret);
        }
        self.inner.show_suggestions(list);
        self.visible = true;
        self.suggesting = true;
    }

    fn is_suggesting(&self) -> bool {
        self.visible && self.suggesting
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
