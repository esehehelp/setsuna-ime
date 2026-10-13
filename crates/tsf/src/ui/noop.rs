//! Headless `CandidateUi` impl for unit tests + cross-platform compile.
//!
//! Records every call into a `Vec<NoopEvent>` so tests can assert
//! show/hide ordering without touching Win32.

use super::{CandidateUi, ScreenPos};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoopEvent {
    Show {
        candidates: Vec<String>,
        selected: usize,
        anchor: ScreenPos,
    },
    Hide,
}

#[derive(Default)]
pub struct NoopUi {
    events: Vec<NoopEvent>,
    visible: bool,
}

impl NoopUi {
    pub fn events(&self) -> &[NoopEvent] {
        &self.events
    }
}

impl CandidateUi for NoopUi {
    fn show(&mut self, candidates: &[String], selected: usize, anchor: ScreenPos) {
        self.events.push(NoopEvent::Show {
            candidates: candidates.to_vec(),
            selected,
            anchor,
        });
        self.visible = true;
    }

    fn hide(&mut self) {
        self.events.push(NoopEvent::Hide);
        self.visible = false;
    }

    fn is_visible(&self) -> bool {
        self.visible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_records_show_hide() {
        let mut ui = NoopUi::default();
        assert!(!ui.is_visible());
        ui.show(&["a".into(), "b".into()], 0, ScreenPos { x: 10, y: 20 });
        assert!(ui.is_visible());
        ui.hide();
        assert!(!ui.is_visible());
        assert_eq!(
            ui.events(),
            &[
                NoopEvent::Show {
                    candidates: vec!["a".into(), "b".into()],
                    selected: 0,
                    anchor: ScreenPos { x: 10, y: 20 },
                },
                NoopEvent::Hide,
            ]
        );
    }
}
