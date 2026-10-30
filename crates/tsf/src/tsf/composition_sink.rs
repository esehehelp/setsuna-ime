//! ITfCompositionSink — handles OnCompositionTerminated.
//!
//! Uses `try_borrow_mut` (karukan pattern) so that reentrant termination
//! triggered from inside an edit session is a safe no-op. This is the exact
//! failure mode that crashed the C++ TSF build on Enter (double Release of
//! the `composition_` member).

use windows::core::*;
use windows::Win32::UI::TextServices::*;

use crate::tsf::text_input_processor::SetsunaTextService_Impl;

impl ITfCompositionSink_Impl for SetsunaTextService_Impl {
    /// The app ended the composition (a click elsewhere, its own commit):
    /// what was shown stays in its text. The engine drops what it was
    /// composing, and the candidate window goes with it.
    fn OnCompositionTerminated(
        &self,
        _ecwrite: u32,
        _pcomposition: Option<&ITfComposition>,
    ) -> Result<()> {
        self.guarded("OnCompositionTerminated", None, Ok(()), || {
            {
                let mut inner = match self.inner.try_borrow_mut() {
                    Ok(inner) => inner,
                    Err(_) => {
                        tracing::debug!("OnCompositionTerminated: reentrant borrow, skipping");
                        return Ok(());
                    }
                };
                // its actions (end the composition, hide the window) are
                // done here: the composition is gone already
                let _ = inner.engine.discard();
                inner.composition = None;
                inner.last_preedit.clear();
            }
            self.hide_window();
            Ok(())
        })
    }
}
