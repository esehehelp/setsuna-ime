//! ITfKeyEventSink — turns Windows key events into engine actions.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use windows::core::*;
use windows::Win32::Foundation::{BOOL, E_FAIL, FALSE, LPARAM, TRUE, WPARAM};
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::PostMessageW;

use crate::keymap::{classify_vk, Mods, OnOff};
use crate::placement::Rect;
use crate::tsf::edit_session::ActionEditSession;
use crate::tsf::text_input_processor::SetsunaTextService_Impl;
use setsuna_core::engine_bridge::{Action, KeyKind};

#[allow(clippy::not_unsafe_ptr_arg_deref)] // OnPreservedKey takes *const GUID per the TSF interface
impl ITfKeyEventSink_Impl for SetsunaTextService_Impl {
    fn OnSetFocus(&self, _fforeground: BOOL) -> Result<()> {
        setsuna_core::diagnostics::log("OnSetFocus");
        Ok(())
    }

    fn OnTestKeyDown(
        &self,
        pic: Option<&ITfContext>,
        wparam: WPARAM,
        _lparam: LPARAM,
    ) -> Result<BOOL> {
        let vk = wparam.0 as u32;
        setsuna_core::diagnostics::log(format!("OnTestKeyDown vk={vk:#x}"));
        self.guarded("OnTestKeyDown", pic, Ok(FALSE), || {
            let eaten = self.test_key_down(pic, vk);
            if !eaten {
                self.note_passthrough(vk);
            }
            Ok(BOOL::from(eaten))
        })
    }

    fn OnKeyDown(&self, pic: Option<&ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        let vk = wparam.0 as u32;
        setsuna_core::diagnostics::log(format!("OnKeyDown vk={vk:#x} pic={}", pic.is_some()));
        self.guarded("OnKeyDown", pic, Ok(FALSE), || self.key_down(pic, vk))
    }

    fn OnTestKeyUp(
        &self,
        _pic: Option<&ITfContext>,
        _wparam: WPARAM,
        _lparam: LPARAM,
    ) -> Result<BOOL> {
        Ok(FALSE)
    }

    fn OnKeyUp(&self, _pic: Option<&ITfContext>, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Ok(FALSE)
    }

    fn OnPreservedKey(&self, pic: Option<&ITfContext>, rguid: *const GUID) -> Result<BOOL> {
        use crate::globals::{
            GUID_PRESERVED_KEY_OFF, GUID_PRESERVED_KEY_ON, GUID_PRESERVED_KEY_ONOFF,
        };
        if rguid.is_null() {
            return Ok(FALSE);
        }
        let role = match unsafe { *rguid } {
            g if g == GUID_PRESERVED_KEY_ONOFF => OnOff::Toggle,
            g if g == GUID_PRESERVED_KEY_ON => OnOff::On,
            g if g == GUID_PRESERVED_KEY_OFF => OnOff::Off,
            _ => return Ok(FALSE),
        };
        setsuna_core::diagnostics::log(format!("OnPreservedKey {role:?}"));
        self.guarded("OnPreservedKey", pic, Ok(FALSE), || {
            self.switch_by_key(role, pic)
        })
    }
}

/// Whether a key the app receives (the IME not composing) may change the
/// text before the caret or move the caret, making the left context stale.
/// Modifiers alone, Escape, function keys and Ctrl+C leave it.
fn breaks_context(vk: u32, ctrl_or_alt: bool) -> bool {
    const MODIFIERS: &[u32] = &[
        0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
    ];
    if MODIFIERS.contains(&vk) {
        return false;
    }
    if ctrl_or_alt {
        return vk != 0x43; // Ctrl+C copies; other shortcuts may edit
    }
    // BS, Tab, Enter, Space, PgUp PgDn End Home and the arrows, Insert, Delete
    matches!(vk, 0x08 | 0x09 | 0x0D | 0x20 | 0x21..=0x28 | 0x2D | 0x2E)
}

impl SetsunaTextService_Impl {
    /// `OnKeyDown`'s body: what `vk` does, and whether the IME ate it.
    fn key_down(&self, pic: Option<&ITfContext>, vk: u32) -> Result<BOOL> {
        // the on / off keys, Ctrl / Alt combinations among them
        if let Some(role) = self.on_off_role(vk) {
            return self.switch_by_key(role, pic);
        }

        if let Some(form) = self.ctrl_conversion(vk) {
            let outcome = self.inner.borrow_mut().engine.handle_convert_to(form);
            if let Some(context) = pic {
                apply_actions(self, context, outcome.actions)?;
            }
            return Ok(BOOL::from(outcome.consumed));
        }

        if self.ctrl_forget(vk) {
            let outcome = self.inner.borrow_mut().engine.handle_forget_candidate();
            if let Some(context) = pic {
                apply_actions(self, context, outcome.actions)?;
            }
            return Ok(BOOL::from(outcome.consumed));
        }

        if ctrl_down() || alt_down() || win_down() {
            return Ok(FALSE);
        }

        if !self.inner.borrow().enabled {
            return Ok(FALSE);
        }
        // a config saved since applies from this input on
        self.maybe_reload(false);
        let symbols = self.inner.borrow().symbols.clone();

        let shift = shift_down();
        let Some(mut kind) =
            classify_vk(vk, shift, crate::keymap::layout_char(vk, shift), &symbols)
        else {
            return Ok(FALSE);
        };
        // kana input on the JIS layout: Shift+0 types no char, but is を
        if vk == 0x30 && shift && setsuna_core::romaji_table::active().is_some_and(|t| t.kana_keys)
        {
            kind = KeyKind::Char('を');
        }

        if let Some(context) = pic {
            // a field that never takes IME input (a password, a PIN)
            if self.read_app_context(context, &kind) {
                self.note_passthrough(vk);
                return Ok(FALSE);
            }
        }

        let outcome = {
            let mut inner = self.inner.borrow_mut();
            match kind {
                KeyKind::Char(c) if c.is_ascii_uppercase() => {
                    // Shift+letter: enter (or extend) the temporary
                    // alphabet mode. Uppercase-only check is safe because
                    // `classify_vk` only returns uppercase when the Shift
                    // modifier was held at the time of the keypress.
                    inner.engine.handle_shifted_alpha(c)
                }
                KeyKind::Char(c) => inner.engine.handle_char(c),
                KeyKind::Backspace => inner.engine.handle_backspace(),
                KeyKind::Enter => inner.engine.handle_enter(),
                KeyKind::Escape => inner.engine.handle_escape(),
                KeyKind::Space => inner.engine.handle_space(),
                KeyKind::Left => inner.engine.handle_move_cursor(-1),
                KeyKind::Right => inner.engine.handle_move_cursor(1),
                KeyKind::ShiftLeft => inner.engine.handle_resize(-1),
                KeyKind::ShiftRight => inner.engine.handle_resize(1),
                KeyKind::Up => inner.engine.handle_up(),
                KeyKind::Down => inner.engine.handle_down(),
                KeyKind::PageUp => inner.engine.handle_page_up(),
                KeyKind::PageDown => inner.engine.handle_page_down(),
                KeyKind::Delete => inner.engine.handle_delete(),
                KeyKind::Home => inner.engine.handle_home_end(false),
                KeyKind::End => inner.engine.handle_home_end(true),
                KeyKind::Tab => inner.engine.handle_tab(false),
                KeyKind::ShiftTab => inner.engine.handle_tab(true),
                KeyKind::ShiftSpace => inner.engine.handle_shift_space(),
                KeyKind::ConvertTo(form) => inner.engine.handle_convert_to(form),
                KeyKind::Comma => inner.engine.handle_punctuation("、"),
                KeyKind::Period => inner.engine.handle_punctuation("。"),
                KeyKind::Exclaim => inner.engine.handle_punctuation("！"),
                KeyKind::Question => inner.engine.handle_punctuation("？"),
                KeyKind::Punctuation(s) => inner.engine.handle_punctuation(&s),
                KeyKind::ForgetCandidate => inner.engine.handle_forget_candidate(),
            }
        };

        if !outcome.consumed {
            setsuna_core::diagnostics::log("OnKeyDown not consumed");
            self.note_passthrough(vk);
            return Ok(FALSE);
        }

        if let Some(context) = pic {
            apply_actions(self, context, outcome.actions)?;
            setsuna_core::diagnostics::log("OnKeyDown consumed with context");
            return Ok(TRUE);
        }
        setsuna_core::diagnostics::log("OnKeyDown consumed without context");
        Ok(BOOL::from(outcome.consumed))
    }

    /// Whether the IME eats `vk` (the `OnTestKeyDown` answer): what
    /// `key_down` will do with it.
    fn test_key_down(&self, pic: Option<&ITfContext>, vk: u32) -> bool {
        if self.on_off_role(vk).is_some() {
            return true;
        }
        if self.ctrl_conversion(vk).is_some() || self.ctrl_forget(vk) {
            return true;
        }
        // Ctrl/Alt/Win never consumed — preserves Ctrl+C, Alt+Tab, Win shortcuts.
        if ctrl_down() || alt_down() || win_down() {
            return false;
        }
        if !self.inner.borrow().enabled {
            return false;
        }
        self.maybe_reload(false);
        let (starts, client_id) = {
            let inner = self.inner.borrow();
            let Some(kind) = classify_vk(
                vk,
                shift_down(),
                crate::keymap::layout_char(vk, shift_down()),
                &inner.symbols,
            ) else {
                return false;
            };
            if !inner.engine.would_consume(&kind) {
                return false;
            }
            (!inner.engine.is_active(), inner.client_id)
        };
        // a composition would start: not in a password field, as `key_down`
        // decides by the same field
        !(starts && pic.is_some_and(|c| field_at_caret(c, client_id).secret))
    }

    /// What `vk`, held with the modifiers down now, does to the mode, if it
    /// is one of the on / off keys.
    fn on_off_role(&self, vk: u32) -> Option<OnOff> {
        let mods = Mods {
            ctrl: ctrl_down(),
            shift: shift_down(),
            alt: alt_down(),
            win: win_down(),
        };
        self.inner.try_borrow().ok()?.on_off_keys.role(vk, mods)
    }

    /// An on / off key was pressed: the user's choice, kept from then on
    /// in this field.
    fn switch_by_key(&self, role: OnOff, pic: Option<&ITfContext>) -> Result<BOOL> {
        let on = self.inner.borrow().enabled;
        let want = match role {
            OnOff::Toggle => !on,
            OnOff::On => true,
            OnOff::Off => false,
        };
        self.switch_to(want, pic);
        Ok(TRUE)
    }

    /// Ctrl+Delete while converting: forget or hide the candidate selected
    /// (otherwise the key is the app's).
    fn ctrl_forget(&self, vk: u32) -> bool {
        vk == 0x2E // VK_DELETE
            && ctrl_down()
            && !alt_down()
            && !win_down()
            && self.inner.borrow().engine.is_converting()
    }

    /// The F6–F10 form of Ctrl+U / I / O / P / T, while something is composed
    /// (otherwise the key is the app's shortcut).
    fn ctrl_conversion(&self, vk: u32) -> Option<setsuna_core::char_forms::Form> {
        if !ctrl_down() || alt_down() || win_down() {
            return None;
        }
        let inner = self.inner.borrow();
        if !inner.enabled || !inner.engine.is_active() {
            return None;
        }
        crate::keymap::ctrl_form(vk)
    }

    /// A key went to the app: forget the left context when it can change
    /// the text or move the caret. Called from both key paths; forgetting
    /// is idempotent.
    fn note_passthrough(&self, vk: u32) {
        if let Ok(mut inner) = self.inner.try_borrow_mut() {
            if !inner.engine.is_active() && breaks_context(vk, ctrl_down() || alt_down()) {
                inner.engine.forget_context();
            }
        }
    }

    /// `kind` starts a composition: take the left context from the app's
    /// text before the caret (kept as it is when the app does not expose
    /// its text), and the kind of field. True when the field never takes
    /// IME input (a password): the key goes to the app.
    fn read_app_context(&self, context: &ITfContext, kind: &KeyKind) -> bool {
        let client_id = {
            let inner = self.inner.borrow();
            if inner.engine.is_active() || !inner.engine.would_consume(kind) {
                return false;
            }
            inner.client_id
        };
        let (text, field) = text_before_caret(context, client_id);
        setsuna_core::diagnostics::log(match &text {
            Some(t) => format!("app context len={} field={field:?}", t.chars().count()),
            None => format!("app context unavailable field={field:?}"),
        });
        if field.secret {
            return true;
        }
        let mut inner = self.inner.borrow_mut();
        // a field that starts off (a URL, a number) the user turned on: as
        // a private one, its text neither read nor learned
        inner
            .engine
            .set_private_field(field.private || field.starts_off);
        inner.engine.set_app_context(text.as_deref());
        false
    }

    /// Turn the IME on or off, as the user asked (a key, the taskbar item).
    pub(crate) fn toggle_enabled(&self, pic: Option<&ITfContext>) -> Result<BOOL> {
        let on = self.inner.borrow().enabled;
        self.switch_to(!on, pic);
        Ok(TRUE)
    }

    /// Be `on` as the user asked: through the on / off compartment when
    /// there is one (its change comes back through `OnChange`, as one made
    /// by another program does), else directly. The field's start state
    /// no longer applies, in this field when it takes the focus again too.
    fn switch_to(&self, on: bool, pic: Option<&ITfContext>) {
        let (compartment, client_id, current) = {
            let mut inner = self.inner.borrow_mut();
            inner.auto_off = false;
            inner.quiet_mode_change = false;
            inner.switched_in = inner.focused;
            (
                inner.open_close.as_ref().map(|(c, _)| c.clone()),
                inner.client_id,
                inner.enabled,
            )
        };
        if on == current {
            return;
        }
        match compartment {
            Some(c) => {
                if crate::tsf::mode::write_open(&c, client_id, on).is_err() {
                    self.set_mode(on, pic);
                }
            }
            None => self.set_mode(on, pic),
        }
    }

    /// The input scopes of the field that took the focus were read
    /// (`check_field`): one that starts off (a URL, a number, a password)
    /// turns the IME off, quietly, unless the user switched in that field
    /// before (`chosen`); a text field after it turns it on again unless
    /// the user switched since. Nothing changes while composing.
    pub(crate) fn on_field(&self, starts_off: bool, chosen: bool) {
        let (compartment, client_id, on) = {
            let mut inner = self.inner.borrow_mut();
            if inner.engine.is_active() {
                return;
            }
            let want = if starts_off && chosen {
                return;
            } else if starts_off && inner.enabled {
                inner.auto_off = true;
                false
            } else if !starts_off && inner.auto_off {
                inner.auto_off = false;
                true
            } else {
                return;
            };
            if want == inner.enabled {
                return;
            }
            inner.quiet_mode_change = true;
            (
                inner.open_close.as_ref().map(|(c, _)| c.clone()),
                inner.client_id,
                want,
            )
        };
        setsuna_core::diagnostics::log(format!(
            "field starts the IME {}",
            if on { "on" } else { "off" }
        ));
        match compartment {
            Some(c) => {
                if crate::tsf::mode::write_open(&c, client_id, on).is_err() {
                    self.set_mode(on, None);
                }
            }
            None => self.set_mode(on, None),
        }
    }

    /// Be on or off: commit what is composed (pending romaji would vanish
    /// otherwise), forget the left context (text typed while off never
    /// reached it), show the mode on the taskbar and by the caret (not for
    /// the IME's own change at a field's start).
    pub(crate) fn set_mode(&self, on: bool, pic: Option<&ITfContext>) {
        let (actions, changed, quiet, converts, client_id) = {
            let mut inner = self.inner.borrow_mut();
            let quiet = std::mem::take(&mut inner.quiet_mode_change);
            let converts = inner.engine.has_client();
            if inner.enabled == on {
                (Vec::new(), false, quiet, converts, inner.client_id)
            } else {
                let flush = inner.engine.handle_enter();
                inner.enabled = on;
                inner.engine.forget_context();
                (flush.actions, true, quiet, converts, inner.client_id)
            }
        };
        tracing::info!("IME mode: enabled={on}");
        let context = pic
            .cloned()
            .or_else(|| crate::tsf::text_input_processor::focused_context(self));
        if let Some(context) = &context {
            if !actions.is_empty() {
                let _ = apply_actions(self, context, actions);
            }
        }
        let state = self.inner.borrow().mode.as_ref().map(|(s, _)| s.clone());
        if let Some(state) = state {
            state.converts.set(converts);
            state.set(on);
        }
        if changed && !quiet {
            let at = context
                .as_ref()
                .and_then(|c| selection_screen_rect(c, client_id));
            if let Some(popup) = self.inner.borrow_mut().mode_popup.as_mut() {
                popup.show(on, converts, at);
            }
        }
    }
}

fn shift_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x10) < 0 }
}

fn ctrl_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x11) < 0 }
}

fn alt_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x12) < 0 }
}

fn win_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x5B) < 0 || GetKeyState(0x5C) < 0 }
}

pub(crate) fn apply_actions(
    service: &SetsunaTextService_Impl,
    context: &ITfContext,
    actions: Vec<Action>,
) -> Result<()> {
    setsuna_core::diagnostics::log(format!("apply_actions n={}", actions.len()));
    // Split: edit-session-bound actions vs candidate-window-bound actions.
    // The edit session needs synchronous TSF access (TF_ES_SYNC); candidate
    // window operations are plain Win32 and run afterwards.
    //
    // Additionally: drop `UpdatePreedit` actions whose text matches the
    // last one we rendered. Some TSF hosts repaint the composition on
    // every SetText even when the content is identical, which is visible
    // flicker. Commit/End actions are never dropped (they change
    // composition lifetime, not just the rendered text).
    //
    // As the fcitx5 panel draws them: a conversion on screen (the candidate
    // window or an F6–F10 form) is one focused segment, set apart from
    // typing; while typing with the caret moved off the end, the caret is
    // shown where it is (and a move alone is not a no-op).
    let (converting, caret) = {
        let inner = service.inner.borrow();
        (inner.engine.is_converting(), inner.engine.caret())
    };
    let preedit_key = |text: &str| match caret {
        Some(c) => format!("{text}\u{1}{c}"),
        None => text.to_string(),
    };
    let mut edit_actions: Vec<Action> = Vec::with_capacity(actions.len());
    let mut window_actions: Vec<Action> = Vec::new();
    {
        let last = service.inner.borrow().last_preedit.clone();
        let mut running_last = last;
        for a in actions {
            let a = match a {
                Action::UpdatePreedit { text } if converting && !text.is_empty() => {
                    Action::UpdateSegments {
                        segments: vec![text],
                        focus: 0,
                    }
                }
                other => other,
            };
            match a {
                Action::ShowCandidates { .. }
                | Action::ShowSegmentCandidates { .. }
                | Action::HideCandidates
                | Action::ShowSuggestions { .. }
                | Action::Status { .. } => window_actions.push(a),
                Action::UpdatePreedit { ref text } if preedit_key(text) == running_last => {
                    // No-op — skip.
                }
                Action::UpdatePreedit { ref text } => {
                    running_last = preedit_key(text);
                    edit_actions.push(a);
                }
                Action::UpdateSegments {
                    ref segments,
                    focus,
                } => {
                    let key = segments_key(segments, focus);
                    if key != running_last {
                        running_last = key;
                        edit_actions.push(a);
                    }
                }
                _ => edit_actions.push(a),
            }
        }
    }

    let (composition_snapshot, client_id, atom_input, atom_converted, candidate_window) = {
        let inner = service.inner.borrow();
        (
            inner.composition.clone(),
            inner.client_id,
            inner.atom_input,
            inner.atom_converted,
            inner.candidate_window.clone(),
        )
    };
    let composition_cell = Rc::new(RefCell::new(composition_snapshot));
    let composition_sink: ITfCompositionSink = service.cast_interface()?;

    let session_actions_snapshot: Vec<Action> = edit_actions.clone();
    let mut panicked = false;
    if !edit_actions.is_empty() {
        let session = ActionEditSession::new(
            context.clone(),
            edit_actions,
            composition_cell.clone(),
            composition_sink,
            atom_input,
            atom_converted,
            caret,
        );
        let failed = session.panicked.clone();
        let edit_session: ITfEditSession = session.into();
        unsafe {
            let hr = context.RequestEditSession(
                client_id,
                &edit_session,
                TF_ES_SYNC | TF_ES_READWRITE,
            )?;
            if hr.is_err() {
                tracing::warn!("RequestEditSession failed: {:?}", hr);
                setsuna_core::diagnostics::log(format!("RequestEditSession failed: {hr:?}"));
            } else {
                setsuna_core::diagnostics::log(format!("RequestEditSession OK: {hr:?}"));
            }
        }
        panicked = failed.get();
    }

    {
        let mut inner = service.inner.borrow_mut();
        inner.composition = composition_cell.borrow().clone();
        // Track the last-rendered preedit so the next apply_actions call
        // can dedup no-op SetText. Walk session_actions in order so the
        // latest UpdatePreedit / Commit wins.
        for a in &session_actions_snapshot {
            match a {
                Action::UpdatePreedit { text } => inner.last_preedit = preedit_key(text),
                Action::UpdateSegments { segments, focus } => {
                    inner.last_preedit = segments_key(segments, *focus)
                }
                Action::Commit { .. } | Action::EndComposition => inner.last_preedit.clear(),
                _ => {}
            }
        }
    }
    if panicked {
        // the session ended the composition; the engine drops it too
        service.recover(Some(context));
        return Err(E_FAIL.into());
    }

    let (active, typing) = {
        let inner = service.inner.borrow();
        (
            inner.engine.is_active(),
            inner.engine.is_active() && !inner.engine.is_converting(),
        )
    };
    let shows_or_hides = |actions: &[Action]| {
        actions.iter().any(|a| {
            matches!(
                a,
                Action::ShowCandidates { .. }
                    | Action::ShowSegmentCandidates { .. }
                    | Action::ShowSuggestions { .. }
                    | Action::HideCandidates
            )
        })
    };
    let window_up = candidate_window
        .as_ref()
        .is_some_and(|cw| cw.borrow().is_visible());
    // typing without live conversion: its conversions beside the kana, or
    // none (unless the list was opened or closed just now)
    if typing
        && !window_actions
            .iter()
            .any(|a| matches!(a, Action::ShowCandidates { .. } | Action::HideCandidates))
    {
        let list = service.inner.borrow().engine.suggestions();
        if !list.is_empty() {
            window_actions.push(Action::ShowSuggestions { list });
        } else if candidate_window
            .as_ref()
            .is_some_and(|cw| cw.borrow().is_suggesting())
        {
            window_actions.push(Action::HideCandidates);
        }
    } else if !active && window_up && !shows_or_hides(&window_actions) {
        // nothing composed any more (committed, cancelled): no window
        // (the suggestions, a list) stays up after it
        window_actions.push(Action::HideCandidates);
    }

    if let Some(cw) = candidate_window {
        let anchor = |at: i32| current_caret_screen_rect(context, client_id, &composition_cell, at);
        for a in window_actions {
            match a {
                Action::ShowCandidates { list, selected } => {
                    // `None` keeps the previous position: only when the
                    // caret query actually returned a rect
                    let at = anchor(0);
                    cw.borrow_mut().show(&list, selected, at);
                }
                Action::ShowSegmentCandidates { list, selected, at } => {
                    // under the focused segment
                    let at = anchor(at as i32);
                    cw.borrow_mut().show(&list, selected, at);
                }
                Action::HideCandidates => {
                    cw.borrow_mut().hide();
                }
                Action::ShowSuggestions { list } => {
                    let at = anchor(0);
                    cw.borrow_mut().show_suggestions(&list, at);
                }
                Action::Status { text } => {
                    cw.borrow_mut().set_status(&text);
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// The `last_preedit` key of a segmented preedit: its text and focus, so a
/// focus move is not skipped as a no-op.
fn segments_key(segments: &[String], focus: usize) -> String {
    format!("{}\u{0}{focus}", segments.concat())
}

/// A rect `GetTextExt` returned, unless it is empty (the app gave none).
fn screen_rect(r: windows::Win32::Foundation::RECT) -> Option<Rect> {
    (r.left != 0 || r.top != 0 || r.right != 0 || r.bottom != 0).then_some(Rect {
        left: r.left,
        top: r.top,
        right: r.right,
        bottom: r.bottom,
    })
}

/// Query the composition's on-screen rect, from UTF-16 offset `offset` of
/// the preedit on, so the candidate window can place itself by it (below
/// it, or above near the bottom of the screen). Returns `None` if any step
/// fails — the window then keeps its last position.
fn current_caret_screen_rect(
    context: &ITfContext,
    client_id: u32,
    composition: &Rc<RefCell<Option<ITfComposition>>>,
    offset: i32,
) -> Option<Rect> {
    // We need a ReadOnly edit session to call GetTextExt. Define a small
    // one-shot session that captures the result into a shared slot.
    struct ExtSession {
        composition: Rc<RefCell<Option<ITfComposition>>>,
        context: ITfContext,
        offset: i32,
        result: Rc<Cell<Option<Rect>>>,
    }
    #[implement(ITfEditSession)]
    struct Sess(ExtSession);
    impl ITfEditSession_Impl for Sess_Impl {
        fn DoEditSession(&self, ec: u32) -> Result<()> {
            crate::guard::catch("caret rect", Err(E_FAIL.into()), || unsafe {
                let comp = self.0.composition.borrow();
                let Some(composition) = comp.as_ref() else {
                    return Ok(());
                };
                let range = composition.GetRange()?;
                if self.0.offset > 0 {
                    let mut moved = 0i32;
                    range.ShiftStart(ec, self.0.offset, &mut moved, core::ptr::null())?;
                }
                let view = self.0.context.GetActiveView()?;
                let mut rect = windows::Win32::Foundation::RECT::default();
                let mut clipped = BOOL::default();
                view.GetTextExt(ec, &range, &mut rect, &mut clipped)?;
                self.0.result.set(screen_rect(rect));
                Ok(())
            })
        }
    }

    let result: Rc<Cell<Option<Rect>>> = Rc::new(Cell::new(None));
    let session = Sess(ExtSession {
        composition: composition.clone(),
        context: context.clone(),
        offset,
        result: result.clone(),
    });
    let edit_session: ITfEditSession = session.into();
    unsafe {
        let _ = context.RequestEditSession(client_id, &edit_session, TF_ES_SYNC | TF_ES_READ);
    }
    result.get()
}

/// Where the selection (the caret) is on screen, for a notice to sit by it.
fn selection_screen_rect(context: &ITfContext, client_id: u32) -> Option<Rect> {
    struct ReadSelection {
        context: ITfContext,
        result: Rc<Cell<Option<Rect>>>,
    }
    #[implement(ITfEditSession)]
    struct Sess(ReadSelection);
    impl ITfEditSession_Impl for Sess_Impl {
        fn DoEditSession(&self, ec: u32) -> Result<()> {
            crate::guard::catch("selection rect", Err(E_FAIL.into()), || unsafe {
                let Some(range) = selection(&self.0.context, ec)? else {
                    return Ok(());
                };
                let view = self.0.context.GetActiveView()?;
                let mut rect = windows::Win32::Foundation::RECT::default();
                let mut clipped = BOOL::default();
                view.GetTextExt(ec, &range, &mut rect, &mut clipped)?;
                self.0.result.set(screen_rect(rect));
                Ok(())
            })
        }
    }
    let result: Rc<Cell<Option<Rect>>> = Rc::new(Cell::new(None));
    let session: ITfEditSession = Sess(ReadSelection {
        context: context.clone(),
        result: result.clone(),
    })
    .into();
    unsafe {
        let _ = context.RequestEditSession(client_id, &session, TF_ES_SYNC | TF_ES_READ);
    }
    result.get()
}

/// The default selection's range (the caret).
unsafe fn selection(context: &ITfContext, ec: u32) -> Result<Option<ITfRange>> {
    let mut sel = [TF_SELECTION::default()];
    let mut fetched = 0u32;
    context.GetSelection(ec, TF_DEFAULT_SELECTION, &mut sel, &mut fetched)?;
    Ok(core::mem::ManuallyDrop::take(&mut sel[0].range))
}

/// UTF-16 units of the app's text read before the caret: the engine bridge
/// keeps the last 80 chars of it.
const CONTEXT_UNITS: i32 = 160;

/// The input scopes of the field at `range` (`GUID_PROP_INPUTSCOPE`), as
/// Mozc reads them (`win32/tip/tip_range_util.cc`).
unsafe fn input_scopes(context: &ITfContext, ec: u32, range: &ITfRange) -> Vec<InputScope> {
    let Ok(prop) = context.GetAppProperty(&GUID_PROP_INPUTSCOPE) else {
        return Vec::new();
    };
    let Ok(value) = prop.GetValue(ec, range) else {
        return Vec::new();
    };
    let Ok(scope) = IUnknown::try_from(&value).and_then(|u| u.cast::<ITfInputScope>()) else {
        return Vec::new();
    };
    let mut ptr: *mut InputScope = core::ptr::null_mut();
    let mut count = 0u32;
    if scope.GetInputScopes(&mut ptr, &mut count).is_err() || ptr.is_null() {
        return Vec::new();
    }
    let out = core::slice::from_raw_parts(ptr, count as usize).to_vec();
    windows::Win32::System::Com::CoTaskMemFree(Some(ptr as *const core::ffi::c_void));
    out
}

/// What a field's input scopes make it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Field {
    /// Marked private: no context, no rewrite, nothing learned.
    pub private: bool,
    /// Starts with the IME off, as Mozc's `tip_input_mode_manager` does
    /// (a URL, an e-mail address, a number, a phone number, a time,
    /// half-width alphanumerics): the user may turn it on there.
    pub starts_off: bool,
    /// A password or a PIN: the IME never composes there (it starts off
    /// too, and keys go to the app even when the user turns it on).
    pub secret: bool,
}

fn field_of(scopes: &[InputScope]) -> Field {
    let secret = scopes.iter().any(|s| {
        matches!(
            *s,
            IS_PASSWORD | IS_NUMERIC_PASSWORD | IS_NUMERIC_PIN | IS_ALPHANUMERIC_PIN
        )
    });
    let starts_off = secret
        || scopes.iter().any(|s| {
            matches!(
                *s,
                IS_URL
                    | IS_EMAIL_USERNAME
                    | IS_EMAIL_SMTPEMAILADDRESS
                    | IS_NUMBER
                    | IS_DIGITS
                    | IS_TELEPHONE_FULLTELEPHONENUMBER
                    | IS_TIME_FULLTIME
                    | IS_ALPHANUMERIC_HALFWIDTH
            )
        });
    Field {
        private: secret || scopes.contains(&IS_PRIVATE),
        starts_off,
        secret,
    }
}

/// The field at the caret (`field_of` its input scopes): read in a
/// synchronous session, `Field::default()` when it cannot be.
fn field_at_caret(context: &ITfContext, client_id: u32) -> Field {
    let field = Rc::new(Cell::new(None));
    let session: ITfEditSession = ReadField {
        context: context.clone(),
        field: field.clone(),
        post_to: None,
        chosen: false,
    }
    .into();
    let _ = unsafe { context.RequestEditSession(client_id, &session, TF_ES_SYNC | TF_ES_READ) };
    field.get().unwrap_or_default()
}

/// The field that took the focus: read when the app allows (at once or a
/// little later), and posted to the service's notify window, whose
/// `on_field` turns the IME off for a field that starts so.
pub(crate) fn check_field(service: &SetsunaTextService_Impl, context: &ITfContext) {
    let (client_id, hwnd, chosen) = match service.inner.try_borrow_mut() {
        Ok(mut inner) => {
            inner.focused = context.as_raw() as usize;
            (
                inner.client_id,
                inner.notify_hwnd,
                inner.switched_in == inner.focused,
            )
        }
        Err(_) => return,
    };
    if hwnd.0 as usize == 0 {
        return;
    }
    let session: ITfEditSession = ReadField {
        context: context.clone(),
        field: Rc::new(Cell::new(None)),
        post_to: Some(hwnd.0 as isize),
        chosen,
    }
    .into();
    let _ = unsafe {
        context.RequestEditSession(client_id, &session, TF_ES_ASYNCDONTCARE | TF_ES_READ)
    };
}

/// Reads the field at the caret; posts `WM_APP_FIELD` when `post_to` is set
/// (with `chosen`: the user switched the IME in this field before).
#[implement(ITfEditSession)]
struct ReadField {
    context: ITfContext,
    field: Rc<Cell<Option<Field>>>,
    post_to: Option<isize>,
    chosen: bool,
}

impl ITfEditSession_Impl for ReadField_Impl {
    fn DoEditSession(&self, ec: u32) -> Result<()> {
        crate::guard::catch("read field", Err(E_FAIL.into()), || unsafe {
            let Some(range) = selection(&self.context, ec)? else {
                return Ok(());
            };
            let field = field_of(&input_scopes(&self.context, ec, &range));
            self.field.set(Some(field));
            if let Some(hwnd) = self.post_to {
                let _ = PostMessageW(
                    windows::Win32::Foundation::HWND(hwnd as *mut core::ffi::c_void),
                    crate::globals::WM_APP_FIELD,
                    WPARAM(usize::from(field.starts_off) | usize::from(self.chosen) << 1),
                    LPARAM(0),
                );
            }
            Ok(())
        })
    }
}

/// The app's text before the caret, at most `CONTEXT_UNITS` UTF-16 units
/// of it (`None` when the app does not let us read it), and what the field
/// is (`field_of`).
fn text_before_caret(context: &ITfContext, client_id: u32) -> (Option<String>, Field) {
    struct ReadBefore {
        context: ITfContext,
        result: Rc<RefCell<Option<String>>>,
        field: Rc<Cell<Field>>,
    }
    #[implement(ITfEditSession)]
    struct Sess(ReadBefore);
    impl ITfEditSession_Impl for Sess_Impl {
        fn DoEditSession(&self, ec: u32) -> Result<()> {
            crate::guard::catch("text before caret", Err(E_FAIL.into()), || unsafe {
                let Some(range) = selection(&self.0.context, ec)? else {
                    return Ok(());
                };
                self.0
                    .field
                    .set(field_of(&input_scopes(&self.0.context, ec, &range)));
                range.Collapse(ec, TF_ANCHOR_START)?;
                let mut moved = 0i32;
                range.ShiftStart(ec, -CONTEXT_UNITS, &mut moved, core::ptr::null())?;
                let mut buf = [0u16; CONTEXT_UNITS as usize];
                let mut n = 0u32;
                range.GetText(ec, 0, &mut buf, &mut n)?;
                let text = String::from_utf16_lossy(&buf[..(n as usize).min(buf.len())]);
                // a surrogate pair cut at the start
                let text = text.trim_start_matches('\u{FFFD}').to_string();
                *self.0.result.borrow_mut() = Some(text);
                Ok(())
            })
        }
    }

    let result: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    let field = Rc::new(Cell::new(Field::default()));
    let session: ITfEditSession = Sess(ReadBefore {
        context: context.clone(),
        result: result.clone(),
        field: field.clone(),
    })
    .into();
    let hr = unsafe { context.RequestEditSession(client_id, &session, TF_ES_SYNC | TF_ES_READ) };
    if !hr.is_ok_and(|hr| hr.is_ok()) {
        return (None, Field::default());
    }
    (result.take(), field.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_that_edit_or_move_break_the_context() {
        for vk in [0x0D, 0x09, 0x20, 0x08, 0x2E, 0x25, 0x28, 0x24, 0x23, 0x21] {
            assert!(breaks_context(vk, false), "{vk:#x}");
        }
        // Ctrl+V pastes, Ctrl+Z undoes
        assert!(breaks_context(0x56, true));
        assert!(breaks_context(0x5A, true));
    }

    #[test]
    fn modifiers_escape_function_keys_and_copy_do_not() {
        for vk in [0x10, 0x11, 0x12, 0xA0, 0x5B, 0x1B, 0x70, 0x7B] {
            assert!(!breaks_context(vk, false), "{vk:#x}");
        }
        assert!(!breaks_context(0x43, true)); // Ctrl+C
        assert!(!breaks_context(0x11, true)); // Ctrl itself
    }

    #[test]
    fn a_url_field_starts_off_a_password_field_never_composes() {
        let url = field_of(&[IS_URL]);
        assert!(url.starts_off && !url.secret && !url.private);
        let password = field_of(&[IS_PASSWORD]);
        assert!(password.starts_off && password.secret && password.private);
        let pin = field_of(&[IS_DEFAULT, IS_NUMERIC_PIN]);
        assert!(pin.secret);
        let text = field_of(&[IS_DEFAULT]);
        assert_eq!(text, Field::default());
        let private = field_of(&[IS_PRIVATE]);
        assert!(private.private && !private.starts_off);
        assert_eq!(field_of(&[]), Field::default());
    }
}
