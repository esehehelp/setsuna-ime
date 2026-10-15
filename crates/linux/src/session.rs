//! One IME session: the `EngineBridge` plus what the addon should show.
//!
//! The bridge answers each key with `Action`s meant to be applied in order
//! (TSF edit sessions). fcitx5 instead renders a state: text to commit, a
//! preedit and a candidate list. `Session` folds the actions into a [`View`]
//! the addon reads after every call and renders whole.
//!
//! The addon keeps one session for all input contexts: only the focused one
//! composes, and the composition is committed when it loses focus
//! ([`Session::flush`]), so a context never inherits another's.

use std::collections::BTreeMap;

use setsuna_core::char_forms::annotation;
use setsuna_core::config::Config;
use setsuna_core::engine_bridge::{Action, EngineBridge, KeyKind, KeyOutcome};
use setsuna_core::engine_client::EngineClient;
use setsuna_core::learning::Learning;

use crate::keys::{self, Key};

/// What the addon shows after a call.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct View {
    /// Text to commit, in order, before the preedit is shown. Only the
    /// commits of the last call.
    pub commits: Vec<String>,
    /// The preedit cut into segments: one while typing, several in segment
    /// editing. Empty when there is no composition.
    pub preedit: Vec<String>,
    /// The segment being edited (drawn highlighted), in segment editing.
    pub focus: Option<usize>,
    /// The open candidate list, empty when it is closed.
    pub candidates: Vec<String>,
    /// A note per candidate (`[全] カタカナ`), empty for none.
    pub comments: Vec<String>,
    /// The caret as a char offset into the preedit, when typing with it
    /// moved off the end.
    pub caret: Option<usize>,
    /// The reading being converted (the panel's heading), empty while
    /// typing.
    pub reading: String,
    /// `[frontend] preedit_cursor_at_start`: [`Self::cursor`] is the start
    /// unless the caret was moved.
    pub cursor_at_start: bool,
    /// The F6–F10 form shown, as (its name, its key).
    pub form: Option<(&'static str, &'static str)>,
    /// Conversions of what is typed, for the suggestion window beside the
    /// kana preedit; empty while converting.
    pub suggestions: Vec<String>,
    pub suggestion_comments: Vec<String>,
    /// The highlighted candidate.
    pub selected: usize,
}

impl View {
    /// Byte offset of the caret in the preedit: where it was moved to, else
    /// the start with `cursor_at_start`, else the start of the focused
    /// segment (where its candidate list belongs), else the end.
    pub fn cursor(&self) -> usize {
        if let (Some(caret), [text]) = (self.caret, self.preedit.as_slice()) {
            return text.chars().take(caret).map(char::len_utf8).sum();
        }
        if self.cursor_at_start {
            return 0;
        }
        let end = self.focus.unwrap_or(self.preedit.len());
        self.preedit[..end.min(self.preedit.len())]
            .iter()
            .map(String::len)
            .sum()
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::UpdatePreedit { text } => {
                self.preedit = vec![text];
                self.focus = None;
            }
            Action::UpdateSegments { segments, focus } => {
                self.preedit = segments;
                self.focus = Some(focus);
            }
            Action::Commit { text } => {
                self.commits.push(text);
                self.preedit.clear();
                self.focus = None;
            }
            Action::EndComposition => {
                self.preedit.clear();
                self.focus = None;
            }
            Action::ShowCandidates { list, selected }
            | Action::ShowSegmentCandidates { list, selected, .. } => {
                self.selected = selected.min(list.len().saturating_sub(1));
                self.comments = list
                    .iter()
                    .map(|c| annotation(c).unwrap_or_default().to_string())
                    .collect();
                self.candidates = list;
            }
            Action::HideCandidates => {
                self.candidates.clear();
                self.comments.clear();
                self.selected = 0;
            }
            // only the TSF frontend has rewrite mode (and its status) yet;
            // the suggestions are read from the bridge (`suggestions`)
            Action::Status { .. } | Action::ShowSuggestions { .. } => {}
        }
    }
}

pub struct Session {
    bridge: EngineBridge,
    symbols: BTreeMap<String, String>,
    /// Off: every key but the toggle goes to the app (\` / 半角/全角).
    enabled: bool,
    page_size: usize,
    view: View,
}

/// Candidates per page of the panel (`[ui.candidate_window] page_size`,
/// 1..=10: the digit keys pick them).
fn page_size(cfg: &Config) -> usize {
    (cfg.ui.candidate_window.page_size as usize).clamp(1, 10)
}

/// `[frontend] live_conversion`, on unless set, as on Windows: the text
/// shows the conversion while typing.
fn live_conversion(cfg: &Config) -> bool {
    cfg.frontend.live_conversion.unwrap_or(true)
}

impl Session {
    /// The field typed into is private (a password): no context, no
    /// rewrite, nothing learned (`EngineBridge::set_private_field`).
    pub fn set_private_field(&mut self, private: bool) {
        self.bridge.set_private_field(private);
    }

    /// A session converting through `client` (`None`: raw hiragana only).
    pub fn new(cfg: &Config, client: Option<Box<dyn EngineClient>>) -> Self {
        let mut bridge = EngineBridge::with_frontend(client, &cfg.frontend);
        bridge.set_learning(Learning::open(&cfg.learning));
        bridge.set_hidden(setsuna_core::hidden::Hidden::open());
        bridge.use_user_dictionary_file();
        bridge.open_predictor(&cfg.frontend);
        bridge.set_candidate_paging(page_size(cfg), true);
        bridge.set_live_conversion(live_conversion(cfg));
        Self {
            bridge,
            symbols: cfg.frontend.symbols.clone(),
            enabled: true,
            page_size: page_size(cfg),
            view: View {
                cursor_at_start: cfg.frontend.preedit_cursor_at_start,
                ..View::default()
            },
        }
    }

    /// Take up a (re)loaded config, dropping any composition. Called when
    /// the input method is activated, as the TSF DLL reloads on ActivateEx.
    pub fn configure(&mut self, cfg: &Config) {
        self.bridge.reset_preserving_client(&cfg.frontend);
        self.bridge.set_learning(Learning::open(&cfg.learning));
        self.bridge.set_hidden(setsuna_core::hidden::Hidden::open());
        self.bridge.use_user_dictionary_file();
        self.bridge.open_predictor(&cfg.frontend);
        self.page_size = page_size(cfg);
        self.bridge.set_candidate_paging(self.page_size, true);
        self.bridge.set_live_conversion(live_conversion(cfg));
        self.symbols = cfg.frontend.symbols.clone();
        self.view = View {
            cursor_at_start: cfg.frontend.preedit_cursor_at_start,
            ..View::default()
        };
    }

    /// How long (ms) until a deferred preedit is shown anyway (0: none).
    pub fn deferred_ms(&self) -> u32 {
        self.bridge
            .deferred_for()
            .map_or(0, |d| d.as_millis().min(10_000) as u32 + 1)
    }

    /// The deferred preedit's deadline passed; true when the view changed.
    pub fn tick(&mut self) -> bool {
        let actions = self.bridge.tick();
        let changed = !actions.is_empty();
        self.apply(actions);
        changed
    }

    /// Turn the IME on or off without committing (a new input context).
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    pub fn has_client(&self) -> bool {
        self.bridge.has_client()
    }

    pub fn view(&self) -> &View {
        &self.view
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Handle a key press; `false` leaves it to the app.
    pub fn key(&mut self, key: &Key) -> bool {
        self.view.commits.clear();
        if keys::is_toggle(key) {
            self.toggle();
            return true;
        }
        if !self.enabled {
            return false;
        }
        let outcome = match keys::classify(key, &self.symbols) {
            Some(kind) => self.dispatch(kind),
            None => KeyOutcome {
                consumed: false,
                actions: Vec::new(),
            },
        };
        if !outcome.consumed {
            if !self.bridge.is_active() && keys::breaks_context(key) {
                self.bridge.forget_context();
            }
            return false;
        }
        self.apply(outcome.actions);
        true
    }

    fn dispatch(&mut self, kind: KeyKind) -> KeyOutcome {
        let b = &mut self.bridge;
        match kind {
            // `classify` returns uppercase only with Shift held: the
            // temporary alphabet mode.
            KeyKind::Char(c) if c.is_ascii_uppercase() => b.handle_shifted_alpha(c),
            KeyKind::Char(c) => b.handle_char(c),
            KeyKind::Backspace => b.handle_backspace(),
            KeyKind::Enter => b.handle_enter(),
            KeyKind::Escape => b.handle_escape(),
            KeyKind::Space => b.handle_space(),
            KeyKind::Left => b.handle_move_cursor(-1),
            KeyKind::Right => b.handle_move_cursor(1),
            KeyKind::ShiftLeft => b.handle_resize(-1),
            KeyKind::ShiftRight => b.handle_resize(1),
            KeyKind::Up => b.handle_up(),
            KeyKind::Down => b.handle_down(),
            KeyKind::PageUp => b.handle_page_up(),
            KeyKind::PageDown => b.handle_page_down(),
            KeyKind::Delete => b.handle_delete(),
            KeyKind::Home => b.handle_home_end(false),
            KeyKind::End => b.handle_home_end(true),
            KeyKind::Tab => b.handle_tab(false),
            KeyKind::ShiftTab => b.handle_tab(true),
            KeyKind::ShiftSpace => b.handle_shift_space(),
            KeyKind::ConvertTo(form) => b.handle_convert_to(form),
            KeyKind::Comma => b.handle_punctuation("、"),
            KeyKind::Period => b.handle_punctuation("。"),
            KeyKind::Exclaim => b.handle_punctuation("！"),
            KeyKind::Question => b.handle_punctuation("？"),
            KeyKind::Punctuation(s) => b.handle_punctuation(&s),
            KeyKind::ForgetCandidate => b.handle_forget_candidate(),
        }
    }

    /// Take the engine's latest result (after its notify); `true` when the
    /// view changed.
    pub fn apply_async_result(&mut self) -> bool {
        self.view.commits.clear();
        let actions = self.bridge.apply_async_result();
        let changed = !actions.is_empty();
        self.apply(actions);
        changed
    }

    /// Pick candidate `index` of the open list (a click in the panel).
    pub fn select(&mut self, index: usize) -> bool {
        self.view.commits.clear();
        let outcome = self.bridge.handle_select(index);
        self.apply(outcome.actions);
        outcome.consumed
    }

    /// Commit the composition as it stands (the input context loses focus
    /// or switches input method), then forget the left context: the next
    /// text is typed elsewhere.
    pub fn flush(&mut self) {
        self.view.commits.clear();
        let outcome = self.bridge.handle_enter();
        self.apply(outcome.actions);
        self.apply(vec![Action::HideCandidates, Action::EndComposition]);
        self.bridge.forget_context();
    }

    /// Turn the IME on or off, committing what is composed. Text typed while
    /// it was off never reached the left context, which `flush` forgets.
    pub fn toggle(&mut self) {
        self.flush();
        self.enabled = !self.enabled;
        setsuna_core::diagnostics::log(format!("fcitx5 toggle: enabled={}", self.enabled));
    }

    /// Drop the composition (the app reset the input context).
    pub fn reset(&mut self) {
        self.view.commits.clear();
        let actions = self.bridge.discard();
        self.apply(actions);
    }

    fn apply(&mut self, actions: Vec<Action>) {
        for action in actions {
            self.view.apply(action);
        }
        self.view.caret = self.bridge.caret();
        self.view.reading = self.bridge.shown_reading().unwrap_or_default();
        self.view.form = self.bridge.shown_form().map(|f| f.label());
        let suggestions = self.bridge.suggestions();
        // as many as a page of the conversion holds, so the panel keeps its
        // height between typing and converting
        self.view.suggestions = suggestions.iter().take(self.page_size).cloned().collect();
        self.view.suggestion_comments = self
            .view
            .suggestions
            .iter()
            .map(|c| annotation(c).unwrap_or_default().to_string())
            .collect();
        // a whole conversion (candidate window or F6–F10 form) is drawn as
        // one focused segment, as Mozc does, so it reads apart from typing
        if self.bridge.is_converting() && self.view.focus.is_none() && self.view.preedit.len() == 1
        {
            self.view.focus = Some(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use setsuna_core::engine_client::EngineResult;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::keys::{MOD_ALT, MOD_CTRL, MOD_SHIFT};

    /// Answers every dispatch with `candidates[reading]` on the next
    /// `take_latest`.
    #[derive(Clone, Default)]
    struct Echo {
        answers: Rc<RefCell<BTreeMap<String, Vec<String>>>>,
        last: Rc<RefCell<Option<(u64, String, String)>>>,
    }

    impl Echo {
        fn answer(&self, reading: &str, candidates: &[&str]) {
            self.answers.borrow_mut().insert(
                reading.into(),
                candidates.iter().map(|s| s.to_string()).collect(),
            );
        }
        fn context(&self) -> String {
            self.last.borrow().as_ref().map(|l| l.1.clone()).unwrap()
        }
    }

    impl EngineClient for Echo {
        fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
            *self.last.borrow_mut() = Some((req_id, context.into(), reading.into()));
        }
        fn take_latest(&self) -> Option<EngineResult> {
            let (req_id, _, reading) = self.last.borrow_mut().clone()?;
            let candidates = self.answers.borrow().get(&reading)?.clone();
            Some(EngineResult {
                req_id,
                reading,
                candidates,
                reading_pos: Vec::new(),
                ..Default::default()
            })
        }
        fn is_ready(&self) -> bool {
            true
        }
    }

    fn session() -> (Session, Echo) {
        session_with(|_| {})
    }

    /// The caret where it is (`preedit_cursor_at_start` off) unless `f`
    /// says otherwise.
    fn session_with(f: impl FnOnce(&mut Config)) -> (Session, Echo) {
        let mut cfg = Config::default();
        cfg.learning.enabled = false;
        cfg.frontend.preedit_cursor_at_start = false;
        f(&mut cfg);
        let echo = Echo::default();
        (Session::new(&cfg, Some(Box::new(echo.clone()))), echo)
    }

    fn press(s: &mut Session, sym: u32, mods: u32) -> bool {
        s.key(&Key { sym, code: 0, mods })
    }

    fn type_str(s: &mut Session, text: &str) {
        for c in text.chars() {
            let mods = if c.is_ascii_uppercase() { MOD_SHIFT } else { 0 };
            assert!(press(s, c as u32, mods), "{c}");
        }
    }

    const SPACE: u32 = 0x20;
    const RETURN: u32 = 0xff0d;
    const RIGHT: u32 = 0xff53;
    const DOWN: u32 = 0xff54;
    const LEFT: u32 = 0xff51;

    #[test]
    fn live_conversion_shows_kana_then_the_conversion() {
        let (mut s, echo) = session();
        type_str(&mut s, "kyouha");
        assert_eq!(s.view().preedit, vec!["きょうは"]);
        assert_eq!(s.view().cursor(), "きょうは".len());
        echo.answer("きょうは", &["今日は"]);
        assert!(s.apply_async_result());
        assert_eq!(s.view().preedit, vec!["今日は"]);
        assert!(s.view().suggestions.is_empty());
        assert!(s.view().commits.is_empty());
    }

    #[test]
    fn space_opens_the_list_and_enter_commits_the_pick() {
        let (mut s, echo) = session();
        type_str(&mut s, "kisha");
        echo.answer("きしゃ", &["記者", "汽車"]);
        s.apply_async_result();
        assert!(press(&mut s, SPACE, 0));
        s.apply_async_result();
        assert_eq!(s.view().candidates[..2], ["記者", "汽車"]);
        assert!(press(&mut s, DOWN, 0));
        assert_eq!(s.view().selected, 1);
        assert_eq!(s.view().preedit, vec!["汽車"]);
        assert!(press(&mut s, RETURN, 0));
        assert_eq!(s.view().commits, vec!["汽車"]);
        assert!(s.view().preedit.is_empty());
        assert!(s.view().candidates.is_empty());
        // the commit is the next conversion's left context
        type_str(&mut s, "desuyo");
        assert_eq!(echo.context(), "汽車");
        // commits are reported once
        press(&mut s, 'k' as u32, 0);
        assert!(s.view().commits.is_empty());
    }

    #[test]
    fn backquote_commits_and_turns_the_ime_off_and_on() {
        let (mut s, _) = session();
        type_str(&mut s, "kana");
        assert!(press(&mut s, '`' as u32, 0));
        assert!(!s.enabled());
        assert_eq!(s.view().commits, vec!["かな"]);
        assert!(s.view().preedit.is_empty());
        // off: keys go to the app
        assert!(!press(&mut s, 'a' as u32, 0));
        assert!(press(&mut s, '`' as u32, MOD_ALT));
        assert!(s.enabled());
        type_str(&mut s, "a");
        assert_eq!(s.view().preedit, vec!["あ"]);
    }

    #[test]
    fn a_conversion_is_highlighted_and_annotated() {
        let (mut s, echo) = session();
        type_str(&mut s, "kana");
        assert_eq!(s.view().focus, None);
        echo.answer("かな", &["仮名", "カナ"]);
        // a 2-char reading is converted on Space
        press(&mut s, SPACE, 0);
        s.apply_async_result();
        assert_eq!(s.view().focus, Some(0));
        assert_eq!(s.view().cursor(), 0);
        let i = s
            .view()
            .candidates
            .iter()
            .position(|c| c == "カナ")
            .unwrap();
        assert_eq!(s.view().comments[i], "[全] カタカナ");
        assert_eq!(s.view().comments[0], "");
        // a digit picks from the page
        assert!(press(&mut s, '1' as u32, 0));
        assert_eq!(s.view().commits, vec!["仮名"]);
    }

    #[test]
    fn f7_shows_katakana_and_home_moves_the_caret() {
        let (mut s, _) = session();
        type_str(&mut s, "kana");
        assert!(press(&mut s, 0xffc4, 0));
        assert_eq!(s.view().preedit, vec!["カナ"]);
        assert_eq!(s.view().focus, Some(0));
        assert_eq!(s.view().form, Some(("全角カタカナ", "F7")));
        assert_eq!(s.view().reading, "かな");
        press(&mut s, 0xff1b, 0); // Escape
        press(&mut s, 0xff50, 0); // Home
        press(&mut s, RIGHT, 0);
        assert_eq!(s.view().caret, Some(1));
        assert_eq!(s.view().cursor(), "か".len());
        // nothing composed: Ctrl+I is the app's
        let (mut s, _) = session();
        assert!(!press(&mut s, 'i' as u32, MOD_CTRL));
    }

    #[test]
    fn typing_keeps_the_kana_and_suggests_the_conversions() {
        let (mut s, echo) = session_with(|c| c.frontend.live_conversion = Some(false));
        type_str(&mut s, "kyouha");
        echo.answer("きょうは", &["今日は", "京は"]);
        assert!(s.apply_async_result());
        assert_eq!(s.view().preedit, vec!["きょうは"]);
        assert_eq!(s.view().suggestions[..2], ["今日は", "京は"]);
        assert!(s.view().candidates.is_empty());
        // Space: the usual window, no suggestions
        press(&mut s, SPACE, 0);
        assert_eq!(s.view().preedit, vec!["今日は"]);
        assert!(s.view().suggestions.is_empty());
        assert!(!s.view().candidates.is_empty());
    }

    #[test]
    fn the_caret_stays_at_the_start_when_asked() {
        let (mut s, echo) = session_with(|c| c.frontend.preedit_cursor_at_start = true);
        type_str(&mut s, "kyouha");
        assert_eq!(s.view().cursor(), 0);
        echo.answer("きょうは", &["今日は", "京は"]);
        press(&mut s, SPACE, 0);
        assert_eq!(s.view().cursor(), 0);
        // segment editing: still the start
        press(&mut s, RIGHT, 0);
        press(&mut s, RIGHT, 0);
        assert_eq!(s.view().cursor(), 0);
        // a caret moved by hand is shown
        let (mut s, _) = session_with(|c| c.frontend.preedit_cursor_at_start = true);
        type_str(&mut s, "kana");
        press(&mut s, LEFT, 0);
        assert_eq!(s.view().cursor(), "か".len());
    }

    #[test]
    fn a_click_picks_a_candidate() {
        let (mut s, echo) = session();
        type_str(&mut s, "kisha");
        echo.answer("きしゃ", &["記者", "汽車"]);
        s.apply_async_result();
        press(&mut s, SPACE, 0);
        assert!(s.select(1));
        assert_eq!(s.view().preedit, vec!["汽車"]);
        assert_eq!(s.view().selected, 1);
    }

    #[test]
    fn segment_editing_marks_the_focused_segment() {
        let (mut s, echo) = session();
        type_str(&mut s, "kishanokisha");
        echo.answer("きしゃのきしゃ", &["記者の記者", "汽車の記者"]);
        s.apply_async_result();
        press(&mut s, SPACE, 0);
        assert!(press(&mut s, RIGHT, 0));
        let v = s.view();
        assert!(v.preedit.len() > 1, "{:?}", v.preedit);
        assert_eq!(v.focus, Some(0));
        assert_eq!(v.cursor(), 0);
        assert!(!v.candidates.is_empty());
        assert!(press(&mut s, RIGHT, 0));
        assert_eq!(s.view().focus, Some(1));
        assert_eq!(s.view().cursor(), s.view().preedit[0].len());
        assert!(press(&mut s, LEFT, 0));
        assert_eq!(s.view().focus, Some(0));
    }

    #[test]
    fn flush_commits_what_is_shown() {
        let (mut s, echo) = session();
        type_str(&mut s, "kyouha");
        echo.answer("きょうは", &["今日は"]);
        s.apply_async_result();
        s.flush();
        assert_eq!(s.view().commits, vec!["今日は"]);
        assert!(s.view().preedit.is_empty());
        // focus moved: the next context starts empty
        type_str(&mut s, "desuyo");
        assert_eq!(echo.context(), "");
    }

    #[test]
    fn reset_drops_the_composition() {
        let (mut s, _echo) = session();
        type_str(&mut s, "kisha");
        press(&mut s, SPACE, 0);
        s.reset();
        assert_eq!(s.view(), &View::default());
        assert!(!press(&mut s, RETURN, 0));
    }

    #[test]
    fn shift_letters_type_ascii_and_shortcuts_pass() {
        let (mut s, _echo) = session();
        type_str(&mut s, "Linux");
        assert_eq!(s.view().preedit, vec!["Linux"]);
        assert!(press(&mut s, RETURN, 0));
        assert_eq!(s.view().commits, vec!["Linux"]);
        assert!(!press(&mut s, 'c' as u32, MOD_CTRL));
    }

    #[test]
    fn keys_the_app_gets_forget_the_context() {
        let (mut s, echo) = session();
        type_str(&mut s, "a");
        press(&mut s, RETURN, 0);
        assert!(!press(&mut s, RETURN, 0)); // a newline in the app
        type_str(&mut s, "desuyo");
        assert_eq!(echo.context(), "");
    }
}
