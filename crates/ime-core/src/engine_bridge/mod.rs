//! Phase state machine with **async** live conversion.
//!
//! Threading model:
//!   * UI thread: advances `ComposingText` and fires an immediate preedit
//!     update with the RAW hiragana (no blocking inference).
//!   * Engine client (`EngineClient`, in production the child
//!     `setsuna-engine daemon` of `child_daemon`): converts each dispatch,
//!     then calls the frontend's `Notify`, which wakes the UI thread (TSF:
//!     a message posted to the service's notify HWND; fcitx5: an event
//!     dispatcher job). The UI thread calls
//!     [`EngineBridge::apply_async_result`] and applies the actions, which
//!     swap the preedit to the kanji top-1.
//!
//! Phases:
//!   * `Empty` — no composition.
//!   * `Composing` — typing; candidates is whatever the last async run
//!     delivered (may be empty while the first job is in flight). Preedit
//!     shows `candidates[0]` if present, else raw hiragana + pending romaji.
//!   * `Converting` — Space was pressed; the candidate window is visible.
//!     ← / → / Shift+← / Shift+→ switch it to segment editing (`segmented`):
//!     the conversion is cut into segments by the engine's alignment, ← / →
//!     move the focused one, Shift+← / Shift+→ resize it, Space / ↑ / ↓ pick
//!     its candidate, Enter commits them all.
//!
//! Rewrite mode (with a rewriter, as the frontend sets it): the conversion key also
//! asks the rewriter (`rewrite_client`) to turn the input into natural
//! Japanese; the rewrite comes first in the open list once it is in.

mod async_jobs;
mod handlers;
mod live_cache;
mod ranking;
mod segment_edit;
mod state;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::char_forms::Form;
use crate::composing_text::ComposingText;
use crate::engine_client::EngineClient;
use crate::learning::Learning;
use crate::rerank_client::RerankClient;
use crate::rewrite_client::RewriteClient;

pub use ranking::order_candidates;

use state::{
    BridgeConfig, CompositionState, ContextState, FormShown, Phase, UserDictSource, CONTEXT_WINDOW,
    PREDICTIONS,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    UpdatePreedit {
        text: String,
    },
    Commit {
        text: String,
    },
    EndComposition,
    ShowCandidates {
        list: Vec<String>,
        selected: usize,
    },
    HideCandidates,
    /// The preedit as segments, the focused one marked (segment editing).
    UpdateSegments {
        segments: Vec<String>,
        focus: usize,
    },
    /// The focused segment's candidates, the window placed at its UTF-16
    /// offset `at` in the preedit.
    ShowSegmentCandidates {
        list: Vec<String>,
        selected: usize,
        at: usize,
    },
    /// A line about the open list (a rewrite in progress or why it
    /// failed); empty clears it. Hiding the candidates clears it too.
    Status {
        text: String,
    },
    /// Predictions while typing, for a window beside the preedit with none
    /// selected (Tab opens them as candidates). Frontends add it
    /// (`EngineBridge::suggestions`).
    ShowSuggestions {
        list: Vec<String>,
    },
}

impl Action {
    /// The action for a log: its kind and the length of its text or list,
    /// never the text.
    pub fn shape(&self) -> String {
        let n = |s: &str| s.chars().count();
        match self {
            Action::UpdatePreedit { text } => format!("UpdatePreedit({})", n(text)),
            Action::Commit { text } => format!("Commit({})", n(text)),
            Action::EndComposition => "EndComposition".into(),
            Action::ShowCandidates { list, selected } => {
                format!("ShowCandidates({}, selected {selected})", list.len())
            }
            Action::HideCandidates => "HideCandidates".into(),
            Action::UpdateSegments { segments, focus } => {
                format!("UpdateSegments({}, focus {focus})", segments.len())
            }
            Action::ShowSegmentCandidates { list, selected, at } => format!(
                "ShowSegmentCandidates({}, selected {selected}, at {at})",
                list.len()
            ),
            Action::Status { text } => format!("Status({})", n(text)),
            Action::ShowSuggestions { list } => format!("ShowSuggestions({})", list.len()),
        }
    }

    /// [`Action::shape`] of each, for a log.
    pub fn shapes(actions: &[Action]) -> String {
        let shapes: Vec<String> = actions.iter().map(Action::shape).collect();
        format!("[{}]", shapes.join(", "))
    }
}

pub struct KeyOutcome {
    pub consumed: bool,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyKind {
    Char(char),
    Punctuation(String),
    Backspace,
    Enter,
    Escape,
    Space,
    Left,
    Right,
    /// Shift+← / Shift+→: resize the focused segment.
    ShiftLeft,
    ShiftRight,
    Up,
    Down,
    PageUp,
    PageDown,
    /// Delete / Home / End: edit or move the caret in the composition;
    /// Home / End pick the first / last candidate while converting.
    Delete,
    Home,
    End,
    /// Tab / Shift+Tab: open the candidates, then step through them.
    Tab,
    ShiftTab,
    /// Shift+Space: Space, but a full-width space when nothing is composed
    /// (the other width than `full_width_space` gives).
    ShiftSpace,
    /// Ctrl+Delete while converting: forget or hide the candidate selected.
    ForgetCandidate,
    /// F6–F10 (Ctrl+U / I / O / P / T): the composition as hiragana,
    /// katakana, half-width katakana, full / half-width alphanumerics.
    ConvertTo(Form),
    Comma,
    Period,
    Exclaim,
    Question,
}

pub struct EngineBridge {
    composing: ComposingText,
    phase: Phase,
    comp: CompositionState,
    ctx: ContextState,
    cfg: BridgeConfig,
    req_counter: AtomicU64,
    /// Results older than this belong to a previous composition.
    result_floor: u64,
    /// Conversion backend. `None` in hosts where the engine daemon failed
    /// to spawn — live conversion then degrades to a raw-hiragana preedit.
    client: Option<Box<dyn EngineClient>>,
    /// The reranker; `None` when reranking is off.
    reranker: Option<Box<dyn RerankClient>>,
    /// The rewriter; `None` when rewriting is off.
    rewriter: Option<Box<dyn RewriteClient>>,
    /// The character n-gram table and its bigram and trigram weights
    /// (`[ngram]`); `None` when off.
    ngram: Option<(Arc<crate::char_ngram::CharNgram>, f32, f32)>,
    /// The reading lexicon (`[lexicon]`): off without a file.
    lexicon: Option<Arc<crate::lexicon::Lexicon>>,
    /// `None`: `[frontend] symbol_dictionary = false`.
    symbol_dict: Option<Arc<crate::symbol_dict::SymbolDict>>,
    /// The user dictionary.
    user_dict: UserDictSource,
    /// Learned candidate choices; `None` when learning is off.
    learning: Option<Learning>,
    /// Candidates hidden with Ctrl+Delete.
    hidden: Option<crate::hidden::Hidden>,
    /// Prediction from the commits (`predict`).
    predictor: Option<std::cell::RefCell<crate::predict::Predictor>>,
}

impl EngineBridge {
    pub fn new(client: Option<Box<dyn EngineClient>>) -> Self {
        Self::with_frontend(client, &crate::config::FrontendCfg::default())
    }

    pub fn with_frontend(
        client: Option<Box<dyn EngineClient>>,
        frontend: &crate::config::FrontendCfg,
    ) -> Self {
        let mut bridge = Self {
            composing: ComposingText::new(),
            phase: Phase::Empty,
            comp: CompositionState::default(),
            ctx: ContextState::default(),
            cfg: BridgeConfig::from_frontend(frontend),
            req_counter: AtomicU64::new(0),
            result_floor: 1,
            client,
            reranker: None,
            rewriter: None,
            ngram: None,
            lexicon: None,
            symbol_dict: frontend
                .symbol_dictionary
                .then(|| crate::symbol_dict::shared(frontend.symbol_dictionary_file.as_deref())),
            user_dict: UserDictSource::None,
            learning: None,
            hidden: None,
            predictor: None,
        };
        if let Err(e) = crate::romaji_table::configure(&frontend.romaji_table) {
            crate::diagnostics::log(format!("romaji table: {e}"));
        }
        // kana input: letters are kana keys, none of them English
        let kana_keys = crate::romaji_table::active().is_some_and(|t| t.kana_keys);
        let mixed = frontend
            .mixed_model
            .as_deref()
            .filter(|dir| !dir.is_empty() && !kana_keys)
            .and_then(|dir| crate::mixed::shared(dir, frontend.name_dictionary_file.as_deref()));
        bridge.set_mixed(mixed, &frontend.symbols, crate::mixed::Casing::of(frontend));
        // つずく read as つづく, by the mixed model's kana model
        bridge.composing.set_voicing(frontend.typo_correction);
        bridge
    }

    /// Mixed Japanese / English input with `model` (`None`: off);
    /// `symbols` is `[frontend.symbols]`, `casing` what the English is
    /// cased.
    fn set_mixed(
        &mut self,
        model: Option<Arc<crate::mixed::MixedModel>>,
        symbols: &std::collections::BTreeMap<String, String>,
        casing: crate::mixed::Casing,
    ) {
        self.cfg.mixed_symbols = if model.is_some() {
            crate::config::SYMBOL_CHARS
                .iter()
                .filter_map(|(name, c)| symbols.get(*name).map(|s| (s.clone(), *c)))
                .collect()
        } else {
            HashMap::new()
        };
        self.composing.set_mixed(model, symbols, casing);
        self.composing.set_before(&self.ctx.context);
    }

    fn next_req_id(&self) -> u64 {
        self.req_counter.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Rerank the conversions with `reranker` (an LM), as `cfg` says.
    pub fn set_reranker(
        &mut self,
        reranker: Option<Box<dyn RerankClient>>,
        cfg: &crate::config::RerankCfg,
    ) {
        self.reranker = reranker;
        self.cfg.rerank_cfg = cfg.clone();
        self.comp.rerank = None;
    }

    /// A reranker whose backend is still up.
    pub fn has_reranker(&self) -> bool {
        self.reranker.as_ref().is_some_and(|r| r.is_alive())
    }

    /// Rewrite with `rewriter` in rewrite mode (`None`: off, and the mode
    /// with it).
    pub fn set_rewriter(&mut self, rewriter: Option<Box<dyn RewriteClient>>) {
        self.stop_rewrite();
        self.rewriter = rewriter;
        self.cfg.rewrite_mode &= self.rewriter.is_some();
    }

    /// A rewriter whose backend is still up.
    pub fn has_rewriter(&self) -> bool {
        self.rewriter.as_ref().is_some_and(|r| r.is_alive())
    }

    pub fn rewrite_mode(&self) -> bool {
        self.cfg.rewrite_mode
    }

    /// Turn rewrite mode on or off (on only with a rewriter). Turned on
    /// with a conversion open, it asks for that conversion's rewrite. A
    /// rewrite already asked for still lands.
    pub fn set_rewrite_mode(&mut self, on: bool) -> Vec<Action> {
        self.cfg.rewrite_mode = on && self.rewriter.is_some();
        if !self.cfg.rewrite_mode || !self.window_open() {
            return Vec::new();
        }
        let Phase::Converting {
            candidates,
            seg: None,
            ..
        } = &self.phase
        else {
            return Vec::new();
        };
        let Some(source) = candidates.first().cloned() else {
            return Vec::new();
        };
        let reading = self.composing.hiragana().to_string();
        self.start_rewrite(&reading, &source).into_iter().collect()
    }

    /// The frontend's candidate window: `page_size` candidates per page,
    /// labelled with digits that pick them when `digit_select`.
    pub fn set_candidate_paging(&mut self, page_size: usize, digit_select: bool) {
        self.cfg.page_size = page_size.max(1);
        self.cfg.digit_select = digit_select;
    }

    /// Whether typing shows the conversion in the preedit; `false` keeps
    /// the kana there for a frontend that lists [`Self::suggestions`].
    pub fn set_live_conversion(&mut self, live: bool) {
        self.cfg.live_conversion = live;
    }

    /// The conversions of what is typed, for a suggestion window beside a
    /// kana preedit: the typing phase's candidates when they are for the
    /// whole reading. Empty otherwise, and with live conversion on: no
    /// window comes up while typing, the predictions open with Tab only
    /// (user 2026-10-23).
    pub fn suggestions(&self) -> Vec<String> {
        match &self.phase {
            Phase::Composing { candidates }
                if !self.cfg.live_conversion
                    && !candidates.is_empty()
                    && self.composing.cursor_at_end()
                    && self.comp.list_reading == self.composing.live_reading() =>
            {
                candidates.clone()
            }
            _ => Vec::new(),
        }
    }

    /// Whether a conversion is on screen (the candidate window or an F6–F10
    /// form), as opposed to typing.
    pub fn is_converting(&self) -> bool {
        matches!(self.phase, Phase::Converting { .. })
    }

    /// The pending romaji the next letter typed is read after, while typing
    /// (`ComposingText::romaji_context`); None while converting.
    pub fn romaji_context(&self) -> Option<&str> {
        if self.is_converting() {
            return None;
        }
        self.composing.romaji_context()
    }

    /// The caret as a char offset into the preedit while typing with the
    /// caret moved off the end (the preedit is then plain kana); `None` for
    /// the end.
    pub fn caret(&self) -> Option<usize> {
        match self.phase {
            Phase::Composing { .. } if !self.composing.cursor_at_end() => {
                Some(self.composing.caret())
            }
            _ => None,
        }
    }

    /// The reading being converted: the focused segment's while editing
    /// segments, else the whole composition's; `None` while typing.
    pub fn shown_reading(&self) -> Option<String> {
        match &self.phase {
            Phase::Converting { seg: Some(s), .. } => Some(s.reading_of(s.focus)),
            Phase::Converting { .. } => Some(self.composing.full_reading()),
            _ => None,
        }
    }

    /// The F6–F10 form on screen, if one is.
    pub fn shown_form(&self) -> Option<Form> {
        self.form_shown().map(|f| f.form)
    }

    /// The F6–F10 conversion Converting shows, if that is what it shows.
    fn form_shown(&self) -> Option<&FormShown> {
        let f = self.comp.form.as_ref()?;
        match &self.phase {
            Phase::Converting {
                candidates,
                seg: None,
                ..
            } if candidates.len() == 1 && candidates[0] == f.text => Some(f),
            _ => None,
        }
    }

    /// Rerank candidates by (and record choices into) `learning`.
    pub fn set_learning(&mut self, learning: Option<Learning>) {
        self.learning = learning;
    }

    /// Leave the candidates hidden with Ctrl+Delete out of the lists.
    pub fn set_hidden(&mut self, hidden: Option<crate::hidden::Hidden>) {
        self.hidden = hidden;
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.phase, Phase::Empty)
    }

    /// An engine client whose backend is still up: false after the daemon
    /// exited (killed by an install, crashed), so that the frontend starts
    /// a new one instead of dispatching to it forever.
    pub fn has_client(&self) -> bool {
        self.client.as_ref().is_some_and(|c| c.is_alive())
    }

    pub fn reset_preserving_client(&mut self, frontend: &crate::config::FrontendCfg) {
        let counter = self.req_counter.load(Ordering::Relaxed);
        let client = self.client.take();
        let reranker = self.reranker.take();
        let rerank_cfg = self.cfg.rerank_cfg.clone();
        let ngram = self.ngram.take();
        let rewriter = self.rewriter.take();
        let rewrite_mode = self.cfg.rewrite_mode;
        let (page_size, digit_select, live) = (
            self.cfg.page_size,
            self.cfg.digit_select,
            self.cfg.live_conversion,
        );
        *self = Self::with_frontend(client, frontend);
        self.req_counter.store(counter, Ordering::Relaxed);
        self.invalidate_engine_results();
        self.set_candidate_paging(page_size, digit_select);
        self.set_reranker(reranker, &rerank_cfg);
        self.ngram = ngram;
        self.rewriter = rewriter;
        self.cfg.rewrite_mode = rewrite_mode && self.rewriter.is_some();
        if frontend.live_conversion.is_none() {
            self.cfg.live_conversion = live;
        }
    }

    /// New rerank settings for the reranker already running.
    pub fn set_rerank_config(&mut self, cfg: &crate::config::RerankCfg) {
        self.cfg.rerank_cfg = cfg.clone();
    }

    /// Reorder the engine's candidates by character n-gram scores as
    /// `[ngram]` says (off without a table, or when it does not load).
    pub fn set_ngram(&mut self, cfg: &crate::config::NgramCfg) {
        self.ngram = cfg
            .table
            .as_deref()
            .filter(|p| !p.is_empty())
            .and_then(crate::char_ngram::shared)
            .map(|t| (t, cfg.bigram_weight, cfg.trigram_weight));
    }

    /// Score the engine's candidates by the reading lexicon as `[lexicon]`
    /// says (off without a file, or when it does not open).
    pub fn set_lexicon(&mut self, cfg: &crate::config::LexiconCfg) {
        self.lexicon = cfg
            .file
            .as_deref()
            .filter(|p| !p.is_empty())
            .and_then(crate::lexicon::shared);
        self.composing.set_lexicon(self.lexicon.clone());
    }

    /// The user dictionary in effect: the file's, read again when it
    /// changed, or the one given.
    fn user_dict(&self) -> Option<Arc<crate::user_dict::UserDict>> {
        match &self.user_dict {
            UserDictSource::None => None,
            UserDictSource::Fixed(d) => Some(d.clone()),
            UserDictSource::File => crate::user_dict::shared(),
        }
    }

    /// Use the user dictionary file of the config directory
    /// (`user_dict::shared`).
    pub fn use_user_dictionary_file(&mut self) {
        self.user_dict = UserDictSource::File;
    }

    /// Use `dict` as the user dictionary (`None`: none).
    pub fn set_user_dict(&mut self, dict: Option<crate::user_dict::UserDict>) {
        self.user_dict = match dict {
            Some(d) => UserDictSource::Fixed(Arc::new(d)),
            None => UserDictSource::None,
        };
    }

    /// Drop the composition without committing it (the host reset the
    /// input context), keeping the client and the learning. The caret may
    /// have moved, so the left context goes too.
    pub fn discard(&mut self) -> Vec<Action> {
        self.invalidate_engine_results();
        let was_active = self.is_active();
        self.phase = Phase::Empty;
        self.composing.reset();
        self.clear_composition();
        self.forget_context();
        if was_active {
            vec![Action::HideCandidates, Action::EndComposition]
        } else {
            Vec::new()
        }
    }

    /// Drop the composition without committing it (Escape on the kana, or
    /// the app ended the composition), keeping the context.
    pub fn cancel(&mut self) -> Vec<Action> {
        if !self.is_active() {
            return Vec::new();
        }
        let converting = self.is_converting();
        self.invalidate_engine_results();
        self.composing.reset();
        self.phase = Phase::Empty;
        self.clear_composition();
        let mut actions = Vec::new();
        if converting {
            actions.push(Action::HideCandidates);
        }
        actions.push(Action::EndComposition);
        actions
    }

    /// Drop the left context: the text before the caret is no longer what
    /// this IME committed (a key went to the app, focus moved, the IME was
    /// toggled). An empty context is what the model sees at the start of a
    /// document; a stale one is text from elsewhere.
    pub fn forget_context(&mut self) {
        self.ctx.at_word_start = None;
        self.ctx.passing = false;
        self.ctx.raw_context.clear();
        self.ctx.context.clear();
        self.composing.set_before(&self.ctx.context);
    }

    /// The text before the caret as the app holds it, read as a composition
    /// starts: it replaces the context kept from this IME's own commits.
    /// `None` or empty (an app that does not expose its text, or the start
    /// of a document) keeps that one. Line breaks and embedded objects are
    /// dropped; the model reads running text.
    pub fn set_app_context(&mut self, text: Option<&str>) {
        if self.ctx.private_field {
            self.forget_context();
            return;
        }
        let Some(text) = text else {
            self.ctx.at_word_start = None;
            return;
        };
        self.ctx.at_word_start = Some(text.chars().last().is_none_or(char::is_whitespace));
        let text: String = text
            .chars()
            .filter(|&c| !c.is_control() && !matches!(c, '\u{2028}' | '\u{2029}' | '\u{FFFC}'))
            .collect();
        if text.is_empty() {
            return;
        }
        self.ctx.raw_context.clear();
        self.append_context(&text);
    }

    /// The field typed into is private (`private_field`), or no longer is.
    /// The frontend tells it as a composition starts.
    pub fn set_private_field(&mut self, private: bool) {
        self.ctx.private_field = private;
        if private {
            self.forget_context();
        }
    }

    #[cfg(test)]
    pub(crate) fn context(&self) -> &str {
        &self.ctx.context
    }

    fn append_context(&mut self, text: &str) {
        if text.is_empty() || self.ctx.private_field {
            return;
        }
        self.ctx.raw_context.push_str(text);
        let n = self.ctx.raw_context.chars().count();
        if n > CONTEXT_WINDOW {
            let skip = n - CONTEXT_WINDOW;
            self.ctx.raw_context = self.ctx.raw_context.chars().skip(skip).collect();
        }
        self.ctx.context = crate::privacy::after_last_hit(&self.ctx.raw_context).to_string();
        self.composing.set_before(&self.ctx.context);
    }

    pub fn would_consume(&self, kind: &KeyKind) -> bool {
        match kind {
            // a command's name goes to the app
            KeyKind::Char(_) if self.ctx.passing && !self.is_active() => false,
            KeyKind::Char(c) => {
                c.is_ascii_alphabetic()
                    || c.is_ascii_digit()
                    || *c == '-'
                    || crate::romaji_table::active().is_some_and(|t| t.uses(*c))
            }
            KeyKind::Comma
            | KeyKind::Period
            | KeyKind::Exclaim
            | KeyKind::Question
            | KeyKind::Punctuation(_) => true,
            KeyKind::Space => self.is_active() || self.cfg.full_width_space,
            KeyKind::ShiftSpace => self.is_active() || !self.cfg.full_width_space,
            KeyKind::Backspace
            | KeyKind::Enter
            | KeyKind::Escape
            | KeyKind::Delete
            | KeyKind::Home
            | KeyKind::End
            | KeyKind::Tab
            | KeyKind::ShiftTab
            | KeyKind::ConvertTo(_)
            | KeyKind::Left
            | KeyKind::Right
            | KeyKind::ShiftLeft
            | KeyKind::ShiftRight
            | KeyKind::Up
            | KeyKind::Down
            | KeyKind::PageUp
            | KeyKind::PageDown => self.is_active(),
            KeyKind::ForgetCandidate => self.is_converting(),
        }
    }

    /// Predictions for what is typed: commits whose reading starts with it
    /// (`predict`), up to `PREDICTIONS`.
    pub fn predictions(&self) -> Vec<crate::predict::Prediction> {
        let Some(p) = self.predictor.as_ref() else {
            return Vec::new();
        };
        if !matches!(self.phase, Phase::Composing { .. })
            || !self.composing.cursor_at_end()
            || self.ctx.private_field
        {
            return Vec::new();
        }
        p.borrow_mut().complete(
            &self.composing.live_reading(),
            &self.ctx.last_commit,
            PREDICTIONS,
        )
    }

    /// Predict from the commits of the config directory when `[frontend]
    /// prediction` is on.
    pub fn open_predictor(&mut self, frontend: &crate::config::FrontendCfg) {
        self.set_predictor(if frontend.prediction {
            crate::predict::Predictor::open()
        } else {
            None
        });
    }

    /// How long until the deferred preedit is shown anyway (`tick`), if one
    /// waits: the frontend sets a timer.
    pub fn deferred_for(&self) -> Option<Duration> {
        self.comp
            .deferred
            .as_ref()
            .map(|d| d.until.saturating_duration_since(Instant::now()))
    }

    /// Predict from `predictor` (`None`: no prediction).
    pub fn set_predictor(&mut self, predictor: Option<crate::predict::Predictor>) {
        self.predictor = predictor.map(std::cell::RefCell::new);
    }
}
