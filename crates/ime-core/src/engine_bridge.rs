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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Local};

use crate::char_forms::{self, Form};
use crate::composing_text::ComposingText;
use crate::engine_client::{EngineClient, EngineResult};
use crate::learning::Learning;
use crate::postprocess::CharRules;
use crate::rerank_client::RerankClient;
use crate::rewrite_client::RewriteClient;
use crate::segmented::Segmented;

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

/// Mixed input while converting: the list offers the reading read with
/// English (`main`, the engine's list for it) and with every letter as
/// romaji (`candidates`, asked for once `main` is in), then the input as
/// typed. Ending in English, its top candidate with a space comes first:
/// the next key commits it, Space again goes on to the Japanese ones.
struct AltList {
    /// `None` when every letter is read as romaji already, or some letter
    /// reads as none (`google` → ごおgぇ: its conversions are noise).
    reading: Option<String>,
    req_id: Option<u64>,
    candidates: Vec<String>,
    main: Vec<String>,
    ends_english: bool,
    typed: String,
    /// The input is one word the model knows as Latin (`made` read まで):
    /// offer it as typed among the first candidates.
    typed_early: bool,
}

/// Candidates of the other reading offered (its first `ALT_EARLY` right
/// after the first `ALT_EARLY` of the mixed one).
const ALT_MAX: usize = 5;
const ALT_EARLY: usize = 3;

/// Predictions offered while typing.
const PREDICTIONS: usize = 3;

/// The engine's top of what is typed, waiting for the reranked one
/// (`[rerank] defer_ms`).
struct Deferred {
    reading: String,
    top: String,
    until: Instant,
}

/// Where the user dictionary comes from.
enum UserDictSource {
    None,
    /// Given (tests, the demo).
    Fixed(Arc<crate::user_dict::UserDict>),
    /// The file in the config directory, read again when it changes.
    File,
}

enum Phase {
    Empty,
    Composing {
        candidates: Vec<String>,
    },
    Converting {
        candidates: Vec<String>,
        selected: usize,
        /// Segment editing, once ← / → / Shift+← / Shift+→ was pressed.
        seg: Option<Segmented>,
    },
    /// Temporary raw-ASCII mode. Entered via Shift+letter, exited by
    /// Enter (commits buffer) or Escape (discards). Mirrors MS-IME /
    /// karukan: letters (case preserved), digits, space and `-` append
    /// verbatim; any other key commits the buffer first and then re-
    /// enters the normal flow.
    Alphabet {
        buffer: String,
    },
}

pub struct EngineBridge {
    composing: ComposingText,
    phase: Phase,
    /// Conversion backend. `None` in hosts where the engine daemon failed
    /// to spawn — live conversion then degrades to a raw-hiragana preedit.
    client: Option<Box<dyn EngineClient>>,
    /// The text before the caret as read or committed (`CONTEXT_WINDOW`
    /// chars): what decides whether the rewriter may see it.
    raw_context: String,
    /// What the models read of it: the text after anything sensitive in it
    /// (`privacy::after_last_hit`: a URL, a token, パスワード).
    context: String,
    req_counter: AtomicU64,
    /// Results older than this belong to a previous composition.
    result_floor: u64,
    /// Reading that the current `Phase::Composing.candidates` matches.
    /// Prevents redundant dispatches while the user types mid-kana and the
    /// hiragana portion is unchanged.
    last_applied_reading: String,
    /// Kanji rendering of the last successfully-converted **prefix** of
    /// hiragana (the reading of the last applied result). Used to render the
    /// preedit as `cached_kanji + raw_tail + pending` instantly on every
    /// keystroke — the user's just-typed characters stay visible as
    /// hiragana while earlier chars settle into kanji once the engine
    /// returns.
    cached_prefix_reading: String,
    cached_prefix_kanji: String,
    live_min_dispatch_chars: usize,
    force_convert_on_space: bool,
    symbol_width: crate::config::SymbolWidth,
    /// Punctuation / middle-dot / replacement rules of the frontend config.
    char_rules: CharRules,
    drop_untyped_symbols: bool,
    /// `None`: `[frontend] symbol_dictionary = false`.
    symbol_dict: Option<std::sync::Arc<crate::symbol_dict::SymbolDict>>,
    date_time_enabled: bool,
    /// Number candidates (`numbers`).
    numbers_enabled: bool,
    /// Typo correction (`typo`).
    typo_correction: bool,
    /// Emoji and kaomoji candidates (`emoji`).
    emoji_enabled: bool,
    /// Clock snapshot shared by local candidates and delayed model results.
    date_time_now: Option<DateTime<Local>>,
    /// With mixed input (`[frontend] mixed_model`): the ASCII char each
    /// `[frontend.symbols]` output was typed as. Typed symbols go into the
    /// composition as ASCII, shown per span.
    mixed_symbols: HashMap<String, char>,
    /// Learned candidate choices; `None` when learning is off.
    learning: Option<Learning>,
    /// Candidates hidden with Ctrl+Delete.
    hidden: Option<crate::hidden::Hidden>,
    /// The user dictionary.
    user_dict: UserDictSource,
    /// Prediction from the commits (`predict`).
    predictor: Option<std::cell::RefCell<crate::predict::Predictor>>,
    /// The last commit: what the next one is predicted after.
    last_commit: String,
    /// The predictions open as the candidate list (Tab while typing).
    prediction: Option<Vec<crate::predict::Prediction>>,
    /// The engine's top waiting for the reranked one.
    deferred: Option<Deferred>,
    /// `/command` and the like go to the app (`pass_through_commands`).
    pass_commands: bool,
    /// Whether the caret is at a line's start or after a space, when the
    /// app's text says (`None`: unknown).
    at_word_start: Option<bool>,
    /// A command's name being typed through to the app.
    passing: bool,
    /// Reading the current candidate list (Composing / Converting) was
    /// produced for; empty for a provisional list. A choice is learned
    /// against it.
    list_reading: String,
    /// Reading start of each char of the current list's candidates, by
    /// candidate text (`segments::char_starts`): what segment editing cuts
    /// them with.
    list_starts: HashMap<String, Vec<usize>>,
    /// Segment conversions in flight: (request id, segment index, its
    /// `continue_beam`: 0 for the segment's own conversion).
    seg_requests: Vec<(u64, usize, u32)>,
    /// Candidates per page of the frontend's window: what PageUp / PageDown
    /// step by and what the digit keys pick from.
    page_size: usize,
    /// The digit keys pick a candidate of the open page (1–9, 0 = 10th)
    /// instead of typing; for a window that labels its candidates.
    digit_select: bool,
    full_width_space: bool,
    /// `[frontend] sentence_case`: the punctuation typed after an English
    /// sentence is English (`mixed::english_after`).
    sentence_case: bool,
    /// Typing shows the conversion in the preedit (else the kana, with the
    /// conversions offered by [`Self::suggestions`]).
    live_conversion: bool,
    /// The last F6–F10 conversion, while its text is what Converting shows.
    form: Option<FormShown>,
    /// The character n-gram table and its bigram and trigram weights
    /// (`[ngram]`); `None` when off.
    ngram: Option<(Arc<crate::char_ngram::CharNgram>, f32, f32)>,
    /// The reading lexicon (`[lexicon]`): off without a file.
    lexicon: Option<Arc<crate::lexicon::Lexicon>>,
    /// The reranker; `None` when reranking is off.
    reranker: Option<Box<dyn RerankClient>>,
    rerank_cfg: crate::config::RerankCfg,
    /// The whole-reading conversion being reranked.
    rerank: Option<RerankJob>,
    /// Mixed input: the other reading of the conversion open.
    alt: Option<AltList>,
    /// The rewriter; `None` when rewriting is off.
    rewriter: Option<Box<dyn RewriteClient>>,
    /// Rewrite mode: the conversion key also asks for a rewrite.
    rewrite_mode: bool,
    /// The rewrite of the conversion open, asked for or in.
    rewrite: Option<RewriteJob>,
    /// The field typed into is private (a password, a PIN, an input scope
    /// marked private): its text is no context, nothing is rewritten and
    /// nothing learned (learned choices still apply).
    private_field: bool,
}

/// A rewrite of the whole reading's conversion.
struct RewriteJob {
    req_id: u64,
    reading: String,
    /// The rewrite, once in.
    text: Option<String>,
}

/// A whole-reading conversion under rerank: the engine's candidates as it
/// sent them, the first `k` of which are being (or were) scored by the LM.
struct RerankJob {
    reading: String,
    context: String,
    raw: Vec<String>,
    scores: Vec<f32>,
    /// Per candidate, whether it spells the reading out
    /// (`order_engine_result`): those stay above the others when reranked.
    spelled: Vec<bool>,
    reading_pos: Vec<Vec<u32>>,
    k: usize,
    req_id: u64,
    /// When the request for the current `k` is due.
    due: Instant,
}

/// An F6–F10 conversion on screen: the form, the press count (F9 / F10
/// cycle the case), the text and the candidates it stands in for, which
/// Space / Escape return to.
struct FormShown {
    form: Form,
    case: usize,
    text: String,
    list: Vec<String>,
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
            client,
            raw_context: String::new(),
            context: String::new(),
            req_counter: AtomicU64::new(0),
            result_floor: 1,
            last_applied_reading: String::new(),
            cached_prefix_reading: String::new(),
            cached_prefix_kanji: String::new(),
            live_min_dispatch_chars: frontend.live_min_dispatch_chars,
            force_convert_on_space: frontend.force_convert_on_space,
            symbol_width: frontend.symbol_width,
            char_rules: CharRules::from_frontend(frontend),
            drop_untyped_symbols: frontend.drop_untyped_symbols,
            symbol_dict: frontend
                .symbol_dictionary
                .then(|| crate::symbol_dict::shared(frontend.symbol_dictionary_file.as_deref())),
            date_time_enabled: frontend.date_time_candidates,
            numbers_enabled: frontend.number_candidates,
            typo_correction: frontend.typo_correction,
            emoji_enabled: frontend.emoji_candidates,
            date_time_now: None,
            mixed_symbols: HashMap::new(),
            learning: None,
            hidden: None,
            user_dict: UserDictSource::None,
            predictor: None,
            last_commit: String::new(),
            prediction: None,
            deferred: None,
            pass_commands: frontend.pass_through_commands,
            at_word_start: None,
            passing: false,
            list_reading: String::new(),
            list_starts: HashMap::new(),
            seg_requests: Vec::new(),
            page_size: 10,
            digit_select: false,
            full_width_space: frontend.full_width_space,
            sentence_case: frontend.sentence_case,
            live_conversion: frontend.live_conversion.unwrap_or(true),
            form: None,
            ngram: None,
            lexicon: None,
            reranker: None,
            rerank_cfg: crate::config::RerankCfg::default(),
            rerank: None,
            alt: None,
            rewriter: None,
            rewrite_mode: false,
            private_field: false,
            rewrite: None,
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
        bridge
    }

    /// Mixed Japanese / English input with `model` (`None`: off);
    /// `symbols` is `[frontend.symbols]`, `casing` what the English is
    /// cased.
    fn set_mixed(
        &mut self,
        model: Option<std::sync::Arc<crate::mixed::MixedModel>>,
        symbols: &std::collections::BTreeMap<String, String>,
        casing: crate::mixed::Casing,
    ) {
        self.mixed_symbols = if model.is_some() {
            crate::config::SYMBOL_CHARS
                .iter()
                .filter_map(|(name, c)| symbols.get(*name).map(|s| (s.clone(), *c)))
                .collect()
        } else {
            HashMap::new()
        };
        self.composing.set_mixed(model, symbols, casing);
        self.composing.set_before(&self.context);
    }

    /// Rerank the conversions with `reranker` (an LM), as `cfg` says.
    pub fn set_reranker(
        &mut self,
        reranker: Option<Box<dyn RerankClient>>,
        cfg: &crate::config::RerankCfg,
    ) {
        self.reranker = reranker;
        self.rerank_cfg = cfg.clone();
        self.rerank = None;
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
        self.rewrite_mode &= self.rewriter.is_some();
    }

    /// A rewriter whose backend is still up.
    pub fn has_rewriter(&self) -> bool {
        self.rewriter.as_ref().is_some_and(|r| r.is_alive())
    }

    pub fn rewrite_mode(&self) -> bool {
        self.rewrite_mode
    }

    /// Turn rewrite mode on or off (on only with a rewriter). Turned on
    /// with a conversion open, it asks for that conversion's rewrite. A
    /// rewrite already asked for still lands.
    pub fn set_rewrite_mode(&mut self, on: bool) -> Vec<Action> {
        self.rewrite_mode = on && self.rewriter.is_some();
        if !self.rewrite_mode || !self.window_open() {
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
        self.page_size = page_size.max(1);
        self.digit_select = digit_select;
    }

    /// Whether typing shows the conversion in the preedit; `false` keeps
    /// the kana there for a frontend that lists [`Self::suggestions`].
    pub fn set_live_conversion(&mut self, live: bool) {
        self.live_conversion = live;
    }

    /// The conversions of what is typed, for a suggestion window beside a
    /// kana preedit: the typing phase's candidates when they are for the
    /// whole reading. Empty otherwise, and with live conversion on: no
    /// window comes up while typing, the predictions open with Tab only
    /// (user 2026-10-23).
    pub fn suggestions(&self) -> Vec<String> {
        match &self.phase {
            Phase::Composing { candidates }
                if !self.live_conversion
                    && !candidates.is_empty()
                    && self.composing.cursor_at_end()
                    && self.list_reading == self.composing.live_reading() =>
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
        let f = self.form.as_ref()?;
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

    /// Ctrl+Delete on the candidate selected (Mozc's
    /// DeleteSelectedCandidate): what was learned to put it up is forgotten;
    /// when nothing was, it is hidden for this reading from now on
    /// (`hidden.jsonl`). Either way it leaves the list open. The kana of the
    /// reading and a list's last candidate stay.
    pub fn handle_forget_candidate(&mut self) -> KeyOutcome {
        let Some(reading) = self.shown_reading() else {
            return not_consumed();
        };
        if self.form_shown().is_some() {
            return consumed(Vec::new());
        }
        let (cand, count) = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &s.segs[s.focus];
                (f.cands.get(f.selected).cloned(), f.cands.len())
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => (candidates.get(*selected).cloned(), candidates.len()),
            _ => return not_consumed(),
        };
        let Some(cand) = cand else {
            return consumed(Vec::new());
        };
        let forgot = self
            .learning
            .as_mut()
            .is_some_and(|l| l.forget(&reading, &cand).unwrap_or(false));
        let message = if forgot {
            "学習を取り消しました"
        } else if count <= 1
            || cand == reading
            || cand == crate::kana_candidates::to_katakana(&reading)
        {
            return consumed(vec![Action::Status {
                text: "この候補は消せません".to_string(),
            }]);
        } else {
            match self.hidden.as_mut().map(|h| h.hide(&reading, &cand)) {
                Some(Ok(())) => "この候補を表示しないようにしました",
                _ => return consumed(Vec::new()),
            }
        };
        // out of the list open
        let mut actions = match &mut self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &mut s.segs[s.focus];
                f.cands.retain(|c| c != &cand);
                f.selected = f.selected.min(f.cands.len().saturating_sub(1));
                self.segment_actions()
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => {
                candidates.retain(|c| c != &cand);
                *selected = (*selected).min(candidates.len().saturating_sub(1));
                let (list, sel) = (candidates.clone(), *selected);
                let text = self.with_pending(list[sel].clone());
                vec![
                    Action::UpdatePreedit { text },
                    Action::ShowCandidates {
                        list,
                        selected: sel,
                    },
                ]
            }
            _ => Vec::new(),
        };
        actions.push(Action::Status {
            text: message.to_string(),
        });
        consumed(actions)
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
        let rerank_cfg = self.rerank_cfg.clone();
        let ngram = self.ngram.take();
        let rewriter = self.rewriter.take();
        let rewrite_mode = self.rewrite_mode;
        let (page_size, digit_select, live) =
            (self.page_size, self.digit_select, self.live_conversion);
        *self = Self::with_frontend(client, frontend);
        self.req_counter.store(counter, Ordering::Relaxed);
        self.invalidate_engine_results();
        self.set_candidate_paging(page_size, digit_select);
        self.set_reranker(reranker, &rerank_cfg);
        self.ngram = ngram;
        self.rewriter = rewriter;
        self.rewrite_mode = rewrite_mode && self.rewriter.is_some();
        if frontend.live_conversion.is_none() {
            self.live_conversion = live;
        }
    }

    /// New rerank settings for the reranker already running.
    pub fn set_rerank_config(&mut self, cfg: &crate::config::RerankCfg) {
        self.rerank_cfg = cfg.clone();
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
    }

    /// The engine's candidates that spell the reading out first, by their
    /// scores with the character n-gram terms added (`[ngram]`: those sums
    /// become their scores, which the LM reranker then adds to) and their
    /// lexicon costs taken off (`[lexicon]`, which also drops those far
    /// above the cheapest), then those that do not, in the engine's order
    /// and scored no higher than the others. A candidate that does not has
    /// lost a char or put a kana out of place (ゼロプラモデル for
    /// ぜろせんぷらもでる): never the gold over wiki_fresh_v2 and AJIMEE,
    /// and one the n-gram would favour, with fewer pairs to lose points on.
    fn order_engine_result(&self, result: &mut EngineResult) {
        use unicode_normalization::UnicodeNormalization;
        let mut n = result.candidates.len();
        if n < 2 {
            return;
        }
        let reading: Vec<char> = result.reading.nfkc().collect();
        let (mut order, mut rest): (Vec<usize>, Vec<usize>) = (0..n)
            .partition(|&i| crate::reading_align::spells_out(&result.candidates[i], &reading));
        if order.is_empty() {
            std::mem::swap(&mut order, &mut rest);
        }
        let scored = result.scores.len() == n;
        if rest.is_empty() && !(scored && (self.ngram.is_some() || self.lexicon.is_some())) {
            return;
        }
        let mut scores = result.scores.clone();
        if let (Some((table, w2, w3)), true) = (&self.ngram, scored && order.len() >= 2) {
            let cands: Vec<String> = order
                .iter()
                .map(|&i| result.candidates[i].clone())
                .collect();
            let own: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
            let (by, total) = table.rescore(&result.context, &cands, &own, *w2, *w3);
            for (k, &i) in order.iter().enumerate() {
                scores[i] = total[k];
            }
            order = by.iter().map(|&k| order[k]).collect();
        }
        if let (Some(lex), true) = (&self.lexicon, scored) {
            let r: String = reading.iter().collect();
            let mut costs: Vec<Option<f32>> = result
                .candidates
                .iter()
                .map(|c| lex.cost(&c.nfkc().collect::<String>(), &r))
                .collect();
            // the lexicon's own conversions the engine missed (大島村 for
            // おおしまむら, where it gave おおしま村), those `ADDED_UNDER`
            // cheaper than every candidate of the engine, scored
            // `ADDED_BELOW` under its top: the lexicon's cost and the
            // reranker place them
            let top = result.scores[0];
            let cheapest = costs.iter().flatten().fold(f32::INFINITY, |a, &b| a.min(b));
            for (surface, cost) in lex.convert(&r, crate::lexicon::CONVERSIONS) {
                if cost > cheapest - crate::lexicon::ADDED_UNDER
                    || result
                        .candidates
                        .iter()
                        .any(|c| c.nfkc().eq(surface.chars()))
                {
                    continue;
                }
                order.push(result.candidates.len());
                result.candidates.push(surface);
                scores.push(top - crate::lexicon::ADDED_BELOW);
                costs.push(Some(cost));
                if result.reading_pos.len() == n {
                    result.reading_pos.push(Vec::new());
                }
                n += 1;
            }
            let best = costs.iter().flatten().fold(f32::INFINITY, |a, &b| a.min(b));
            if best.is_finite() {
                let worst = best + crate::lexicon::MARGIN;
                // the engine's top stays whatever it costs
                let keep = |i: &usize| *i == 0 || costs[*i].is_some_and(|c| c <= worst);
                order.retain(keep);
                rest.retain(keep);
                if order.is_empty() {
                    std::mem::swap(&mut order, &mut rest);
                }
                for (i, c) in costs.iter().enumerate() {
                    scores[i] -= crate::lexicon::WEIGHT * c.unwrap_or(worst);
                }
                order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            }
        }
        if scored {
            let floor = order
                .iter()
                .map(|&i| scores[i])
                .fold(f32::INFINITY, f32::min);
            for &i in &rest {
                scores[i] = scores[i].min(floor);
            }
        }
        order.extend(rest);
        fn permute<T>(v: &mut Vec<T>, order: &[usize]) {
            let mut old: Vec<Option<T>> = std::mem::take(v).into_iter().map(Some).collect();
            *v = order.iter().map(|&i| old[i].take().unwrap()).collect();
        }
        permute(&mut result.candidates, &order);
        if result.reading_pos.len() == n {
            permute(&mut result.reading_pos, &order);
        }
        if scored {
            result.scores = order.iter().map(|&i| scores[i]).collect();
        }
    }

    /// Drop the composition without committing it (the host reset the
    /// input context), keeping the client and the learning. The caret may
    /// have moved, so the left context goes too.
    pub fn discard(&mut self) -> Vec<Action> {
        self.invalidate_engine_results();
        let was_active = self.is_active();
        self.phase = Phase::Empty;
        self.composing.reset();
        self.seg_requests.clear();
        self.list_reading.clear();
        self.list_starts.clear();
        self.last_applied_reading.clear();
        self.cached_prefix_reading.clear();
        self.cached_prefix_kanji.clear();
        self.stop_rerank();
        self.stop_rewrite();
        self.forget_context();
        if was_active {
            vec![Action::HideCandidates, Action::EndComposition]
        } else {
            Vec::new()
        }
    }

    const CONTEXT_WINDOW: usize = 80;

    /// Drop the left context: the text before the caret is no longer what
    /// this IME committed (a key went to the app, focus moved, the IME was
    /// toggled). An empty context is what the model sees at the start of a
    /// document; a stale one is text from elsewhere.
    pub fn forget_context(&mut self) {
        self.at_word_start = None;
        self.passing = false;
        self.raw_context.clear();
        self.context.clear();
        self.composing.set_before(&self.context);
    }

    /// The text before the caret as the app holds it, read as a composition
    /// starts: it replaces the context kept from this IME's own commits.
    /// `None` or empty (an app that does not expose its text, or the start
    /// of a document) keeps that one. Line breaks and embedded objects are
    /// dropped; the model reads running text.
    pub fn set_app_context(&mut self, text: Option<&str>) {
        if self.private_field {
            self.forget_context();
            return;
        }
        let Some(text) = text else {
            self.at_word_start = None;
            return;
        };
        self.at_word_start = Some(text.chars().last().is_none_or(char::is_whitespace));
        let text: String = text
            .chars()
            .filter(|&c| !c.is_control() && !matches!(c, '\u{2028}' | '\u{2029}' | '\u{FFFC}'))
            .collect();
        if text.is_empty() {
            return;
        }
        self.raw_context.clear();
        self.append_context(&text);
    }

    /// The field typed into is private (`private_field`), or no longer is.
    /// The frontend tells it as a composition starts.
    pub fn set_private_field(&mut self, private: bool) {
        self.private_field = private;
        if private {
            self.forget_context();
        }
    }

    #[cfg(test)]
    pub(crate) fn context(&self) -> &str {
        &self.context
    }

    fn append_context(&mut self, text: &str) {
        if text.is_empty() || self.private_field {
            return;
        }
        self.raw_context.push_str(text);
        let n = self.raw_context.chars().count();
        if n > Self::CONTEXT_WINDOW {
            let skip = n - Self::CONTEXT_WINDOW;
            self.raw_context = self.raw_context.chars().skip(skip).collect();
        }
        self.context = crate::privacy::after_last_hit(&self.raw_context).to_string();
        self.composing.set_before(&self.context);
    }

    pub fn would_consume(&self, kind: &KeyKind) -> bool {
        match kind {
            // a command's name goes to the app
            KeyKind::Char(_) if self.passing && !self.is_active() => false,
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
            KeyKind::Space => self.is_active() || self.full_width_space,
            KeyKind::ShiftSpace => self.is_active() || !self.full_width_space,
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

    /// Called on the UI thread when the engine client notifies a completed
    /// convert. We accept the result as long as its `reading`
    /// is still a **prefix** of the current hiragana — if the user kept
    /// typing while the engine was converting, the newly-typed tail will
    /// show as raw kana after the cached prefix (no whole-string
    /// flicker). If the user back-spaced past the cached prefix the
    /// result is dropped.
    pub fn apply_async_result(&mut self) -> Vec<Action> {
        let mut actions = self.apply_engine_result();
        actions.extend(self.apply_rerank_result());
        actions.extend(self.apply_rewrite_result());
        actions
    }

    fn apply_engine_result(&mut self) -> Vec<Action> {
        let Some(client) = self.client.as_ref() else {
            return Vec::new();
        };
        let Some(mut result) = client.take_latest() else {
            return Vec::new();
        };
        if result.req_id < self.result_floor {
            return Vec::new();
        }
        self.order_engine_result(&mut result);
        if self
            .alt
            .as_ref()
            .is_some_and(|alt| alt.req_id == Some(result.req_id))
        {
            return self.apply_alt_result(result);
        }
        if let Some(actions) = self.apply_segment_result(&result) {
            return actions;
        }
        let raw = std::mem::take(&mut result.candidates);
        let list = self.finish_list(&result.reading, raw.clone(), true);
        let starts = self.list_starts_of(&result.reading, &raw, &result.reading_pos, &list);
        result.candidates = list;
        let current_hira = self.composing.live_reading();

        // Live typing: accept the result while its reading is still a prefix
        // of the current reading; the freshly-typed tail then shows as raw
        // kana after the cached prefix.
        if matches!(self.phase, Phase::Composing { .. }) {
            if !current_hira.starts_with(&result.reading) {
                return Vec::new();
            }
            let top = result.candidates.first().cloned().unwrap_or_default();
            if top.is_empty() {
                return Vec::new();
            }
            if let Phase::Composing { candidates } = &mut self.phase {
                *candidates = result.candidates.clone();
            }
            self.list_reading = result.reading.clone();
            self.list_starts = starts;
            // the conversion of all that is typed: rerank it once typing pauses
            let whole = result.reading == current_hira;
            // one that reads a pending n as ん is what Enter commits; the
            // preview keeps the conversion before it and shows the n, as
            // the next key may make it な (共犯 for きょうはn, then the whole
            // in kana while きょうはな was converted)
            let n_read = whole && self.composing.n_pending();
            if whole {
                self.start_rerank(
                    &result.reading,
                    &raw,
                    &result.scores,
                    &result.reading_pos,
                    false,
                );
            }
            let reranking = self
                .rerank
                .as_ref()
                .is_some_and(|j| j.reading == result.reading);
            if n_read {
                self.deferred = None;
            } else if whole && reranking && self.rerank_cfg.defer_ms > 0 {
                // shown once the reranked top is in, or at the deadline: the
                // preedit changes once (what was shown stays meanwhile)
                self.deferred = Some(Deferred {
                    reading: result.reading.clone(),
                    top,
                    until: Instant::now()
                        + Duration::from_millis(self.rerank_cfg.idle_ms + self.rerank_cfg.defer_ms),
                });
            } else {
                self.deferred = None;
                self.cached_prefix_reading = result.reading.clone();
                self.cached_prefix_kanji = top;
            }
            self.last_applied_reading = current_hira;
            return vec![Action::UpdatePreedit {
                text: self.live_preedit(),
            }];
        }

        // Candidate window open: the reading is frozen, so accept only an
        // exact match and swap the provisional list for the real one.
        if self.form_shown().is_some() {
            return Vec::new();
        }
        let (selected, chosen) = match &self.phase {
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => (
                *selected,
                (*selected > 0)
                    .then(|| candidates.get(*selected).cloned())
                    .flatten(),
            ),
            _ => return Vec::new(),
        };
        if current_hira != result.reading || result.candidates.is_empty() {
            return Vec::new();
        }
        let candidates = self.shown_list(result.candidates.clone());
        // Locally generated candidates may move when the model arrives.
        // Keep a deliberate choice on its text rather than its old index.
        let sel = chosen
            .and_then(|c| candidates.iter().position(|t| *t == c))
            .unwrap_or(if selected < candidates.len() {
                selected
            } else {
                0
            });
        let text = self.with_pending(candidates[sel].clone());
        let list = candidates;
        // the candidate window is open: rerank at once
        self.start_rerank(
            &result.reading,
            &raw,
            &result.scores,
            &result.reading_pos,
            true,
        );
        self.list_reading = result.reading;
        self.list_starts = starts;
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        self.request_alt();
        vec![
            Action::UpdatePreedit { text },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
        ]
    }

    /// Mixed input, Space on a composition with English in it: the other
    /// reading to offer (every letter as romaji), or `None`.
    fn alt_for_space(&self) -> Option<AltList> {
        // a typo: the reading one edit away, when a letter reads as no kana
        // and nothing was read as English
        let typo = || {
            // typed in capitals, English was meant (mixed input reads a
            // lowercase word it does not know as English too: kais + が)
            let shifted = self
                .composing
                .romaji()
                .chars()
                .any(|c| c.is_ascii_uppercase());
            if !self.typo_correction || shifted {
                return None;
            }
            crate::typo::correct(
                &self.composing.romaji(),
                self.composing.mixed_model().as_deref(),
            )
        };
        if !self.composing.is_mixed() {
            return typo().map(|reading| AltList {
                reading: Some(reading),
                req_id: None,
                candidates: Vec::new(),
                main: Vec::new(),
                ends_english: false,
                typed: String::new(),
                typed_early: false,
            });
        }
        let reading = self.composing.kana_reading();
        let mixed = self.composing.full_reading();
        let other = reading != mixed && !reading.chars().any(|c| c.is_ascii_alphabetic());
        Some(AltList {
            reading: if other { Some(reading) } else { typo() },
            req_id: None,
            candidates: Vec::new(),
            main: Vec::new(),
            ends_english: self.composing.ends_in_english(),
            typed: self.composing.romaji(),
            typed_early: self.composing.is_known_word(),
        })
    }

    /// The list to show for the engine's list `main` of the reading open:
    /// with mixed input's other candidates, and the rewrite first.
    fn shown_list(&mut self, main: Vec<String>) -> Vec<String> {
        let mut list = self.mixed_list(main);
        if let Some(text) = self.rewrite_shown() {
            list.retain(|c| *c != text);
            list.insert(0, text);
        }
        list
    }

    /// The rewrite of the reading typed, once it is in.
    fn rewrite_shown(&self) -> Option<String> {
        self.rewrite
            .as_ref()
            .filter(|j| j.reading == self.composing.hiragana())
            .and_then(|j| j.text.clone())
    }

    /// The list to show for the engine's list `main` of the mixed reading:
    /// with the other reading's candidates and the input as typed (see
    /// `AltList`). Unchanged without one.
    fn mixed_list(&mut self, main: Vec<String>) -> Vec<String> {
        let Some(alt) = self.alt.as_mut() else {
            return main;
        };
        alt.main = main.clone();
        let other = &alt.candidates[..alt.candidates.len().min(ALT_MAX)];
        let (other_early, other_late) = other.split_at(other.len().min(ALT_EARLY));
        let mut list: Vec<String> = Vec::new();
        if alt.ends_english {
            list.extend(main.first().map(|top| format!("{top} ")));
            list.extend(other_early.iter().cloned());
            list.extend(main);
        } else {
            let (early, late) = main.split_at(main.len().min(ALT_EARLY));
            list.extend(early.iter().cloned());
            list.extend(other_early.iter().cloned());
            list.extend(late.iter().cloned());
        }
        list.extend(other_late.iter().cloned());
        if alt.typed_early {
            list.insert(list.len().min(2), alt.typed.clone());
        } else {
            list.push(alt.typed.clone());
        }
        let mut seen = std::collections::HashSet::new();
        list.retain(|c| !c.is_empty() && seen.insert(c.clone()));
        list
    }

    /// Ask the engine for the other reading's list, once the mixed one is in.
    fn request_alt(&mut self) {
        let Some(alt) = self.alt.as_mut() else {
            return;
        };
        let Some(reading) = &alt.reading else {
            return;
        };
        if alt.req_id.is_some() {
            return;
        }
        if let Some(client) = &self.client {
            let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
            client.dispatch(req_id, &self.context, reading);
            alt.req_id = Some(req_id);
        }
    }

    /// The other reading's list arrived: add it to the open list, the
    /// selection staying on its candidate.
    fn apply_alt_result(&mut self, result: EngineResult) -> Vec<Action> {
        let Some(alt) = self.alt.as_ref() else {
            return Vec::new();
        };
        let Some(reading) = alt.reading.clone().filter(|r| *r == result.reading) else {
            return Vec::new();
        };
        let main = alt.main.clone();
        let candidates = self.finish_list(&reading, result.candidates, false);
        if let Some(alt) = self.alt.as_mut() {
            alt.candidates = candidates;
        }
        let shown = match &self.phase {
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => candidates.get(*selected).cloned(),
            _ => return Vec::new(),
        };
        let list = self.shown_list(main);
        let sel = shown
            .and_then(|t| list.iter().position(|c| *c == t))
            .unwrap_or(0);
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        vec![
            Action::UpdatePreedit {
                text: self.with_pending(list[sel].clone()),
            },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
        ]
    }

    /// Shift+letter: enter (or extend) the temporary `Alphabet` phase.
    /// Commits any pending hiragana/kanji composition first so the raw
    /// ASCII buffer is cleanly separated from what came before.
    pub fn handle_shifted_alpha(&mut self, c: char) -> KeyOutcome {
        if !c.is_ascii_alphabetic() {
            return not_consumed();
        }
        // kana input: Shift+letter is a kana key (Shift+E: ぃ)
        if crate::romaji_table::active().is_some_and(|t| t.kana_keys) {
            return self.handle_char(c);
        }
        // Mixed input: the letter joins the composition, its case marking
        // it English.
        if self.composing.is_mixed() {
            return self.handle_char(c);
        }
        let mut actions = match &self.phase {
            Phase::Empty => Vec::new(),
            Phase::Composing { .. } => {
                let outcome = self.handle_enter();
                outcome.actions
            }
            Phase::Converting { .. } => {
                let mut a = self.commit_converting();
                a.push(Action::HideCandidates);
                a
            }
            Phase::Alphabet { buffer } => {
                // Already in alphabet mode — just append.
                let mut buf = buffer.clone();
                buf.push(c);
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                return KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                };
            }
        };
        let buffer = c.to_string();
        let preedit = buffer.clone();
        self.phase = Phase::Alphabet { buffer };
        actions.push(Action::UpdatePreedit { text: preedit });
        KeyOutcome {
            consumed: true,
            actions,
        }
    }

    /// Whether `c` goes to the app as part of a command's name being typed
    /// (`pass_through_commands`).
    fn passes(&mut self, c: char) -> bool {
        if !self.passing || !matches!(self.phase, Phase::Empty) {
            self.passing = false;
            return false;
        }
        self.passing = c.is_ascii_alphanumeric() || "-_:./".contains(c);
        self.passing
    }

    pub fn handle_char(&mut self, c: char) -> KeyOutcome {
        if self.passes(c) {
            return not_consumed();
        }
        // a key the table reads, or a kana typed by a key of its own
        let table_key =
            crate::romaji_table::active().is_some_and(|t| t.uses(c) || ('ぁ'..='ゖ').contains(&c));
        if !c.is_ascii_alphabetic() && !c.is_ascii_digit() && c != '-' && !table_key {
            return not_consumed();
        }
        // In Alphabet phase, letters / digits / hyphen append verbatim
        // (case preserved) to the buffer.
        if let Phase::Alphabet { buffer } = &self.phase {
            let mut buf = buffer.clone();
            buf.push(c);
            let preedit = buf.clone();
            self.phase = Phase::Alphabet { buffer: buf };
            return KeyOutcome {
                consumed: true,
                actions: vec![Action::UpdatePreedit { text: preedit }],
            };
        }
        if self.digit_select && c.is_ascii_digit() && self.window_open() {
            return self.pick_by_digit(c);
        }
        self.type_char(c)
    }

    /// `c` typed into the composition, committing a conversion first.
    fn type_char(&mut self, c: char) -> KeyOutcome {
        if matches!(self.phase, Phase::Empty | Phase::Converting { .. }) {
            self.invalidate_engine_results();
        }
        match &self.phase {
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.composing.reset();
                self.composing.input_char(c);
                self.phase = Phase::Composing {
                    candidates: Vec::new(),
                };
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            _ => {
                self.composing.input_char(c);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_backspace(&mut self) -> KeyOutcome {
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.pop();
                if buf.is_empty() {
                    self.phase = Phase::Empty;
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                self.back_to_composing();
                let preedit = self.live_preedit();
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. } => {
                self.composing.delete_left();
                if self.composing.is_empty() {
                    self.phase = Phase::Empty;
                    self.last_applied_reading.clear();
                    self.cached_prefix_reading.clear();
                    self.cached_prefix_kanji.clear();
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_space(&mut self) -> KeyOutcome {
        // a command's name ends
        self.passing = false;
        if self.form_shown().is_some() {
            // Space after F6–F10 converts the reading as usual.
            self.back_to_composing();
        }
        // Mixed input: Space after an English word, or a lone `i` (`i
        // think`), types a space (`Hello world`); Space again, or after
        // Japanese, converts.
        if self.composing.is_mixed()
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.cursor_at_end()
        {
            if self.composing.ends_with_space() {
                self.composing.trim_end_spaces();
            } else if self.composing.ends_in_english() || self.composing.ends_in_lone_i() {
                return self.type_char(' ');
            }
        }
        let mut previous_dates = Vec::new();
        if matches!(self.phase, Phase::Composing { .. }) {
            self.alt = self.alt_for_space();
            if self.date_time_enabled {
                previous_dates = self
                    .char_rules
                    .apply(self.date_time_candidates(&self.composing.full_reading()));
                // Each explicit conversion gets a fresh clock; its later
                // model/rerank/segment results reuse this exact instant.
                self.date_time_now = Some(Local::now());
            }
        }
        match &self.phase {
            Phase::Empty if self.full_width_space => self.commit_space(),
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.push(' ');
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => self.handle_down(),
            Phase::Composing { .. } => {
                // Async model: the live path already dispatches on each
                // keystroke for longer readings. Space is an explicit
                // conversion request, so it must dispatch even for 1-2
                // character readings; otherwise dictionary-backed short
                // conversions never reach the daemon and the UI is stuck
                // with a single provisional candidate.
                let candidates = if self.force_convert_on_space {
                    self.dispatch_live_forced()
                } else {
                    // typing is done for this conversion, as in the forced
                    // dispatch (a pending n is ん)
                    self.composing.commit_reading();
                    self.dispatch_live()
                };
                // the engine has the list of this reading already
                let reading = self.composing.hiragana().to_string();
                let ready = reading == self.list_reading;
                let dynamic = self.date_time_enabled && crate::date_time::is_reading(&reading);
                // A prefix list is not a conversion of the frozen whole reading.
                let candidates = if ready && dynamic {
                    let normal = candidates
                        .into_iter()
                        .filter(|c| !previous_dates.contains(c))
                        .collect();
                    crate::date_time::insert(
                        &reading,
                        normal,
                        self.char_rules.apply(self.date_time_candidates(&reading)),
                    )
                } else if ready {
                    candidates
                } else {
                    Vec::new()
                };
                if candidates.is_empty() {
                    let provisional = self.provisional_conversion();
                    if provisional.is_empty() {
                        return self.handle_enter();
                    }
                    // This also works without a model or while it loads.
                    let list = if dynamic {
                        self.finish_list(&reading, vec![reading.clone()], true)
                    } else {
                        vec![provisional]
                    };
                    self.list_starts = self.list_starts_of(&reading, &[], &[], &list);
                    self.list_reading = if dynamic {
                        reading.clone()
                    } else {
                        String::new()
                    };
                    let source = list[0].clone();
                    let list = self.shown_list(list);
                    let first = self.with_pending(list[0].clone());
                    self.phase = Phase::Converting {
                        candidates: list.clone(),
                        selected: 0,
                        seg: None,
                    };
                    let mut actions = vec![
                        Action::UpdatePreedit { text: first },
                        Action::ShowCandidates { list, selected: 0 },
                    ];
                    actions.extend(self.start_rewrite(&reading, &source));
                    return consumed(actions);
                }
                if dynamic && self.cached_prefix_reading == reading {
                    self.cached_prefix_kanji = candidates.first().cloned().unwrap_or_default();
                }
                // Preserve the model's alignments while adding starts for
                // fresh local candidates (which each cover the whole reading).
                for (text, starts) in self.list_starts_of(&reading, &[], &[], &candidates) {
                    self.list_starts.entry(text).or_insert(starts);
                }
                let source = candidates[0].clone();
                let candidates = self.shown_list(candidates);
                if ready {
                    self.request_alt();
                }
                let first = self.with_pending(candidates.first().cloned().unwrap_or_default());
                self.phase = Phase::Converting {
                    candidates: candidates.clone(),
                    selected: 0,
                    seg: None,
                };
                // the conversion key: a rerank waiting for the typing pause goes now
                self.rerank_now();
                let mut actions = vec![
                    Action::UpdatePreedit { text: first },
                    Action::ShowCandidates {
                        list: candidates,
                        selected: 0,
                    },
                ];
                actions.extend(self.start_rewrite(&reading, &source));
                consumed(actions)
            }
        }
    }

    pub fn handle_enter(&mut self) -> KeyOutcome {
        self.passing = false;
        self.invalidate_engine_results();
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let text = buffer.clone();
                self.phase = Phase::Empty;
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            Phase::Composing { candidates } => {
                // with the caret moved the preedit is the kana, and so is
                // the commit; so it is when mixed input reads the whole
                // otherwise than what was typed so far (`ok`: OK, not お
                // and k)
                let done_otherwise = self.composing.reads_otherwise_when_done();
                let reading = self.composing.full_reading();
                let cand_top = candidates
                    .first()
                    .filter(|_| {
                        self.live_conversion
                            && self.composing.cursor_at_end()
                            && self.list_reading == self.composing.live_reading()
                            && !done_otherwise
                    })
                    .cloned();
                let text = match cand_top {
                    _ if done_otherwise => self.composing.commit(),
                    Some(t) if !t.is_empty() => format!("{t}{}", self.composing.commit_live_tail()),
                    _ if self.live_conversion && self.composing.cursor_at_end() => {
                        let text = self.provisional_conversion();
                        format!("{text}{}", self.composing.commit_live_tail())
                    }
                    _ => self.composing.commit(),
                };
                self.stop_rerank();
                self.stop_rewrite();
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.record_commit(&reading, &text);
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
        }
    }

    pub fn handle_escape(&mut self) -> KeyOutcome {
        self.passing = false;
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { .. } => {
                self.phase = Phase::Empty;
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::EndComposition],
                }
            }
            Phase::Converting { .. } => {
                self.back_to_composing();
                let preedit = self.live_preedit();
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. } => {
                self.invalidate_engine_results();
                self.composing.reset();
                self.stop_rerank();
                self.stop_rewrite();
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::EndComposition],
                }
            }
        }
    }

    pub fn handle_move_cursor(&mut self, delta: i32) -> KeyOutcome {
        match &self.phase {
            Phase::Composing { .. } => {
                self.composing.move_cursor(delta);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { seg, .. } => {
                // The first ← / → switches to segment editing on the first
                // segment; later ones move the focus.
                let entering = seg.is_none();
                if !self.ensure_segmented() {
                    return consumed(Vec::new());
                }
                if !entering {
                    if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
                        s.move_focus(delta);
                    }
                }
                self.query_focused_segment();
                consumed(self.segment_actions())
            }
            _ => not_consumed(),
        }
    }

    /// Shift+← / Shift+→: resize the focused segment by one reading char
    /// (moving the cursor while composing).
    pub fn handle_resize(&mut self, delta: i32) -> KeyOutcome {
        match &self.phase {
            Phase::Composing { .. } => self.handle_move_cursor(delta),
            Phase::Converting { .. } => {
                if !self.ensure_segmented() {
                    return consumed(Vec::new());
                }
                let changed = match &mut self.phase {
                    Phase::Converting { seg: Some(s), .. } => s.resize(delta),
                    _ => false,
                };
                if changed {
                    // Results for the old segments no longer fit.
                    self.seg_requests.clear();
                    self.query_next_stale();
                }
                consumed(self.segment_actions())
            }
            _ => not_consumed(),
        }
    }

    /// Switch the Converting phase to segment editing unless it is already;
    /// false when the conversion cannot be segmented.
    fn ensure_segmented(&mut self) -> bool {
        let date_now = self.date_time_now.unwrap_or_else(Local::now);
        let date_time_enabled = self.date_time_enabled;
        let char_rules = self.char_rules.clone();
        let Phase::Converting {
            candidates,
            selected,
            seg,
        } = &mut self.phase
        else {
            return false;
        };
        if seg.is_some() {
            return true;
        }
        if self.list_reading.is_empty() {
            return false;
        }
        // a segment's own conversions take over from the whole sentence's rerank
        if self.rerank.take().is_some() {
            if let Some(r) = &self.reranker {
                r.cancel();
            }
        }
        *seg = Segmented::build(&self.list_reading, candidates, *selected, &self.list_starts);
        if let Some(s) = seg.as_mut() {
            // each segment offers its own hiragana / katakana, as a whole
            // conversion does
            for k in 0..s.segs.len() {
                let reading = s.reading_of(k);
                let cands = std::mem::take(&mut s.segs[k].cands);
                let cands = crate::kana_candidates::arrange(&reading, cands);
                let generated = if date_time_enabled {
                    char_rules.apply(crate::date_time::candidates(&reading, &date_now))
                } else {
                    Vec::new()
                };
                let chosen = cands.first().cloned();
                s.segs[k].cands = crate::date_time::insert(&reading, cands, generated);
                if let Some(chosen) = chosen {
                    s.segs[k].selected = s.segs[k]
                        .cands
                        .iter()
                        .position(|c| *c == chosen)
                        .unwrap_or(0);
                }
            }
        }
        self.seg_requests.clear();
        seg.is_some()
    }

    fn segmented(&self) -> Option<&Segmented> {
        match &self.phase {
            Phase::Converting { seg: Some(s), .. } => Some(s),
            _ => None,
        }
    }

    /// The preedit and candidate window of segment editing.
    fn segment_actions(&self) -> Vec<Action> {
        let Some(s) = self.segmented() else {
            return Vec::new();
        };
        let focused = &s.segs[s.focus];
        let mut texts = s.texts();
        if let Some(last) = texts.last_mut() {
            *last = self.with_pending(std::mem::take(last));
        }
        vec![
            Action::UpdateSegments {
                segments: texts,
                focus: s.focus,
            },
            Action::ShowSegmentCandidates {
                list: focused.cands.clone(),
                selected: focused.selected,
                at: s.focus_utf16_offset(),
            },
        ]
    }

    /// Convert segment `k`'s own reading, with the segments before it as
    /// left context; with `more` > 0, ask for more candidates of that
    /// conversion (the engine's `continue_beam`).
    fn query_segment(&mut self, k: usize, more: u32) {
        // Serialize segment jobs: the client's pending slot is replaceable.
        if !self.seg_requests.is_empty() {
            return;
        }
        let Some(s) = self.segmented() else {
            return;
        };
        let reading = s.reading_of(k);
        let context = format!("{}{}", self.context, s.text_before(k));
        if let Some(client) = &self.client {
            let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if more == 0 {
                client.dispatch(req_id, &context, &reading);
            } else {
                client.dispatch_more(req_id, &context, &reading, more);
            }
            self.seg_requests.push((req_id, k, more));
        }
    }

    /// Past the focused segment's last candidate: ask for more of them
    /// instead of going round to the first, and move to the first new one
    /// when they arrive. False when the segment has no more to give (its
    /// last continuation brought nothing new, or it is waiting for its
    /// first conversion after a resize).
    fn continue_focused(&mut self) -> bool {
        let Some(s) = self.segmented() else {
            return false;
        };
        let k = s.focus;
        let seg = &s.segs[k];
        if seg.exhausted || seg.stale || self.client.is_none() {
            return false;
        }
        let in_flight = !self.seg_requests.is_empty();
        let more = (seg.queried && !in_flight).then_some(seg.more + 1);
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            s.segs[k].advance = true;
            if let Some(m) = more {
                s.segs[k].more = m;
            }
        }
        if !in_flight {
            // its own conversion first, if it has not had one
            self.query_segment(k, more.unwrap_or(0));
        }
        true
    }

    /// Convert the focused segment unless that was done or is under way.
    fn query_focused_segment(&mut self) {
        let Some(s) = self.segmented() else {
            return;
        };
        let k = s.focus;
        if s.segs[k].advance && s.segs[k].queried {
            self.continue_focused();
            return;
        }
        if !s.segs[k].queried && self.seg_requests.is_empty() {
            self.query_segment(k, 0);
        }
    }

    fn query_next_stale(&mut self) {
        if let Some(k) = self.segmented().and_then(|s| s.next_stale()) {
            if !self.seg_requests.iter().any(|&(_, i, _)| i == k) {
                self.query_segment(k, 0);
            }
        }
    }

    /// A result while editing segments: the conversion of one of them (put
    /// into its list), or a stale whole-sentence result (dropped). `None`
    /// when not editing segments.
    fn apply_segment_result(&mut self, result: &EngineResult) -> Option<Vec<Action>> {
        self.segmented()?;
        let Some(at) = self
            .seg_requests
            .iter()
            .position(|&(id, _, _)| id == result.req_id)
        else {
            return Some(Vec::new());
        };
        let (_, k, more) = self.seg_requests.remove(at);
        let reading = self.segmented()?.reading_of(k);
        if reading != result.reading {
            return Some(Vec::new());
        }
        let mut list = self.finish_list(&reading, result.candidates.clone(), false);
        let dates = self.char_rules.apply(self.date_time_candidates(&reading));
        if let Some(s) = self.segmented() {
            let chosen = s.text_of(k);
            let stale = s.segs[k].stale;
            // a resized segment has no chosen text yet: judge by the list's top
            let chosen = if stale {
                list.first().cloned().unwrap_or(chosen)
            } else {
                chosen
            };
            list.retain(|c| dates.contains(c) || crate::segments::fits(&reading, &chosen, c));
        }
        let mut again = false;
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            let before = s.segs[k].cands.len();
            s.take(k, list);
            let seg = &mut s.segs[k];
            let added = seg.cands.len() > before;
            if more > 0 && !added {
                seg.exhausted = true;
            }
            if seg.advance {
                if added && s.focus == k {
                    seg.selected = before;
                    seg.advance = false;
                } else if s.focus == k && !seg.exhausted && more == 0 {
                    // its own conversion added nothing: ask for more
                    again = true;
                } else {
                    // nothing more: round to the first, as without continuing
                    if s.focus == k && seg.exhausted {
                        seg.selected = 0;
                    }
                    seg.advance = false;
                }
            }
        }
        if again {
            self.continue_focused();
        }
        self.query_next_stale();
        self.query_focused_segment();
        Some(self.segment_actions())
    }

    /// Rerank the whole-reading conversion just shown: its first candidates
    /// once typing pauses (`now`: at once, the candidate window being open),
    /// then more while the user stays idle. The same conversion again only
    /// hurries a request still waiting.
    fn start_rerank(
        &mut self,
        reading: &str,
        raw: &[String],
        scores: &[f32],
        reading_pos: &[Vec<u32>],
        now: bool,
    ) {
        if self.reranker.is_none() || raw.len() < 2 || scores.len() != raw.len() {
            return;
        }
        if let Some(job) = &self.rerank {
            if job.reading == reading && job.raw == raw {
                if now {
                    self.rerank_now();
                }
                return;
            }
        }
        let max = self.rerank_max(raw.len());
        let k = self
            .rerank_cfg
            .first_candidates
            .clamp(2, max.max(2))
            .min(max);
        let due = if now {
            Instant::now()
        } else {
            Instant::now() + Duration::from_millis(self.rerank_cfg.idle_ms)
        };
        let spelled = {
            use unicode_normalization::UnicodeNormalization;
            let r: Vec<char> = reading.nfkc().collect();
            raw.iter()
                .map(|c| crate::reading_align::spells_out(c, &r))
                .collect()
        };
        self.rerank = Some(RerankJob {
            reading: reading.to_string(),
            context: self.context.clone(),
            raw: raw.to_vec(),
            scores: scores.to_vec(),
            spelled,
            reading_pos: reading_pos.to_vec(),
            k,
            req_id: 0,
            due,
        });
        self.send_rerank(due);
    }

    /// Candidates of a list of `n` reranked at most.
    fn rerank_max(&self, n: usize) -> usize {
        let info = self.reranker.as_ref().and_then(|r| r.info());
        let mut limit = self
            .rerank_cfg
            .max_candidates
            .or(info.map(|i| i.max_candidates))
            .unwrap_or(20);
        if let Some(i) = info {
            limit = limit.min(i.max_candidates);
        }
        limit.min(n)
    }

    fn rerank_lambda(&self) -> f32 {
        self.rerank_cfg
            .lambda
            .or(self
                .reranker
                .as_ref()
                .and_then(|r| r.info())
                .map(|i| i.lambda))
            .unwrap_or(0.3)
    }

    fn send_rerank(&mut self, due: Instant) {
        let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let (Some(client), Some(job)) = (self.reranker.as_ref(), self.rerank.as_mut()) else {
            return;
        };
        job.req_id = req_id;
        job.due = due;
        client.dispatch(req_id, &job.context, &job.raw[..job.k], due);
    }

    /// The conversion key: a rerank waiting for the typing pause goes now.
    fn rerank_now(&mut self) {
        if self.rerank.as_ref().is_some_and(|j| j.due > Instant::now()) {
            self.send_rerank(Instant::now());
        }
    }

    /// The conversion being reranked is gone (typed over, committed,
    /// dropped): no more requests for it.
    fn stop_rerank(&mut self) {
        if self.rerank.take().is_some() {
            if let Some(r) = &self.reranker {
                r.cancel();
            }
        }
    }

    /// Rewrite mode, a conversion just opened: ask for the rewrite of
    /// `reading`'s conversion `text` (once per reading) and say so.
    fn start_rewrite(&mut self, reading: &str, text: &str) -> Option<Action> {
        // nothing sensitive leaves the machine: no rewrite of a token, a
        // URL or a password, and no context that holds one
        if !self.rewrite_mode
            || self.private_field
            || text.trim().is_empty()
            || crate::privacy::is_sensitive(reading)
            || crate::privacy::is_sensitive(text)
        {
            return None;
        }
        if let Some(job) = self.rewrite.as_ref().filter(|j| j.reading == reading) {
            // asked already: in, or still on its way
            return job.text.is_none().then(rewriting);
        }
        let client = self.rewriter.as_ref()?;
        let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let context = if crate::privacy::is_sensitive(&self.raw_context) {
            ""
        } else {
            &self.raw_context
        };
        client.dispatch(req_id, context, reading, text.trim_end());
        self.rewrite = Some(RewriteJob {
            req_id,
            reading: reading.to_string(),
            text: None,
        });
        Some(rewriting())
    }

    /// The conversion is committed or dropped: no rewrite for it.
    fn stop_rewrite(&mut self) {
        if self.rewrite.take().is_some() {
            if let Some(r) = &self.rewriter {
                r.cancel();
            }
        }
    }

    /// A rewrite came back: first in the list open for its reading,
    /// selected unless the user moved off the top; or why it failed.
    fn apply_rewrite_result(&mut self) -> Vec<Action> {
        let Some(result) = self.rewriter.as_ref().and_then(|r| r.take_latest()) else {
            return Vec::new();
        };
        let Some(reading) = self
            .rewrite
            .as_ref()
            .filter(|j| j.req_id == result.req_id)
            .map(|j| j.reading.clone())
        else {
            return Vec::new();
        };
        let open = self.composing.hiragana() == reading
            && self.form_shown().is_none()
            && matches!(self.phase, Phase::Converting { seg: None, .. });
        let Some(text) = result.text.filter(|t| !t.is_empty()) else {
            self.rewrite = None;
            let code = result.code.unwrap_or_default();
            return if open {
                vec![Action::Status {
                    text: crate::rewrite_client::failure_message(&code).to_string(),
                }]
            } else {
                Vec::new()
            };
        };
        if let Some(job) = self.rewrite.as_mut() {
            job.text = Some(text.clone());
        }
        crate::diagnostics::log(format!("rewrite applied ms={:.0}", result.ms));
        if !open {
            return Vec::new();
        }
        // segment editing cuts the rewrite too, as best the kana allow
        if let Some(starts) = crate::segments::char_starts(&reading, &text, None) {
            self.list_starts.insert(text.clone(), starts);
        }
        let Phase::Converting {
            candidates,
            selected,
            ..
        } = &self.phase
        else {
            return Vec::new();
        };
        let shown = candidates.get(*selected).cloned();
        let at_top = *selected == 0;
        let mut list = candidates.clone();
        list.retain(|c| *c != text);
        list.insert(0, text);
        let sel = if at_top {
            0
        } else {
            shown
                .and_then(|t| list.iter().position(|c| *c == t))
                .unwrap_or(0)
        };
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        vec![
            Action::UpdatePreedit {
                text: self.with_pending(list[sel].clone()),
            },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
            Action::Status {
                text: String::new(),
            },
        ]
    }

    /// A rerank result: the scored candidates reordered by `converter +
    /// lambda * LM` (the rest kept after them), the list rebuilt and shown,
    /// then more candidates asked for while the user stays idle.
    fn apply_rerank_result(&mut self) -> Vec<Action> {
        let Some(result) = self.reranker.as_ref().and_then(|r| r.take_latest()) else {
            return Vec::new();
        };
        let Some(job) = self.rerank.as_ref() else {
            return Vec::new();
        };
        if result.req_id != job.req_id || result.logp.len() != job.k {
            return Vec::new();
        }
        let lambda = f64::from(self.rerank_lambda());
        let order = rerank_order(
            &job.scores[..job.k],
            &result.logp,
            self.rerank_lambda(),
            &job.spelled[..job.k],
        );
        // the scored candidates far below the best go (`RERANK_GAP`)
        let total = |i: usize| f64::from(job.scores[i]) + lambda * result.logp[i];
        let best = total(order[0]);
        let order: Vec<usize> = order
            .into_iter()
            .filter(|&i| best - total(i) <= RERANK_GAP)
            .collect();
        let reorder = |v: &[String]| -> Vec<String> {
            order
                .iter()
                .map(|&i| v[i].clone())
                .chain(v[job.k..].iter().cloned())
                .collect()
        };
        let raw = reorder(&job.raw);
        let reading_pos: Vec<Vec<u32>> = if job.reading_pos.len() == job.raw.len() {
            order
                .iter()
                .map(|&i| job.reading_pos[i].clone())
                .chain(job.reading_pos[job.k..].iter().cloned())
                .collect()
        } else {
            Vec::new()
        };
        let reading = job.reading.clone();
        let (k, n) = (job.k, job.raw.len());
        crate::diagnostics::log(format!(
            "rerank applied k={k} ms={:.1} top={:?} (was {:?})",
            result.score_ms, raw[0], job.raw[0]
        ));
        let list = self.finish_list(&reading, raw.clone(), true);
        let starts = self.list_starts_of(&reading, &raw, &reading_pos, &list);
        let actions = self.show_reranked(&reading, list, starts);
        let max = self.rerank_max(n);
        if k < max {
            if let Some(job) = self.rerank.as_mut() {
                job.k = (k * 2).min(max);
            }
            self.send_rerank(Instant::now());
        }
        actions
    }

    /// Show a reranked list of `reading` where its conversion is shown: the
    /// typing preedit, or the open window (the selection kept on the
    /// candidate it was on, unless that was the top).
    fn show_reranked(
        &mut self,
        reading: &str,
        list: Vec<String>,
        starts: HashMap<String, Vec<usize>>,
    ) -> Vec<Action> {
        if self.list_reading != reading || list.is_empty() || self.form_shown().is_some() {
            return Vec::new();
        }
        match &mut self.phase {
            Phase::Composing { candidates } if self.composing.live_reading() == reading => {
                *candidates = list.clone();
            }
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => {
                let shown = candidates.get(*selected).cloned();
                let at_top = *selected == 0;
                let list = self.shown_list(list);
                let sel = if at_top {
                    0
                } else {
                    shown
                        .and_then(|t| list.iter().position(|c| *c == t))
                        .unwrap_or(0)
                };
                if let Phase::Converting {
                    candidates,
                    selected,
                    ..
                } = &mut self.phase
                {
                    *candidates = list.clone();
                    *selected = sel;
                }
                self.list_starts = starts;
                return vec![
                    Action::UpdatePreedit {
                        text: self.with_pending(list[sel].clone()),
                    },
                    Action::ShowCandidates {
                        list,
                        selected: sel,
                    },
                ];
            }
            _ => return Vec::new(),
        }
        self.list_starts = starts;
        // the reranked top is what the deferred preedit waited for
        if self.deferred.as_ref().is_some_and(|d| d.reading == reading) {
            self.deferred = None;
            self.cached_prefix_reading = reading.to_string();
        }
        if self.cached_prefix_reading == reading {
            self.cached_prefix_kanji = list[0].clone();
        }
        vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]
    }

    /// The engine's candidates for `reading` as the IME lists them:
    /// post-processing, kana forms, symbols, char rules and (for a whole
    /// reading) the learned reranking.
    fn finish_list(&mut self, reading: &str, raw: Vec<String>, rerank: bool) -> Vec<String> {
        let processed = crate::postprocess::apply(reading, raw, self.symbol_width, &self.context);
        let processed = if self.drop_untyped_symbols {
            crate::postprocess::drop_untyped_symbols(reading, processed)
        } else {
            processed
        };
        let processed = crate::postprocess::drop_changed_digits(reading, processed);
        let processed = crate::postprocess::drop_script_flips(processed);
        // The English words of a mixed reading stay as typed.
        let processed = if self.composing.is_mixed() {
            processed
                .into_iter()
                .filter(|c| crate::mixed::keeps_latin(reading, c))
                .collect()
        } else {
            processed
        };
        let mut arranged = crate::kana_candidates::arrange(reading, processed);
        if let Some(dict) = &self.symbol_dict {
            arranged = dict.insert(reading, arranged);
        }
        if self.date_time_enabled && crate::date_time::is_reading(reading) {
            self.date_time_now.get_or_insert_with(Local::now);
            arranged =
                crate::date_time::insert(reading, arranged, self.date_time_candidates(reading));
        }
        if self.numbers_enabled {
            arranged =
                crate::numbers::insert(reading, arranged, crate::numbers::candidates(reading));
        }
        let mut list = self.char_rules.apply(arranged);
        // after the symbol widths: a kaomoji stays as it is drawn
        if self.emoji_enabled {
            list = crate::emoji::insert(reading, list);
        }
        if let Some(dict) = self.user_dict() {
            list = dict.apply(reading, list);
        }
        if rerank {
            if let Some(learning) = self.learning.as_mut() {
                list = learning.rerank(reading, list);
            }
        }
        if let Some(hidden) = self.hidden.as_mut() {
            list = hidden.filter(reading, list);
        }
        list
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

    /// Reading starts of the chars of each candidate in `list`: from the
    /// engine's alignment of the raw candidate it was made from (when the
    /// post-processing kept its length), else from the kana anchors.
    fn list_starts_of(
        &self,
        reading: &str,
        raw: &[String],
        reading_pos: &[Vec<u32>],
        list: &[String],
    ) -> HashMap<String, Vec<usize>> {
        let mut out: HashMap<String, Vec<usize>> = HashMap::new();
        for (r, pos) in raw.iter().zip(reading_pos) {
            let processed = crate::postprocess::apply(
                reading,
                vec![r.clone()],
                self.symbol_width,
                &self.context,
            );
            let processed = if self.drop_untyped_symbols {
                crate::postprocess::drop_untyped_symbols(reading, processed)
            } else {
                processed
            };
            let Some(one) = processed.first() else {
                continue;
            };
            let shown = self.char_rules.apply_one(one);
            if out.contains_key(&shown)
                || !list.contains(&shown)
                || shown.chars().count() != r.chars().count()
            {
                continue;
            }
            if let Some(starts) = crate::segments::char_starts(reading, &shown, Some(pos)) {
                out.insert(shown, starts);
            }
        }
        for c in self.char_rules.apply(self.date_time_candidates(reading)) {
            if list.contains(&c) {
                out.insert(c.clone(), vec![0; c.chars().count()]);
            }
        }
        for c in list {
            if !out.contains_key(c) {
                if let Some(starts) = crate::segments::char_starts(reading, c, None) {
                    out.insert(c.clone(), starts);
                }
            }
        }
        out
    }

    pub fn handle_up(&mut self) -> KeyOutcome {
        self.move_selection(-1)
    }
    /// ↓ steps down the list; while typing it opens the conversion, as
    /// Tab does.
    pub fn handle_down(&mut self) -> KeyOutcome {
        if matches!(self.phase, Phase::Composing { .. }) {
            return self.handle_space();
        }
        self.move_selection(1)
    }
    pub fn handle_page_up(&mut self) -> KeyOutcome {
        self.move_page(-1)
    }
    pub fn handle_page_down(&mut self) -> KeyOutcome {
        self.move_page(1)
    }

    /// Shift+Space: Space while composing; with nothing composed, the space
    /// width plain Space does not type.
    pub fn handle_shift_space(&mut self) -> KeyOutcome {
        match self.phase {
            Phase::Empty if !self.full_width_space => self.commit_space(),
            Phase::Empty => not_consumed(),
            _ => self.handle_space(),
        }
    }

    fn commit_space(&mut self) -> KeyOutcome {
        let text = "\u{3000}".to_string();
        self.append_context(&text);
        consumed(vec![Action::Commit { text }])
    }

    /// Tab opens the candidates like Space, then steps down (Shift+Tab up).
    pub fn handle_tab(&mut self, back: bool) -> KeyOutcome {
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { .. } => consumed(Vec::new()),
            Phase::Converting { .. } if self.form_shown().is_none() => {
                self.move_selection(if back { -1 } else { 1 })
            }
            // typing with predictions: Tab opens them (as Mozc's)
            Phase::Composing { .. } => match self.open_predictions(0) {
                Some(out) => out,
                None => self.handle_space(),
            },
            _ => self.handle_space(),
        }
    }

    /// Delete: remove the kana right of the caret.
    pub fn handle_delete(&mut self) -> KeyOutcome {
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Composing { .. } => {
                self.composing.delete_right();
                if self.composing.is_empty() {
                    return self.handle_escape();
                }
                let candidates = self.dispatch_live();
                self.phase = Phase::Composing { candidates };
                consumed(vec![Action::UpdatePreedit {
                    text: self.live_preedit(),
                }])
            }
            _ => consumed(Vec::new()),
        }
    }

    /// Home / End: the caret to the start / end of the composition, or the
    /// first / last candidate of the open list.
    pub fn handle_home_end(&mut self, end: bool) -> KeyOutcome {
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Composing { .. } => {
                self.composing.move_cursor_to_edge(end);
                let candidates = self.dispatch_live();
                self.phase = Phase::Composing { candidates };
                consumed(vec![Action::UpdatePreedit {
                    text: self.live_preedit(),
                }])
            }
            Phase::Converting { .. } if self.window_open() => {
                let Some((selected, n)) = self.selection() else {
                    return consumed(Vec::new());
                };
                let target = if end { n - 1 } else { 0 };
                self.move_selection(target as i32 - selected as i32)
            }
            _ => consumed(Vec::new()),
        }
    }

    /// F6–F10: show the whole composition in `form`. Pressing F9 / F10
    /// again cycles the case (`tokyo` → `TOKYO` → `Tokyo`). Enter commits
    /// it, Space converts the reading as usual, Escape / Backspace go back
    /// to typing.
    pub fn handle_convert_to(&mut self, form: Form) -> KeyOutcome {
        let mut actions = Vec::new();
        match &self.phase {
            Phase::Empty => return not_consumed(),
            Phase::Alphabet { buffer } => {
                // the typed letters become the composition
                let buffer = buffer.clone();
                self.composing.reset();
                for c in buffer.chars() {
                    self.composing.input_char(c);
                }
                self.phase = Phase::Composing {
                    candidates: Vec::new(),
                };
            }
            _ => {}
        }
        // every letter as romaji: a word taken for English converts too
        let reading = self.composing.kana_reading();
        if reading.is_empty() {
            return consumed(actions);
        }
        let (case, list) = match self.form_shown() {
            Some(f) => (
                if f.form == form && form.cycles_case() {
                    f.case + 1
                } else {
                    0
                },
                f.list.clone(),
            ),
            None => (0, self.current_list()),
        };
        let text = char_forms::convert(form, &reading, &self.composing.romaji(), case);
        self.seg_requests.clear();
        self.form = Some(FormShown {
            form,
            case,
            text: text.clone(),
            list,
        });
        self.phase = Phase::Converting {
            candidates: vec![text.clone()],
            selected: 0,
            seg: None,
        };
        actions.push(Action::HideCandidates);
        actions.push(Action::UpdatePreedit { text });
        consumed(actions)
    }

    /// The candidates of the whole reading the screen holds: the typing
    /// phase's, or the open list's when it is not provisional.
    fn current_list(&self) -> Vec<String> {
        match &self.phase {
            Phase::Composing { candidates } => candidates.clone(),
            Phase::Converting { candidates, .. } if !self.list_reading.is_empty() => {
                candidates.clone()
            }
            _ => Vec::new(),
        }
    }

    /// Leave Converting for typing, keeping the reading's candidates (Enter
    /// then commits the conversion the preedit shows).
    fn back_to_composing(&mut self) {
        // the predictions are no conversion of what is typed
        let predicted = self.prediction.take().is_some();
        let mut list = match self.form_shown() {
            Some(f) => f.list.clone(),
            None if predicted => Vec::new(),
            None => self.current_list(),
        };
        // typing shows (and Enter commits) the engine's conversion; the
        // rewrite stays for the next conversion of the same reading
        if let Some(text) = self.rewrite_shown() {
            list.retain(|c| *c != text);
        }
        self.form = None;
        self.seg_requests.clear();
        self.phase = Phase::Composing {
            candidates: Vec::new(),
        };
        let dispatched = self.dispatch_live();
        self.phase = Phase::Composing {
            candidates: if list.is_empty() { dispatched } else { list },
        };
    }

    /// Whether the candidate window is open (Converting, not an F6–F10
    /// form).
    fn window_open(&self) -> bool {
        self.is_converting() && self.form_shown().is_none()
    }

    /// (selected index, list length) of the open list: the focused
    /// segment's while editing segments.
    fn selection(&self) -> Option<(usize, usize)> {
        let (selected, n) = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &s.segs[s.focus];
                (f.selected, f.cands.len())
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => (*selected, candidates.len()),
            _ => return None,
        };
        (n > 0).then_some((selected, n))
    }

    /// PageUp / PageDown: the same slot on the previous / next page,
    /// wrapping around (past a segment's last page: its continuation).
    fn move_page(&mut self, dir: i32) -> KeyOutcome {
        let Some((selected, n)) = self.selection() else {
            return not_consumed();
        };
        let ps = self.page_size;
        let pages = n.div_ceil(ps) as i32;
        let page = (selected / ps) as i32;
        if dir > 0 && page + 1 >= pages && self.continue_focused() {
            return consumed(self.segment_actions());
        }
        let target_page = (page + dir).rem_euclid(pages) as usize;
        let target = (target_page * ps + selected % ps).min(n - 1);
        self.move_selection(target as i32 - selected as i32)
    }

    /// Digit `c` with the window open: pick that candidate of the page and
    /// commit it, or, editing segments, fix the focused segment's and move
    /// to the next one (committing after the last).
    fn pick_by_digit(&mut self, c: char) -> KeyOutcome {
        let slot = match c.to_digit(10) {
            Some(0) => 10,
            Some(d) => d as usize,
            None => return not_consumed(),
        };
        let Some((selected, n)) = self.selection() else {
            return consumed(Vec::new());
        };
        let index = selected / self.page_size * self.page_size + slot - 1;
        if slot > self.page_size || index >= n {
            return consumed(Vec::new());
        }
        let mut actions = self.handle_select(index).actions;
        let last_segment = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => s.focus + 1 == s.segs.len(),
            _ => true,
        };
        let next = if last_segment {
            self.handle_enter()
        } else {
            self.handle_move_cursor(1)
        };
        actions.extend(next.actions);
        consumed(actions)
    }

    /// Pick candidate `index` of the open list (a click in the candidate
    /// window): the same as moving the selection there with ↑ / ↓.
    /// While typing it picks from [`Self::suggestions`]: the conversion
    /// opens on that candidate.
    pub fn handle_select(&mut self, index: usize) -> KeyOutcome {
        if matches!(self.phase, Phase::Composing { .. }) && index < self.suggestions().len() {
            let mut actions = self.handle_space().actions;
            actions.extend(self.handle_select(index).actions);
            return consumed(actions);
        }
        let current = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => s.segs[s.focus].selected,
            Phase::Converting { selected, .. } => *selected,
            _ => return not_consumed(),
        };
        self.move_selection(index as i32 - current as i32)
    }

    fn move_selection(&mut self, delta: i32) -> KeyOutcome {
        if let Some(s) = self.segmented() {
            let focused = &s.segs[s.focus];
            // past the last candidate: more of them rather than round to the first
            if delta > 0
                && focused.selected + delta as usize >= focused.cands.len()
                && self.continue_focused()
            {
                return consumed(self.segment_actions());
            }
        }
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            let focused = &mut s.segs[s.focus];
            let n = focused.cands.len().max(1) as i32;
            focused.selected = (focused.selected as i32 + delta).rem_euclid(n) as usize;
            self.query_focused_segment();
            return consumed(self.segment_actions());
        }
        let Phase::Converting {
            candidates,
            selected,
            ..
        } = &mut self.phase
        else {
            return not_consumed();
        };
        let n = candidates.len();
        if n == 0 {
            return not_consumed();
        }
        let cur = *selected as i32;
        let new_sel = ((cur + delta).rem_euclid(n as i32)) as usize;
        *selected = new_sel;
        let text = candidates[new_sel].clone();
        let list_snap = candidates.clone();
        let text = self.with_pending(text);
        KeyOutcome {
            consumed: true,
            actions: vec![
                Action::UpdatePreedit { text },
                Action::ShowCandidates {
                    list: list_snap,
                    selected: new_sel,
                },
            ],
        }
    }

    pub fn handle_punctuation(&mut self, kana: &str) -> KeyOutcome {
        // `/command`, `$skill`, `@file` at a line's start or after a space:
        // to the app as typed, its name too
        let ascii = |c: char| match c {
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            '・' => '/',
            c => c,
        };
        let typed: Vec<char> = kana.chars().map(ascii).collect();
        if let [c] = typed[..] {
            if self.passes(c) {
                return not_consumed();
            }
            if self.pass_commands
                && matches!(self.phase, Phase::Empty)
                && self.at_word_start == Some(true)
                && matches!(c, '/' | '$' | '@')
            {
                self.passing = true;
                return not_consumed();
            }
        }
        // a key the romaji table reads (AZIK's `;`, kana input's `,`) goes
        // into the composition
        if let Some(table) = crate::romaji_table::active() {
            let ascii: Vec<char> = kana
                .chars()
                .map(|c| match c {
                    '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                    '、' => ',',
                    '。' => '.',
                    '・' => '/',
                    '「' => '[',
                    '」' => ']',
                    '￥' => '¥',
                    c => c,
                })
                .collect();
            if let [c] = ascii[..] {
                if table.uses(c) && !matches!(self.phase, Phase::Alphabet { .. }) {
                    return self.handle_char(c);
                }
            }
        }
        // Sentence punctuation right after English, in a sentence with no
        // Japanese (`mixed::english_after`): as typed, not 、。！. After the
        // English typed with Shift it goes on with it; after an English text
        // committed it is committed. Mixed input decides it per span.
        let sentence_punct = match typed[..] {
            ['。'] => Some('.'),
            ['、'] => Some(','),
            [c @ ('.' | ',' | '!' | '?' | ';' | ':')] => Some(c),
            _ => None,
        };
        if let Some(c) = sentence_punct {
            match &self.phase {
                _ if !self.sentence_case => {}
                Phase::Alphabet { buffer }
                    if crate::mixed::english_after(&format!("{}{buffer}", self.context)) =>
                {
                    let buf = format!("{buffer}{c}");
                    self.phase = Phase::Alphabet {
                        buffer: buf.clone(),
                    };
                    return consumed(vec![Action::UpdatePreedit { text: buf }]);
                }
                Phase::Empty if crate::mixed::english_after(&self.context) => {
                    let text = c.to_string();
                    self.append_context(&text);
                    return consumed(vec![Action::Commit { text }]);
                }
                _ => {}
            }
        }
        // The period key right after a digit types `.` (3.5, and 3. before
        // the 5): a 。 there is next to never meant. Mixed input shows it so
        // itself (`mixed::render_parts`).
        let kana = if kana == "。"
            && !self.composing.is_mixed()
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.digit_before_cursor()
        {
            "."
        } else {
            kana
        };
        if matches!(
            self.phase,
            Phase::Empty | Phase::Alphabet { .. } | Phase::Converting { .. }
        ) {
            self.invalidate_engine_results();
        }
        // Mixed input: the symbol goes in as typed and shows per span
        // (`example.com`, 。 after Japanese).
        if self.composing.is_mixed() && !matches!(self.phase, Phase::Alphabet { .. }) {
            // A symbol with no name in the config comes full width.
            let typed = self.mixed_symbols.get(kana).copied().or_else(|| {
                let mut chars = kana.chars();
                match (chars.next(), chars.next()) {
                    (Some(c @ '\u{FF01}'..='\u{FF5E}'), None) => {
                        char::from_u32(c as u32 - 0xFF01 + 0x21)
                    }
                    _ => None,
                }
            });
            if let Some(c) = typed {
                return self.type_char(c);
            }
        }
        // In Alphabet mode punctuation commits the buffer (same as Enter)
        // and then starts a fresh punctuation composition. Matches MS-IME:
        // typing `!` while in temp-alphabet mode ends it.
        if let Phase::Alphabet { buffer } = &self.phase {
            let text = buffer.clone();
            self.phase = Phase::Empty;
            let mut actions = Vec::new();
            if !text.is_empty() {
                self.append_context(&text);
                actions.push(Action::Commit { text });
            }
            self.composing.reset();
            self.composing.flush_romaji_and_insert(kana);
            let candidates = self.dispatch_live();
            let preedit = self.live_preedit();
            let _ = &candidates;
            self.phase = Phase::Composing { candidates };
            actions.push(Action::UpdatePreedit { text: preedit });
            return KeyOutcome {
                consumed: true,
                actions,
            };
        }
        match &self.phase {
            Phase::Alphabet { .. } => unreachable!(), // handled above
            Phase::Empty | Phase::Composing { .. } => {
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                self.composing.reset();
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                actions.push(Action::HideCandidates);
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
        }
    }

    /// Fire an async convert job for the current full hiragana. Returns
    /// the candidate list currently held by the Composing phase (possibly
    /// stale — the keystroke always renders immediately with raw hiragana
    /// and the kanji version lands 20 ms later, or 50 ms when the conversion
    /// is slower than that; see `child_daemon::PACING_STEPS`).
    fn dispatch_live(&mut self) -> Vec<String> {
        self.dispatch_live_inner(false)
    }

    /// Same as `dispatch_live`, but bypasses the short-reading OOD guard.
    /// Used only for explicit conversion requests such as Space.
    fn dispatch_live_forced(&mut self) -> Vec<String> {
        self.dispatch_live_inner(true)
    }

    fn dispatch_live_inner(&mut self, force: bool) -> Vec<String> {
        let hira = if force {
            self.composing.commit_reading()
        } else {
            self.composing.live_reading()
        };
        if hira.is_empty() {
            self.last_applied_reading.clear();
            self.cached_prefix_reading.clear();
            self.cached_prefix_kanji.clear();
            return Vec::new();
        }
        // Skip OOD partial-input conversions during live typing.
        // Clear the cache so `live_preedit` falls through to raw hira;
        // we'll resume dispatching once the reading grows past the gate.
        if !force && hira.chars().count() < self.live_min_dispatch_chars {
            self.last_applied_reading.clear();
            self.cached_prefix_reading.clear();
            self.cached_prefix_kanji.clear();
            return Vec::new();
        }
        // Keep the cache even when the user backspaces — `live_preedit`
        // handles the shrink case by trimming the cached kanji tail,
        // which preserves visual continuity until the next async result
        // lands. We only drop the cache explicitly (in commit / escape /
        // empty-composition paths) elsewhere. Space may find the list in for
        // a reading the preview does not show (a pending n read as ん): it
        // takes that list, and its rerank goes on.
        let listed = force
            && hira == self.list_reading
            && matches!(&self.phase, Phase::Composing { candidates } if !candidates.is_empty());
        if hira != self.cached_prefix_reading && !listed {
            self.stop_rerank();
            if let Some(client) = &self.client {
                let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
                client.dispatch(req_id, &self.context, &hira);
            }
        }
        if let Phase::Composing { candidates } = &self.phase {
            candidates.clone()
        } else {
            Vec::new()
        }
    }

    /// Preedit for the current composition. Three cases:
    ///
    /// * **Grow** (`hira.starts_with(cached_reading)`, user kept typing):
    ///   `cached_kanji + (hira - cached_reading) + pending`. Everything
    ///   already converted stays kanji; only the delta is raw kana.
    ///
    /// * **Shrink** (`cached_reading.starts_with(hira)`, user back-spaced):
    ///   trim trailing chars from `cached_kanji` by the delta count in
    ///   chars. This is an approximation (1 hiragana char ≠ 1 kanji char
    ///   in general), but it keeps the preedit visually continuous until
    ///   the async convert returns and fixes up the cache. Avoids the
    ///   `kanji → raw → kanji` flash that the naive "drop cache" version
    ///   caused.
    ///
    /// * **Diverge** (the end of the reading changed, mixed input reading
    ///   letters otherwise): the shrink of the cache to the reading they
    ///   share, then the rest as raw kana; raw `hira + pending` only when
    ///   they share nothing.
    ///
    /// `hira` is the kana typed, without the ん `live_reading` reads a
    /// pending n as: the n shows as typed (今日はn), and a conversion with
    /// that ん is never cached (`apply_async_result`).
    fn live_preedit(&self) -> String {
        // editing mid-composition: the kana as typed, so the caret has
        // something to sit between
        if !self.composing.cursor_at_end() || !self.live_conversion {
            return self.composing.preedit();
        }
        let hira = self.composing.hiragana();
        let pending = self.composing.pending_romaji();
        if hira.is_empty() {
            return pending.to_string();
        }
        let cached_r = &self.cached_prefix_reading;
        let cached_k = &self.cached_prefix_kanji;
        if !cached_r.is_empty() && !cached_k.is_empty() {
            let common = cached_r
                .chars()
                .zip(hira.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if common > 0 {
                let shrink_chars = cached_r.chars().count() - common;
                let kanji_chars: Vec<char> = cached_k.chars().collect();
                let keep = kanji_chars.len().saturating_sub(shrink_chars);
                let trimmed: String = kanji_chars[..keep].iter().collect();
                let tail: String = hira.chars().skip(common).collect();
                return format!("{trimmed}{tail}{pending}");
            }
        }
        // nothing converted yet: as typed (けっこn)
        format!("{hira}{pending}")
    }

    fn invalidate_engine_results(&mut self) {
        self.result_floor = self.req_counter.load(Ordering::Relaxed) + 1;
        self.date_time_now = None;
    }

    fn date_time_candidates(&self, reading: &str) -> Vec<String> {
        if !self.date_time_enabled || !crate::date_time::is_reading(reading) {
            return Vec::new();
        }
        crate::date_time::candidates(reading, &self.date_time_now.unwrap_or_else(Local::now))
    }

    /// A provisional whole reading, never a stale candidate for a different
    /// reading. Only a matching converted prefix may replace its raw kana.
    fn provisional_conversion(&self) -> String {
        let reading = self.composing.live_reading();
        if self.live_conversion
            && self.composing.cursor_at_end()
            && !self.cached_prefix_reading.is_empty()
            && reading.starts_with(&self.cached_prefix_reading)
        {
            format!(
                "{}{}",
                self.cached_prefix_kanji,
                &reading[self.cached_prefix_reading.len()..]
            )
        } else {
            reading.to_string()
        }
    }

    fn with_pending(&self, mut text: String) -> String {
        // a prediction covers what is still being typed
        if self.form_shown().is_none() && self.prediction.is_none() {
            text.push_str(self.composing.pending_romaji());
        }
        text
    }

    /// Record a commit for prediction (and as the commit before the next).
    fn record_commit(&mut self, reading: &str, text: &str) {
        if self.private_field {
            return;
        }
        if let Some(p) = self.predictor.as_ref() {
            if let Err(e) = p.borrow_mut().record(reading, text, &self.last_commit) {
                tracing::warn!("prediction record failed: {e}");
            }
        }
        self.last_commit = text.to_string();
    }

    /// Predictions for what is typed: commits whose reading starts with it
    /// (`predict`), up to `PREDICTIONS`.
    pub fn predictions(&self) -> Vec<crate::predict::Prediction> {
        let Some(p) = self.predictor.as_ref() else {
            return Vec::new();
        };
        if !matches!(self.phase, Phase::Composing { .. })
            || !self.composing.cursor_at_end()
            || self.private_field
        {
            return Vec::new();
        }
        p.borrow_mut().complete(
            &self.composing.live_reading(),
            &self.last_commit,
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
        self.deferred
            .as_ref()
            .map(|d| d.until.saturating_duration_since(Instant::now()))
    }

    /// The deferred preedit's deadline passed: the engine's top, without the
    /// reranked one.
    pub fn tick(&mut self) -> Vec<Action> {
        let Some(d) = self.deferred.take() else {
            return Vec::new();
        };
        if Instant::now() < d.until {
            self.deferred = Some(d);
            return Vec::new();
        }
        if !matches!(self.phase, Phase::Composing { .. })
            || self.composing.live_reading() != d.reading
        {
            return Vec::new();
        }
        self.cached_prefix_reading = d.reading;
        self.cached_prefix_kanji = d.top;
        vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]
    }

    /// Predict from `predictor` (`None`: no prediction).
    pub fn set_predictor(&mut self, predictor: Option<crate::predict::Predictor>) {
        self.predictor = predictor.map(std::cell::RefCell::new);
    }

    /// Open the predictions as the candidate list, `selected` chosen: Tab
    /// while typing.
    fn open_predictions(&mut self, selected: usize) -> Option<KeyOutcome> {
        let preds = self.predictions();
        if preds.is_empty() {
            return None;
        }
        let list: Vec<String> = preds.iter().map(|p| p.surface.clone()).collect();
        let selected = selected.min(list.len() - 1);
        self.stop_rerank();
        self.stop_rewrite();
        self.prediction = Some(preds);
        self.list_reading.clear();
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected,
            seg: None,
        };
        Some(consumed(vec![
            Action::UpdatePreedit {
                text: list[selected].clone(),
            },
            Action::ShowCandidates { list, selected },
        ]))
    }

    fn commit_converting(&mut self) -> Vec<Action> {
        let Phase::Converting {
            candidates,
            selected,
            seg,
        } = &self.phase
        else {
            return vec![];
        };
        // What is committed, and the choice to learn: the sentence the user
        // ended up with over the one shown first.
        let (text, learn) = match seg {
            Some(s) => {
                let text = s.text();
                let learn = (text != s.initial).then(|| (s.full_reading(), s.initial.clone()));
                (text, learn)
            }
            None => {
                let text = candidates.get(*selected).cloned().unwrap_or_default();
                // over a rewrite first, the engine's top is what was passed over
                let top = match self.rewrite_shown() {
                    Some(r) if candidates.first() == Some(&r) => 1,
                    _ => 0,
                };
                let learn = (*selected > top && !self.list_reading.is_empty())
                    .then(|| (self.list_reading.clone(), candidates[top].clone()));
                (text, learn)
            }
        };
        // a prediction: no choice to learn, but its use to record
        let prediction = self.prediction.take().map(|p| {
            let reading = p
                .iter()
                .find(|x| x.surface == text)
                .map(|x| x.reading.clone())
                .unwrap_or_default();
            (reading, text.clone())
        });
        let learn = learn.filter(|_| !self.private_field && prediction.is_none());
        if let (Some((reading, rejected)), Some(learning)) = (learn, self.learning.as_mut()) {
            if let Err(e) = learning.record(&reading, &text, &rejected) {
                tracing::warn!("learning record failed: {e}");
            }
        }
        let reading = match &prediction {
            Some((r, _)) => r.clone(),
            None if !self.list_reading.is_empty() && seg.is_none() => self.list_reading.clone(),
            None => self.composing.full_reading(),
        };
        let tail = if self.form_shown().is_none() && prediction.is_none() {
            self.composing.commit_tail()
        } else {
            // a prediction covers what was still being typed
            self.composing.reset();
            String::new()
        };
        self.record_commit(&reading, &text);
        let text = format!("{text}{tail}");
        self.invalidate_engine_results();
        self.seg_requests.clear();
        self.stop_rerank();
        self.stop_rewrite();
        self.list_reading.clear();
        self.last_applied_reading.clear();
        self.cached_prefix_reading.clear();
        self.cached_prefix_kanji.clear();
        if !text.is_empty() {
            self.append_context(&text);
            vec![Action::Commit { text }]
        } else {
            vec![Action::EndComposition]
        }
    }
}

/// The order of candidates by `score + lambda * logp`, best first; a tie
/// keeps the converter's order.
/// How far below the reranked best (the engine's score plus lambda times the
/// LM's log-probability) a scored candidate may be and stay in the list
/// (user 2026-10-23: candidates as clean as Microsoft IME's). Most of the
/// model's near-misses (でルタ, eルタ and デル多 for でるた, 続頭 for ぞくとう)
/// score far below: with sarashina2.2's SentencePiece scores at lambda 0.8,
/// over the engine's top 20 of wiki_fresh_v2 and AJIMEE 20 keeps 49% / 57%
/// of the candidates and every gold among them, 15 loses one.
const RERANK_GAP: f64 = 20.0;

/// The first candidates by the engine's score plus lambda times the LM's,
/// those that spell the reading out above those that do not (the LM lifted
/// おしま村 for おおしまむら over おおしま村; kept below, EM1 over the top 20
/// of wiki_fresh_v2 79.21 -> 79.27, AJIMEE 81.0 -> 81.5).
fn rerank_order(scores: &[f32], logp: &[f64], lambda: f32, spelled: &[bool]) -> Vec<usize> {
    let total = |i: usize| f64::from(scores[i]) + f64::from(lambda) * logp[i];
    let first = |i: usize| spelled.get(i).copied().unwrap_or(true);
    let mut order: Vec<usize> = (0..scores.len().min(logp.len())).collect();
    order.sort_by(|&a, &b| {
        first(b).cmp(&first(a)).then(
            total(b)
                .partial_cmp(&total(a))
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    order
}

/// The status while a rewrite is on its way.
fn rewriting() -> Action {
    Action::Status {
        text: "書き直し中…".to_string(),
    }
}

fn consumed(actions: Vec<Action>) -> KeyOutcome {
    KeyOutcome {
        consumed: true,
        actions,
    }
}

fn not_consumed() -> KeyOutcome {
    KeyOutcome {
        consumed: false,
        actions: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_client::EngineResult;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct MockState {
        dispatched: RefCell<Vec<(u64, String, String)>>,
        /// (request id, continue_beam) of the `dispatch_more` requests.
        continued: RefCell<Vec<(u64, u32)>>,
        next: RefCell<Option<EngineResult>>,
        /// The backend has exited.
        gone: std::cell::Cell<bool>,
    }

    /// In-memory `EngineClient` for phase-machine tests. Cloning shares the
    /// same state, so a test can keep a handle, hand a clone to the bridge,
    /// and still inspect dispatches / inject results.
    #[derive(Clone, Default)]
    struct MockClient {
        state: Rc<MockState>,
    }

    impl MockClient {
        fn set_result(&self, reading: &str, candidates: &[&str]) {
            *self.state.next.borrow_mut() = Some(EngineResult {
                req_id: self
                    .state
                    .dispatched
                    .borrow()
                    .iter()
                    .rev()
                    .find(|(_, _, r)| r == reading)
                    .map_or(1, |r| r.0),
                reading: reading.to_string(),
                candidates: candidates.iter().map(|s| s.to_string()).collect(),
                reading_pos: Vec::new(),
                ..Default::default()
            });
        }
        fn set_aligned(&self, reading: &str, candidates: &[&str], pos: &[&[u32]]) {
            *self.state.next.borrow_mut() = Some(EngineResult {
                req_id: self
                    .state
                    .dispatched
                    .borrow()
                    .iter()
                    .rev()
                    .find(|(_, _, r)| r == reading)
                    .map_or(1, |r| r.0),
                reading: reading.to_string(),
                candidates: candidates.iter().map(|s| s.to_string()).collect(),
                reading_pos: pos.iter().map(|p| p.to_vec()).collect(),
                ..Default::default()
            });
        }

        /// Answer the last dispatched request (its id and reading).
        fn answer_last(&self, candidates: &[&str]) -> (String, String) {
            let (req_id, context, reading) =
                self.state.dispatched.borrow().last().cloned().unwrap();
            *self.state.next.borrow_mut() = Some(EngineResult {
                req_id,
                reading: reading.clone(),
                candidates: candidates.iter().map(|s| s.to_string()).collect(),
                reading_pos: Vec::new(),
                ..Default::default()
            });
            (context, reading)
        }

        fn dispatched_readings(&self) -> Vec<String> {
            self.state
                .dispatched
                .borrow()
                .iter()
                .map(|(_, _, r)| r.clone())
                .collect()
        }
    }

    impl EngineClient for MockClient {
        fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
            self.state.dispatched.borrow_mut().push((
                req_id,
                context.to_string(),
                reading.to_string(),
            ));
        }
        fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
            self.dispatch(req_id, context, reading);
            self.state
                .continued
                .borrow_mut()
                .push((req_id, continue_beam));
        }
        fn take_latest(&self) -> Option<EngineResult> {
            self.state.next.borrow_mut().take()
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn is_alive(&self) -> bool {
            !self.state.gone.get()
        }
    }

    #[derive(Default)]
    struct RerankState {
        /// (request id, context, candidates, due)
        dispatched: RefCell<Vec<(u64, String, Vec<String>, Instant)>>,
        cancels: RefCell<usize>,
        next: RefCell<Option<crate::rerank_client::RerankResult>>,
    }

    #[derive(Clone, Default)]
    struct MockReranker {
        state: Rc<RerankState>,
    }

    impl MockReranker {
        /// Answer the last request with these log-probabilities.
        fn answer(&self, logp: &[f64]) {
            let req_id = self.state.dispatched.borrow().last().unwrap().0;
            *self.state.next.borrow_mut() = Some(crate::rerank_client::RerankResult {
                req_id,
                logp: logp.to_vec(),
                score_ms: 1.0,
            });
        }
        fn last(&self) -> (Vec<String>, Instant) {
            let d = self.state.dispatched.borrow();
            let (_, _, c, due) = d.last().unwrap();
            (c.clone(), *due)
        }
    }

    impl RerankClient for MockReranker {
        fn dispatch(&self, req_id: u64, context: &str, candidates: &[String], due: Instant) {
            self.state.dispatched.borrow_mut().push((
                req_id,
                context.to_string(),
                candidates.to_vec(),
                due,
            ));
        }
        fn cancel(&self) {
            *self.state.cancels.borrow_mut() += 1;
        }
        fn take_latest(&self) -> Option<crate::rerank_client::RerankResult> {
            self.state.next.borrow_mut().take()
        }
        fn info(&self) -> Option<crate::rerank_client::RerankInfo> {
            None
        }
    }

    /// A bridge with a reranker that scores the first 2 candidates first.
    fn bridge_with_reranker() -> (EngineBridge, MockClient, MockReranker) {
        let (mut bridge, mock) = bridge_with_mock();
        let reranker = MockReranker::default();
        let cfg = crate::config::RerankCfg {
            first_candidates: 2,
            lambda: Some(0.5),
            ..Default::default()
        };
        bridge.set_reranker(Some(Box::new(reranker.clone())), &cfg);
        (bridge, mock, reranker)
    }

    #[derive(Default)]
    struct RewriteState {
        /// (request id, context, reading, text)
        dispatched: RefCell<Vec<(u64, String, String, String)>>,
        next: RefCell<Option<crate::rewrite_client::RewriteResult>>,
    }

    #[derive(Clone, Default)]
    struct MockRewriter {
        state: Rc<RewriteState>,
    }

    impl MockRewriter {
        /// Answer the last request with `text`, or fail it with `code`.
        fn answer(&self, text: Result<&str, &str>) {
            let req_id = self.state.dispatched.borrow().last().unwrap().0;
            *self.state.next.borrow_mut() = Some(crate::rewrite_client::RewriteResult {
                req_id,
                text: text.ok().map(str::to_string),
                code: text.err().map(str::to_string),
                ms: 1.0,
            });
        }
        fn requests(&self) -> usize {
            self.state.dispatched.borrow().len()
        }
    }

    impl RewriteClient for MockRewriter {
        fn dispatch(&self, req_id: u64, context: &str, reading: &str, text: &str) {
            self.state.dispatched.borrow_mut().push((
                req_id,
                context.to_string(),
                reading.to_string(),
                text.to_string(),
            ));
        }
        fn cancel(&self) {}
        fn take_latest(&self) -> Option<crate::rewrite_client::RewriteResult> {
            self.state.next.borrow_mut().take()
        }
    }

    /// `kasano` typed and converted by the engine, with a rewriter in
    /// rewrite mode.
    fn rewriting_bridge() -> (EngineBridge, MockClient, MockRewriter) {
        let (mut bridge, mock) = bridge_with_mock();
        let rewriter = MockRewriter::default();
        bridge.set_rewriter(Some(Box::new(rewriter.clone())));
        assert!(bridge.set_rewrite_mode(true).is_empty());
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        (bridge, mock, rewriter)
    }

    fn status(actions: &[Action]) -> Option<&str> {
        actions.iter().find_map(|a| match a {
            Action::Status { text } => Some(text.as_str()),
            _ => None,
        })
    }

    fn shown(actions: &[Action]) -> Option<(Vec<String>, usize)> {
        actions.iter().rev().find_map(|a| match a {
            Action::ShowCandidates { list, selected } => Some((list.clone(), *selected)),
            _ => None,
        })
    }

    #[test]
    fn the_models_read_the_context_after_anything_sensitive() {
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.set_app_context(Some("詳細はhttps://example.com/aを見て、"));
        assert_eq!(bridge.context(), "を見て、");
        bridge.set_app_context(Some("今日は晴れ。"));
        assert_eq!(bridge.context(), "今日は晴れ。");
    }

    #[test]
    fn a_private_field_has_no_context_and_learns_nothing() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_private_field(true);
        bridge.set_app_context(Some("今日は"));
        assert_eq!(bridge.context(), "");
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        bridge.handle_space();
        bridge.handle_down();
        bridge.handle_enter();
        // the commit is no context either
        assert_eq!(bridge.context(), "");
        bridge.set_private_field(false);
        bridge.set_app_context(Some("今日は"));
        assert_eq!(bridge.context(), "今日は");
    }

    #[test]
    fn the_rewriter_gets_no_sensitive_context_nor_text() {
        let (mut bridge, mock) = bridge_with_mock();
        let rewriter = MockRewriter::default();
        bridge.set_rewriter(Some(Box::new(rewriter.clone())));
        bridge.set_rewrite_mode(true);
        // the context holds a password word: none of it is sent, not even
        // the text after it
        bridge.set_app_context(Some("パスワードは"));
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の"]);
        bridge.apply_async_result();
        bridge.handle_space();
        assert_eq!(rewriter.state.dispatched.borrow()[0].1, "");
        bridge.handle_escape();
        bridge.handle_escape();
        bridge.set_app_context(Some("今日は"));
        type_str(&mut bridge, "kasano");
        bridge.handle_space();
        assert_eq!(
            rewriter.state.dispatched.borrow().last().unwrap().1,
            "今日は"
        );
        // a conversion that is a token itself is not sent at all
        bridge.handle_escape();
        bridge.handle_escape();
        let before = rewriter.state.dispatched.borrow().len();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["AKIAIOSFODNN7EXAMPLE"]);
        bridge.apply_async_result();
        bridge.handle_space();
        assert_eq!(rewriter.state.dispatched.borrow().len(), before);
    }

    #[test]
    fn rewrite_mode_needs_a_rewriter() {
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.set_rewrite_mode(true);
        assert!(!bridge.rewrite_mode());
    }

    #[test]
    fn the_conversion_key_asks_for_a_rewrite_that_comes_first() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        let out = bridge.handle_space();
        assert_eq!(status(&out.actions), Some("書き直し中…"));
        {
            let sent = rewriter.state.dispatched.borrow();
            assert_eq!((sent[0].2.as_str(), sent[0].3.as_str()), ("かさの", "傘の"));
        }
        rewriter.answer(Ok("傘の"));
        // the same text as the engine's top: it moves first, once
        let actions = bridge.apply_async_result();
        let (list, selected) = shown(&actions).unwrap();
        assert_eq!(
            (list[0].as_str(), list[1].as_str(), selected),
            ("傘の", "笠の", 0)
        );
        assert_eq!(status(&actions), Some(""));
    }

    #[test]
    fn a_rewrite_keeps_a_selection_off_the_top() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.handle_space();
        bridge.handle_down();
        rewriter.answer(Ok("傘をさして"));
        let actions = bridge.apply_async_result();
        let (list, selected) = shown(&actions).unwrap();
        assert_eq!(list[0], "傘をさして");
        assert_eq!(list[selected], "笠の");
        assert!(actions.contains(&Action::UpdatePreedit {
            text: "笠の".into()
        }));
        // Enter commits the selection, the context grows by it
        let out = bridge.handle_enter();
        assert!(out.actions.contains(&Action::Commit {
            text: "笠の".into()
        }));
    }

    #[test]
    fn enter_commits_the_rewrite() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.handle_space();
        rewriter.answer(Ok("傘をさして"));
        bridge.apply_async_result();
        let out = bridge.handle_enter();
        assert!(out.actions.contains(&Action::Commit {
            text: "傘をさして".into()
        }));
        assert_eq!(bridge.context(), "傘をさして");
    }

    #[test]
    fn escape_leaves_the_rewrite_out_of_typing() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.handle_space();
        rewriter.answer(Ok("傘をさして"));
        bridge.apply_async_result();
        // back to typing: Enter commits the engine's conversion
        bridge.handle_escape();
        let out = bridge.handle_enter();
        assert!(out.actions.contains(&Action::Commit {
            text: "傘の".into()
        }));
    }

    #[test]
    fn the_same_reading_is_rewritten_once() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.handle_space();
        rewriter.answer(Ok("傘をさして"));
        bridge.apply_async_result();
        bridge.handle_escape();
        let out = bridge.handle_space();
        assert_eq!(rewriter.requests(), 1);
        assert_eq!(status(&out.actions), None);
        assert_eq!(shown(&out.actions).unwrap().0[0], "傘をさして");
    }

    #[test]
    fn a_failed_rewrite_says_why() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.handle_space();
        rewriter.answer(Err("usage_limit"));
        let actions = bridge.apply_async_result();
        assert_eq!(
            status(&actions),
            Some(crate::rewrite_client::failure_message("usage_limit"))
        );
        assert!(shown(&actions).is_none());
    }

    #[test]
    fn rewrite_mode_turned_on_over_an_open_conversion_asks_at_once() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        bridge.set_rewrite_mode(false);
        bridge.handle_space();
        assert_eq!(rewriter.requests(), 0);
        let actions = bridge.set_rewrite_mode(true);
        assert_eq!(status(&actions), Some("書き直し中…"));
        assert_eq!(rewriter.requests(), 1);
    }

    #[test]
    fn user_words_come_first_and_inside_longer_readings() {
        let (mut bridge, mock) = bridge_with_mock();
        let (entries, _) = crate::user_dict::parse("かさ\t笠\t名詞\n");
        bridge.set_user_dict(Some(crate::user_dict::UserDict::new(&entries)));
        type_str(&mut bridge, "kasa");
        mock.set_result("かさ", &["傘", "嵩"]);
        bridge.apply_async_result();
        let (list, _) = shown(&bridge.handle_space().actions).unwrap();
        assert_eq!(list[0], "笠");
        bridge.handle_escape();
        bridge.handle_escape();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "嵩の"]);
        bridge.apply_async_result();
        let (list, _) = shown(&bridge.handle_space().actions).unwrap();
        assert_eq!(list[0], "笠の");
    }

    #[test]
    fn ctrl_delete_forgets_a_learned_candidate_or_hides_it() {
        let dir =
            std::env::temp_dir().join(format!("setsuna-bridge-forget-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (mut bridge, mock) = bridge_with_mock();
        let cfg = crate::config::LearningCfg::default();
        bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
        bridge.set_hidden(Some(crate::hidden::Hidden::at(dir.join("hidden.jsonl"))));
        let convert = |bridge: &mut EngineBridge| {
            type_str(bridge, "kasano");
            mock.set_result("かさの", &["傘の", "笠の", "嵩の"]);
            bridge.apply_async_result();
            bridge.handle_space()
        };
        // nothing learned for 笠の: hidden from now on
        convert(&mut bridge);
        bridge.handle_down();
        let out = bridge.handle_forget_candidate();
        assert_eq!(
            status(&out.actions),
            Some("この候補を表示しないようにしました")
        );
        assert!(!shown(&out.actions).unwrap().0.contains(&"笠の".to_string()));
        bridge.handle_escape();
        bridge.handle_escape();
        let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
        assert!(!list.contains(&"笠の".to_string()));
        // 嵩の chosen, so learned first; Ctrl+Delete forgets that
        for _ in 0..list.iter().position(|c| c == "嵩の").unwrap() {
            bridge.handle_down();
        }
        bridge.handle_enter();
        let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
        assert_eq!(list[0], "嵩の");
        let out = bridge.handle_forget_candidate();
        assert_eq!(status(&out.actions), Some("学習を取り消しました"));
        bridge.handle_escape();
        bridge.handle_escape();
        let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
        assert_eq!(list[0], "傘の");
        // the kana stay
        let i = list.iter().position(|c| c == "かさの").unwrap();
        for _ in 0..i {
            bridge.handle_down();
        }
        let out = bridge.handle_forget_candidate();
        assert_eq!(status(&out.actions), Some("この候補は消せません"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn picking_the_engines_second_over_a_rewrite_learns_against_its_top() {
        let (mut bridge, _mock, rewriter) = rewriting_bridge();
        let dir = std::env::temp_dir().join(format!(
            "setsuna-bridge-rewrite-learn-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = crate::config::LearningCfg::default();
        bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
        bridge.handle_space();
        rewriter.answer(Ok("傘をさして"));
        bridge.apply_async_result();
        // the rewrite, 傘の, 笠の: pick 笠の
        bridge.handle_down();
        bridge.handle_down();
        bridge.handle_enter();
        let log = std::fs::read_to_string(dir.join("learning.jsonl")).unwrap();
        assert!(log.contains("笠の") && log.contains("傘の"), "{log}");
        assert!(!log.contains("傘をさして"), "{log}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The engine's answer to the last request, with converter scores.
    fn answer_scored(mock: &MockClient, candidates: &[&str], scores: &[f32]) {
        let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
        *mock.state.next.borrow_mut() = Some(EngineResult {
            req_id,
            reading,
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
            scores: scores.to_vec(),
            ..Default::default()
        });
    }

    /// A bridge on a mock engine. Emoji are off: the lists the tests
    /// compare are the engine's (`emoji_join_the_list` has them on).
    fn bridge_with_mock() -> (EngineBridge, MockClient) {
        let mock = MockClient::default();
        let mut bridge = EngineBridge::new(Some(Box::new(mock.clone())));
        bridge.emoji_enabled = false;
        (bridge, mock)
    }

    #[test]
    fn commands_typed_at_a_word_start_go_to_the_app() {
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.pass_commands = true;
        bridge.set_app_context(Some("hello "));
        assert!(!bridge.handle_punctuation("・").consumed);
        for c in "review-pr".chars() {
            assert!(!bridge.handle_char(c).consumed, "{c}");
        }
        assert!(!bridge.would_consume(&KeyKind::Char('x')));
        // the space ends it: romaji again
        bridge.handle_space();
        assert!(bridge.handle_char('k').consumed);
        bridge.handle_escape();
        // not at a word's start, or the text unknown: a symbol as usual
        bridge.set_app_context(Some("今日は"));
        assert!(bridge.handle_punctuation("・").consumed);
        bridge.handle_escape();
        bridge.forget_context();
        assert!(bridge.handle_punctuation("＠").consumed);
    }

    #[test]
    fn a_deferred_preedit_changes_once() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        bridge.rerank_cfg.defer_ms = 250;
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
        // the engine's top waits for the reranked one: the kana meanwhile
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("かさの")
        );
        assert!(bridge.deferred_for().is_some());
        assert!(bridge.tick().is_empty());
        rr.answer(&[-30.0, -20.0]);
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("笠の")
        );
        assert!(bridge.deferred_for().is_none());
        // none by the deadline: the engine's top
        bridge.handle_escape();
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
        bridge.apply_async_result();
        bridge.deferred.as_mut().unwrap().until = Instant::now();
        assert_eq!(preedit(&bridge.tick()).as_deref(), Some("傘の"));
    }

    #[test]
    fn commits_are_predicted_and_tab_opens_them() {
        let path = std::env::temp_dir().join(format!(
            "setsuna-bridge-predict-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let (mut bridge, mock) = bridge_with_mock();
        bridge.live_conversion = true;
        bridge.set_predictor(Some(crate::predict::Predictor::at(path.clone())));
        type_str(&mut bridge, "kasanoshita");
        mock.set_result("かさのした", &["傘の下"]);
        bridge.apply_async_result();
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の下"]);
        // its start typed: predicted, but no window comes up by itself;
        // Tab opens them
        type_str(&mut bridge, "kasanos");
        assert_eq!(bridge.predictions().len(), 1);
        assert!(bridge.suggestions().is_empty());
        let out = bridge.handle_tab(false);
        assert_eq!(preedit(&out.actions).as_deref(), Some("傘の下"));
        // the s still being typed is no part of it
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の下"]);
        // Escape from them goes back to typing
        type_str(&mut bridge, "kasano");
        bridge.handle_tab(false);
        bridge.handle_escape();
        assert!(!bridge.is_converting() && bridge.is_active());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn emoji_join_the_list() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.emoji_enabled = true;
        type_str(&mut bridge, "nikoniko");
        mock.set_result("にこにこ", &["ニコニコ"]);
        bridge.apply_async_result();
        let (list, _) = shown(&bridge.handle_space().actions).unwrap();
        assert!(list.contains(&"😊".to_string()));
        assert!(list.last().unwrap().contains('^'));
    }

    #[test]
    fn a_client_whose_daemon_exited_counts_as_none() {
        // the frontend starts a new daemon when `has_client` is false; a dead
        // one kept counting as a client left the IME converting nothing
        let (bridge, mock) = bridge_with_mock();
        assert!(bridge.has_client());
        mock.state.gone.set(true);
        assert!(!bridge.has_client());
        assert!(!EngineBridge::new(None).has_client());
    }

    fn type_str(bridge: &mut EngineBridge, s: &str) {
        for c in s.chars() {
            bridge.handle_char(c);
        }
    }

    /// A bridge with the small mixed-input model of the `mixed` tests.
    fn mixed_bridge() -> (EngineBridge, MockClient) {
        let (mut bridge, mock) = bridge_with_mock();
        let model = std::sync::Arc::new(crate::mixed::tests::model());
        bridge.set_mixed(
            Some(model),
            &crate::config::default_symbol_map(),
            crate::mixed::Casing::default(),
        );
        (bridge, mock)
    }

    #[test]
    fn mixed_shift_letters_type_into_the_composition() {
        let (mut bridge, _mock) = mixed_bridge();
        for c in "GitHub".chars() {
            if c.is_ascii_uppercase() {
                bridge.handle_shifted_alpha(c);
            } else {
                bridge.handle_char(c);
            }
        }
        let out = {
            type_str(&mut bridge, "ni");
            bridge.handle_enter()
        };
        assert_eq!(commits(&out.actions), vec!["GitHubに"]);
    }

    #[test]
    fn mixed_space_after_english_types_a_space() {
        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "hello");
        let out = bridge.handle_space();
        assert_eq!(preedit(&out.actions).as_deref(), Some("hello "));
        assert!(!bridge.is_converting());
        type_str(&mut bridge, "google");
        // two words: an English sentence, with its capital
        assert_eq!(
            preedit(&bridge.handle_space().actions).as_deref(),
            Some("Hello google ")
        );
        assert_eq!(
            commits(&bridge.handle_enter().actions),
            vec!["Hello google "]
        );
    }

    #[test]
    fn mixed_space_twice_after_english_offers_it_with_a_space() {
        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "hello");
        bridge.handle_space();
        let out = bridge.handle_space();
        assert!(bridge.is_converting());
        assert_eq!(preedit(&out.actions).as_deref(), Some("hello "));
        // the next key commits it, as between English words
        assert_eq!(commits(&bridge.handle_char('w').actions), vec!["hello "]);
    }

    #[test]
    fn live_conversion_reads_a_pending_n_as_n_kana() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.live_conversion = true;
        type_str(&mut bridge, "kekkon");
        // the engine converts けっこん, not けっこ, for Enter; the preview
        // shows the n as typed
        assert_eq!(mock.dispatched_readings().last().unwrap(), "けっこん");
        mock.set_result("けっこん", &["結婚"]);
        let out = bridge.apply_async_result();
        assert_eq!(preedit(&out).as_deref(), Some("けっこn"));
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["結婚"]);
        // a vowel after it converts again
        type_str(&mut bridge, "kekkona");
        assert_eq!(mock.dispatched_readings().last().unwrap(), "けっこな");
        // typed to its end (nn), the conversion of けっこん is shown
        bridge.handle_escape();
        type_str(&mut bridge, "kekkonn");
        mock.set_result("けっこん", &["結婚"]);
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("結婚")
        );
    }

    #[test]
    fn a_pending_n_shows_as_typed_after_the_conversion_before_it() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.live_conversion = true;
        type_str(&mut bridge, "kyouha");
        mock.set_result("きょうは", &["今日は"]);
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("今日は")
        );
        let out = bridge.handle_char('n');
        assert_eq!(preedit(&out.actions).as_deref(), Some("今日はn"));
        // the conversion of きょうはん does not take its place
        mock.set_result("きょうはん", &["共犯"]);
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("今日はn")
        );
        // nor does the whole go back to kana when the n becomes な
        let out = bridge.handle_char('a');
        assert_eq!(preedit(&out.actions).as_deref(), Some("今日はな"));
        mock.set_result("きょうはな", &["今日花"]);
        assert_eq!(
            preedit(&bridge.apply_async_result()).as_deref(),
            Some("今日花")
        );
        // Enter on a pending n commits the conversion with ん
        bridge.handle_escape();
        type_str(&mut bridge, "kyouhan");
        mock.set_result("きょうはん", &["共犯"]);
        bridge.apply_async_result();
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["共犯"]);
        // and Space shows it, not asking the engine again
        type_str(&mut bridge, "kyouhan");
        mock.set_result("きょうはん", &["共犯"]);
        bridge.apply_async_result();
        let sent = mock.dispatched_readings().len();
        let (list, _) = shown(&bridge.handle_space().actions).unwrap();
        assert_eq!(list[0], "共犯");
        assert_eq!(mock.dispatched_readings().len(), sent);
    }

    #[test]
    fn a_reading_whose_end_changed_keeps_the_conversion_it_shares() {
        // mixed input reading letters otherwise, a kana replaced: the shared
        // start stays converted while the new end is
        let (mut bridge, mock) = bridge_with_mock();
        bridge.live_conversion = true;
        type_str(&mut bridge, "watashiha");
        mock.set_result("わたしは", &["私は"]);
        bridge.apply_async_result();
        bridge.cached_prefix_reading = "わたしわ".into();
        bridge.cached_prefix_kanji = "私わ".into();
        assert_eq!(bridge.live_preedit(), "私は");
    }

    #[test]
    fn mixed_enter_commits_what_the_whole_reads_as() {
        // the shipped model knows ok
        let (mut bridge, _mock) = bridge_with_mock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        bridge.set_mixed(
            Some(std::sync::Arc::new(
                crate::mixed::MixedModel::load(&dir).unwrap(),
            )),
            &crate::config::default_symbol_map(),
            crate::mixed::Casing::default(),
        );
        bridge.live_conversion = true;
        type_str(&mut bridge, "ok");
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["OK"]);
    }

    #[test]
    fn mixed_a_slip_shown_as_english_reads_again_when_deleted() {
        // the shipped model reads sp as an acronym; deleting the p leaves
        // the s typed, which reads as romaji again (frozen as SP, the
        // capitals kept it English)
        let (mut bridge, _mock) = bridge_with_mock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        bridge.set_mixed(
            Some(std::sync::Arc::new(
                crate::mixed::MixedModel::load(&dir).unwrap(),
            )),
            &crate::config::default_symbol_map(),
            crate::mixed::Casing::default(),
        );
        type_str(&mut bridge, "nanidemosp");
        assert_eq!(bridge.composing.preedit(), "なにでもSP");
        bridge.handle_backspace();
        assert_eq!(bridge.composing.preedit(), "なにでもs");
        type_str(&mut bridge, "u");
        assert_eq!(bridge.composing.preedit(), "なにでもす");
    }

    #[test]
    fn mixed_space_after_a_lone_i_waits_for_the_next_word() {
        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "i");
        bridge.handle_space();
        assert!(!bridge.is_converting());
        type_str(&mut bridge, "hello");
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["I hello"]);
        // Space again converts い
        type_str(&mut bridge, "i");
        bridge.handle_space();
        bridge.handle_space();
        assert!(bridge.is_converting());
    }

    #[test]
    fn period_after_a_digit_types_a_point() {
        let (mut bridge, _mock) = bridge_with_mock();
        type_str(&mut bridge, "3");
        let out = bridge.handle_punctuation("。");
        assert_eq!(preedit(&out.actions).as_deref(), Some("3."));
        type_str(&mut bridge, "desu");
        let out = bridge.handle_punctuation("。");
        assert_eq!(preedit(&out.actions).as_deref(), Some("3.です。"));

        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "3");
        let out = bridge.handle_punctuation("。");
        assert_eq!(preedit(&out.actions).as_deref(), Some("3."));
    }

    #[test]
    fn mixed_conversion_offers_the_reading_without_english() {
        let (mut bridge, mock) = mixed_bridge();
        type_str(&mut bridge, "hello");
        // the first Space types a space
        bridge.handle_space();
        bridge.handle_space();
        mock.answer_last(&["hello"]);
        bridge.apply_async_result();
        // with the English list in, the reading with every letter as romaji
        assert_eq!(mock.dispatched_readings().last().unwrap(), "へっぉ");
        mock.answer_last(&["減っぉ"]);
        let out = bridge.apply_async_result();
        let list = out
            .iter()
            .find_map(|a| match a {
                Action::ShowCandidates { list, .. } => Some(list.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(list[0], "hello ");
        assert_eq!(list[1], "減っぉ");
        assert!(list.contains(&"hello".to_string()));
        // Space again goes on to it
        assert_eq!(
            preedit(&bridge.handle_space().actions).as_deref(),
            Some("減っぉ")
        );
    }

    #[test]
    fn mixed_kana_forms_read_every_letter_as_romaji() {
        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "hello");
        let out = bridge.handle_convert_to(Form::Hiragana);
        assert_eq!(preedit(&out.actions).as_deref(), Some("へっぉ"));
    }

    #[test]
    fn mixed_symbols_show_per_span() {
        let (mut bridge, _mock) = mixed_bridge();
        type_str(&mut bridge, "example");
        bridge.handle_punctuation("。");
        type_str(&mut bridge, "com");
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["example.com"]);
        type_str(&mut bridge, "desu");
        bridge.handle_punctuation("。");
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["です。"]);
    }

    #[test]
    fn punctuation_after_an_english_sentence_is_english() {
        let (mut bridge, _mock) = bridge_with_mock();
        // typed with Shift: it goes on with the English
        bridge.handle_shifted_alpha('T');
        type_str(&mut bridge, "his");
        bridge.handle_space();
        type_str(&mut bridge, "works");
        let out = bridge.handle_punctuation("！");
        assert_eq!(preedit(&out.actions).as_deref(), Some("This works!"));
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["This works!"]);
        // after English committed: committed as typed
        bridge.set_app_context(Some("This is a pen"));
        let out = bridge.handle_punctuation("。");
        assert_eq!(commits(&out.actions), vec!["."]);
        // English inside Japanese: Japanese punctuation
        bridge.set_app_context(Some("これはGoogle"));
        let out = bridge.handle_punctuation("。");
        assert_eq!(preedit(&out.actions).as_deref(), Some("。"));
    }

    #[test]
    fn mixed_punctuation_follows_the_sentence() {
        let (mut bridge, mock) = mixed_bridge();
        type_str(&mut bridge, "hello");
        bridge.handle_punctuation("、");
        // Space after English punctuation types a space, as after a word
        let out = bridge.handle_space();
        assert_eq!(preedit(&out.actions).as_deref(), Some("hello, "));
        type_str(&mut bridge, "google");
        bridge.handle_punctuation("。");
        assert_eq!(
            commits(&bridge.handle_enter().actions),
            vec!["Hello, google."]
        );
        // a candidate that turned it into 。 is a misconversion
        type_str(&mut bridge, "korehagoogle");
        bridge.handle_punctuation("。");
        mock.set_result("これはgoogle。", &["これはgoogle。"]);
        bridge.apply_async_result();
        assert_eq!(
            commits(&bridge.handle_enter().actions),
            vec!["これはgoogle。"]
        );
        assert!(crate::mixed::keeps_latin("hello, world.", "hello, world."));
        assert!(!crate::mixed::keeps_latin(
            "hello, world.",
            "hello、world。"
        ));
    }

    #[test]
    fn mixed_candidates_keep_the_english_as_typed() {
        let (mut bridge, mock) = mixed_bridge();
        type_str(&mut bridge, "googlede");
        mock.set_result("googleで", &["Googleで", "googleで"]);
        bridge.apply_async_result();
        let out = bridge.handle_space();
        assert!(out.actions.iter().any(|a| matches!(
            a,
            Action::ShowCandidates { list, .. } if list[0] == "googleで" && !list.iter().any(|c| c == "Googleで")
        )));
    }

    #[test]
    fn char_enters_composing_with_raw_preedit() {
        let (mut bridge, _mock) = bridge_with_mock();
        let outcome = bridge.handle_char('k');
        assert!(outcome.consumed);
        assert_eq!(
            outcome.actions,
            vec![Action::UpdatePreedit { text: "k".into() }]
        );
        assert!(bridge.is_active());
    }

    #[test]
    fn completed_kana_dispatches_convert() {
        let (mut bridge, mock) = bridge_with_mock();
        // 3+ char reading required: dispatch_live skips OOD short partials
        // (see MIN_DISPATCH_CHARS). The guard only fires when hira chars < 3,
        // so we type 3 chars to verify dispatch in the normal case.
        type_str(&mut bridge, "kasa");
        // type "ka"->"か" (1 char, no dispatch), "sa"->"かさ" (2 chars, no
        // dispatch) — actually we need 3 chars, so type one more:
        type_str(&mut bridge, "no");
        assert_eq!(
            mock.dispatched_readings().last(),
            Some(&"かさの".to_string())
        );
    }

    #[test]
    fn short_reading_skips_dispatch() {
        // Regression test for the dispatch guard: 1- and 2-char readings
        // must NOT trigger a daemon dispatch — they are OOD on the
        // sentence-trained CTC-NAT and produce nonsense flicker. The
        // preedit still shows raw hira; conversion fires once the reading
        // grows past MIN_DISPATCH_CHARS.
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        assert!(mock.dispatched_readings().is_empty(), "1-char skipped");
        type_str(&mut bridge, "sa");
        assert!(mock.dispatched_readings().is_empty(), "2-char skipped");
        type_str(&mut bridge, "no");
        // Now ≥3 chars → dispatch.
        assert_eq!(
            mock.dispatched_readings().last(),
            Some(&"かさの".to_string())
        );
    }

    #[test]
    fn space_forces_dispatch_for_short_reading() {
        // Live typing still skips 1-char readings, but explicit Space must
        // ask the daemon so dictionary/model fallback can return more than
        // the provisional single candidate.
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        assert!(mock.dispatched_readings().is_empty(), "live skip remains");

        let outcome = bridge.handle_space();
        assert!(outcome.consumed);
        assert_eq!(mock.dispatched_readings().last(), Some(&"か".to_string()));
        assert_eq!(
            outcome.actions,
            vec![
                Action::UpdatePreedit { text: "か".into() },
                Action::ShowCandidates {
                    list: vec!["か".into()],
                    selected: 0,
                },
            ]
        );
    }

    #[test]
    fn space_flushes_pending_n_before_dispatch() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "shin");
        assert!(mock.dispatched_readings().is_empty());

        let outcome = bridge.handle_space();
        assert!(outcome.consumed);
        assert_eq!(mock.dispatched_readings().last(), Some(&"しん".to_string()));
        assert_eq!(
            outcome.actions,
            vec![
                Action::UpdatePreedit {
                    text: "しん".into()
                },
                Action::ShowCandidates {
                    list: vec!["しん".into()],
                    selected: 0,
                },
            ]
        );
    }

    #[test]
    fn async_result_swaps_preedit_to_kanji() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        mock.set_result("か", &["可", "蚊"]);
        let actions = bridge.apply_async_result();
        assert_eq!(actions, vec![Action::UpdatePreedit { text: "可".into() }]);
    }

    #[test]
    fn a_typed_number_stays_as_typed_when_the_model_merges_its_digits() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "19911225");
        mock.set_result("19911225", &["199125", "1991125"]);
        let actions = bridge.apply_async_result();
        assert_eq!(
            actions,
            vec![Action::UpdatePreedit {
                text: "19911225".into()
            }]
        );
        let outcome = bridge.handle_space();
        let Some(Action::ShowCandidates { list, .. }) = outcome.actions.last() else {
            panic!("no candidate window: {:?}", outcome.actions);
        };
        assert_eq!(list[0], "19911225");
        assert!(!list.iter().any(|c| c == "199125" || c == "1991125"));
    }

    #[test]
    fn space_opens_candidate_window_with_async_list() {
        let (mut bridge, mock) = bridge_with_mock();
        // ≥3 char reading so dispatch_live actually fires (otherwise the
        // MIN_DISPATCH_CHARS guard short-circuits and the mock daemon
        // never gets the reading).
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        let outcome = bridge.handle_space();
        assert!(outcome.consumed);
        assert_eq!(
            outcome.actions,
            vec![
                Action::UpdatePreedit {
                    text: "傘の".into()
                },
                // The reading and its katakana are always offered last.
                Action::ShowCandidates {
                    list: vec![
                        "傘の".into(),
                        "笠の".into(),
                        "かさの".into(),
                        "カサノ".into()
                    ],
                    selected: 0,
                },
            ]
        );
    }

    #[test]
    fn space_before_result_shows_provisional_then_async_swaps() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        // No result applied yet: Space opens a provisional 1-item window.
        let outcome = bridge.handle_space();
        assert_eq!(
            outcome.actions,
            vec![
                Action::UpdatePreedit { text: "か".into() },
                Action::ShowCandidates {
                    list: vec!["か".into()],
                    selected: 0,
                },
            ]
        );
        // The async result then swaps in the real candidate list.
        mock.set_result("か", &["可", "蚊"]);
        let actions = bridge.apply_async_result();
        assert_eq!(
            actions,
            vec![
                Action::UpdatePreedit { text: "可".into() },
                Action::ShowCandidates {
                    list: vec!["可".into(), "蚊".into(), "か".into(), "カ".into()],
                    selected: 0,
                },
            ]
        );
    }

    #[test]
    fn wrong_hiragana_candidate_is_replaced_by_the_reading() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "moji");
        mock.set_result("もじ", &["文字", "もし"]);
        bridge.apply_async_result();
        let outcome = bridge.handle_space();
        assert_eq!(
            outcome.actions[1],
            Action::ShowCandidates {
                list: vec!["文字".into(), "もじ".into(), "モジ".into()],
                selected: 0,
            }
        );
    }

    fn converted_kasano() -> (EngineBridge, MockClient) {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        (bridge, mock)
    }

    #[test]
    fn enter_keeps_the_unconverted_completed_tail() {
        let (mut bridge, _) = converted_kasano();
        type_str(&mut bridge, "ame");
        assert_eq!(
            bridge.handle_enter().actions,
            vec![Action::Commit {
                text: "傘のあめ".into()
            }]
        );
    }

    #[test]
    fn enter_does_not_commit_a_deleted_or_edited_reading() {
        let (mut bridge, _) = converted_kasano();
        bridge.handle_backspace();
        assert_eq!(
            bridge.handle_enter().actions,
            vec![Action::Commit {
                text: "かさ".into()
            }]
        );
        let (mut bridge, _) = converted_kasano();
        bridge.handle_home_end(false);
        type_str(&mut bridge, "a");
        bridge.handle_home_end(true);
        assert_eq!(
            bridge.handle_enter().actions,
            vec![Action::Commit {
                text: "あかさの".into()
            }]
        );
    }

    #[test]
    fn space_uses_the_whole_reading_while_its_conversion_is_pending() {
        let (mut bridge, _) = converted_kasano();
        type_str(&mut bridge, "ame");
        let out = bridge.handle_space();
        assert_eq!(
            out.actions[0],
            Action::UpdatePreedit {
                text: "傘のあめ".into()
            }
        );
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "傘のあめ".into()
            }
        );
    }

    #[test]
    fn conversion_keeps_pending_romaji_with_or_without_a_result() {
        for answer in [false, true] {
            let (mut bridge, mock) = converted_kasano();
            type_str(&mut bridge, "k");
            assert_eq!(
                bridge.handle_space().actions[0],
                Action::UpdatePreedit {
                    text: "傘のk".into()
                }
            );
            if answer {
                mock.set_result("かさの", &["傘の", "笠の"]);
                assert_eq!(
                    bridge.apply_async_result()[0],
                    Action::UpdatePreedit {
                        text: "傘のk".into()
                    }
                );
                assert_eq!(
                    bridge.handle_down().actions[0],
                    Action::UpdatePreedit {
                        text: "笠のk".into()
                    }
                );
            }
            let text = if answer { "笠のk" } else { "傘のk" };
            assert_eq!(
                bridge.handle_enter().actions[0],
                Action::Commit { text: text.into() }
            );
        }
    }

    #[test]
    fn provisional_pending_romaji_is_not_appended_twice() {
        let (mut bridge, _) = bridge_with_mock();
        type_str(&mut bridge, "nekok");
        assert_eq!(
            bridge.handle_space().actions[0],
            Action::UpdatePreedit {
                text: "ねこk".into()
            }
        );
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "ねこk".into()
            }
        );
    }

    #[test]
    fn late_results_do_not_cross_composition_boundaries() {
        for end in 0..5 {
            let (mut bridge, mock) = bridge_with_mock();
            type_str(&mut bridge, "kasano");
            mock.set_result("かさの", &["笠の"]);
            let old = mock.state.next.borrow_mut().take().unwrap();
            match end {
                0 => {
                    bridge.discard();
                }
                1 => {
                    bridge.handle_enter();
                }
                2 => {
                    bridge.handle_escape();
                }
                3 => {
                    bridge.reset_preserving_client(&crate::config::FrontendCfg::default());
                }
                _ => {
                    for _ in 0..3 {
                        bridge.handle_backspace();
                    }
                    assert!(!bridge.is_active());
                }
            }
            bridge.set_app_context(Some("別の文脈"));
            type_str(&mut bridge, "kasano");
            assert!(mock.state.dispatched.borrow().last().unwrap().0 > old.req_id);
            *mock.state.next.borrow_mut() = Some(old);
            assert!(bridge.apply_async_result().is_empty());
            mock.answer_last(&["傘の"]);
            assert_eq!(
                bridge.apply_async_result(),
                vec![Action::UpdatePreedit {
                    text: "傘の".into()
                }]
            );
        }
    }

    #[test]
    fn date_time_candidates_work_without_a_model_including_short_readings() {
        for (typed, reading) in [
            ("kinou", "きのう"),
            ("kyou", "きょう"),
            ("ima", "いま"),
            ("raishuu", "らいしゅう"),
            ("suiyoubi", "すいようび"),
            ("1jikango", "1じかんご"),
            ("30punmae", "30ぷんまえ"),
            ("raishuunosuiyoubi", "らいしゅうのすいようび"),
        ] {
            let mut bridge = EngineBridge::new(None);
            type_str(&mut bridge, typed);
            let out = bridge.handle_space();
            let expected = crate::date_time::candidates(reading, &bridge.date_time_now.unwrap());
            let list = match &out.actions[1] {
                Action::ShowCandidates { list, .. } => list,
                _ => panic!("no candidate list"),
            };
            assert!(list.contains(&reading.to_string()), "{typed}");
            assert!(expected.iter().all(|c| list.contains(c)), "{typed}");
            let chosen = expected[0].clone();
            bridge.handle_select(list.iter().position(|c| *c == chosen).unwrap());
            assert_eq!(
                bridge.handle_enter().actions[0],
                Action::Commit { text: chosen }
            );
        }
    }

    #[test]
    fn local_date_choice_survives_a_delayed_model_and_segment_result() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kyou");
        let out = bridge.handle_space();
        let now = bridge.date_time_now.unwrap();
        let date = crate::date_time::candidates("きょう", &now)[0].clone();
        let list = match &out.actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!(),
        };
        bridge.handle_select(list.iter().position(|c| *c == date).unwrap());
        mock.answer_last(&["今日", "京"]);
        let actions = bridge.apply_async_result();
        assert_eq!(preedit(&actions), Some(date.clone()));
        assert_eq!(bridge.date_time_now, Some(now));
        let list = match &actions[1] {
            Action::ShowCandidates { list, selected } => {
                assert_eq!(list[*selected], date);
                list
            }
            _ => panic!(),
        };
        assert!(list.contains(&"今日".to_string()));
        assert_eq!(bridge.list_starts[&date], vec![0; date.chars().count()]);
        let actions = bridge.handle_move_cursor(1).actions;
        assert_eq!(segments_shown(&actions).0, vec![date.clone()]);
        mock.answer_last(&["今日", "京"]);
        let actions = bridge.apply_async_result();
        assert_eq!(segments_shown(&actions).0, vec![date.clone()]);
        let (list, selected, _) = segment_list(&actions);
        assert_eq!(list[selected], date);
        assert!(crate::date_time::candidates("きょう", &now)
            .iter()
            .all(|c| list.contains(c)));
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit { text: date }
        );
    }

    #[test]
    fn ngram_scores_reorder_the_engines_candidates_for_the_reranker_too() {
        let path = std::env::temp_dir().join(format!(
            "setsuna-ngram-{}-{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        let text = "博士が来る。猫が白い。白紙に書く。".repeat(20);
        std::fs::write(&path, crate::char_ngram::tests::table(&text)).unwrap();
        let (mut bridge, mock, _rr) = bridge_with_reranker();
        bridge.set_ngram(&crate::config::NgramCfg {
            table: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        });
        type_str(&mut bridge, "hakusi");
        bridge.handle_space();
        // 博紙, the model's top, joins two chars no text has together
        answer_scored(&mock, &["博紙", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
        let actions = bridge.apply_async_result();
        let list = match &actions[1] {
            Action::ShowCandidates { list, .. } => list.clone(),
            _ => panic!(),
        };
        assert_eq!(list[..2], ["博士", "白紙"]);
        let pos = |c: &str| list.iter().position(|x| x == c).unwrap();
        assert!(pos("博紙") > pos("白紙"), "{list:?}");
        let job = bridge.rerank.as_ref().unwrap();
        assert_eq!(job.raw, ["博士", "白紙", "博紙"]);
        assert_eq!(job.scores[..2], [-1.2, -1.3]);
        assert!(job.scores[2] < -1.3);
        // off: the engine's order
        bridge.handle_escape();
        bridge.set_ngram(&crate::config::NgramCfg::default());
        type_str(&mut bridge, "hakusi");
        bridge.handle_space();
        answer_scored(&mock, &["博紙", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
        bridge.apply_async_result();
        assert_eq!(
            bridge.rerank.as_ref().unwrap().raw,
            ["博紙", "博士", "白紙"]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn candidates_that_leave_reading_out_go_last() {
        let (mut bridge, mock, _rr) = bridge_with_reranker();
        type_str(&mut bridge, "hakusi");
        bridge.handle_space();
        // はし leaves く out
        answer_scored(&mock, &["はし", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
        bridge.apply_async_result();
        let job = bridge.rerank.as_ref().unwrap();
        assert_eq!(job.raw, ["博士", "白紙", "はし"]);
        assert_eq!(job.scores, [-1.2, -1.3, -1.3]);
        // and the n-gram does not lift one: し has no pair to lose points on
        let path = std::env::temp_dir().join(format!(
            "setsuna-ngram-spell-{}-{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        let text = "博士が来る。猫が白い。白紙に書く。".repeat(20);
        std::fs::write(&path, crate::char_ngram::tests::table(&text)).unwrap();
        bridge.handle_escape();
        bridge.set_ngram(&crate::config::NgramCfg {
            table: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        });
        type_str(&mut bridge, "hakusi");
        bridge.handle_space();
        answer_scored(&mock, &["博紙", "し", "博士"], &[-1.0, -1.1, -1.2]);
        bridge.apply_async_result();
        let job = bridge.rerank.as_ref().unwrap();
        assert_eq!(job.raw, ["博士", "博紙", "し"]);
        assert!(job.scores[2] <= job.scores[1], "{:?}", job.scores);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn time_snapshot_and_selection_survive_reranking() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        type_str(&mut bridge, "ima");
        bridge.handle_space();
        answer_scored(&mock, &["今", "居間"], &[-1.0, -1.5]);
        let actions = bridge.apply_async_result();
        let now = bridge.date_time_now.unwrap();
        let time = crate::date_time::candidates("いま", &now)[1].clone();
        let list = match &actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!(),
        };
        bridge.handle_select(list.iter().position(|c| *c == time).unwrap());
        rr.answer(&[-30.0, -20.0]);
        let actions = bridge.apply_async_result();
        assert_eq!(bridge.date_time_now, Some(now));
        assert_eq!(preedit(&actions), Some(time.clone()));
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit { text: time }
        );
    }

    #[test]
    fn explicit_conversion_refreshes_old_dates_without_reapplying_character_rules() {
        let mock = MockClient::default();
        let frontend = crate::config::FrontendCfg {
            replacements: vec![crate::config::Replacement {
                from: "今日".into(),
                to: "今日今日".into(),
                enabled: true,
            }],
            ..Default::default()
        };
        let mut bridge = EngineBridge::with_frontend(Some(Box::new(mock.clone())), &frontend);
        type_str(&mut bridge, "kyou");
        let old = Local::now() - chrono::Duration::days(1);
        bridge.date_time_now = Some(old);
        mock.answer_last(&["今日", "京"]);
        bridge.apply_async_result();
        let old_date = crate::date_time::candidates("きょう", &old)[0].clone();
        let out = bridge.handle_space();
        let now = bridge.date_time_now.unwrap();
        let new_date = crate::date_time::candidates("きょう", &now)[0].clone();
        assert_ne!(old_date, new_date);
        let list = match &out.actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!(),
        };
        assert!(!list.contains(&old_date));
        assert!(list.contains(&new_date));
        assert_eq!(list[0], "今日今日");
        bridge.handle_escape();
        let out = bridge.handle_space();
        let list = match &out.actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!(),
        };
        assert!(!list.contains(&old_date));
        assert_eq!(list[0], "今日今日");
    }

    #[test]
    fn local_date_reconversion_keeps_the_normal_kana_candidates() {
        let mut bridge = EngineBridge::new(None);
        type_str(&mut bridge, "ima");
        for _ in 0..3 {
            let out = bridge.handle_space();
            let list = match &out.actions[1] {
                Action::ShowCandidates { list, .. } => list,
                _ => panic!(),
            };
            assert_eq!(&list[..2], ["いま", "イマ"]);
            assert!(list.len() > 2);
            bridge.handle_escape();
        }
    }

    #[test]
    fn date_time_candidates_can_be_disabled_and_do_not_match_sentence_prefixes() {
        let frontend = crate::config::FrontendCfg {
            date_time_candidates: false,
            ..Default::default()
        };
        let mut bridge = EngineBridge::with_frontend(None, &frontend);
        type_str(&mut bridge, "kyou");
        assert_eq!(
            bridge.handle_space().actions[1],
            Action::ShowCandidates {
                list: v(&["きょう"]),
                selected: 0
            }
        );
        let mut bridge = EngineBridge::new(None);
        type_str(&mut bridge, "kyouha");
        assert_eq!(
            bridge.handle_space().actions[1],
            Action::ShowCandidates {
                list: v(&["きょうは"]),
                selected: 0
            }
        );
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kyouha");
        mock.answer_last(&["今日は"]);
        bridge.apply_async_result();
        assert!(bridge.current_list().iter().all(|c| !c.contains('/')));
    }

    #[test]
    fn enter_commits_top_candidate() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        mock.set_result("か", &["可", "蚊"]);
        bridge.apply_async_result();
        let outcome = bridge.handle_enter();
        assert_eq!(outcome.actions, vec![Action::Commit { text: "可".into() }]);
        assert!(!bridge.is_active());
    }

    #[test]
    fn escape_discards_composition() {
        let (mut bridge, _mock) = bridge_with_mock();
        type_str(&mut bridge, "ka");
        let outcome = bridge.handle_escape();
        assert_eq!(outcome.actions, vec![Action::EndComposition]);
        assert!(!bridge.is_active());
    }

    #[test]
    fn punctuation_rules_apply_to_the_candidate_list() {
        let mock = MockClient::default();
        let frontend = crate::config::FrontendCfg {
            punctuation: crate::config::PunctuationStyle::CommaPeriod,
            ..crate::config::FrontendCfg::default()
        };
        let mut bridge = EngineBridge::with_frontend(Some(Box::new(mock.clone())), &frontend);
        type_str(&mut bridge, "hai");
        bridge.handle_punctuation("、");
        mock.set_result("はい、", &["はい、", "肺、"]);
        bridge.apply_async_result();
        let outcome = bridge.handle_space();
        assert_eq!(
            outcome.actions[1],
            Action::ShowCandidates {
                list: vec!["はい，".into(), "肺，".into(), "ハイ，".into()],
                selected: 0,
            }
        );
    }

    #[test]
    fn a_picked_candidate_is_learned_and_ranked_first_next_time() {
        let dir = std::env::temp_dir().join(format!("setsuna-bridge-learn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let learning = || {
            Some(Learning::at(
                dir.join("learning.jsonl"),
                &crate::config::LearningCfg::default(),
            ))
        };
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_learning(learning());
        type_str(&mut bridge, "kami");
        mock.set_result("かみ", &["神", "紙", "髪"]);
        bridge.apply_async_result();
        bridge.handle_space();
        bridge.handle_down();
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit { text: "紙".into() }
        );

        // Another bridge (another application) sees the choice.
        let (mut other, mock) = bridge_with_mock();
        other.set_learning(learning());
        type_str(&mut other, "kami");
        mock.set_result("かみ", &["神", "紙", "髪"]);
        assert_eq!(
            other.apply_async_result(),
            vec![Action::UpdatePreedit { text: "紙".into() }]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn committing_the_top_candidate_learns_nothing() {
        let dir = std::env::temp_dir().join(format!("setsuna-bridge-top-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("learning.jsonl");
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_learning(Some(Learning::at(
            path.clone(),
            &crate::config::LearningCfg::default(),
        )));
        type_str(&mut bridge, "kami");
        mock.set_result("かみ", &["神", "紙"]);
        bridge.apply_async_result();
        bridge.handle_space();
        bridge.handle_enter();
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn committed_text_is_the_next_context_until_forgotten() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の"]);
        bridge.apply_async_result();
        bridge.handle_enter();
        assert_eq!(bridge.context(), "傘の");
        type_str(&mut bridge, "sitade");
        let last = mock.state.dispatched.borrow().last().cloned().unwrap();
        assert_eq!((last.1.as_str(), last.2.as_str()), ("傘の", "したで"));
        bridge.handle_escape();
        bridge.forget_context();
        type_str(&mut bridge, "sitade");
        let last = mock.state.dispatched.borrow().last().cloned().unwrap();
        assert_eq!(last.1, "");
    }

    #[test]
    fn the_apps_text_replaces_the_kept_context() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.handle_punctuation("。");
        bridge.handle_enter();
        assert_eq!(bridge.context(), "。");
        // line breaks and embedded objects are dropped, the tail kept
        let long = format!("{}\r\n三相\u{FFFC}交流の", "あ".repeat(100));
        bridge.set_app_context(Some(&long));
        assert_eq!(
            bridge.context().chars().count(),
            EngineBridge::CONTEXT_WINDOW
        );
        assert!(bridge.context().ends_with("あ三相交流の"));
        type_str(&mut bridge, "teiden");
        let last = mock.state.dispatched.borrow().last().cloned().unwrap();
        assert!(last.1.ends_with("三相交流の"));
        // the next read replaces it again
        bridge.handle_escape();
        bridge.set_app_context(Some("ビー"));
        assert_eq!(bridge.context(), "ビー");
    }

    #[test]
    fn unreadable_app_text_keeps_the_kept_context() {
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.handle_punctuation("。");
        bridge.handle_enter();
        bridge.set_app_context(None);
        assert_eq!(bridge.context(), "。");
        bridge.set_app_context(Some(""));
        assert_eq!(bridge.context(), "。");
        bridge.set_app_context(Some("\r\n"));
        assert_eq!(bridge.context(), "。");
    }

    /// きしゃのきしゃ converted, Space pressed: the whole-sentence list.
    fn converting_kisha() -> (EngineBridge, MockClient) {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kishanokisha");
        let pos: &[u32] = &[0, 2, 3, 4, 6];
        mock.set_aligned(
            "きしゃのきしゃ",
            &["記者の記者", "汽車の記者", "記者の汽車"],
            &[pos, pos, pos],
        );
        bridge.apply_async_result();
        bridge.handle_space();
        (bridge, mock)
    }

    fn segments_shown(actions: &[Action]) -> (Vec<String>, usize) {
        actions
            .iter()
            .find_map(|a| match a {
                Action::UpdateSegments { segments, focus } => Some((segments.clone(), *focus)),
                _ => None,
            })
            .expect("UpdateSegments")
    }

    fn segment_list(actions: &[Action]) -> (Vec<String>, usize, usize) {
        actions
            .iter()
            .find_map(|a| match a {
                Action::ShowSegmentCandidates { list, selected, at } => {
                    Some((list.clone(), *selected, *at))
                }
                _ => None,
            })
            .expect("ShowSegmentCandidates")
    }

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn discard_drops_the_composition_and_a_late_result() {
        let (mut bridge, mock) = converting_kisha();
        assert_eq!(
            bridge.discard(),
            vec![Action::HideCandidates, Action::EndComposition]
        );
        assert!(!bridge.is_active());
        assert_eq!(bridge.context(), "");
        mock.set_result("きしゃのきしゃ", &["記者の記者"]);
        assert!(bridge.apply_async_result().is_empty());
        assert!(bridge.discard().is_empty());
        let out = bridge.handle_char('a');
        assert_eq!(
            out.actions,
            vec![Action::UpdatePreedit { text: "あ".into() }]
        );
    }

    #[test]
    fn select_picks_a_candidate_of_the_open_list() {
        let (mut bridge, _mock) = converting_kisha();
        let out = bridge.handle_select(4);
        assert!(out.consumed);
        assert!(out.actions.contains(&Action::UpdatePreedit {
            text: "記者の汽車".into()
        }));
        // in segment editing it picks the focused segment's candidate (the
        // rest keep the whole candidate picked before, 記者の汽車)
        bridge.handle_move_cursor(1);
        let out = bridge.handle_select(1);
        assert_eq!(segments_shown(&out.actions), (v(&["汽車の", "汽車"]), 0));
        // nothing to pick without a list
        let (mut idle, _mock) = bridge_with_mock();
        assert!(!idle.handle_select(0).consumed);
    }

    #[test]
    fn arrows_edit_the_conversion_segment_by_segment() {
        let (mut bridge, mock) = converting_kisha();
        // → enters segment editing on the first segment
        let out = bridge.handle_move_cursor(1);
        assert!(out.consumed);
        assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 0));
        let (list, selected, at) = segment_list(&out.actions);
        assert_eq!(list, v(&["記者の", "汽車の", "きしゃの", "キシャノ"]));
        assert_eq!((selected, at), (0, 0));
        // the focused segment's own reading is converted
        assert_eq!(mock.dispatched_readings().last().unwrap(), "きしゃの");
        // → again focuses the second; its window sits after 記者の
        let out = bridge.handle_move_cursor(1);
        assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 1));
        let (list, _, at) = segment_list(&out.actions);
        assert_eq!(list, v(&["記者", "汽車", "きしゃ", "キシャ"]));
        assert_eq!(at, 3);
        // Segment jobs are serialized; finishing the first starts the
        // now-focused second, with the first as its left context.
        assert_eq!(mock.answer_last(&["記者の"]).1, "きしゃの");
        bridge.apply_async_result();
        let (context, reading) = mock.answer_last(&["記者", "帰社"]);
        assert_eq!((context.as_str(), reading.as_str()), ("記者の", "きしゃ"));
        let out = bridge.apply_async_result();
        let (list, _, _) = segment_list(&out);
        assert_eq!(list, v(&["記者", "汽車", "きしゃ", "キシャ", "帰社"]));
        // ↓ picks 汽車, Enter commits both segments
        bridge.handle_down();
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "記者の汽車".into()
            }
        );
    }

    #[test]
    fn rapid_segment_moves_do_not_queue_overwritable_jobs() {
        let (mut bridge, mock) = converting_kisha();
        bridge.handle_move_cursor(1);
        let count = mock.state.dispatched.borrow().len();
        for _ in 0..3 {
            bridge.handle_move_cursor(1);
            bridge.handle_move_cursor(-1);
        }
        bridge.handle_move_cursor(1);
        assert_eq!(mock.state.dispatched.borrow().len(), count);
        assert_eq!(bridge.seg_requests.len(), 1);
        // Finishing the first job dispatches the currently focused segment.
        assert_eq!(mock.answer_last(&["記者の"]).1, "きしゃの");
        bridge.apply_async_result();
        assert_eq!(mock.answer_last(&["記者"]).1, "きしゃ");
        bridge.apply_async_result();
        assert!(bridge.seg_requests.is_empty());
    }

    #[test]
    fn a_deferred_segment_continuation_is_dispatched_after_the_current_job() {
        let (mut bridge, mock) = converting_kisha();
        bridge.handle_move_cursor(1);
        mock.answer_last(&["記者の"]);
        bridge.apply_async_result();
        bridge.handle_move_cursor(1); // second segment's conversion is running
        bridge.handle_move_cursor(-1);
        bridge.handle_home_end(true);
        bridge.handle_down(); // defer the first segment's continuation
        assert!(mock.state.continued.borrow().is_empty());
        assert_eq!(mock.answer_last(&["記者"]).1, "きしゃ");
        bridge.apply_async_result();
        assert_eq!(mock.state.continued.borrow().last().unwrap().1, 1);
        assert_eq!(mock.answer_last(&["記者の", "貴社の"]).1, "きしゃの");
        let (list, selected, _) = segment_list(&bridge.apply_async_result());
        assert_eq!(list[selected], "貴社の");
    }

    #[test]
    fn segment_editing_and_forms_keep_the_pending_tail_once() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kishanokishak");
        mock.set_aligned("きしゃのきしゃ", &["記者の記者"], &[&[0, 2, 3, 4, 6]]);
        bridge.apply_async_result();
        bridge.handle_space();
        assert_eq!(
            segments_shown(&bridge.handle_move_cursor(1).actions).0,
            v(&["記者の", "記者k"])
        );
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "記者の記者k".into()
            }
        );
        let (mut bridge, _) = converted_kasano();
        type_str(&mut bridge, "k");
        bridge.handle_convert_to(Form::HalfAlnum);
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "kasanok".into()
            }
        );
    }

    #[test]
    fn past_a_segments_last_candidate_come_more_not_the_first() {
        let (mut bridge, mock) = converting_kisha();
        bridge.handle_move_cursor(1);
        mock.answer_last(&["記者の", "帰社の"]);
        let (list, _, _) = segment_list(&bridge.apply_async_result());
        assert_eq!(list.len(), 5);
        for _ in 0..4 {
            bridge.handle_down();
        }
        // ↓ on the last asks for more of the segment's conversion and waits there
        let out = bridge.handle_down();
        assert_eq!(segment_list(&out.actions).1, 4);
        let (req, more) = *mock.state.continued.borrow().last().unwrap();
        assert_eq!(more, 1);
        let (context, reading) = mock.answer_last(&["記者の", "帰社の", "貴社の"]);
        assert_eq!((context.as_str(), reading.as_str()), ("", "きしゃの"));
        assert_eq!(mock.state.dispatched.borrow().last().unwrap().0, req);
        // the first new candidate is selected when it arrives
        let (list, selected, _) = segment_list(&bridge.apply_async_result());
        assert_eq!((list[5].as_str(), selected), ("貴社の", 5));
        // a continuation with nothing new: round to the first, as before
        bridge.handle_down();
        assert_eq!(mock.state.continued.borrow().last().unwrap().1, 2);
        mock.answer_last(&["記者の"]);
        assert_eq!(segment_list(&bridge.apply_async_result()).1, 0);
        // and from then on without asking again
        for _ in 0..5 {
            bridge.handle_down();
        }
        assert_eq!(segment_list(&bridge.handle_down().actions).1, 0);
        assert_eq!(mock.state.continued.borrow().len(), 2);
    }

    #[test]
    fn a_whole_sentence_list_still_goes_round() {
        let (mut bridge, mock) = converting_kisha();
        for _ in 0..5 {
            bridge.handle_down();
        }
        assert!(mock.state.continued.borrow().is_empty());
        assert!(bridge
            .handle_down()
            .actions
            .contains(&Action::UpdatePreedit {
                text: "汽車の記者".into()
            }));
    }

    #[test]
    fn rerank_order_adds_the_weighted_lm_score() {
        // 0: -1 - 0.5*30 = -16, 1: -2 - 0.5*20 = -12, 2: -3 - 0.5*18 = -12
        assert_eq!(
            rerank_order(&[-1.0, -2.0, -3.0], &[-30.0, -20.0, -18.0], 0.5, &[true; 3]),
            [1, 2, 0]
        );
        assert_eq!(
            rerank_order(&[-1.0, -2.0], &[-30.0, -20.0], 0.0, &[true; 2]),
            [0, 1]
        );
        // one that does not spell the reading stays below one that does
        assert_eq!(
            rerank_order(&[-1.0, -2.0], &[-30.0, -10.0], 0.5, &[true, false]),
            [0, 1]
        );
    }

    #[test]
    fn a_typing_pause_reranks_the_preedit_then_more_candidates() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
        let t0 = Instant::now();
        assert_eq!(
            bridge.apply_async_result(),
            vec![Action::UpdatePreedit {
                text: "傘の".into()
            }]
        );
        // the first 2, due once typing has paused
        let (cands, due) = rr.last();
        assert_eq!(cands, v(&["傘の", "笠の"]));
        assert!(due >= t0 + Duration::from_millis(100));
        // the LM prefers 笠の: -1.5 - 0.5*20 > -1.0 - 0.5*30
        rr.answer(&[-30.0, -20.0]);
        assert_eq!(
            bridge.apply_async_result(),
            vec![Action::UpdatePreedit {
                text: "笠の".into()
            }]
        );
        // then all three, at once
        let (cands, due) = rr.last();
        assert_eq!(cands, v(&["傘の", "笠の", "嵩の"]));
        assert!(due <= Instant::now());
        // Enter commits the reranked top
        assert_eq!(
            bridge.handle_enter().actions,
            vec![Action::Commit {
                text: "笠の".into()
            }]
        );
    }

    #[test]
    fn the_lexicon_scores_the_candidates_and_drops_the_near_misses() {
        let table = "続投\tぞくとう\t15.0\tnoun\n続\tぞく\t13.3\tpref\n頭\tとう\t11.0\tnoun\n\
                     続騰\tぞくとう\t17.0\tnoun\n";
        let path = std::env::temp_dir().join(format!(
            "setsuna-bridge-lexicon-{}.lexicon",
            std::process::id()
        ));
        std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_lexicon(&crate::config::LexiconCfg {
            file: Some(path.to_string_lossy().into_owned()),
        });
        type_str(&mut bridge, "zokutou");
        // the engine liked 続頭 and a kanji no word reads (続当) best
        answer_scored(
            &mock,
            &["続頭", "続当", "続投", "続騰"],
            &[-0.5, -0.6, -1.0, -1.5],
        );
        bridge.apply_async_result();
        let (list, _) = shown(&bridge.handle_space().actions).unwrap();
        // 続投 15.0 and 続騰 17.0 by words, 続頭 24.3 by a prefix and a noun:
        // 続投 first; 続当 (30 for 当) is more than MARGIN above 15.0 and goes,
        // 続頭 stays as the engine's top was
        assert_eq!(list[0], "続投");
        assert!(!list.contains(&"続当".to_string()), "{list:?}");
        assert!(list.contains(&"続頭".to_string()), "{list:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_candidate_the_reranker_scores_far_below_the_best_goes() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "傘ノ", "嵩の"], &[-1.0, -1.5, -4.0]);
        bridge.apply_async_result();
        bridge.handle_space();
        // 傘ノ: -1.5 - 0.5*70 is 21 below -1.0 - 0.5*30; 嵩の is not scored
        rr.answer(&[-30.0, -70.0]);
        bridge.apply_async_result();
        rr.answer(&[-30.0, -70.0, -35.0]);
        let (list, _) = shown(&bridge.apply_async_result()).unwrap();
        assert_eq!(list, v(&["傘の", "嵩の", "かさの", "カサノ"]));
    }

    #[test]
    fn typing_on_drops_the_rerank_and_a_late_result() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "笠の"], &[-1.0, -1.5]);
        bridge.apply_async_result();
        type_str(&mut bridge, "si");
        assert_eq!(*rr.state.cancels.borrow(), 1);
        rr.answer(&[-30.0, -20.0]);
        assert!(bridge.apply_async_result().is_empty());
    }

    #[test]
    fn the_conversion_key_reranks_at_once_and_keeps_a_moved_selection() {
        let (mut bridge, mock, rr) = bridge_with_reranker();
        type_str(&mut bridge, "kasano");
        answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
        bridge.apply_async_result();
        bridge.handle_space();
        let (_, due) = rr.last();
        assert!(due <= Instant::now());
        // the user moved to かさの before the result came: it stays selected
        bridge.handle_down();
        bridge.handle_down();
        rr.answer(&[-30.0, -20.0]);
        let out = bridge.apply_async_result();
        assert_eq!(
            out,
            vec![
                Action::UpdatePreedit {
                    text: "かさの".into()
                },
                Action::ShowCandidates {
                    list: v(&["笠の", "傘の", "かさの", "カサノ", "嵩の"]),
                    selected: 2,
                }
            ]
        );
    }

    #[test]
    fn shift_arrows_resize_and_reconvert() {
        let (mut bridge, mock) = converting_kisha();
        bridge.handle_move_cursor(1);
        // Shift+← gives の to the next segment: both start over as hiragana
        let out = bridge.handle_resize(-1);
        assert_eq!(
            segments_shown(&out.actions),
            (v(&["きしゃ", "のきしゃ"]), 0)
        );
        assert_eq!(mock.dispatched_readings().last().unwrap(), "きしゃ");
        mock.answer_last(&["汽車", "記者"]);
        let out = bridge.apply_async_result();
        assert_eq!(segments_shown(&out).0, v(&["汽車", "のきしゃ"]));
        // then the next stale segment, with the new first one as context
        let (context, reading) = mock.answer_last(&["の記者"]);
        assert_eq!((context.as_str(), reading.as_str()), ("汽車", "のきしゃ"));
        let out = bridge.apply_async_result();
        assert_eq!(segments_shown(&out).0, v(&["汽車", "の記者"]));
    }

    #[test]
    fn a_whole_sentence_result_is_ignored_while_editing_segments() {
        let (mut bridge, mock) = converting_kisha();
        bridge.handle_move_cursor(1);
        mock.set_result("きしゃのきしゃ", &["帰社の記者"]);
        assert!(bridge.apply_async_result().is_empty());
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit {
                text: "記者の記者".into()
            }
        );
    }

    #[test]
    fn a_segment_edit_is_learned_as_the_sentence_choice() {
        let dir = std::env::temp_dir().join(format!("setsuna-bridge-seg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("learning.jsonl");
        let (mut bridge, _mock) = converting_kisha();
        bridge.set_learning(Some(Learning::at(
            path.clone(),
            &crate::config::LearningCfg::default(),
        )));
        bridge.handle_move_cursor(1);
        bridge.handle_move_cursor(1);
        bridge.handle_down();
        bridge.handle_enter();
        let log = std::fs::read_to_string(&path).unwrap();
        assert!(log.contains("\"chosen\":\"記者の汽車\""), "{log}");
        assert!(log.contains("\"rejected\":\"記者の記者\""), "{log}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn commits(actions: &[Action]) -> Vec<String> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Commit { text } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    fn preedit(actions: &[Action]) -> Option<String> {
        actions.iter().rev().find_map(|a| match a {
            Action::UpdatePreedit { text } => Some(text.clone()),
            _ => None,
        })
    }

    /// `kasano` typed and converted, with `n` candidates (傘の, 笠の, …).
    fn converting_many(n: usize) -> (EngineBridge, MockClient) {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        let cands: Vec<String> = (0..n).map(|i| format!("候補{i}")).collect();
        let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
        mock.set_result("かさの", &refs);
        bridge.apply_async_result();
        bridge.handle_space();
        (bridge, mock)
    }

    #[test]
    fn function_keys_convert_the_composition_to_a_form() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "tokyo");
        mock.set_result("ときょ", &["時許"]);
        bridge.apply_async_result();
        let out = bridge.handle_convert_to(Form::Katakana);
        assert_eq!(preedit(&out.actions).as_deref(), Some("トキョ"));
        assert!(out.actions.contains(&Action::HideCandidates));
        let out = bridge.handle_convert_to(Form::HalfKatakana);
        assert_eq!(preedit(&out.actions).as_deref(), Some("ﾄｷｮ"));
        // F10 again and again cycles the case
        let out = bridge.handle_convert_to(Form::HalfAlnum);
        assert_eq!(preedit(&out.actions).as_deref(), Some("tokyo"));
        let out = bridge.handle_convert_to(Form::HalfAlnum);
        assert_eq!(preedit(&out.actions).as_deref(), Some("TOKYO"));
        let out = bridge.handle_convert_to(Form::HalfAlnum);
        assert_eq!(preedit(&out.actions).as_deref(), Some("Tokyo"));
        let out = bridge.handle_convert_to(Form::FullAlnum);
        assert_eq!(preedit(&out.actions).as_deref(), Some("ｔｏｋｙｏ"));
        // a late result of the reading does not replace the form
        mock.set_result("ときょ", &["時許"]);
        assert!(bridge.apply_async_result().is_empty());
        // Enter commits it, unlearned
        let out = bridge.handle_enter();
        assert_eq!(commits(&out.actions), vec!["ｔｏｋｙｏ"]);
        assert!(!bridge.is_active());
        // nothing composed: left to the app
        assert!(!bridge.handle_convert_to(Form::Katakana).consumed);
    }

    #[test]
    fn escape_and_space_leave_a_form_for_the_conversion() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        bridge.handle_convert_to(Form::Katakana);
        // Escape: back to typing, the conversion still what Enter commits
        let out = bridge.handle_escape();
        assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
        bridge.handle_convert_to(Form::Katakana);
        // Space: the usual candidate window
        let out = bridge.handle_space();
        assert!(out.actions.iter().any(|a| matches!(
            a,
            Action::ShowCandidates { list, .. } if list[0] == "傘の"
        )));
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の"]);
    }

    #[test]
    fn shift_letters_convert_with_function_keys_too() {
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.handle_shifted_alpha('N');
        type_str(&mut bridge, "eko");
        let out = bridge.handle_convert_to(Form::Hiragana);
        assert_eq!(preedit(&out.actions).as_deref(), Some("ねこ"));
    }

    #[test]
    fn digits_pick_from_the_open_page_when_enabled() {
        let (mut bridge, _mock) = converting_many(15);
        // off by default: a digit commits and starts typing
        let out = bridge.handle_char('2');
        assert_eq!(commits(&out.actions), vec!["候補0"]);
        bridge.handle_escape();

        let (mut bridge, _mock) = converting_many(15);
        bridge.set_candidate_paging(9, true);
        // the reading and its katakana are 3rd and 4th
        let out = bridge.handle_char('5');
        assert_eq!(commits(&out.actions), vec!["候補2"]);
        assert!(!bridge.is_active());

        // on the second page, 1 is the page's first
        let (mut bridge, _mock) = converting_many(15);
        bridge.set_candidate_paging(9, true);
        bridge.handle_page_down();
        let out = bridge.handle_char('1');
        assert_eq!(commits(&out.actions), vec!["候補7"]);

        // past the list: nothing
        let (mut bridge, _mock) = converting_many(3);
        bridge.set_candidate_paging(9, true);
        let out = bridge.handle_char('6');
        assert!(out.consumed && out.actions.is_empty());
        // typing digits is unaffected
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.set_candidate_paging(9, true);
        type_str(&mut bridge, "1");
        assert!(bridge.is_active());
    }

    #[test]
    fn page_keys_step_by_the_window_page() {
        // 候補0, 候補1, かさの, カサノ, 候補2, …, 候補11: 14 candidates
        let (mut bridge, _mock) = converting_many(12);
        bridge.set_candidate_paging(5, false);
        bridge.handle_down();
        let out = bridge.handle_page_down();
        assert_eq!(preedit(&out.actions).as_deref(), Some("候補4"));
        // the last page is short: its last candidate
        bridge.handle_down();
        bridge.handle_down();
        bridge.handle_down();
        let out = bridge.handle_page_down();
        assert_eq!(preedit(&out.actions).as_deref(), Some("候補11"));
        // and it wraps
        let out = bridge.handle_page_down();
        assert_eq!(preedit(&out.actions).as_deref(), Some("カサノ"));
        let out = bridge.handle_home_end(true);
        assert_eq!(preedit(&out.actions).as_deref(), Some("候補11"));
        let out = bridge.handle_home_end(false);
        assert_eq!(preedit(&out.actions).as_deref(), Some("候補0"));
    }

    #[test]
    fn tab_opens_then_steps_through_the_candidates() {
        let (mut bridge, mock) = bridge_with_mock();
        assert!(!bridge.handle_tab(false).consumed);
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        let out = bridge.handle_tab(false);
        assert!(bridge.is_converting());
        assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
        let out = bridge.handle_tab(false);
        assert_eq!(preedit(&out.actions).as_deref(), Some("笠の"));
        let out = bridge.handle_tab(true);
        assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
    }

    #[test]
    fn caret_edits_show_and_commit_the_kana() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の"]);
        bridge.apply_async_result();
        assert_eq!(bridge.caret(), None);
        let out = bridge.handle_home_end(false);
        assert_eq!(preedit(&out.actions).as_deref(), Some("かさの"));
        assert_eq!(bridge.caret(), Some(0));
        let out = bridge.handle_delete();
        assert_eq!(preedit(&out.actions).as_deref(), Some("さの"));
        bridge.handle_move_cursor(1);
        assert_eq!(bridge.caret(), Some(1));
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["さの"]);
        // Delete of the last char ends the composition
        type_str(&mut bridge, "a");
        bridge.handle_move_cursor(-1);
        let out = bridge.handle_delete();
        assert_eq!(out.actions, vec![Action::EndComposition]);
        assert!(!bridge.is_active());
    }

    #[test]
    fn space_widths() {
        let (mut bridge, _mock) = bridge_with_mock();
        assert!(!bridge.handle_space().consumed);
        assert_eq!(
            commits(&bridge.handle_shift_space().actions),
            vec!["\u{3000}"]
        );
        let cfg = crate::config::FrontendCfg {
            full_width_space: true,
            ..Default::default()
        };
        let mut bridge = EngineBridge::with_frontend(None, &cfg);
        assert_eq!(commits(&bridge.handle_space().actions), vec!["\u{3000}"]);
        assert!(!bridge.handle_shift_space().consumed);
        // composing: Shift+Space converts like Space
        type_str(&mut bridge, "ka");
        bridge.handle_shift_space();
        assert!(bridge.is_converting());
    }

    #[test]
    fn without_live_conversion_the_kana_stays_and_conversions_are_suggested() {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_live_conversion(false);
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        let actions = bridge.apply_async_result();
        assert_eq!(preedit(&actions).as_deref(), Some("かさの"));
        assert_eq!(bridge.suggestions()[..2], ["傘の", "笠の"]);
        // a stale list (for a shorter reading) is not suggested
        type_str(&mut bridge, "k");
        assert!(
            !bridge.suggestions().is_empty(),
            "pending romaji keeps them"
        );
        type_str(&mut bridge, "a");
        assert!(bridge.suggestions().is_empty());
        bridge.handle_backspace();
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        // Enter commits the kana shown
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["かさの"]);

        // a click on a suggestion opens the conversion on it
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        let out = bridge.handle_select(1);
        assert!(bridge.is_converting());
        assert_eq!(preedit(&out.actions).as_deref(), Some("笠の"));
        assert!(bridge.suggestions().is_empty());
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["笠の"]);

        // ↓ opens it too
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の"]);
        bridge.apply_async_result();
        bridge.handle_down();
        assert!(bridge.is_converting());
    }

    #[test]
    fn live_conversion_suggests_nothing() {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["傘の"]);
        let actions = bridge.apply_async_result();
        assert_eq!(preedit(&actions).as_deref(), Some("傘の"));
        assert!(bridge.suggestions().is_empty());
    }

    #[test]
    fn no_client_degrades_to_raw_hiragana() {
        // With no backend the bridge must still accept input and render a
        // raw-kana preedit rather than panic.
        let mut bridge = EngineBridge::new(None);
        type_str(&mut bridge, "ka");
        let outcome = bridge.handle_char('n');
        assert!(outcome.consumed);
        assert_eq!(
            outcome.actions,
            vec![Action::UpdatePreedit {
                text: "かn".into()
            }]
        );
    }
}
