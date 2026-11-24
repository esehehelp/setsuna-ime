//! Internal state structs, configuration, and constants for [`super::EngineBridge`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Local};

use crate::char_forms::Form;
use crate::postprocess::CharRules;
use crate::segmented::Segmented;

use super::{Action, KeyOutcome};

/// Without a reading lexicon (or with a partial one, `Lexicon::partial`), the
/// engine's candidates more than this below its top (log-probability) go: the lexicon drops what reads otherwise than the
/// reading (今陽, 況日, 日 for きょう), and without it they filled the list
/// (user 2026-11-10, the phone without the dictionaries: "今日 日 今陽 昨日
/// 今ウ 況日 京日 のような壊れた候補が多い"). Of the golds among
/// Cymbidium-v1.0-xsmall's 20 candidates, 99.47% (wiki_fresh_v2) / 98.35%
/// (AJIMEE) stay, of what the lexicon would drop 13% (at 8: 99.77 / 98.90,
/// 29%), the list 5.3 long; Bletilla-v0.1-small alike.
pub(super) const NO_LEXICON_MARGIN: f32 = 6.0;

/// Candidates of the other reading offered (its first `ALT_EARLY` right
/// after the first `ALT_EARLY` of the mixed one), besides its user
/// dictionary words.
pub(super) const ALT_MAX: usize = 5;
pub(super) const ALT_EARLY: usize = 3;

/// Predictions offered while typing.
pub(super) const PREDICTIONS: usize = 3;

/// How long segment editing waits for the engine's analysis of the
/// conversion it cuts (`bunsetsu`) before cutting it by the char alignment.
pub(super) const ANALYSIS_WAIT: Duration = Duration::from_millis(150);

/// How long Enter waits for the conversion of what is typed when it is not
/// in yet (`await_live_conversion`): a phone without SDOT takes 100-300 ms
/// a conversion, and one may be in flight before it.
pub(super) const ENTER_WAIT: Duration = Duration::from_millis(1000);

/// How long a key waits for the conversion it asked for before the frontend
/// draws it (`EngineBridge::settle_live`): the engine answers in about 1 ms
/// and an edit of a TSF host's text takes 3 to 8, so one edit with the
/// conversion shows it sooner than the kana and then it.
pub const LIVE_SETTLE: Duration = Duration::from_millis(4);

/// How long Space waits for the other reading's list (`await_alt`) before
/// the list opens without it: it is added when it comes, the selection
/// staying where it is (`apply_alt_result`), so a candidate the user's
/// choices put first is first but not chosen. One conversion of
/// Cymbidium-v1.1-xsmall, warm, not cached, measured 2026-11-19 on a 4-vCPU
/// Xeon at 2.1 GHz with AVX-512 VNNI, median of 30 (max): 2.6 ms (IMEの),
/// 5.8 ms a reading of 18 chars, 10.4 (11.2) of 42, 17.7 (18.6) of 80; v1.0
/// takes 33 ms a sentence on a CPU with none of its SIMD kernels (README).
/// User 2026-11-19: 20 ms.
pub(super) const ALT_WAIT: Duration = Duration::from_millis(20);

/// How far below the reranked best (the engine's score plus lambda times the
/// LM's log-probability) a scored candidate may be and stay in the list
/// (user 2026-10-23: candidates as clean as Microsoft IME's). Most of the
/// model's near-misses (でルタ, eルタ and デル多 for でるた, 続頭 for ぞくとう)
/// score far below: with sarashina2.2's SentencePiece scores at lambda 0.8,
/// over the engine's top 20 of wiki_fresh_v2 and AJIMEE 20 keeps 49% / 57%
/// of the candidates and every gold among them, 15 loses one.
pub(super) const RERANK_GAP: f64 = 20.0;

pub(super) const CONTEXT_WINDOW: usize = 80;

/// Mixed input while converting: the list offers the reading read with
/// English (`main`, the engine's list for it) and with every letter as
/// romaji (`candidates`, asked for once `main` is in), then the input as
/// typed. Ending in English, its top candidate with a space comes first:
/// the next key commits it, Space again goes on to the Japanese ones.
pub(super) struct AltList {
    /// `None` when every letter is read as romaji already, or some letter
    /// reads as none (`google` → ごおgぇ: its conversions are noise).
    pub(super) reading: Option<String>,
    pub(super) req_id: Option<u64>,
    /// Space waited for its list once (`await_alt`): never again, even
    /// when it did not come (a later request took its place).
    pub(super) awaited: bool,
    pub(super) candidates: Vec<String>,
    pub(super) main: Vec<String>,
    pub(super) ends_english: bool,
    pub(super) typed: String,
    /// The input is one word the model knows as Latin (`made` read まで):
    /// offer it as typed among the first candidates.
    pub(super) typed_early: bool,
    /// The user dictionary's words of the other reading as a whole: they
    /// come first of its candidates, and all of them (`ALT_MAX` counts the
    /// others).
    pub(super) user_words: Vec<String>,
}

/// The engine's top of what is typed, waiting for the reranked one
/// (`[rerank] defer_ms`).
pub(super) struct Deferred {
    pub(super) reading: String,
    pub(super) top: String,
    pub(super) until: Instant,
}

/// Where the user dictionary comes from.
pub(super) enum UserDictSource {
    None,
    /// Given (tests, the demo).
    Fixed(Arc<crate::user_dict::UserDict>),
    /// The file in the config directory, read again when it changes.
    File,
}

pub(super) enum Phase {
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

/// A rewrite of the whole reading's conversion.
pub(super) struct RewriteJob {
    pub(super) req_id: u64,
    pub(super) reading: String,
    /// The rewrite, once in.
    pub(super) text: Option<String>,
}

/// A whole-reading conversion under rerank: the engine's candidates as it
/// sent them, the first `k` of which are being (or were) scored by the LM.
pub(super) struct RerankJob {
    pub(super) reading: String,
    /// The engine's reading corrector's reading (`EngineResult::corrected_reading`).
    pub(super) corrected_reading: Option<String>,
    pub(super) context: String,
    pub(super) raw: Vec<String>,
    pub(super) scores: Vec<f32>,
    /// Per candidate, whether it spells the reading out
    /// (`order_engine_result`): those stay above the others when reranked.
    pub(super) spelled: Vec<bool>,
    pub(super) reading_pos: Vec<Vec<u32>>,
    pub(super) k: usize,
    pub(super) req_id: u64,
    /// When the request for the current `k` is due.
    pub(super) due: Instant,
}

/// An F6–F10 conversion on screen: the form, the press count (F9 / F10
/// cycle the case), the text and the candidates it stands in for, which
/// Space / Escape return to.
pub(super) struct FormShown {
    pub(super) form: Form,
    pub(super) case: usize,
    pub(super) text: String,
    pub(super) list: Vec<String>,
}

/// Per-composition transient state reset together whenever a composition ends
/// (commit, cancel, discard, or backspacing to empty). Keeping every
/// composition-scoped field here lets [`Self::clear`] reset the whole struct
/// via `Default`, preventing stale state from leaking into the next
/// composition.
#[derive(Default)]
pub(super) struct CompositionState {
    /// Reading that the current `Phase::Composing.candidates` matches.
    /// Prevents redundant dispatches while the user types mid-kana and the
    /// hiragana portion is unchanged.
    pub(super) last_applied_reading: String,
    /// Kanji rendering of the last successfully-converted **prefix** of
    /// hiragana (the reading of the last applied result). Used to render the
    /// preedit as `cached_kanji + raw_tail + pending` instantly on every
    /// keystroke — the user's just-typed characters stay visible as
    /// hiragana while earlier chars settle into kanji once the engine
    /// returns.
    pub(super) cached_prefix_reading: String,
    pub(super) cached_prefix_kanji: String,
    /// A shortcut word of the user dictionary (短縮よみ) on top of the
    /// conversion of `cached_prefix_reading`: shown while the reading is
    /// that one only, `cached_prefix_kanji` standing for it as the start of
    /// a longer reading (`cache_conversion`).
    pub(super) cached_shortcut: Option<String>,
    /// Clock snapshot shared by local candidates and delayed model results.
    pub(super) date_time_now: Option<DateTime<Local>>,
    /// The predictions open as the candidate list (Tab while typing).
    pub(super) prediction: Option<Vec<crate::predict::Prediction>>,
    /// The open list stands for all that is typed, pending romaji too: the
    /// user dictionary's words of a reading of letters (`w`: (笑)).
    pub(super) covers_typed: bool,
    /// The engine's top waiting for the reranked one.
    pub(super) deferred: Option<Deferred>,
    /// Reading the current candidate list (Composing / Converting) was
    /// produced for; empty for a provisional list. A choice is learned
    /// against it.
    pub(super) list_reading: String,
    /// Reading start of each char of the current list's candidates, by
    /// candidate text (`segments::char_starts`): what segment editing cuts
    /// them with.
    pub(super) list_starts: HashMap<String, Vec<usize>>,
    /// Segment conversions in flight: (request id, segment index, its
    /// `continue_beam`: 0 for the segment's own conversion, whether it is
    /// the conversion of the segment's letters read as romaji).
    pub(super) seg_requests: Vec<(u64, usize, u32, bool)>,
    /// Escape showed the kana typed in place of the conversion: Escape
    /// again clears, Enter commits the kana; typing on converts again.
    pub(super) kana_shown: bool,
    /// What Backspace left of a conversion shown, with its reading and its
    /// chars' reading starts: the list of that reading keeps it first, so
    /// the preview stays as it was (until the reading changes), with those
    /// starts.
    pub(super) kept: Option<(String, String, Vec<usize>)>,
    /// The last F6–F10 conversion, while its text is what Converting shows.
    pub(super) form: Option<FormShown>,
    /// The last F6–F10 conversion of a segment: (segment, form, press
    /// count, text), while that segment shows it.
    pub(super) seg_form: Option<(usize, Form, usize, String)>,
    /// The whole-reading conversion being reranked.
    pub(super) rerank: Option<RerankJob>,
    /// Mixed input: the other reading of the conversion open.
    pub(super) alt: Option<AltList>,
    /// The rewrite of the conversion open, asked for or in.
    pub(super) rewrite: Option<RewriteJob>,
    /// The engine's result (as it came) whose list the preview went without
    /// the lexicon's costs for (`apply_engine_result`): its whole list is
    /// made when wanted (`finish_list_now`).
    pub(super) unfinished: Option<crate::engine_client::EngineResult>,
    /// The live list's reading and the model's first, when a candidate
    /// ending in kana went before it (`hold_noun_end`): Space puts it back
    /// first.
    pub(super) end_held: Option<(String, String)>,
    /// The reading the last live conversion was asked for (`dispatch_live`):
    /// what `settle_live` waits for.
    pub(super) live_out: Option<String>,
    /// Space opened the whole sentence's list before the engine's list of
    /// it came, where it opens the last segment's (`on_last_segment`): it
    /// does so once the list comes.
    pub(super) last_segment_due: bool,
}

impl CompositionState {
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(super) fn clear_preview_cache(&mut self) {
        self.cached_prefix_reading.clear();
        self.cached_prefix_kanji.clear();
        self.cached_shortcut = None;
    }
}

/// Surrounding text and app-field state kept across compositions.
#[derive(Default)]
pub(super) struct ContextState {
    /// The text before the caret as read or committed (`CONTEXT_WINDOW`
    /// chars): what decides whether the rewriter may see it.
    pub(super) raw_context: String,
    /// What the models read of it: the text after anything sensitive in it
    /// (`privacy::after_last_hit`: a URL, a token, パスワード).
    pub(super) context: String,
    /// The last commit: what the next one is predicted after.
    pub(super) last_commit: String,
    /// Whether the caret is at a line's start or after a space, when the
    /// app's text says (`None`: unknown).
    pub(super) at_word_start: Option<bool>,
    /// A command's name being typed through to the app.
    pub(super) passing: bool,
    /// The field typed into is private (a password, a PIN, an input scope
    /// marked private): its text is no context, nothing is rewritten and
    /// nothing learned (learned choices still apply).
    pub(super) private_field: bool,
}

/// Frontend and runtime knobs applied to conversions and key handling.
pub(super) struct BridgeConfig {
    pub(super) live_min_dispatch_chars: usize,
    pub(super) force_convert_on_space: bool,
    pub(super) symbol_width: crate::config::SymbolWidth,
    /// Punctuation / middle-dot / replacement rules of the frontend config.
    pub(super) char_rules: CharRules,
    pub(super) drop_untyped_symbols: bool,
    pub(super) date_time_enabled: bool,
    /// Number candidates (`numbers`).
    pub(super) numbers_enabled: bool,
    /// Typo correction (`typo`).
    pub(super) typo_correction: bool,
    /// Emoji and kaomoji candidates (`emoji`).
    pub(super) emoji_enabled: bool,
    /// With mixed input (`[frontend] mixed_model`): the ASCII char each
    /// `[frontend.symbols]` output was typed as. Typed symbols go into the
    /// composition as ASCII, shown per span.
    pub(super) mixed_symbols: HashMap<String, char>,
    /// `/command` and the like go to the app (`pass_through_commands`).
    pub(super) pass_commands: bool,
    /// Candidates per page of the frontend's window: what PageUp / PageDown
    /// step by and what the digit keys pick from.
    pub(super) page_size: usize,
    /// The digit keys pick a candidate of the open page (1–9, 0 = 10th)
    /// instead of typing; for a window that labels its candidates.
    pub(super) digit_select: bool,
    pub(super) full_width_space: bool,
    /// `[frontend] space_converts_last_segment`.
    pub(super) space_last_segment: bool,
    /// `[frontend] live_noun_end_penalty`.
    pub(super) noun_end_penalty: f32,
    /// `[frontend] sentence_case`: the punctuation typed after an English
    /// sentence is English (`mixed::english_after`).
    pub(super) sentence_case: bool,
    /// Typing shows the conversion in the preedit (else the kana, with the
    /// conversions offered by [`super::EngineBridge::suggestions`]).
    pub(super) live_conversion: bool,
    /// `[frontend] backspace_deletes_shown` (`backspace_shown`).
    pub(super) backspace_deletes_shown: bool,
    /// `[frontend] escape_to_kana` (`handle_escape`).
    pub(super) escape_to_kana: bool,
    pub(super) rerank_cfg: crate::config::RerankCfg,
    /// Rewrite mode: the conversion key also asks for a rewrite.
    pub(super) rewrite_mode: bool,
}

impl BridgeConfig {
    pub(super) fn from_frontend(frontend: &crate::config::FrontendCfg) -> Self {
        Self {
            live_min_dispatch_chars: frontend.live_min_dispatch_chars,
            force_convert_on_space: frontend.force_convert_on_space,
            symbol_width: frontend.symbol_width,
            char_rules: CharRules::from_frontend(frontend),
            drop_untyped_symbols: frontend.drop_untyped_symbols,
            date_time_enabled: frontend.date_time_candidates,
            numbers_enabled: frontend.number_candidates,
            typo_correction: frontend.typo_correction,
            emoji_enabled: frontend.emoji_candidates,
            mixed_symbols: HashMap::new(),
            pass_commands: frontend.pass_through_commands,
            page_size: 10,
            digit_select: false,
            full_width_space: frontend.full_width_space,
            space_last_segment: frontend.space_converts_last_segment,
            noun_end_penalty: frontend.live_noun_end_penalty,
            sentence_case: frontend.sentence_case,
            live_conversion: frontend.live_conversion.unwrap_or(true),
            backspace_deletes_shown: frontend.backspace_deletes_shown,
            escape_to_kana: frontend.escape_to_kana,
            rerank_cfg: crate::config::RerankCfg::default(),
            rewrite_mode: false,
        }
    }
}

/// The status while a rewrite is on its way.
pub(super) fn rewriting() -> Action {
    Action::Status {
        text: "書き直し中…".to_string(),
    }
}

pub(super) fn consumed(actions: Vec<Action>) -> KeyOutcome {
    KeyOutcome {
        consumed: true,
        actions,
    }
}

pub(super) fn not_consumed() -> KeyOutcome {
    KeyOutcome {
        consumed: false,
        actions: vec![],
    }
}
