//! Runtime configuration loaded from `<config_dir>/config.toml`
//! (`%APPDATA%\setsuna-ime` on Windows, `~/.config/setsuna-ime` on Linux).
//!
//! Why a config module
//! -------------------
//! Without it every tunable of the IME would be hardcoded — daemon
//! beam_width, the polluted-vocab blacklist, the candidate window font,
//! the engine binary path — and every change would need a rebuild and
//! a reinstall. With Config:
//!   - the Windows IME reads the file again when an input starts after it
//!     was saved (and at activation); fcitx5 at activation
//!   - unset engine knobs are not filled in here: the daemon takes them from
//!     the model manifest's `decode_defaults`, then from its own defaults,
//!     so the recommended decode recipe travels with the model
//!   - missing UI / frontend fields fall back to defaults (`#[serde(default)]`)
//!
//! Path resolution
//! ---------------
//! `Config::load_default()` reads `%APPDATA%\setsuna-ime\config.toml` (Windows
//! Roaming AppData). When the file is missing or unreadable, a default
//! `Config` is returned and the daemon proceeds normally. The DLL's
//! `ActivateEx` call site logs the resolved path and parse result via
//! `tracing` so behavior is debuggable.
//!
//! Default template
//! ----------------
//! `Config::emit_default_toml` writes the commented template, keeping the
//! format documented in one place; a test checks it parses back.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Top-level config root. All sections carry their own `#[serde(default)]`
/// derives so a `[engine]` block with only `beam_width = 100` leaves every
/// other field at its default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub engine: EngineCfg,
    pub ui: UiCfg,
    pub frontend: FrontendCfg,
    pub learning: LearningCfg,
    pub rerank: RerankCfg,
    pub ngram: NgramCfg,
    pub lexicon: LexiconCfg,
    pub rewriter: RewriterCfg,
    /// Settings for one application each (`[[apps]]`), over the others.
    pub apps: Vec<AppCfg>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine: EngineCfg::default(),
            ui: UiCfg::default(),
            frontend: FrontendCfg::default(),
            learning: LearningCfg::default(),
            rerank: RerankCfg::default(),
            ngram: NgramCfg::default(),
            lexicon: LexiconCfg::default(),
            rewriter: RewriterCfg::default(),
            apps: Vec::new(),
        }
    }
}

/// How the IME starts in an application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppStart {
    /// As the thread's on / off state says.
    #[default]
    Keep,
    /// On (hiragana).
    On,
    /// Off (direct input: a game, a terminal).
    Off,
}

/// One application's settings (`[[apps]]`): its process (the executable's
/// name, `code.exe`; case and `.exe` do not matter) and what it changes.
/// An unset field keeps the setting of the rest of the config.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppCfg {
    pub process: String,
    pub start: AppStart,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_conversion: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewrite: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed_input: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<bool>,
}

impl AppCfg {
    /// Whether this entry is the application `process` (a path or a name).
    pub fn matches(&self, process: &str) -> bool {
        let stem = |s: &str| {
            let name = s
                .rsplit(['\\', '/'])
                .next()
                .unwrap_or(s)
                .to_ascii_lowercase();
            name.strip_suffix(".exe")
                .map(str::to_string)
                .unwrap_or(name)
        };
        !self.process.trim().is_empty() && stem(self.process.trim()) == stem(process)
    }
}

impl Config {
    /// This config as the application `process` uses it: its `[[apps]]`
    /// entry's settings over the rest, and the entry itself.
    pub fn for_app(&self, process: &str) -> (Config, Option<AppCfg>) {
        let Some(app) = self.apps.iter().find(|a| a.matches(process)).cloned() else {
            return (self.clone(), None);
        };
        let mut cfg = self.clone();
        if let Some(v) = app.live_conversion {
            cfg.frontend.live_conversion = Some(v);
        }
        if let Some(v) = app.learning {
            cfg.learning.enabled = v;
        }
        if let Some(v) = app.rewrite {
            cfg.rewriter.enabled = v;
        }
        if app.mixed_input == Some(false) {
            // empty is off
            cfg.frontend.mixed_model = Some(String::new());
        }
        if let Some(v) = app.prediction {
            cfg.frontend.prediction = v;
        }
        (cfg, Some(app))
    }
}

/// Engine section. Every decode knob is optional: a knob set here is passed
/// to the daemon as an explicit value; an unset one falls back to the model
/// manifest's `decode_defaults`, then to the engine default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineCfg {
    /// Model artifact manifest (`*.manifest.json`). Wins over `model_dir`
    /// and automatic discovery.
    pub model_manifest: Option<PathBuf>,
    /// Directory holding exactly one `*.manifest.json`.
    pub model_dir: Option<PathBuf>,
    /// Optional explicit `setsuna-engine.exe` path. Empty = next to the DLL.
    pub engine_exe: Option<PathBuf>,
    /// ONNX Runtime library the engine loads (`--ort-dylib`). Empty = the
    /// engine's default, `onnxruntime.dll` next to it.
    pub ort_dylib: Option<PathBuf>,
    /// Prefix-beam width (1 = greedy).
    pub beam_width: Option<usize>,
    /// Number of candidates returned per request.
    pub top_k: Option<usize>,
    /// Per-step expansion breadth in the prefix-beam loop.
    pub top_k_per_step: Option<usize>,
    /// Maximum left-context characters fed to the model.
    pub max_context: Option<usize>,
    /// Split bracket / annotation symbols out of the neural decoding span.
    pub symbol_split: Option<bool>,
    /// Read each run of symbols in the reading as one token (models trained
    /// on rows with symbol tokens); wins over `symbol_split`.
    pub symbol_token: Option<bool>,
    /// Cut typed numbers out of the reading and splice them back verbatim
    /// (models whose training readings spell numbers in kana).
    pub digit_split: Option<bool>,
    /// Token IDs to forbid in the proposal output. `[]` clears a manifest's
    /// blacklist.
    pub vocab_blacklist: Option<Vec<u32>>,
    /// Ask the engine for its output frame by frame (`--emit-frames`, this
    /// many tokens per frame), for inspection tools such as the demo. Not
    /// for the IME: it only costs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emit_frames: Option<usize>,
    /// Analyzer manifest the engine loads (`--analyzer`): segment editing
    /// cuts a conversion by its words (`bunsetsu`). Empty = none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analyzer_manifest: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiCfg {
    pub candidate_window: CandidateWindowCfg,
}

impl Default for UiCfg {
    fn default() -> Self {
        Self {
            candidate_window: CandidateWindowCfg::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CandidateWindowCfg {
    /// `auto` (`setsuna` or `setsuna-dark` as Windows' app mode is light or
    /// dark), `setsuna` (light panel, amber-to-violet accent),
    /// `setsuna-dark`, or `classic` (flat, in the colors below).
    pub theme: String,
    /// The window's size, in percent (80-200).
    pub size_percent: u32,
    /// The panel's corner radius at 96 DPI (0-24).
    pub corner_radius: u32,
    /// The window's opacity, in percent (20-100).
    pub opacity_percent: u32,
    /// The candidates' font weight (100-900: 400 normal, 700 bold).
    pub font_weight: u32,
    pub font_face: String,
    /// Candidate text size in pixels at 96 DPI.
    pub font_pt: u32,
    /// Candidates per page (1-10: the digit keys pick them).
    pub page_size: u32,
    /// Foreground / background pairs as 0xRRGGBB hex strings (parsed via
    /// `parse_color`). Stored as String for human-friendliness in TOML.
    pub highlight_color: String,
    pub highlight_text_color: String,
    pub bg_color: String,
    pub border_color: String,
    pub footer_color: String,
}

impl Default for CandidateWindowCfg {
    fn default() -> Self {
        // The colors are the `classic` theme's; the others have their own.
        Self {
            theme: "auto".to_string(),
            size_percent: 100,
            corner_radius: 12,
            opacity_percent: 100,
            font_weight: 400,
            font_face: "Meiryo UI".to_string(),
            font_pt: 16,
            page_size: 10,
            highlight_color: "#D77800".to_string(),
            highlight_text_color: "#FFFFFF".to_string(),
            bg_color: "#FFFFFF".to_string(),
            border_color: "#808080".to_string(),
            footer_color: "#707070".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrontendCfg {
    /// Minimum hiragana length before automatic live conversion dispatch
    /// while typing. Explicit Space conversion can bypass this via
    /// `force_convert_on_space`.
    pub live_min_dispatch_chars: usize,
    /// When true, Space dispatches conversion even below
    /// `live_min_dispatch_chars`, so short readings can use dictionary/model
    /// candidates instead of a single provisional raw-kana candidate.
    pub force_convert_on_space: bool,
    /// Virtual-key symbol outputs keyed by stable names such as
    /// `slash`, `left_bracket`, `shift_8`. Values are inserted into the
    /// composing buffer exactly as written.
    pub symbols: BTreeMap<String, String>,
    /// Width of the symbols typed into a reading, in the candidates: the
    /// engine sees the reading NFKC-normalized, so a typed `（` comes back as
    /// `(`; `as_typed` restores what was typed, `full` / `half` force a width,
    /// `auto` picks the width of digits, letters and symbols from the text
    /// before the caret (`postprocess::auto_width`).
    pub symbol_width: SymbolWidth,
    /// Touten / kuten style of the candidates (`、。` / `，．` / …).
    /// `as_is` leaves them as the engine wrote them.
    pub punctuation: PunctuationStyle,
    /// Write every middle-dot variant (`･` `·` `‧` `⋅` `∙`) as `・`.
    pub normalize_middle_dot: bool,
    /// Literal replacements applied to every candidate, in order, after the
    /// rules above.
    pub replacements: Vec<Replacement>,
    /// Drop candidates with a symbol (punctuation, brackets, …) the reading
    /// does not have (`postprocess::drop_untyped_symbols`).
    pub drop_untyped_symbols: bool,
    /// Offer symbols for readings that name them (やじるし -> →, `symbol_dict`).
    pub symbol_dictionary: bool,
    /// A larger symbol dictionary merged into the built-in one: JSON lines
    /// with `reading` and `symbol` (the distributed symbol dictionary).
    /// Unset: the built-in table only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_dictionary_file: Option<String>,
    /// Generate dates and times from readings such as きょう / いま /
    /// らいしゅう / すいようび / 1じかんご, using the local clock.
    pub date_time_candidates: bool,
    /// Number candidates (`numbers`): the forms of digits (1,234, 千二百
    /// 三十四, 12:34), numbers read in kana, era years, calculations
    /// (`1+2=`), `U+3042`.
    pub number_candidates: bool,
    /// Emoji and kaomoji candidates (`emoji`): にこにこ -> 😊 …, えもじ /
    /// かおもじ list them all.
    pub emoji_candidates: bool,
    /// Prediction from what was committed (`predict`): Tab while typing a
    /// reading's start opens the longer conversions committed for it (no
    /// window comes up by itself).
    pub prediction: bool,
    /// The romaji table: "standard" (built in), "azik", "kana_jis" (kana
    /// input on the JIS layout) or the path of a table in Mozc's format.
    pub romaji_table: String,
    /// Typo correction (`typo`): where a letter reads as no kana, the
    /// reading one edit away (a neighbouring key, a key too many, two keys
    /// swapped) is converted too, its candidates after the first ones.
    pub typo_correction: bool,
    /// `/`, `$` or `@` typed at a line's start or after a space, and the
    /// name after it, go to the app as typed (an AI agent's `/command`,
    /// `$skill`, `@file`, whose completion the app offers). Only where the
    /// text before the caret can be read.
    pub pass_through_commands: bool,
    /// Space with nothing composed types a full-width space (U+3000), as
    /// MS-IME and Mozc do in hiragana mode; false leaves it to the app (a
    /// half-width space). Shift+Space types the other width.
    pub full_width_space: bool,
    /// While typing, show the conversion in the text (true) or keep the
    /// typed kana there and list the conversions in a suggestion window
    /// (false, as Mozc does). Unset: true on every frontend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_conversion: Option<bool>,
    /// With live conversion, Backspace on a conversion shown deletes its
    /// last char and the reading under it, the rest staying as shown (true;
    /// user 2026-10-28: delete what is seen), or the last kana of the
    /// reading, the rest converted again (false).
    pub backspace_deletes_shown: bool,
    /// Escape steps back as MS-IME's does: on a conversion shown (live or
    /// in the candidate window) it shows the kana typed, and Escape again
    /// clears the composition (true); false goes from the candidate window
    /// back to the live conversion and clears from there.
    pub escape_to_kana: bool,
    /// Keep the preedit caret at the start of the composition (fcitx5-mozc's
    /// "preedit cursor position at beginning"). Apps put the candidate
    /// window at the caret, so it then stays put while typing, converting
    /// and moving between segments; some draw the caret there too. A caret
    /// moved with ← / → is still shown. fcitx5 only; off by default (user
    /// 2026-10-20: the caret seemed stuck at the start of the sentence).
    pub preedit_cursor_at_start: bool,
    /// Directory of the mixed-input model (`setsuna_core::mixed`; the IME
    /// ships one, `data/mixed`, which the Windows installer sets here and
    /// fcitx5 takes when this is unset): English words typed among the
    /// romaji stay as typed, Shift+letter types into the composition. Unset
    /// (Windows) or empty: every letter is romaji.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed_model: Option<String>,
    /// With `mixed_model`: the Latin names dictionary (the distributed
    /// `latin-names.tsv`, `<name as written>\t<weight>[\tcommon]` lines).
    /// Its names count as English words, and one typed in lowercase takes
    /// its case as `name_case` says (github -> GitHub). Unset: the model's
    /// words only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_dictionary_file: Option<String>,
    /// What English typed in lowercase takes its case: the names of the
    /// names dictionary, and acronyms.
    pub name_case: NameCase,
    /// An English sentence (two words or more, with `mixed_model`) starts
    /// with a capital and `i` is `I`; the `.` `,` `!` `?` right after an
    /// English word stay as typed in a sentence with no Japanese
    /// (`mixed::english_after`).
    pub sentence_case: bool,
    /// Keys that turn the IME on or off (Windows): a key name with
    /// `Ctrl+` / `Shift+` / `Alt+` / `Win+` before it, such as `Zenkaku`
    /// (半角/全角), `Alt+\``, `Henkan`, `Muhenkan`, `Ctrl+Space`. Ctrl+Space
    /// is not in by default: it is a shortcut of many apps.
    pub toggle_keys: Vec<String>,
    /// Keys that only turn it on (`ImeOn`: VK_IME_ON, which some keyboards
    /// send).
    pub on_keys: Vec<String>,
    /// Keys that only turn it off (`ImeOff`: VK_IME_OFF).
    pub off_keys: Vec<String>,
}

/// How far mixed input cases English typed in lowercase
/// (`mixed::MixedModel::cased`). A name that is an ordinary word too (word,
/// apple: the dictionary marks it `common`) stays as typed at every level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NameCase {
    /// As typed.
    Off,
    /// Proper nouns of the names dictionary (github -> GitHub, iphone ->
    /// iPhone, twitter -> Twitter).
    Proper,
    /// Those, and acronyms by rule (`mixed::acronym`): a word that cannot
    /// be read as English (pc, sns, kpi, usb -> PC, SNS, KPI, USB) and the
    /// few that can but are written so (ok, api, id -> OK, API, ID).
    #[default]
    Acronyms,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolWidth {
    #[default]
    AsTyped,
    Full,
    Half,
    /// Digits, letters and symbols as the text before the caret writes
    /// them; without one, half-width digits and letters, and each symbol as
    /// Japanese text usually writes it (`data/symbol_width.tsv`: ！（）：
    /// full-width, " ' [ ] half-width, 10:30).
    Auto,
}

/// Touten (comma) / kuten (period) pair written into the candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PunctuationStyle {
    /// Leave the candidates' punctuation as the engine wrote it.
    #[default]
    AsIs,
    /// `、` `。`
    TotenKuten,
    /// `，` `．`
    CommaPeriod,
    /// `，` `。`
    CommaKuten,
    /// `、` `．`
    TotenPeriod,
}

impl PunctuationStyle {
    pub const ALL: [PunctuationStyle; 5] = [
        PunctuationStyle::AsIs,
        PunctuationStyle::TotenKuten,
        PunctuationStyle::CommaPeriod,
        PunctuationStyle::CommaKuten,
        PunctuationStyle::TotenPeriod,
    ];

    /// The (comma, period) pair to write, or `None` for `AsIs`.
    pub fn marks(self) -> Option<(char, char)> {
        match self {
            PunctuationStyle::AsIs => None,
            PunctuationStyle::TotenKuten => Some(('、', '。')),
            PunctuationStyle::CommaPeriod => Some(('，', '．')),
            PunctuationStyle::CommaKuten => Some(('，', '。')),
            PunctuationStyle::TotenPeriod => Some(('、', '．')),
        }
    }
}

/// One literal replacement rule (`from` -> `to`) for the candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replacement {
    pub from: String,
    pub to: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl Default for FrontendCfg {
    fn default() -> Self {
        Self {
            live_min_dispatch_chars: 3,
            force_convert_on_space: true,
            symbols: default_symbol_map(),
            symbol_width: SymbolWidth::AsTyped,
            punctuation: PunctuationStyle::AsIs,
            normalize_middle_dot: false,
            replacements: Vec::new(),
            drop_untyped_symbols: true,
            symbol_dictionary: true,
            symbol_dictionary_file: None,
            date_time_candidates: true,
            number_candidates: true,
            emoji_candidates: true,
            prediction: true,
            romaji_table: "standard".to_string(),
            typo_correction: true,
            pass_through_commands: false,
            full_width_space: false,
            live_conversion: None,
            backspace_deletes_shown: true,
            escape_to_kana: true,
            preedit_cursor_at_start: false,
            mixed_model: None,
            name_dictionary_file: None,
            name_case: NameCase::Acronyms,
            sentence_case: true,
            toggle_keys: ["Zenkaku", "Alt+Zenkaku", "Alt+`"]
                .map(String::from)
                .to_vec(),
            on_keys: vec!["ImeOn".to_string()],
            off_keys: vec!["ImeOff".to_string()],
        }
    }
}

/// Reranking of the candidates by the user's past choices (`learning`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearningCfg {
    /// Record which candidate the user picks and rerank by it.
    pub enabled: bool,
    /// Weight of a learned choice against the model's order. At 1.0 a choice
    /// learned for the exact same reading moves to the top.
    pub strength: f32,
    /// Minimum cosine similarity (reading character bigrams) for a choice
    /// learned on another reading to apply as a whole sentence. 1.0 = exact
    /// reading only.
    pub similarity_threshold: f32,
    /// Also learn the words a choice changed (きしゃ: 記者 -> 貴社) and apply
    /// them wherever their reading recurs next to the same reading chars.
    pub words: bool,
}

impl Default for LearningCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 1.0,
            similarity_threshold: 0.6,
            words: true,
        }
    }
}

/// The ASCII char of each `[frontend.symbols]` name (the frontends map their
/// keys to these names).
pub const SYMBOL_CHARS: [(&str, char); 31] = [
    ("semicolon", ';'),
    ("colon", ':'),
    ("equals", '='),
    ("plus", '+'),
    ("comma", ','),
    ("less", '<'),
    ("period", '.'),
    ("greater", '>'),
    ("slash", '/'),
    ("question", '?'),
    ("backquote", '`'),
    ("tilde", '~'),
    ("left_bracket", '['),
    ("left_brace", '{'),
    ("yen", '\\'),
    ("pipe", '|'),
    ("right_bracket", ']'),
    ("right_brace", '}'),
    ("quote", '\''),
    ("double_quote", '"'),
    ("shift_1", '!'),
    ("shift_3", '#'),
    ("shift_4", '$'),
    ("shift_5", '%'),
    ("shift_6", '&'),
    ("shift_8", '('),
    ("shift_9", ')'),
    ("underscore", '_'),
    ("at", '@'),
    ("asterisk", '*'),
    ("caret", '^'),
];

/// The `[frontend.symbols]` name of the ASCII symbol `c`, whatever key typed
/// it (the keyboard layout decides: `:` is its own key on JIS, Shift+`;` on
/// US). `\` is `oem102_backslash` on the JIS ろ key (`ro_key`), else `yen`.
pub fn symbol_name(c: char, ro_key: bool) -> Option<&'static str> {
    if c == '\\' {
        return Some(if ro_key { "oem102_backslash" } else { "yen" });
    }
    SYMBOL_CHARS
        .iter()
        .find(|(_, s)| *s == c)
        .map(|(name, _)| *name)
}

/// What typing the ASCII symbol `c` inserts: its `[frontend.symbols]` entry,
/// else its full-width form.
pub fn typed_symbol(c: char, ro_key: bool, symbols: &BTreeMap<String, String>) -> String {
    symbol_name(c, ro_key)
        .and_then(|name| symbols.get(name).cloned())
        .unwrap_or_else(|| full_width(c).to_string())
}

/// The full-width form of a printable ASCII char.
fn full_width(c: char) -> char {
    match c {
        '!'..='~' => char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c),
        _ => c,
    }
}

/// What each symbol key types by default: `、。・「」` for `, . / [ ]`, `￥`
/// for the yen key, and every other symbol in the width Japanese text
/// usually writes it in when no digit or letter is beside it
/// (`data/symbol_width.tsv`): `！？（）：％` full-width, `" ' # _ | { }`
/// half-width.
pub fn default_symbol_map() -> BTreeMap<String, String> {
    let mut map: BTreeMap<String, String> = JAPANESE_SYMBOLS
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    for (name, c) in SYMBOL_CHARS {
        if !map.contains_key(name) {
            let c = if crate::postprocess::usually_full(c, 0) {
                full_width(c)
            } else {
                c
            };
            map.insert(name.to_string(), c.to_string());
        }
    }
    map
}

/// The Japanese symbols the comma, period, slash, bracket and yen keys type
/// by default.
const JAPANESE_SYMBOLS: [(&str, &str); 6] = [
    ("comma", "、"),
    ("period", "。"),
    ("slash", "・"),
    ("left_bracket", "「"),
    ("right_bracket", "」"),
    ("yen", "￥"),
];

/// What the symbol key `name` (a `SYMBOL_CHARS` one) may type, as the
/// settings app offers it: its Japanese symbol (、。・「」 for , . / [ ]),
/// its full-width form and its half-width one (the yen key: ￥ and `\`).
pub struct SymbolForms {
    pub japanese: Option<&'static str>,
    pub full: String,
    pub half: String,
}

pub fn symbol_forms(name: &str) -> Option<SymbolForms> {
    let &(_, c) = SYMBOL_CHARS.iter().find(|(k, _)| *k == name)?;
    if name == "yen" {
        return Some(SymbolForms {
            japanese: None,
            full: "￥".into(),
            half: c.to_string(),
        });
    }
    Some(SymbolForms {
        japanese: JAPANESE_SYMBOLS
            .iter()
            .find(|(k, _)| *k == name)
            .map(|&(_, s)| s),
        full: full_width(c).to_string(),
        half: c.to_string(),
    })
}

/// `symbols` with every symbol key typing its full-width form (`full`:
/// ！（）＃) or its half-width one (! ( ) #), but those that type their
/// Japanese symbol (、。・「」), which stay: the settings app's 全角 / 半角
/// for the typed symbols. Other keys are kept.
pub fn symbols_in_width(
    symbols: &BTreeMap<String, String>,
    full: bool,
) -> BTreeMap<String, String> {
    let mut out = symbols.clone();
    for (name, _) in SYMBOL_CHARS {
        let Some(forms) = symbol_forms(name) else {
            continue;
        };
        let now = symbols.get(name).map(String::as_str);
        if now.is_some() && now == forms.japanese {
            continue;
        }
        let to = if full { forms.full } else { forms.half };
        out.insert(name.to_string(), to);
    }
    out
}

/// The reranker: a causal LM (`setsuna-rerank daemon`) rescoring the
/// converter's top candidates once the user pauses. Unset `manifest`: off.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RerankCfg {
    /// LM scorer manifest (`neo-ime-lm-scorer-export-v1`).
    pub manifest: Option<PathBuf>,
    /// Optional explicit `setsuna-rerank.exe`. Empty = next to the module.
    pub exe: Option<PathBuf>,
    /// Threads the reranker computes with. Empty = its default.
    pub threads: Option<usize>,
    /// Candidates reranked first; doubled while the user stays idle.
    pub first_candidates: usize,
    /// Most candidates reranked. Empty = the manifest's.
    pub max_candidates: Option<usize>,
    /// Weight of the LM score. Empty = the manifest's.
    pub lambda: Option<f32>,
    /// Typing pause (ms) before reranking; the conversion key reranks at once.
    pub idle_ms: u64,
    /// While typing, how long (ms, after the pause) the engine's top waits
    /// for the reranked one before it is shown, so that the preedit does
    /// not change twice (Mozkey's deferred display). 0: off, the engine's
    /// top at once.
    pub defer_ms: u64,
}

impl Default for RerankCfg {
    fn default() -> Self {
        Self {
            manifest: None,
            exe: None,
            threads: None,
            first_candidates: 5,
            max_candidates: None,
            lambda: None,
            idle_ms: 120,
            defer_ms: 0,
        }
    }
}

/// Character n-gram scores added to the engine's (`char_ngram`): the
/// engine's candidates are reordered by them before anything else sees the
/// list, the LM reranker included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NgramCfg {
    /// The table (`CHNGRAM1`, counted from the training text). Unset or
    /// empty: off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    /// Weight of `Σ min(0, PMI)` over the candidate's adjacent chars.
    pub bigram_weight: f32,
    /// Weight of `Σ min(0, PMI)` over its triples (a pair and the char
    /// after it).
    pub trigram_weight: f32,
}

/// `[lexicon]`: the reading lexicon (`lexicon`), which scores the engine's
/// candidates by the words text uses and drops the near-misses.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LexiconCfg {
    /// The compiled lexicon (`LEXICON1`). Unset or empty: off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

impl Default for NgramCfg {
    fn default() -> Self {
        // picked on the distributed model's wiki-fresh conversions
        Self {
            table: None,
            bigram_weight: 0.7,
            trigram_weight: 0.5,
        }
    }
}

/// The rewriter (`setsuna-rewrite daemon`): an OpenAI model, or one behind
/// an OpenAI-compatible API, rewriting the conversion into natural Japanese.
/// While it is on, the conversion key also asks for a rewrite and puts it
/// first among the candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewriterCfg {
    /// Rewrite on the conversion key.
    pub enabled: bool,
    pub provider: RewriteProvider,
    /// The API's base URL (openai / compatible). Empty = OpenAI's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// The API requests go to. Empty = responses for OpenAI,
    /// chat_completions for a compatible server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<RewriteApi>,
    pub model: String,
    /// What the model is told. Empty = the built-in instructions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// `reasoning.effort` (Responses API), e.g. "low". Empty = the model's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Send the text before the caret along as context.
    pub send_context: bool,
    /// Longest a rewrite may take (ms).
    pub timeout_ms: u64,
    /// Optional explicit `setsuna-rewrite.exe`. Empty = next to the module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exe: Option<PathBuf>,
}

impl Default for RewriterCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: RewriteProvider::Chatgpt,
            base_url: None,
            api: None,
            model: String::new(),
            instructions: None,
            reasoning_effort: None,
            send_context: true,
            timeout_ms: 20_000,
            exe: None,
        }
    }
}

/// Who serves (and pays for) the rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteProvider {
    /// The ChatGPT plan, signed in with ChatGPT.
    Chatgpt,
    /// OpenAI with an API key.
    Openai,
    /// An OpenAI-compatible server.
    Compatible,
}

impl RewriteProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chatgpt => "chatgpt",
            Self::Openai => "openai",
            Self::Compatible => "compatible",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteApi {
    Responses,
    ChatCompletions,
}

impl RewriteApi {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
        }
    }
}

impl Config {
    /// Load from `<config_dir>/config.toml`. On any error (missing file,
    /// parse error, IO error), returns `Default::default()` so the IME
    /// never fails activation because of a config issue.
    pub fn load_default() -> Self {
        match Self::load_from_appdata() {
            Ok(c) => c,
            Err(_) => Self::default(),
        }
    }

    /// Like `load_default` but propagates errors. Use when you want to
    /// log the failure reason via `tracing`.
    pub fn load_from_appdata() -> Result<Self> {
        let path = Self::appdata_config_path()
            .ok_or_else(|| anyhow::anyhow!("no config directory (APPDATA / HOME not set)"))?;
        Self::load_from_path(&path)
    }

    pub fn load_from_path(path: &Path) -> Result<Self> {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("open config {}", path.display()))?;
        let mut s = String::new();
        file.read_to_string(&mut s)
            .with_context(|| format!("read config {}", path.display()))?;
        Self::load_from_str(&s)
    }

    pub fn load_from_str(s: &str) -> Result<Self> {
        let mut cfg: Self = toml::from_str(s).with_context(|| "parse config.toml")?;
        for (k, v) in default_symbol_map() {
            cfg.frontend.symbols.entry(k).or_insert(v);
        }
        Ok(cfg)
    }

    /// Resolve `<config_dir>/config.toml` (`%APPDATA%\setsuna-ime\config.toml`
    /// on Windows).
    pub fn appdata_config_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("config.toml"))
    }

    /// Resolve `<config_dir>/learning.jsonl`, the log of learned choices.
    pub fn learning_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("learning.jsonl"))
    }

    /// Where the candidates hidden with Ctrl+Delete are kept.
    pub fn hidden_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("hidden.jsonl"))
    }

    /// What prediction learns from: the conversions committed.
    pub fn prediction_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("prediction.jsonl"))
    }

    /// The user dictionary (Mozc / Google Japanese Input text format).
    pub fn user_dictionary_path() -> Option<PathBuf> {
        config_dir().map(|d| d.join("user_dictionary.txt"))
    }

    /// Write this config to `path`, keeping the comments and layout of the
    /// file already there (or of the default template when there is none).
    /// Every key this config serializes is written; keys it leaves out (an
    /// unset engine knob, a removed key) are left as they are in the file.
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        let existing = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::emit_default_toml(),
            Err(e) => return Err(e).with_context(|| format!("read config {}", path.display())),
        };
        let mut doc: toml_edit::DocumentMut = existing
            .parse()
            .with_context(|| format!("parse config {}", path.display()))?;
        let new: toml_edit::DocumentMut = toml::to_string(self)
            .context("serialize config")?
            .parse()
            .context("reparse serialized config")?;
        merge_table(doc.as_table_mut(), new.as_table());
        // every application's entry removed
        if self.apps.is_empty() {
            doc.remove("apps");
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        // Write then rename, so a reader never sees a half-written file.
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, doc.to_string())
            .with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
        Ok(())
    }

    /// Emit a fully-commented `config.toml` template. The output is
    /// hand-formatted; round-tripping `toml::to_string(&Config::default())`
    /// would lose the comments that document each knob.
    pub fn emit_default_toml() -> String {
        let d = Self::default();
        format!(
            "# Setsuna TSF IME configuration. Edit and save; the IME picks up\n\
             # the new values when the next input starts.\n\
             \n\
             [engine]\n\
             # Model artifact manifest written by export. Edit this to switch\n\
             # models without rebuilding or reinstalling.\n\
             # model_manifest = \"models/onnx/model.int4.manifest.json\"\n\
             # Alternatively a directory holding exactly one `*.manifest.json`.\n\
             # model_dir = \"models/onnx\"\n\
             # Optional explicit engine daemon executable.\n\
             # engine_exe = \"setsuna-engine.exe\"\n\
             # ONNX Runtime the engine loads (default: onnxruntime.dll next to it).\n\
             # ort_dylib = \"onnxruntime.dll\"\n\
             # Decode knobs. An unset knob falls back to the manifest's\n\
             # `decode_defaults`, then to the engine default.\n\
             # beam_width = 20\n\
             # top_k = 20\n\
             # top_k_per_step = 16\n\
             # Left-context chars fed to the model.\n\
             # max_context = 64\n\
             # symbol_split = true\n\
             # Each run of symbols read as one token (models trained so).\n\
             # symbol_token = true\n\
             # Typed numbers kept as typed, out of the model's reading.\n\
             # digit_split = true\n\
             # Token IDs forced to -inf log-prob.\n\
             # vocab_blacklist = [724, 730, 734, 737, 776, 788]\n\
             \n\
             [ui.candidate_window]\n\
             # setsuna, setsuna-dark, or classic (flat, in the colors below).\n\
             theme = \"{theme}\"\n\
             font_face = \"{font_face}\"\n\
             font_pt = {font_pt}\n\
             page_size = {page_size}\n\
             highlight_color = \"{highlight_color}\"\n\
             highlight_text_color = \"{highlight_text_color}\"\n\
             bg_color = \"{bg_color}\"\n\
             border_color = \"{border_color}\"\n\
             footer_color = \"{footer_color}\"\n\
             \n\
             [frontend]\n\
             # Live typing waits for this many hiragana chars before asking\n\
             # the daemon. Space conversion bypasses it when\n\
             # force_convert_on_space=true.\n\
             live_min_dispatch_chars = 3\n\
             force_convert_on_space = true\n\
             # Width of typed symbols in the candidates: as_typed, full, half,\n\
             # or auto (digits, letters and symbols as the text before the\n\
             # caret writes them).\n\
             symbol_width = \"as_typed\"\n\
             # Touten / kuten of the candidates: as_is (as converted),\n\
             # touten_kuten (、。), comma_period (，．), comma_kuten (，。),\n\
             # touten_period (、．).\n\
             punctuation = \"as_is\"\n\
             # Write every middle-dot variant (･ · ‧ ⋅ ∙) as ・.\n\
             normalize_middle_dot = false\n\
             # Drop candidates with a symbol (punctuation, brackets, ...) the\n\
             # reading does not have.\n\
             drop_untyped_symbols = true\n\
             # Offer symbols for readings that name them (やじるし -> →).\n\
             symbol_dictionary = true\n\
             # A larger symbol dictionary merged into the built-in one (JSON\n\
             # lines with reading and symbol). Unset: off.\n\
             # symbol_dictionary_file = \"C:/path/to/symbols.jsonl\"\n\
             # Local date/time candidates (きょう, いま, 1じかんご, ...).\n\
             date_time_candidates = true\n\
             # Number candidates: 1,234 / 千二百三十四 / 12:34, じゅうに -> 12,\n\
             # 2025ねん -> 令和7年, 1+2= -> 3, U+3042 -> あ.\n\
             number_candidates = true\n\
             # Emoji and kaomoji candidates (にこにこ, えもじ, かおもじ).\n\
             emoji_candidates = true\n\
             # Prediction from what was committed (Tab opens it).\n\
             prediction = true\n\
             # The romaji table: \"standard\", \"azik\", \"kana_jis\" (kana input on the\n\
             # JIS layout) or the path of a table in Mozc's format.\n\
             romaji_table = \"standard\"\n\
             # Typo correction where a letter reads as no kana (kaisga: かいしゃ).\n\
             typo_correction = true\n\
             # /command, $skill, @file typed at a line's start or after a space\n\
             # go to the app as typed.\n\
             pass_through_commands = false\n\
             # Space with nothing composed types a full-width space (U+3000);\n\
             # Shift+Space types the other width.\n\
             full_width_space = false\n\
             # While typing, show the conversion in the text (true) or the\n\
             # typed kana with the conversions in a suggestion window (false).\n\
             # Default true.\n\
             # live_conversion = false\n\
             # With live conversion, Backspace deletes the last char shown and\n\
             # the reading under it, the rest staying as shown (true), or the\n\
             # last kana typed, the rest converted again (false).\n\
             backspace_deletes_shown = true\n\
             # Escape on a conversion shown (live or in the candidate window)\n\
             # shows the kana typed, and Escape again clears (true); false goes\n\
             # from the candidate window back to the live conversion.\n\
             escape_to_kana = true\n\
             # fcitx5: keep the preedit caret at the start of the composition,\n\
             # so the candidate window stays put (the caret is then not seen\n\
             # to move while typing). Default false.\n\
             # preedit_cursor_at_start = true\n\
             # Directory of the mixed Japanese / English input model: English\n\
             # words typed among the romaji stay as typed. The installer sets\n\
             # the one it installs; \"\" turns mixed input off.\n\
             # mixed_model = \"C:/path/to/mixed-model\"\n\
             # With it, the Latin names dictionary: names typed in lowercase\n\
             # take their case (github -> GitHub). Unset: off.\n\
             # name_dictionary_file = \"C:/path/to/latin-names.tsv\"\n\
             # What English typed in lowercase takes its case: \"acronyms\"\n\
             # (proper nouns and acronyms: github -> GitHub, pc -> PC),\n\
             # \"proper\" (proper nouns only) or \"off\".\n\
             name_case = \"acronyms\"\n\
             # An English sentence starts with a capital (hello world -> Hello\n\
             # world), and . , ! ? after English stay as typed where no\n\
             # Japanese is in the sentence.\n\
             sentence_case = true\n\
             # Keys that turn the IME on and off (Windows): Zenkaku (半角/全角),\n\
             # Henkan, Muhenkan, Eisu, Kana, Space, `, a letter, F1-F24, ...\n\
             # after Ctrl+ / Shift+ / Alt+ / Win+. Ctrl+Space is a shortcut of\n\
             # many apps (VS Code, Excel, Emacs): add it only if you want it.\n\
             toggle_keys = [\"Zenkaku\", \"Alt+Zenkaku\", \"Alt+`\"]\n\
             # Keys that only turn it on / off (ImeOn, ImeOff: VK_IME_ON / OFF).\n\
             on_keys = [\"ImeOn\"]\n\
             off_keys = [\"ImeOff\"]\n\
             # Literal replacements applied to every candidate, in order:\n\
             # [[frontend.replacements]]\n\
             # from = \"～\"\n\
             # to = \"〜\"\n\
             \n\
             [frontend.symbols]\n\
             comma = \"、\"\n\
             period = \"。\"\n\
             slash = \"・\"\n\
             question = \"？\"\n\
             left_bracket = \"「\"\n\
             right_bracket = \"」\"\n\
             yen = \"￥\"\n\
             shift_8 = \"（\"\n\
             shift_9 = \"）\"\n\
             \n\
             [learning]\n\
             # Rerank candidates by the ones picked before (learning.jsonl).\n\
             enabled = {learning_enabled}\n\
             # 1.0: a choice learned for the same reading moves to the top.\n\
             strength = {learning_strength:?}\n\
             # Reading similarity (cosine over character bigrams) a choice\n\
             # learned on another reading needs to apply; 1.0 = same reading.\n\
             similarity_threshold = {learning_threshold:?}\n\
             # Also learn the changed words (きしゃ: 記者 -> 貴社) and apply them\n\
             # wherever their reading recurs next to the same reading chars.\n\
             words = {learning_words}\n\
             \n\
             [rerank]\n\
             # A language model rescoring the top candidates once typing\n\
             # pauses. Off while no manifest is set.\n\
             # manifest = \"models/rerank/model.q8_0.manifest.json\"\n\
             # exe = \"setsuna-rerank.exe\"\n\
             # threads = 1\n\
             # Candidates reranked first; doubled while typing stays paused,\n\
             # up to max_candidates (default: the manifest's).\n\
             first_candidates = {rerank_first}\n\
             # max_candidates = 20\n\
             # Weight of the LM score against the converter's (default: the manifest's).\n\
             # lambda = 0.3\n\
             # Typing pause (ms) before reranking; the conversion key reranks at once.\n\
             idle_ms = {rerank_idle}\n\
             \n\
             [ngram]\n\
             # Character bigram / trigram counts of text: a candidate joining\n\
             # chars rarer than chance (博紙 for はくし) moves down. Off while no\n\
             # table is set.\n\
             # table = \"C:/path/to/chars.ngram\"\n\
             # Weights of the bigram and trigram terms against the converter's score.\n\
             bigram_weight = {ngram_bigram:?}\n\
             trigram_weight = {ngram_trigram:?}\n\
             \n\
             [lexicon]\n\
             # Reading lexicon: candidates are scored by the words text uses,\n\
             # and the model's near-misses (続頭 for ぞくとう) dropped. Off while\n\
             # no file is set.\n\
             # file = \"C:/path/to/words.lexicon\"\n\
             \n\
             [rewriter]\n\
             # Rewriting: the conversion key also asks a model to rewrite the\n\
             # input into natural Japanese and puts the result first. Sign in,\n\
             # keys and models: the settings GUI.\n\
             enabled = {rewriter_enabled}\n\
             # chatgpt (your ChatGPT plan, signed in), openai (an API key) or\n\
             # compatible (an OpenAI-compatible server at base_url).\n\
             provider = \"{rewriter_provider}\"\n\
             # base_url = \"http://127.0.0.1:8080/v1\"\n\
             # responses or chat_completions (default: responses for OpenAI,\n\
             # chat_completions for a compatible server).\n\
             # api = \"chat_completions\"\n\
             model = \"\"\n\
             # What the model is told (default: the built-in instructions).\n\
             # instructions = \"...\"\n\
             # reasoning_effort = \"low\"\n\
             # Send the text before the caret along as context.\n\
             send_context = {rewriter_context}\n\
             timeout_ms = {rewriter_timeout}\n",
            theme = d.ui.candidate_window.theme,
            font_face = d.ui.candidate_window.font_face,
            font_pt = d.ui.candidate_window.font_pt,
            page_size = d.ui.candidate_window.page_size,
            highlight_color = d.ui.candidate_window.highlight_color,
            highlight_text_color = d.ui.candidate_window.highlight_text_color,
            bg_color = d.ui.candidate_window.bg_color,
            border_color = d.ui.candidate_window.border_color,
            footer_color = d.ui.candidate_window.footer_color,
            learning_enabled = d.learning.enabled,
            learning_strength = d.learning.strength,
            learning_threshold = d.learning.similarity_threshold,
            learning_words = d.learning.words,
            rerank_first = d.rerank.first_candidates,
            rerank_idle = d.rerank.idle_ms,
            ngram_bigram = d.ngram.bigram_weight,
            ngram_trigram = d.ngram.trigram_weight,
            rewriter_enabled = d.rewriter.enabled,
            rewriter_provider = d.rewriter.provider.as_str(),
            rewriter_context = d.rewriter.send_context,
            rewriter_timeout = d.rewriter.timeout_ms,
        )
    }

    /// Parse `#RRGGBB` → COLORREF u32 (0x00BBGGRR). Returns 0 on parse
    /// failure so a typo in a color string never aborts activation.
    pub fn parse_color(s: &str) -> u32 {
        let h = s.trim_start_matches('#');
        if h.len() != 6 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
            return 0;
        }
        let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(0) as u32;
        let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(0) as u32;
        let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(0) as u32;
        (b << 16) | (g << 8) | r
    }
}

/// Directory holding `config.toml` and the learning log: `%APPDATA%\setsuna-ime`
/// on Windows, `~/Library/Application Support/setsuna-ime` on macOS,
/// `$XDG_CONFIG_HOME/setsuna-ime` (or `~/.config/setsuna-ime`) elsewhere.
pub fn config_dir() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        env("APPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library").join("Application Support"))
    } else {
        env("XDG_CONFIG_HOME").or_else(|| env("HOME").map(|h| h.join(".config")))
    };
    base.map(|b| b.join("setsuna-ime"))
}

/// Copy every key of `new` into `doc`: tables are merged key by key, any
/// other item replaces the old one. A replaced value keeps its old
/// decoration (its trailing comment); the key keeps the comments above it.
fn merge_table(doc: &mut toml_edit::Table, new: &toml_edit::Table) {
    for (key, item) in new.iter() {
        match (doc.get_mut(key), item) {
            (Some(toml_edit::Item::Table(old)), toml_edit::Item::Table(t)) => merge_table(old, t),
            (Some(toml_edit::Item::Value(old)), toml_edit::Item::Value(v)) => {
                let decor = old.decor().clone();
                *old = v.clone();
                *old.decor_mut() = decor;
            }
            _ => {
                doc.insert(key, item.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_application_takes_its_settings_over_the_rest() {
        let cfg = Config::load_from_str(
            "[frontend]\nprediction = true\n\n[[apps]]\nprocess = \"Code.exe\"\nstart = \"off\"\nprediction = false\nmixed_input = false\n",
        )
        .unwrap();
        let (c, app) = cfg.for_app("C:\\Program Files\\VS Code\\code.EXE");
        assert_eq!(app.unwrap().start, AppStart::Off);
        assert!(!c.frontend.prediction);
        assert_eq!(c.frontend.mixed_model.as_deref(), Some(""));
        let (c, app) = cfg.for_app("notepad.exe");
        assert!(app.is_none() && c.frontend.prediction);
        // saved and read back; none left removes them
        let path = std::env::temp_dir().join(format!("setsuna-apps-{}.toml", std::process::id()));
        cfg.save_to_path(&path).unwrap();
        assert_eq!(Config::load_from_path(&path).unwrap().apps, cfg.apps);
        let mut none = cfg.clone();
        none.apps.clear();
        none.save_to_path(&path).unwrap();
        assert!(Config::load_from_path(&path).unwrap().apps.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_keeps_comments_and_unset_knobs() {
        let dir = std::env::temp_dir().join(format!("setsuna-config-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("config.toml");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            "# top comment\n[engine]\n# the model\nmodel_manifest = \"m.json\"\nbeam_width = 30\n\n\
             [frontend]\n# wait for this many\nlive_min_dispatch_chars = 3 # chars\n",
        )
        .unwrap();
        let mut cfg = Config::load_from_path(&path).unwrap();
        cfg.frontend.live_min_dispatch_chars = 5;
        cfg.frontend.replacements.push(Replacement {
            from: "～".into(),
            to: "〜".into(),
            enabled: true,
        });
        cfg.frontend.punctuation = PunctuationStyle::CommaPeriod;
        cfg.learning.strength = 0.5;
        cfg.save_to_path(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# top comment"), "{text}");
        assert!(
            text.contains("# wait for this many\nlive_min_dispatch_chars = 5 # chars"),
            "{text}"
        );
        // A knob the GUI does not touch is left as written.
        assert!(text.contains("beam_width = 30"), "{text}");
        let back = Config::load_from_path(&path).unwrap();
        assert_eq!(back.engine.model_manifest, Some(PathBuf::from("m.json")));
        assert_eq!(back.frontend.punctuation, PunctuationStyle::CommaPeriod);
        assert_eq!(back.frontend.live_min_dispatch_chars, 5);
        assert_eq!(back.frontend.replacements, cfg.frontend.replacements);
        assert_eq!(back.learning.strength, 0.5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_without_a_file_starts_from_the_template() {
        let dir = std::env::temp_dir().join(format!("setsuna-config-new-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("config.toml");
        Config::default().save_to_path(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# Setsuna TSF IME configuration"), "{text}");
        Config::load_from_path(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn defaults_match_deploy_hardcodes() {
        let d = Config::default();
        // candidate_window: must reproduce candidate_window.rs constants.
        assert_eq!(d.ui.candidate_window.font_face, "Meiryo UI");
        assert_eq!(d.ui.candidate_window.font_pt, 16);
        assert_eq!(d.ui.candidate_window.page_size, 10);
        assert_eq!(d.ui.candidate_window.highlight_color, "#D77800");
    }

    #[test]
    fn unset_engine_knobs_stay_unset() {
        let c = Config::load_from_str("[engine]\nbeam_width = 100").expect("parse");
        assert_eq!(c.engine.beam_width, Some(100));
        // Unset knobs are left to the manifest and the engine.
        assert_eq!(c.engine.top_k, None);
        assert_eq!(c.engine.symbol_split, None);
        assert_eq!(c.engine.symbol_token, None);
        assert_eq!(c.engine.digit_split, None);
        assert_eq!(c.engine.vocab_blacklist, None);
        assert_eq!(c.ui.candidate_window.font_face, "Meiryo UI");
    }

    #[test]
    fn name_case_parses() {
        let cfg =
            Config::load_from_str("[frontend]\nname_case = \"proper\"\nsentence_case = false")
                .unwrap();
        assert_eq!(cfg.frontend.name_case, NameCase::Proper);
        assert!(!cfg.frontend.sentence_case);
        assert_eq!(Config::default().frontend.name_case, NameCase::Acronyms);
    }

    #[test]
    fn symbol_width_auto_parses() {
        let cfg = Config::load_from_str("[frontend]\nsymbol_width = \"auto\"").unwrap();
        assert_eq!(cfg.frontend.symbol_width, SymbolWidth::Auto);
    }

    #[test]
    fn date_time_candidates_default_on_and_can_be_disabled() {
        let cfg = Config::load_from_str("[frontend]").unwrap();
        assert!(cfg.frontend.date_time_candidates);
        let cfg = Config::load_from_str("[frontend]\ndate_time_candidates = false").unwrap();
        assert!(!cfg.frontend.date_time_candidates);
        let saved = toml::to_string(&cfg).unwrap();
        assert!(
            !Config::load_from_str(&saved)
                .unwrap()
                .frontend
                .date_time_candidates
        );
    }

    #[test]
    fn parses_symbol_split() {
        let s = r#"
[engine]
symbol_split = true
symbol_token = true
digit_split = true
        "#;
        let c = Config::load_from_str(s).expect("parse");
        assert_eq!(c.engine.symbol_split, Some(true));
        assert_eq!(c.engine.symbol_token, Some(true));
        assert_eq!(c.engine.digit_split, Some(true));
    }

    #[test]
    fn emit_default_toml_is_parseable() {
        let s = Config::emit_default_toml();
        let c = Config::load_from_str(&s).expect("default-emit parses");
        // Every engine knob is commented out in the template.
        assert!(c.engine.model_manifest.is_none());
        assert_eq!(c.engine.beam_width, None);
        assert_eq!(c.ui.candidate_window.font_pt, 16);
        assert_eq!(
            c.frontend.toggle_keys,
            Config::default().frontend.toggle_keys
        );
    }

    #[test]
    fn on_off_keys_default_and_parse() {
        let d = Config::default().frontend;
        assert!(d.toggle_keys.iter().all(|k| !k.contains("Ctrl")));
        assert_eq!(d.on_keys, ["ImeOn"]);
        let c = Config::load_from_str("[frontend]\ntoggle_keys = [\"Ctrl+Space\"]\noff_keys = []")
            .unwrap();
        assert_eq!(c.frontend.toggle_keys, ["Ctrl+Space"]);
        assert!(c.frontend.off_keys.is_empty());
        assert_eq!(c.frontend.on_keys, ["ImeOn"]);
    }

    #[test]
    fn partial_symbol_map_merges_defaults() {
        let s = r#"
[frontend.symbols]
slash = "・"
        "#;
        let c = Config::load_from_str(s).expect("parse");
        assert_eq!(c.frontend.symbols.get("slash"), Some(&"・".to_string()));
        assert_eq!(c.frontend.symbols.get("comma"), Some(&"、".to_string()));
    }

    #[test]
    fn typed_symbols_go_full_or_half_width_but_the_japanese_ones() {
        // as the installer writes them: ／ for the slash key, （ ） ？
        let mut symbols = default_symbol_map();
        for (k, v) in [("slash", "／"), ("shift_8", "（"), ("question", "？")] {
            symbols.insert(k.into(), v.into());
        }
        let half = symbols_in_width(&symbols, false);
        let typed = |m: &BTreeMap<String, String>, c: char| typed_symbol(c, false, m);
        for (c, s) in [
            ('(', "("),
            (')', ")"),
            ('?', "?"),
            ('!', "!"),
            ('/', "/"),
            ('\\', "\\"),
        ] {
            assert_eq!(typed(&half, c), s);
        }
        // 、。「」 stay
        for (c, s) in [(',', "、"), ('.', "。"), ('[', "「"), (']', "」")] {
            assert_eq!(typed(&half, c), s);
        }
        let full = symbols_in_width(&half, true);
        for (c, s) in [
            ('(', "（"),
            ('?', "？"),
            ('#', "＃"),
            ('/', "／"),
            ('\\', "￥"),
            ('[', "「"),
        ] {
            assert_eq!(typed(&full, c), s);
        }
        // the slash key typing ・ keeps it
        let mut dot = symbols.clone();
        dot.insert("slash".into(), "・".into());
        assert_eq!(typed(&symbols_in_width(&dot, false), '/'), "・");
        // every key types a symbol of its own, which mixed input reads back
        for m in [&half, &full] {
            let outs: std::collections::HashSet<&String> =
                SYMBOL_CHARS.iter().map(|(k, _)| &m[*k]).collect();
            assert_eq!(outs.len(), SYMBOL_CHARS.len());
        }
    }

    #[test]
    fn parse_color_basic() {
        // #D77800 = R=0xD7 G=0x78 B=0x00 → COLORREF = 0x00_00_78_D7
        assert_eq!(Config::parse_color("#D77800"), 0x00_00_78_D7);
        assert_eq!(Config::parse_color("D77800"), 0x00_00_78_D7);
        assert_eq!(Config::parse_color("#FFFFFF"), 0x00_FF_FF_FF);
        assert_eq!(Config::parse_color("#000000"), 0x00_00_00_00);
        // Bad input → 0 (safe default), not panic
        assert_eq!(Config::parse_color("nope"), 0);
        assert_eq!(Config::parse_color("#abc"), 0);
        assert_eq!(Config::parse_color("aéaaa"), 0);
        assert_eq!(Config::parse_color("#aéaaa"), 0);
        assert_eq!(Config::parse_color("#GG0000"), 0);
    }
}
