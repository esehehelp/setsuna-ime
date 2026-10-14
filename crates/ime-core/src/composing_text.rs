//! Romaji → hiragana composing buffer.
//!
//! Romaji composition is IME behaviour, so it lives in the frontend; the
//! engine only ever sees the finished hiragana. The buffer keeps the raw
//! input text and derives hiragana + pending romaji after each edit. That
//! keeps `nn` disambiguation deterministic across typing and commit.
//! Backspace and cursor moves first rewrite the raw text as the shown kana
//! (plus pending romaji), so they act on what the user sees: one kana, not
//! one romaji letter hidden behind it.
//!
//! With a mixed-input model (`set_mixed`), the letters are split into
//! Japanese and English spans (`crate::mixed`): the reading keeps the
//! English words as typed (the engine passes them through), letters keep
//! their case and typed ASCII symbols are shown per span (`.` in
//! `example.com`, 。 after Japanese).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::mixed::{Lang, MixedModel};

const ROMAJI_TABLE: &[(&str, &str)] = &[
    // four-letter — explicit small-tsu spellings (`ltsu` / `xtsu`). Without
    // these the 3-letter `tsu` → つ entry would win and leave the `l`/`x`
    // prefix stranded in the buffer.
    ("ltsu", "っ"),
    ("xtsu", "っ"),
    // three-letter (yōon + small kana with y-row + extended th/dh/v)
    ("shi", "し"),
    ("chi", "ち"),
    ("tsu", "つ"),
    ("sha", "しゃ"),
    ("shu", "しゅ"),
    ("sho", "しょ"),
    ("cha", "ちゃ"),
    ("chu", "ちゅ"),
    ("cho", "ちょ"),
    ("tya", "ちゃ"),
    ("tyu", "ちゅ"),
    ("tyo", "ちょ"),
    ("sya", "しゃ"),
    ("syu", "しゅ"),
    ("syo", "しょ"),
    ("nya", "にゃ"),
    ("nyu", "にゅ"),
    ("nyo", "にょ"),
    ("hya", "ひゃ"),
    ("hyu", "ひゅ"),
    ("hyo", "ひょ"),
    ("mya", "みゃ"),
    ("myu", "みゅ"),
    ("myo", "みょ"),
    ("rya", "りゃ"),
    ("ryu", "りゅ"),
    ("ryo", "りょ"),
    ("gya", "ぎゃ"),
    ("gyu", "ぎゅ"),
    ("gyo", "ぎょ"),
    ("bya", "びゃ"),
    ("byu", "びゅ"),
    ("byo", "びょ"),
    ("pya", "ぴゃ"),
    ("pyu", "ぴゅ"),
    ("pyo", "ぴょ"),
    ("kya", "きゃ"),
    ("kyu", "きゅ"),
    ("kyo", "きょ"),
    ("jya", "じゃ"),
    ("jyu", "じゅ"),
    ("jyo", "じょ"),
    // small-kana — `x`- and `l`-prefixed aliases. MS-IME defaults to `l`
    // for small kana, so both families are accepted.
    ("xtu", "っ"),
    ("ltu", "っ"),
    ("xya", "ゃ"),
    ("xyu", "ゅ"),
    ("xyo", "ょ"),
    ("xwa", "ゎ"),
    ("lya", "ゃ"),
    ("lyu", "ゅ"),
    ("lyo", "ょ"),
    ("lwa", "ゎ"),
    // th/dh-row — small vowel on て / で base (katakana カ spellings).
    ("tha", "てぁ"),
    ("thi", "てぃ"),
    ("thu", "てゅ"),
    ("the", "てぇ"),
    ("tho", "てょ"),
    ("dha", "でぁ"),
    ("dhi", "でぃ"),
    ("dhu", "でゅ"),
    ("dhe", "でぇ"),
    ("dho", "でょ"),
    // v-row yōon
    ("vya", "ゔゃ"),
    ("vyu", "ゔゅ"),
    ("vyo", "ゔょ"),
    // extended foreign-sound yōon that MS-IME accepts but were missing here:
    // ち/し + small え, つ + small vowel, ふ + small y-row. Without them these
    // sequences stranded in the buffer and committed as raw latin.
    ("che", "ちぇ"),
    ("she", "しぇ"),
    ("tsa", "つぁ"),
    ("tsi", "つぃ"),
    ("tse", "つぇ"),
    ("tso", "つぉ"),
    ("fya", "ふゃ"),
    ("fyu", "ふゅ"),
    ("fyo", "ふょ"),
    // two-letter
    ("ka", "か"),
    ("ki", "き"),
    ("ku", "く"),
    ("ke", "け"),
    ("ko", "こ"),
    ("sa", "さ"),
    ("si", "し"),
    ("su", "す"),
    ("se", "せ"),
    ("so", "そ"),
    ("ta", "た"),
    ("ti", "ち"),
    ("tu", "つ"),
    ("te", "て"),
    ("to", "と"),
    ("na", "な"),
    ("ni", "に"),
    ("nu", "ぬ"),
    ("ne", "ね"),
    ("no", "の"),
    ("ha", "は"),
    ("hi", "ひ"),
    ("hu", "ふ"),
    ("he", "へ"),
    ("ho", "ほ"),
    ("ma", "ま"),
    ("mi", "み"),
    ("mu", "む"),
    ("me", "め"),
    ("mo", "も"),
    ("ya", "や"),
    ("yi", "い"),
    ("yu", "ゆ"),
    ("ye", "いぇ"),
    ("yo", "よ"),
    ("ra", "ら"),
    ("ri", "り"),
    ("ru", "る"),
    ("re", "れ"),
    ("ro", "ろ"),
    ("wa", "わ"),
    ("wi", "うぃ"),
    ("wu", "う"),
    ("we", "うぇ"),
    ("wo", "を"),
    ("ga", "が"),
    ("gi", "ぎ"),
    ("gu", "ぐ"),
    ("ge", "げ"),
    ("go", "ご"),
    ("za", "ざ"),
    ("zi", "じ"),
    ("zu", "ず"),
    ("ze", "ぜ"),
    ("zo", "ぞ"),
    ("da", "だ"),
    ("di", "ぢ"),
    ("du", "づ"),
    ("de", "で"),
    ("do", "ど"),
    ("ba", "ば"),
    ("bi", "び"),
    ("bu", "ぶ"),
    ("be", "べ"),
    ("bo", "ぼ"),
    ("pa", "ぱ"),
    ("pi", "ぴ"),
    ("pu", "ぷ"),
    ("pe", "ぺ"),
    ("po", "ぽ"),
    ("fa", "ふぁ"),
    ("fi", "ふぃ"),
    ("fu", "ふ"),
    ("fe", "ふぇ"),
    ("fo", "ふぉ"),
    ("ja", "じゃ"),
    ("ji", "じ"),
    ("ju", "じゅ"),
    ("je", "じぇ"),
    ("jo", "じょ"),
    ("va", "ゔぁ"),
    ("vi", "ゔぃ"),
    ("vu", "ゔ"),
    ("ve", "ゔぇ"),
    ("vo", "ゔぉ"),
    ("xa", "ぁ"),
    ("xi", "ぃ"),
    ("xu", "ぅ"),
    ("xe", "ぇ"),
    ("xo", "ぉ"),
    ("la", "ぁ"),
    ("li", "ぃ"),
    ("lu", "ぅ"),
    ("le", "ぇ"),
    ("lo", "ぉ"),
    // Note: "nn" intentionally NOT in the table; the raw converter treats it
    // as a hard ん boundary, so `nne` becomes んえ.
    // single vowels
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
];

#[derive(Debug, Clone, Default)]
pub struct ComposingText {
    raw: String,
    hiragana: String,
    pending: String,
    /// Cursor measured in UTF-8 code points (chars) into `raw`.
    cursor: usize,
    mixed: Option<MixedInput>,
}

/// Mixed Japanese / English input: the model splitting the typed text, what
/// each ASCII symbol shows in Japanese, and the text before the caret.
#[derive(Debug, Clone)]
struct MixedInput {
    model: Arc<MixedModel>,
    symbols: HashMap<char, String>,
    before: String,
    casing: crate::mixed::Casing,
}

/// Chars of the text before the caret the mixed model looks at.
const MIXED_BEFORE: usize = 32;

impl ComposingText {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Completed hiragana portion (used as the ONNX model's reading input
    /// during live conversion — pending romaji is excluded on purpose).
    pub fn hiragana(&self) -> &str {
        &self.hiragana
    }

    /// Un-converted romaji tail (e.g. `"k"` while the user is in the middle
    /// of typing `kak`). Shown verbatim at the end of the preedit and
    /// excluded from backend conversion.
    pub fn pending_romaji(&self) -> &str {
        &self.pending
    }

    /// Split the input with `model` (`None`: every letter is romaji);
    /// `symbols` is `[frontend.symbols]`, what ASCII symbols show in
    /// Japanese, `casing` what the English is cased.
    pub fn set_mixed(
        &mut self,
        model: Option<Arc<MixedModel>>,
        symbols: &BTreeMap<String, String>,
        casing: crate::mixed::Casing,
    ) {
        self.mixed = model.map(|model| MixedInput {
            model,
            symbols: crate::config::SYMBOL_CHARS
                .iter()
                .filter_map(|(name, c)| symbols.get(*name).map(|s| (*c, s.clone())))
                .collect(),
            before: String::new(),
            casing,
        });
        self.recompute(false);
    }

    pub fn is_mixed(&self) -> bool {
        self.mixed.is_some()
    }

    /// The text before the caret, for the mixed model.
    pub fn set_before(&mut self, before: &str) {
        if let Some(mixed) = self.mixed.as_mut() {
            let n = before.chars().count();
            mixed.before = before
                .chars()
                .skip(n.saturating_sub(MIXED_BEFORE))
                .collect();
            self.recompute(false);
        }
    }

    /// The whole input read as romaji, English or not, with a pending `n`
    /// as ん (what F6–F8 convert, and the other reading of a mixed one).
    pub fn kana_reading(&self) -> String {
        let converted = convert_raw(&self.raw, true);
        converted.text + &converted.pending
    }

    /// Whether the whole input is one word the mixed model has seen as
    /// Latin (`made`, `google`), however it reads.
    pub fn is_known_word(&self) -> bool {
        self.mixed.as_ref().is_some_and(|mixed| {
            self.raw.chars().all(|c| c.is_ascii_alphabetic()) && mixed.model.knows(&self.raw)
        })
    }

    /// Whether the composition ends in an English word.
    pub fn ends_in_english(&self) -> bool {
        let Some(mixed) = &self.mixed else {
            return false;
        };
        mixed
            .model
            .cased(&self.raw, Some(&mixed.before), true, mixed.casing)
            .1
            .last()
            .is_some_and(|span| span.lang == Lang::En && span.end == self.raw.chars().count())
    }

    /// Whether the composition ends in a lone `i`, which the word after it
    /// may make English (`i think`: `mixed::ends_in_lone_i`).
    pub fn ends_in_lone_i(&self) -> bool {
        self.mixed.is_some() && crate::mixed::ends_in_lone_i(&self.raw)
    }

    /// Whether the composition ends in a space (one typed after English).
    pub fn ends_with_space(&self) -> bool {
        self.raw.ends_with(' ')
    }

    /// Drop the spaces at the end of the composition, the cursor at its end.
    pub fn trim_end_spaces(&mut self) {
        let trimmed = self.raw.trim_end_matches(' ').len();
        self.raw.truncate(trimmed);
        self.cursor = self.raw.chars().count();
        self.recompute(false);
    }

    /// Whether the char left of the cursor is a digit.
    pub fn digit_before_cursor(&self) -> bool {
        self.cursor > 0
            && self
                .raw
                .chars()
                .nth(self.cursor - 1)
                .is_some_and(|c| c.is_ascii_digit())
    }

    pub fn input_char(&mut self, c: char) {
        // Mixed input keeps the case: Shift marks English.
        let c = if self.mixed.is_some() {
            c
        } else {
            c.to_ascii_lowercase()
        };
        if !c.is_ascii() {
            return;
        }
        self.insert_raw_at_cursor(c);
        self.recompute(false);
    }

    fn insert_raw_at_cursor(&mut self, c: char) {
        let byte_pos = byte_pos_for_char(&self.raw, self.cursor);
        self.raw.insert(byte_pos, c);
        self.cursor += 1;
    }

    fn recompute(&mut self, flush_final_n: bool) {
        let converted = self.convert(&self.raw, flush_final_n);
        self.hiragana = converted.text;
        self.pending = converted.pending;
    }

    /// Delete the character shown left of the cursor: a whole kana (`moji` →
    /// `も`, not `もj`), or one letter of still-pending romaji (`kak` → `か`).
    pub fn delete_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.freeze_to_preedit();
        let prev_char_start = byte_pos_for_char(&self.raw, self.cursor - 1);
        let next_char_start = byte_pos_for_char(&self.raw, self.cursor);
        self.raw.drain(prev_char_start..next_char_start);
        self.cursor -= 1;
        self.recompute(false);
    }

    /// Delete the character shown right of the cursor (Delete).
    pub fn delete_right(&mut self) {
        self.freeze_to_preedit();
        let total = self.raw.chars().count();
        if self.cursor >= total {
            return;
        }
        let start = byte_pos_for_char(&self.raw, self.cursor);
        let end = byte_pos_for_char(&self.raw, self.cursor + 1);
        self.raw.drain(start..end);
        self.recompute(false);
    }

    /// Put the cursor at the start (`false`) or the end (`true`).
    pub fn move_cursor_to_edge(&mut self, end: bool) {
        self.freeze_to_preedit();
        self.cursor = if end { self.raw.chars().count() } else { 0 };
    }

    /// Whether the cursor is after the last character.
    pub fn cursor_at_end(&self) -> bool {
        self.cursor == self.raw.chars().count()
    }

    /// The cursor as a char offset into [`Self::preedit`].
    pub fn caret(&self) -> usize {
        let left = self.convert(
            &self.raw[..byte_pos_for_char(&self.raw, self.cursor)],
            false,
        );
        left.text.chars().count() + left.pending.chars().count()
    }

    /// The whole reading with a pending `n` as ん and other pending romaji
    /// kept (what F6–F8 convert), leaving the buffer as it is.
    pub fn full_reading(&self) -> String {
        let converted = self.convert(&self.raw, true);
        converted.text + &converted.pending
    }

    /// The romaji typed for the composition (what F9 / F10 spell): the raw
    /// input, with any kana a cursor edit froze into it spelled back.
    pub fn romaji(&self) -> String {
        crate::char_forms::romaji_of(&self.raw)
    }

    /// Move the cursor by `delta` characters of the preedit, so it steps over
    /// kana rather than over the romaji letters behind them.
    pub fn move_cursor(&mut self, delta: i32) {
        self.freeze_to_preedit();
        let total = self.raw.chars().count() as i32;
        let new_cursor = (self.cursor as i32 + delta).clamp(0, total) as usize;
        self.cursor = new_cursor;
    }

    /// Rewrite `raw` as exactly what the preedit shows (converted kana plus
    /// any still-pending romaji, on each side of the cursor), keeping the
    /// cursor where it is on screen. After this, one `raw` char is one shown
    /// char. Pending romaji stays raw, so a trailing `n` or `k` still combines
    /// with the next letter typed.
    fn freeze_to_preedit(&mut self) {
        let split = byte_pos_for_char(&self.raw, self.cursor);
        let left = self.convert(&self.raw[..split], false);
        let right = self.convert(&self.raw[split..], false);
        let mut raw = left.text;
        raw.push_str(&left.pending);
        self.cursor = raw.chars().count();
        raw.push_str(&right.text);
        raw.push_str(&right.pending);
        self.raw = raw;
    }

    /// Preedit text = converted raw text plus its pending romaji tail.
    pub fn preedit(&self) -> String {
        if self.cursor == self.raw.chars().count() {
            return format!("{}{}", self.hiragana, self.pending);
        }
        let byte_pos = byte_pos_for_char(&self.raw, self.cursor);
        let left = self.convert(&self.raw[..byte_pos], false);
        let right = self.convert(&self.raw[byte_pos..], false);
        let mut out =
            String::with_capacity(left.text.len() + left.pending.len() + right.text.len());
        out.push_str(&left.text);
        out.push_str(&left.pending);
        out.push_str(&right.text);
        out.push_str(&right.pending);
        out
    }

    /// Whether mixed input reads the composition otherwise once typing is
    /// done than while it is typed (`ok`: お and a pending k while typed,
    /// OK when done), a pending `n` read as ん either way: the candidates
    /// of the reading typed so far are then not what Enter commits.
    pub fn reads_otherwise_when_done(&self) -> bool {
        if self.mixed.is_none() {
            return false;
        }
        let read = |c: Converted| {
            let pending = if c.pending == "n" { "ん" } else { &c.pending };
            c.text + pending
        };
        read(self.convert(&self.raw, true)) != read(self.convert(&self.raw, false))
    }

    /// Commit flushes a pending raw `n` as ん, appends any other pending
    /// partial romaji verbatim, then resets.
    pub fn commit(&mut self) -> String {
        let converted = self.convert(&self.raw, true);
        let mut out = converted.text;
        out.push_str(&converted.pending);
        self.reset();
        out
    }

    /// The raw romaji tail to append AFTER a chosen candidate has replaced the
    /// hiragana reading (so typing `neko` + `k` + Enter on a candidate commits
    /// `猫k`, not `猫`). A pending `n` becomes ん; anything else stays raw. The
    /// superseded hiragana is discarded by the caller; this only resets state.
    pub fn commit_tail(&mut self) -> String {
        // Use the reading's existing tail. Re-segmenting mixed input with
        // fin=false could reinterpret an English word frozen by Space and
        // append letters that are already part of the chosen candidate.
        let tail = if self.pending == "n" {
            "ん".to_string()
        } else {
            self.pending.clone()
        };
        self.reset();
        tail
    }

    /// Flush pending `n` as ん and return the full hiragana text, leaving
    /// the buffer otherwise in place for potential further editing (callers
    /// can `reset()` themselves). Used by the conversion path so the ONNX
    /// model sees a clean reading even if the user pressed Space before
    /// typing the final vowel.
    pub fn commit_reading(&mut self) -> String {
        let converted = self.convert(&self.raw, true);
        let reading = converted.text.clone();
        self.raw = converted.text;
        self.raw.push_str(&converted.pending);
        self.cursor = self.raw.chars().count();
        // typing is done for this conversion: a mixed input reads as it did
        // above (`how` English, not ほ + a pending w)
        self.recompute(true);
        reading
    }

    /// Drop any pending romaji (flushing `n` → ん first, unless already
    /// converted) and splice the given kana at the end. Used for punctuation
    /// (`,` → `、`, `.` → `。`) so the composition keeps going instead of
    /// auto-committing the half-width character.
    pub fn flush_romaji_and_insert(&mut self, kana: &str) {
        let converted = self.convert(&self.raw, false);
        let mut raw = converted.text;
        if converted.pending == "n" {
            raw.push('ん');
        }
        raw.push_str(kana);
        self.raw = raw;
        self.cursor = self.raw.chars().count();
        self.recompute(false);
    }

    pub fn reset(&mut self) {
        self.raw.clear();
        self.hiragana.clear();
        self.pending.clear();
        self.cursor = 0;
    }

    /// `raw` as shown: romaji as kana, or with a mixed model its spans.
    fn convert(&self, raw: &str, flush_final_n: bool) -> Converted {
        let Some(mixed) = &self.mixed else {
            return convert_raw(raw, flush_final_n);
        };
        let (chars, spans) =
            mixed
                .model
                .cased(raw, Some(&mixed.before), flush_final_n, mixed.casing);
        let (text, pending) =
            crate::mixed::render_parts(&chars, &spans, flush_final_n, &mixed.symbols);
        Converted { text, pending }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Converted {
    text: String,
    pending: String,
}

fn convert_raw(raw: &str, flush_final_n: bool) -> Converted {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = Converted {
        text: String::new(),
        pending: String::new(),
    };
    for unit in romaji_units(&chars, flush_final_n) {
        match unit.kind {
            UnitKind::Kana(kana) => out.text.push_str(kana),
            UnitKind::Char(c) => out.text.push(c),
            UnitKind::Pending => out.pending = chars[unit.start..unit.end].iter().collect(),
        }
    }
    out
}

/// The kana romaji still being typed (`k`, `ky`, `n`) may become: the
/// first kana of each table entry it starts, ん for `n`, っ for a lone
/// consonant (doubled next).
pub(crate) fn pending_kana(pending: &str) -> Vec<char> {
    let pending = pending.to_ascii_lowercase();
    let mut out: Vec<char> = ROMAJI_TABLE
        .iter()
        .filter(|(romaji, _)| romaji.starts_with(&pending))
        .filter_map(|(_, kana)| kana.chars().next())
        .collect();
    match pending.as_str() {
        "n" => out.push('ん'),
        p if p.len() == 1 && !"aiueo".contains(p) => out.push('っ'),
        _ => {}
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// One step of reading romaji: the chars `start..end` of the input and
/// what they become.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RomajiUnit {
    pub start: usize,
    pub end: usize,
    pub kind: UnitKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitKind {
    /// A syllable (`ka` → か, `-` → ー, a doubled consonant's first letter
    /// → っ).
    Kana(&'static str),
    /// A char kept as it is: a non-letter (digit, symbol) or a letter that
    /// starts no romaji (lowercased).
    Char(char),
    /// The end of the input, a romaji not yet complete (`k`, `ky`, `n`).
    Pending,
}

/// `chars` read as romaji, left to right (letters in either case). A final
/// `n` is ん when `flush_final_n`, else pending.
pub(crate) fn romaji_units(chars: &[char], flush_final_n: bool) -> Vec<RomajiUnit> {
    let chars: Vec<char> = chars.iter().map(|c| c.to_ascii_lowercase()).collect();
    let mut units = Vec::new();
    let mut push = |start: usize, len: usize, kind: UnitKind| {
        units.push(RomajiUnit {
            start,
            end: start + len,
            kind,
        })
    };
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '-' {
            push(i, 1, UnitKind::Kana("ー"));
            i += 1;
            continue;
        }
        if !c.is_ascii_alphabetic() {
            push(i, 1, UnitKind::Char(c));
            i += 1;
            continue;
        }

        if c == 'n' {
            if i + 1 >= chars.len() {
                if flush_final_n {
                    push(i, 1, UnitKind::Kana("ん"));
                } else {
                    push(i, 1, UnitKind::Pending);
                }
                i += 1;
                continue;
            }
            let next = chars[i + 1];
            if next == 'n' || next == '\'' {
                push(i, 2, UnitKind::Kana("ん"));
                i += 2;
                continue;
            }
            if !matches!(next, 'a' | 'i' | 'u' | 'e' | 'o' | 'y') {
                push(i, 1, UnitKind::Kana("ん"));
                i += 1;
                continue;
            }
        }

        if i + 1 < chars.len()
            && chars[i] == chars[i + 1]
            && !matches!(c, 'a' | 'i' | 'u' | 'e' | 'o' | 'n')
        {
            push(i, 1, UnitKind::Kana("っ"));
            i += 1;
            continue;
        }

        if let Some((romaji, kana)) = ROMAJI_TABLE
            .iter()
            .find(|(romaji, _)| matches_at(&chars, i, romaji))
            .copied()
        {
            let len = romaji.chars().count();
            push(i, len, UnitKind::Kana(kana));
            i += len;
            continue;
        }

        let rest = chars[i..].iter().collect::<String>();
        if rest.chars().all(|ch| ch.is_ascii_alphabetic())
            && ROMAJI_TABLE.iter().any(|(r, _)| r.starts_with(&rest))
        {
            push(i, chars.len() - i, UnitKind::Pending);
            break;
        }

        push(i, 1, UnitKind::Char(c));
        i += 1;
    }
    units
}

fn matches_at(chars: &[char], start: usize, key: &str) -> bool {
    let key_len = key.chars().count();
    if start + key_len > chars.len() {
        return false;
    }
    chars[start..start + key_len]
        .iter()
        .copied()
        .eq(key.chars())
}

fn byte_pos_for_char(s: &str, char_pos: usize) -> usize {
    s.char_indices()
        .nth(char_pos)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_all(s: &str) -> ComposingText {
        let mut c = ComposingText::new();
        for ch in s.chars() {
            c.input_char(ch);
        }
        c
    }

    fn commit_all(s: &str) -> String {
        type_all(s).commit()
    }

    #[test]
    fn basic_vowels() {
        assert_eq!(commit_all("aiueo"), "あいうえお");
    }

    #[test]
    fn konnichiha() {
        assert_eq!(commit_all("konnichiha"), "こんいちは");
    }

    #[test]
    fn sokuon() {
        assert_eq!(commit_all("kitte"), "きって");
    }

    #[test]
    fn n_flush_on_commit() {
        assert_eq!(commit_all("shin"), "しん");
    }

    #[test]
    fn double_n_commit_is_single_n() {
        assert_eq!(commit_all("nn"), "ん");
    }

    #[test]
    fn nn_visible_in_preedit() {
        let mut c = ComposingText::new();
        c.input_char('n');
        assert_eq!(c.preedit(), "n");
        c.input_char('n');
        assert_eq!(c.preedit(), "ん");
    }

    #[test]
    fn nn_then_vowel_is_hard_n_boundary() {
        assert_eq!(commit_all("nna"), "んあ");
        assert_eq!(commit_all("nni"), "んい");
        assert_eq!(commit_all("nne"), "んえ");
    }

    #[test]
    fn nn_then_yoon() {
        assert_eq!(commit_all("nnyo"), "んよ");
        assert_eq!(commit_all("nnya"), "んや");
    }

    #[test]
    fn nnn_is_one_kana_plus_pending() {
        // "nn" is a hard ん boundary; the trailing single n flushes on commit.
        assert_eq!(commit_all("nnn"), "んん");
    }

    #[test]
    fn nnnn_is_two_kana() {
        // Two complete `nn` pairs.
        assert_eq!(commit_all("nnnn"), "んん");
    }

    #[test]
    fn nn_then_consonant_does_not_double_flush() {
        assert_eq!(commit_all("nnka"), "んか");
        assert_eq!(commit_all("nnta"), "んた");
    }

    #[test]
    fn nn_choon_is_n_choon() {
        // `n-` → んー. `nn-` should also be んー (single ん, single ー).
        assert_eq!(commit_all("nn-"), "んー");
    }

    #[test]
    fn nnk_a_combined_sequence() {
        assert_eq!(commit_all("nnka"), "んか");
    }

    #[test]
    fn triple_n_keeps_legacy_konnichiha_shape() {
        assert_eq!(commit_all("konnnichiha"), "こんにちは");
    }

    #[test]
    fn shinbun() {
        assert_eq!(commit_all("shinbun"), "しんぶん");
    }

    #[test]
    fn preedit_includes_pending() {
        let mut c = ComposingText::new();
        c.input_char('k');
        assert_eq!(c.preedit(), "k");
        c.input_char('o');
        assert_eq!(c.preedit(), "こ");
    }

    #[test]
    fn choon_single_hyphen() {
        assert_eq!(commit_all("ra-men"), "らーめん");
    }

    #[test]
    fn choon_double_hyphen_is_two_marks_not_sokuon() {
        assert_eq!(commit_all("a--"), "あーー");
    }

    #[test]
    fn n_before_choon_flushes() {
        assert_eq!(commit_all("san-"), "さんー");
    }

    // ---- small-kana family ----

    #[test]
    fn l_prefix_small_vowels() {
        assert_eq!(commit_all("la"), "ぁ");
        assert_eq!(commit_all("li"), "ぃ");
        assert_eq!(commit_all("lu"), "ぅ");
        assert_eq!(commit_all("le"), "ぇ");
        assert_eq!(commit_all("lo"), "ぉ");
    }

    #[test]
    fn x_prefix_small_vowels_still_work() {
        assert_eq!(commit_all("xa"), "ぁ");
        assert_eq!(commit_all("xi"), "ぃ");
        assert_eq!(commit_all("xu"), "ぅ");
        assert_eq!(commit_all("xe"), "ぇ");
        assert_eq!(commit_all("xo"), "ぉ");
    }

    #[test]
    fn l_prefix_small_y_row() {
        assert_eq!(commit_all("lya"), "ゃ");
        assert_eq!(commit_all("lyu"), "ゅ");
        assert_eq!(commit_all("lyo"), "ょ");
    }

    #[test]
    fn small_tsu_spellings() {
        assert_eq!(commit_all("ltu"), "っ");
        assert_eq!(commit_all("xtu"), "っ");
        assert_eq!(commit_all("ltsu"), "っ");
        assert_eq!(commit_all("xtsu"), "っ");
    }

    #[test]
    fn small_wa() {
        assert_eq!(commit_all("lwa"), "ゎ");
        assert_eq!(commit_all("xwa"), "ゎ");
    }

    #[test]
    fn modern_wi_we() {
        assert_eq!(commit_all("wi"), "うぃ");
        assert_eq!(commit_all("we"), "うぇ");
    }

    #[test]
    fn ll_triggers_sokuon_then_small_vowel() {
        // Matches MS-IME: first `l` becomes っ, then `la` → ぁ.
        assert_eq!(commit_all("lla"), "っぁ");
        assert_eq!(commit_all("xxi"), "っぃ");
    }

    // ---- extended consonants ----

    #[test]
    fn th_row() {
        assert_eq!(commit_all("thi"), "てぃ");
        assert_eq!(commit_all("tha"), "てぁ");
        assert_eq!(commit_all("thu"), "てゅ");
    }

    #[test]
    fn dh_row() {
        assert_eq!(commit_all("dhi"), "でぃ");
        assert_eq!(commit_all("dha"), "でぁ");
    }

    #[test]
    fn v_row_full() {
        // All five vowels + yōon, hiragana form of the MS-IME ヴ mapping.
        assert_eq!(commit_all("va"), "ゔぁ");
        assert_eq!(commit_all("vi"), "ゔぃ");
        assert_eq!(commit_all("vu"), "ゔ");
        assert_eq!(commit_all("ve"), "ゔぇ");
        assert_eq!(commit_all("vo"), "ゔぉ");
        assert_eq!(commit_all("vya"), "ゔゃ");
        assert_eq!(commit_all("vyu"), "ゔゅ");
        assert_eq!(commit_all("vyo"), "ゔょ");
    }

    #[test]
    fn vy_alone_stays_pending_until_vowel() {
        // `vy` without a trailing vowel must not auto-collapse to ゔ — a
        // buffered `vy` waits for the next vowel to disambiguate ゃ/ゅ/ょ.
        // (Commit without a vowel simply drops the incomplete romaji.)
        let mut c = ComposingText::new();
        c.input_char('v');
        c.input_char('y');
        assert_eq!(c.preedit(), "vy");
        c.input_char('u');
        assert_eq!(c.commit(), "ゔゅ");
    }

    #[test]
    fn fa_fi_fe_fo() {
        assert_eq!(commit_all("fa"), "ふぁ");
        assert_eq!(commit_all("fi"), "ふぃ");
        assert_eq!(commit_all("fe"), "ふぇ");
        assert_eq!(commit_all("fo"), "ふぉ");
    }

    #[test]
    fn ja_family() {
        assert_eq!(commit_all("ja"), "じゃ");
        assert_eq!(commit_all("ji"), "じ");
        assert_eq!(commit_all("ju"), "じゅ");
        assert_eq!(commit_all("jo"), "じょ");
        assert_eq!(commit_all("jya"), "じゃ");
        assert_eq!(commit_all("jyu"), "じゅ");
    }

    // ---- regression cases ----

    #[test]
    fn tsu_still_works_with_four_letter_prefix_in_table() {
        // Adding `ltsu`/`xtsu` must not break plain `tsu`.
        assert_eq!(commit_all("tsu"), "つ");
        assert_eq!(commit_all("matsuri"), "まつり");
    }

    #[test]
    fn konpyu_ta_choon_chain() {
        assert_eq!(commit_all("konpyu-ta-"), "こんぴゅーたー");
    }

    // ---- commit must not drop un-converted romaji ----

    #[test]
    fn garbage_does_not_block_embedded_conversion() {
        // The reported bug: a dead leading prefix ("cw") used to swallow the
        // whole buffer, so the embedded わ never converted and Enter dropped
        // it all. Now stuck chars flush as latin and わ still converts; the
        // doubled consonant still sokuons normally (ww → っ).
        assert_eq!(commit_all("cwwasd"), "cっわsd");
        assert_eq!(commit_all("abc"), "あbc");
        assert_eq!(commit_all("asdf"), "あsdf");
    }

    #[test]
    fn sokuon_still_standard_after_flush() {
        assert_eq!(commit_all("wwa"), "っわ");
        assert_eq!(commit_all("watto"), "わっと");
        assert_eq!(commit_all("xqkitte"), "xqきって");
    }

    #[test]
    fn commit_keeps_partial_trailing_romaji() {
        assert_eq!(commit_all("kak"), "かk");
        assert_eq!(commit_all("ky"), "ky");
    }

    #[test]
    fn commit_tail_after_candidate() {
        // hiragana replaced by a candidate; tail keeps the raw romaji, and a
        // pending n becomes ん.
        let mut c = type_all("nekok");
        assert_eq!(c.commit_tail(), "k");
        let mut c = type_all("nekon");
        assert_eq!(c.commit_tail(), "ん");
        let mut c = type_all("neko");
        assert_eq!(c.commit_tail(), "");
    }

    #[test]
    fn commit_tail_uses_the_frozen_mixed_reading() {
        let model = Arc::new(crate::mixed::tests::model());
        for word in [
            "how", "made", "hello", "google", "github", "k", "go", "Tokyo",
        ] {
            let mut c = ComposingText::new();
            c.set_mixed(
                Some(model.clone()),
                &crate::config::default_symbol_map(),
                crate::mixed::Casing::default(),
            );
            for ch in word.chars() {
                c.input_char(ch);
            }
            c.commit_reading();
            let expected = if c.pending == "n" {
                "ん".to_string()
            } else {
                c.pending.clone()
            };
            assert_eq!(c.commit_tail(), expected, "{word}");
            assert!(c.is_empty());
        }
    }

    // ---- backspace / cursor act on shown characters ----

    fn backspace(c: &mut ComposingText, n: usize) {
        for _ in 0..n {
            c.delete_left();
        }
    }

    #[test]
    fn backspace_deletes_a_whole_kana() {
        let mut c = type_all("moji");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "も");
        assert_eq!(c.hiragana(), "も");
        assert_eq!(c.commit(), "も");
    }

    #[test]
    fn backspace_deletes_one_small_kana_of_a_yoon() {
        let mut c = type_all("kya");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "き");
    }

    #[test]
    fn backspace_deletes_pending_romaji_letter_by_letter() {
        let mut c = type_all("kaky");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "かk");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "か");
    }

    #[test]
    fn pending_n_survives_backspace_elsewhere() {
        // `kan` + BS drops the pending n; `kann` + BS drops the whole ん.
        let mut c = type_all("kan");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "か");
        let mut c = type_all("kanji");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "かん");
        c.input_char('a');
        assert_eq!(c.commit(), "かんあ");
    }

    #[test]
    fn typing_after_backspace_continues_romaji() {
        let mut c = type_all("mojik");
        backspace(&mut c, 2);
        assert_eq!(c.preedit(), "も");
        for ch in "ji".chars() {
            c.input_char(ch);
        }
        assert_eq!(c.commit(), "もじ");
    }

    #[test]
    fn backspace_to_empty() {
        let mut c = type_all("moji");
        backspace(&mut c, 3);
        assert!(c.is_empty());
        assert_eq!(c.preedit(), "");
    }

    #[test]
    fn cursor_moves_over_kana() {
        let mut c = type_all("moji");
        c.move_cursor(-1);
        assert_eq!(c.preedit(), "もじ");
        backspace(&mut c, 1);
        assert_eq!(c.preedit(), "じ");
        for ch in "ka".chars() {
            c.input_char(ch);
        }
        assert_eq!(c.commit(), "かじ");
    }

    #[test]
    fn cursor_move_keeps_pending_romaji_combinable() {
        let mut c = type_all("kan");
        c.move_cursor(0);
        c.input_char('a');
        assert_eq!(c.commit(), "かな");
    }

    // ---- extended foreign-sound romaji ----

    #[test]
    fn extended_yoon() {
        assert_eq!(commit_all("che"), "ちぇ");
        assert_eq!(commit_all("she"), "しぇ");
        assert_eq!(commit_all("tsa"), "つぁ");
        assert_eq!(commit_all("tsi"), "つぃ");
        assert_eq!(commit_all("tse"), "つぇ");
        assert_eq!(commit_all("tso"), "つぉ");
        assert_eq!(commit_all("fya"), "ふゃ");
        assert_eq!(commit_all("fyu"), "ふゅ");
        assert_eq!(commit_all("fyo"), "ふょ");
    }

    #[test]
    fn extended_does_not_break_base_forms() {
        assert_eq!(commit_all("tsu"), "つ");
        assert_eq!(commit_all("chi"), "ち");
        assert_eq!(commit_all("shi"), "し");
        assert_eq!(commit_all("fu"), "ふ");
    }
}
