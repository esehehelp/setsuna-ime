//! IME-side post-processing of the engine's candidates (no inference here).
//!
//! The reading goes to the engine NFKC-normalized (child_daemon), so symbols
//! the engine splices back verbatim come out in their NFKC form (`（` -> `(`).
//! Two rules then run on every candidate, keyed on the reading as typed:
//!
//! 1. Symbol width: each typed symbol whose NFKC form differs (full-width
//!    `！（％`…) is found again in the candidate and written back as typed
//!    (`as_typed`), or forced to full / half width. Typed symbols are matched
//!    in order to the occurrences closest to their relative position in the
//!    reading; one the model dropped is skipped, and a symbol the model
//!    produced on its own (`例:(括))` for `れい：（かっこ）`) is left alone.
//!    With `auto`, the symbols go back half-width and `auto_width` then
//!    writes every digit, letter and symbol of the candidate in the width
//!    the text before the caret uses (see there).
//! 2. Reading in parentheses: a `(…)` group of the reading whose content is
//!    kana and equals the kana right before it (士幌町（しほろちょう）) is a
//!    reading annotation, so the same-numbered group of the candidate gets the
//!    typed kana back instead of a conversion. Skipped when the candidate's
//!    group count differs from the reading's.

use std::collections::HashMap;
use std::sync::OnceLock;

use unicode_normalization::UnicodeNormalization;

use crate::config::{FrontendCfg, SymbolWidth};

fn nfkc(c: char) -> String {
    c.to_string().nfkc().collect()
}

fn is_open(c: char) -> bool {
    matches!(c, '(' | '（')
}

fn is_close(c: char) -> bool {
    matches!(c, ')' | '）')
}

fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | 'ー')
}

/// Full-width form of an ASCII symbol (U+0021..U+007E -> U+FF01..U+FF5E).
fn to_full(c: char) -> char {
    match c {
        '!'..='~' if !c.is_ascii_alphanumeric() => {
            char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c)
        }
        _ => c,
    }
}

/// Top-level parenthesis groups: (open byte, close byte) pairs.
fn groups(s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut open: Option<usize> = None;
    for (i, c) in s.char_indices() {
        if is_open(c) && open.is_none() {
            open = Some(i);
        } else if is_close(c) {
            if let Some(o) = open.take() {
                out.push((o, i));
            }
        }
    }
    out
}

/// Cost of leaving a typed symbol without a match in the candidate. Above
/// the largest position cost (1.0), so a match is always preferred.
const SKIP_COST: f32 = 1.5;

/// Relative position of char `i` in a string of `n` chars, in (0, 1).
fn rel(i: usize, n: usize) -> f32 {
    (i as f32 + 0.5) / n as f32
}

fn restore_width(reading: &str, cand: &str, width: SymbolWidth) -> String {
    // typed symbols: relative position in the reading, the char typed and
    // the single char NFKC makes of it (what the engine saw)
    let n_reading = reading.chars().count();
    let typed: Vec<(f32, char, char)> = reading
        .chars()
        .enumerate()
        .filter(|&(_, c)| !c.is_alphanumeric() && !c.is_whitespace() && !is_kana(c))
        .filter_map(|(i, c)| {
            let n = nfkc(c);
            let mut it = n.chars();
            match (it.next(), it.next()) {
                (Some(m), None) if !m.is_alphanumeric() => Some((rel(i, n_reading), c, m)),
                _ => None,
            }
        })
        .collect();
    if typed.is_empty() {
        return cand.to_string();
    }
    let mut chars: Vec<char> = cand.chars().collect();
    // Either width of a typed symbol's NFKC form may come back (the engine
    // splices `!` verbatim but normalizes a converted `！`), and the engine
    // writes every `,` / `.` as `、` / `。` (so a typed `1，000` came back as
    // `1、000`).
    let matches = |t: &(f32, char, char), c: char| {
        c == t.2 || c == to_full(t.2) || (t.2 == ',' && c == '、') || (t.2 == '.' && c == '。')
    };
    let occ: Vec<usize> = (0..chars.len())
        .filter(|&j| typed.iter().any(|t| matches(t, chars[j])))
        .collect();
    // Order-preserving alignment of typed symbols to candidate occurrences,
    // minimizing the distance of their relative positions. A typed symbol
    // the model dropped is skipped instead of blocking the ones after it,
    // and a symbol the model wrote on its own stays unmatched (left as is).
    let (k_n, o_n) = (typed.len(), occ.len());
    let mut cost = vec![vec![0.0f32; o_n + 1]; k_n + 1];
    for k in 1..=k_n {
        cost[k][0] = k as f32 * SKIP_COST;
        for o in 1..=o_n {
            let mut best = (cost[k - 1][o] + SKIP_COST).min(cost[k][o - 1]);
            let t = &typed[k - 1];
            if matches(t, chars[occ[o - 1]]) {
                let d = (t.0 - rel(occ[o - 1], chars.len())).abs();
                best = best.min(cost[k - 1][o - 1] + d);
            }
            cost[k][o] = best;
        }
    }
    let (mut k, mut o) = (k_n, o_n);
    while k > 0 && o > 0 {
        let t = &typed[k - 1];
        let j = occ[o - 1];
        if matches(t, chars[j])
            && cost[k][o] == cost[k - 1][o - 1] + (t.0 - rel(j, chars.len())).abs()
        {
            chars[j] = match width {
                SymbolWidth::AsTyped => t.1,
                SymbolWidth::Full => to_full(t.2),
                SymbolWidth::Half | SymbolWidth::Auto => t.2,
            };
            k -= 1;
            o -= 1;
        } else if cost[k][o] == cost[k][o - 1] {
            o -= 1;
        } else {
            k -= 1;
        }
    }
    chars.into_iter().collect()
}

fn keep_reading_in_parens(reading: &str, cand: &str) -> String {
    let rg = groups(reading);
    if rg.is_empty() {
        return cand.to_string();
    }
    // which reading groups are reading annotations, and their typed content
    let keep: Vec<Option<&str>> = rg
        .iter()
        .map(|&(o, c)| {
            let inner = &reading[o + reading[o..].chars().next().unwrap().len_utf8()..c];
            let before = &reading[..o];
            (inner.chars().count() >= 2 && inner.chars().all(is_kana) && before.ends_with(inner))
                .then_some(inner)
        })
        .collect();
    if keep.iter().all(Option::is_none) {
        return cand.to_string();
    }
    let cg = groups(cand);
    if cg.len() != rg.len() {
        return cand.to_string();
    }
    let mut out = String::new();
    let mut last = 0;
    for (&(o, c), k) in cg.iter().zip(&keep) {
        if let Some(inner) = k {
            let open_len = cand[o..].chars().next().unwrap().len_utf8();
            out.push_str(&cand[last..o + open_len]);
            out.push_str(inner);
            last = c;
        }
    }
    out.push_str(&cand[last..]);
    out
}

/// A character that is neither a letter, digit, kanji nor kana: punctuation,
/// brackets, other symbols, whitespace.
fn is_symbol(c: char) -> bool {
    !c.is_alphanumeric() && !is_kana(c)
}

/// The class a symbol is counted in: its NFKC form, with the forms the
/// engine or the frontend rules swap between folded together.
fn symbol_class(c: char) -> char {
    let n = nfkc(c);
    let mut it = n.chars();
    let c = match (it.next(), it.next()) {
        (Some(m), None) => m,
        _ => c,
    };
    match c {
        '、' | ',' => ',',
        '。' | '.' => '.',
        '·' | '‧' | '⋅' | '∙' => '・',
        '〜' => '~',
        '‐' | '‑' | '‒' | '–' | '—' | '―' | '−' => '-',
        c if c.is_whitespace() => ' ',
        c => c,
    }
}

/// Drop the candidates that contain a symbol the user did not type: more of
/// a symbol class than the reading has. The model writes punctuation,
/// brackets and junk (`〓`) into readings that have none; with
/// Aster-v1.2-small すうじときごう converts to 数字と: | 数字と) | 数字と号 | …
/// and 数字と記号 is 7th. A typed ー also allows a `-` (3ー4 -> 3-4).
///
/// A `,` / `.` between digits is part of the number, not a symbol; the
/// engine writes those as `、` / `。` (6.2 -> 6。2), so they are put back.
pub fn drop_untyped_symbols(reading: &str, candidates: Vec<String>) -> Vec<String> {
    let mut typed: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for c in reading.chars() {
        if is_symbol(c) {
            *typed.entry(symbol_class(c)).or_default() += 1;
        } else if c == 'ー' {
            *typed.entry('-').or_default() += 1;
        }
    }
    candidates
        .into_iter()
        .filter_map(|cand| {
            let mut chars: Vec<char> = cand.chars().collect();
            let mut used: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
            for i in 0..chars.len() {
                let c = chars[i];
                if !is_symbol(c) {
                    continue;
                }
                let class = symbol_class(c);
                let between_digits = i > 0
                    && i + 1 < chars.len()
                    && chars[i - 1].is_ascii_digit()
                    && chars[i + 1].is_ascii_digit();
                if between_digits && matches!(class, ',' | '.') {
                    if matches!(c, '、' | '。') {
                        chars[i] = class;
                    }
                    continue;
                }
                let n = used.entry(class).or_default();
                *n += 1;
                if *n > typed.get(&class).copied().unwrap_or(0) {
                    return None;
                }
            }
            Some(chars.into_iter().collect())
        })
        .collect()
}

/// Drop the candidates with one kana of a script between kana of the other
/// on both sides (続投がキまっ, デるタ, 依頼がクるように): the model's
/// char-by-char output puts a kana in the wrong script, and no word is
/// written so. A particle between katakana (ゲームをプレイ) and the
/// prolonged sound mark stay, and so does a kana next to a kanji, a digit
/// or a symbol (カシの木, 寺ノ台). Over the engine's candidates of
/// wiki_fresh_v2 and AJIMEE it drops no gold, and 1.5% of what the
/// reranker's score gap (`RERANK_GAP`) leaves.
pub fn drop_script_flips(candidates: Vec<String>) -> Vec<String> {
    candidates
        .into_iter()
        .filter(|c| !has_script_flip(c))
        .collect()
}

fn has_script_flip(text: &str) -> bool {
    let hira = |c: char| matches!(c, '\u{3041}'..='\u{3096}');
    let kata = |c: char| matches!(c, '\u{30A1}'..='\u{30FA}' | 'ー');
    let chars: Vec<char> = text.chars().collect();
    chars.windows(3).any(|w| {
        let (p, c, n) = (w[0], w[1], w[2]);
        (hira(c) && kata(p) && kata(n) && !"をがにはのでとへもやか".contains(c))
            || (kata(c) && c != 'ー' && hira(p) && hira(n))
    })
}

/// Drop the candidates that change a number typed in digits: each run of
/// digits of the reading must be a run of the candidate, in order (more
/// runs, from kana, may come between). The model's CTC merges a repeated
/// digit (19911225 -> 199125) and swaps digits for kanji (2025・11・2二), so
/// a typed number is kept as it was typed, as Mozc keeps it.
pub fn drop_changed_digits(reading: &str, candidates: Vec<String>) -> Vec<String> {
    if !reading.chars().any(|c| to_half(c).is_ascii_digit()) {
        return candidates;
    }
    candidates
        .into_iter()
        .filter(|cand| keeps_digits(reading, cand))
        .collect()
}

/// Whether `candidate` keeps the numbers typed in digits in `reading`
/// (`drop_changed_digits`).
/// Drop the candidates whose sentence ends are not the reading's: its 。！？
/// in order (a period, full-width or half-width, as 。), no more and no
/// fewer. The model writes a char a frame, and a beam's runners-up swap
/// one, a 。 among them (の展覧会に出展した『1900年4月, からである、こうした
/// for からである。こうした, a 。 left out); a reading of several sentences
/// goes to it whole, so they come into the list. Over Shiran-v2 s7080's
/// wiki_fresh_v2 lists of several sentences, 34 of 1,055 had one in the
/// first 5 and none at the top; no gold goes.
pub fn drop_moved_sentence_ends(reading: &str, candidates: Vec<String>) -> Vec<String> {
    let want = sentence_ends(reading);
    candidates
        .into_iter()
        .filter(|c| sentence_ends(c) == want)
        .collect()
}

fn sentence_ends(s: &str) -> Vec<char> {
    s.chars()
        .filter_map(|c| match c {
            '。' | '｡' | '.' | '．' => Some('。'),
            '!' | '！' => Some('!'),
            '?' | '？' => Some('?'),
            _ => None,
        })
        .collect()
}

pub fn keeps_digits(reading: &str, candidate: &str) -> bool {
    let mut runs = digit_runs(candidate).into_iter();
    digit_runs(reading).iter().all(|t| runs.any(|r| r == *t))
}

/// The runs of digits of `s`, full-width ones as half-width.
fn digit_runs(s: &str) -> Vec<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut in_run = false;
    for c in s.chars().map(to_half) {
        if c.is_ascii_digit() {
            if !in_run {
                runs.push(String::new());
            }
            runs.last_mut().unwrap().push(c);
        }
        in_run = c.is_ascii_digit();
    }
    runs
}

/// A kanji numeral of a number below 万: a digit, 十, 百 or 千.
fn is_kanji_numeral(c: char) -> bool {
    matches!(
        c,
        '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千'
    )
}

/// Drop the candidates with a number spelled half in digits and half in
/// kanji, or with a 0 the reading does not have. The model writes a char a
/// frame, and its runners-up swap one digit of a number for a kanji one
/// (1六番街, 五0年, 50歩1百歩, 十3歳) or put a 0 before it (03年 for さんねん
/// after 196, 09月, 戦後0年). A digit before 千万億兆 stays (3千人, 1万5000),
/// and so does a 0 where the reading has ぜろ, れい or a typed 0. Readings
/// spelling their numbers in kana are AJIMEE's; over wiki_fresh_v2, whose
/// numbers are typed in digits, Bletila-v0.1-small's lists lost 1,090
/// candidates and no gold.
pub fn drop_broken_numbers(reading: &str, candidates: Vec<String>) -> Vec<String> {
    let zero_read = reading.contains("ぜろ")
        || reading.contains("れい")
        || reading.chars().any(|c| to_half(c) == '0');
    candidates
        .into_iter()
        .filter(|c| !breaks_a_number(c, zero_read))
        .collect()
}

fn breaks_a_number(candidate: &str, zero_read: bool) -> bool {
    let chars: Vec<char> = candidate.chars().map(to_half).collect();
    let mixed = chars.windows(2).any(|w| {
        (w[0].is_ascii_digit() && is_kanji_numeral(w[1]) && w[1] != '千')
            || (is_kanji_numeral(w[0]) && w[1].is_ascii_digit())
    });
    // a run of digits that starts with 0 (not a decimal's digits)
    let zero = !zero_read
        && chars.iter().enumerate().any(|(i, &c)| {
            c == '0' && (i == 0 || !(chars[i - 1].is_ascii_digit() || chars[i - 1] == '.'))
        });
    mixed || zero
}

/// What `order_engine_result` takes off the score of a candidate that
/// `mixes_numerals`.
pub const MIXED_NUMERALS: f32 = 5.0;

/// Whether a candidate writes its numbers both ways: in digits, and in
/// kanji right before 年, 月 or 日 (新暦八月1日, 一日の…10時間). A runner-up
/// writes them in digits throughout as the text does (8月1日, 1日), and the
/// one with both is scored down (`MIXED_NUMERALS`): on AJIMEE, whose
/// readings spell numbers in kana, Bletila-v0.1-small's top 1 with the
/// lexicon 76.0 -> 77.5 together with `drop_broken_numbers` (+3, none lost),
/// with the reranker over the top 20 86.5 -> 87.0; wiki_fresh_v2 unchanged.
pub fn mixes_numerals(candidate: &str) -> bool {
    let chars: Vec<char> = candidate.chars().map(to_half).collect();
    chars.iter().any(|c| c.is_ascii_digit())
        && chars
            .windows(2)
            .any(|w| is_kanji_numeral(w[0]) && matches!(w[1], '年' | '月' | '日'))
}

/// Half-width form of a full-width ASCII char (U+FF01..U+FF5E), else `c`.
pub(crate) fn to_half(c: char) -> char {
    match c {
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFF01 + 0x21).unwrap_or(c),
        _ => c,
    }
}

/// Full-width form of a printable ASCII char, digits and letters included.
fn to_full_ascii(c: char) -> char {
    match c {
        '!'..='~' => char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c),
        _ => c,
    }
}

/// The widths the text before the caret writes, each the last one seen:
/// `true` is full-width.
#[derive(Debug, Default)]
struct WidthHint {
    digit: Option<bool>,
    letter: Option<bool>,
    /// by the symbol's half-width form
    symbols: std::collections::HashMap<char, bool>,
}

impl WidthHint {
    fn of(context: &str) -> Self {
        let mut hint = Self::default();
        for c in context.chars() {
            let h = to_half(c);
            if !('!'..='~').contains(&h) {
                continue;
            }
            let full = h != c;
            if h.is_ascii_digit() {
                hint.digit = Some(full);
            } else if h.is_ascii_alphabetic() {
                hint.letter = Some(full);
            } else {
                hint.symbols.insert(h, full);
            }
        }
        hint
    }

    /// Whether a digit / letter is written full-width; `None` for a symbol.
    fn alnum_full(&self, c: char) -> Option<bool> {
        if c.is_ascii_digit() {
            Some(self.digit.unwrap_or(false))
        } else if c.is_ascii_alphabetic() {
            Some(self.letter.unwrap_or(false))
        } else {
            None
        }
    }
}

const WIDTH_TABLE: &str = include_str!("../data/symbol_width.tsv");

/// Whether Japanese text writes ASCII symbol `c` full-width with `beside`
/// (0, 1, 2) digits or letters right beside it (`data/symbol_width.tsv`).
pub(crate) fn usually_full(c: char, beside: usize) -> bool {
    static TABLE: OnceLock<HashMap<char, [u8; 3]>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        WIDTH_TABLE
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let mut f = l.split('\t');
                let c = f.next()?.chars().next()?;
                let mut pct = [0u8; 3];
                for p in &mut pct {
                    *p = f.next()?.parse().ok()?;
                }
                Some((c, pct))
            })
            .collect()
    });
    table.get(&c).is_some_and(|p| p[beside.min(2)] > 50)
}

/// `symbol_width = "auto"`: every ASCII or full-width ASCII char of `cand`
/// in the width the text before the caret (`context`) uses.
///
/// - A digit / letter: the width of the last digit / letter of the context;
///   none there, half-width.
/// - A symbol between two digits / letters: full-width when Japanese text
///   usually writes it so there (`data/symbol_width.tsv`: （ in 1（2）),
///   else the width of the one before it (10:30, １０：３０, 1,000).
/// - Any other symbol: the width the context last wrote it in. Never written
///   there, full-width when Japanese text usually writes it so (凄い！,
///   （本当）, 50％), else the width of the digit / letter beside it (C#, Ｃ＃)
///   or half-width (" ' [ ] | -).
/// - But a symbol of English, in the width of the letters: one inside a
///   word (https://example.com, e.g), or at a word's edge in a sentence
///   with no Japanese before it (`Thank you!`, `hello, world`; これはGoogle！
///   is Japanese's).
pub fn auto_width(cand: &str, context: &str) -> String {
    let hint = WidthHint::of(context);
    let chars: Vec<char> = cand.chars().collect();
    let half: Vec<char> = chars.iter().map(|&c| to_half(c)).collect();
    let alnum_at = |j: Option<usize>| {
        j.and_then(|j| half.get(j))
            .and_then(|&n| hint.alnum_full(n))
    };
    let ascii = |j: usize| ('!'..='~').contains(&half[j]);
    let english = |i: usize| {
        // the word of ASCII chars around it
        let (mut a, mut b) = (i, i + 1);
        while a > 0 && ascii(a - 1) {
            a -= 1;
        }
        while b < half.len() && ascii(b) {
            b += 1;
        }
        if !half[a..b].iter().any(|c| c.is_ascii_alphabetic()) {
            return false;
        }
        let inside = a < i && i + 1 < b;
        inside || {
            let before: String = context.chars().chain(chars[..i].iter().copied()).collect();
            !crate::mixed::sentence_of(&before).0
        }
    };
    (0..chars.len())
        .map(|i| {
            let h = half[i];
            if !ascii(i) {
                return chars[i];
            }
            let full = hint.alnum_full(h).unwrap_or_else(|| {
                if english(i) {
                    return hint.letter.unwrap_or(false);
                }
                let (left, right) = (alnum_at(i.checked_sub(1)), alnum_at(Some(i + 1)));
                let beside = left.or(right);
                let n = usize::from(left.is_some()) + usize::from(right.is_some());
                let by_text = || usually_full(h, n) || beside.unwrap_or(false);
                if n == 2 {
                    by_text()
                } else {
                    hint.symbols.get(&h).copied().unwrap_or_else(by_text)
                }
            });
            if full {
                to_full_ascii(h)
            } else {
                h
            }
        })
        .collect()
}

/// Apply both rules to every candidate (dropping duplicates they create).
/// `context` is the text before the caret, for `SymbolWidth::Auto`.
pub fn apply(
    reading: &str,
    candidates: Vec<String>,
    width: SymbolWidth,
    context: &str,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    candidates
        .into_iter()
        .map(|c| keep_reading_in_parens(reading, &restore_width(reading, &c, width)))
        .map(|c| match width {
            SymbolWidth::Auto => auto_width(&c, context),
            _ => c,
        })
        .filter(|c| seen.insert(c.clone()))
        .collect()
}

/// Character rules of the `[frontend]` config: touten / kuten style, the
/// middle dot and literal replacements. They run on the final list (after
/// `kana_candidates::arrange`), so the typed-reading entries get them too.
///
/// Only the ideographic / full-width / half-width-katakana forms of the
/// comma and period are rewritten, never ASCII `,` `.`, so numbers such as
/// `6.2` and `2,727` keep theirs.
#[derive(Debug, Clone, Default)]
pub struct CharRules {
    marks: Option<(char, char)>,
    middle_dot: bool,
    replacements: Vec<(String, String)>,
}

impl CharRules {
    pub fn from_frontend(f: &FrontendCfg) -> Self {
        Self {
            marks: f.punctuation.marks(),
            middle_dot: f.normalize_middle_dot,
            replacements: f
                .replacements
                .iter()
                .filter(|r| r.enabled && !r.from.is_empty())
                .map(|r| (r.from.clone(), r.to.clone()))
                .collect(),
        }
    }

    pub fn apply_one(&self, s: &str) -> String {
        let mut out: String = s
            .chars()
            .map(|c| match (c, self.marks) {
                ('、' | '，' | '､', Some((comma, _))) => comma,
                ('。' | '．' | '｡', Some((_, period))) => period,
                ('･' | '·' | '‧' | '⋅' | '∙', _) if self.middle_dot => '・',
                _ => c,
            })
            .collect();
        for (from, to) in &self.replacements {
            if out.contains(from.as_str()) {
                out = out.replace(from.as_str(), to);
            }
        }
        out
    }

    /// Apply to every candidate, dropping empty results and the duplicates
    /// the rules create (first occurrence wins).
    pub fn apply(&self, candidates: Vec<String>) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        candidates
            .into_iter()
            .map(|c| self.apply_one(&c))
            .filter(|c| !c.is_empty() && seen.insert(c.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PunctuationStyle, Replacement};

    #[test]
    fn a_kana_in_the_wrong_script_between_two_drops_the_candidate() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let got = drop_script_flips(v(&[
            "続投が決まっ",
            "続投がキまっ",
            "デるタ",
            "依頼がクるように",
            "ゲームをプレイ",
            "あのテレビ",
            "ピリッとした",
            "カシの木",
            "寺ノ台",
            "コーヒー",
        ]));
        assert_eq!(
            got,
            v(&[
                "続投が決まっ",
                "ゲームをプレイ",
                "あのテレビ",
                "ピリッとした",
                "カシの木",
                "寺ノ台",
                "コーヒー",
            ])
        );
    }

    fn one(reading: &str, cand: &str, w: SymbolWidth) -> String {
        apply(reading, vec![cand.to_string()], w, "").remove(0)
    }

    fn auto(reading: &str, cand: &str, context: &str) -> String {
        apply(reading, vec![cand.to_string()], SymbolWidth::Auto, context).remove(0)
    }

    #[test]
    fn auto_width_without_context_writes_symbols_as_japanese_text_does() {
        assert_eq!(
            auto("すごい！（ほんとう）", "凄い!(本当)", ""),
            "凄い！（本当）"
        );
        assert_eq!(auto("10：30に", "10:30に", ""), "10:30に");
        assert_eq!(auto("50％", "50%", ""), "50％");
        assert_eq!(auto("（2020ねん）", "(2020年)", ""), "（2020年）");
        assert_eq!(auto("1，000えん", "1,000円", ""), "1,000円");
        // symbols Japanese text writes half-width
        assert_eq!(auto("＂ほげ＂と", "\"ほげ\"と", ""), "\"ほげ\"と");
        assert_eq!(auto("［ちゅう］", "[注]", ""), "[注]");
        assert_eq!(auto("ｃ＃で", "C#で", ""), "C#で");
        assert_eq!(auto("ｃ＃で", "C#で", "ＸＹＺ"), "Ｃ＃で");
        assert_eq!(auto("れい：", "例:", ""), "例：");
    }

    #[test]
    fn auto_width_follows_the_context() {
        // full-width digits before: digits and the symbols beside them too
        assert_eq!(
            auto("10：30に", "10:30に", "会議は１２時から"),
            "１０：３０に"
        );
        assert_eq!(auto("50％", "50%", "税率は１０％"), "５０％");
        // letters and digits are judged apart
        assert_eq!(auto("ABCと1", "ABCと1", "ＸＹＺと9"), "ＡＢＣと1");
        // a symbol on its own: as the context last wrote it
        assert_eq!(auto("すごい！", "凄い!", "やった!"), "凄い!");
        assert_eq!(auto("すごい！", "凄い!", "やった!よし！"), "凄い！");
        // the last one written wins
        assert_eq!(auto("1こ", "1個", "１つ、2つ"), "1個");
        assert_eq!(auto("1こ", "1個", "2つ、３つ"), "１個");
    }

    #[test]
    fn auto_width_writes_english_symbols_as_its_letters() {
        assert_eq!(
            auto("https://example.com", "https://example.com", ""),
            "https://example.com"
        );
        assert_eq!(auto("Thank you!", "Thank you!", "やった！"), "Thank you!");
        assert_eq!(auto("hello, world.", "hello, world.", ""), "hello, world.");
        // a URL inside Japanese too
        assert_eq!(
            auto("しょうさいはhttps://a.com", "詳細はhttps://a.com", ""),
            "詳細はhttps://a.com"
        );
        // full-width letters: their symbols too
        assert_eq!(auto("Ｏｋ！", "Ok!", "ＸＹＺ"), "Ｏｋ！");
        // English inside Japanese ends as Japanese does
        assert_eq!(
            auto("これはGoogle！", "これはGoogle!", ""),
            "これはGoogle！"
        );
    }

    #[test]
    fn auto_width_leaves_other_chars() {
        assert_eq!(auto("まる", "○", "ＡＢＣ"), "○");
        assert_eq!(auto("ほんとう、", "本当、", "１２３"), "本当、");
    }

    #[test]
    fn typed_width_comes_back() {
        assert_eq!(
            one("すごい！（ほんとう）", "凄い!(本当)", SymbolWidth::AsTyped),
            "凄い！（本当）"
        );
        assert_eq!(one("すごい！", "凄い!", SymbolWidth::Half), "凄い!");
        assert_eq!(one("すごい!", "凄い!", SymbolWidth::Full), "凄い！");
    }

    #[test]
    fn a_dropped_symbol_does_not_block_the_later_ones() {
        // the model dropped the colon; the ！ after it still comes back
        assert_eq!(
            one("じかん：いまから！", "時間今から!", SymbolWidth::AsTyped),
            "時間今から！"
        );
    }

    #[test]
    fn a_symbol_the_model_added_is_not_taken_for_a_typed_one() {
        // Aster-v1.2-small writes 括) for かっこ: the typed ） is the last one
        assert_eq!(
            one("れい：（かっこ）", "例:(括))", SymbolWidth::AsTyped),
            "例：（括)）"
        );
        // the typed colon sits between ひ and ふ, the model's own one at the end
        assert_eq!(one("ひ：ふ", "比:ふ:", SymbolWidth::AsTyped), "比：ふ:");
    }

    #[test]
    fn candidates_that_move_a_sentence_end_are_dropped() {
        let v = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // Shiran-v2 s7080's runners-up for a reading of two sentences
        let reading = "のてんらんかいにしゅってんした。1900ねん4がつ";
        assert_eq!(
            drop_moved_sentence_ends(
                reading,
                v(&[
                    "の展覧会に出展した。1900年4月",
                    "の展覧会に出展した『1900年4月",
                    "の展覧会に出展した1900年4月",
                    "の展覧会に出展したば1900年4月",
                    "の展覧会に出展した。。1900年4月",
                ])
            ),
            v(&["の展覧会に出展した。1900年4月"])
        );
        // full-width and half-width forms are the same ends; numbers and
        // URLs keep theirs
        assert_eq!(
            drop_moved_sentence_ends(
                "ほんとう？うん！",
                v(&["本当?うん!", "本当？うん", "本当、うん！"])
            ),
            v(&["本当?うん!"])
        );
        assert_eq!(
            drop_moved_sentence_ends("３．５ばい", v(&["3.5倍", "35倍"])),
            v(&["3.5倍"])
        );
        assert_eq!(
            drop_moved_sentence_ends("いいてんき", v(&["いい天気", "いい天気。"])),
            v(&["いい天気"])
        );
    }

    fn digits(reading: &str, cands: &[&str]) -> Vec<String> {
        drop_changed_digits(reading, cands.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn candidates_that_change_a_typed_number_are_dropped() {
        // shion-v4.10 for 19911225 and 11じ22ふん: CTC merges a repeated digit
        assert_eq!(
            digits(
                "19911225",
                &["199125", "1991125", "19911225", "１９９１１２２５"]
            ),
            ["19911225", "１９９１１２２５"]
        );
        assert_eq!(
            digits(
                "11じ22ふん",
                &[
                    "11時22分",
                    "11時に2分",
                    "11時2二分",
                    "1一時22分",
                    "11時229分"
                ]
            ),
            ["11時22分"]
        );
        assert_eq!(
            digits(
                "2025・11・22",
                &["2025/11/22", "2025・11・2二", "2025011・22"]
            ),
            ["2025/11/22"]
        );
        assert_eq!(digits("100まん", &["100万", "1000万", "10〇万"]), ["100万"]);
        assert_eq!(digits("3.14", &["3.14", "314"]), ["3.14"]);
        // a number from kana may come beside the typed ones
        assert_eq!(digits("2がつついたち", &["2月1日"]), ["2月1日"]);
        // no digit typed: nothing to keep
        assert_eq!(digits("いちばん", &["1番", "一番"]), ["1番", "一番"]);
    }

    fn numbers(reading: &str, cands: &[&str]) -> Vec<String> {
        drop_broken_numbers(reading, cands.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn numbers_half_in_digits_half_in_kanji_are_dropped() {
        // Bletila-v0.1-small's lists on AJIMEE
        assert_eq!(
            numbers(
                "じゅうろくばんがい",
                &["16番街", "1六番街", "十六番街", "十6番街"]
            ),
            ["16番街", "十六番街"]
        );
        assert_eq!(
            numbers(
                "ごじゅっぽひゃっぽ",
                &["50歩100歩", "50歩1百歩", "5十歩100歩", "五十歩百歩"]
            ),
            ["50歩100歩", "五十歩百歩"]
        );
        // digits before 千万億兆 are how numbers are written
        assert_eq!(
            numbers(
                "さんぜんにん、いちまんごせん",
                &["3千人、1万5000", "3千人、1万5〇〇〇"]
            ),
            ["3千人、1万5000"]
        );
        // full-width digits too
        assert_eq!(numbers("じゅうさんさい", &["１３歳", "十３歳"]), ["１３歳"]);
    }

    #[test]
    fn a_zero_the_reading_does_not_have_is_dropped() {
        assert_eq!(
            numbers(
                "さんねんくがつじゅうごにち",
                &["03年9月15日", "3年9月15日", "3年09月15日"]
            ),
            ["3年9月15日"]
        );
        assert_eq!(
            numbers("せんごじゅうねん", &["戦後0年", "戦後10年"]),
            ["戦後10年"]
        );
        // read or typed, and a decimal's digits, stay
        assert_eq!(numbers("れいじ", &["0時", "零時"]), ["0時", "零時"]);
        assert_eq!(numbers("ぜろてんご", &["0.5"]), ["0.5"]);
        assert_eq!(numbers("0ばん", &["0番"]), ["0番"]);
        assert_eq!(numbers("いってんまるご", &["1.05"]), ["1.05"]);
        assert_eq!(numbers("にせんねん", &["2000年"]), ["2000年"]);
    }

    #[test]
    fn numbers_in_digits_and_kanji_before_a_date_unit_mix() {
        assert!(mixes_numerals("は新暦八月1日もしくは月遅れの九月1日"));
        assert!(mixes_numerals("、一日の将棋の勉強時間が10時間を超える"));
        assert!(mixes_numerals("一日の勉強時間が１０時間"));
        assert!(!mixes_numerals("は新暦8月1日もしくは月遅れの9月1日"));
        assert!(!mixes_numerals("九月一日"));
        // kanji numerals elsewhere are words
        assert!(!mixes_numerals("一人前になり、30も半ばにして"));
        assert!(!mixes_numerals("一時的に10人"));
    }

    fn drop(reading: &str, cands: &[&str]) -> Vec<String> {
        drop_untyped_symbols(reading, cands.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn candidates_with_untyped_symbols_are_dropped() {
        // Aster-v1.2-small for すうじときごう
        assert_eq!(
            drop(
                "すうじときごう",
                &[
                    "数字と:",
                    "数字と)",
                    "数字と号",
                    "数字と記:",
                    "数字と  ",
                    "数字と記号"
                ]
            ),
            ["数字と号", "数字と記号"]
        );
        assert_eq!(
            drop(
                "めーるあどれす",
                &[
                    "メールアドレス",
                    "メール アドレス",
                    "メール・アドレス",
                    "メールアドレ〓"
                ]
            ),
            ["メールアドレス"]
        );
        assert_eq!(
            drop("そうですか", &["そうですか", "そうですか？"]),
            ["そうですか"]
        );
    }

    #[test]
    fn typed_symbols_are_allowed_as_often_as_typed() {
        // either width / style of a typed symbol
        assert_eq!(drop("はい、そう。", &["はい，そう．"]), ["はい，そう．"]);
        // the model's extra ) for かっこ is one more than typed
        assert_eq!(
            drop("れい：（かっこ）", &["例：（括)）", "例：（括弧）"]),
            ["例：（括弧）"]
        );
        // a typed ー allows a hyphen
        assert_eq!(drop("3ー4", &["3-4"]), ["3-4"]);
    }

    #[test]
    fn a_separator_between_digits_is_part_of_the_number() {
        assert_eq!(
            drop("ろくてんにぱーせんと", &["6。2パーセント", "6。2パーセン%"]),
            ["6.2パーセント"]
        );
        assert_eq!(drop("にせんななひゃくにじゅうなな", &["2、727"]), ["2,727"]);
        // a typed full-width comma put back by restore_width stays
        assert_eq!(drop("1，000えん", &["1，000円"]), ["1，000円"]);
    }

    #[test]
    fn a_typed_comma_the_engine_wrote_as_touten_comes_back() {
        // Aster-v1.2-small via the engine's normalization: `,` -> `、`
        assert_eq!(
            one(
                "ねだん：￥1，000（ぜいこみ）",
                "値段:¥1、000(税込)",
                SymbolWidth::AsTyped
            ),
            "値段：￥1，000（税込）"
        );
        // a touten the model wrote on its own stays
        assert_eq!(
            one("1，000えん、やすい", "1、000円、安い", SymbolWidth::AsTyped),
            "1，000円、安い"
        );
    }

    #[test]
    fn a_full_width_form_from_the_engine_is_matched_too() {
        assert_eq!(one("すごい！", "凄い！", SymbolWidth::Half), "凄い!");
    }

    #[test]
    fn model_symbols_are_left_alone() {
        // the reading has no full-width symbol: 6.2 and 2,727 stay as the model wrote them
        assert_eq!(one("ろくてんに", "6.2", SymbolWidth::AsTyped), "6.2");
        assert_eq!(
            one("にせんななひゃく", "2,727円", SymbolWidth::Full),
            "2,727円"
        );
    }

    #[test]
    fn reading_annotation_stays_kana() {
        let r = "しほろちょう（しほろちょう）は";
        assert_eq!(
            one(r, "士幌町(士幌町)は", SymbolWidth::AsTyped),
            "士幌町（しほろちょう）は"
        );
    }

    #[test]
    fn other_parentheses_are_converted() {
        let r = "けっか（ちゅうりゃく）";
        assert_eq!(one(r, "結果(中略)", SymbolWidth::AsTyped), "結果（中略）");
    }

    #[test]
    fn group_count_mismatch_is_left_alone() {
        let r = "しほろちょう（しほろちょう）";
        assert_eq!(one(r, "士幌町", SymbolWidth::AsTyped), "士幌町");
    }

    #[test]
    fn duplicates_are_dropped() {
        let r = "しほろちょう（しほろちょう）";
        let got = apply(
            r,
            vec!["士幌町(士幌町)".into(), "士幌町(志幌町)".into()],
            SymbolWidth::AsTyped,
            "",
        );
        assert_eq!(got, vec!["士幌町（しほろちょう）".to_string()]);
    }

    fn rules(punctuation: PunctuationStyle, middle_dot: bool, repl: &[(&str, &str)]) -> CharRules {
        CharRules::from_frontend(&FrontendCfg {
            punctuation,
            normalize_middle_dot: middle_dot,
            replacements: repl
                .iter()
                .map(|(f, t)| Replacement {
                    from: f.to_string(),
                    to: t.to_string(),
                    enabled: true,
                })
                .collect(),
            ..FrontendCfg::default()
        })
    }

    #[test]
    fn punctuation_styles_leave_ascii_numbers_alone() {
        let s = "はい、そう。6.2と2,727，よ．";
        assert_eq!(rules(PunctuationStyle::AsIs, false, &[]).apply_one(s), s);
        assert_eq!(
            rules(PunctuationStyle::CommaPeriod, false, &[]).apply_one(s),
            "はい，そう．6.2と2,727，よ．"
        );
        assert_eq!(
            rules(PunctuationStyle::CommaKuten, false, &[]).apply_one(s),
            "はい，そう。6.2と2,727，よ。"
        );
        assert_eq!(
            rules(PunctuationStyle::TotenPeriod, false, &[]).apply_one(s),
            "はい、そう．6.2と2,727、よ．"
        );
        assert_eq!(
            rules(PunctuationStyle::TotenKuten, false, &[]).apply_one(s),
            "はい、そう。6.2と2,727、よ。"
        );
    }

    #[test]
    fn middle_dot_variants_and_replacements() {
        let r = rules(PunctuationStyle::AsIs, true, &[("～", "〜"), ("", "x")]);
        assert_eq!(r.apply_one("A･B·C‧D⋅E∙F"), "A・B・C・D・E・F");
        assert_eq!(r.apply_one("1～2"), "1〜2");
        let off = rules(PunctuationStyle::AsIs, false, &[]);
        assert_eq!(off.apply_one("A･B"), "A･B");
    }

    #[test]
    fn disabled_replacements_are_skipped_and_duplicates_dropped() {
        let mut f = FrontendCfg {
            punctuation: PunctuationStyle::CommaPeriod,
            ..FrontendCfg::default()
        };
        f.replacements.push(Replacement {
            from: "a".into(),
            to: "b".into(),
            enabled: false,
        });
        let r = CharRules::from_frontend(&f);
        assert_eq!(
            r.apply(vec!["a、b".into(), "a，b".into(), "c".into()]),
            vec!["a，b".to_string(), "c".to_string()]
        );
    }
}
