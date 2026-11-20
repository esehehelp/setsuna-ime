//! Word-level choices derived from a sentence-level one.
//!
//! The engine's candidates are whole sentences with no segmentation, so a
//! learned choice (記者に連絡する -> 貴社に連絡する for きしゃにれんらくする)
//! is split here into the words it changed (きしゃ: 記者 -> 貴社) together
//! with up to two reading characters on either side (none / にれ). The word
//! choice then applies wherever that reading recurs next to the same context
//! on one side (きしゃにれんらくします), however different the rest of the
//! reading is. One character is not enough context: with Aster-v1.2-small,
//! かみ: 髪 -> 紙 learned before を turned まいにちかみをあらう into 毎日紙を洗う,
//! and こうえん: 公園 -> 公演 before に turned こうえんにあつまる into 公演.
//!
//! Surfaces are aligned to the reading without a dictionary: a kana (either
//! script) must match its reading char, a symbol matches any reading symbol,
//! and a run of other characters (kanji, digits, latin) takes 1 to 4 reading
//! chars per character. The diff of the two surfaces is widened to whole
//! runs (思 / 志 -> 意思 / 意志), and the reading span of the widened chosen
//! part is the word's reading.

/// One word choice: `chosen` over `rejected` for the reading `span`, seen
/// with the reading chars `left` / `right` next to it (up to `CONTEXT` each,
/// fewer at an end of the reading).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(super) struct Word {
    pub span: String,
    pub chosen: String,
    pub rejected: String,
    pub left: String,
    pub right: String,
}

/// Reading chars of context kept on each side of a word.
const CONTEXT: usize = 2;

use crate::reading_align::{align, is_free};

/// The longest pair of texts diffed (chars of one times chars of the
/// other): longer ones learn no words.
const MAX_DIFF_CELLS: usize = 1 << 22;

/// Differing regions of `a` and `b` (longest common subsequence diff), as
/// (a_start, a_end, b_start, b_end); none for texts too long to diff.
fn hunks(a: &[char], b: &[char]) -> Vec<(usize, usize, usize, usize)> {
    let (n, m) = (a.len(), b.len());
    if (n + 1).saturating_mul(m + 1) > MAX_DIFF_CELLS {
        return Vec::new();
    }
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j, mut start) = (0, 0, (0, 0));
    loop {
        let matched = i < n && j < m && a[i] == b[j] && lcs[i][j] == lcs[i + 1][j + 1] + 1;
        if matched || (i == n && j == m) {
            if (i, j) != start {
                out.push((start.0, i, start.1, j));
            }
            if i == n && j == m {
                return out;
            }
            i += 1;
            j += 1;
            start = (i, j);
        } else if j == m || (i < n && lcs[i + 1][j] >= lcs[i][j + 1]) {
            i += 1;
        } else {
            j += 1;
        }
    }
}

/// The word choices in choosing `chosen` over `rejected` for `reading`.
/// Empty when `chosen` cannot be aligned to the reading.
pub(super) fn derive(reading: &str, chosen: &str, rejected: &str) -> Vec<Word> {
    let r: Vec<char> = reading.chars().collect();
    let a: Vec<char> = chosen.chars().collect();
    let b: Vec<char> = rejected.chars().collect();
    let Some(ranges) = align(&a, &r) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (mut a0, mut a1, mut b0, mut b1) in hunks(&a, &b) {
        if a0 == a1 || b0 == b1 {
            continue;
        }
        // Widen over the common characters to whole free runs.
        while a0 > 0 && b0 > 0 && a[a0 - 1] == b[b0 - 1] && is_free(a[a0 - 1]) && is_free(a[a0]) {
            a0 -= 1;
            b0 -= 1;
        }
        while a1 < a.len() && b1 < b.len() && a[a1] == b[b1] && is_free(a[a1]) && is_free(a[a1 - 1])
        {
            a1 += 1;
            b1 += 1;
        }
        let (s, e) = (ranges[a0].0, ranges[a1 - 1].1);
        if s >= e {
            continue;
        }
        out.push(Word {
            span: r[s..e].iter().collect(),
            chosen: a[a0..a1].iter().collect(),
            rejected: b[b0..b1].iter().collect(),
            left: r[s.saturating_sub(CONTEXT)..s].iter().collect(),
            right: r[e..(e + CONTEXT).min(r.len())].iter().collect(),
        });
    }
    out
}

/// Where `w` applies in `reading`: the starts of its span where it occurs
/// with the same context on at least one side (both sides for a one-char
/// span, which would otherwise match almost anywhere).
pub(super) fn places(w: &Word, reading: &[char]) -> Vec<usize> {
    let span: Vec<char> = w.span.chars().collect();
    let n = span.len();
    if n == 0 || n > reading.len() {
        return Vec::new();
    }
    let left: Vec<char> = w.left.chars().collect();
    let right: Vec<char> = w.right.chars().collect();
    (0..=reading.len() - n)
        .filter(|&p| {
            if reading[p..p + n] != span[..] {
                return false;
            }
            let left = !left.is_empty() && reading[..p].ends_with(&left);
            let right = !right.is_empty() && reading[p + n..].starts_with(&right);
            (n >= 2 && (left || right)) || (left && right)
        })
        .collect()
}

/// What `surface`, aligned to the reading as `ranges` (`align`), writes
/// for the reading chars `p..e`; `None` when a char of it reads across `p`
/// or `e`: the reading there is no word of that surface (今 of 今では
/// reads across まで of いまでは).
fn form_at(surface: &[char], ranges: &[(usize, usize)], p: usize, e: usize) -> Option<String> {
    let mut out = String::new();
    let (mut lo, mut hi) = (usize::MAX, 0);
    for (&c, &(s, t)) in surface.iter().zip(ranges) {
        if t <= p || s >= e {
            continue;
        }
        if s < p || t > e {
            return None;
        }
        out.push(c);
        lo = lo.min(s);
        hi = hi.max(t);
    }
    (lo == p && hi == e).then_some(out)
}

/// Whether `surface` writes `form` for the reading chars `p..e`: what it
/// writes there (`form_at`, `ranges` its alignment, `None` when it does not
/// fit the reading). Where that is no word of it, a form with a kanji (or
/// another char whose reading is not written out) still counts when the
/// surface holds it, inside a longer run of them (迄 in 今日迄は for まで of
/// きょうまでは); a kana form does not (まで is not written in 今では).
pub(super) fn writes(
    surface: &[char],
    ranges: Option<&[(usize, usize)]>,
    p: usize,
    e: usize,
    form: &str,
) -> bool {
    match ranges.and_then(|r| form_at(surface, r, p, e)) {
        Some(f) => f == form,
        None => form.chars().any(is_free) && surface.iter().collect::<String>().contains(form),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn word(span: &str, chosen: &str, rejected: &str, l: &str, r: &str) -> Word {
        Word {
            span: span.into(),
            chosen: chosen.into(),
            rejected: rejected.into(),
            left: l.into(),
            right: r.into(),
        }
    }

    #[test]
    fn aligns_kanji_runs_between_kana() {
        let got = align(&chars("意志を伝える"), &chars("いしをつたえる")).unwrap();
        assert_eq!(got, [(0, 2), (0, 2), (2, 3), (3, 5), (5, 6), (6, 7)]);
        // katakana matches its hiragana reading
        assert!(align(&chars("メール@アドレス"), &chars("めーる@あどれす")).is_some());
        // a surface that does not fit its reading
        assert!(align(&chars("今日は"), &chars("あした")).is_none());
    }

    #[test]
    fn hunks_are_the_differing_regions() {
        assert_eq!(
            hunks(&chars("貴社に連絡"), &chars("記者に連絡")),
            [(0, 2, 0, 2)]
        );
        assert_eq!(hunks(&chars("abc"), &chars("abc")), []);
        assert_eq!(
            hunks(&chars("aXbY"), &chars("aZbW")),
            [(1, 2, 1, 2), (3, 4, 3, 4)]
        );
        // texts too long to diff: no hunks (a 16-bit count wrapped past
        // 65535 chars in common); 1500 chars diff as any other
        let long = "a".repeat(70_000);
        assert_eq!(
            hunks(&chars(&format!("{long}X")), &chars(&format!("{long}Y"))),
            []
        );
        let (a, b) = ("a".repeat(1500), "a".repeat(1500) + "b");
        assert_eq!(hunks(&chars(&a), &chars(&b)), [(1500, 1500, 1500, 1501)]);
    }

    #[test]
    fn derives_the_changed_words_with_their_context() {
        assert_eq!(
            derive("きしゃにれんらくする", "貴社に連絡する", "記者に連絡する"),
            [word("きしゃ", "貴社", "記者", "", "にれ")]
        );
        // only 志 / 思 differ: widened to the whole run
        assert_eq!(
            derive("いしをつたえる", "意志を伝える", "意思を伝える"),
            [word("いし", "意志", "意思", "", "をつ")]
        );
        assert_eq!(
            derive("こうえんにいく", "公演に行く", "公園に行く"),
            [word("こうえん", "公演", "公園", "", "にい")]
        );
        // two separate changes
        assert_eq!(
            derive("かみをきっておく", "紙を切って置く", "髪を切っておく"),
            [
                word("かみ", "紙", "髪", "", "をき"),
                word("お", "置", "お", "って", "く"),
            ]
        );
        // unalignable chosen surface: nothing
        assert!(derive("かみ", "紙を", "髪").is_empty());
    }

    fn applies(w: &Word, reading: &[char]) -> bool {
        !places(w, reading).is_empty()
    }

    fn writes_at(surface: &str, reading: &str, p: usize, e: usize, form: &str) -> bool {
        let s = chars(surface);
        let r = align(&s, &chars(reading));
        writes(&s, r.as_deref(), p, e, form)
    }

    #[test]
    fn a_form_is_what_the_surface_writes_for_that_reading() {
        // まで of いまでは
        assert!(writes_at("いまでは", "いまでは", 1, 3, "まで"));
        assert!(!writes_at("今では", "いまでは", 1, 3, "まで"));
        assert!(!writes_at("今では", "いまでは", 1, 3, "迄"));
        // まで of きょうまでは
        assert!(writes_at("今日までは", "きょうまでは", 3, 5, "まで"));
        // a kanji form inside a longer run of kanji still counts
        assert!(writes_at("今日迄は", "きょうまでは", 3, 5, "迄"));
        // かみ at the start, not the 紙 at the end
        assert!(writes_at(
            "髪を切ってから紙",
            "かみをきってからかみ",
            0,
            2,
            "髪"
        ));
        assert!(!writes_at(
            "髪を切ってから紙",
            "かみをきってからかみ",
            0,
            2,
            "紙"
        ));
    }

    #[test]
    fn applies_where_the_span_recurs_next_to_the_same_context() {
        let kami = word("かみ", "紙", "髪", "", "をき");
        assert!(applies(&kami, &chars("かみをきっておく")));
        assert!(applies(&kami, &chars("あかいかみをきざむ")));
        assert!(!applies(&kami, &chars("まいにちかみをあらう")));
        assert!(!applies(&kami, &chars("かみがながい")));
        let kisha = word("きしゃ", "貴社", "記者", "", "にれ");
        assert!(applies(&kisha, &chars("きしゃにれんらくします")));
        assert!(!applies(&kisha, &chars("きしゃのたんとうしゃ")));
        // a one-char span needs both sides
        let o = word("お", "置", "お", "って", "く");
        assert!(applies(&o, &chars("もっておく")));
        assert!(!applies(&o, &chars("おくる")));
    }
}
