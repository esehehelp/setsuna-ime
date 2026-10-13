//! Segmentation of a whole-sentence conversion into bunsetsu-like segments,
//! for converting segment by segment.
//!
//! The engine returns, per candidate char, the reading char its token was
//! emitted at (`candidate_reading_pos`). That splits even kanji runs by
//! reading (東[とう]京[きょう]都[と]庁[ちょう]), but a boundary can come one
//! char early: Aster-v1.2-small emits は of 今日は at the う of きょう. So
//! the positions are made monotone and every kana is snapped to the matching
//! reading kana within a char or two. Without engine positions the
//! dictionary-free kana-anchor alignment (`reading_align`) is used.
//!
//! Chars sharing a start form a unit (今日 -> きょう); units are grouped into
//! segments with a boundary before a unit that starts a word (kanji,
//! katakana, digit, latin, opening bracket) right after hiragana or closing
//! punctuation: 記者の|記者が|汽車で|帰社した, 凄い！|（本当）.

use crate::reading_align::{align, is_kana, to_hira};

/// One segment: its reading chars `[reading.0, reading.1)` and the chars
/// `[text.0, text.1)` of the candidate it was cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub reading: (usize, usize),
    pub text: (usize, usize),
}

/// For each char of `text`, the reading char it starts at: non-decreasing
/// and 0 for the first char. `None` when the text cannot be aligned.
pub fn char_starts(reading: &str, text: &str, engine_pos: Option<&[u32]>) -> Option<Vec<usize>> {
    let r: Vec<char> = reading.chars().collect();
    let t: Vec<char> = text.chars().collect();
    if r.is_empty() || t.is_empty() {
        return None;
    }
    let usable =
        engine_pos.filter(|p| p.len() == t.len() && p.iter().all(|&x| (x as usize) < r.len()));
    let Some(pos) = usable else {
        return align(&t, &r).map(|ranges| ranges.iter().map(|&(a, _)| a).collect());
    };
    let mut starts = Vec::with_capacity(t.len());
    let mut lo = 0usize;
    for (i, &c) in t.iter().enumerate() {
        let mut q = (pos[i] as usize).max(lo);
        if is_kana(c) {
            // the emitting frame can be a char or two off the kana it spells
            let h = to_hira(c);
            let near = [Some(q), Some(q + 1), q.checked_sub(1), Some(q + 2)];
            if let Some(j) = near
                .into_iter()
                .flatten()
                .find(|&j| j >= lo && j < r.len() && to_hira(r[j]) == h)
            {
                q = j;
            }
        }
        starts.push(q);
        lo = q;
    }
    starts[0] = 0;
    // A kana spells exactly its own reading char, so what follows it starts
    // right after. The engine can emit the next kanji a char late (使 of
    // 方式を使う at the か of をつかう, which would leave つ with を) or
    // early (merging it into the kana's unit).
    for i in 1..t.len() {
        if is_kana(t[i - 1]) && !is_kana(t[i]) {
            starts[i] = (starts[i - 1] + 1).min(r.len() - 1);
        }
        starts[i] = starts[i].max(starts[i - 1]);
    }
    Some(starts)
}

fn is_hiragana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}')
}

fn is_open_bracket(c: char) -> bool {
    "（「『【［｛〈《〔([{\"＂".contains(c)
}

fn starts_word(c: char) -> bool {
    (c.is_alphanumeric() && !is_hiragana(c) && c != 'ー') || is_open_bracket(c)
}

fn ends_word(c: char) -> bool {
    is_hiragana(c) || "、。，．！？!?）」』】］｝〉》〕)]}".contains(c)
}

/// A hiragana word right after は / が / を that follows a content word:
/// 彼女は|そういった, 今日は|いい. The particle must follow a kanji,
/// katakana, digit or latin char, so ありがとう (が after り) and には stay.
fn after_case_particle(t: &[char], a: usize) -> bool {
    a >= 2
        && is_hiragana(t[a])
        && matches!(t[a - 1], 'は' | 'が' | 'を')
        && starts_word(t[a - 2])
        && !is_open_bracket(t[a - 2])
}

/// Segments of `text` over a reading of `reading_len` chars, given the
/// per-char reading `starts` (`char_starts`). They cover the reading and
/// the text in order, with no gap.
pub fn segment(text: &str, starts: &[usize], reading_len: usize) -> Vec<Segment> {
    let t: Vec<char> = text.chars().collect();
    if t.is_empty() || starts.len() != t.len() || reading_len == 0 {
        return Vec::new();
    }
    // units: runs of chars sharing a start
    let mut units: Vec<(usize, usize)> = Vec::new(); // text ranges
    for i in 0..t.len() {
        match units.last_mut() {
            Some(u) if starts[u.0] == starts[i] => u.1 = i + 1,
            _ => units.push((i, i + 1)),
        }
    }
    let unit_reading = |k: usize| -> (usize, usize) {
        let s = starts[units[k].0];
        let e = units.get(k + 1).map_or(reading_len, |u| starts[u.0]);
        (s, e.max(s))
    };
    let mut out: Vec<Segment> = Vec::new();
    for k in 0..units.len() {
        let (a, b) = units[k];
        let (rs, re) = unit_reading(k);
        let boundary = k > 0
            && ((starts_word(t[a]) && ends_word(t[units[k - 1].1 - 1]))
                || after_case_particle(&t, a));
        match out.last_mut() {
            Some(seg) if !boundary => {
                seg.reading.1 = re;
                seg.text.1 = b;
            }
            _ => out.push(Segment {
                reading: (rs, re),
                text: (a, b),
            }),
        }
    }
    if let Some(last) = out.last_mut() {
        last.reading.1 = reading_len;
    }
    out
}

/// Whether `text` can stand for a segment read `reading` whose chosen text
/// is `chosen`: it aligns to the reading (its kana match), and when the
/// chosen text ends in the reading's last kana (a particle, okurigana), so
/// does it. Slices cut at boundaries another candidate does not share
/// (雨降 for あめが next to 雨が) and span conversions that spell the
/// reading wrong (る降るらしい for ふるらしい) fail.
pub fn fits(reading: &str, chosen: &str, text: &str) -> bool {
    let r: Vec<char> = reading.chars().collect();
    let t: Vec<char> = text.chars().collect();
    // a symbol-dictionary entry (かっこ -> （）) spells no reading
    if !t.is_empty() && t.iter().all(|&c| !c.is_alphanumeric() && !is_kana(c)) {
        return true;
    }
    if r.is_empty() || t.is_empty() || align(&t, &r).is_none() {
        return false;
    }
    let last = *r.last().unwrap();
    let ends_in = |s: &str| {
        s.chars()
            .last()
            .is_some_and(|c| is_kana(c) && to_hira(c) == to_hira(last))
    };
    !ends_in(chosen) || ends_in(text)
}

/// The slice of `text` for each of the reading `ranges`, by the chars'
/// reading `starts`. A char belongs to the range its start falls in.
pub fn slice(text: &str, starts: &[usize], ranges: &[(usize, usize)]) -> Vec<String> {
    let mut out = vec![String::new(); ranges.len()];
    for (c, &s) in text.chars().zip(starts) {
        if let Some(k) = ranges.iter().position(|&(a, b)| a <= s && s < b) {
            out[k].push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(text: &str, reading: &str, segs: &[Segment]) -> Vec<(String, String)> {
        let t: Vec<char> = text.chars().collect();
        let r: Vec<char> = reading.chars().collect();
        segs.iter()
            .map(|s| {
                (
                    t[s.text.0..s.text.1].iter().collect(),
                    r[s.reading.0..s.reading.1].iter().collect(),
                )
            })
            .collect()
    }

    fn pairs(xs: &[(&str, &str)]) -> Vec<(String, String)> {
        xs.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn engine_positions_are_snapped_to_the_kana() {
        // Aster-v1.2-small: 今日[きょ] は[う] い[はい] い[い] 天[て] 気[んき] で す ね
        let (r, t) = ("きょうはいいてんきですね", "今日はいい天気ですね");
        let pos = [0, 0, 2, 3, 5, 6, 7, 9, 10, 11];
        let starts = char_starts(r, t, Some(&pos)).unwrap();
        assert_eq!(starts, [0, 0, 3, 4, 5, 6, 7, 9, 10, 11]);
        let segs = segment(t, &starts, 12);
        assert_eq!(
            texts(t, r, &segs),
            pairs(&[
                ("今日は", "きょうは"),
                ("いい", "いい"),
                ("天気ですね", "てんきですね")
            ])
        );
    }

    #[test]
    fn segments_break_before_a_word_after_hiragana() {
        // Aster-v1.2-small's positions for 記者の記者が汽車で帰社した
        let r = "きしゃのきしゃがきしゃできしゃした";
        let t = "記者の記者が汽車で帰社した";
        let pos = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16];
        let starts = char_starts(r, t, Some(&pos)).unwrap();
        let segs = segment(t, &starts, r.chars().count());
        assert_eq!(
            texts(t, r, &segs),
            pairs(&[
                ("記者の", "きしゃの"),
                ("記者が", "きしゃが"),
                ("汽車で", "きしゃで"),
                ("帰社した", "きしゃした"),
            ])
        );
    }

    #[test]
    fn a_hiragana_word_after_a_case_particle_is_its_own_segment() {
        let (r, t) = ("かのじょはそういった", "彼女はそういった");
        let starts = char_starts(r, t, None).unwrap();
        assert_eq!(
            texts(t, r, &segment(t, &starts, 10)),
            pairs(&[("彼女は", "かのじょは"), ("そういった", "そういった")])
        );
        // が after hiragana, and には, do not split
        for (r, t) in [
            ("ありがとう", "有りがとう"),
            ("にほんにはいる", "日本にはいる"),
        ] {
            let starts = char_starts(r, t, None).unwrap();
            let n = r.chars().count();
            assert_eq!(segment(t, &starts, n).len(), 1, "{t}");
        }
    }

    #[test]
    fn a_kanji_after_kana_starts_right_after_it() {
        // Aster-v1.2-small emitted 使 at the か of つかう
        let (r, t) = ("ほうしきをつかう", "方式を使う");
        let starts = char_starts(r, t, Some(&[0, 2, 4, 6, 7])).unwrap();
        assert_eq!(starts, [0, 2, 4, 5, 7]);
        // emitted early, at を: still right after it
        assert_eq!(
            char_starts(r, t, Some(&[0, 2, 4, 4, 7])).unwrap(),
            [0, 2, 4, 5, 7]
        );
        assert_eq!(
            texts(t, r, &segment(t, &starts, 8)),
            pairs(&[("方式を", "ほうしきを"), ("使う", "つかう")])
        );
    }

    #[test]
    fn a_kanji_run_stays_one_segment_and_brackets_start_one() {
        let (r, t) = ("とうきょうとちょうにいく", "東京都庁に行く");
        let starts = char_starts(r, t, Some(&[0, 2, 5, 6, 9, 10, 11])).unwrap();
        assert_eq!(
            texts(t, r, &segment(t, &starts, 12)),
            pairs(&[("東京都庁に", "とうきょうとちょうに"), ("行く", "いく")])
        );
        let (r, t) = ("すごい！（ほんとう）", "凄い！（本当）");
        // the engine put 凄 at ご; the first char always starts at 0
        let starts = char_starts(r, t, Some(&[1, 2, 3, 4, 5, 7, 9])).unwrap();
        assert_eq!(
            texts(t, r, &segment(t, &starts, 10)),
            pairs(&[("凄い！", "すごい！"), ("（本当）", "（ほんとう）")])
        );
    }

    #[test]
    fn without_engine_positions_the_kana_anchors_are_used() {
        let (r, t) = ("かみをきる", "紙を切る");
        let starts = char_starts(r, t, None).unwrap();
        assert_eq!(starts, [0, 2, 3, 4]);
        assert_eq!(
            texts(t, r, &segment(t, &starts, 5)),
            pairs(&[("紙を", "かみを"), ("切る", "きる")])
        );
        // unusable positions (wrong length) fall back too
        assert_eq!(char_starts(r, t, Some(&[0])).unwrap(), starts);
        assert!(char_starts("あした", "今日は", None).is_none());
    }

    #[test]
    fn candidates_that_do_not_fit_the_segment_are_told() {
        // Aster-v1.2-small's slices and span conversions
        assert!(fits("あめが", "雨が", "雨が"));
        assert!(!fits("あめが", "雨が", "雨降"));
        assert!(!fits("ふるらしい", "降るらしい", "る降るらしい"));
        assert!(fits("ふるらしい", "降るらしい", "ふるらしい"));
        // no particle at the end: no ending required
        assert!(fits("かいぎ", "会議", "会義"));
        assert!(fits("かっこ", "括弧", "（）"));
    }

    #[test]
    fn other_candidates_are_sliced_by_the_same_ranges() {
        let r = "きしゃのきしゃ";
        let ranges = [(0, 4), (4, 7)];
        let starts = char_starts(r, "汽車の記者", Some(&[0, 2, 3, 4, 6])).unwrap();
        assert_eq!(slice("汽車の記者", &starts, &ranges), ["汽車の", "記者"]);
    }
}
