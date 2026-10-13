//! Digit split: training readings spell numbers in kana (3.5 is read
//! さんてんご) and never hold ASCII digits, so a typed number fed to the model
//! in the reading comes out garbled. The reading is cut into runs of digits
//! and the rest; the digits are spliced back verbatim (`symsplit::splice`)
//! and reach the model only in the context of the segments after them,
//! which training does have (contexts carry digits).

use anyhow::Result;

use crate::candidate::Candidate;
use crate::pipeline::symsplit::{self, Seg, Vary};

/// Whether `chars[i]` is in a run of digits: an ASCII digit, or a `.` / `,`
/// between two of them (3.5, 1,000, 2.5.1).
pub fn in_digit_run(chars: &[char], i: usize) -> bool {
    let c = chars[i];
    c.is_ascii_digit()
        || (matches!(c, '.' | ',')
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i + 1].is_ascii_digit())
}

/// Whether `reading` has a digit to split out.
pub fn has_digit(reading: &str) -> bool {
    reading.chars().any(|c| c.is_ascii_digit())
}

/// `reading` as alternating segments: the runs of digits (`Seg::Sym`, kept
/// verbatim) and the rest (`Seg::Kana`, converted).
pub(crate) fn segments(reading: &str) -> Vec<Seg> {
    let chars: Vec<char> = reading.chars().collect();
    let mut segs: Vec<Seg> = Vec::new();
    let mut buf = String::new();
    let mut cur: Option<bool> = None;
    for i in 0..chars.len() {
        let digit = in_digit_run(&chars, i);
        if let Some(prev) = cur.filter(|&d| d != digit) {
            let t = std::mem::take(&mut buf);
            segs.push(if prev { Seg::Sym(t) } else { Seg::Kana(t) });
        }
        buf.push(chars[i]);
        cur = Some(digit);
    }
    if let Some(d) = cur {
        segs.push(if d { Seg::Sym(buf) } else { Seg::Kana(buf) });
    }
    segs
}

/// Convert a reading with its runs of digits spliced back verbatim: every
/// other segment through `convert_span`, after the context and the output
/// so far; the candidates vary the longest one.
pub fn convert(
    context: &str,
    reading: &str,
    pool_limit: usize,
    convert_span: impl FnMut(&str, &str) -> Result<Vec<Candidate>>,
) -> Result<Vec<Candidate>> {
    symsplit::splice(
        context,
        &segments(reading),
        pool_limit,
        Vary::Longest,
        convert_span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::CandidateSource;

    fn segs(s: &str) -> Vec<(bool, String)> {
        segments(s)
            .into_iter()
            .map(|seg| match seg {
                Seg::Sym(t) => (true, t),
                Seg::Kana(t) => (false, t),
            })
            .collect()
    }

    fn d(s: &str) -> (bool, String) {
        (true, s.to_string())
    }

    fn k(s: &str) -> (bool, String) {
        (false, s.to_string())
    }

    #[test]
    fn numbers_with_their_separators_are_one_run() {
        assert_eq!(
            segs("ぜんたいの3.5ぱーせんと"),
            [k("ぜんたいの"), d("3.5"), k("ぱーせんと")]
        );
        assert_eq!(
            segs("かかく(ぜいこみ)は1,000えんです"),
            [k("かかく(ぜいこみ)は"), d("1,000"), k("えんです")]
        );
        assert_eq!(segs("ばーじょん2.5.1"), [k("ばーじょん"), d("2.5.1")]);
    }

    #[test]
    fn a_separator_not_between_digits_stays_out() {
        assert_eq!(segs("3.。"), [d("3"), k(".。")]);
        assert_eq!(segs("10。"), [d("10"), k("。")]);
        assert_eq!(segs(".5,"), [k("."), d("5"), k(",")]);
        assert_eq!(segs("1、2"), [d("1"), k("、"), d("2")]);
    }

    #[test]
    fn digits_at_the_start_and_the_end() {
        assert_eq!(
            segs("2025ねん10がつ11にち"),
            [d("2025"), k("ねん"), d("10"), k("がつ"), d("11"), k("にち")]
        );
        assert_eq!(segs("だい2"), [k("だい"), d("2")]);
        assert_eq!(segs("42"), [d("42")]);
    }

    #[test]
    fn a_reading_without_digits_is_one_segment() {
        assert_eq!(segs("きょうはいいてんき"), [k("きょうはいいてんき")]);
        assert!(segs("").is_empty());
        assert!(!has_digit("きょう.,"));
        assert!(has_digit("だい2"));
    }

    fn cand(text: &str, score: f32) -> Candidate {
        Candidate {
            text: text.to_string(),
            score,
            source: CandidateSource::Beam,
            reading_pos: (0..text.chars().count() as u32).collect(),
        }
    }

    #[test]
    fn digits_reach_the_model_only_as_context() {
        let mut calls = Vec::new();
        let out = convert(
            "前",
            "ぜんたいの3.5ぱーせんと",
            5,
            |context, reading| {
                calls.push((context.to_string(), reading.to_string()));
                Ok(match reading {
                    "ぜんたいの" => vec![cand("全体の", -0.1), cand("前隊の", -4.0)],
                    _ => vec![cand("パーセント", -0.2), cand("ぱーせんと", -3.0)],
                })
            },
        )
        .unwrap();
        assert_eq!(
            calls,
            [
                ("前".to_string(), "ぜんたいの".to_string()),
                ("前全体の3.5".to_string(), "ぱーせんと".to_string())
            ]
        );
        let texts: Vec<_> = out.iter().map(|c| c.text.as_str()).collect();
        // the longest segment varies (the first of equals: ぜんたいの)
        assert_eq!(texts, ["全体の3.5パーセント", "前隊の3.5パーセント"]);
        // 全 体 の at ぜ ん た, 3 . 5 at their own chars, パーセント at ぱーせんと
        assert_eq!(out[0].reading_pos, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn an_all_digit_reading_comes_back_as_typed() {
        let out = convert("", "2025", 5, |_, _| unreachable!()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "2025");
        assert_eq!(out[0].source, CandidateSource::Fallback);
    }
}
