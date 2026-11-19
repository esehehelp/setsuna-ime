//! A reading corrector: a CTC model over the same vocabulary that reads the reading as typed so far (with its
//! context, as a conversion does) and writes the reading it should have been, kana to kana, so that a slip of the
//! keys is caught while the reading is typed. Its greedy output is the corrected reading; how likely the model finds
//! it and the reading as typed (`ctc_score`, both over the same frames) says how sure it is.
//!
//! The session (`EngineSession`, a research knob until it is measured) converts the corrected reading first when
//! the corrector changed the reading, the reading as typed can be an output of it on its frames (a reading with
//! more doubled neighbours than spare frames cannot, so the model could not have left it alone) and log P(corrected)
//! - log P(as typed) is above the threshold; else the reading as typed. Either way the other reading's best
//! candidate comes second, so both stay on offer ([`merge`]), and the corrected reading's candidates have their
//! reading positions on the reading as typed ([`to_typed_positions`]), the one the frontend aligns them to.

use std::collections::HashSet;

use crate::candidate::{Candidate, CandidateSource};

/// What the corrector made of one reading.
#[derive(Debug, Clone, PartialEq)]
pub struct Correction {
    /// The corrected reading (the reading as typed when the corrector gave nothing or did not run).
    pub reading: String,
    /// The corrected reading differs from the one typed.
    pub changed: bool,
    /// The corrected reading was converted first (see the module).
    pub applied: bool,
    /// log P(corrected) and log P(as typed) under the corrector less one constant (the same for both: the frames'
    /// log-sum-exps are left out, `proposal::greedy_scores`), so only their difference is a log-probability ratio;
    /// -inf for a sequence no alignment to the frames fits, NaN when the corrector did not run (a reading too long
    /// for one pass, or with a token its logits leave out).
    pub score_fix: f32,
    pub score_typed: f32,
}

impl Correction {
    /// The reading left as typed.
    pub fn unchanged(reading: &str) -> Self {
        Self {
            reading: reading.to_string(),
            changed: false,
            applied: false,
            score_fix: f32::NAN,
            score_typed: f32::NAN,
        }
    }

    /// The corrector's output for `typed` judged against `threshold`.
    pub fn judged(
        typed: &str,
        corrected: String,
        score_fix: f32,
        score_typed: f32,
        threshold: f32,
    ) -> Self {
        let changed = !corrected.is_empty() && corrected != typed;
        let reading = if changed {
            corrected
        } else {
            typed.to_string()
        };
        let applied =
            changed && Self::margin_of(score_fix, score_typed).is_some_and(|m| m > threshold);
        Self {
            reading,
            changed,
            applied,
            score_fix,
            score_typed,
        }
    }

    /// log P(corrected) - log P(as typed), when both are possible outputs.
    pub fn margin(&self) -> Option<f32> {
        Self::margin_of(self.score_fix, self.score_typed)
    }

    fn margin_of(score_fix: f32, score_typed: f32) -> Option<f32> {
        (score_fix.is_finite() && score_typed.is_finite()).then(|| score_fix - score_typed)
    }
}

/// For each char of `corrected`, the char of `typed` it stands for, by an alignment of the fewest edits: a char kept
/// or substituted, its counterpart; one inserted, the typed char after it (the last when none follows); a typed char
/// deleted stands for nothing. Non-decreasing and each below the typed reading's length; empty for an empty typed
/// reading.
pub fn typed_positions(typed: &str, corrected: &str) -> Vec<u32> {
    let t: Vec<char> = typed.chars().collect();
    let c: Vec<char> = corrected.chars().collect();
    let (n, m) = (t.len(), c.len());
    if n == 0 {
        return Vec::new();
    }
    // d[i * w + j]: the edits from t[..i] to c[..j]
    let w = m + 1;
    let mut d = vec![0u32; (n + 1) * w];
    for i in 0..=n {
        d[i * w] = i as u32;
    }
    for j in 0..=m {
        d[j] = j as u32;
    }
    let diag =
        |d: &[u32], i: usize, j: usize| d[(i - 1) * w + j - 1] + u32::from(t[i - 1] != c[j - 1]);
    for i in 1..=n {
        for j in 1..=m {
            d[i * w + j] = diag(&d, i, j)
                .min(d[(i - 1) * w + j] + 1)
                .min(d[i * w + j - 1] + 1);
        }
    }
    let mut map = vec![0u32; m];
    let (mut i, mut j) = (n, m);
    while j > 0 {
        if i > 0 && d[i * w + j] == diag(&d, i, j) {
            map[j - 1] = (i - 1) as u32;
            i -= 1;
            j -= 1;
        } else if d[i * w + j] == d[i * w + j - 1] + 1 {
            map[j - 1] = i.min(n - 1) as u32;
            j -= 1;
        } else {
            i -= 1;
        }
    }
    map
}

/// The corrected reading's `candidates` with their reading positions on the typed reading instead, as a conversion
/// gives every candidate's: `map` is `typed_positions(typed, corrected)`. A position it cannot place leaves the
/// candidate's positions unknown (empty).
pub fn to_typed_positions(candidates: &mut [Candidate], map: &[u32]) {
    for c in candidates {
        let mapped: Option<Vec<u32>> = c
            .reading_pos
            .iter()
            .map(|&p| map.get(p as usize).copied())
            .collect();
        c.reading_pos = mapped.unwrap_or_default();
    }
}

/// The candidates of the reading converted first, with the other reading's best second and its others after the
/// first's (`CandidateSource::OtherReading`), duplicates dropped, cut to `limit`.
pub fn merge(first: Vec<Candidate>, other: Vec<Candidate>, limit: usize) -> Vec<Candidate> {
    let mut out = Vec::with_capacity(limit.min(first.len() + other.len()));
    let mut seen = HashSet::new();
    let mut first = first.into_iter();
    let mut other = other.into_iter().map(|mut c| {
        c.source = CandidateSource::OtherReading;
        c
    });
    let order = first
        .next()
        .into_iter()
        .chain(other.next())
        .chain(first)
        .chain(other);
    for c in order {
        if out.len() >= limit {
            break;
        }
        if seen.insert(c.text.clone()) {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(text: &str) -> Candidate {
        Candidate {
            text: text.into(),
            score: 0.0,
            source: CandidateSource::Beam,
            reading_pos: Vec::new(),
            parts: None,
        }
    }

    fn texts(c: &[Candidate]) -> Vec<&str> {
        c.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn the_other_readings_best_comes_second() {
        let merged = merge(
            vec![cand("今日は"), cand("京は"), cand("経は")],
            vec![cand("今日わ"), cand("京は"), cand("教派")],
            10,
        );
        assert_eq!(texts(&merged), ["今日は", "今日わ", "京は", "経は", "教派"]);
        assert_eq!(merged[1].source, CandidateSource::OtherReading);
        assert_eq!(merged[2].source, CandidateSource::Beam);
        assert_eq!(merged[4].source, CandidateSource::OtherReading);
        assert_eq!(
            texts(&merge(vec![cand("a"), cand("b")], vec![cand("c")], 2)),
            ["a", "c"]
        );
        assert_eq!(
            texts(&merge(Vec::new(), vec![cand("c"), cand("d")], 5)),
            ["c", "d"]
        );
    }

    #[test]
    fn corrected_positions_land_on_the_typed_reading() {
        // inserted: the typed char after it
        assert_eq!(typed_positions("かみをる", "かみをきる"), [0, 1, 2, 3, 3]);
        assert_eq!(typed_positions("ありがと", "ありがとう"), [0, 1, 2, 3, 3]);
        // substituted: its counterpart
        assert_eq!(typed_positions("かに", "かみ"), [0, 1]);
        // deleted: nothing stands for it
        assert_eq!(typed_positions("ととも", "とも"), [1, 2]);
        assert_eq!(typed_positions("きょううは", "きょうは"), [0, 1, 3, 4]);
        assert!(typed_positions("", "か").is_empty());
        let mut cands = vec![
            Candidate {
                reading_pos: vec![0, 1, 2, 3, 4],
                ..cand("可美を切る")
            },
            Candidate {
                reading_pos: vec![0, 9],
                ..cand("壊れた")
            },
            cand("位置なし"),
        ];
        to_typed_positions(&mut cands, &typed_positions("かみをる", "かみをきる"));
        assert_eq!(cands[0].reading_pos, [0, 1, 2, 3, 3]);
        assert!(cands[1].reading_pos.is_empty() && cands[2].reading_pos.is_empty());
    }

    #[test]
    fn applied_only_above_the_threshold_and_when_both_are_possible() {
        let c = Correction::judged("ととも", "とも".into(), -0.1, -9.0, 7.0);
        assert!(c.changed && c.applied && c.reading == "とも");
        assert!(!Correction::judged("ととも", "とも".into(), -0.1, -5.0, 7.0).applied);
        // the reading as typed cannot be an output on its frames: never applied
        let c = Correction::judged("ととも", "とも".into(), -0.1, f32::NEG_INFINITY, 7.0);
        assert!(c.changed && !c.applied && c.margin().is_none());
        // unchanged or empty: nothing to apply
        let c = Correction::judged("とも", "とも".into(), -0.1, -0.1, -100.0);
        assert!(!c.changed && !c.applied);
        let c = Correction::judged("とも", String::new(), f32::NEG_INFINITY, -0.1, -100.0);
        assert!(!c.changed && c.reading == "とも");
        // no threshold given: changed, never applied
        assert!(!Correction::judged("ととも", "とも".into(), -0.1, -9.0, f32::INFINITY).applied);
    }
}
