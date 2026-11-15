//! Scores: an output is right when, NFKC-normalized, without whitespace and
//! in lower case, it equals one of the row's accepted forms; its CER is the
//! edit distance to the nearest accepted form over that form's length. The
//! style of the full-width punctuation (，． or 、。, a setting of the input
//! method) does not count.

use std::collections::BTreeMap;

use unicode_normalization::UnicodeNormalization;

use crate::set::Row;

pub fn norm(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '，' => '、',
            '．' => '。',
            _ => c,
        })
        .collect::<String>()
        .nfkc()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Whether `output` is right for `row`, and its CER.
pub fn judge(row: &Row, output: &str) -> (bool, f64) {
    let o: Vec<char> = norm(output).chars().collect();
    let mut best = f64::INFINITY;
    for a in &row.answers {
        let a: Vec<char> = norm(a).chars().collect();
        let d = distance(&o, &a) as f64 / a.len().max(1) as f64;
        best = best.min(d);
    }
    (best == 0.0, best)
}

pub fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let k = ((v.len() - 1) as f64 * p).round() as usize;
    v[k]
}

#[derive(Default)]
pub struct Tally {
    pub n: usize,
    pub right: usize,
    pub cer: f64,
}

/// Per category and over all: rows, right, summed CER; `outputs` by row id.
pub fn tally(rows: &[Row], outputs: &BTreeMap<String, String>) -> BTreeMap<String, Tally> {
    let mut t: BTreeMap<String, Tally> = BTreeMap::new();
    for row in rows {
        let Some(out) = outputs.get(&row.id) else {
            continue;
        };
        let (ok, cer) = judge(row, out);
        for key in [row.category.as_str(), "all"] {
            let e = t.entry(key.to_string()).or_default();
            e.n += 1;
            e.right += usize::from(ok);
            e.cer += cer;
        }
    }
    t
}
