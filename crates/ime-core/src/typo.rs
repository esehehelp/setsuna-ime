//! Typo correction of romaji, as azooKey's noisy channel (MIT): edits of
//! what was typed cost by kind (a neighbouring key α · distance, an extra
//! key β, two keys swapped γ), and the language model judges the kana
//! they make; tried only where a letter reads as no kana (`kaisga`), as
//! Meltype waits for one, so that what reads is never "corrected". The
//! correction is a second reading converted next to the one typed
//! (`EngineBridge::alt_for_space`), whose top stays first.
//!
//! ず and づ, which read the same, are put as the kana model reads them
//! (`voicing`): つずく is converted as つづく (user 2026-11-10: ずとづの補正).

use crate::composing_text::{romaji_units, UnitKind};
use crate::mixed::MixedModel;

/// How much better (−ln P by the kana model) the reading with ず and づ
/// swapped must read for the reading converted to be it (つずく as つづく,
/// かたずける, みかずき, みづ as みず); the reading as typed is converted
/// too short of `VOICING_SURE`, its candidates after the first (築いたら
/// for きずいたら, made きづいたら). Measured: readings spelled right read
/// best as they are, but for ず/づ both in use (きずく 0.95 築く / 気づく,
/// いなづま 0.73, つまずく 1.51); the misspelt ones 1.9 (よこずな) to 18.
/// Of the readings of wiki_fresh v2 / v3 with ず or づ (384, 377), 3 each
/// are swapped: names (鹿角 かづの, 兎束 うづか, 牛頭 ごず) and a reading
/// starting inside a word (づけた).
pub const VOICING_FIX: f64 = 2.0;
/// Short of `VOICING_FIX`, the reading with them swapped is converted too,
/// its candidates after the first (気づく for きずく, 横綱 for よこずな).
pub const VOICING_OFFER: f64 = 0.5;
/// From this much better, the reading as typed is not converted too: it
/// is no word (つずく 17.8, かたずける 7.3, おこずかい 8.7), its conversions
/// noise (ツズク, 勝たずける); the words spelled so read better by 4.2 at
/// most (築かれた, きずかれた).
pub const VOICING_SURE: f64 = 6.0;

/// Costs of the edits (azooKey's α, β, γ), and of a vowel left out.
const ALPHA: f64 = 2.0;
const BETA: f64 = 3.0;
const GAMMA: f64 = 2.0;
const VOWEL: f64 = 2.5;
/// A correction no better than this (channel + language model, against the
/// best) is not offered.
const MAX_COST: f64 = 6.0;

/// The keys around each letter on a QWERTY keyboard (row, column).
fn position(c: char) -> Option<(f64, f64)> {
    const ROWS: [(&str, f64); 3] = [("qwertyuiop", 0.0), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75)];
    ROWS.iter()
        .enumerate()
        .find_map(|(r, (keys, shift))| keys.find(c).map(|x| (r as f64, x as f64 + shift)))
}

fn distance(a: char, b: char) -> Option<f64> {
    let (pa, pb) = (position(a)?, position(b)?);
    Some(((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt())
}

/// The kana `typed` reads as, when every letter reads.
fn reads(typed: &[char]) -> Option<String> {
    let mut out = String::new();
    for u in romaji_units(typed, true) {
        match u.kind {
            UnitKind::Kana(k) => out.push_str(k),
            UnitKind::Char(c) if c.is_ascii_alphabetic() => return None,
            UnitKind::Char(c) => out.push(c),
            UnitKind::Pending => return None,
        }
    }
    Some(out)
}

/// The reading `typed` was probably meant as, when a letter of it reads as
/// no kana and one edit near it makes all of it read; `lm` judges the
/// kana (without one, the cheapest edit).
pub fn correct(typed: &str, lm: Option<&crate::mixed::MixedModel>) -> Option<String> {
    let chars: Vec<char> = typed.chars().map(|c| c.to_ascii_lowercase()).collect();
    if chars.is_empty() || !chars.iter().all(|c| c.is_ascii_lowercase() || *c == '-') {
        return None;
    }
    // where a letter reads as no kana
    let bad: Vec<usize> = romaji_units(&chars, true)
        .iter()
        .filter(|u| matches!(u.kind, UnitKind::Char(c) if c.is_ascii_alphabetic()))
        .map(|u| u.start)
        .collect();
    let &first = bad.first()?;
    let mut edits: Vec<(Vec<char>, f64)> = Vec::new();
    let around = first.saturating_sub(1)..(first + 2).min(chars.len());
    for p in around.clone() {
        // a neighbouring key
        for c in 'a'..='z' {
            if let Some(d) = distance(chars[p], c).filter(|&d| d > 0.0 && d <= 1.65) {
                let mut v = chars.clone();
                v[p] = c;
                edits.push((v, ALPHA * d));
            }
        }
        // an extra key
        let mut v = chars.clone();
        v.remove(p);
        edits.push((v, BETA));
        // two keys swapped
        if p + 1 < chars.len() {
            let mut v = chars.clone();
            v.swap(p, p + 1);
            edits.push((v, GAMMA));
        }
    }
    // a vowel left out after the letter
    for vowel in ['a', 'i', 'u', 'e', 'o'] {
        let mut v = chars.clone();
        v.insert(first + 1, vowel);
        edits.push((v, VOWEL));
    }
    let original_cost = lm.map(|m| m.kana_cost(&chars.iter().collect::<String>()));
    let mut best: Option<(f64, String)> = None;
    for (v, channel) in edits {
        let Some(kana) = reads(&v) else {
            continue;
        };
        let cost = channel
            + lm.map_or(0.0, |m| {
                m.kana_cost(&kana) - original_cost.unwrap_or(0.0).min(m.kana_cost(&kana))
            });
        if best.as_ref().is_none_or(|(b, _)| cost < *b) {
            best = Some((cost, kana));
        }
    }
    let (cost, kana) = best?;
    // with a language model, a correction must also read well
    let fine = match lm {
        Some(m) => m.kana_cost(&kana) / (kana.chars().count() as f64 + 1.0) < MAX_COST + 3.0,
        None => cost <= MAX_COST,
    };
    fine.then_some(kana)
}

/// `kana` with each ず and づ swapped where the other reads better by the
/// kana model, by `margin` or more; `None` when none is. Only ず/づ: じ/ぢ
/// would make 間近's まぢか まじか, as the kana model has it.
pub fn voicing(kana: &str, lm: &MixedModel, margin: f64) -> Option<String> {
    let mut chars: Vec<char> = kana.chars().collect();
    let at: Vec<usize> = (0..chars.len())
        .filter(|&i| matches!(chars[i], 'ず' | 'づ'))
        .collect();
    if at.is_empty() {
        return None;
    }
    let swap = |c: char| if c == 'ず' { 'づ' } else { 'ず' };
    let mut cost = lm.kana_cost(kana);
    let mut changed = false;
    // two passes: a swap can change how a ず/づ beside it reads
    for _ in 0..2 {
        let mut again = false;
        for &i in &at {
            chars[i] = swap(chars[i]);
            let swapped = lm.kana_cost(&chars.iter().collect::<String>());
            if cost - swapped >= margin {
                cost = swapped;
                again = true;
            } else {
                chars[i] = swap(chars[i]);
            }
        }
        changed |= again;
        if !again {
            break;
        }
    }
    changed.then(|| chars.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lm() -> crate::mixed::MixedModel {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        crate::mixed::MixedModel::load(&dir).unwrap()
    }

    #[test]
    fn a_letter_that_reads_as_no_kana_is_corrected() {
        let m = lm();
        // g for h, its neighbour: かいしゃ
        assert_eq!(correct("kaisga", Some(&m)).as_deref(), Some("かいしゃ"));
        // a swap (konnichiwa, read こんいちわ: nn is ん)
        assert_eq!(
            correct("konnihciwa", Some(&m)).as_deref(),
            Some("こんいちわ")
        );
        // a key too many
        assert_eq!(
            correct("konnichiwqa", Some(&m)).as_deref(),
            Some("こんいちわ")
        );
    }

    #[test]
    fn what_reads_is_left_alone() {
        let m = lm();
        assert_eq!(correct("kaisha", Some(&m)), None);
        // all of it reads (ありがつお): nothing to correct
        assert_eq!(correct("arigatuo", Some(&m)), None);
        assert_eq!(correct("", Some(&m)), None);
        assert_eq!(correct("Kaisga", Some(&m)).as_deref(), Some("かいしゃ"));
    }

    #[test]
    fn zu_and_du_as_the_kana_model_reads_them() {
        let m = lm();
        let fix = |k: &str| voicing(k, &m, VOICING_FIX);
        for (typed, meant) in [
            ("つずく", "つづく"),
            ("つずけてください", "つづけてください"),
            ("かたずける", "かたづける"),
            ("みかずき", "みかづき"),
            ("こずつみ", "こづつみ"),
            ("もとずく", "もとづく"),
            ("おこずかい", "おこづかい"),
            ("わかりずらい", "わかりづらい"),
            ("みづ", "みず"),
            ("づっと", "ずっと"),
            ("おとづれる", "おとずれる"),
        ] {
            assert_eq!(fix(typed).as_deref(), Some(meant), "{typed}");
        }
        // spelled right, or both in use
        for k in [
            "つづく",
            "みず",
            "ずっと",
            "すこしずつ",
            "きずな",
            "わずか",
            "いなずま",
            "きずく",
            "つまずく",
            "ひざまずく",
            "きずつく",
            "ちず",
            "じめん",
            "まぢか",
            "はなぢ",
            "かんじ",
        ] {
            assert_eq!(fix(k), None, "{k}");
        }
        // short of it, offered: 気づく as well as 築く
        assert_eq!(
            voicing("きずく", &m, VOICING_OFFER).as_deref(),
            Some("きづく")
        );
        assert_eq!(
            voicing("よこずな", &m, VOICING_OFFER).as_deref(),
            Some("よこづな")
        );
        assert_eq!(voicing("かな", &m, VOICING_OFFER), None);
    }
}
