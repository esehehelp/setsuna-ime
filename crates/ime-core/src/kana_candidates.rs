//! Kana entries of the candidate list.
//!
//! The engine's candidates are model hypotheses, so an all-hiragana one can
//! differ from what was typed (`もし` for `もじ`). As an IME the list should
//! offer the typed reading itself as hiragana and as katakana, and no other
//! pure-hiragana spelling. `arrange` drops all-hiragana candidates that are
//! not the reading and puts the reading and its katakana right after the
//! model's top two (3rd / 4th): below that the model's candidates are mostly
//! near-misses, and at the end of a 20-candidate list the kana forms sat on
//! the third page of the window. A kana form the model ranked in its top two
//! (は for a particle, パソコン) stays there. Order is otherwise kept and
//! duplicates dropped.

use std::collections::HashSet;

pub fn arrange(reading: &str, candidates: Vec<String>) -> Vec<String> {
    if reading.is_empty() {
        return candidates;
    }
    let kana = [reading.to_string(), to_katakana(reading)];
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::with_capacity(candidates.len() + 2);
    for c in candidates {
        if c.is_empty() || (is_hiragana(&c) && c != reading) {
            continue;
        }
        // A kana form below the model's top two moves up with the others.
        if out.len() >= KEEP_ON_TOP && kana.contains(&c) {
            continue;
        }
        if seen.insert(c.clone()) {
            out.push(c);
        }
    }
    let at = out.len().min(KEEP_ON_TOP);
    let missing: Vec<String> = kana
        .into_iter()
        .filter(|k| seen.insert(k.clone()))
        .collect();
    out.splice(at..at, missing);
    out
}

/// Model candidates kept above the reading and its katakana.
const KEEP_ON_TOP: usize = 2;

/// Every char is hiragana (including ゝゞ) or the prolonged sound mark.
fn is_hiragana(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| matches!(c, '\u{3041}'..='\u{3096}' | '\u{309D}' | '\u{309E}' | 'ー'))
}

pub fn to_katakana(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{3041}'..='\u{3096}' | '\u{309D}' | '\u{309E}' => {
                char::from_u32(c as u32 + 0x60).unwrap_or(c)
            }
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn wrong_hiragana_is_dropped_and_reading_inserted() {
        let got = arrange("もじ", v(&["文字", "もし", "模字"]));
        assert_eq!(got, v(&["文字", "模字", "もじ", "モジ"]));
    }

    #[test]
    fn reading_and_katakana_go_third_and_fourth() {
        let got = arrange("かみ", v(&["紙", "神", "髪", "加味", "カミ"]));
        assert_eq!(got, v(&["紙", "神", "かみ", "カミ", "髪", "加味"]));
    }

    #[test]
    fn reading_keeps_a_top_two_rank() {
        let got = arrange("は", v(&["は", "葉", "歯"]));
        assert_eq!(got, v(&["は", "葉", "ハ", "歯"]));
    }

    #[test]
    fn model_katakana_keeps_its_rank_and_is_not_repeated() {
        let got = arrange("ぱそこん", v(&["パソコン", "ぱそこん"]));
        assert_eq!(got, v(&["パソコン", "ぱそこん"]));
        let got = arrange("ぱそこん", v(&["パソコン", "PC", "ぱそこん"]));
        assert_eq!(got, v(&["パソコン", "PC", "ぱそこん"]));
    }

    #[test]
    fn katakana_variants_are_kept_and_wrong_hiragana_dropped() {
        // Katakana spellings other than the reading's are legitimate
        // conversions; only the pure-hiragana misspelling goes. The reading's
        // katakana is already in the list, so only the reading is inserted.
        let got = arrange(
            "こんぴゅーたー",
            v(&["コンピューター", "コンピュータ", "こんぴゅーた"]),
        );
        assert_eq!(
            got,
            v(&["コンピューター", "コンピュータ", "こんぴゅーたー"])
        );
    }

    #[test]
    fn empty_model_output_still_offers_reading_and_katakana() {
        assert_eq!(arrange("あい", Vec::new()), v(&["あい", "アイ"]));
    }

    #[test]
    fn reading_with_digits() {
        let got = arrange("3がつ", v(&["3月", "さんがつ"]));
        assert_eq!(got, v(&["3月", "3がつ", "3ガツ"]));
    }

    #[test]
    fn duplicates_are_dropped() {
        let got = arrange("かみ", v(&["紙", "紙", "神", "紙"]));
        assert_eq!(got, v(&["紙", "神", "かみ", "カミ"]));
    }

    #[test]
    fn a_short_list_gets_the_kana_after_it() {
        assert_eq!(arrange("か", v(&["可"])), v(&["可", "か", "カ"]));
    }
}
