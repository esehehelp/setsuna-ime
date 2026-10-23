//! Kana entries of the candidate list.
//!
//! The engine's candidates are model hypotheses, so an all-hiragana one can
//! differ from what was typed (`もし` for `もじ`). As an IME the list should
//! offer the typed reading itself as hiragana and as katakana, and no other
//! pure-hiragana spelling. `arrange` drops all-hiragana candidates that are
//! not the reading, and kana-only ones with katakana that do not spell it
//! (`spells_as_kana`: デタ or テルタ for でるた, ウルス for ういるす; a
//! katakana spelling of its own, コンピュータ for こんぴゅーたー, stays), and
//! puts the reading and its katakana right after the
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
    let reading_chars: Vec<char> = reading.chars().collect();
    for c in candidates {
        if c.is_empty()
            || (is_hiragana(&c) && c != reading)
            || (is_kana_only(&c) && !spells_as_kana(&c, &reading_chars))
        {
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

/// Every char is kana of either script or the prolonged sound mark.
fn is_kana_only(s: &str) -> bool {
    !s.is_empty() && s.chars().all(crate::reading_align::is_kana)
}

/// Whether a kana-only `surface` spells `reading` out, char for char as
/// hiragana, but for the ways katakana writes a word otherwise than its
/// reading: a prolonged sound mark added, left out or for a vowel
/// (コンピュータ, ケータイ for けいたい), a small vowel for a large one or
/// the other way (ウィルス for ういるす), ヴ and a small vowel for a b-row
/// kana (ヴァイオリン for ばいおりん), ケ for が. A reading char left out
/// (デタ for でるた) or another kana (テルタ) does not.
pub fn spells_as_kana(surface: &str, reading: &[char]) -> bool {
    let s: Vec<char> = surface.chars().map(crate::reading_align::to_hira).collect();
    let (n, m) = (s.len(), reading.len());
    // ok[i][j]: s[i..] spells reading[j..]
    let mut ok = vec![vec![false; m + 1]; n + 1];
    ok[n][m] = true;
    for i in (0..=n).rev() {
        for j in (0..=m).rev() {
            if i == n && j == m {
                continue;
            }
            let mut v = false;
            if i < n && j < m {
                let (a, b) = (s[i], reading[j]);
                v |= (a == b || (a == 'け' && b == 'が') || small_for(a, b) || small_for(b, a))
                    && ok[i + 1][j + 1];
                // ー for a vowel
                v |= a == 'ー' && "あいうえお".contains(b) && ok[i + 1][j + 1];
                // ヴ (and a small vowel) for a b-row kana
                if a == 'ゔ' {
                    v |= b == 'ぶ' && ok[i + 1][j + 1];
                    if let Some(&small) = s.get(i + 1) {
                        v |= b_row(small) == Some(b) && ok[i + 2][j + 1];
                    }
                }
            }
            // a prolonged sound mark added or left out
            v |= i < n && s[i] == 'ー' && ok[i + 1][j];
            v |= j < m && reading[j] == 'ー' && ok[i][j + 1];
            ok[i][j] = v;
        }
    }
    ok[0][0]
}

/// `small` is the small form of the vowel `large` (ぃ for い).
fn small_for(small: char, large: char) -> bool {
    matches!(
        (small, large),
        ('ぁ', 'あ') | ('ぃ', 'い') | ('ぅ', 'う') | ('ぇ', 'え') | ('ぉ', 'お')
    )
}

/// The b-row kana ヴ and the small vowel `small` write (ヴァ: ば).
fn b_row(small: char) -> Option<char> {
    Some(match small {
        'ぁ' => 'ば',
        'ぃ' => 'び',
        'ぇ' => 'べ',
        'ぉ' => 'ぼ',
        _ => return None,
    })
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
    fn kana_only_candidates_that_lose_a_reading_char_are_dropped() {
        let got = arrange("でるた", v(&["デルタ", "デタ", "デル", "テルタ", "出ルタ"]));
        assert_eq!(got, v(&["デルタ", "出ルタ", "でるた"]));
        let got = arrange("ういるす", v(&["ウイルス", "ウルス", "ウィルス"]));
        assert_eq!(got, v(&["ウイルス", "ウィルス", "ういるす"]));
        // a kana-only candidate that leaves out what is not kana in the reading
        let got = arrange("3でるた", v(&["3デルタ", "デルタ"]));
        assert_eq!(got, v(&["3デルタ", "3でるた"]));
    }

    #[test]
    fn katakana_writes_some_words_otherwise_than_their_reading() {
        let r = |s: &str| s.chars().collect::<Vec<_>>();
        assert!(spells_as_kana("コンピュータ", &r("こんぴゅーたー")));
        assert!(spells_as_kana("ケータイ", &r("けいたい")));
        assert!(spells_as_kana("ウィルス", &r("ういるす")));
        assert!(spells_as_kana("ヴァイオリン", &r("ばいおりん")));
        assert!(spells_as_kana("ヴァイオリン", &r("ゔぁいおりん")));
        assert!(spells_as_kana("ヴ", &r("ぶ")));
        assert!(spells_as_kana("ケ", &r("が")));
        assert!(spells_as_kana("でルタ", &r("でるた")));
        assert!(!spells_as_kana("デタ", &r("でるた")));
        assert!(!spells_as_kana("テルタ", &r("でるた")));
        assert!(!spells_as_kana("バオリン", &r("ばいおりん")));
        assert!(!spells_as_kana("デルタ", &r("3でるた")));
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
