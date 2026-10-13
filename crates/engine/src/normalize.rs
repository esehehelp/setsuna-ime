//! Candidate normalization for IME emission.

/// `.` / `,` (half- or fullwidth) right after a Latin letter or a digit stay ASCII (3.5, 1,000, e.g., v2.0,
/// Hello, world); elsewhere they become 。 / 、. Char for char, so positions carry over.
pub fn normalize_candidate(s: &str) -> String {
    let mut prev: Option<char> = None;
    let mapped: String = s
        .chars()
        .map(|c| {
            let after_ascii = prev.is_some_and(|p| p.is_ascii_alphanumeric());
            let m = match c {
                ',' | '\u{FF0C}' if after_ascii => ',',
                '.' | '\u{FF0E}' if after_ascii => '.',
                ',' | '\u{FF0C}' => '\u{3001}',
                '.' | '\u{FF0E}' => '\u{3002}',
                '?' | '\u{FF1F}' => '\u{FF1F}',
                '!' | '\u{FF01}' => '\u{FF01}',
                '\u{FF02}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                other => other,
            };
            prev = Some(m);
            m
        })
        .collect();
    if mapped.chars().all(|c| c.is_whitespace()) {
        return String::new();
    }
    mapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullwidth_ascii_to_halfwidth_preserves_case() {
        assert_eq!(normalize_candidate("ＡＩ"), "AI");
        assert_eq!(normalize_candidate("ａｂｃ"), "abc");
        assert_eq!(normalize_candidate("Ａpple"), "Apple");
        assert_eq!(normalize_candidate("PC は"), "PC は");
    }

    #[test]
    fn whitespace_only_collapses() {
        assert_eq!(normalize_candidate("  "), "");
        assert_eq!(normalize_candidate(""), "");
    }

    #[test]
    fn unifies_punctuation_family() {
        assert_eq!(normalize_candidate("あ，い．う"), "あ、い。う");
        assert_eq!(normalize_candidate("あ,い.う"), "あ、い。う");
        assert_eq!(normalize_candidate("今日は、"), "今日は、");
        assert_eq!(normalize_candidate("です."), "です。");
    }

    #[test]
    fn period_and_comma_after_latin_or_digit_stay_ascii() {
        assert_eq!(normalize_candidate("3.5パーセント"), "3.5パーセント");
        assert_eq!(normalize_candidate("１．５倍"), "1.5倍");
        assert_eq!(normalize_candidate("1,000円"), "1,000円");
        assert_eq!(normalize_candidate("e.g."), "e.g.");
        assert_eq!(normalize_candidate("Hello, world."), "Hello, world.");
        assert_eq!(normalize_candidate("Appleが発表した."), "Appleが発表した。");
        assert_eq!(normalize_candidate("a,b.c"), "a,b.c");
    }

    #[test]
    fn question_and_exclamation_stay_fullwidth() {
        assert_eq!(normalize_candidate("そうですか？"), "そうですか？");
        assert_eq!(normalize_candidate("そうですか?"), "そうですか？");
        assert_eq!(normalize_candidate("すごい！"), "すごい！");
        assert_eq!(normalize_candidate("すごい!"), "すごい！");
    }

    #[test]
    fn other_fullwidth_ascii_still_normalizes() {
        assert_eq!(normalize_candidate("ＡＢＣ"), "ABC");
        assert_eq!(normalize_candidate("１２３"), "123");
    }
}
