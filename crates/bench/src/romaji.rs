//! The keys a person types for a reading: Hepburn-style romaji as the common
//! Japanese input methods take it (し shi, ち chi, つ tsu, ふ fu, じ ji, を wo),
//! ん as `nn` before a vowel, y, n or anything not kana and as `n` otherwise,
//! っ as the next consonant doubled (`xtu` when there is none), ー as `-`, the
//! Japanese punctuation as the ASCII key that types it, and digits and ASCII
//! symbols as they are.

const ROWS: [(&str, [&str; 5]); 19] = [
    ("", ["あ", "い", "う", "え", "お"]),
    ("k", ["か", "き", "く", "け", "こ"]),
    ("g", ["が", "ぎ", "ぐ", "げ", "ご"]),
    ("s", ["さ", "", "す", "せ", "そ"]),
    ("z", ["ざ", "", "ず", "ぜ", "ぞ"]),
    ("t", ["た", "", "", "て", "と"]),
    ("d", ["だ", "ぢ", "づ", "で", "ど"]),
    ("n", ["な", "に", "ぬ", "ね", "の"]),
    ("h", ["は", "ひ", "", "へ", "ほ"]),
    ("b", ["ば", "び", "ぶ", "べ", "ぼ"]),
    ("p", ["ぱ", "ぴ", "ぷ", "ぺ", "ぽ"]),
    ("m", ["ま", "み", "む", "め", "も"]),
    ("y", ["や", "", "ゆ", "", "よ"]),
    ("r", ["ら", "り", "る", "れ", "ろ"]),
    ("w", ["わ", "", "", "", "を"]),
    ("x", ["ぁ", "ぃ", "ぅ", "ぇ", "ぉ"]),
    ("xy", ["ゃ", "", "ゅ", "", "ょ"]),
    ("v", ["", "", "ゔ", "", ""]),
    ("xw", ["ゎ", "", "", "", ""]),
];
const VOWELS: [&str; 5] = ["a", "i", "u", "e", "o"];

/// The romaji of one kana, without the small kana after it.
fn single(c: char) -> Option<String> {
    let special = match c {
        'し' => Some("shi"),
        'じ' => Some("ji"),
        'ち' => Some("chi"),
        'つ' => Some("tsu"),
        'ふ' => Some("fu"),
        'ん' => Some("n"),
        'っ' => Some("xtu"),
        _ => None,
    };
    if let Some(s) = special {
        return Some(s.to_string());
    }
    let s = c.to_string();
    for (cons, row) in ROWS {
        if let Some(i) = row.iter().position(|k| *k == s) {
            return Some(format!("{cons}{}", VOWELS[i]));
        }
    }
    None
}

/// A kana and the small kana after it as one syllable (きゃ kya, ふぁ fa, てぃ thi).
fn pair(c: char, small: char) -> Option<String> {
    let yoon = |v: &str| -> Option<String> {
        Some(match c {
            'し' => format!("sh{v}"),
            'ち' => format!("ch{v}"),
            'じ' => format!("j{v}"),
            'き' | 'ぎ' | 'に' | 'ひ' | 'び' | 'ぴ' | 'み' | 'り' | 'ぢ' => {
                let base = single(c)?;
                format!("{}y{v}", &base[..base.len() - 1])
            }
            _ => return None,
        })
    };
    match small {
        'ゃ' => yoon("a"),
        'ゅ' => yoon("u"),
        'ょ' => yoon("o"),
        'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' => {
            let v = VOWELS[['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ']
                .iter()
                .position(|&k| k == small)?];
            Some(match (c, small) {
                ('ふ', _) if small != 'ぅ' => format!("f{v}"),
                ('ゔ', _) => format!("v{v}"),
                ('て', 'ぃ') => "thi".into(),
                ('で', 'ぃ') => "dhi".into(),
                ('て', 'ぅ') => "thu".into(),
                ('で', 'ぅ') => "dhu".into(),
                ('と', 'ぅ') => "twu".into(),
                ('ど', 'ぅ') => "dwu".into(),
                ('う', 'ぃ') => "wi".into(),
                ('う', 'ぇ') => "we".into(),
                ('い', 'ぇ') => "ye".into(),
                ('し', 'ぇ') => "she".into(),
                ('ち', 'ぇ') => "che".into(),
                ('じ', 'ぇ') => "je".into(),
                ('く', _) => format!("qw{v}").replace("qwu", "qu"),
                ('ぐ', _) => format!("gw{v}"),
                _ => return None,
            })
        }
        _ => None,
    }
}

fn is_small(c: char) -> bool {
    matches!(c, 'ゃ' | 'ゅ' | 'ょ' | 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ')
}

/// The key a Japanese punctuation mark or symbol is typed with.
fn punctuation(c: char) -> Option<char> {
    Some(match c {
        '、' => ',',
        '。' => '.',
        '？' => '?',
        '！' => '!',
        '「' => '[',
        '」' => ']',
        '・' => '/',
        '～' => '~',
        '（' => '(',
        '）' => ')',
        'ー' => '-',
        _ => return None,
    })
}

/// The keys typing `reading`; `Err` names the first char that cannot be typed.
pub fn keys(reading: &str) -> Result<String, char> {
    let chars: Vec<char> = reading.chars().collect();
    // syllables: (romaji, is kana)
    let mut syl: Vec<(String, bool)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_graphic() {
            syl.push((c.to_string(), false));
            i += 1;
            continue;
        }
        if let Some(k) = punctuation(c) {
            syl.push((k.to_string(), false));
            i += 1;
            continue;
        }
        if i + 1 < chars.len() && is_small(chars[i + 1]) {
            if let Some(p) = pair(c, chars[i + 1]) {
                syl.push((p, true));
                i += 2;
                continue;
            }
        }
        match single(c) {
            Some(s) => syl.push((s, true)),
            None => return Err(c),
        }
        i += 1;
    }
    let mut out = String::new();
    for (k, (s, kana)) in syl.iter().enumerate() {
        let next = syl.get(k + 1);
        if *kana && s == "n" {
            let starts_soft = next.is_none_or(|(n, kana)| {
                !*kana || n.starts_with(['a', 'i', 'u', 'e', 'o', 'y', 'n'])
            });
            out.push_str(if starts_soft { "nn" } else { "n" });
        } else if *kana && s == "xtu" {
            match next {
                Some((n, true))
                    if n.starts_with(|c: char| {
                        c.is_ascii_alphabetic() && !"aiueon".contains(c)
                    }) =>
                {
                    out.push(n.chars().next().unwrap_or('x'));
                }
                _ => out.push_str("xtu"),
            }
        } else {
            out.push_str(s);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::keys;

    #[test]
    fn romaji_as_typed() {
        assert_eq!(
            keys("きょうはいちにちあめだったね。").unwrap(),
            "kyouhaichinichiamedattane."
        );
        assert_eq!(keys("しんかんせん").unwrap(), "shinkansenn");
        assert_eq!(keys("こんや").unwrap(), "konnya");
        assert_eq!(keys("ほんの").unwrap(), "honnno");
        assert_eq!(keys("いっちょう").unwrap(), "icchou");
        assert_eq!(keys("ふぁいる").unwrap(), "fairu");
        assert_eq!(keys("ぱーてぃー").unwrap(), "pa-thi-");
        assert_eq!(keys("2026ねん11がつ").unwrap(), "2026nenn11gatsu");
        assert_eq!(keys("あっ！").unwrap(), "axtu!");
        assert_eq!(keys("「はい」、どうぞ").unwrap(), "[hai],douzo");
        assert!(keys("カタカナ").is_err());
    }
}
