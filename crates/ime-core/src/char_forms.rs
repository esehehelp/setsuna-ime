//! Character-form conversions of a composition (F6–F10 in every Japanese
//! IME) and the width / script annotations of candidates.
//!
//! F6 hiragana, F7 katakana, F8 half-width katakana, F9 full-width
//! alphanumeric, F10 half-width alphanumeric. The alphanumeric forms spell
//! the composition in the romaji that was typed; kana that no longer has
//! its romaji (after a cursor edit) is spelled back from the romaji table.

use crate::kana_candidates::to_katakana;

/// A form F6–F10 convert the composition to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Hiragana,
    Katakana,
    HalfKatakana,
    FullAlnum,
    HalfAlnum,
}

impl Form {
    /// The form's name and its key, as the panel shows them.
    pub fn label(self) -> (&'static str, &'static str) {
        match self {
            Form::Hiragana => ("ひらがな", "F6"),
            Form::Katakana => ("全角カタカナ", "F7"),
            Form::HalfKatakana => ("半角カタカナ", "F8"),
            Form::FullAlnum => ("全角英数", "F9"),
            Form::HalfAlnum => ("半角英数", "F10"),
        }
    }

    /// Whether pressing the key again cycles the letter case (F9 / F10:
    /// `kana` → `KANA` → `Kana`).
    pub fn cycles_case(self) -> bool {
        matches!(self, Form::FullAlnum | Form::HalfAlnum)
    }
}

/// `reading` (hiragana, maybe with symbols) and `romaji` (what was typed
/// for it) in `form`. `case` is the F9 / F10 press count from 0: lower,
/// upper, capitalized.
pub fn convert(form: Form, reading: &str, romaji: &str, case: usize) -> String {
    match form {
        Form::Hiragana => reading.to_string(),
        Form::Katakana => to_katakana(reading),
        Form::HalfKatakana => to_half_katakana(&to_katakana(reading)),
        Form::FullAlnum => to_full_ascii(&with_case(romaji, case)),
        Form::HalfAlnum => with_case(romaji, case),
    }
}

fn with_case(s: &str, case: usize) -> String {
    match case % 3 {
        0 => s.to_ascii_lowercase(),
        1 => s.to_ascii_uppercase(),
        _ => {
            let lower = s.to_ascii_lowercase();
            let mut chars = lower.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        }
    }
}

/// Printable ASCII as its full-width form, space as U+3000.
pub fn to_full_ascii(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => '\u{3000}',
            '!'..='~' => char::from_u32(c as u32 - 0x21 + 0xff01).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// Full-width ASCII and the kana punctuation as the ASCII typed for them.
fn to_half_ascii(c: char) -> Option<char> {
    Some(match c {
        '\u{ff01}'..='\u{ff5e}' => char::from_u32(c as u32 - 0xff01 + 0x21)?,
        '\u{3000}' => ' ',
        '、' => ',',
        '。' => '.',
        'ー' => '-',
        '「' => '[',
        '」' => ']',
        '・' => '/',
        '￥' => '\\',
        _ if c.is_ascii() => c,
        _ => return None,
    })
}

/// The romaji for `text` as the user would have typed it: ASCII stays, kana
/// is spelled back from the romaji table (longest kana match first), the
/// kana punctuation goes back to its key.
pub fn romaji_of(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        // small っ: double the next consonant
        if chars[i] == 'っ' && i + 1 < chars.len() {
            if let Some((r, _)) = spell(&chars[i + 1..]) {
                if let Some(c) = r.chars().next().filter(|c| !"aiueon".contains(*c)) {
                    out.push(c);
                    i += 1;
                    continue;
                }
            }
        }
        if let Some((r, n)) = spell(&chars[i..]) {
            out.push_str(r);
            i += n;
            continue;
        }
        let c = chars[i];
        if let Some(h) = to_half_ascii(c) {
            out.push(h);
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// The romaji of the longest kana prefix of `chars` (2 chars for yōon), and
/// how many chars it spelled.
fn spell(chars: &[char]) -> Option<(&'static str, usize)> {
    for n in [2, 1] {
        if chars.len() < n {
            continue;
        }
        let kana: String = chars[..n].iter().collect();
        if let Some((r, _)) = KANA_ROMAJI.iter().find(|(_, k)| *k == kana) {
            return Some((r, n));
        }
    }
    None
}

/// Kana → the romaji spelling F9 / F10 write (Hepburn-leaning, as Mozc and
/// MS-IME do: し `shi`, つ `tsu`, ん `nn`).
#[rustfmt::skip]
const KANA_ROMAJI: &[(&str, &str)] = &[
    ("kya", "きゃ"), ("kyu", "きゅ"), ("kyo", "きょ"),
    ("gya", "ぎゃ"), ("gyu", "ぎゅ"), ("gyo", "ぎょ"),
    ("sha", "しゃ"), ("shu", "しゅ"), ("sho", "しょ"), ("she", "しぇ"),
    ("ja", "じゃ"), ("ju", "じゅ"), ("jo", "じょ"), ("je", "じぇ"),
    ("cha", "ちゃ"), ("chu", "ちゅ"), ("cho", "ちょ"), ("che", "ちぇ"),
    ("nya", "にゃ"), ("nyu", "にゅ"), ("nyo", "にょ"),
    ("hya", "ひゃ"), ("hyu", "ひゅ"), ("hyo", "ひょ"),
    ("bya", "びゃ"), ("byu", "びゅ"), ("byo", "びょ"),
    ("pya", "ぴゃ"), ("pyu", "ぴゅ"), ("pyo", "ぴょ"),
    ("mya", "みゃ"), ("myu", "みゅ"), ("myo", "みょ"),
    ("rya", "りゃ"), ("ryu", "りゅ"), ("ryo", "りょ"),
    ("thi", "てぃ"), ("dhi", "でぃ"), ("fa", "ふぁ"), ("fi", "ふぃ"),
    ("fe", "ふぇ"), ("fo", "ふぉ"), ("wi", "うぃ"), ("we", "うぇ"),
    ("va", "ゔぁ"), ("vi", "ゔぃ"), ("ve", "ゔぇ"), ("vo", "ゔぉ"),
    ("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お"),
    ("ka", "か"), ("ki", "き"), ("ku", "く"), ("ke", "け"), ("ko", "こ"),
    ("ga", "が"), ("gi", "ぎ"), ("gu", "ぐ"), ("ge", "げ"), ("go", "ご"),
    ("sa", "さ"), ("shi", "し"), ("su", "す"), ("se", "せ"), ("so", "そ"),
    ("za", "ざ"), ("ji", "じ"), ("zu", "ず"), ("ze", "ぜ"), ("zo", "ぞ"),
    ("ta", "た"), ("chi", "ち"), ("tsu", "つ"), ("te", "て"), ("to", "と"),
    ("da", "だ"), ("di", "ぢ"), ("du", "づ"), ("de", "で"), ("do", "ど"),
    ("na", "な"), ("ni", "に"), ("nu", "ぬ"), ("ne", "ね"), ("no", "の"),
    ("ha", "は"), ("hi", "ひ"), ("fu", "ふ"), ("he", "へ"), ("ho", "ほ"),
    ("ba", "ば"), ("bi", "び"), ("bu", "ぶ"), ("be", "べ"), ("bo", "ぼ"),
    ("pa", "ぱ"), ("pi", "ぴ"), ("pu", "ぷ"), ("pe", "ぺ"), ("po", "ぽ"),
    ("ma", "ま"), ("mi", "み"), ("mu", "む"), ("me", "め"), ("mo", "も"),
    ("ya", "や"), ("yu", "ゆ"), ("yo", "よ"),
    ("ra", "ら"), ("ri", "り"), ("ru", "る"), ("re", "れ"), ("ro", "ろ"),
    ("wa", "わ"), ("wo", "を"), ("nn", "ん"), ("vu", "ゔ"),
    ("xa", "ぁ"), ("xi", "ぃ"), ("xu", "ぅ"), ("xe", "ぇ"), ("xo", "ぉ"),
    ("xya", "ゃ"), ("xyu", "ゅ"), ("xyo", "ょ"), ("xtsu", "っ"), ("xwa", "ゎ"),
];

/// Full-width katakana (and the kana punctuation) as half-width.
pub fn to_half_katakana(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match half_kana(c) {
            Some(h) => out.push_str(h),
            None => out.push(c),
        }
    }
    out
}

fn half_kana(c: char) -> Option<&'static str> {
    const PLAIN: &str =
        "ｧｱｨｲｩｳｪｴｫｵｶｶｷｷｸｸｹｹｺｺｻｻｼｼｽｽｾｾｿｿﾀﾀﾁﾁｯﾂﾂﾃﾃﾄﾄﾅﾆﾇﾈﾉﾊﾊﾊﾋﾋﾋﾌﾌﾌﾍﾍﾍﾎﾎﾎﾏﾐﾑﾒﾓｬﾔｭﾕｮﾖﾗﾘﾙﾚﾛﾜﾜｲｴｦﾝｳ";
    match c {
        'ー' => return Some("ｰ"),
        '・' => return Some("･"),
        '、' => return Some("､"),
        '。' => return Some("｡"),
        '「' => return Some("｢"),
        '」' => return Some("｣"),
        _ => {}
    }
    // U+30A1 ァ ..= U+30F4 ヴ: the half-width base in PLAIN, then its mark
    let i = (c as u32).checked_sub(0x30a1)? as usize;
    if i > 0x30f4 - 0x30a1 {
        return None;
    }
    let base = PLAIN.chars().nth(i)?;
    let mark = voicing(c);
    let s: &'static str = HALF_KANA
        .iter()
        .find(|(b, m, _)| *b == base && *m == mark)
        .map(|(_, _, s)| *s)?;
    Some(s)
}

/// 0 plain, 1 dakuten, 2 handakuten.
fn voicing(c: char) -> u8 {
    match c {
        'ガ' | 'ギ' | 'グ' | 'ゲ' | 'ゴ' | 'ザ' | 'ジ' | 'ズ' | 'ゼ' | 'ゾ' | 'ダ' | 'ヂ'
        | 'ヅ' | 'デ' | 'ド' | 'バ' | 'ビ' | 'ブ' | 'ベ' | 'ボ' | 'ヴ' => 1,
        'パ' | 'ピ' | 'プ' | 'ペ' | 'ポ' => 2,
        _ => 0,
    }
}

/// (half-width base, voicing, output). Built for every base char so the
/// lookup above returns a `'static` str.
#[rustfmt::skip]
const HALF_KANA: &[(char, u8, &str)] = &[
    ('ｧ', 0, "ｧ"), ('ｱ', 0, "ｱ"), ('ｨ', 0, "ｨ"), ('ｲ', 0, "ｲ"), ('ｩ', 0, "ｩ"),
    ('ｳ', 0, "ｳ"), ('ｳ', 1, "ｳﾞ"), ('ｪ', 0, "ｪ"), ('ｴ', 0, "ｴ"), ('ｫ', 0, "ｫ"),
    ('ｵ', 0, "ｵ"), ('ｶ', 0, "ｶ"), ('ｶ', 1, "ｶﾞ"), ('ｷ', 0, "ｷ"), ('ｷ', 1, "ｷﾞ"),
    ('ｸ', 0, "ｸ"), ('ｸ', 1, "ｸﾞ"), ('ｹ', 0, "ｹ"), ('ｹ', 1, "ｹﾞ"), ('ｺ', 0, "ｺ"),
    ('ｺ', 1, "ｺﾞ"), ('ｻ', 0, "ｻ"), ('ｻ', 1, "ｻﾞ"), ('ｼ', 0, "ｼ"), ('ｼ', 1, "ｼﾞ"),
    ('ｽ', 0, "ｽ"), ('ｽ', 1, "ｽﾞ"), ('ｾ', 0, "ｾ"), ('ｾ', 1, "ｾﾞ"), ('ｿ', 0, "ｿ"),
    ('ｿ', 1, "ｿﾞ"), ('ﾀ', 0, "ﾀ"), ('ﾀ', 1, "ﾀﾞ"), ('ﾁ', 0, "ﾁ"), ('ﾁ', 1, "ﾁﾞ"),
    ('ｯ', 0, "ｯ"), ('ﾂ', 0, "ﾂ"), ('ﾂ', 1, "ﾂﾞ"), ('ﾃ', 0, "ﾃ"), ('ﾃ', 1, "ﾃﾞ"),
    ('ﾄ', 0, "ﾄ"), ('ﾄ', 1, "ﾄﾞ"), ('ﾅ', 0, "ﾅ"), ('ﾆ', 0, "ﾆ"), ('ﾇ', 0, "ﾇ"),
    ('ﾈ', 0, "ﾈ"), ('ﾉ', 0, "ﾉ"), ('ﾊ', 0, "ﾊ"), ('ﾊ', 1, "ﾊﾞ"), ('ﾊ', 2, "ﾊﾟ"),
    ('ﾋ', 0, "ﾋ"), ('ﾋ', 1, "ﾋﾞ"), ('ﾋ', 2, "ﾋﾟ"), ('ﾌ', 0, "ﾌ"), ('ﾌ', 1, "ﾌﾞ"),
    ('ﾌ', 2, "ﾌﾟ"), ('ﾍ', 0, "ﾍ"), ('ﾍ', 1, "ﾍﾞ"), ('ﾍ', 2, "ﾍﾟ"), ('ﾎ', 0, "ﾎ"),
    ('ﾎ', 1, "ﾎﾞ"), ('ﾎ', 2, "ﾎﾟ"), ('ﾏ', 0, "ﾏ"), ('ﾐ', 0, "ﾐ"), ('ﾑ', 0, "ﾑ"),
    ('ﾒ', 0, "ﾒ"), ('ﾓ', 0, "ﾓ"), ('ｬ', 0, "ｬ"), ('ﾔ', 0, "ﾔ"), ('ｭ', 0, "ｭ"),
    ('ﾕ', 0, "ﾕ"), ('ｮ', 0, "ｮ"), ('ﾖ', 0, "ﾖ"), ('ﾗ', 0, "ﾗ"), ('ﾘ', 0, "ﾘ"),
    ('ﾙ', 0, "ﾙ"), ('ﾚ', 0, "ﾚ"), ('ﾛ', 0, "ﾛ"), ('ﾜ', 0, "ﾜ"), ('ｲ', 0, "ｲ"),
    ('ｴ', 0, "ｴ"), ('ｦ', 0, "ｦ"), ('ﾝ', 0, "ﾝ"),
];

/// What the candidate window notes beside a candidate whose script or width
/// is not obvious at a glance (as Mozc does): `[全] カタカナ`, `[半] 英数`.
/// `None` for kanji and mixed text.
pub fn annotation(candidate: &str) -> Option<&'static str> {
    let all = |f: fn(char) -> bool| !candidate.is_empty() && candidate.chars().all(f);
    let is_full_kata = |c: char| matches!(c, '\u{30a1}'..='\u{30fa}' | 'ー' | '・');
    let is_half_kata = |c: char| matches!(c, '\u{ff66}'..='\u{ff9f}');
    let is_hira = |c: char| matches!(c, '\u{3041}'..='\u{3096}' | 'ー');
    let is_full_alnum = |c: char| matches!(c, '\u{ff10}'..='\u{ff19}' | '\u{ff21}'..='\u{ff3a}' | '\u{ff41}'..='\u{ff5a}');
    let is_half_alnum = |c: char| c.is_ascii_alphanumeric();
    if all(is_full_kata) {
        Some("[全] カタカナ")
    } else if all(is_half_kata) {
        Some("[半] カタカナ")
    } else if all(is_hira) {
        Some("ひらがな")
    } else if all(is_full_alnum) {
        Some("[全] 英数")
    } else if all(is_half_alnum) {
        Some("[半] 英数")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kana_forms() {
        assert_eq!(convert(Form::Hiragana, "がっこう", "gakkou", 0), "がっこう");
        assert_eq!(convert(Form::Katakana, "がっこう", "gakkou", 0), "ガッコウ");
        assert_eq!(convert(Form::HalfKatakana, "がっこう", "gakkou", 0), "ｶﾞｯｺｳ");
        assert_eq!(convert(Form::HalfKatakana, "ぱーてぃ。", "", 0), "ﾊﾟｰﾃｨ｡");
        assert_eq!(convert(Form::HalfKatakana, "ゔぁ", "", 0), "ｳﾞｧ");
    }

    #[test]
    fn alnum_forms_cycle_the_case() {
        assert_eq!(convert(Form::HalfAlnum, "", "tokyo", 0), "tokyo");
        assert_eq!(convert(Form::HalfAlnum, "", "tokyo", 1), "TOKYO");
        assert_eq!(convert(Form::HalfAlnum, "", "tokyo", 2), "Tokyo");
        assert_eq!(convert(Form::HalfAlnum, "", "tokyo", 3), "tokyo");
        assert_eq!(convert(Form::FullAlnum, "", "tokyo", 0), "ｔｏｋｙｏ");
        assert_eq!(convert(Form::FullAlnum, "", "a1-", 1), "Ａ１－");
    }

    #[test]
    fn romaji_is_spelled_back_from_kana() {
        assert_eq!(romaji_of("がっこう"), "gakkou");
        assert_eq!(romaji_of("しゃしん"), "shashinn");
        assert_eq!(romaji_of("abc"), "abc");
        assert_eq!(romaji_of("らーめん、"), "ra-menn,");
        assert_eq!(romaji_of("ちょっと"), "chotto");
    }

    #[test]
    fn annotations() {
        assert_eq!(annotation("カタカナ"), Some("[全] カタカナ"));
        assert_eq!(annotation("ｶﾀｶﾅ"), Some("[半] カタカナ"));
        assert_eq!(annotation("ひらがな"), Some("ひらがな"));
        assert_eq!(annotation("ＡＢＣ１"), Some("[全] 英数"));
        assert_eq!(annotation("abc1"), Some("[半] 英数"));
        assert_eq!(annotation("漢字"), None);
        assert_eq!(annotation("漢かな"), None);
        assert_eq!(annotation(""), None);
    }
}
