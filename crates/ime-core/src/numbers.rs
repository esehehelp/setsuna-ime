//! Candidates for numbers, made by rule as Mozc's number, date and
//! calculator rewriters and azooKey's special conversions make them (the
//! rules are facts: era start years, numeral characters; no code taken):
//!
//! - digits: 1,234 / １２３４ / 千二百三十四 / 一二三四 / 壱阡弐百参拾四 /
//!   1万2345, circled and Roman numerals for small ones, and for three or
//!   four digits the times and dates they may be (1:23, 1時23分, 1/23,
//!   1月23日);
//! - a number read in kana (じゅうに, さんぜんまん): 12 and 十二;
//! - a year: 2025ねん -> 令和7年 (both eras in a year of a change),
//!   れいわ7ねん -> 2025年;
//! - a calculation typed with `=` before or after it (1+2=): its result
//!   first, then the expression with it;
//! - `U+3042`: the character.

/// Eras from Meiji, with their first years.
const ERAS: &[(&str, &str, i64)] = &[
    ("令和", "れいわ", 2019),
    ("平成", "へいせい", 1989),
    ("昭和", "しょうわ", 1926),
    ("大正", "たいしょう", 1912),
    ("明治", "めいじ", 1868),
];

/// Candidates for `reading` that are a number (empty when it is none).
pub fn candidates(reading: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(c) = calculation(reading) {
        out.extend(c);
    } else if let Some(c) = unicode(reading) {
        out.push(c);
    } else if let Some(c) = year(reading) {
        out.extend(c);
    } else if reading.chars().all(|c| c.is_ascii_digit()) && !reading.is_empty() {
        out.extend(digit_forms(reading));
    } else if let Some(n) = kana_number(reading) {
        out.push(n.to_string());
        if n >= 1000 {
            out.push(group(n));
        }
        out.push(kanji(n));
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|c| c != reading && seen.insert(c.clone()));
    out
}

/// Whether the result of `reading` is a calculation (its result goes first).
pub fn is_calculation(reading: &str) -> bool {
    calculation(reading).is_some()
}

/// `candidates` with the number candidates in: a calculation's first, the
/// others after the kana forms (as the date and time candidates go).
pub fn insert(reading: &str, candidates: Vec<String>, generated: Vec<String>) -> Vec<String> {
    if generated.is_empty() {
        return candidates;
    }
    if is_calculation(reading) {
        let mut out = generated;
        for c in candidates {
            if !out.contains(&c) {
                out.push(c);
            }
        }
        return out;
    }
    crate::date_time::insert(reading, candidates, generated)
}

/// The forms of a number typed in digits.
fn digit_forms(digits: &str) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = digits.trim_start_matches('0');
    let n: Option<u64> = digits.parse().ok().filter(|_| digits.len() <= 16);
    if let Some(n) = n {
        if n >= 1000 && trimmed.len() == digits.len() {
            out.push(group(n));
        }
    }
    out.push(crate::char_forms::to_full_ascii(digits));
    if let Some(n) = n.filter(|_| trimmed.len() == digits.len() || digits == "0") {
        out.push(kanji(n));
        if digits.len() > 1 {
            out.push(digits.chars().map(kanji_digit).collect());
        }
        if n >= 10_000 {
            out.push(man(n));
        }
        out.push(daiji(n));
        if let Some(c) = circled(n) {
            out.push(c);
        }
        if let Some(r) = roman(n) {
            out.push(r);
        }
    } else {
        out.push(digits.chars().map(kanji_digit).collect());
    }
    out.extend(times_and_dates(digits));
    out
}

/// 1,234,567
fn group(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn kanji_digit(c: char) -> char {
    "〇一二三四五六七八九"
        .chars()
        .nth(c.to_digit(10).unwrap_or(0) as usize)
        .unwrap_or(c)
}

const BIG: [(u64, &str); 4] = [
    (1_0000_0000_0000_0000, "京"),
    (1_0000_0000_0000, "兆"),
    (1_0000_0000, "億"),
    (1_0000, "万"),
];

/// 千二百三十四万五千六百七十八 (positional kanji with 十百千 and 万億兆京).
fn kanji(n: u64) -> String {
    if n == 0 {
        return "〇".to_string();
    }
    fn below(n: u64, units: [&str; 3], digits: &str) -> String {
        let d: Vec<char> = digits.chars().collect();
        let mut out = String::new();
        for (i, (p, u)) in [(1000, units[2]), (100, units[1]), (10, units[0])]
            .iter()
            .enumerate()
        {
            let _ = i;
            let k = (n / p) % 10;
            if k > 1 {
                out.push(d[k as usize]);
            }
            if k > 0 {
                out.push_str(u);
            }
        }
        let k = n % 10;
        if k > 0 {
            out.push(d[k as usize]);
        }
        out
    }
    let mut out = String::new();
    let mut rest = n;
    for (unit, name) in BIG {
        if rest >= unit {
            out.push_str(&below(
                rest / unit,
                ["十", "百", "千"],
                "〇一二三四五六七八九",
            ));
            out.push_str(name);
            rest %= unit;
        }
    }
    out.push_str(&below(rest, ["十", "百", "千"], "〇一二三四五六七八九"));
    out
}

/// 大字: 壱萬弐阡参百四拾伍 as used on documents (壱弐参, 拾 阡 萬; 1 is
/// always written).
fn daiji(n: u64) -> String {
    if n == 0 {
        return "零".to_string();
    }
    let d: Vec<char> = "零壱弐参四伍六七八九".chars().collect();
    let below = |n: u64| {
        let mut out = String::new();
        for (p, u) in [(1000, '阡'), (100, '百'), (10, '拾')] {
            let k = (n / p) % 10;
            if k > 0 {
                out.push(d[k as usize]);
                out.push(u);
            }
        }
        if n % 10 > 0 {
            out.push(d[(n % 10) as usize]);
        }
        out
    };
    let mut out = String::new();
    let mut rest = n;
    for (unit, name) in BIG {
        if rest >= unit {
            out.push_str(&below(rest / unit));
            out.push_str(if name == "万" { "萬" } else { name });
            rest %= unit;
        }
    }
    out.push_str(&below(rest));
    out
}

/// 1万2345, 3億
fn man(n: u64) -> String {
    let mut out = String::new();
    let mut rest = n;
    for (unit, name) in BIG {
        if rest >= unit {
            out.push_str(&(rest / unit).to_string());
            out.push_str(name);
            rest %= unit;
        }
    }
    if rest > 0 {
        out.push_str(&rest.to_string());
    }
    out
}

fn circled(n: u64) -> Option<String> {
    let c = match n {
        1..=20 => 0x2460 + n as u32 - 1,
        21..=35 => 0x3251 + n as u32 - 21,
        36..=50 => 0x32B1 + n as u32 - 36,
        _ => return None,
    };
    char::from_u32(c).map(String::from)
}

fn roman(n: u64) -> Option<String> {
    match n {
        1..=12 => char::from_u32(0x2160 + n as u32 - 1).map(String::from),
        _ => None,
    }
}

/// 3 digits: 1:23 1時23分 1/23 1月23日; 4 digits: 12:34 12時34分 12/31
/// 12月31日 (each when the parts are a time or a date).
fn times_and_dates(d: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (a, b) = match d.len() {
        3 => d.split_at(1),
        4 => d.split_at(2),
        _ => return out,
    };
    let (Ok(h), Ok(m)) = (a.parse::<u32>(), b.parse::<u32>()) else {
        return out;
    };
    if h <= 24 && m < 60 {
        out.push(format!("{h}:{b}"));
        out.push(format!("{h}時{m}分"));
    }
    if (1..=12).contains(&h) && (1..=31).contains(&m) {
        out.push(format!("{h}/{m}"));
        out.push(format!("{h}月{m}日"));
    }
    out
}

/// A number read in kana, up to the 兆s: じゅうに, にせんにじゅうご,
/// さんおくごせんまん.
pub fn kana_number(s: &str) -> Option<u64> {
    if s.is_empty() || s.chars().any(|c| !('ぁ'..='ゖ').contains(&c)) {
        return None;
    }
    // a unit alone is a word far more often (線, 満, 奥, 町, 銃)
    if ["じゅう", "ひゃく", "せん", "まん", "おく", "ちょう"].contains(&s) {
        return None;
    }
    let mut total = 0u64;
    let mut rest = s.to_string();
    let mut any = false;
    for (unit, name) in [
        (1_0000_0000_0000u64, "ちょう"),
        (1_0000_0000, "おく"),
        (1_0000, "まん"),
    ] {
        if let Some((head, tail)) = rest.split_once(name) {
            let k = if head.is_empty() {
                1
            } else {
                below_10000(head)?
            };
            total += k * unit;
            rest = tail.to_string();
            any = true;
        }
    }
    if rest.is_empty() {
        return any.then_some(total);
    }
    let k = below_10000(&rest)?;
    // a bare digit word (に, ご, し) is too often something else
    if !any && k < 10 && s.chars().count() < 3 {
        return None;
    }
    Some(total + k)
}

fn below_10000(s: &str) -> Option<u64> {
    let digit = |s: &str| -> Option<u64> {
        Some(match s {
            "" => 0,
            "いち" | "いっ" => 1,
            "に" => 2,
            "さん" => 3,
            // よ as in よねん, よじ
            "よん" | "よ" | "し" => 4,
            "ご" => 5,
            "ろく" | "ろっ" => 6,
            "なな" | "しち" => 7,
            "はち" | "はっ" => 8,
            "きゅう" | "く" => 9,
            _ => return None,
        })
    };
    let s = s.replace("じゅっ", "じゅう").replace("じっ", "じゅう");
    let mut n = 0u64;
    let mut rest = s.as_str();
    for (prefixes, value) in [
        (
            &[
                "きゅうせん",
                "はっせん",
                "ななせん",
                "ろくせん",
                "ごせん",
                "よんせん",
                "さんぜん",
                "にせん",
                "いっせん",
                "せん",
            ][..],
            1000u64,
        ),
        (
            &[
                "きゅうひゃく",
                "はっぴゃく",
                "ななひゃく",
                "ろっぴゃく",
                "ごひゃく",
                "よんひゃく",
                "さんびゃく",
                "にひゃく",
                "ひゃく",
            ][..],
            100,
        ),
    ] {
        for p in prefixes {
            if let Some(r) = rest.strip_prefix(p) {
                let head = &p[..p.len()
                    - if value == 1000 {
                        "せん".len()
                    } else {
                        "ひゃく".len()
                    }];
                let k = match head {
                    "" => 1,
                    "さん" | "さんぜ" => 3,
                    "は" | "はっ" => 8,
                    "ろっ" | "ろ" => 6,
                    "いっ" => 1,
                    h => digit(h)?,
                };
                n += k * value;
                rest = r;
                break;
            }
        }
    }
    if let Some((a, b)) = rest.split_once("じゅう") {
        n += if a.is_empty() { 1 } else { digit(a)? } * 10 + digit(b)?;
    } else {
        n += digit(rest)?;
    }
    (n > 0 || s == "ぜろ").then_some(n)
}

/// A year: `2025ねん` (digits) -> its era years; `れいわ7ねん` /
/// `れいわななねん` / `れいわがんねん` -> the Western year.
fn year(reading: &str) -> Option<Vec<String>> {
    let body = reading.strip_suffix("ねん")?;
    if !body.is_empty() && body.chars().all(|c| c.is_ascii_digit()) {
        let y: i64 = body.parse().ok()?;
        let mut out = Vec::new();
        for (i, (name, _, start)) in ERAS.iter().enumerate() {
            if y >= *start {
                out.push(era_year(name, y - start + 1));
                // the year an era began is the last of the one before
                if y == *start {
                    if let Some((prev, _, prev_start)) = ERAS.get(i + 1) {
                        out.push(era_year(prev, y - prev_start + 1));
                    }
                }
                break;
            }
        }
        return (!out.is_empty()).then_some(out);
    }
    for (name, kana, start) in ERAS {
        let Some(n) = body.strip_prefix(kana) else {
            continue;
        };
        let k = if n == "がん" {
            1
        } else if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) {
            n.parse().ok()?
        } else {
            kana_number(n).or_else(|| below_10000(n))? as i64
        };
        if k < 1 || k > 99 {
            return None;
        }
        return Some(vec![format!("{}年", start + k - 1), era_year(name, k)]);
    }
    None
}

fn era_year(name: &str, k: i64) -> String {
    if k == 1 {
        format!("{name}元年")
    } else {
        format!("{name}{k}年")
    }
}

/// `U+3042` (typed as うﾌ+3042 too: `u` is う in romaji) -> the character.
fn unicode(reading: &str) -> Option<String> {
    let rest = ["U+", "u+", "う+", "U＋", "u＋", "う＋"]
        .iter()
        .find_map(|p| reading.strip_prefix(p))?;
    if !(4..=6).contains(&rest.len()) || !rest.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let c = char::from_u32(u32::from_str_radix(rest, 16).ok()?)?;
    (!c.is_control()).then(|| c.to_string())
}

/// A calculation with `=` before or after it: the result, then the
/// expression with it (`1+2=` -> 3, 1+2=3).
fn calculation(reading: &str) -> Option<Vec<String>> {
    let ascii: String = reading
        .chars()
        .map(|c| match c {
            '＋' => '+',
            '－' | 'ー' | '−' => '-',
            '＊' | '×' => '*',
            '／' | '÷' | '・' => '/',
            '＝' => '=',
            '（' => '(',
            '）' => ')',
            '＾' => '^',
            '．' => '.',
            c @ '０'..='９' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            c => c,
        })
        .collect();
    let expr = ascii
        .strip_suffix('=')
        .or_else(|| ascii.strip_prefix('='))?;
    if !expr.chars().any(|c| "+-*/^".contains(c))
        || !expr
            .chars()
            .all(|c| c.is_ascii_digit() || "+-*/^().".contains(c))
    {
        return None;
    }
    let mut p = Parser {
        s: expr.as_bytes(),
        i: 0,
    };
    let v = p.sum()?;
    if p.i != p.s.len() || !v.is_finite() {
        return None;
    }
    let result = format_number(v);
    Some(vec![result.clone(), format!("{expr}={result}")])
}

fn format_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.10}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// A recursive-descent calculator: + - * / ^ ( ), unary minus.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn sum(&mut self) -> Option<f64> {
        let mut v = self.product()?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.i += 1;
            let r = self.product()?;
            v = if op == b'+' { v + r } else { v - r };
        }
        Some(v)
    }

    fn product(&mut self) -> Option<f64> {
        let mut v = self.power()?;
        while let Some(op @ (b'*' | b'/')) = self.peek() {
            self.i += 1;
            let r = self.power()?;
            if op == b'/' && r == 0.0 {
                return None;
            }
            v = if op == b'*' { v * r } else { v / r };
        }
        Some(v)
    }

    fn power(&mut self) -> Option<f64> {
        let base = self.unary()?;
        if self.peek() == Some(b'^') {
            self.i += 1;
            let e = self.power()?;
            return Some(base.powf(e));
        }
        Some(base)
    }

    fn unary(&mut self) -> Option<f64> {
        if self.peek() == Some(b'-') {
            self.i += 1;
            return Some(-self.unary()?);
        }
        self.atom()
    }

    fn atom(&mut self) -> Option<f64> {
        if self.peek() == Some(b'(') {
            self.i += 1;
            let v = self.sum()?;
            if self.peek() != Some(b')') {
                return None;
            }
            self.i += 1;
            return Some(v);
        }
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || c == b'.') {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[start..self.i])
            .ok()?
            .parse()
            .ok()
    }
}

/// How a number is read where a reading ends: as it is said alone (いち,
/// にじゅう), or in a form only a counter after it takes (いっ of いっかい, よ
/// of よじ, ひと of ひとつ), with the kana the counter's reading may start
/// with ("" for any).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Said {
    Alone,
    BeforeCounter(&'static str),
}

/// The kana a geminated number (いっ, はっ, じゅっ) goes before.
const VOICELESS: &str = "かきくけこさしすせそたちつてとぱぴぷぺぽ";
const GEMINATED: Said = Said::BeforeCounter(VOICELESS);
/// よ (よじ, よねん, よにん), and し and く (しがつ, くじ).
const YO: Said = Said::BeforeCounter("じねに");
const SHI_KU: Said = Said::BeforeCounter("がじ");
const NATIVE: Said = Said::BeforeCounter("");

/// The ways a digit 1-9 is read in a number, and before a counter.
fn ones(d: u64) -> &'static [(&'static str, Said)] {
    match d {
        1 => &[("いち", Said::Alone), ("いっ", GEMINATED)],
        2 => &[("に", Said::Alone)],
        3 => &[("さん", Said::Alone)],
        4 => &[
            ("よん", Said::Alone),
            ("よっ", GEMINATED),
            ("よ", YO),
            ("し", SHI_KU),
        ],
        5 => &[("ご", Said::Alone)],
        6 => &[("ろく", Said::Alone), ("ろっ", GEMINATED)],
        7 => &[("なな", Said::Alone), ("しち", Said::Alone)],
        8 => &[("はち", Said::Alone), ("はっ", GEMINATED)],
        9 => &[("きゅう", Said::Alone), ("く", SHI_KU)],
        _ => &[],
    }
}

fn tens(d: u64) -> &'static [(&'static str, Said)] {
    match d {
        1 => &[
            ("じゅう", Said::Alone),
            ("じゅっ", GEMINATED),
            ("じっ", GEMINATED),
        ],
        2 => &[
            ("にじゅう", Said::Alone),
            ("にじゅっ", GEMINATED),
            ("にじっ", GEMINATED),
        ],
        3 => &[
            ("さんじゅう", Said::Alone),
            ("さんじゅっ", GEMINATED),
            ("さんじっ", GEMINATED),
        ],
        4 => &[
            ("よんじゅう", Said::Alone),
            ("しじゅう", Said::Alone),
            ("よんじゅっ", GEMINATED),
            ("よんじっ", GEMINATED),
        ],
        5 => &[
            ("ごじゅう", Said::Alone),
            ("ごじゅっ", GEMINATED),
            ("ごじっ", GEMINATED),
        ],
        6 => &[
            ("ろくじゅう", Said::Alone),
            ("ろくじゅっ", GEMINATED),
            ("ろくじっ", GEMINATED),
        ],
        7 => &[
            ("ななじゅう", Said::Alone),
            ("しちじゅう", Said::Alone),
            ("ななじゅっ", GEMINATED),
            ("ななじっ", GEMINATED),
        ],
        8 => &[
            ("はちじゅう", Said::Alone),
            ("はちじゅっ", GEMINATED),
            ("はちじっ", GEMINATED),
        ],
        9 => &[
            ("きゅうじゅう", Said::Alone),
            ("きゅうじゅっ", GEMINATED),
            ("きゅうじっ", GEMINATED),
        ],
        _ => &[],
    }
}

fn hundreds(d: u64) -> &'static [(&'static str, Said)] {
    match d {
        1 => &[("ひゃく", Said::Alone), ("ひゃっ", GEMINATED)],
        2 => &[("にひゃく", Said::Alone)],
        3 => &[("さんびゃく", Said::Alone)],
        4 => &[("よんひゃく", Said::Alone)],
        5 => &[("ごひゃく", Said::Alone)],
        6 => &[("ろっぴゃく", Said::Alone)],
        7 => &[("ななひゃく", Said::Alone)],
        8 => &[("はっぴゃく", Said::Alone)],
        9 => &[("きゅうひゃく", Said::Alone)],
        _ => &[],
    }
}

fn thousands(d: u64) -> &'static [(&'static str, Said)] {
    match d {
        1 => &[("せん", Said::Alone), ("いっせん", Said::Alone)],
        2 => &[("にせん", Said::Alone)],
        3 => &[("さんぜん", Said::Alone)],
        4 => &[("よんせん", Said::Alone)],
        5 => &[("ごせん", Said::Alone)],
        6 => &[("ろくせん", Said::Alone)],
        7 => &[("ななせん", Said::Alone)],
        8 => &[("はっせん", Said::Alone)],
        9 => &[("きゅうせん", Said::Alone)],
        _ => &[],
    }
}

/// The native readings, only said before a counter (ひとつ, ふたり, みっか,
/// ついたち, はつか, はたち).
fn native(n: u64) -> &'static [(&'static str, Said)] {
    match n {
        1 => &[("ひと", NATIVE), ("つい", NATIVE)],
        2 => &[("ふた", NATIVE), ("ふつ", NATIVE)],
        3 => &[("み", NATIVE), ("みっ", GEMINATED)],
        5 => &[("いつ", NATIVE)],
        6 => &[("む", NATIVE), ("むい", NATIVE), ("むっ", GEMINATED)],
        7 => &[("なの", NATIVE)],
        8 => &[("や", NATIVE), ("よう", NATIVE), ("やっ", GEMINATED)],
        9 => &[("ここの", NATIVE)],
        10 => &[("とお", NATIVE), ("と", NATIVE)],
        20 => &[("はつ", NATIVE), ("はた", NATIVE)],
        _ => &[],
    }
}

/// A digit read alone, as in a phone number (ぜろいちにぜろ).
fn digit_alone(d: u32) -> &'static [&'static str] {
    match d {
        0 => &["ぜろ", "れい", "まる"],
        1 => &["いち"],
        2 => &["に", "にい"],
        3 => &["さん"],
        4 => &["よん"],
        5 => &["ご", "ごお"],
        6 => &["ろく"],
        7 => &["なな"],
        8 => &["はち"],
        9 => &["きゅう"],
        _ => &[],
    }
}

/// Where a reading of the number `digits` (ASCII digits) that starts at
/// `reading[at]` can end, and how it is said there: the number as a whole
/// (にせんご, さんびゃく, いっかい's いっ, ひとつ's ひと), or each digit
/// alone. Empty when no reading of it starts there.
pub(crate) fn reading_ends(digits: &str, reading: &[char], at: usize) -> Vec<(usize, Said)> {
    let starts = |j: usize, text: &str| {
        let mut k = j;
        for c in text.chars() {
            if reading.get(k) != Some(&c) {
                return None;
            }
            k += 1;
        }
        Some(k)
    };
    let mut out: Vec<(usize, Said)> = Vec::new();
    let leading_zero = digits.len() > 1 && digits.starts_with('0');
    if let (false, Ok(n)) = (leading_zero || digits.len() > 16, digits.parse::<u64>()) {
        if n == 0 {
            for t in digit_alone(0) {
                out.extend(starts(at, t).map(|k| (k, Said::Alone)));
            }
        }
        // the parts of the number, the 兆, 億 and 万 groups and then the
        // ones, each read in turn: only its last read before a counter
        let mut parts: Vec<Vec<(&'static str, Said)>> = Vec::new();
        for (unit, name) in [
            (1_0000_0000_0000u64, "ちょう"),
            (1_0000_0000, "おく"),
            (1_0000, "まん"),
            (1, ""),
        ] {
            let g = n / unit % 1_0000;
            if g == 0 {
                continue;
            }
            for (v, table) in [
                (
                    g / 1000,
                    thousands as fn(u64) -> &'static [(&'static str, Said)],
                ),
                (g / 100 % 10, hundreds),
                (g / 10 % 10, tens),
                (g % 10, ones),
            ] {
                if v > 0 {
                    parts.push(table(v).to_vec());
                }
            }
            if !name.is_empty() {
                parts.push(vec![(name, Said::Alone)]);
            }
        }
        let mut at_ends: Vec<(usize, Said)> = vec![(at, Said::Alone)];
        for (i, options) in parts.iter().enumerate() {
            let last = i + 1 == parts.len();
            // いっちょう, はっちょう: a geminated digit before 兆
            let before_cho = parts.get(i + 1).is_some_and(|p| p[0].0 == "ちょう");
            let mut next = Vec::new();
            for &(j, _) in &at_ends {
                for &(text, said) in options {
                    if said != Said::Alone && !last && !(before_cho && said == GEMINATED) {
                        continue;
                    }
                    if let Some(k) = starts(j, text) {
                        if !next.contains(&(k, said)) {
                            next.push((k, said));
                        }
                    }
                }
            }
            at_ends = next;
            if at_ends.is_empty() {
                break;
            }
        }
        if !parts.is_empty() {
            out.extend(at_ends);
        }
        for &(text, said) in native(n) {
            out.extend(starts(at, text).map(|k| (k, said)));
        }
    }
    if digits.len() > 1 {
        let mut at_ends = vec![at];
        for d in digits.chars().filter_map(|c| c.to_digit(10)) {
            let mut next = Vec::new();
            for &j in &at_ends {
                for t in digit_alone(d) {
                    if let Some(k) = starts(j, t) {
                        if !next.contains(&k) {
                            next.push(k);
                        }
                    }
                }
            }
            at_ends = next;
        }
        out.extend(at_ends.into_iter().map(|k| (k, Said::Alone)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has(reading: &str, c: &str) -> bool {
        candidates(reading).iter().any(|x| x == c)
    }

    #[test]
    fn digits_get_their_forms() {
        for c in [
            "1,234",
            "１２３４",
            "千二百三十四",
            "一二三四",
            "壱阡弐百参拾四",
            "12:34",
            "12時34分",
        ] {
            assert!(has("1234", c), "{c}: {:?}", candidates("1234"));
        }
        assert!(has("123", "1:23") && has("123", "1月23日") && has("123", "1/23"));
        assert!(has("12345", "1万2345") && has("12345", "一万二千三百四十五"));
        assert!(has("7", "⑦") && has("7", "Ⅶ") && has("7", "七"));
        assert!(has("100000000", "一億") && has("100000000", "1億"));
        // no date for 1399, no time for 1299 (99 minutes)
        assert!(!has("1399", "13月99日") && !has("1299", "12時99分"));
        // a leading zero keeps the digits
        assert!(has("0120", "〇一二〇") && !has("0120", "百二十"));
    }

    #[test]
    fn kana_numbers() {
        assert_eq!(kana_number("じゅうに"), Some(12));
        assert_eq!(kana_number("にせんにじゅうご"), Some(2025));
        assert_eq!(kana_number("さんびゃくろくじゅうご"), Some(365));
        assert_eq!(kana_number("はっせんまん"), Some(8000_0000));
        assert_eq!(kana_number("さんおくごせんまん"), Some(3_5000_0000));
        assert_eq!(kana_number("いっせん"), Some(1000));
        // single digit words and units alone are left alone
        assert_eq!(kana_number("に"), None);
        assert_eq!(kana_number("せん"), None);
        assert_eq!(kana_number("おく"), None);
        assert_eq!(kana_number("かさ"), None);
        assert!(has("じゅうに", "12") && has("じゅうに", "十二"));
    }

    #[test]
    fn years_and_eras() {
        assert_eq!(candidates("2025ねん"), ["令和7年"]);
        assert_eq!(candidates("2019ねん"), ["令和元年", "平成31年"]);
        assert_eq!(candidates("1989ねん"), ["平成元年", "昭和64年"]);
        assert_eq!(candidates("れいわ7ねん"), ["2025年", "令和7年"]);
        assert_eq!(candidates("へいせいがんねん"), ["1989年", "平成元年"]);
        assert_eq!(candidates("しょうわろくじゅうよねん")[0], "1989年");
    }

    #[test]
    fn calculations_and_code_points() {
        assert_eq!(candidates("1＋2＝"), ["3", "1+2=3"]);
        assert_eq!(candidates("(1+2)*3="), ["9", "(1+2)*3=9"]);
        assert_eq!(candidates("10／4＝")[0], "2.5");
        assert_eq!(candidates("2^10=")[0], "1024");
        assert_eq!(candidates("5ー3＝")[0], "2");
        assert!(candidates("1/0=").is_empty());
        assert_eq!(candidates("う＋3042"), ["あ"]);
        assert_eq!(candidates("U+1F600"), ["😀"]);
        let out = insert("1＋2＝", vec!["1＋2＝".to_string()], candidates("1＋2＝"));
        assert_eq!(out[0], "3");
    }
}
