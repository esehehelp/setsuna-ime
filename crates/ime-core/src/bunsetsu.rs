//! Segments of a conversion from the engine's analysis of it (`--analyzer`:
//! its words, with their readings and parts of speech), where the char
//! alignment alone (`segments`) cuts compound verbs (申し|出る, 引き|続き)
//! and leaves runs uncut (パリオリンピックに、なでしこ).
//!
//! Each word is placed on the reading by the cheapest way the words read as
//! it: a word by its own reading, by one of its other readings (what the
//! analysis says the path costs more with it), or as written when it holds
//! no kanji; a word whose ASCII digits lead (16年 じゅうろくねん) by the
//! digits and a tail of its reading, one whose digits trail by a head of
//! its reading and the digits; a reading that differs only by voicing
//! (ほんかわ for ほんがわ) costs [`VOICE`] more. Where they do, each char's
//! reading start is kept within its word's (the engine can emit a kanji a
//! char off: ま|ほうの for 魔法の), else the engine's are kept.
//!
//! A segment starts at a word that stands on its own (a noun, verb,
//! adjective, adverb, ...) unless it leans on the word before, after
//! KWDLC's bunsetsu: a 非自立可能 word after て / で, a noun or a verb (い
//! of ている, し of 保管し, 始め of 働き始め), a noun after a noun or a
//! prefix, ある after で (である), ともに after と, a suffix, a particle,
//! an auxiliary, punctuation; a numeral after a numeral starts one (2000年|
//! 3月|24日), so does an opening bracket. Over KWDLC's dev sentences (the
//! rules picked there; the engine's char positions where its top is the
//! sentence) the boundaries' F1 is 94.3, the char alignment's 85.4; over its
//! test sentences 94.4 against 83.4.

use std::collections::HashMap;

use crate::engine_client::Word;
use crate::segments::Segment;

/// Cost of a reading that differs from a word's only by voicing.
pub const VOICE: i32 = 1000;

fn hira(s: &str) -> Vec<char> {
    use unicode_normalization::UnicodeNormalization;
    s.nfkc().map(crate::reading_align::to_hira).collect()
}

const VOICED: [(char, char); 26] = [
    ('が', 'か'),
    ('ぎ', 'き'),
    ('ぐ', 'く'),
    ('げ', 'け'),
    ('ご', 'こ'),
    ('ざ', 'さ'),
    ('じ', 'し'),
    ('ず', 'す'),
    ('ぜ', 'せ'),
    ('ぞ', 'そ'),
    ('だ', 'た'),
    ('ぢ', 'ち'),
    ('づ', 'つ'),
    ('で', 'て'),
    ('ど', 'と'),
    ('ば', 'は'),
    ('び', 'ひ'),
    ('ぶ', 'ふ'),
    ('べ', 'へ'),
    ('ぼ', 'ほ'),
    ('ぱ', 'は'),
    ('ぴ', 'ひ'),
    ('ぷ', 'ふ'),
    ('ぺ', 'へ'),
    ('ぽ', 'ほ'),
    ('ゔ', 'う'),
];

fn unvoiced(c: char) -> char {
    VOICED.iter().find(|(v, _)| *v == c).map_or(c, |&(_, u)| u)
}

fn is_kanji(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{9fff}' | '々' | '〆' | 'ヵ' | 'ヶ')
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// What `w` can read as: (chars, cost).
fn options(w: &Word) -> Vec<(Vec<char>, i32)> {
    let s = hira(&w.surface);
    let mut readings: Vec<(Vec<char>, i32)> = Vec::new();
    if let Some(r) = &w.reading {
        readings.push((hira(r), 0));
    }
    readings.extend(w.alts.iter().map(|(r, d)| (hira(r), *d)));
    let mut out = readings.clone();
    if !s.iter().any(|&c| is_kanji(c)) {
        out.push((s.clone(), 0));
    }
    let lead = s.iter().take_while(|&&c| is_digit(c)).count();
    let trail = s.iter().rev().take_while(|&&c| is_digit(c)).count();
    for (r, d) in &readings {
        if lead > 0 && lead < s.len() {
            // 16年 じゅうろくねん: 16 and a tail of the reading
            for k in 1..r.len() {
                out.push((s[..lead].iter().chain(&r[k..]).copied().collect(), *d));
            }
        } else if trail > 0 && trail < s.len() {
            for k in 1..r.len() {
                out.push((
                    r[..k]
                        .iter()
                        .chain(&s[s.len() - trail..])
                        .copied()
                        .collect(),
                    *d,
                ));
            }
        }
    }
    out.retain(|(o, _)| !o.is_empty());
    out
}

/// The reading char each word starts at, by the cheapest way the words
/// read as `reading`; `None` when no way does.
pub fn word_starts(words: &[Word], reading: &str) -> Option<Vec<usize>> {
    let r = hira(reading);
    let ru: Vec<char> = r.iter().map(|&c| unvoiced(c)).collect();
    // reading position -> (cost, each word's start so far)
    let mut best: HashMap<usize, (i64, Vec<usize>)> = HashMap::from([(0, (0, Vec::new()))]);
    for w in words {
        let opts = options(w);
        let mut next: HashMap<usize, (i64, Vec<usize>)> = HashMap::new();
        for (&p, (c, path)) in &best {
            for (o, d) in &opts {
                let q = p + o.len();
                if q > r.len() {
                    continue;
                }
                let e = if r[p..q] == o[..] {
                    *d
                } else if ru[p..q].iter().copied().eq(o.iter().map(|&c| unvoiced(c))) {
                    d + VOICE
                } else {
                    continue;
                };
                let cost = c + i64::from(e);
                if next.get(&q).is_none_or(|(b, _)| cost < *b) {
                    let mut path = path.clone();
                    path.push(p);
                    next.insert(q, (cost, path));
                }
            }
        }
        if next.is_empty() {
            return None;
        }
        best = next;
    }
    best.remove(&r.len()).map(|(_, path)| path)
}

const OWN: [&str; 11] = [
    "名詞",
    "代名詞",
    "動詞",
    "形容詞",
    "形状詞",
    "副詞",
    "連体詞",
    "接続詞",
    "感動詞",
    "接頭辞",
    "記号",
];
const NOUNISH: [&str; 4] = ["名詞", "代名詞", "接頭辞", "記号"];
const BREAK_AFTER: [&str; 4] = ["句点", "読点", "括弧閉", "空白"];
const ARU: [&str; 6] = ["あっ", "ある", "あり", "あれ", "あら", "あろ"];

/// Whether a segment starts at `w`, after `prev`.
fn starts_segment(prev: &Word, w: &Word) -> bool {
    let (pos, pos2, s) = (w.pos.as_str(), w.pos2.as_str(), w.surface.as_str());
    let noun = |p: &str| NOUNISH.contains(&p);
    if pos == "補助記号" {
        return pos2 == "括弧開";
    }
    if prev.pos == "補助記号" {
        // after 、。」 and spaces unless a function word follows; none after
        // ・ - / and an opening bracket, inside 1,310 and Q.E.D. and between
        // latin words
        let latin =
            prev.surface.is_ascii() && s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric());
        return BREAK_AFTER.contains(&prev.pos2.as_str()) && OWN.contains(&pos) && !latin;
    }
    if !OWN.contains(&pos) {
        return false;
    }
    if pos2 == "非自立可能" && (prev.pos2 == "接続助詞" || noun(&prev.pos) || prev.pos == "動詞")
    {
        return false;
    }
    if pos == "動詞"
        && prev.surface == "で"
        && matches!(prev.pos.as_str(), "助詞" | "助動詞")
        && ARU.contains(&s)
    {
        return false;
    }
    if prev.surface == "と" && matches!(s, "ともに" | "共に") {
        return false;
    }
    if pos == "接頭辞" {
        return prev.pos != "接頭辞";
    }
    if pos2 == "数詞" && prev.pos2 == "数詞" {
        return true;
    }
    !(noun(pos) && noun(&prev.pos)) && prev.pos != "接頭辞"
}

/// The words each segment after the first starts at.
fn segment_starts(words: &[Word]) -> Vec<usize> {
    (1..words.len())
        .filter(|&k| starts_segment(&words[k - 1], &words[k]))
        .collect()
}

/// `text` (read `reading`, its chars' reading starts `starts` as
/// `segments::char_starts` gives them) cut into segments by its analysis
/// `words`, with each char's reading start kept within its word's where the
/// words read as the reading. `None` when the words are not the text's.
pub fn cut(
    text: &str,
    reading: &str,
    words: &[Word],
    starts: &[usize],
) -> Option<(Vec<usize>, Vec<Segment>)> {
    let n = text.chars().count();
    let reading_len = reading.chars().count();
    if words.is_empty() || starts.len() != n || reading_len == 0 {
        return None;
    }
    // the words cover the text in order
    let covered = words.first()?.start == 0
        && words.last()?.end == n
        && words
            .windows(2)
            .all(|p| p[0].end == p[1].start && p[0].start < p[0].end);
    if !covered {
        return None;
    }
    let mut fixed = starts.to_vec();
    if let Some(ws) = word_starts(words, reading) {
        for (k, w) in words.iter().enumerate() {
            let (a, b) = (ws[k], ws.get(k + 1).copied().unwrap_or(reading_len));
            let last = b.max(a + 1) - 1;
            for (i, f) in fixed.iter_mut().enumerate().take(w.end).skip(w.start) {
                *f = if i == w.start { a } else { (*f).clamp(a, last) };
            }
        }
        for i in 1..n {
            fixed[i] = fixed[i].max(fixed[i - 1]);
        }
    }
    let mut out: Vec<Segment> = Vec::new();
    let (mut text_start, mut read_start) = (0, 0);
    for k in segment_starts(words) {
        let t = words[k].start;
        let r = fixed[t];
        // each segment reads at least one char
        if r <= read_start || r >= reading_len {
            continue;
        }
        out.push(Segment {
            reading: (read_start, r),
            text: (text_start, t),
        });
        (text_start, read_start) = (t, r);
    }
    out.push(Segment {
        reading: (read_start, reading_len),
        text: (text_start, n),
    });
    Some((fixed, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words of `spec`: "surface/reading/pos/pos2" each, a space apart
    /// (reading `-` for none).
    fn words(spec: &str) -> Vec<Word> {
        let mut at = 0;
        spec.split(' ')
            .map(|w| {
                let f: Vec<&str> = w.split('/').collect();
                let n = f[0].chars().count();
                let word = Word {
                    surface: f[0].to_string(),
                    start: at,
                    end: at + n,
                    reading: (f[1] != "-").then(|| f[1].to_string()),
                    pos: f[2].to_string(),
                    pos2: f.get(3).unwrap_or(&"*").to_string(),
                    alts: Vec::new(),
                    cost: 0,
                };
                at += n;
                word
            })
            .collect()
    }

    fn texts(text: &str, reading: &str, segs: &[Segment]) -> Vec<(String, String)> {
        let t: Vec<char> = text.chars().collect();
        let r: Vec<char> = reading.chars().collect();
        segs.iter()
            .map(|s| {
                (
                    t[s.text.0..s.text.1].iter().collect(),
                    r[s.reading.0..s.reading.1].iter().collect(),
                )
            })
            .collect()
    }

    fn cut_of(text: &str, reading: &str, spec: &str, starts: &[usize]) -> Vec<(String, String)> {
        let (_, segs) = cut(text, reading, &words(spec), starts).unwrap();
        texts(text, reading, &segs)
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn compound_verbs_stay_whole() {
        // the char alignment: 前までに|申し|出る
        let starts = [0, 1, 2, 3, 4, 5, 7, 8];
        assert_eq!(
            cut_of(
                "前までに申し出る",
                "まえまでにもうしでる",
                "前/まえ/名詞/普通名詞 まで/まで/助詞/副助詞 に/に/助詞/格助詞 申し出る/もうしでる/動詞/一般",
                &starts
            ),
            pairs(&[("前までに", "まえまでに"), ("申し出る", "もうしでる")])
        );
    }

    #[test]
    fn auxiliaries_lean_and_content_words_start() {
        let spec = "工場/こうじょう/名詞/普通名詞 を/を/助詞/格助詞 置い/おい/動詞/非自立可能 て/て/助詞/接続助詞 \
                    いる/いる/動詞/非自立可能";
        let text = "工場を置いている";
        let reading = "こうじょうをおいている";
        let starts = [0, 2, 5, 6, 7, 8, 9, 10];
        assert_eq!(
            cut_of(text, reading, spec, &starts),
            pairs(&[("工場を", "こうじょうを"), ("置いている", "おいている")])
        );
        // する of a verbal noun, numerals each, ある of である
        let spec = "保管/ほかん/名詞/普通名詞 し/し/動詞/非自立可能 た/た/助動詞 2000年/にせんねん/名詞/数詞 \
                    3月/さんがつ/名詞/数詞 で/で/助動詞 ある/ある/動詞/非自立可能";
        let text = "保管した2000年3月である";
        let reading = "ほかんした2000ねん3がつである";
        let starts = (0..text.chars().count()).collect::<Vec<_>>();
        assert_eq!(
            cut_of(text, reading, spec, &starts),
            pairs(&[
                ("保管した", "ほかんした"),
                ("2000年", "2000ねん"),
                ("3月である", "3がつである")
            ])
        );
    }

    #[test]
    fn punctuation_breaks_and_brackets_start() {
        let spec = "なり/なり/動詞/非自立可能 、/、/補助記号/読点 また/また/接続詞 「/「/補助記号/括弧開 \
                    歌/うた/名詞/普通名詞 」/」/補助記号/括弧閉 1/1/名詞/数詞 ,/,/補助記号/読点 310/310/名詞/数詞";
        let text = "なり、また「歌」1,310";
        let reading = "なり、また「うた」1,310";
        let starts = (0..text.chars().count()).collect::<Vec<_>>();
        assert_eq!(
            cut_of(text, reading, spec, &starts),
            pairs(&[
                ("なり、", "なり、"),
                ("また", "また"),
                ("「歌」", "「うた」"),
                ("1,310", "1,310")
            ])
        );
    }

    #[test]
    fn the_words_correct_the_engines_char_starts() {
        // the engine put 法 at the の: ま|ほうの
        let spec = "魔法/まほう/名詞/普通名詞 の/の/助詞/格助詞 呪文/じゅもん/名詞/普通名詞";
        let (fixed, segs) = cut(
            "魔法の呪文",
            "まほうのじゅもん",
            &words(spec),
            &[0, 3, 3, 4, 6],
        )
        .unwrap();
        // 法 within 魔法's まほう
        assert_eq!(fixed, [0, 2, 3, 4, 6]);
        assert_eq!(
            texts("魔法の呪文", "まほうのじゅもん", &segs),
            pairs(&[("魔法の", "まほうの"), ("呪文", "じゅもん")])
        );
    }

    #[test]
    fn readings_by_digits_alternatives_and_voicing() {
        let mut w = words("16年/じゅうろくねん/名詞/数詞 本川/もとかわ/名詞/固有名詞");
        assert_eq!(word_starts(&w, "16ねんほんかわ"), None);
        w[1].alts = vec![("ほんかわ".into(), 0)];
        assert_eq!(word_starts(&w, "16ねんほんかわ"), Some(vec![0, 4]));
        // ほんがわ by voicing
        assert_eq!(word_starts(&w, "16ねんほんがわ"), Some(vec![0, 4]));
        assert_eq!(word_starts(&w, "16ねんかわ"), None);
    }

    #[test]
    fn words_that_are_not_the_texts_cut_nothing() {
        let w = words("今日/きょう/名詞/普通名詞");
        assert!(cut("今日は", "きょうは", &w, &[0, 0, 3]).is_none());
        assert!(cut("今日", "きょう", &[], &[0, 0]).is_none());
    }
}
