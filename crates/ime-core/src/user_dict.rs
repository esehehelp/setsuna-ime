//! The user dictionary: words the user registered, in Mozc's / Google
//! Japanese Input's text format (`reading<TAB>word<TAB>品詞<TAB>comment`),
//! `user_dictionary.txt` next to the config, so that an export of either
//! imports as it is. The model has no dictionary to add them to, so they
//! come in around it:
//!
//! - a word whose reading is the whole reading converted (or the segment
//!   edited) is a candidate of its own, first;
//! - a word whose reading is part of a longer one replaces the chars the
//!   model wrote for that part (aligned to the reading by
//!   `segments::char_starts`), in a copy of each of the first candidates;
//!   the copies go before them (a sentence with the word, as Mozkey keeps
//!   the user's words in what its neural model writes);
//! - the part of speech 短縮よみ / サジェストのみ (a shortcut: きょう ->
//!   an address) matches the whole reading only, and 抑制単語 hides a word
//!   (for its reading, or for every reading when it has none).
//!
//! Imports read the other IMEs' exports (`import`): Mozc / Google TSV,
//! MS-IME's text, ATOK's and Gboard's.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// What an entry does, from its part of speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A word: the whole reading or a part of a longer one.
    Word,
    /// 短縮よみ / サジェストのみ: the whole reading only.
    Shortcut,
    /// 抑制単語: never shown.
    Suppress,
}

impl Kind {
    pub fn of(pos: &str) -> Self {
        match pos.trim() {
            "短縮よみ" | "短縮読み" | "サジェストのみ" => Kind::Shortcut,
            "抑制単語" => Kind::Suppress,
            _ => Kind::Word,
        }
    }
}

/// One line of the dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub reading: String,
    pub word: String,
    pub pos: String,
    pub comment: String,
}

impl Entry {
    pub fn kind(&self) -> Kind {
        Kind::of(&self.pos)
    }

    fn line(&self) -> String {
        let clean = |s: &str| s.replace(['\t', '\n', '\r'], " ");
        let mut line = format!(
            "{}\t{}\t{}",
            clean(&self.reading),
            clean(&self.word),
            clean(&self.pos)
        );
        if !self.comment.is_empty() {
            line.push('\t');
            line.push_str(&clean(&self.comment));
        }
        line
    }
}

/// The default part of speech of an entry that names none.
pub const DEFAULT_POS: &str = "名詞";

/// A reading as the dictionary keys it: hiragana (katakana folded, romaji
/// typed as an IME would read it), or `None` when it is no reading.
pub fn normalize_reading(reading: &str) -> Option<String> {
    let reading = reading.trim();
    if reading.is_empty() {
        return None;
    }
    let kana = if reading
        .chars()
        .all(|c| c.is_ascii_alphabetic() || c == '-' || c == '\'')
    {
        let mut c = crate::composing_text::ComposingText::new();
        for ch in reading.chars() {
            c.input_char(ch.to_ascii_lowercase());
        }
        let k = c.kana_reading();
        if k.chars().any(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        k
    } else {
        reading
            .chars()
            .map(|c| match c {
                'ァ'..='ヶ' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
                c => c,
            })
            .collect()
    };
    kana.chars()
        .all(|c| ('ぁ'..='ゖ').contains(&c) || "ーゝゞ".contains(c) || c.is_ascii_digit())
        .then_some(kana)
}

/// The entries of a dictionary text (any format `import` reads), with the
/// lines it skipped and why.
pub fn parse(text: &str) -> (Vec<Entry>, Vec<String>) {
    let text = text.trim_start_matches('\u{FEFF}');
    let atok = text.starts_with("!!ATOK");
    let gboard = text.starts_with("# Gboard Dictionary");
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').collect()
        } else if line.starts_with('"') {
            // Kotoeri CSV: "reading","word","pos"
            line.split("\",\"").map(|c| c.trim_matches('"')).collect()
        } else {
            skipped.push(format!("{line}（区切りがありません）"));
            continue;
        };
        if cols.len() < 2 {
            skipped.push(format!("{line}（列が足りません）"));
            continue;
        }
        let pos = match cols.get(2).map(|p| p.trim()) {
            // Gboard's third column is the locale
            _ if gboard => DEFAULT_POS.to_string(),
            // ATOK marks some parts of speech with * or $
            Some(p) if atok => p.trim_end_matches(['*', '$']).to_string(),
            Some(p) if !p.is_empty() => p.to_string(),
            _ => DEFAULT_POS.to_string(),
        };
        // a suppression word without a reading is hidden for every reading
        let reading = if cols[0].trim().is_empty() && Kind::of(&pos) == Kind::Suppress {
            String::new()
        } else if let Some(r) = normalize_reading(cols[0]) {
            r
        } else {
            skipped.push(format!("{line}（読みがひらがなではありません）"));
            continue;
        };
        let word = cols[1].trim().to_string();
        if word.is_empty() {
            skipped.push(format!("{line}（語がありません）"));
            continue;
        }
        let comment = cols
            .get(3)
            .map(|c| c.trim().to_string())
            .unwrap_or_default();
        if seen.insert((reading.clone(), word.clone(), pos.clone())) {
            entries.push(Entry {
                reading,
                word,
                pos,
                comment,
            });
        }
    }
    (entries, skipped)
}

/// The dictionary text of `entries`, in Mozc / Google TSV.
pub fn to_text(entries: &[Entry]) -> String {
    let mut out =
        String::from("# Setsuna IME user dictionary: reading\tword\tpart of speech\tcomment\n");
    for e in entries {
        out.push_str(&e.line());
        out.push('\n');
    }
    out
}

/// The dictionary as conversion uses it.
#[derive(Default)]
pub struct UserDict {
    /// Reading -> (word, kind), in file order.
    by_reading: HashMap<String, Vec<(String, Kind)>>,
    /// The words (of `Kind::Word`) that may sit inside a longer reading,
    /// longest reading first.
    inner: Vec<(String, String)>,
    /// 抑制単語 for a reading, and for any reading.
    suppressed: HashSet<(String, String)>,
    suppressed_anywhere: HashSet<String>,
}

/// How many of the first candidates get a copy with a word inside them.
const INNER_CANDIDATES: usize = 3;

impl UserDict {
    pub fn new(entries: &[Entry]) -> Self {
        let mut d = Self::default();
        for e in entries {
            match e.kind() {
                Kind::Suppress if e.reading.is_empty() => {
                    d.suppressed_anywhere.insert(e.word.clone());
                }
                Kind::Suppress => {
                    d.suppressed.insert((e.reading.clone(), e.word.clone()));
                }
                kind => {
                    d.by_reading
                        .entry(e.reading.clone())
                        .or_default()
                        .push((e.word.clone(), kind));
                    // one char matches too much inside a reading
                    if kind == Kind::Word && e.reading.chars().count() >= 2 {
                        d.inner.push((e.reading.clone(), e.word.clone()));
                    }
                }
            }
        }
        d.inner
            .sort_by_key(|(r, _)| std::cmp::Reverse(r.chars().count()));
        d
    }

    pub fn is_empty(&self) -> bool {
        self.by_reading.is_empty()
            && self.suppressed.is_empty()
            && self.suppressed_anywhere.is_empty()
    }

    /// `candidates` for `reading` with the user's words: those of the whole
    /// reading first, then the first candidates with the words inside the
    /// reading put in, then the rest; the suppressed ones left out.
    pub fn apply(&self, reading: &str, candidates: Vec<String>) -> Vec<String> {
        if self.is_empty() {
            return candidates;
        }
        let mut out: Vec<String> = Vec::new();
        let push = |out: &mut Vec<String>, c: String| {
            if !out.contains(&c) {
                out.push(c);
            }
        };
        if let Some(words) = self.by_reading.get(reading) {
            for (w, _) in words {
                push(&mut out, w.clone());
            }
        }
        let spans = self.spans_in(reading);
        if !spans.is_empty() {
            for c in candidates.iter().take(INNER_CANDIDATES) {
                if let Some(with) = put_in(reading, c, &spans) {
                    push(&mut out, with);
                }
            }
        }
        for c in candidates {
            push(&mut out, c);
        }
        out.retain(|c| {
            !self.suppressed_anywhere.contains(c)
                && !self.suppressed.contains(&(reading.to_string(), c.clone()))
        });
        out
    }

    /// The words that sit inside `reading` (but are not all of it), as
    /// (start, end, word) in reading chars, not overlapping, the longest
    /// readings first and then from the left.
    fn spans_in(&self, reading: &str) -> Vec<(usize, usize, String)> {
        let chars: Vec<char> = reading.chars().collect();
        let mut taken = vec![false; chars.len()];
        let mut out = Vec::new();
        for (r, w) in &self.inner {
            let rc: Vec<char> = r.chars().collect();
            if rc.len() >= chars.len() {
                continue;
            }
            let mut s = 0;
            while s + rc.len() <= chars.len() {
                let e = s + rc.len();
                if chars[s..e] == rc[..] && !taken[s..e].iter().any(|&t| t) {
                    taken[s..e].iter_mut().for_each(|t| *t = true);
                    out.push((s, e, w.clone()));
                    s = e;
                } else {
                    s += 1;
                }
            }
        }
        out.sort_by_key(|(s, _, _)| *s);
        out
    }
}

/// `candidate` with each span's chars (those whose reading starts fall in
/// it, and only they) replaced by its word; `None` when no span lines up
/// with the candidate's chars or nothing changes.
fn put_in(reading: &str, candidate: &str, spans: &[(usize, usize, String)]) -> Option<String> {
    let starts = crate::segments::char_starts(reading, candidate, None)?;
    let chars: Vec<char> = candidate.chars().collect();
    let n = reading.chars().count();
    let mut out = String::new();
    let mut i = 0;
    let mut changed = false;
    for (s, e, word) in spans {
        // the chars from the one starting the span to the one starting at its end
        let Some(a) = (i..chars.len()).find(|&k| starts[k] == *s) else {
            continue;
        };
        let b = (a..chars.len())
            .find(|&k| starts[k] >= *e)
            .unwrap_or(chars.len());
        if (b < chars.len() && starts[b] != *e) || (b == chars.len() && *e != n) {
            continue;
        }
        if starts[a..b].iter().any(|&x| x < *s || x >= *e) {
            continue;
        }
        out.extend(&chars[i..a]);
        out.push_str(word);
        changed |= chars[a..b].iter().collect::<String>() != *word;
        i = b;
    }
    out.extend(&chars[i..]);
    changed.then_some(out)
}

/// Where the user dictionary is.
pub fn path() -> Option<PathBuf> {
    crate::config::Config::user_dictionary_path()
}

/// The entries of the file at `path` (none when there is none).
pub fn load(path: &Path) -> Vec<Entry> {
    std::fs::read_to_string(path)
        .map(|t| parse(&t).0)
        .unwrap_or_default()
}

/// Write `entries` to `path`.
pub fn save(path: &Path, entries: &[Entry]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, to_text(entries))
}

type Cached = Option<(PathBuf, Option<SystemTime>, Arc<UserDict>)>;
static SHARED: Mutex<Cached> = Mutex::new(None);

/// The user dictionary of the default config directory, read again when
/// the file changed (the settings GUI saves it while the IME runs).
pub fn shared() -> Option<Arc<UserDict>> {
    let path = path()?;
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    let mut cache = SHARED.lock().ok()?;
    if let Some((p, t, d)) = cache.as_ref() {
        if *p == path && *t == mtime {
            return Some(d.clone());
        }
    }
    let dict = Arc::new(UserDict::new(&load(&path)));
    *cache = Some((path, mtime, dict.clone()));
    Some(dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(text: &str) -> UserDict {
        UserDict::new(&parse(text).0)
    }

    fn list(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn formats_are_read() {
        let (e, skipped) =
            parse("# comment\nせつな\t刹那\t名詞\tIME の名前\nsetsuna\tSetsuna\t固有名詞\n");
        assert_eq!(e.len(), 2);
        assert_eq!(
            (e[1].reading.as_str(), e[1].word.as_str()),
            ("せつな", "Setsuna")
        );
        assert_eq!(e[0].comment, "IME の名前");
        assert!(skipped.is_empty());
        // MS-IME
        let (e, _) = parse("!Microsoft IME Dictionary Tool\nかなかな\t仮名々\t名詞\n");
        assert_eq!(e[0].word, "仮名々");
        // ATOK
        let (e, _) = parse("!!ATOK_TANGO_TEXT_HEADER_1\nあとっく\tATOK\t固有一般*\n");
        assert_eq!(e[0].pos, "固有一般");
        // Gboard: the third column is the locale
        let (e, _) = parse("# Gboard Dictionary version:1\nじーぼーど\tGboard\tja-JP\n");
        assert_eq!(e[0].pos, DEFAULT_POS);
        // a reading that is no reading
        let (e, skipped) = parse("漢字\t語\t名詞\n");
        assert!(e.is_empty() && skipped.len() == 1);
    }

    #[test]
    fn a_whole_reading_puts_its_word_first() {
        let d = dict("せつな\t刹那\t名詞\n");
        assert_eq!(
            d.apply("せつな", list(&["切な", "せつな"])),
            list(&["刹那", "切な", "せつな"])
        );
        assert_eq!(d.apply("べつ", list(&["別"])), list(&["別"]));
    }

    #[test]
    fn a_word_inside_a_reading_replaces_its_part() {
        let d = dict("せつな\tSetsuna\t固有名詞\n");
        let out = d.apply("せつなをつかう", list(&["切なを使う", "刹那を使う"]));
        assert_eq!(out[0], "Setsunaを使う");
        assert_eq!(out[1..], list(&["切なを使う", "刹那を使う"])[..]);
        // a word of one char of reading stays out of longer readings
        let d = dict("き\t樹\t名詞\n");
        assert_eq!(
            d.apply("きをきる", list(&["木を切る"])),
            list(&["木を切る"])
        );
    }

    #[test]
    fn shortcuts_and_suppression() {
        let d =
            dict("じゅうしょ\t東京都千代田区1-1\t短縮よみ\nかみ\t加味\t抑制単語\n\t糞\t抑制単語\n");
        assert_eq!(
            d.apply("じゅうしょ", list(&["住所"]))[0],
            "東京都千代田区1-1"
        );
        // a shortcut is never put inside a longer reading
        assert_eq!(
            d.apply("じゅうしょを", list(&["住所を"])),
            list(&["住所を"])
        );
        assert_eq!(
            d.apply("かみ", list(&["紙", "加味", "神"])),
            list(&["紙", "神"])
        );
        assert_eq!(d.apply("くそ", list(&["糞", "くそ"])), list(&["くそ"]));
    }

    #[test]
    fn readings_are_normalized() {
        assert_eq!(normalize_reading("セツナ").as_deref(), Some("せつな"));
        assert_eq!(normalize_reading("setsuna").as_deref(), Some("せつな"));
        assert_eq!(normalize_reading("漢字"), None);
    }
}
