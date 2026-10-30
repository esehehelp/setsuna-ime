//! Learning of the user's candidate choices, and reranking by them.
//!
//! When the user commits a candidate other than the first one shown, the
//! choice is recorded as (reading, chosen, rejected), `rejected` being the
//! candidate that was on top. A later candidate list is reranked by every
//! recorded choice whose reading is similar enough to the current one:
//!
//! ```text
//! bonus(c) = Σ sim(R, r_i) · w_i · (cos(c, chosen_i) − cos(c, rejected_i)) / span_i
//! score(c) = −rank(c) / n + strength · bonus(c)
//! ```
//!
//! `sim` and `cos` are cosine similarities of character-bigram count
//! vectors, `span_i = 1 − cos(chosen_i, rejected_i)` scales the difference so
//! the chosen surface itself gets +1 and the rejected one −1, and
//! `w_i = 1 + ln(count_i)` grows with repeated choices. What the chosen and
//! rejected surfaces share cancels out, so on a similar reading only the part
//! the user chose between (貴社 / 記者) counts.
//!
//! The best-scoring candidate moves to the top when it outscores the current
//! top; every other candidate keeps the engine's order. (Sorting the whole
//! list by score let misconversions that merely share the chosen characters,
//! 意志をつ伝えたい / 医志を伝えたい, jump over the engine's real
//! alternatives.) With the default strength 1.0 a choice learned for the
//! same reading moves to the top.
//!
//! Each choice also yields word-level choices (`words`): the words the two
//! surfaces differ in, with their reading and up to two reading chars on
//! each side (きしゃ: 記者 -> 貴社, before にれ). A word choice applies wherever
//! its reading recurs next to the same context on one side, however
//! different the rest of the reading is, and adds ±w to candidates containing the chosen / rejected
//! word. A candidate list is reranked when a sentence-level or a word-level
//! choice applies.
//!
//! Choices are appended to `learning.jsonl` in the config directory, one
//! JSON object per line. Every process that loads the IME appends to the
//! same file and reads the lines others appended before reranking, so a
//! choice made in one application applies in the others. Deleting or
//! truncating the file (the settings GUI's "clear") resets the learning.
//!
//! The same choice made again is a line again, and one cancelled (chosen
//! back, Ctrl+Delete) stays in the file, so the file grows for good while
//! what it holds does not. Loading it compacts it (`compact`) when it has
//! more than `COMPACT_FLOOR` lines and more than twice the lines of what is
//! in effect: a first line `{"compacted": <ns>}`, each sentence choice in
//! effect once with its `count` (and `no_words`: its word choices are lines
//! of their own), and each word choice in effect as `{"word": {...},
//! "count": n}`; nothing cancelled or forgotten. It is written to a file of
//! its own and moved over the log, unless the log changed meanwhile (another
//! process appended: compacted another time). Another process sees that
//! the file is another one by its first bytes (or its length going down)
//! and reads it again from the start. A version from before compaction
//! reads a compacted file too: it skips the first line and the word lines,
//! and takes each sentence line as one choice with its word choices, so
//! going back to it loses the counts (each choice is as if made once).

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::LearningCfg;

mod words;

/// Sparse character-bigram vector: (bigram key, weight) sorted by key, with
/// unit L2 norm.
type Vector = Vec<(u64, f32)>;

const BOS: char = '\u{2}';
const EOS: char = '\u{3}';

fn bigrams(s: &str) -> Vector {
    let chars: Vec<char> = std::iter::once(BOS)
        .chain(s.chars())
        .chain(std::iter::once(EOS))
        .collect();
    let mut counts: HashMap<u64, f32> = HashMap::new();
    for w in chars.windows(2) {
        *counts
            .entry(((w[0] as u64) << 32) | w[1] as u64)
            .or_default() += 1.0;
    }
    let norm = counts.values().map(|v| v * v).sum::<f32>().sqrt();
    let mut v: Vector = counts.into_iter().map(|(k, c)| (k, c / norm)).collect();
    v.sort_unstable_by_key(|e| e.0);
    v
}

fn cosine(a: &Vector, b: &Vector) -> f32 {
    let (mut i, mut j, mut dot) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

/// One line of `learning.jsonl`: a choice, or (`forget`) that the choices
/// of `chosen` for `reading`, and the word choices inside it, are
/// forgotten (Ctrl+Delete on the candidate, as Mozc deletes it from its
/// history); in a compacted file also its first line (`compacted`) and the
/// word choices (`word`).
#[derive(Debug, Default, Serialize, Deserialize)]
struct Record {
    // without them, a line versions before compaction skip
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reading: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    chosen: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    rejected: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    forget: bool,
    /// Times the choice was made (1 when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    count: Option<u32>,
    /// A sentence choice whose word choices are lines of their own.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    no_words: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    word: Option<words::Word>,
    /// The first line of a compacted file: when it was compacted (ns since
    /// the Unix epoch), which tells it from the file it replaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compacted: Option<u128>,
}

/// A file of fewer lines is not compacted.
const COMPACT_FLOOR: usize = 500;
/// The first bytes of the file kept to tell a compacted file from the one
/// read before (`sync`).
const HEAD: usize = 64;

struct Entry {
    /// Times this choice was made; 0 once the opposite choice cancelled it.
    count: u32,
    reading: Vector,
    chosen: Vector,
    rejected: Vector,
    /// `1 − cos(chosen, rejected)`, floored so it never divides by ~0.
    span: f32,
}

struct WordEntry {
    /// As `Entry::count`.
    count: u32,
    word: words::Word,
}

/// A learned choice, by index.
enum Learned {
    Sentence(usize),
    Word(usize),
}

pub struct Learning {
    path: PathBuf,
    strength: f32,
    threshold: f32,
    /// Whether word-level choices apply (`LearningCfg::words`).
    use_words: bool,
    /// Bytes of the file already applied.
    offset: u64,
    /// Its first bytes (up to `HEAD`) and lines, as read.
    head: Vec<u8>,
    lines: usize,
    entries: Vec<Entry>,
    by_key: HashMap<(String, String, String), usize>,
    /// Reading bigram -> entries whose reading has it.
    postings: HashMap<u64, Vec<usize>>,
    words: Vec<WordEntry>,
    word_key: HashMap<words::Word, usize>,
}

impl Learning {
    /// The learning of the default config directory, or `None` when learning
    /// is disabled or there is no config directory.
    pub fn open(cfg: &LearningCfg) -> Option<Self> {
        if !cfg.enabled {
            return None;
        }
        crate::config::Config::learning_path().map(|p| Self::at(p, cfg))
    }

    /// The learning stored at `path` (which need not exist yet), the file
    /// compacted when it is due (the module docs).
    pub fn at(path: PathBuf, cfg: &LearningCfg) -> Self {
        let mut l = Self::load(path, cfg);
        let live = l.entries.iter().filter(|e| e.count > 0).count()
            + l.words.iter().filter(|w| w.count > 0).count()
            + 1;
        if l.lines > COMPACT_FLOOR && l.lines > 2 * live {
            // a failure leaves the log as it was
            let _ = l.compact();
        }
        l
    }

    fn load(path: PathBuf, cfg: &LearningCfg) -> Self {
        let mut l = Self {
            path,
            strength: cfg.strength,
            threshold: cfg.similarity_threshold,
            use_words: cfg.words,
            offset: 0,
            head: Vec::new(),
            lines: 0,
            entries: Vec::new(),
            by_key: HashMap::new(),
            postings: HashMap::new(),
            words: Vec::new(),
            word_key: HashMap::new(),
        };
        l.sync();
        l
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of choices in effect.
    pub fn len(&mut self) -> usize {
        self.sync();
        self.entries.iter().filter(|e| e.count > 0).count()
    }

    pub fn is_empty(&mut self) -> bool {
        self.len() == 0
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.head.clear();
        self.lines = 0;
        self.entries.clear();
        self.by_key.clear();
        self.postings.clear();
        self.words.clear();
        self.word_key.clear();
    }

    /// Apply the lines appended to the file since the last sync. A file that
    /// shrank or disappeared was cleared, and one that starts otherwise than
    /// it did was compacted (by another process): start over.
    fn sync(&mut self) {
        let len = match std::fs::metadata(&self.path) {
            Ok(m) => m.len(),
            Err(_) => {
                self.reset();
                return;
            }
        };
        if len < self.offset {
            self.reset();
        }
        if len == self.offset {
            return;
        }
        let Ok(mut f) = File::open(&self.path) else {
            return;
        };
        if self.offset > 0 {
            let mut head = vec![0; self.head.len()];
            if f.read_exact(&mut head).is_err() || head != self.head {
                self.reset();
            }
        }
        let mut buf = Vec::new();
        let read = f
            .seek(SeekFrom::Start(self.offset))
            .and_then(|_| f.read_to_end(&mut buf));
        if read.is_err() {
            return;
        }
        // Only whole lines: another process may be mid-append.
        let Some(end) = buf.iter().rposition(|&b| b == b'\n') else {
            return;
        };
        if self.offset == 0 {
            self.head = buf[..(end + 1).min(HEAD)].to_vec();
        }
        for line in buf[..end].split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            self.lines += 1;
            if let Ok(rec) = serde_json::from_slice::<Record>(line) {
                self.apply(rec);
            }
        }
        self.offset += end as u64 + 1;
    }

    /// Write what is in effect as a compacted file (the module docs) and
    /// move it over the log; false when the log changed meanwhile (another
    /// process appended a line) and was left as it is.
    fn compact(&mut self) -> std::io::Result<bool> {
        let len = std::fs::metadata(&self.path)?.len();
        if len != self.offset {
            // a line being appended
            return Ok(false);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut out = Vec::new();
        let mut line = |r: &Record| -> std::io::Result<()> {
            serde_json::to_writer(&mut out, r)?;
            out.push(b'\n');
            Ok(())
        };
        line(&Record {
            compacted: Some(now),
            ..Record::default()
        })?;
        let mut keys: Vec<(&(String, String, String), usize)> =
            self.by_key.iter().map(|(k, &i)| (k, i)).collect();
        keys.sort_unstable_by_key(|&(_, i)| i);
        let count = |n: u32| (n > 1).then_some(n);
        for ((reading, chosen, rejected), i) in keys {
            let n = self.entries[i].count;
            if n > 0 {
                line(&Record {
                    reading: reading.clone(),
                    chosen: chosen.clone(),
                    rejected: rejected.clone(),
                    count: count(n),
                    no_words: true,
                    ..Record::default()
                })?;
            }
        }
        for w in self.words.iter().filter(|w| w.count > 0) {
            line(&Record {
                word: Some(w.word.clone()),
                count: count(w.count),
                ..Record::default()
            })?;
        }
        let name = self
            .path
            .file_name()
            .map_or("learning.jsonl".into(), |n| n.to_string_lossy());
        let tmp = self
            .path
            .with_file_name(format!("{name}.{}.{now}.tmp", std::process::id()));
        let moved = std::fs::write(&tmp, &out).and_then(|()| {
            // another process appended meanwhile: its line stays, compacted
            // another time
            if std::fs::metadata(&self.path)?.len() != len {
                return Ok(false);
            }
            std::fs::rename(&tmp, &self.path).map(|()| true)
        });
        if !matches!(moved, Ok(true)) {
            let _ = std::fs::remove_file(&tmp);
        }
        if moved? {
            self.reset();
            self.sync();
            return Ok(true);
        }
        Ok(false)
    }

    fn apply(&mut self, r: Record) {
        let n = r.count.unwrap_or(1);
        if r.compacted.is_some() || n == 0 {
            return;
        }
        if let Some(w) = r.word {
            self.apply_word(w, n);
            return;
        }
        if r.forget {
            for i in self.learned(&r.reading, &r.chosen) {
                match i {
                    Learned::Sentence(i) => self.entries[i].count = 0,
                    Learned::Word(i) => self.words[i].count = 0,
                }
            }
            return;
        }
        if r.reading.is_empty() || r.chosen.is_empty() || r.chosen == r.rejected {
            return;
        }
        if !r.no_words {
            for w in words::derive(&r.reading, &r.chosen, &r.rejected) {
                self.apply_word(w, n);
            }
        }
        self.apply_sentence(r, n);
    }

    /// Apply a word choice made `n` times.
    fn apply_word(&mut self, w: words::Word, n: u32) {
        let opposite = words::Word {
            chosen: w.rejected.clone(),
            rejected: w.chosen.clone(),
            ..w.clone()
        };
        if let Some(&i) = self.word_key.get(&opposite) {
            self.words[i].count = 0;
        }
        if let Some(&i) = self.word_key.get(&w) {
            self.words[i].count += n;
            return;
        }
        self.word_key.insert(w.clone(), self.words.len());
        self.words.push(WordEntry { count: n, word: w });
    }

    /// Apply a sentence choice made `n` times.
    fn apply_sentence(&mut self, r: Record, n: u32) {
        // Choosing back what was rejected cancels the earlier choice.
        let opposite = (r.reading.clone(), r.rejected.clone(), r.chosen.clone());
        if let Some(&i) = self.by_key.get(&opposite) {
            self.entries[i].count = 0;
        }
        let key = (r.reading, r.chosen, r.rejected);
        if let Some(&i) = self.by_key.get(&key) {
            self.entries[i].count += n;
            return;
        }
        let reading = bigrams(&key.0);
        let chosen = bigrams(&key.1);
        let rejected = bigrams(&key.2);
        let span = (1.0 - cosine(&chosen, &rejected)).max(0.05);
        let id = self.entries.len();
        for (k, _) in &reading {
            self.postings.entry(*k).or_default().push(id);
        }
        self.entries.push(Entry {
            count: n,
            reading,
            chosen,
            rejected,
            span,
        });
        self.by_key.insert(key, id);
    }

    /// The choices in effect that put `surface` up for `reading`: the
    /// sentence choices of it, and the word choices of a word inside it
    /// that apply to the reading (`words::applies`, as reranking takes
    /// them): a word learned elsewhere is no part of it (傘 learned before
    /// のし stays when 傘の下 is deleted for かさの).
    fn learned(&self, reading: &str, surface: &str) -> Vec<Learned> {
        let reading_chars: Vec<char> = reading.chars().collect();
        let mut out: Vec<Learned> = self
            .by_key
            .iter()
            .filter(|((r, c, _), &i)| r == reading && c == surface && self.entries[i].count > 0)
            .map(|(_, &i)| Learned::Sentence(i))
            .collect();
        out.extend(
            self.words
                .iter()
                .enumerate()
                .filter(|(_, w)| {
                    w.count > 0
                        && surface.contains(&w.word.chosen)
                        && (words::applies(&w.word, &reading_chars)
                            // the word of a choice of the whole reading,
                            // which has no context to apply by
                            || (w.word.span == reading
                                && w.word.left.is_empty()
                                && w.word.right.is_empty()))
                })
                .map(|(i, _)| Learned::Word(i)),
        );
        out
    }

    /// Forget what was learned to put `surface` up for `reading`; false
    /// when nothing was.
    pub fn forget(&mut self, reading: &str, surface: &str) -> std::io::Result<bool> {
        self.sync();
        if reading.is_empty() || surface.is_empty() || self.learned(reading, surface).is_empty() {
            return Ok(false);
        }
        self.append(&Record {
            reading: reading.to_string(),
            chosen: surface.to_string(),
            rejected: String::new(),
            forget: true,
            ..Record::default()
        })?;
        Ok(true)
    }

    /// Record that the user committed `chosen` for `reading` while `rejected`
    /// was shown first.
    pub fn record(&mut self, reading: &str, chosen: &str, rejected: &str) -> std::io::Result<()> {
        if reading.is_empty() || chosen.is_empty() || chosen == rejected {
            return Ok(());
        }
        self.append(&Record {
            reading: reading.to_string(),
            chosen: chosen.to_string(),
            rejected: rejected.to_string(),
            ..Record::default()
        })
    }

    fn append(&mut self, record: &Record) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        // One append-mode write per line keeps lines from different
        // processes whole.
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(&line)?;
        self.sync();
        Ok(())
    }

    /// Reorder `candidates` (best first) for `reading` by the learned choices.
    /// Without an applicable choice the order is kept.
    pub fn rerank(&mut self, reading: &str, candidates: Vec<String>) -> Vec<String> {
        self.sync();
        if candidates.len() < 2
            || (self.entries.is_empty() && self.words.is_empty())
            || reading.is_empty()
        {
            return candidates;
        }
        let reading_chars: Vec<char> = reading.chars().collect();
        let word_hits: Vec<&WordEntry> = self
            .words
            .iter()
            .filter(|w| self.use_words && w.count > 0 && words::applies(&w.word, &reading_chars))
            .collect();
        let rv = bigrams(reading);
        let mut seen: HashSet<usize> = HashSet::new();
        let mut hits: Vec<(usize, f32)> = Vec::new();
        for (k, _) in &rv {
            for &i in self.postings.get(k).into_iter().flatten() {
                if !seen.insert(i) || self.entries[i].count == 0 {
                    continue;
                }
                let sim = cosine(&rv, &self.entries[i].reading);
                if sim >= self.threshold {
                    hits.push((i, sim));
                }
            }
        }
        if hits.is_empty() && word_hits.is_empty() {
            return candidates;
        }
        let n = candidates.len() as f32;
        let scores: Vec<f32> = candidates
            .iter()
            .enumerate()
            .map(|(rank, c)| {
                let cv = bigrams(c);
                let bonus: f32 = hits
                    .iter()
                    .map(|&(i, sim)| {
                        let e = &self.entries[i];
                        let w = 1.0 + (e.count as f32).ln();
                        sim * w * (cosine(&cv, &e.chosen) - cosine(&cv, &e.rejected)) / e.span
                    })
                    .sum();
                let word_bonus: f32 = word_hits
                    .iter()
                    .map(|w| {
                        let has = |s: &str| if c.contains(s) { 1.0 } else { 0.0 };
                        (1.0 + (w.count as f32).ln())
                            * (has(&w.word.chosen) - has(&w.word.rejected))
                    })
                    .sum();
                -(rank as f32) / n + self.strength * (bonus + word_bonus)
            })
            .collect();
        // First index of the highest score: ties keep the engine's order.
        let best = (1..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
        let mut candidates = candidates;
        if best > 0 {
            let c = candidates.remove(best);
            candidates.insert(0, c);
        }
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempFile(PathBuf);

    impl TempFile {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("setsuna-learning-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir.join("learning.jsonl"))
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            if let Some(dir) = self.0.parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    fn cfg() -> LearningCfg {
        LearningCfg::default()
    }

    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cosine_of_bigrams() {
        let a = bigrams("きしゃにれんらく");
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
        let b = bigrams("きしゃにれんらくします");
        let s = cosine(&a, &b);
        assert!(s > 0.7 && s < 0.8, "{s}");
        assert!(cosine(&a, &bigrams("あいうえお")) < 0.01);
    }

    #[test]
    fn exact_reading_choice_moves_to_the_top() {
        let t = TempFile::new("exact");
        let mut l = Learning::at(t.0.clone(), &cfg());
        let cands = list(&["記者に連絡", "汽車に連絡", "帰社に連絡", "貴社に連絡"]);
        assert_eq!(l.rerank("きしゃにれんらく", cands.clone()), cands);
        l.record("きしゃにれんらく", "貴社に連絡", "記者に連絡")
            .unwrap();
        let got = l.rerank("きしゃにれんらく", cands);
        // Only the learned choice moves; the rest keep the engine's order.
        assert_eq!(
            got,
            list(&["貴社に連絡", "記者に連絡", "汽車に連絡", "帰社に連絡"])
        );
    }

    #[test]
    fn candidates_sharing_the_chosen_characters_do_not_jump() {
        // Aster-v1.2-small's list for いしをつたえたい after 意思 -> 意志 was
        // learned on いしをつたえる.
        let t = TempFile::new("no-jump");
        let mut l = Learning::at(t.0.clone(), &cfg());
        l.record("いしをつたえる", "意志を伝える", "意思を伝える")
            .unwrap();
        let cands = list(&[
            "意思を伝えたい",
            "意志を伝えたい",
            "医思を伝えたい",
            "意師を伝えたい",
            "意志をつ伝えたい",
            "医志を伝えたい",
        ]);
        let got = l.rerank("いしをつたえたい", cands);
        assert_eq!(
            got[..4],
            list(&[
                "意志を伝えたい",
                "意思を伝えたい",
                "医思を伝えたい",
                "意師を伝えたい"
            ])[..]
        );
    }

    #[test]
    fn similar_reading_gets_the_choice_unrelated_does_not() {
        let t = TempFile::new("similar");
        let mut l = Learning::at(t.0.clone(), &cfg());
        l.record("きしゃにれんらく", "貴社に連絡", "記者に連絡")
            .unwrap();
        let got = l.rerank(
            "きしゃにれんらくします",
            list(&["記者に連絡します", "貴社に連絡します"]),
        );
        assert_eq!(got[0], "貴社に連絡します");
        let other = list(&["天気", "転機"]);
        assert_eq!(l.rerank("てんき", other.clone()), other);
    }

    #[test]
    fn a_word_choice_applies_next_to_the_same_reading_only() {
        // かみをきる / かみをきっておく are only 0.54 similar as sentences,
        // below the 0.6 threshold: the word choice かみ: 髪 -> 紙 (before を)
        // carries it. かみがながい (before が) keeps 髪.
        let t = TempFile::new("words");
        let mut l = Learning::at(t.0.clone(), &cfg());
        l.record("かみをきる", "紙を切る", "髪を切る").unwrap();
        let cands = list(&["髪を切っておく", "紙を切っておく", "髪を切って置く"]);
        assert_eq!(
            l.rerank("かみをきっておく", cands),
            list(&["紙を切っておく", "髪を切っておく", "髪を切って置く"])
        );
        let long = list(&["髪が長い", "紙が長い"]);
        assert_eq!(l.rerank("かみがながい", long.clone()), long);

        let off = LearningCfg {
            words: false,
            ..cfg()
        };
        let mut l = Learning::at(t.0.clone(), &off);
        let cands = list(&["髪を切っておく", "紙を切っておく"]);
        assert_eq!(l.rerank("かみをきっておく", cands.clone()), cands);
    }

    #[test]
    fn threshold_one_means_exact_reading_only() {
        let t = TempFile::new("exact-only");
        let strict = LearningCfg {
            similarity_threshold: 1.0,
            words: false,
            ..cfg()
        };
        let mut l = Learning::at(t.0.clone(), &strict);
        l.record("きしゃにれんらく", "貴社に連絡", "記者に連絡")
            .unwrap();
        let cands = list(&["記者に連絡します", "貴社に連絡します"]);
        assert_eq!(l.rerank("きしゃにれんらくします", cands.clone()), cands);
    }

    #[test]
    fn forgetting_a_candidate_drops_its_choices_everywhere() {
        let t = TempFile::new("forget");
        let mut a = Learning::at(t.0.clone(), &cfg());
        let mut b = Learning::at(t.0.clone(), &cfg());
        a.record("かみ", "紙", "神").unwrap();
        assert_eq!(a.rerank("かみ", list(&["神", "紙"]))[0], "紙");
        // nothing learned for 神: nothing to forget
        assert!(!a.forget("かみ", "神").unwrap());
        assert!(a.forget("かみ", "紙").unwrap());
        assert_eq!(a.rerank("かみ", list(&["神", "紙"]))[0], "神");
        assert_eq!(a.len(), 0);
        // another process sees it
        assert_eq!(b.rerank("かみ", list(&["神", "紙"]))[0], "神");
    }

    #[test]
    fn forgetting_leaves_a_word_learned_elsewhere() {
        let t = TempFile::new("forget-word");
        let mut l = Learning::at(t.0.clone(), &cfg());
        l.record("かさのした", "傘の下", "笠の下").unwrap();
        // 傘 was learned before のし: not what put 傘の下 up for かさの
        assert!(!l.forget("かさの", "傘の下").unwrap());
        assert_eq!(
            l.rerank("かさのしたで", list(&["笠の下で", "傘の下で"]))[0],
            "傘の下で"
        );
        // where it applies, it goes
        assert!(l.forget("かさのしたで", "傘の下で").unwrap());
    }

    #[test]
    fn choosing_back_cancels_and_other_processes_see_appends() {
        let t = TempFile::new("share");
        let mut a = Learning::at(t.0.clone(), &cfg());
        let mut b = Learning::at(t.0.clone(), &cfg());
        a.record("かみ", "紙", "神").unwrap();
        // `b` picks up the line `a` appended.
        assert_eq!(b.rerank("かみ", list(&["神", "紙"]))[0], "紙");
        b.record("かみ", "神", "紙").unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(a.rerank("かみ", list(&["紙", "神"]))[0], "神");
    }

    #[test]
    fn deleting_the_file_clears_and_partial_lines_wait() {
        let t = TempFile::new("clear");
        let mut l = Learning::at(t.0.clone(), &cfg());
        l.record("かみ", "紙", "神").unwrap();
        assert_eq!(l.len(), 1);
        std::fs::remove_file(&t.0).unwrap();
        assert_eq!(l.len(), 0);
        // A line without its newline yet is not applied.
        std::fs::write(&t.0, r#"{"reading":"かみ","chosen":"紙","rejected":"神"}"#).unwrap();
        assert_eq!(l.len(), 0);
        let mut f = OpenOptions::new().append(true).open(&t.0).unwrap();
        f.write_all(b"\n").unwrap();
        assert_eq!(l.len(), 1);
    }

    /// The choices and word choices in effect, with their counts.
    fn in_effect(l: &Learning) -> (Vec<(String, u32)>, Vec<(String, u32)>) {
        let mut s: Vec<(String, u32)> = l
            .by_key
            .iter()
            .filter(|(_, &i)| l.entries[i].count > 0)
            .map(|(k, &i)| (format!("{k:?}"), l.entries[i].count))
            .collect();
        let mut w: Vec<(String, u32)> = l
            .words
            .iter()
            .filter(|w| w.count > 0)
            .map(|w| (format!("{:?}", w.word), w.count))
            .collect();
        s.sort();
        w.sort();
        (s, w)
    }

    fn line(reading: &str, chosen: &str, rejected: &str) -> String {
        format!(
            "{{\"reading\":\"{reading}\",\"chosen\":\"{chosen}\",\"rejected\":\"{rejected}\"}}\n"
        )
    }

    /// A log of many lines and few choices: the same ones made again and
    /// again, one chosen back, one forgotten.
    fn long_log() -> String {
        let mut log = String::new();
        for _ in 0..300 {
            log += &line("きしゃにれんらく", "貴社に連絡", "記者に連絡");
            log += &line("かみをきる", "紙を切る", "髪を切る");
        }
        log += &line("かみ", "紙", "神");
        log += &line("かみ", "神", "紙");
        log += &line("こうえん", "公演", "公園");
        log += "{\"reading\":\"こうえん\",\"chosen\":\"公演\",\"forget\":true}\n";
        log
    }

    #[test]
    fn a_long_log_is_compacted_to_what_is_in_effect() {
        let t = TempFile::new("compact");
        std::fs::create_dir_all(t.0.parent().unwrap()).unwrap();
        std::fs::write(&t.0, long_log()).unwrap();
        let before = Learning::load(t.0.clone(), &cfg());
        let mut l = Learning::at(t.0.clone(), &cfg());
        let text = std::fs::read_to_string(&t.0).unwrap();
        // a first line, two choices and their word choices (and 神 over 紙)
        assert!(text.lines().count() < 10, "{text}");
        assert!(text.starts_with("{\"compacted\":"), "{text}");
        assert!(text.contains("\"count\":300"), "{text}");
        assert!(!text.contains("公演"), "{text}");
        assert_eq!(in_effect(&l), in_effect(&before));
        // read again, it is the same, and is not compacted again
        let again = Learning::at(t.0.clone(), &cfg());
        assert_eq!(in_effect(&again), in_effect(&before));
        assert_eq!(std::fs::read_to_string(&t.0).unwrap(), text);
        // and reranks as before
        let cands = list(&["髪を切っておく", "紙を切っておく"]);
        assert_eq!(l.rerank("かみをきっておく", cands)[0], "紙を切っておく");
        // no file of its own is left
        let files = std::fs::read_dir(t.0.parent().unwrap()).unwrap().count();
        assert_eq!(files, 1);
    }

    #[test]
    fn a_short_or_tight_log_is_left_as_it_is() {
        let t = TempFile::new("no-compact");
        std::fs::create_dir_all(t.0.parent().unwrap()).unwrap();
        let log = line("かみ", "紙", "神").repeat(COMPACT_FLOOR / 2);
        std::fs::write(&t.0, &log).unwrap();
        Learning::at(t.0.clone(), &cfg());
        assert_eq!(std::fs::read_to_string(&t.0).unwrap(), log);
    }

    #[test]
    fn other_processes_read_a_compacted_log_again() {
        let t = TempFile::new("compact-share");
        std::fs::create_dir_all(t.0.parent().unwrap()).unwrap();
        // `short` read the log while it was a line, `long` while it was
        // long: the compacted file is longer than the one and shorter than
        // the other
        std::fs::write(&t.0, line("かみ", "神", "紙")).unwrap();
        let mut short = Learning::load(t.0.clone(), &cfg());
        let log = long_log();
        let mut f = OpenOptions::new().append(true).open(&t.0).unwrap();
        f.write_all(log.as_bytes()).unwrap();
        let mut long = Learning::load(t.0.clone(), &cfg());
        let before = in_effect(&long);
        let mut l = Learning::at(t.0.clone(), &cfg());
        assert_eq!(in_effect(&l), before);
        assert!(std::fs::metadata(&t.0).unwrap().len() > short.offset);
        assert!(std::fs::metadata(&t.0).unwrap().len() < long.offset);
        short.sync();
        long.sync();
        assert_eq!(in_effect(&short), before);
        assert_eq!(in_effect(&long), before);
        // a choice after it is appended as before and seen by all
        l.record("てんき", "転機", "天気").unwrap();
        assert_eq!(short.rerank("てんき", list(&["天気", "転機"]))[0], "転機");
    }

    #[test]
    fn a_log_appended_to_meanwhile_is_not_replaced() {
        let t = TempFile::new("compact-race");
        std::fs::create_dir_all(t.0.parent().unwrap()).unwrap();
        std::fs::write(&t.0, long_log()).unwrap();
        let mut l = Learning::load(t.0.clone(), &cfg());
        // another process appends after this one read the log
        let mut f = OpenOptions::new().append(true).open(&t.0).unwrap();
        f.write_all(line("てんき", "転機", "天気").as_bytes())
            .unwrap();
        let text = std::fs::read_to_string(&t.0).unwrap();
        assert!(!l.compact().unwrap());
        assert_eq!(std::fs::read_to_string(&t.0).unwrap(), text);
        let files = std::fs::read_dir(t.0.parent().unwrap()).unwrap().count();
        assert_eq!(files, 1);
    }

    #[test]
    fn a_version_before_compaction_reads_a_compacted_log() {
        // what the lines of the log were to it
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Before {
            reading: String,
            chosen: String,
            #[serde(default)]
            rejected: String,
            #[serde(default)]
            forget: bool,
        }
        let t = TempFile::new("compact-old");
        std::fs::create_dir_all(t.0.parent().unwrap()).unwrap();
        std::fs::write(&t.0, long_log()).unwrap();
        Learning::at(t.0.clone(), &cfg());
        let text = std::fs::read_to_string(&t.0).unwrap();
        let read: Vec<Before> = text
            .lines()
            .filter_map(|l| serde_json::from_str::<Before>(l).ok())
            .collect();
        // the sentence choices, each once; the first line and the word
        // choices skipped
        let mut chosen: Vec<&str> = read.iter().map(|r| r.chosen.as_str()).collect();
        chosen.sort();
        assert_eq!(chosen, ["神", "紙を切る", "貴社に連絡"]);
    }
}
