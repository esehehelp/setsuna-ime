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
/// history).
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    reading: String,
    chosen: String,
    #[serde(default)]
    rejected: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    forget: bool,
}

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

    /// The learning stored at `path` (which need not exist yet).
    pub fn at(path: PathBuf, cfg: &LearningCfg) -> Self {
        let mut l = Self {
            path,
            strength: cfg.strength,
            threshold: cfg.similarity_threshold,
            use_words: cfg.words,
            offset: 0,
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
        self.entries.clear();
        self.by_key.clear();
        self.postings.clear();
        self.words.clear();
        self.word_key.clear();
    }

    /// Apply the lines appended to the file since the last sync. A file that
    /// shrank or disappeared was cleared: start over.
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
        let mut buf = Vec::new();
        let read = File::open(&self.path).and_then(|mut f| {
            f.seek(SeekFrom::Start(self.offset))?;
            f.read_to_end(&mut buf)
        });
        if read.is_err() {
            return;
        }
        // Only whole lines: another process may be mid-append.
        let Some(end) = buf.iter().rposition(|&b| b == b'\n') else {
            return;
        };
        for line in buf[..end].split(|&b| b == b'\n') {
            if let Ok(rec) = serde_json::from_slice::<Record>(line) {
                self.apply(rec);
            }
        }
        self.offset += end as u64 + 1;
    }

    fn apply(&mut self, r: Record) {
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
        for w in words::derive(&r.reading, &r.chosen, &r.rejected) {
            self.apply_word(w);
        }
        self.apply_sentence(r);
    }

    fn apply_word(&mut self, w: words::Word) {
        let opposite = words::Word {
            chosen: w.rejected.clone(),
            rejected: w.chosen.clone(),
            ..w.clone()
        };
        if let Some(&i) = self.word_key.get(&opposite) {
            self.words[i].count = 0;
        }
        if let Some(&i) = self.word_key.get(&w) {
            self.words[i].count += 1;
            return;
        }
        self.word_key.insert(w.clone(), self.words.len());
        self.words.push(WordEntry { count: 1, word: w });
    }

    fn apply_sentence(&mut self, r: Record) {
        // Choosing back what was rejected cancels the earlier choice.
        let opposite = (r.reading.clone(), r.rejected.clone(), r.chosen.clone());
        if let Some(&i) = self.by_key.get(&opposite) {
            self.entries[i].count = 0;
        }
        let key = (r.reading, r.chosen, r.rejected);
        if let Some(&i) = self.by_key.get(&key) {
            self.entries[i].count += 1;
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
            count: 1,
            reading,
            chosen,
            rejected,
            span,
        });
        self.by_key.insert(key, id);
    }

    /// The choices in effect that put `surface` up for `reading`: the
    /// sentence choices of it, and the word choices of a word inside it.
    fn learned(&self, reading: &str, surface: &str) -> Vec<Learned> {
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
                        && reading.contains(&w.word.span)
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
            forget: false,
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
        if candidates.len() < 2 || self.entries.is_empty() || reading.is_empty() {
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
}
