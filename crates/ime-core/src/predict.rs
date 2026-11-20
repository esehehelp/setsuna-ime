//! Prediction from what the user committed, as Mozc's user history
//! predictor does it (the model cannot complete a reading: its output is
//! bounded by what is typed). Every conversion committed is recorded in
//! `prediction.jsonl` next to the config (`{"r": reading, "s": surface,
//! "p": the commit before it}`, appended so that every process sees it);
//! typing a reading's start offers the longer surfaces committed for it,
//! the more often and the more recently the higher, and higher still after
//! the same commit as before (Mozc's bigram links: the next word).
//! Ctrl+Delete on a prediction appends the same with `"d": true`: its
//! commits are forgotten.
//!
//! A log that grew to many times the entries it holds is compacted to them
//! (`{"r", "s", "n": count, "a": {commit before: count}}` after a first
//! line of its own, which tells other processes to read it again), as the
//! learning's is; every append and the compaction take the file's lock.
//!
//! As Mozc's IsValidSuggestion, a surface committed once is offered from
//! three chars typed, one committed again or after the same commit as
//! before from two. Live conversion commits whole sentences, which are
//! seldom typed again: a reading longer than a phrase (`MAX_READING`) is
//! neither recorded nor offered, or a sentence's first chars would bring
//! the whole of an earlier input back.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One line of `prediction.jsonl`: a commit, or (`d`) that the commits of
/// `s` for `r` are forgotten (Ctrl+Delete on the prediction).
#[derive(Debug, Default, Serialize, Deserialize)]
struct Record {
    r: String,
    s: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    p: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    d: bool,
    /// Commits it stands for (a compacted log's), 1 when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    /// The commits before it, with their counts (a compacted log's).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    a: HashMap<String, u32>,
}

/// A compacted log's first line: when, so that it differs from the head
/// any process read before.
#[derive(Serialize)]
struct Compacted {
    compacted: u128,
}

#[derive(Default)]
struct Stat {
    count: u32,
    /// Order of the last use (higher is later).
    last: u64,
    /// The commits seen right before it, with their counts.
    after: HashMap<String, u32>,
}

/// Whether a commit holds what looks like a number, a key or an address.
fn sensitive(reading: &str, surface: &str) -> bool {
    crate::privacy::is_sensitive(reading) || crate::privacy::is_sensitive(surface)
}

/// A prediction: the surface and the reading it was committed for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction {
    pub reading: String,
    pub surface: String,
}

pub struct Predictor {
    path: PathBuf,
    /// Bytes of the file applied, its time of change then, its first bytes
    /// (up to `HEAD`) and its lines.
    offset: u64,
    stamp: Option<std::time::SystemTime>,
    head: Vec<u8>,
    lines: usize,
    seq: u64,
    stats: HashMap<(String, String), Stat>,
}

/// Bytes of the file's start kept to tell a file replaced (compacted).
const HEAD: usize = 256;
/// Commits before a surface kept for it, the most frequent.
const AFTER_CAP: usize = 16;
/// Lines from which a log may be compacted (when most are no longer in
/// effect).
const COMPACT_LINES: usize = 2 * CAP;

impl Stat {
    fn add_after(&mut self, prev: String, n: u32) {
        let c = self.after.entry(prev).or_default();
        *c = c.saturating_add(n);
        if self.after.len() > AFTER_CAP {
            if let Some(least) = self
                .after
                .iter()
                .min_by_key(|&(p, &c)| (c, std::cmp::Reverse(p.clone())))
                .map(|(p, _)| p.clone())
            {
                self.after.remove(&least);
            }
        }
    }
}

/// Readings shorter than this are not predicted from (too many surfaces).
pub const MIN_PREFIX: usize = 2;
/// What has to be typed of a surface committed only once, after another
/// commit than the one before it then.
const MIN_PREFIX_ONCE: usize = 3;
/// The longest reading recorded and offered: a phrase, not a sentence.
const MAX_READING: usize = 16;
/// Entries kept (the least recently used are forgotten on load).
const CAP: usize = 10_000;

impl Predictor {
    /// The predictions of the default config directory.
    pub fn open() -> Option<Self> {
        crate::config::Config::prediction_path().map(Self::at)
    }

    /// The predictions stored at `path` (which need not exist yet).
    pub fn at(path: PathBuf) -> Self {
        let mut p = Self {
            path,
            offset: 0,
            stamp: None,
            head: Vec::new(),
            lines: 0,
            seq: 0,
            stats: HashMap::new(),
        };
        p.sync();
        p
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.stamp = None;
        self.head.clear();
        self.lines = 0;
        self.seq = 0;
        self.stats.clear();
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Apply the lines appended since the last sync. A file that shrank or
    /// went away was cleared, and one that starts otherwise than it did was
    /// compacted (by another process): start over.
    fn sync(&mut self) {
        let (len, stamp) = match std::fs::metadata(&self.path) {
            Ok(m) => (m.len(), m.modified().ok()),
            Err(_) => {
                self.reset();
                return;
            }
        };
        if len < self.offset {
            self.reset();
        }
        if len == self.offset && stamp == self.stamp {
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
        self.stamp = stamp;
        let mut buf = Vec::new();
        let read = f
            .seek(SeekFrom::Start(self.offset))
            .and_then(|_| f.read_to_end(&mut buf));
        if read.is_err() {
            return;
        }
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
            if let Ok(r) = serde_json::from_slice::<Record>(line) {
                if r.d {
                    self.stats.remove(&(r.r, r.s));
                    continue;
                }
                // nor one recorded before such commits were kept out
                let n = r.n.unwrap_or(1);
                if n == 0 || sensitive(&r.r, &r.s) {
                    continue;
                }
                self.seq += 1;
                let s = self.stats.entry((r.r, r.s)).or_default();
                s.count = s.count.saturating_add(n);
                s.last = self.seq;
                if !r.p.is_empty() {
                    s.add_after(r.p, 1);
                }
                for (p, c) in r.a {
                    s.add_after(p, c);
                }
            }
        }
        self.offset += end as u64 + 1;
        if self.stats.len() > CAP {
            let mut lasts: Vec<u64> = self.stats.values().map(|s| s.last).collect();
            lasts.sort_unstable();
            let cut = lasts[lasts.len() - CAP];
            self.stats.retain(|_, s| s.last >= cut);
        }
    }

    /// Whether a commit is worth predicting: it differs from its reading
    /// (kana alone needs no prediction), the reading is neither too short
    /// nor a sentence's, and it keeps the numbers typed in digits (a commit
    /// the model garbled, 19911225 -> 199125, before
    /// `postprocess::drop_changed_digits`).
    pub fn worth(reading: &str, surface: &str) -> bool {
        (MIN_PREFIX + 1..=MAX_READING).contains(&reading.chars().count())
            && surface != reading
            && !surface.trim().is_empty()
            && crate::postprocess::keeps_digits(reading, surface)
    }

    /// Record that `surface` was committed for `reading` after `prev`; not
    /// what looks like a number, a key or an address to keep from the
    /// history (`privacy`), nor such a commit as the one before.
    pub fn record(&mut self, reading: &str, surface: &str, prev: &str) -> std::io::Result<()> {
        if !Self::worth(reading, surface) || sensitive(reading, surface) {
            return Ok(());
        }
        // the commit before as a whole is judged, not the end kept of it
        // (`Password: the value is hunter2` ends in no announcing word)
        let prev: String = if crate::privacy::is_sensitive(prev) {
            String::new()
        } else {
            prev.chars()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        };
        self.append(&Record {
            r: reading.to_string(),
            s: surface.to_string(),
            p: prev,
            ..Record::default()
        })
    }

    /// Forget the commits of `surface` for `reading`: it is no longer
    /// predicted (here and in every process, which read the line appended).
    /// False when there was none.
    pub fn forget(&mut self, reading: &str, surface: &str) -> std::io::Result<bool> {
        self.sync();
        if !self
            .stats
            .contains_key(&(reading.to_string(), surface.to_string()))
        {
            return Ok(false);
        }
        self.append(&Record {
            r: reading.to_string(),
            s: surface.to_string(),
            d: true,
            ..Record::default()
        })?;
        Ok(true)
    }

    fn append(&mut self, record: &Record) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        crate::private_file::append_locked(&self.path, &line)?;
        self.sync();
        if self.lines > COMPACT_LINES && self.lines > 2 * self.stats.len() + 1 {
            // a failure leaves the log as it was
            let _ = self.compact();
        }
        Ok(())
    }

    /// Write what is in effect as a compacted file and move it over the
    /// log, under the lock every append takes; false when the log changed
    /// since it was read, or another process holds the lock.
    fn compact(&mut self) -> std::io::Result<bool> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut out = serde_json::to_vec(&Compacted { compacted: now })?;
        out.push(b'\n');
        let mut entries: Vec<(&(String, String), &Stat)> = self.stats.iter().collect();
        entries.sort_unstable_by_key(|&(_, s)| s.last);
        for ((r, s), stat) in entries {
            serde_json::to_writer(
                &mut out,
                &Record {
                    r: r.clone(),
                    s: s.clone(),
                    n: (stat.count != 1).then_some(stat.count),
                    a: stat.after.clone(),
                    ..Record::default()
                },
            )?;
            out.push(b'\n');
        }
        let (offset, path) = (self.offset, self.path.clone());
        let moved = crate::private_file::try_locked(&self.path, || {
            if std::fs::metadata(&path)?.len() != offset {
                return Ok(false);
            }
            crate::private_file::write(&path, &out).map(|()| true)
        })?;
        if moved == Some(true) {
            self.reset();
            self.sync();
            return Ok(true);
        }
        Ok(false)
    }

    /// Up to `n` predictions for `prefix` typed after `prev`: surfaces
    /// committed for a longer reading that starts with it, best first.
    pub fn complete(&mut self, prefix: &str, prev: &str, n: usize) -> Vec<Prediction> {
        self.sync();
        let typed = prefix.chars().count();
        if typed < MIN_PREFIX || n == 0 {
            return Vec::new();
        }
        let prev: String = prev
            .chars()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let seq = self.seq.max(1) as f64;
        let mut scored: Vec<(f64, &(String, String))> = self
            .stats
            .iter()
            .filter_map(|(k, s)| {
                let (r, surface) = k;
                if !(r.len() > prefix.len() && r.starts_with(prefix) && Self::worth(r, surface)) {
                    return None;
                }
                let after = if prev.is_empty() {
                    0
                } else {
                    *s.after.get(&prev).unwrap_or(&0)
                };
                if s.count < 2 && after == 0 && typed < MIN_PREFIX_ONCE {
                    return None;
                }
                let score = (1.0 + s.count as f64).ln()
                    + 2.0 * (1.0 + after as f64).ln()
                    + 0.3 * s.last as f64 / seq
                    // the less is left to type, the likelier
                    - 0.02 * (r.chars().count() - typed) as f64;
                Some((score, k))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        let mut out: Vec<Prediction> = Vec::new();
        for (_, (r, s)) in scored {
            if out.iter().any(|p| &p.surface == s) {
                continue;
            }
            out.push(Prediction {
                reading: r.clone(),
                surface: s.clone(),
            });
            if out.len() == n {
                break;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "setsuna-predict-{name}-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// A log of many lines for few entries is compacted to them, read again
    /// by a process that read it before; the commits before a surface are
    /// kept for it up to a few.
    #[test]
    fn a_long_log_is_compacted_and_read_again_elsewhere() {
        let path = temp("compact");
        let line = "{\"r\":\"よろしく\",\"s\":\"宜しく\",\"p\":\"では\"}\n";
        std::fs::write(&path, line.repeat(COMPACT_LINES)).unwrap();
        let mut other = Predictor::at(path.clone());
        let mut p = Predictor::at(path.clone());
        p.record("よろこび", "喜び", "").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("{\"compacted\":"), "{text}");
        assert_eq!(text.lines().count(), 3, "{text}");
        let key = ("よろしく".to_string(), "宜しく".to_string());
        for q in [&mut p, &mut other] {
            q.sync();
            assert_eq!(q.stats[&key].count, COMPACT_LINES as u32);
            assert_eq!(q.stats[&key].after["では"], COMPACT_LINES as u32);
            assert!(q
                .stats
                .contains_key(&("よろこび".to_string(), "喜び".to_string())));
        }
        assert!(p.complete("よろし", "", 0).is_empty());
        for i in 0..AFTER_CAP + 4 {
            p.record("よろしく", "宜しく", &format!("前{i}")).unwrap();
        }
        assert_eq!(p.stats[&key].after.len(), AFTER_CAP);
        assert!(p.stats[&key].after.contains_key("では"));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("jsonl.lock"));
    }

    #[test]
    fn a_number_or_a_key_is_kept_out_of_the_history() {
        let path = temp("private");
        let mut p = Predictor::at(path.clone());
        p.record("ばんごう", "090-1234-5678", "").unwrap();
        p.record("きーは", "sk-projAbCdEf123", "").unwrap();
        p.record("よろしく", "宜しく", "パスワードは").unwrap();
        p.record("よろしく", "宜しく", "Password: the value is hunter2")
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("090") && !text.contains("sk-"), "{text}");
        assert!(
            !text.contains("パスワード") && !text.contains("hunter2"),
            "{text}"
        );
        assert_eq!(p.complete("よろし", "", 5).len(), 1);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("jsonl.lock"));
    }

    #[test]
    fn longer_commits_are_predicted_from_their_start() {
        let path = temp("start");
        let mut p = Predictor::at(path.clone());
        p.record("よろしくおねがいします", "よろしくお願いします", "")
            .unwrap();
        p.record("よろしくおねがいします", "よろしくお願いします", "")
            .unwrap();
        p.record("よろしく", "宜しく", "").unwrap();
        p.record("よろこび", "喜び", "").unwrap();
        let out: Vec<String> = p
            .complete("よろし", "", 3)
            .into_iter()
            .map(|x| x.surface)
            .collect();
        assert_eq!(out, ["よろしくお願いします", "宜しく"]);
        // kana alone and short readings are not recorded
        p.record("よろしく", "よろしく", "").unwrap();
        assert_eq!(p.complete("よろ", "", 5).len(), 1);
        assert!(p.complete("よ", "", 5).is_empty());
        // another process sees them
        let mut q = Predictor::at(path.clone());
        assert_eq!(
            q.complete("よろし", "", 1)[0].surface,
            "よろしくお願いします"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_forgotten_prediction_is_offered_no_more() {
        let path = temp("forget");
        let mut p = Predictor::at(path.clone());
        let mut q = Predictor::at(path.clone());
        p.record("かさのした", "傘の下", "").unwrap();
        p.record("かさのした", "傘の下", "").unwrap();
        p.record("かさのしたで", "傘の下で", "").unwrap();
        assert!(!p.forget("かさのした", "笠の下").unwrap());
        assert!(p.forget("かさのした", "傘の下").unwrap());
        let surfaces = |p: &mut Predictor| -> Vec<String> {
            p.complete("かさの", "", 5)
                .into_iter()
                .map(|x| x.surface)
                .collect()
        };
        assert_eq!(surfaces(&mut p), ["傘の下で"]);
        // another process reads it too; a new commit counts again
        assert_eq!(surfaces(&mut q), ["傘の下で"]);
        p.record("かさのした", "傘の下", "").unwrap();
        assert_eq!(surfaces(&mut q).len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_garbled_number_is_neither_recorded_nor_predicted() {
        let path = temp("digits");
        // one recorded before commits were checked
        std::fs::write(&path, "{\"r\":\"19911225\",\"s\":\"199125\"}\n").unwrap();
        let mut p = Predictor::at(path.clone());
        p.record("19911225", "1991125", "").unwrap();
        p.record("1991ねん", "1991年", "").unwrap();
        let out: Vec<String> = p
            .complete("1991", "", 5)
            .into_iter()
            .map(|x| x.surface)
            .collect();
        assert_eq!(out, ["1991年"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_commit_before_ranks_the_next_word() {
        let path = temp("next");
        let mut p = Predictor::at(path.clone());
        p.record("かいぎしつ", "会議室", "").unwrap();
        p.record("かいぎしつ", "会議室", "").unwrap();
        p.record("かいぎろく", "会議録", "議事").unwrap();
        assert_eq!(p.complete("かいぎ", "", 1)[0].surface, "会議室");
        assert_eq!(p.complete("かいぎ", "議事", 1)[0].surface, "会議録");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_surface_committed_once_needs_three_chars_or_the_same_commit_before() {
        let path = temp("once");
        let mut p = Predictor::at(path.clone());
        p.record("あとで", "後で", "").unwrap();
        p.record("あとがき", "後書き", "本文").unwrap();
        let surfaces = |p: &mut Predictor, prefix: &str, prev: &str| -> Vec<String> {
            p.complete(prefix, prev, 5)
                .into_iter()
                .map(|x| x.surface)
                .collect()
        };
        assert!(surfaces(&mut p, "あと", "").is_empty());
        assert_eq!(surfaces(&mut p, "あと", "本文"), ["後書き"]);
        assert_eq!(surfaces(&mut p, "あとで", ""), Vec::<String>::new());
        assert_eq!(surfaces(&mut p, "あとが", ""), ["後書き"]);
        p.record("あとで", "後で", "").unwrap();
        assert_eq!(surfaces(&mut p, "あと", ""), ["後で"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_sentence_is_neither_recorded_nor_offered() {
        let path = temp("sentence");
        // one recorded before sentences were left out
        let old = "あとこれはどうようのもんだいがあるっぽい";
        std::fs::write(
            &path,
            format!(
                "{{\"r\":\"{old}\",\"s\":\"あとこれは同様の問題があるっぽい\"}}
"
            ),
        )
        .unwrap();
        let mut p = Predictor::at(path.clone());
        for _ in 0..3 {
            p.record(old, "あとこれは同様の問題があるっぽい", "")
                .unwrap();
        }
        assert!(p.complete("あとこれは", "", 5).is_empty());
        let len = std::fs::metadata(&path).unwrap().len();
        p.record("よろしくおねがいいたします", "よろしくお願いいたします", "")
            .unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > len);
        let _ = std::fs::remove_file(&path);
    }
}
