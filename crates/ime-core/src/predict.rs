//! Prediction from what the user committed, as Mozc's user history
//! predictor does it (the model cannot complete a reading: its output is
//! bounded by what is typed). Every conversion committed is recorded in
//! `prediction.jsonl` next to the config (`{"r": reading, "s": surface,
//! "p": the commit before it}`, appended so that every process sees it);
//! typing a reading's start offers the longer surfaces committed for it,
//! the more often and the more recently the higher, and higher still after
//! the same commit as before (Mozc's bigram links: the next word).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    r: String,
    s: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    p: String,
}

#[derive(Default)]
struct Stat {
    count: u32,
    /// Order of the last use (higher is later).
    last: u64,
    /// The commits seen right before it, with their counts.
    after: HashMap<String, u32>,
}

/// A prediction: the surface and the reading it was committed for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction {
    pub reading: String,
    pub surface: String,
}

pub struct Predictor {
    path: PathBuf,
    offset: u64,
    seq: u64,
    stats: HashMap<(String, String), Stat>,
}

/// Readings shorter than this are not predicted from (too many surfaces).
pub const MIN_PREFIX: usize = 2;
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
            seq: 0,
            stats: HashMap::new(),
        };
        p.sync();
        p
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn sync(&mut self) {
        let len = match std::fs::metadata(&self.path) {
            Ok(m) => m.len(),
            Err(_) => {
                self.offset = 0;
                self.stats.clear();
                return;
            }
        };
        if len < self.offset {
            self.offset = 0;
            self.stats.clear();
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
        let Some(end) = buf.iter().rposition(|&b| b == b'\n') else {
            return;
        };
        for line in buf[..end].split(|&b| b == b'\n') {
            if let Ok(r) = serde_json::from_slice::<Record>(line) {
                self.seq += 1;
                let s = self.stats.entry((r.r, r.s)).or_default();
                s.count += 1;
                s.last = self.seq;
                if !r.p.is_empty() {
                    *s.after.entry(r.p).or_default() += 1;
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
    /// (kana alone needs no prediction) and the reading is not too short.
    pub fn worth(reading: &str, surface: &str) -> bool {
        reading.chars().count() > MIN_PREFIX && surface != reading && !surface.trim().is_empty()
    }

    /// Record that `surface` was committed for `reading` after `prev`.
    pub fn record(&mut self, reading: &str, surface: &str, prev: &str) -> std::io::Result<()> {
        if !Self::worth(reading, surface) {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_vec(&Record {
            r: reading.to_string(),
            s: surface.to_string(),
            p: prev
                .chars()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect(),
        })?;
        line.push(b'\n');
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(&line)?;
        self.sync();
        Ok(())
    }

    /// Up to `n` predictions for `prefix` typed after `prev`: surfaces
    /// committed for a longer reading that starts with it, best first.
    pub fn complete(&mut self, prefix: &str, prev: &str, n: usize) -> Vec<Prediction> {
        self.sync();
        if prefix.chars().count() < MIN_PREFIX {
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
            .filter(|((r, _), _)| r.len() > prefix.len() && r.starts_with(prefix))
            .map(|(k, s)| {
                let after = if prev.is_empty() {
                    0
                } else {
                    *s.after.get(&prev).unwrap_or(&0)
                };
                let score = (1.0 + s.count as f64).ln()
                    + 2.0 * (1.0 + after as f64).ln()
                    + 0.3 * s.last as f64 / seq
                    // the less is left to type, the likelier
                    - 0.02 * (k.0.chars().count() - prefix.chars().count()) as f64;
                (score, k)
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
        assert_eq!(p.complete("よろ", "", 5).len(), 3);
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
}
