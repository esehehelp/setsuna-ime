//! Candidates the user hid (Ctrl+Delete on one nothing learned put up):
//! `hidden.jsonl` next to the config, one `{"reading", "surface"}` per
//! line, appended so every process sees them, as Mozc keeps such pairs in
//! a "非表示候補" dictionary of suppression words. A line with `"show":
//! true` takes a pair back. Kept apart from the learning, so that clearing
//! it does not bring them back.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    reading: String,
    surface: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    show: bool,
}

pub struct Hidden {
    path: PathBuf,
    /// Bytes of the file already applied.
    offset: u64,
    pairs: HashSet<(String, String)>,
}

impl Hidden {
    /// The hidden candidates of the default config directory.
    pub fn open() -> Option<Self> {
        crate::config::Config::hidden_path().map(Self::at)
    }

    /// The hidden candidates stored at `path` (which need not exist yet).
    pub fn at(path: PathBuf) -> Self {
        let mut h = Self {
            path,
            offset: 0,
            pairs: HashSet::new(),
        };
        h.sync();
        h
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The pairs hidden, sorted.
    pub fn list(&mut self) -> Vec<(String, String)> {
        self.sync();
        let mut v: Vec<_> = self.pairs.iter().cloned().collect();
        v.sort();
        v
    }

    /// Apply the lines appended since the last sync; a file that shrank or
    /// went away was cleared.
    fn sync(&mut self) {
        let len = match std::fs::metadata(&self.path) {
            Ok(m) => m.len(),
            Err(_) => {
                self.offset = 0;
                self.pairs.clear();
                return;
            }
        };
        if len < self.offset {
            self.offset = 0;
            self.pairs.clear();
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
                let pair = (r.reading, r.surface);
                if r.show {
                    self.pairs.remove(&pair);
                } else {
                    self.pairs.insert(pair);
                }
            }
        }
        self.offset += end as u64 + 1;
    }

    fn append(&mut self, record: &Record) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(&line)?;
        self.sync();
        Ok(())
    }

    /// Hide `surface` for `reading`.
    pub fn hide(&mut self, reading: &str, surface: &str) -> std::io::Result<()> {
        self.append(&Record {
            reading: reading.to_string(),
            surface: surface.to_string(),
            show: false,
        })
    }

    /// Show `surface` for `reading` again.
    pub fn show(&mut self, reading: &str, surface: &str) -> std::io::Result<()> {
        self.append(&Record {
            reading: reading.to_string(),
            surface: surface.to_string(),
            show: true,
        })
    }

    /// `candidates` for `reading` without the hidden ones.
    pub fn filter(&mut self, reading: &str, candidates: Vec<String>) -> Vec<String> {
        self.sync();
        if self.pairs.is_empty() {
            return candidates;
        }
        candidates
            .into_iter()
            .filter(|c| !self.pairs.contains(&(reading.to_string(), c.clone())))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_pairs_are_filtered_shared_and_shown_back() {
        let path =
            std::env::temp_dir().join(format!("setsuna-hidden-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut a = Hidden::at(path.clone());
        let mut b = Hidden::at(path.clone());
        a.hide("かみ", "加味").unwrap();
        let list = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            a.filter("かみ", list(&["紙", "加味", "神"])),
            list(&["紙", "神"])
        );
        // another reading keeps it
        assert_eq!(a.filter("かみの", list(&["加味の"])), list(&["加味の"]));
        // another process sees it
        assert_eq!(b.filter("かみ", list(&["加味"])), Vec::<String>::new());
        b.show("かみ", "加味").unwrap();
        assert_eq!(a.filter("かみ", list(&["加味"])), list(&["加味"]));
        let _ = std::fs::remove_file(&path);
    }
}
