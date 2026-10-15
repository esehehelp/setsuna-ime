//! Romaji tables other than the built-in one: AZIK, kana input on the JIS
//! layout, or the user's own, in Mozc's romaji table format (`input<TAB>
//! output[<TAB>pending]`, `#` comments; a table is case-sensitive when an
//! input has a capital, as Mozc's). `[frontend] romaji_table` picks one
//! ("standard", the built-in table, is none of these); the one active reads
//! every composition of the process (`composing_text::romaji_units`).
//!
//! Matching is Mozc's: the longest input that matches; input that may
//! still become a longer one waits (`Pending`) at the end of what is typed;
//! a rule's pending text (`kk	っ	k`) is read again.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use crate::composing_text::{RomajiUnit, UnitKind};

/// The presets: name -> table text.
pub const PRESETS: &[(&str, &str)] = &[
    ("azik", include_str!("../data/romaji/azik.tsv")),
    ("kana_jis", include_str!("../data/romaji/kana_jis.tsv")),
];

pub struct RomajiTable {
    /// Input -> (output, chars of the input read again).
    rules: HashMap<String, (&'static str, usize)>,
    /// Every proper prefix of an input.
    prefixes: HashSet<String>,
    max_len: usize,
    case_sensitive: bool,
    /// The chars inputs use (a punctuation key typed into the composition
    /// when one does: AZIK's `;`).
    chars: HashSet<char>,
    /// Kana input: every printable key goes into the composition.
    pub kana_keys: bool,
}

impl RomajiTable {
    /// Read a table; the lines it could not read come back as errors.
    pub fn parse(text: &str) -> Result<Self, Vec<String>> {
        let mut t = Self {
            rules: HashMap::new(),
            prefixes: HashSet::new(),
            max_len: 0,
            case_sensitive: false,
            chars: HashSet::new(),
            kana_keys: false,
        };
        let mut errors = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 2
                || cols[0].is_empty()
                || (cols[1].is_empty() && cols.get(2).is_none_or(|p| p.is_empty()))
            {
                errors.push(format!("{}行目: 入力と出力がありません", n + 1));
                continue;
            }
            let input = cols[0].to_string();
            let pending = cols.get(2).copied().unwrap_or("");
            // the pending text must be the input's end, read again
            let again =
                if !pending.is_empty() && input.ends_with(pending) && pending.len() < input.len() {
                    pending.chars().count()
                } else if pending.is_empty() {
                    0
                } else {
                    errors.push(format!(
                        "{}行目: 保留の文字は入力の末尾にしてください",
                        n + 1
                    ));
                    continue;
                };
            let output: &'static str = Box::leak(cols[1].to_string().into_boxed_str());
            t.case_sensitive |= input.chars().any(|c| c.is_ascii_uppercase());
            t.chars.extend(input.chars());
            let chars: Vec<char> = input.chars().collect();
            t.max_len = t.max_len.max(chars.len());
            for k in 1..chars.len() {
                t.prefixes.insert(chars[..k].iter().collect());
            }
            t.rules.insert(input, (output, again));
        }
        if t.rules.is_empty() {
            errors.push("規則がありません".to_string());
        }
        // kana input: digits and symbols are kana keys too
        t.kana_keys = t.rules.contains_key("3") && t.rules.contains_key("@");
        if errors.is_empty() {
            Ok(t)
        } else {
            Err(errors)
        }
    }

    /// Whether `c` is part of some input (a key the table reads).
    pub fn uses(&self, c: char) -> bool {
        self.chars.contains(&c)
            || (!self.case_sensitive && self.chars.contains(&c.to_ascii_lowercase()))
    }

    pub fn case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// `chars` read with the table, as `composing_text::romaji_units`.
    pub(crate) fn units(&self, chars: &[char], flush: bool) -> Vec<RomajiUnit> {
        let chars: Vec<char> = if self.case_sensitive {
            chars.to_vec()
        } else {
            chars.iter().map(|c| c.to_ascii_lowercase()).collect()
        };
        let n = chars.len();
        let key = |a: usize, b: usize| -> String { chars[a..b].iter().collect() };
        let mut units = Vec::new();
        let mut i = 0;
        while i < n {
            // the longest run from `i` that is an input or the start of one
            let mut reach = 0;
            for len in 1..=self.max_len.min(n - i) {
                let k = key(i, i + len);
                if self.rules.contains_key(&k) || self.prefixes.contains(&k) {
                    reach = len;
                } else {
                    break;
                }
            }
            if reach == 0 {
                units.push(RomajiUnit {
                    start: i,
                    end: i + 1,
                    kind: UnitKind::Char(chars[i]),
                });
                i += 1;
                continue;
            }
            let rest = key(i, n);
            if i + reach == n
                && self.prefixes.contains(&rest)
                && !(flush && self.rules.contains_key(&rest))
            {
                units.push(RomajiUnit {
                    start: i,
                    end: n,
                    kind: UnitKind::Pending,
                });
                break;
            }
            let best = (1..=reach)
                .rev()
                .find_map(|len| self.rules.get(&key(i, i + len)).map(|r| (len, *r)));
            match best {
                Some((len, (out, again))) => {
                    let used = len - again;
                    units.push(RomajiUnit {
                        start: i,
                        end: i + used,
                        kind: UnitKind::Kana(out),
                    });
                    i += used;
                }
                None => {
                    units.push(RomajiUnit {
                        start: i,
                        end: i + 1,
                        kind: UnitKind::Char(chars[i]),
                    });
                    i += 1;
                }
            }
        }
        units
    }

    /// The kana `pending` may still become: the first of each rule it starts.
    pub fn pending_kana(&self, pending: &str) -> Vec<char> {
        let pending = if self.case_sensitive {
            pending.to_string()
        } else {
            pending.to_ascii_lowercase()
        };
        let mut out: Vec<char> = self
            .rules
            .iter()
            .filter(|(k, _)| k.starts_with(&pending))
            .filter_map(|(_, (o, _))| o.chars().next())
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

static ACTIVE: RwLock<Option<Arc<RomajiTable>>> = RwLock::new(None);
/// Tables read so far, by their text (their outputs are leaked: read each once).
static LOADED: Mutex<Option<HashMap<String, Arc<RomajiTable>>>> = Mutex::new(None);

/// The table in effect (`None`: the built-in romaji).
pub fn active() -> Option<Arc<RomajiTable>> {
    ACTIVE.read().ok()?.clone()
}

/// The text of `spec`: a preset's name, or a file's path ("standard" and ""
/// are the built-in table: `None`).
pub fn text_of(spec: &str) -> Result<Option<String>, String> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "standard" {
        return Ok(None);
    }
    if let Some((_, text)) = PRESETS.iter().find(|(name, _)| *name == spec) {
        return Ok(Some(text.to_string()));
    }
    std::fs::read_to_string(spec)
        .map(Some)
        .map_err(|e| format!("{spec}: {e}"))
}

/// Make `spec` the table in effect; on an error the built-in table is.
pub fn configure(spec: &str) -> Result<(), String> {
    let table = match text_of(spec) {
        Ok(Some(text)) => {
            let mut loaded = LOADED.lock().map_err(|e| e.to_string())?;
            let cache = loaded.get_or_insert_with(HashMap::new);
            match cache.get(&text) {
                Some(t) => Ok(Some(t.clone())),
                None => match RomajiTable::parse(&text) {
                    Ok(t) => {
                        let t = Arc::new(t);
                        cache.insert(text, t.clone());
                        Ok(Some(t))
                    }
                    Err(errors) => Err(errors.join("; ")),
                },
            }
        }
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    };
    let (table, result) = match table {
        Ok(t) => (t, Ok(())),
        Err(e) => (None, Err(e)),
    };
    if let Ok(mut a) = ACTIVE.write() {
        *a = table;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(t: &RomajiTable, typed: &str, flush: bool) -> String {
        let chars: Vec<char> = typed.chars().collect();
        t.units(&chars, flush)
            .iter()
            .map(|u| match u.kind {
                UnitKind::Kana(k) => k.to_string(),
                UnitKind::Char(c) => c.to_string(),
                UnitKind::Pending => {
                    format!("[{}]", chars[u.start..u.end].iter().collect::<String>())
                }
            })
            .collect()
    }

    #[test]
    fn azik_reads_its_shortcuts() {
        let t = RomajiTable::parse(PRESETS[0].1).unwrap();
        assert_eq!(read(&t, "kzsr", true), "かんする");
        assert_eq!(read(&t, "ds", true), "です");
        assert_eq!(read(&t, "ka;ta", true), "かった");
        assert!(t.uses(';'));
        // a start of a longer input waits at the end
        assert!(read(&t, "k", false).starts_with('['));
    }

    #[test]
    fn kana_input_reads_keys_and_voiced_marks() {
        let t = RomajiTable::parse(PRESETS[1].1).unwrap();
        assert!(t.kana_keys && t.case_sensitive());
        // t か, t@ が, 3 あ, E ぃ
        assert_eq!(read(&t, "t@3", true), "があ");
        assert_eq!(read(&t, "fE", true), "はぃ");
        assert_eq!(read(&t, "f[", true), "ぱ");
        // か may still take a ゛
        assert_eq!(read(&t, "t", false), "[t]");
        assert_eq!(read(&t, "t", true), "か");
    }

    #[test]
    fn mozc_tables_with_pending_text() {
        let t = RomajiTable::parse("ka\tか\nkk\tっ\tk\nn\tん\nna\tな\n").unwrap();
        assert_eq!(read(&t, "kka", true), "っか");
        assert_eq!(read(&t, "na", true), "な");
        assert_eq!(read(&t, "n", false), "[n]");
        assert_eq!(read(&t, "n", true), "ん");
        assert!(RomajiTable::parse("ka\tか\tx\n").is_err());
    }
}
