//! Symbol dictionary: readings that name symbols (やじるし -> →, かっこ ->
//! （）). The model converts those readings to words (矢印, 括弧) or to junk,
//! and candidates with symbols the user did not type are dropped anyway
//! (`postprocess::drop_untyped_symbols`), so the symbols come from here.
//!
//! The built-in table is `data/symbols.tsv`. A reading that names only a
//! symbol (`first`) gets its symbols before the model's candidates; one that
//! is also an ordinary word (`after`, まる / 丸) gets them after the model's
//! top two and the kana forms, so the word stays on top. Only a whole reading
//! is looked up.
//!
//! A larger dictionary (`[frontend] symbol_dictionary_file`, JSON lines with
//! `reading` and `symbol`, the format of the distributed symbol dictionary)
//! is merged into it. Its symbols for a reading the table has go at the end
//! of the list; a reading only it has is placed `after`, with its first
//! `FILE_INLINE` symbols there and the rest at the end, so a reading with
//! hundreds of symbols (だんせい, かお) keeps the model's candidates near the
//! top.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};

const TABLE: &str = include_str!("../data/symbols.tsv");

/// Symbols of a reading only the file has, listed at its place; the others
/// go at the end of the list.
const FILE_INLINE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    First,
    After,
}

#[derive(Debug, Clone)]
struct Entry {
    place: Place,
    /// From the built-in table, whose symbols are all listed at `place`.
    curated: bool,
    /// Listed at `place`.
    inline: Vec<String>,
    /// Listed at the end of the list.
    rest: Vec<String>,
}

impl Entry {
    fn has(&self, symbol: &str) -> bool {
        self.inline.iter().chain(&self.rest).any(|s| s == symbol)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SymbolDict {
    map: HashMap<String, Entry>,
}

/// One line of the file; other fields (desc, kind, source) are ignored.
#[derive(serde::Deserialize)]
struct FileLine {
    reading: String,
    symbol: String,
}

impl SymbolDict {
    /// The built-in table.
    pub fn builtin() -> Self {
        let map = TABLE
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let mut f = l.split('\t');
                let (reading, place, symbols) = (f.next()?, f.next()?, f.next()?);
                let place = match place {
                    "first" => Place::First,
                    "after" => Place::After,
                    _ => return None,
                };
                let entry = Entry {
                    place,
                    curated: true,
                    inline: symbols
                        .split(' ')
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                    rest: Vec::new(),
                };
                Some((reading.to_string(), entry))
            })
            .collect();
        Self { map }
    }

    /// The built-in table with the dictionary file at `path` merged in.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut dict = Self::builtin();
        dict.merge_jsonl(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(dict)
    }

    /// Merge JSON lines of `{"reading", "symbol"}`, in their order.
    fn merge_jsonl(&mut self, text: &str) -> Result<()> {
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let l: FileLine =
                serde_json::from_str(line).with_context(|| format!("line {}", n + 1))?;
            if l.reading.is_empty() || l.symbol.is_empty() {
                continue;
            }
            let e = self.map.entry(l.reading).or_insert_with(|| Entry {
                place: Place::After,
                curated: false,
                inline: Vec::new(),
                rest: Vec::new(),
            });
            if e.has(&l.symbol) {
                continue;
            }
            if !e.curated && e.inline.len() < FILE_INLINE {
                e.inline.push(l.symbol);
            } else {
                e.rest.push(l.symbol);
            }
        }
        Ok(())
    }

    /// `candidates` (after `kana_candidates::arrange`) with the symbols
    /// `reading` names inserted, or unchanged when it names none.
    pub fn insert(&self, reading: &str, candidates: Vec<String>) -> Vec<String> {
        let Some(e) = self.map.get(reading) else {
            return candidates;
        };
        let mut out: Vec<String> = candidates.into_iter().filter(|c| !e.has(c)).collect();
        let at = match e.place {
            Place::First => 0,
            Place::After => {
                // right after the kana forms, which `arrange` put 3rd / 4th (or
                // left in the model's top two)
                let kana = [
                    reading.to_string(),
                    crate::kana_candidates::to_katakana(reading),
                ];
                out.iter()
                    .rposition(|c| kana.contains(c))
                    .map_or(out.len().min(4), |i| i + 1)
            }
        };
        out.splice(at..at, e.inline.iter().cloned());
        out.extend(e.rest.iter().cloned());
        out
    }
}

/// The dictionary for `file` (`None`: the built-in table), loaded once per
/// process; the built-in table (logged) when the file cannot be read.
pub fn shared(file: Option<&str>) -> Arc<SymbolDict> {
    static CACHE: OnceLock<Mutex<HashMap<Option<String>, Arc<SymbolDict>>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(file.map(String::from))
        .or_insert_with(|| {
            let dict = match file {
                None => SymbolDict::builtin(),
                Some(path) => SymbolDict::load(Path::new(path)).unwrap_or_else(|e| {
                    crate::diagnostics::log(format!("symbol dictionary {path}: {e:#}"));
                    SymbolDict::builtin()
                }),
            };
            Arc::new(dict)
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    fn builtin() -> &'static SymbolDict {
        static D: OnceLock<SymbolDict> = OnceLock::new();
        D.get_or_init(SymbolDict::builtin)
    }

    fn with_file(lines: &[(&str, &str)]) -> SymbolDict {
        let text: String = lines
            .iter()
            .map(|(r, s)| format!("{{\"reading\":\"{r}\",\"symbol\":\"{s}\",\"kind\":\"x\"}}\n"))
            .collect();
        let mut d = SymbolDict::builtin();
        d.merge_jsonl(&text).unwrap();
        d
    }

    #[test]
    fn the_table_parses() {
        let map = &builtin().map;
        assert!(map.len() > 50);
        assert!(!map.contains_key("きごう"));
        assert!(map
            .values()
            .all(|e| !e.inline.is_empty() && e.rest.is_empty()));
    }

    #[test]
    fn a_symbol_name_puts_its_symbols_first() {
        let got = builtin().insert("やじるし", v(&["矢印", "矢印し", "やじるし", "ヤジルシ"]));
        assert_eq!(&got[..3], &v(&["→", "←", "↑"])[..]);
        assert_eq!(got[8], "矢印");
    }

    #[test]
    fn a_word_keeps_the_top_and_gets_symbols_after_the_kana() {
        let got = builtin().insert("まる", v(&["丸", "円", "まる", "マル", "丸る"]));
        assert_eq!(
            got,
            v(&["丸", "円", "まる", "マル", "○", "●", "◎", "◯", "丸る"])
        );
        // a symbol the model already had is not repeated
        let got = builtin().insert("ほし", v(&["星", "☆", "ほし", "ホシ"]));
        assert_eq!(got, v(&["星", "ほし", "ホシ", "☆", "★"]));
    }

    #[test]
    fn other_readings_are_left_alone() {
        let c = v(&["記号", "号", "きごう", "キゴウ"]);
        assert_eq!(builtin().insert("きごう", c.clone()), c);
        let c = v(&["矢印を書く"]);
        assert_eq!(builtin().insert("やじるしをかく", c.clone()), c);
    }

    #[test]
    fn the_file_adds_to_a_table_reading_at_the_end() {
        let d = with_file(&[("まる", "○"), ("まる", "⭕"), ("まる", "🔴")]);
        let got = d.insert("まる", v(&["丸", "円", "まる", "マル", "丸る"]));
        assert_eq!(
            got,
            v(&[
                "丸", "円", "まる", "マル", "○", "●", "◎", "◯", "丸る", "⭕", "🔴"
            ])
        );
    }

    #[test]
    fn a_file_reading_lists_a_few_after_the_kana_and_the_rest_last() {
        let d = with_file(&[
            ("いぬ", "🐕"),
            ("いぬ", "🐶"),
            ("いぬ", "🐩"),
            ("いぬ", "🐶"),
            ("いぬ", "🦮"),
            ("いぬ", "🐾"),
            ("いぬ", "🦴"),
        ]);
        let got = d.insert("いぬ", v(&["犬", "戌", "いぬ", "イヌ", "狗"]));
        assert_eq!(
            got,
            v(&[
                "犬", "戌", "いぬ", "イヌ", "🐕", "🐶", "🐩", "🦮", "狗", "🐾", "🦴"
            ])
        );
    }

    #[test]
    fn a_bad_line_fails_the_file() {
        let mut d = SymbolDict::builtin();
        let err = d
            .merge_jsonl("{\"reading\":\"いぬ\",\"symbol\":\"🐕\"}\nnot json\n")
            .unwrap_err();
        assert!(format!("{err:#}").contains("line 2"), "{err:#}");
    }

    #[test]
    fn an_unreadable_file_falls_back_to_the_table() {
        let d = shared(Some("Z:/no/such/symbols.jsonl"));
        assert!(d.map.contains_key("やじるし"));
    }
}
