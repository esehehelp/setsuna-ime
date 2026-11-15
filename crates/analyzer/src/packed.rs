//! Packed dictionaries (`vibrato-packed-zstd`, `vendor/vibrato-packed`): the
//! same lattice and costs as a vibrato one in a fraction of the memory, the
//! words sorted by surface (no posting lists) and each word's features
//! packed into a few ASCII chars that the manifest's codes read
//! (`setsuna_artifact::PackedCodes`): one for the part of speech, then the
//! reading, `=` (the surface in hiragana), `*` (none), `~` and the reading
//! as it is, or one char a kana.
//!
//! [`build`] writes one from a trained analyzer's lexicon and connection
//! costs, optionally with only the words a usage count ranks first.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use setsuna_artifact::PackedCodes;

/// The code chars, in order: printable ASCII but `"` and `,` (which a
/// lexicon line quotes or splits on) and the markers `=`, `*`, `~`.
pub const CODE_CHARS: &str =
    "!#$%&'()+-./0123456789:;<>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}";

fn code_index(c: char) -> Option<usize> {
    CODE_CHARS.find(c)
}

fn code_char(i: usize) -> Option<char> {
    CODE_CHARS.chars().nth(i)
}

/// Katakana as hiragana, the rest as it is.
pub fn hiragana(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'ァ'..='ヶ' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// A packed dictionary's codes, to read a word's features.
#[derive(Debug, Clone)]
pub struct Codes {
    pos: Vec<(String, String)>,
    kana: Vec<char>,
}

impl Codes {
    pub fn new(codes: &PackedCodes) -> Result<Self> {
        let kana: Vec<char> = codes.kana.chars().collect();
        let n = CODE_CHARS.chars().count();
        ensure!(
            codes.pos.len() <= n && kana.len() <= n,
            "packed codes: {} parts of speech and {} kana, {n} codes at most",
            codes.pos.len(),
            kana.len()
        );
        Ok(Self {
            pos: codes
                .pos
                .iter()
                .map(|[a, b]| (a.clone(), b.clone()))
                .collect(),
            kana,
        })
    }

    /// The (pos1, pos2, reading) a word `surface` with packed `feature`
    /// has; an unknown code reads as no part of speech.
    pub fn read(&self, surface: &str, feature: &str) -> (String, String, Option<String>) {
        let mut chars = feature.chars();
        let (pos1, pos2) = chars
            .next()
            .and_then(code_index)
            .and_then(|i| self.pos.get(i))
            .cloned()
            .unwrap_or_else(|| (String::new(), "*".to_string()));
        let rest = chars.as_str();
        let reading = match rest {
            "" | "*" => None,
            "=" => Some(hiragana(surface)),
            _ => match rest.strip_prefix('~') {
                Some(raw) => Some(raw.to_string()),
                None => rest
                    .chars()
                    .map(|c| code_index(c).and_then(|i| self.kana.get(i).copied()))
                    .collect(),
            },
        };
        (pos1, pos2, reading)
    }
}

/// One lexicon line: the surface (unquoted), then the fields after it.
fn split_line(line: &str) -> Result<(String, Vec<&str>)> {
    let (surface, rest) = match line.strip_prefix('"') {
        Some(q) => {
            let end = q
                .find("\",")
                .with_context(|| format!("unclosed quote: {line}"))?;
            (q[..end].replace("\"\"", "\""), &q[end + 2..])
        }
        None => {
            let end = line
                .find(',')
                .with_context(|| format!("no fields: {line}"))?;
            (line[..end].to_string(), &line[end + 1..])
        }
    };
    Ok((surface, rest.split(',').collect()))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// A lexicon or unknown-word entry: its key (surface or char class), the
/// ids and cost as written, and its part of speech and reading.
struct Entry {
    key: String,
    params: String,
    pos: (String, String),
    reading: String,
}

/// Entries of a lexicon or unk.def whose features are the columns
/// `features` names (pos1, pos2 and reading among them).
fn read_entries(path: &Path, features: &[String]) -> Result<Vec<Entry>> {
    let col = |name: &str| {
        features
            .iter()
            .position(|f| f == name)
            .with_context(|| format!("features name no {name}"))
    };
    let (p1, p2, rd) = (col("pos1")?, col("pos2")?, col("reading")?);
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let (key, f) = split_line(line)?;
        let get = |i: usize| -> Result<String> {
            f.get(3 + i)
                .map(|s| s.to_string())
                .with_context(|| format!("{}:{}: too few fields", path.display(), n + 1))
        };
        ensure!(
            f.len() >= 3,
            "{}:{}: no ids and cost",
            path.display(),
            n + 1
        );
        out.push(Entry {
            key,
            params: f[..3].join(","),
            pos: (get(p1)?, get(p2)?),
            reading: get(rd)?,
        });
    }
    Ok(out)
}

/// How often an analyzer used each (surface, pos1, pos2, reading): lines
/// `surface TAB pos1 TAB pos2 TAB reading TAB count` (a surface may hold a
/// tab: the fields are split from the right).
pub fn read_usage(path: &Path) -> Result<HashMap<(String, String, String, String), u64>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let f: Vec<&str> = line.trim_end_matches('\r').rsplitn(5, '\t').collect();
        if f.len() < 5 {
            continue;
        }
        let key = (
            f[4].to_string(),
            f[3].to_string(),
            f[2].to_string(),
            f[1].to_string(),
        );
        *out.entry(key).or_insert(0) += f[0].parse::<u64>().context("usage count")?;
    }
    Ok(out)
}

/// What [`build`] reads.
pub struct BuildInput {
    /// The trained lexicon: surface, left id, right id, cost, features.
    pub lex: PathBuf,
    /// Its unknown-word entries, features as in `lex`.
    pub unk: PathBuf,
    pub char_def: PathBuf,
    /// The connection costs: a matrix.def, or a directory of the model's
    /// bigram.right / .left / .cost (vibrato's raw connector).
    pub connection: PathBuf,
    /// The names of the features' columns (pos1, pos2 and reading among
    /// them).
    pub features: Vec<String>,
    /// Keep only the `top` words `usage` ranks first ([`read_usage`]; a
    /// word it lacks counts 0, ties keep the lexicon's order).
    pub usage: Option<PathBuf>,
    pub top: Option<usize>,
}

/// What [`build`] wrote.
#[derive(Debug)]
pub struct Built {
    pub words: usize,
    pub bytes: u64,
    pub manifest: PathBuf,
}

/// Write a packed dictionary and its manifest (`system.dic.zst`,
/// `analyzer.manifest.json`) into `out`.
pub fn build(input: &BuildInput, out: &Path) -> Result<Built> {
    let mut words = read_entries(&input.lex, &input.features)?;
    let unk = read_entries(&input.unk, &input.features)?;
    if let Some(top) = input.top {
        if let Some(usage) = &input.usage {
            let use_of = read_usage(usage)?;
            let count = |e: &Entry| {
                use_of
                    .get(&(
                        e.key.clone(),
                        e.pos.0.clone(),
                        e.pos.1.clone(),
                        e.reading.clone(),
                    ))
                    .copied()
                    .unwrap_or(0)
            };
            // stable: ties keep the lexicon's order
            words.sort_by_key(|e| std::cmp::Reverse(count(e)));
        }
        words.truncate(top);
    }
    // a surface's words follow one another: the map needs no postings
    words.sort_by(|a, b| a.key.cmp(&b.key));

    let pos: Vec<(String, String)> = words
        .iter()
        .chain(&unk)
        .map(|e| e.pos.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let kana: Vec<char> = words
        .iter()
        .flat_map(|e| e.reading.chars())
        .filter(|&c| matches!(c, '\u{3041}'..='\u{309f}' | 'ー'))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let n = CODE_CHARS.chars().count();
    ensure!(pos.len() <= n, "{} parts of speech, {n} codes", pos.len());
    // the commonest kana coded when there are more than codes (the rest raw)
    let kana: Vec<char> = if kana.len() > n {
        let mut freq: HashMap<char, usize> = HashMap::new();
        for c in words.iter().flat_map(|e| e.reading.chars()) {
            *freq.entry(c).or_insert(0) += 1;
        }
        let mut k = kana;
        k.sort_by_key(|c| std::cmp::Reverse(freq[c]));
        k.truncate(n);
        k.sort();
        k
    } else {
        kana
    };
    let pos_code: HashMap<&(String, String), char> = pos
        .iter()
        .enumerate()
        .map(|(i, p)| (p, code_char(i).unwrap()))
        .collect();
    let kana_code: HashMap<char, char> = kana
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, code_char(i).unwrap()))
        .collect();
    let feature = |e: &Entry, surface: Option<&str>| -> String {
        let r = &e.reading;
        let reading = if r.is_empty() || r == "*" {
            "*".to_string()
        } else if surface.is_some_and(|s| hiragana(s) == *r) {
            "=".to_string()
        } else if let Some(coded) = r
            .chars()
            .map(|c| kana_code.get(&c))
            .collect::<Option<String>>()
        {
            coded
        } else {
            format!("~{r}")
        };
        format!("{}{reading}", pos_code[&e.pos])
    };
    let mut lex = String::new();
    for e in &words {
        lex.push_str(&format!(
            "{},{},{}\n",
            csv_field(&e.key),
            e.params,
            csv_field(&feature(e, Some(&e.key)))
        ));
    }
    let mut unk_def = String::new();
    for e in &unk {
        unk_def.push_str(&format!("{},{},{}\n", e.key, e.params, feature(e, None)));
    }

    let char_def = File::open(&input.char_def)
        .with_context(|| format!("open {}", input.char_def.display()))?;
    let dict = if input.connection.is_dir() {
        let open = |n: &str| -> Result<File> {
            let p = input.connection.join(n);
            File::open(&p).with_context(|| format!("open {}", p.display()))
        };
        vibrato_packed::SystemDictionaryBuilder::from_readers_with_bigram_info(
            lex.as_bytes(),
            open("bigram.right")?,
            open("bigram.left")?,
            open("bigram.cost")?,
            char_def,
            unk_def.as_bytes(),
            false,
        )
    } else {
        let matrix = File::open(&input.connection)
            .with_context(|| format!("open {}", input.connection.display()))?;
        vibrato_packed::SystemDictionaryBuilder::from_readers(
            lex.as_bytes(),
            matrix,
            char_def,
            unk_def.as_bytes(),
        )
    }
    .context("build the dictionary")?;

    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let dic = out.join("system.dic.zst");
    let mut enc = zstd::Encoder::new(File::create(&dic)?, 19)?;
    dict.write(&mut enc)?;
    enc.finish()?.flush()?;
    let manifest = out.join("analyzer.manifest.json");
    let body = serde_json::json!({
        "schema": setsuna_artifact::ANALYZER_SCHEMA,
        "model_family": setsuna_artifact::ANALYZER_FAMILY,
        "dictionary": {"path": "system.dic.zst", "format": setsuna_artifact::DICTIONARY_PACKED},
        "packed": {
            "pos": pos.iter().map(|(a, b)| [a, b]).collect::<Vec<_>>(),
            "kana": kana.iter().collect::<String>(),
        },
        "analyze_defaults": {"max_grouping_len": crate::MAX_GROUPING_LEN},
    });
    std::fs::write(&manifest, serde_json::to_string_pretty(&body)? + "\n")?;
    Ok(Built {
        words: words.len(),
        bytes: std::fs::metadata(&dic)?.len(),
        manifest,
    })
}

/// Counts of the words `analyzer` gives `texts`, most used first, for
/// [`BuildInput::usage`].
pub fn count_usage<'a>(
    analyzer: &crate::Analyzer,
    texts: impl IntoIterator<Item = &'a str>,
    out: &Path,
) -> Result<usize> {
    let mut count: HashMap<(String, String, String, String), u64> = HashMap::new();
    for text in texts {
        for t in analyzer.analyze(text) {
            let reading = t.reading.unwrap_or_else(|| "*".to_string());
            *count
                .entry((t.surface, t.pos, t.pos2, reading))
                .or_insert(0) += 1;
        }
    }
    let mut rows: Vec<_> = count.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut w = std::io::BufWriter::new(File::create(out)?);
    for ((s, p1, p2, r), n) in &rows {
        writeln!(w, "{s}\t{p1}\t{p2}\t{r}\t{n}")?;
    }
    w.flush()?;
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_chars_leave_out_the_markers_and_csv_chars() {
        assert_eq!(CODE_CHARS.chars().count(), 89);
        assert!(!CODE_CHARS.contains(['"', ',', '=', '*', '~', ' ']));
        let codes = Codes::new(&PackedCodes {
            pos: vec![
                ["名詞".into(), "普通名詞".into()],
                ["助詞".into(), "係助詞".into()],
            ],
            kana: "きょう".into(),
        })
        .unwrap();
        let p = |s: &str| (s.to_string());
        assert_eq!(
            codes.read("今日", "!!#$"),
            (p("名詞"), p("普通名詞"), Some(p("きょう")))
        );
        assert_eq!(
            codes.read("ハ", "#="),
            (p("助詞"), p("係助詞"), Some(p("は")))
        );
        assert_eq!(codes.read("x", "!*"), (p("名詞"), p("普通名詞"), None));
        assert_eq!(
            codes.read("・", "!~てん"),
            (p("名詞"), p("普通名詞"), Some(p("てん")))
        );
    }

    #[test]
    fn lines_split_with_quoted_surfaces() {
        let (s, f) = split_line("\"1,000円\",5,5,100,名詞,数詞,せんえん").unwrap();
        assert_eq!(s, "1,000円");
        assert_eq!(f, ["5", "5", "100", "名詞", "数詞", "せんえん"]);
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("ab"), "ab");
    }
}
