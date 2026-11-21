//! Setsuna analyzer: morphological analysis, text -> words with their
//! readings, and nothing else.
//!
//! A vibrato system dictionary (a MeCab-compatible lattice of dictionary
//! words, and unknown words grouped by character class, with a CRF's word
//! and connection costs) named by an analyzer manifest cuts a text into its
//! cheapest path of words; each word carries the reading and part of speech
//! its dictionary entry gives. With the manifest's alternates table, each
//! word also carries the other readings the dictionary holds for its surface
//! and how much more the path costs with each in its place. What the
//! readings are used for (a reading for text the user did not type, words to
//! learn, a check of a conversion's candidates) is the host's.
//!
//! The dictionary is upstream vibrato's (`vibrato-zstd`), or a packed one
//! (`vibrato-packed-zstd`): the same analyses in a fraction of the memory,
//! read with `vendor/vibrato-packed`, without alternates.
//!
//! Modules:
//!   - alternates: the other readings' table, built from the trained lexicon
//!   - packed: packed dictionaries, their codes and their build

pub mod alternates;
pub mod packed;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use setsuna_artifact::AnalyzerManifest;

pub use alternates::Alternates;

/// Defaults where the manifest's `analyze_defaults` sets none.
pub const MAX_GROUPING_LEN: usize = 24;
pub const ALT_MAX_COST: i32 = 3000;
pub const MAX_ALTS: usize = 8;
/// The most bytes a dictionary may be once decompressed.
const MAX_DICTIONARY_BYTES: u64 = 1 << 30;

/// One word of an analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Token {
    pub surface: String,
    /// The surface's chars `[start, end)` in the text.
    pub start: usize,
    pub end: usize,
    /// Hiragana; none for a word the dictionary does not hold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reading: Option<String>,
    /// The part of speech (the feature column `pos1`, else the first).
    pub pos: String,
    /// Its subdivision (`pos2`, else the second column; `*` for none):
    /// 普通名詞, 係助詞, 非自立可能.
    pub pos2: String,
    /// Other readings and how much more the path costs with each (in the
    /// dictionary's cost units), cheapest first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub alts: Vec<(String, i32)>,
    /// The path's cost from the text's start through this word (the
    /// dictionary's units): the last word's is the analysis's.
    pub cost: i32,
}

/// Analysis settings: the manifest's, else the defaults above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyzeConfig {
    pub max_grouping_len: usize,
    pub alt_max_cost: i32,
    pub max_alts: usize,
}

/// The dictionary and how its words' features read.
enum Engine {
    Vibrato {
        tokenizer: vibrato::Tokenizer,
        alternates: Option<Alternates>,
        reading_col: usize,
        pos_cols: (usize, usize),
    },
    Packed {
        tokenizer: vibrato_packed::Tokenizer,
        codes: packed::Codes,
    },
}

pub struct Analyzer {
    engine: Engine,
    pub config: AnalyzeConfig,
}

impl Analyzer {
    pub fn load(manifest: &AnalyzerManifest) -> Result<Self> {
        let path = &manifest.dictionary_path;
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let d = &manifest.analyze_defaults;
        let config = AnalyzeConfig {
            max_grouping_len: d.max_grouping_len.unwrap_or(MAX_GROUPING_LEN),
            alt_max_cost: d.alt_max_cost.unwrap_or(ALT_MAX_COST),
            max_alts: d.max_alts.unwrap_or(MAX_ALTS),
        };
        let read = || format!("read dictionary {}", path.display());
        // no more than a dictionary may be once decompressed: a small file
        // that decompresses past it (damaged or crafted) ends there
        let unzip = |file| -> Result<_> {
            use std::io::Read;
            Ok(zstd::Decoder::new(file)?.take(MAX_DICTIONARY_BYTES))
        };
        let engine = match &manifest.packed {
            // the manifest's format, checked when it loaded
            Some(codes) => Engine::Packed {
                tokenizer: vibrato_packed::Tokenizer::new(
                    vibrato_packed::Dictionary::read(unzip(file)?).with_context(read)?,
                )
                .max_grouping_len(config.max_grouping_len),
                codes: packed::Codes::new(codes)?,
            },
            None => Engine::Vibrato {
                tokenizer: vibrato::Tokenizer::new(
                    vibrato::Dictionary::read(unzip(file)?).with_context(read)?,
                )
                .max_grouping_len(config.max_grouping_len),
                alternates: manifest
                    .alternates_path
                    .as_deref()
                    .map(Alternates::open)
                    .transpose()?,
                // checked by the manifest
                reading_col: manifest.feature("reading").unwrap_or(0),
                pos_cols: (
                    manifest.feature("pos1").unwrap_or(0),
                    manifest.feature("pos2").unwrap_or(1),
                ),
            },
        };
        Ok(Self { engine, config })
    }

    /// Load the analyzer a manifest file names.
    pub fn from_manifest_path(path: &Path) -> Result<Self> {
        let manifest = AnalyzerManifest::load(path)
            .with_context(|| format!("load analyzer manifest {}", path.display()))?;
        Self::load(&manifest)
    }

    /// A reusable worker holding the lattice scratch buffers for repeated
    /// [`AnalyzeWorker::analyze`] calls without per-sentence allocation.
    pub fn worker(&self) -> AnalyzeWorker<'_> {
        let inner = match &self.engine {
            Engine::Vibrato { tokenizer, .. } => WorkerInner::Vibrato(tokenizer.new_worker()),
            Engine::Packed { tokenizer, .. } => WorkerInner::Packed(tokenizer.new_worker()),
        };
        AnalyzeWorker {
            analyzer: self,
            inner,
        }
    }

    /// The words of `text`, in order.
    pub fn analyze(&self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            return Vec::new();
        }
        self.worker().analyze(text)
    }

    /// Analyze a batch of texts reusing one worker across the whole batch.
    pub fn analyze_many<S: AsRef<str>>(&self, texts: &[S]) -> Vec<Vec<Token>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut worker = self.worker();
        texts.iter().map(|t| worker.analyze(t.as_ref())).collect()
    }

    /// Token `i`'s other readings: each reading's cheapest entry, what the
    /// path costs more with it, cheapest first.
    fn alts(
        &self,
        a: &Alternates,
        worker: &vibrato::tokenizer::worker::Worker,
        i: usize,
        own: &str,
    ) -> Vec<(String, i32)> {
        let t = worker.token(i);
        let prev = if i > 0 {
            worker.token(i - 1).right_id()
        } else {
            0
        };
        let next = if i + 1 < worker.num_tokens() {
            worker.token(i + 1).left_id()
        } else {
            0
        };
        let base = i32::from(t.word_cost())
            + a.connection(prev, t.left_id())
            + a.connection(t.right_id(), next);
        let mut best: HashMap<&str, i32> = HashMap::new();
        for e in a.entries(t.surface()) {
            if e.reading == own {
                continue;
            }
            let d = e.cost + a.connection(prev, e.left) + a.connection(e.right, next) - base;
            if d <= self.config.alt_max_cost {
                let x = best.entry(e.reading).or_insert(d);
                *x = (*x).min(d);
            }
        }
        let mut out: Vec<(String, i32)> =
            best.into_iter().map(|(r, d)| (r.to_string(), d)).collect();
        out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(self.config.max_alts);
        out
    }
}

enum WorkerInner<'a> {
    Vibrato(vibrato::tokenizer::worker::Worker<'a>),
    Packed(vibrato_packed::tokenizer::worker::Worker<'a>),
}

/// Reusable analysis worker tied to an [`Analyzer`] (`Analyzer::worker`).
pub struct AnalyzeWorker<'a> {
    analyzer: &'a Analyzer,
    inner: WorkerInner<'a>,
}

impl AnalyzeWorker<'_> {
    /// The words of `text`, in order, reusing the worker's lattice buffers.
    pub fn analyze(&mut self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            return Vec::new();
        }
        match (&self.analyzer.engine, &mut self.inner) {
            (
                Engine::Vibrato {
                    alternates,
                    reading_col,
                    pos_cols,
                    ..
                },
                WorkerInner::Vibrato(worker),
            ) => {
                worker.reset_sentence(text);
                worker.tokenize();
                let max_col = (*reading_col).max(pos_cols.0).max(pos_cols.1);
                (0..worker.num_tokens())
                    .map(|i| {
                        let t = worker.token(i);
                        let (mut rd, mut p1, mut p2) = (None, "", "*");
                        for (idx, col) in t.feature().split(',').enumerate() {
                            if idx == *reading_col {
                                rd = Some(col);
                            }
                            if idx == pos_cols.0 {
                                p1 = col;
                            }
                            if idx == pos_cols.1 {
                                p2 = col;
                            }
                            if idx >= max_col {
                                break;
                            }
                        }
                        let reading = rd
                            .filter(|r| *r != "*" && !r.is_empty())
                            .map(|r| r.to_string());
                        // a symbol the analysis took as an unknown word can be
                        // one the dictionary reads (/ as すらっしゅ)
                        let alts = match alternates {
                            Some(a) => {
                                self.analyzer
                                    .alts(a, worker, i, reading.as_deref().unwrap_or("*"))
                            }
                            None => Vec::new(),
                        };
                        let range = t.range_char();
                        Token {
                            surface: t.surface().to_string(),
                            start: range.start,
                            end: range.end,
                            reading,
                            pos: p1.to_string(),
                            pos2: p2.to_string(),
                            alts,
                            cost: t.total_cost(),
                        }
                    })
                    .collect()
            }
            (Engine::Packed { codes, .. }, WorkerInner::Packed(worker)) => {
                worker.reset_sentence(text);
                worker.tokenize();
                (0..worker.num_tokens())
                    .map(|i| {
                        let t = worker.token(i);
                        let (pos, pos2, reading) = codes.read(t.surface(), t.feature());
                        let range = t.range_char();
                        Token {
                            surface: t.surface().to_string(),
                            start: range.start,
                            end: range.end,
                            reading,
                            pos,
                            pos2,
                            alts: Vec::new(),
                            cost: t.total_cost(),
                        }
                    })
                    .collect()
            }
            _ => unreachable!("worker engine matches analyzer engine"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A dictionary of a few words, its alternates and its manifest in a
    /// directory of its own.
    fn tiny(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("setsuna-analyzer-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lex = "今日,1,1,100,名詞,普通名詞,*,*,きょう,今日\n\
                   今日,1,1,1500,名詞,普通名詞,*,*,こんにち,今日\n\
                   は,2,2,50,助詞,係助詞,*,*,は,は\n\
                   晴れ,1,1,300,名詞,普通名詞,*,*,はれ,晴れ\n";
        let matrix =
            "3 3\n0 0 0\n0 1 0\n0 2 0\n1 0 0\n1 1 500\n1 2 -100\n2 0 0\n2 1 -100\n2 2 500\n";
        let char_def = "DEFAULT 0 1 0\nKANJI 0 0 2\nHIRAGANA 0 1 0\n\
                        0x3041..0x309F HIRAGANA\n0x4E00..0x9FFF KANJI\n";
        let unk = "DEFAULT,1,1,3000,名詞,一般,*,*,*,*\n\
                   KANJI,1,1,3000,名詞,一般,*,*,*,*\n\
                   HIRAGANA,1,1,3000,名詞,一般,*,*,*,*\n";
        let dict = vibrato::SystemDictionaryBuilder::from_readers(
            lex.as_bytes(),
            matrix.as_bytes(),
            char_def.as_bytes(),
            unk.as_bytes(),
        )
        .unwrap();
        let mut enc =
            zstd::Encoder::new(File::create(dir.join("system.dic.zst")).unwrap(), 3).unwrap();
        dict.write(&mut enc).unwrap();
        enc.finish().unwrap();
        std::fs::write(dir.join("lex.csv"), lex).unwrap();
        std::fs::write(dir.join("matrix.def"), matrix).unwrap();
        alternates::build(
            &dir.join("lex.csv"),
            &dir.join("matrix.def"),
            4,
            &dir.join("alternates.bin"),
        )
        .unwrap();
        std::fs::write(
            dir.join("analyzer.manifest.json"),
            r#"{"schema":"neo-ime-analyzer-export-v1","model_family":"lattice_crf",
            "dictionary":{"path":"system.dic.zst","format":"vibrato-zstd"},
            "alternates":"alternates.bin",
            "features":["pos1","pos2","cType","cForm","reading","orth"],
            "analyze_defaults":{"alt_max_cost":2000}}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn analyzes_with_readings_and_alternates() {
        let dir = tiny("ok");
        let a = Analyzer::from_manifest_path(&dir.join("analyzer.manifest.json")).unwrap();
        assert_eq!(a.config.alt_max_cost, 2000);
        assert_eq!(a.config.max_alts, MAX_ALTS);
        let t = a.analyze("今日は晴れ");
        let words: Vec<(&str, Option<&str>, usize, usize)> = t
            .iter()
            .map(|t| (t.surface.as_str(), t.reading.as_deref(), t.start, t.end))
            .collect();
        assert_eq!(
            words,
            [
                ("今日", Some("きょう"), 0, 2),
                ("は", Some("は"), 2, 3),
                ("晴れ", Some("はれ"), 3, 5)
            ]
        );
        assert_eq!(
            (t[0].pos.as_str(), t[0].pos2.as_str()),
            ("名詞", "普通名詞")
        );
        // こんにち: 1500 against 100, the same connections
        assert_eq!(t[0].alts, [("こんにち".to_string(), 1400)]);
        assert!(t[1].alts.is_empty());
        // a word the dictionary lacks has no reading
        let t = a.analyze("雨");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].reading, None);
        assert!(a.analyze("").is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_packed_dictionary_analyzes_as_the_vibrato_one() {
        let src = tiny("packed-src");
        // the same lexicon with its features as pos1, pos2, reading, and a
        // word read as it is written
        let lex = "今日,1,1,100,名詞,普通名詞,きょう\n\
                   今日,1,1,1500,名詞,普通名詞,こんにち\n\
                   は,2,2,50,助詞,係助詞,は\n\
                   晴れ,1,1,300,名詞,普通名詞,はれ\n\
                   ハレ,1,1,4000,名詞,普通名詞,はれ\n\
                   \"1,000\",1,1,200,名詞,数詞,せん\n";
        std::fs::write(src.join("lex3.csv"), lex).unwrap();
        std::fs::write(
            src.join("unk3.def"),
            "DEFAULT,1,1,3000,名詞,一般,*\nKANJI,1,1,3000,名詞,一般,*\nHIRAGANA,1,1,3000,名詞,一般,*\n",
        )
        .unwrap();
        std::fs::write(
            src.join("char.def"),
            "DEFAULT 0 1 0\nKANJI 0 0 2\nHIRAGANA 0 1 0\n0x3041..0x309F HIRAGANA\n0x4E00..0x9FFF KANJI\n",
        )
        .unwrap();
        let input = |top: Option<usize>, usage: Option<PathBuf>| packed::BuildInput {
            lex: src.join("lex3.csv"),
            unk: src.join("unk3.def"),
            char_def: src.join("char.def"),
            connection: src.join("matrix.def"),
            features: ["pos1", "pos2", "reading"].map(String::from).to_vec(),
            usage,
            top,
        };
        let out = src.join("packed");
        let built = packed::build(&input(None, None), &out).unwrap();
        assert_eq!(built.words, 6);
        let plain = Analyzer::from_manifest_path(&src.join("analyzer.manifest.json")).unwrap();
        let a = Analyzer::from_manifest_path(&built.manifest).unwrap();
        let strip = |t: Vec<Token>| -> Vec<Token> {
            t.into_iter()
                .map(|t| Token {
                    alts: Vec::new(),
                    ..t
                })
                .collect()
        };
        let texts = ["今日は晴れ", "", "雨", "今日", "はれ"];
        for text in texts {
            assert_eq!(strip(plain.analyze(text)), a.analyze(text), "{text}");
        }
        assert_eq!(
            plain.analyze_many(&texts),
            texts.iter().map(|t| plain.analyze(t)).collect::<Vec<_>>()
        );
        assert_eq!(
            a.analyze_many(&texts),
            texts.iter().map(|t| a.analyze(t)).collect::<Vec<_>>()
        );
        let t = a.analyze("1,000はハレ");
        assert_eq!(t[0].reading.as_deref(), Some("せん"));
        assert_eq!((t[0].pos.as_str(), t[0].pos2.as_str()), ("名詞", "数詞"));
        assert_eq!(t[2].reading.as_deref(), Some("はれ"));
        // the words the usage counts most, and those only
        std::fs::write(
            src.join("usage.tsv"),
            "晴れ\t名詞\t普通名詞\tはれ\t9\nは\t助詞\t係助詞\tは\t5\n",
        )
        .unwrap();
        let built = packed::build(
            &input(Some(2), Some(src.join("usage.tsv"))),
            &src.join("top"),
        )
        .unwrap();
        assert_eq!(built.words, 2);
        let a = Analyzer::from_manifest_path(&built.manifest).unwrap();
        let words: Vec<(String, Option<String>)> = a
            .analyze("今日は晴れ")
            .into_iter()
            .map(|t| (t.surface, t.reading))
            .collect();
        assert_eq!(
            words[1..],
            [
                ("は".into(), Some("は".into())),
                ("晴れ".into(), Some("はれ".into()))
            ]
        );
        assert_eq!(words[0].1, None);
        std::fs::remove_dir_all(&src).ok();
    }
}
