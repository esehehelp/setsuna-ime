//! Benchmark of mixed Japanese / English romaji input: what is committed
//! when a line is typed and Enter pressed, Japanese left as hiragana.
//!
//! ```text
//! mixed_input prepare --out <cases.jsonl> [--corpus <reading jsonl> --corpus-n 2000]
//!                     [--mixed <reading jsonl> --mixed-n 1500] [--vocab <latin.tsv>]
//! mixed_input run --cases <cases.jsonl> --out <predictions.jsonl>
//!                 [--model <dir> [--names <latin-names.tsv>] [--param name=value]... [--bridge]]
//! mixed_input tune --cases <dev cases.jsonl> --model <dir> [--names <tsv>] [--fix name]... [--write]
//! mixed_input live --cases <cases.jsonl> --model <dir> [--names <tsv>] [--show 20]
//! mixed_input trace <typed> [--before <text>] [--no-mixed | --model <dir>]
//! mixed_input score --cases <cases.jsonl> --predictions <predictions.jsonl> [--show 20]
//! ```
//!
//! `prepare` writes the hand-made cases of `bench/mixed_input.tsv` and, with
//! `--corpus`, Japanese-only cases typed from the `reading` of a corpus
//! (hiragana with 、。 only): how often plain Japanese comes out with ASCII
//! in it. With `--mixed`, cases typed from readings that keep Latin words
//! and digits as written (Japanese sentences with English words, the
//! Japanese read by a morphological analyzer): once as written, capitals
//! typed with Shift, and once all lowercase. A case where a Latin word's
//! last letter runs into the romaji after it (`5s` + は typed `5sha`,
//! `Japan` + の typed `Japanno`) is put in a `-fused` category: the keys do
//! not tell where the word ends.
//! A case is `{"id", "cat", "typed", "expected", "before"?}`.
//! `run` types each case into the engine bridge without an engine, or with
//! `--model` segments it with `setsuna_core::mixed` (Japanese as kana);
//! `--bridge` types it into the bridge with that model as
//! `[frontend] mixed_model`.
//! `tune` searches the model's parameters on dev cases, one at a time, for
//! the best mean accuracy on the mixed categories (`web-mixed*` but the
//! fused ones, `multi-*-mixed*`) less ten times the error rate on
//! `corpus-ja` and `multi-ja` (Japanese turned English is worse than
//! English left as kana); `--fix` keeps a parameter as it is (`known`:
//! new names are rarer in the dev cases than in typing); `--write` saves
//! them as the model's `params.txt`.
//! `live` checks what typing shows: for each case whose expected text has
//! no Latin letter, whether some preedit on the way showed letters other
//! than the romaji still being typed (`。ko` while typing これ).
//! `trace` types into the bridge with the user's config and its engine,
//! printing the preedit after each key (waiting for the engine) and what
//! Enter commits.
//! With `--vocab` (a mixed model's `latin.tsv`), a web case with a Latin
//! word the model has not seen goes to a `-oov` category: how it does on
//! new names.
//! With both, `multi-*` cases type two sentences in one composition: two
//! Japanese ones, and a Japanese one and a mixed one either way round.
//! `score` compares any system's `{"id", "actual"}` lines with the cases,
//! per category: with the case (a lowercase case's `cased`, its English as
//! written, as a names dictionary would give it) and ignoring it (where the
//! English is, whatever its case). `--names` adds a names dictionary to the
//! model.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use setsuna_core::engine_bridge::{Action, EngineBridge};
use setsuna_core::mixed::{self, MixedModel, MixedParams};

const HAND_MADE: &str = include_str!("../bench/mixed_input.tsv");

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Case {
    id: String,
    cat: String,
    typed: String,
    expected: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    before: Option<String>,
    /// A case typed in lowercase: what it should show, its English cased as
    /// written (`expected` is the same in lowercase).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cased: Option<String>,
}

impl Case {
    /// What the case should show, case and all.
    fn truth(&self) -> &str {
        self.cased.as_deref().unwrap_or(&self.expected)
    }
}

/// What `model` shows for `typed` once typing is done, cased.
fn shown(model: &MixedModel, typed: &str, before: Option<&str>) -> String {
    let (chars, spans) = model.cased(typed, before, true);
    let (text, pending) = mixed::render_parts(&chars, &spans, true, &HashMap::new());
    text + &pending
}

#[derive(Debug, Serialize, Deserialize)]
struct Prediction {
    id: String,
    actual: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let need = |name: &str| flag(name).ok_or_else(|| anyhow!("{name} is required"));
    match args.first().map(String::as_str) {
        Some("prepare") => {
            let mut cases = hand_made()?;
            let mut corpus = Vec::new();
            if let Some(path) = flag("--corpus") {
                let n = flag("--corpus-n").map_or(Ok(2000), |s| s.parse())?;
                corpus = corpus_cases(&path, n)?;
            }
            let mut mixed = Vec::new();
            if let Some(path) = flag("--mixed") {
                let n = flag("--mixed-n").map_or(Ok(1500), |s| s.parse())?;
                mixed = mixed_cases(&path, n)?;
            }
            if let Some(path) = flag("--vocab") {
                tag_oov(
                    &mut mixed,
                    &std::fs::read_to_string(&path).with_context(|| path.clone())?,
                );
            }
            let multi = multi_cases(&corpus, &mixed);
            cases.extend(corpus);
            cases.extend(mixed);
            cases.extend(multi);
            write_jsonl(&need("--out")?, &cases)?;
            eprintln!("{} cases", cases.len());
        }
        Some("run") => {
            let cases: Vec<Case> = read_jsonl(&need("--cases")?)?;
            let mut frontend = setsuna_core::config::FrontendCfg::default();
            let bridge = args.iter().any(|a| a == "--bridge");
            if bridge {
                frontend.mixed_model = Some(need("--model")?);
                frontend.name_dictionary_file = flag("--names");
            }
            let model = flag("--model")
                .filter(|_| !bridge)
                .map(|dir| load_model(&dir, &args))
                .transpose()?;
            let started = std::time::Instant::now();
            let predictions: Vec<Prediction> = cases
                .iter()
                .map(|c| Prediction {
                    id: c.id.clone(),
                    actual: match &model {
                        Some(model) => shown(model, &c.typed, c.before.as_deref()),
                        None => type_line(&c.typed, c.before.as_deref(), &frontend),
                    },
                })
                .collect();
            eprintln!(
                "{:.2} ms per case",
                started.elapsed().as_secs_f64() * 1000.0 / cases.len().max(1) as f64
            );
            write_jsonl(&need("--out")?, &predictions)?;
        }
        Some("tune") => {
            let cases: Vec<Case> = read_jsonl(&need("--cases")?)?;
            let dir = need("--model")?;
            let mut model = load_model(&dir, &args)?;
            let fixed: Vec<&str> = args
                .iter()
                .enumerate()
                .filter(|(_, a)| *a == "--fix")
                .filter_map(|(i, _)| args.get(i + 1).map(String::as_str))
                .collect();
            tune(&mut model, &cases, &fixed);
            print!("{}", model.params.to_text());
            if args.iter().any(|a| a == "--write") {
                std::fs::write(
                    std::path::Path::new(&dir).join("params.txt"),
                    model.params.to_text(),
                )?;
            }
        }
        Some("live") => {
            let cases: Vec<Case> = read_jsonl(&need("--cases")?)?;
            let show = flag("--show").map_or(Ok(20), |s| s.parse())?;
            let mut frontend = setsuna_core::config::FrontendCfg::default();
            frontend.mixed_model = Some(need("--model")?);
            frontend.name_dictionary_file = flag("--names");
            live(&cases, &frontend, show);
        }
        Some("timing") => {
            let dir = need("--model")?;
            let model = load_model(&dir, &args)?;
            let mut frontend = setsuna_core::config::FrontendCfg::default();
            frontend.mixed_model = Some(dir);
            let text = "kyouhagoogledekensakushitekaraashitanomeetingnisankashimasu.";
            for len in [10, 20, 40, 60, 90, 120] {
                let typed: String = text.chars().cycle().take(len).collect();
                let runs = 200;
                let t = std::time::Instant::now();
                for _ in 0..runs {
                    std::hint::black_box(model.segment(&typed, None, false));
                }
                let seg = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
                // the key that makes it `len` long, through the bridge
                let mut bridge = EngineBridge::with_frontend(None, &frontend);
                for c in typed.chars().take(len - 1) {
                    bridge.handle_char(c);
                }
                let t = std::time::Instant::now();
                bridge.handle_char(typed.chars().last().unwrap());
                let key = t.elapsed().as_secs_f64() * 1000.0;
                println!("{len:>4} chars: segment {seg:.2} ms, key {key:.2} ms");
            }
        }
        Some("trace") => {
            let typed = args.get(1).context("trace <typed>")?;
            trace(
                typed,
                flag("--before").as_deref(),
                args.iter().any(|a| a == "--no-mixed"),
                flag("--model"),
            );
        }
        Some("score") => {
            let cases: Vec<Case> = read_jsonl(&need("--cases")?)?;
            let predictions: Vec<Prediction> = read_jsonl(&need("--predictions")?)?;
            let show = flag("--show").map_or(Ok(20), |s| s.parse())?;
            score(&cases, &predictions, show);
        }
        _ => bail!("usage: mixed_input prepare|run|score (see the source)"),
    }
    Ok(())
}

fn hand_made() -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    for (i, line) in HAND_MADE.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if !(3..=4).contains(&cols.len()) {
            bail!("bench/mixed_input.tsv:{}: expected 3 or 4 columns", i + 1);
        }
        cases.push(Case {
            id: format!("hand-{:04}", i + 1),
            cat: cols[0].to_string(),
            typed: cols[1].to_string(),
            expected: cols[2].to_string(),
            before: cols.get(3).map(|s| s.replace('␣', " ")),
            cased: None,
        });
    }
    Ok(cases)
}

/// The first `n` usable readings of a corpus, typed as romaji. A reading
/// whose romaji the IME does not type back (a spelling the table lacks) is
/// skipped, so with no detection every corpus case passes.
fn corpus_cases(path: &str, n: usize) -> Result<Vec<Case>> {
    #[derive(Deserialize)]
    struct Line {
        reading: String,
    }
    let reader = BufReader::new(File::open(path).with_context(|| path.to_string())?);
    let mut cases = Vec::new();
    let mut skipped = 0;
    for line in reader.lines() {
        if cases.len() >= n {
            break;
        }
        let Ok(line) = serde_json::from_str::<Line>(&line?) else {
            continue;
        };
        let reading = line.reading;
        let len = reading.chars().count();
        if !(4..=40).contains(&len) {
            continue;
        }
        let Some(typed) = romaji(&reading) else {
            continue;
        };
        if type_line(&typed, None, &Default::default()) != reading {
            skipped += 1;
            continue;
        }
        cases.push(Case {
            id: format!("corpus-{:05}", cases.len() + 1),
            cat: "corpus-ja".to_string(),
            typed,
            expected: reading,
            before: None,
            cased: None,
        });
    }
    eprintln!("corpus: {} cases, {skipped} skipped", cases.len());
    Ok(cases)
}

/// Move the web cases with a Latin word (2+ letters) missing from
/// `latin_tsv` to a `-oov` category (`web-mixed-oov`).
fn tag_oov(cases: &mut [Case], latin_tsv: &str) {
    let vocab: std::collections::HashSet<&str> = latin_tsv
        .lines()
        .filter_map(|l| l.split('\t').next())
        .collect();
    for case in cases.iter_mut() {
        if case.cat.ends_with("-fused") {
            continue;
        }
        let oov = case
            .expected
            .split(|c: char| !c.is_ascii_alphabetic())
            .filter(|w| w.len() >= 2)
            .any(|w| !vocab.contains(w.to_ascii_lowercase().as_str()));
        if oov {
            case.cat.push_str("-oov");
        }
    }
}

/// Two sentences typed in one composition: Japanese + Japanese, Japanese +
/// mixed and mixed + Japanese (the first one ending in 。).
fn multi_cases(corpus: &[Case], mixed: &[Case]) -> Vec<Case> {
    let ends = |c: &&Case| c.expected.ends_with('。');
    let japanese: Vec<&Case> = corpus.iter().filter(ends).collect();
    let web: Vec<&Case> = mixed.iter().filter(|c| c.cat == "web-mixed").collect();
    let web_ending: Vec<&Case> = web.iter().copied().filter(ends).collect();
    let join = |cat: &str, k: usize, a: &Case, b: &Case| Case {
        id: format!("{cat}-{k:05}"),
        cat: cat.to_string(),
        typed: format!("{}{}", a.typed, b.typed),
        expected: format!("{}{}", a.expected, b.expected),
        before: None,
        cased: None,
    };
    let mut out = Vec::new();
    for (k, pair) in japanese.chunks(2).take(500).enumerate() {
        if let [a, b] = pair {
            out.push(join("multi-ja", k + 1, a, b));
        }
    }
    for (k, (a, b)) in japanese.iter().zip(&web).take(500).enumerate() {
        out.push(join("multi-ja-mixed", k + 1, a, b));
    }
    for (k, (a, b)) in web_ending
        .iter()
        .zip(japanese.iter().rev())
        .take(500)
        .enumerate()
    {
        out.push(join("multi-mixed-ja", k + 1, a, b));
    }
    out
}

/// Cases from readings with Latin words and digits kept as written. A
/// reading whose Japanese parts the IME does not type back is skipped.
fn mixed_cases(path: &str, n: usize) -> Result<Vec<Case>> {
    #[derive(Deserialize)]
    struct Line {
        reading: String,
    }
    let reader = BufReader::new(File::open(path).with_context(|| path.to_string())?);
    let mut written = Vec::new();
    let mut lower = Vec::new();
    let mut skipped = 0;
    for line in reader.lines() {
        if written.len() >= n {
            break;
        }
        let Ok(line) = serde_json::from_str::<Line>(&line?) else {
            continue;
        };
        let reading = line.reading;
        let Some(typed) = romaji(&reading) else {
            skipped += 1;
            continue;
        };
        let japanese_types_back = reading
            .split(is_kept)
            .filter(|run| !run.is_empty())
            .all(|run| {
                romaji(run).is_some_and(|r| type_line(&r, None, &Default::default()) == run)
            });
        if !japanese_types_back {
            skipped += 1;
            continue;
        }
        let fused = if fused_boundary(&reading) {
            "-fused"
        } else {
            ""
        };
        if typed.chars().any(|c| c.is_ascii_uppercase()) {
            lower.push(Case {
                id: format!("web-lower-{:05}", written.len() + 1),
                cat: format!("web-mixed-lower{fused}"),
                typed: typed.to_ascii_lowercase(),
                expected: reading.to_ascii_lowercase(),
                before: None,
                cased: Some(reading.clone()),
            });
        }
        written.push(Case {
            id: format!("web-{:05}", written.len() + 1),
            cat: format!("web-mixed{fused}"),
            typed,
            expected: reading,
            before: None,
            cased: None,
        });
    }
    eprintln!(
        "mixed: {} cases ({} lowercase), {skipped} skipped",
        written.len(),
        lower.len()
    );
    written.extend(lower);
    Ok(written)
}

/// Whether a Latin word's last letter and the romaji typed after it read as
/// a syllable together (a consonant before a vowel or `y`, a doubled
/// consonant, `sh` / `ch` / `th` …).
fn fused_boundary(reading: &str) -> bool {
    let chars: Vec<char> = reading.chars().collect();
    let vowel = |c: char| "aiueo".contains(c);
    for i in 1..chars.len() {
        if !chars[i - 1].is_ascii_alphabetic() || is_kept(chars[i]) {
            continue;
        }
        let rest: String = chars[i..].iter().take_while(|&&c| !is_kept(c)).collect();
        let Some(first) = romaji(&rest).and_then(|r| r.chars().next()) else {
            continue;
        };
        let last = chars[i - 1].to_ascii_lowercase();
        if (!vowel(last) && (vowel(first) || first == 'y'))
            || (last == first && first.is_ascii_alphabetic())
            || ("sctdw".contains(last) && first == 'h')
        {
            return true;
        }
    }
    false
}

/// A character a mixed reading keeps as typed (Latin words, digits).
fn is_kept(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-' | '\'')
}

/// Romaji a user types for `reading` (hiragana, ー, 、。, and Latin words or
/// digits typed as they are), or `None` when it has another character.
fn romaji(reading: &str) -> Option<String> {
    let chars: Vec<char> = reading.chars().collect();
    // Each kana (or kana + small kana) as its romaji, in order; `true` for
    // a char typed as it is.
    let mut units: Vec<(String, bool)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if is_kept(chars[i]) {
            units.push((chars[i].to_string(), true));
            i += 1;
            continue;
        }
        if let Some(&next) = chars.get(i + 1) {
            if let Some(r) = digraph(chars[i], next) {
                units.push((r.to_string(), false));
                i += 2;
                continue;
            }
        }
        let unit = match chars[i] {
            'っ' => "っ".to_string(),
            'ん' => "ん".to_string(),
            c => single(c)?.to_string(),
        };
        units.push((unit, false));
        i += 1;
    }
    let mut out = String::new();
    for (k, (unit, _)) in units.iter().enumerate() {
        // The next kana's first letter; none before a char typed as it is.
        let next = units
            .get(k + 1)
            .filter(|(_, kept)| !kept)
            .and_then(|(u, _)| u.chars().next());
        match unit.as_str() {
            "っ" => match next {
                Some(c) if c.is_ascii_alphabetic() && !"aiueon".contains(c) => out.push(c),
                _ => out.push_str("xtu"),
            },
            "ん" => match next {
                Some(c) if c.is_ascii_alphabetic() && !"aiueoyn".contains(c) => out.push('n'),
                _ => out.push_str("nn"),
            },
            u => out.push_str(u),
        }
    }
    Some(out)
}

fn digraph(a: char, b: char) -> Option<&'static str> {
    let base = match a {
        'き' => "ky",
        'ぎ' => "gy",
        'に' => "ny",
        'ひ' => "hy",
        'び' => "by",
        'ぴ' => "py",
        'み' => "my",
        'り' => "ry",
        'し' => "sh",
        'ち' => "ch",
        'じ' => "j",
        _ => "",
    };
    if !base.is_empty() {
        let v = match b {
            'ゃ' => "a",
            'ゅ' => "u",
            'ょ' => "o",
            'ぇ' if matches!(a, 'し' | 'ち' | 'じ') => "e",
            _ => return None,
        };
        return Some(match (base, v) {
            ("ky", "a") => "kya",
            ("ky", "u") => "kyu",
            ("ky", "o") => "kyo",
            ("gy", "a") => "gya",
            ("gy", "u") => "gyu",
            ("gy", "o") => "gyo",
            ("ny", "a") => "nya",
            ("ny", "u") => "nyu",
            ("ny", "o") => "nyo",
            ("hy", "a") => "hya",
            ("hy", "u") => "hyu",
            ("hy", "o") => "hyo",
            ("by", "a") => "bya",
            ("by", "u") => "byu",
            ("by", "o") => "byo",
            ("py", "a") => "pya",
            ("py", "u") => "pyu",
            ("py", "o") => "pyo",
            ("my", "a") => "mya",
            ("my", "u") => "myu",
            ("my", "o") => "myo",
            ("ry", "a") => "rya",
            ("ry", "u") => "ryu",
            ("ry", "o") => "ryo",
            ("sh", "a") => "sha",
            ("sh", "u") => "shu",
            ("sh", "o") => "sho",
            ("sh", "e") => "she",
            ("ch", "a") => "cha",
            ("ch", "u") => "chu",
            ("ch", "o") => "cho",
            ("ch", "e") => "che",
            ("j", "a") => "ja",
            ("j", "u") => "ju",
            ("j", "o") => "jo",
            ("j", "e") => "je",
            _ => return None,
        });
    }
    Some(match (a, b) {
        ('ふ', 'ぁ') => "fa",
        ('ふ', 'ぃ') => "fi",
        ('ふ', 'ぇ') => "fe",
        ('ふ', 'ぉ') => "fo",
        ('て', 'ぃ') => "thi",
        ('で', 'ぃ') => "dhi",
        ('う', 'ぃ') => "wi",
        ('う', 'ぇ') => "we",
        ('ゔ', 'ぁ') => "va",
        ('ゔ', 'ぃ') => "vi",
        ('ゔ', 'ぇ') => "ve",
        ('ゔ', 'ぉ') => "vo",
        _ => return None,
    })
}

fn single(c: char) -> Option<&'static str> {
    const ROWS: [(&str, [&str; 5]); 15] = [
        ("あいうえお", ["a", "i", "u", "e", "o"]),
        ("かきくけこ", ["ka", "ki", "ku", "ke", "ko"]),
        ("さしすせそ", ["sa", "shi", "su", "se", "so"]),
        ("たちつてと", ["ta", "chi", "tsu", "te", "to"]),
        ("なにぬねの", ["na", "ni", "nu", "ne", "no"]),
        ("はひふへほ", ["ha", "hi", "fu", "he", "ho"]),
        ("まみむめも", ["ma", "mi", "mu", "me", "mo"]),
        ("らりるれろ", ["ra", "ri", "ru", "re", "ro"]),
        ("がぎぐげご", ["ga", "gi", "gu", "ge", "go"]),
        ("ざじずぜぞ", ["za", "ji", "zu", "ze", "zo"]),
        ("だぢづでど", ["da", "di", "du", "de", "do"]),
        ("ばびぶべぼ", ["ba", "bi", "bu", "be", "bo"]),
        ("ぱぴぷぺぽ", ["pa", "pi", "pu", "pe", "po"]),
        ("ぁぃぅぇぉ", ["xa", "xi", "xu", "xe", "xo"]),
        ("ゃゅょゎゔ", ["xya", "xyu", "xyo", "xwa", "vu"]),
    ];
    for (kana, romaji) in ROWS {
        if let Some(k) = kana.chars().position(|k| k == c) {
            return Some(romaji[k]);
        }
    }
    Some(match c {
        'や' => "ya",
        'ゆ' => "yu",
        'よ' => "yo",
        'わ' => "wa",
        'を' => "wo",
        'ー' => "-",
        '、' => ",",
        '。' => ".",
        _ => return None,
    })
}

/// What the IME commits for `typed` then Enter, given the text before the
/// caret. Keys go to the bridge as the frontends send them (Shift+letter as
/// uppercase, symbols through the default symbol map); a key the bridge
/// does not take reaches the app as typed.
fn type_line(
    typed: &str,
    before: Option<&str>,
    frontend: &setsuna_core::config::FrontendCfg,
) -> String {
    let symbols = &frontend.symbols;
    let mut bridge = EngineBridge::with_frontend(None, frontend);
    if let Some(before) = before {
        bridge.set_app_context(Some(before));
    }
    let mut out = String::new();
    let collect = |out: &mut String, actions: Vec<Action>| {
        for action in actions {
            if let Action::Commit { text } = action {
                out.push_str(&text);
            }
        }
    };
    for c in typed.chars() {
        let outcome = match c {
            'A'..='Z' => bridge.handle_shifted_alpha(c),
            'a'..='z' | '0'..='9' | '-' => bridge.handle_char(c),
            ' ' => bridge.handle_space(),
            _ => {
                let kana = setsuna_core::config::SYMBOL_CHARS
                    .iter()
                    .find(|(_, ch)| *ch == c)
                    .and_then(|(name, _)| symbols.get(*name).cloned())
                    .unwrap_or_else(|| full_width(c).to_string());
                bridge.handle_punctuation(&kana)
            }
        };
        if !outcome.consumed {
            out.push(c);
        }
        collect(&mut out, outcome.actions);
    }
    collect(&mut out, bridge.handle_enter().actions);
    out
}

fn full_width(c: char) -> char {
    char::from_u32(c as u32 - 0x21 + 0xff01).unwrap_or(c)
}

fn trace(typed: &str, before: Option<&str>, no_mixed: bool, model: Option<String>) {
    let mut cfg = setsuna_core::config::Config::load_default();
    if no_mixed {
        cfg.frontend.mixed_model = None;
    } else if model.is_some() {
        cfg.frontend.mixed_model = model;
    }
    let (tx, rx) = std::sync::mpsc::channel::<u64>();
    let notify: setsuna_core::engine_client::Notify = Box::new(move |id| {
        let _ = tx.send(id);
    });
    let install =
        std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("setsuna-ime"));
    let client = setsuna_core::child_daemon::spawn_with_config(&cfg, install.as_deref(), notify)
        .map(|c| Box::new(c) as Box<dyn setsuna_core::engine_client::EngineClient>);
    println!("engine: {}", client.is_some());
    let mut bridge = EngineBridge::with_frontend(client, &cfg.frontend);
    bridge.set_live_conversion(cfg.frontend.live_conversion.unwrap_or(true));
    bridge.set_app_context(before);
    let symbols = &cfg.frontend.symbols;
    let show = |label: &str, actions: &[Action]| {
        for a in actions {
            match a {
                Action::UpdatePreedit { text } => println!("{label:>6}  preedit {text}"),
                Action::Commit { text } => println!("{label:>6}  COMMIT  {text:?}"),
                Action::ShowCandidates { list, .. } => println!("{label:>6}  list    {list:?}"),
                _ => {}
            }
        }
    };
    let settle = |bridge: &mut EngineBridge, label: &str| {
        // the engine's answer(s) to this key
        while rx
            .recv_timeout(std::time::Duration::from_millis(400))
            .is_ok()
        {
            let actions = bridge.apply_async_result();
            show(&format!("{label}*"), &actions);
        }
    };
    for c in typed.chars() {
        let label = c.to_string();
        let outcome = match c {
            'A'..='Z' => bridge.handle_shifted_alpha(c),
            'a'..='z' | '0'..='9' | '-' => bridge.handle_char(c),
            ' ' => bridge.handle_space(),
            _ => {
                let kana = setsuna_core::config::SYMBOL_CHARS
                    .iter()
                    .find(|(_, ch)| *ch == c)
                    .and_then(|(name, _)| symbols.get(*name).cloned())
                    .unwrap_or_else(|| full_width(c).to_string());
                bridge.handle_punctuation(&kana)
            }
        };
        show(&label, &outcome.actions);
        settle(&mut bridge, &label);
    }
    let out = bridge.handle_enter();
    show("Enter", &out.actions);
}

fn live(cases: &[Case], frontend: &setsuna_core::config::FrontendCfg, show: usize) {
    let mut by_cat: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut shown = 0;
    for case in cases {
        if case.expected.chars().any(|c| c.is_ascii_alphabetic()) || case.typed.contains(' ') {
            continue;
        }
        let mut bridge = EngineBridge::with_frontend(None, frontend);
        bridge.set_app_context(case.before.as_deref());
        let mut bad = None;
        for (k, c) in case.typed.chars().enumerate() {
            let outcome = match c {
                'a'..='z' | '0'..='9' | '-' => bridge.handle_char(c),
                _ => {
                    let kana = setsuna_core::config::SYMBOL_CHARS
                        .iter()
                        .find(|(_, ch)| *ch == c)
                        .and_then(|(name, _)| frontend.symbols.get(*name).cloned())
                        .unwrap_or_else(|| full_width(c).to_string());
                    bridge.handle_punctuation(&kana)
                }
            };
            let preedit = outcome.actions.iter().rev().find_map(|a| match a {
                Action::UpdatePreedit { text } => Some(text.clone()),
                _ => None,
            });
            // Letters before the trailing romaji being typed.
            if let Some(p) = preedit {
                let body = p.trim_end_matches(|c: char| c.is_ascii_alphabetic());
                if body.chars().any(|c| c.is_ascii_alphabetic()) && bad.is_none() {
                    bad = Some(format!("{} → {p}", &case.typed[..=k]));
                }
            }
        }
        let e = by_cat.entry(case.cat.as_str()).or_default();
        e.1 += 1;
        match bad {
            None => e.0 += 1,
            Some(b) if shown < show => {
                println!("  [{}] {b}", case.cat);
                shown += 1;
            }
            Some(_) => {}
        }
    }
    for (cat, (p, t)) in &by_cat {
        println!(
            "{cat:<22} {p:>5}/{t:<5} {:>6.1}% shown as Japanese all along",
            100.0 * *p as f64 / *t as f64
        );
    }
}

fn load_model(dir: &str, args: &[String]) -> Result<MixedModel> {
    let mut model = MixedModel::load(std::path::Path::new(dir))?;
    for (i, a) in args.iter().enumerate() {
        if a == "--names" {
            let path = args.get(i + 1).map_or("", String::as_str);
            model.add_names(&std::fs::read_to_string(path).with_context(|| path.to_string())?);
        }
        if a == "--param" {
            model
                .params
                .set(args.get(i + 1).map_or("", String::as_str))?;
        }
    }
    Ok(model)
}

/// The tuning objective of `model` on `cases` (see the module docs).
fn objective(model: &MixedModel, cases: &[Case]) -> f64 {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = cases.len().div_ceil(threads).max(1);
    let mut by_cat: HashMap<String, (usize, usize)> = HashMap::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
                    for c in part {
                        let ok = shown(model, &c.typed, c.before.as_deref()) == c.truth();
                        let e = counts.entry(c.cat.clone()).or_default();
                        e.0 += ok as usize;
                        e.1 += 1;
                    }
                    counts
                })
            })
            .collect();
        for handle in handles {
            for (cat, (p, t)) in handle.join().expect("worker panicked") {
                let e = by_cat.entry(cat).or_default();
                e.0 += p;
                e.1 += t;
            }
        }
    });
    let acc = |cat: &str| {
        by_cat
            .get(cat)
            .map_or(1.0, |&(p, t)| p as f64 / t.max(1) as f64)
    };
    let mixed = [
        "web-mixed",
        "web-mixed-lower",
        "web-mixed-oov",
        "web-mixed-lower-oov",
        "multi-ja-mixed",
        "multi-mixed-ja",
    ];
    let japanese = ["corpus-ja", "multi-ja"];
    mixed.iter().map(|c| acc(c)).sum::<f64>() / mixed.len() as f64
        - 10.0 * japanese.iter().map(|c| 1.0 - acc(c)).sum::<f64>()
}

/// Coordinate search: each parameter scaled up and down in turn, a change
/// kept when it raises the objective; the step halves when a round keeps
/// none.
fn tune(model: &mut MixedModel, cases: &[Case], fixed: &[&str]) {
    let mut best = objective(model, cases);
    eprintln!("start {best:.4}");
    let mut step = 0.4;
    for round in 0..8 {
        let mut improved = false;
        for name in MixedParams::NAMES
            .into_iter()
            .filter(|n| !fixed.contains(n))
        {
            for dir in [1.0, -1.0] {
                let old = *model.params.field(name).expect("known name");
                let new = if matches!(name, "known" | "ja_text") {
                    // a weight in (0, 1): a step in log-odds
                    let logit = (old / (1.0 - old)).ln() + dir * step * 4.0;
                    1.0 / (1.0 + (-logit).exp())
                } else {
                    old * (1.0 + step).powf(dir)
                };
                *model.params.field(name).expect("known name") = new;
                let score = objective(model, cases);
                if score > best + 1e-9 {
                    eprintln!("round {round}: {name} {old:.4} -> {new:.4}: {score:.4}");
                    best = score;
                    improved = true;
                    break;
                }
                *model.params.field(name).expect("known name") = old;
            }
        }
        if !improved {
            step /= 2.0;
            if step < 0.05 {
                break;
            }
        }
    }
    eprintln!("best {best:.4}");
}

fn score(cases: &[Case], predictions: &[Prediction], show: usize) {
    let actual: HashMap<&str, &str> = predictions
        .iter()
        .map(|p| (p.id.as_str(), p.actual.as_str()))
        .collect();
    // per category: (pass, total, pass ignoring case)
    let mut by_cat: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut missing = 0;
    for case in cases {
        let Some(&got) = actual.get(case.id.as_str()) else {
            missing += 1;
            continue;
        };
        let entry = by_cat.entry(case.cat.as_str()).or_default();
        entry.1 += 1;
        entry.2 += (got.to_ascii_lowercase() == case.truth().to_ascii_lowercase()
            || got == case.expected) as usize;
        if got == case.truth() {
            entry.0 += 1;
        } else {
            let before = case
                .before
                .as_deref()
                .map_or(String::new(), |b| format!("「{b}」+ "));
            failures.push(format!(
                "[{}] {before}{} → {got} (expected {})",
                case.cat,
                case.typed,
                case.truth()
            ));
        }
    }
    let (pass, total, pass_ci) = by_cat.values().fold((0, 0, 0), |(p, t, c), (cp, ct, cc)| {
        (p + cp, t + ct, c + cc)
    });
    println!("{:<22} {:>11} {:>7}  ignoring case", "", "", "");
    let pct = |p: usize, t: usize| 100.0 * p as f64 / t.max(1) as f64;
    for (cat, &(p, t, c)) in &by_cat {
        println!(
            "{cat:<22} {p:>5}/{t:<5} {:>6.1}%  {:>6.1}%",
            pct(p, t),
            pct(c, t)
        );
    }
    println!(
        "{:<22} {pass:>5}/{total:<5} {:>6.1}%  {:>6.1}%",
        "all",
        pct(pass, total),
        pct(pass_ci, total)
    );
    if missing > 0 {
        println!("{missing} cases have no prediction");
    }
    for failure in failures.iter().take(show) {
        println!("  {failure}");
    }
    if failures.len() > show {
        println!("  … {} more", failures.len() - show);
    }
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Result<Vec<T>> {
    let reader = BufReader::new(File::open(path).with_context(|| path.to_string())?);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line).with_context(|| line.clone())?);
        }
    }
    Ok(out)
}

fn write_jsonl<T: Serialize>(path: &str, items: &[T]) -> Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path).with_context(|| path.to_string())?);
    for item in items {
        serde_json::to_writer(&mut w, item)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}
