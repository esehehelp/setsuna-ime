//! setsuna-bench: the IME benchmark (`sets/`): `check` the sets, type them
//! into the IME session (`harness`), or through a Windows input method in a
//! real text field (`tsf`), and `score` the runs.

mod harness;
mod romaji;
mod score;
mod set;
#[cfg(windows)]
mod tsf;

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Benchmark of the IME as a whole")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// The keys typed for a reading.
    Keys { reading: String },
    /// Check the sets: every row typeable, ids unique; with --analyze, the
    /// analyzer's reading of each answer against the row's reading.
    Check {
        #[arg(long, default_value = "crates/bench/sets/v1")]
        sets: PathBuf,
        #[arg(long)]
        analyze: bool,
    },
    /// Type the sets into the IME session (no desktop needed).
    Harness {
        #[arg(long, default_value = "crates/bench/sets/v1")]
        sets: PathBuf,
        /// JSONL of the outputs, one row a line.
        #[arg(long)]
        out: PathBuf,
        /// Only these categories (comma-separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        #[arg(long)]
        no_rerank: bool,
        #[arg(long)]
        no_lexicon: bool,
        /// Keep the context of the rows before, as a field TSF cannot read.
        #[arg(long)]
        carry: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Settings from this file instead of the user's config.toml, each
        /// ${KAIRO_*} in it replaced by that variable: a run under Kairo
        /// (kairo.toml) names the directories it is handed.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Keys this far apart; 0: one after another without waiting.
        #[arg(long, default_value_t = 0)]
        key_ms: u64,
        #[arg(long, default_value_t = 100)]
        quiet_ms: u64,
        /// Enter at the earliest this long after Space.
        #[arg(long, default_value_t = 500)]
        look_ms: u64,
        #[arg(long, default_value_t = usize::MAX)]
        limit: usize,
    },
    /// Type the sets through a Windows input method into a real text field.
    #[cfg(windows)]
    Tsf(tsf::Args),
    /// Score runs (name=path.jsonl ...) on the sets.
    Score {
        #[arg(long, default_value = "crates/bench/sets/v1")]
        sets: PathBuf,
        runs: Vec<String>,
    },
    /// The `tsf` runs under a directory side by side: accuracy per
    /// category, latency, CPU, memory and storage.
    Report {
        #[arg(long, default_value = "crates/bench/sets/v1")]
        sets: PathBuf,
        /// The directory holding one directory per run (rows.jsonl, meta.json).
        runs: PathBuf,
    },
    /// The IME's order of an engine bench run's candidates (neo-ime's
    /// `*__beam.full.jsonl`: reading, context, candidates, candidate_scores,
    /// references) with each reading lexicon given and with none: EM1 of
    /// the top before and after.
    Rank {
        /// The bench run's directory.
        run: PathBuf,
        /// Lexicon files (name=path) to order with; "none" orders without.
        #[arg(num_args = 1.., required = true)]
        lexicons: Vec<String>,
        /// Also write each item's ordered list (its first 20) by lexicon, as
        /// JSON lines: bench, index, category, references, lists.
        #[arg(long)]
        items: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Keys { reading } => {
            println!(
                "{}",
                romaji::keys(&reading).map_err(|c| anyhow::anyhow!("cannot type {c:?}"))?
            );
        }
        Cmd::Check { sets, analyze } => check(&sets, analyze)?,
        Cmd::Harness {
            sets,
            out,
            only,
            no_rerank,
            no_lexicon,
            carry,
            manifest,
            config,
            key_ms,
            quiet_ms,
            look_ms,
            limit,
        } => {
            let rows = set::load(&sets, &only)?;
            let opts = harness::Options {
                config,
                rerank: !no_rerank,
                lexicon: !no_lexicon,
                carry,
                manifest,
                key_ms,
                quiet_ms,
                look_ms,
                limit,
            };
            let mut h = harness::Harness::new(opts)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            let n = rows.len().min(limit);
            let t0 = std::time::Instant::now();
            let mut k = 0;
            // a row that cannot be written (a full disk) is no row of the
            // run: the first such error ends it, reported
            let mut written: std::io::Result<()> = Ok(());
            let typed = h.run(&rows, |t| {
                written = serde_json::to_string(t)
                    .map_err(std::io::Error::other)
                    .and_then(|line| writeln!(f, "{line}"))
                    .and_then(|()| f.flush());
                k += 1;
                if k % 50 == 0 {
                    eprintln!("{k}/{n} in {:.1} min", t0.elapsed().as_secs_f64() / 60.0);
                }
                written.is_ok()
            })?;
            written.with_context(|| format!("write {}", out.display()))?;
            let mut engine = h.engine_ms.lock().unwrap().clone();
            let mut space: Vec<f64> = typed.iter().map(|t| t.space_ms).collect();
            eprintln!(
                "{} rows in {:.1} min; engine request p50 {:.1} / p95 {:.1} ms; Space to settled p50 {:.0} / p95 {:.0} ms",
                typed.len(),
                t0.elapsed().as_secs_f64() / 60.0,
                score::percentile(&mut engine, 0.5),
                score::percentile(&mut engine, 0.95),
                score::percentile(&mut space, 0.5),
                score::percentile(&mut space, 0.95),
            );
        }
        #[cfg(windows)]
        Cmd::Tsf(args) => tsf::run(args)?,
        Cmd::Score { sets, runs } => score_runs(&sets, &runs)?,
        Cmd::Report { sets, runs } => report(&sets, &runs)?,
        Cmd::Rank {
            run,
            lexicons,
            items,
        } => rank(&run, &lexicons, items.as_deref())?,
    }
    Ok(())
}

/// One `tsf` run: its outputs by row id and its meta.json.
fn read_run(dir: &std::path::Path) -> Result<(BTreeMap<String, String>, serde_json::Value)> {
    let mut outputs = BTreeMap::new();
    let rows = std::fs::read_to_string(dir.join("rows.jsonl"))
        .with_context(|| format!("read {}", dir.display()))?;
    for line in rows.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        outputs.insert(
            v["id"].as_str().unwrap_or_default().to_string(),
            v["output"].as_str().unwrap_or_default().to_string(),
        );
    }
    let meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    Ok((outputs, meta))
}

fn report(sets: &std::path::Path, runs: &std::path::Path) -> Result<()> {
    let rows = set::load(sets, &[])?;
    let mut cats: Vec<String> = rows.iter().map(|r| r.category.clone()).collect();
    cats.sort();
    cats.dedup();
    cats.push("all".into());
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(runs)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("meta.json").is_file())
        .collect();
    dirs.sort();
    let name = |d: &PathBuf| {
        d.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    };
    let mut loaded = Vec::new();
    for d in &dirs {
        loaded.push((name(d), read_run(d)?));
    }
    // the runner's own cost: its CPU per row with no input method
    let base_cpu_row = loaded
        .iter()
        .find(|(n, _)| n.starts_with("none"))
        .map(|(_, (_, m))| {
            m["cpu_ms"][0].as_f64().unwrap_or(0.0) / m["rows"].as_f64().unwrap_or(1.0).max(1.0)
        })
        .unwrap_or(0.0);

    println!("Accuracy: exact % per category (rows), CER over all");
    print!("{:18}", "run");
    for c in &cats {
        print!("{:>9}", c);
    }
    println!("{:>8}", "CER");
    for (n, (outputs, _)) in &loaded {
        if n.starts_with("none") {
            continue;
        }
        let t = score::tally(&rows, outputs);
        print!("{n:18}");
        for c in &cats {
            match t.get(c) {
                Some(x) if x.n > 0 => print!("{:>9.1}", 100.0 * x.right as f64 / x.n as f64),
                _ => print!("{:>9}", "-"),
            }
        }
        let all = t.get("all");
        println!(
            "{:>8.3}",
            all.map_or(f64::NAN, |x| x.cer / x.n.max(1) as f64)
        );
    }

    println!();
    println!("Speed and resources, ms p50/p95: a row's keys sent at once to the field still; a key typed alone to");
    println!("the key shown and to its conversion in; Space to the last change (rows where it changed anything).");
    println!("CPU a row: in this process less the runner's own ({base_cpu_row:.0} ms), and in the input method's processes.");
    println!(
        "{:18}{:>8}{:>12}{:>14}{:>14}{:>16}{:>12}{:>12}{:>11}{:>12}{:>12}{:>10}",
        "run",
        "min",
        "row keys",
        "key shown",
        "key conv",
        "Space (rows)",
        "CPU/row in",
        "CPU/row own",
        "idle CPU%",
        "mem in MB",
        "mem own MB",
        "disk MB"
    );
    for (n, (_, m)) in &loaded {
        if n.starts_with("none") {
            continue;
        }
        let f = |k: &str| m[k].as_f64().unwrap_or(f64::NAN);
        let rows_n = f("rows").max(1.0);
        let cpu_in = m["cpu_ms"][0].as_f64().unwrap_or(0.0) / rows_n - base_cpu_row;
        let cpu_own = m["cpu_ms"][1].as_f64().unwrap_or(0.0) / rows_n;
        let idle_pct = (m["idle_cpu_ms"][0].as_f64().unwrap_or(0.0)
            + m["idle_cpu_ms"][1].as_f64().unwrap_or(0.0))
            / 5000.0
            * 100.0;
        let mem_in = m["peak_private_mb"][0].as_f64().unwrap_or(0.0)
            - m["base"]["private_mb"].as_f64().unwrap_or(0.0);
        let mem_own = m["peak_private_mb"][1].as_f64().unwrap_or(0.0);
        let p = |k: &str| {
            format!(
                "{:.0}/{:.0}",
                f(&format!("{k}_p50")),
                f(&format!("{k}_p95"))
            )
        };
        println!(
            "{n:18}{:>8.1}{:>12}{:>14}{:>14}{:>16}{:>12.0}{:>12.0}{:>11.1}{:>12.1}{:>12.1}{:>10.0}",
            f("minutes"),
            p("burst_ms"),
            p("key_first_ms"),
            p("key_last_ms"),
            format!("{} ({:.0})", p("space_last_ms"), f("space_changed")),
            cpu_in,
            cpu_own,
            idle_pct,
            mem_in,
            mem_own,
            f("storage_mb"),
        );
    }
    Ok(())
}

fn rank(run: &std::path::Path, lexicons: &[String], items: Option<&std::path::Path>) -> Result<()> {
    use setsuna_core::engine_bridge::order_candidates;
    use setsuna_core::engine_client::EngineResult;
    use setsuna_core::lexicon::Lexicon;
    let mut lex: Vec<(String, Option<Lexicon>)> = Vec::new();
    for spec in lexicons {
        match spec.split_once('=') {
            Some((name, path)) => lex.push((
                name.to_string(),
                Some(Lexicon::open(std::path::Path::new(path))?),
            )),
            None if spec == "none" => lex.push(("none".to_string(), None)),
            None => bail!("{spec}: name=path or none"),
        }
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(run)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".full.jsonl"))
        .collect();
    files.sort();
    let mut items = match items {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    print!("{:24}{:>7}{:>9}", "bench", "n", "engine");
    for (name, _) in &lex {
        print!("{:>14}", name);
    }
    println!();
    for file in files {
        let text = std::fs::read_to_string(&file)?;
        let (mut n, mut engine) = (0usize, 0usize);
        let mut right = vec![0usize; lex.len()];
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line)?;
            let strs = |k: &str| -> Vec<String> {
                v[k].as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let refs = strs("references");
            let candidates = strs("candidates");
            let scores: Vec<f32> = v["candidate_scores"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default();
            if candidates.is_empty() {
                continue;
            }
            n += 1;
            engine += usize::from(refs.contains(&candidates[0]));
            let mut lists = serde_json::Map::new();
            lists.insert(
                "engine".to_string(),
                serde_json::json!(candidates.iter().take(20).collect::<Vec<_>>()),
            );
            for (k, (name, l)) in lex.iter().enumerate() {
                let mut r = EngineResult {
                    reading: v["reading"].as_str().unwrap_or("").to_string(),
                    context: v["context"].as_str().unwrap_or("").to_string(),
                    candidates: candidates.clone(),
                    scores: scores.clone(),
                    ..Default::default()
                };
                order_candidates(l.as_ref(), None, &mut r);
                right[k] += usize::from(r.candidates.first().is_some_and(|c| refs.contains(c)));
                lists.insert(
                    name.clone(),
                    serde_json::json!(r.candidates.iter().take(20).collect::<Vec<_>>()),
                );
            }
            if let Some(w) = items.as_mut() {
                let line = serde_json::json!({
                    "bench": v["bench"],
                    "index": v["index"],
                    "category": v["category"],
                    "references": refs,
                    "lists": lists,
                });
                writeln!(w, "{line}")?;
            }
        }
        let name = file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .replace("__beam.full.jsonl", "");
        let pct = |x: usize| 100.0 * x as f64 / n.max(1) as f64;
        print!("{name:24}{n:>7}{:>9.2}", pct(engine));
        for r in &right {
            print!("{:>14.2}", pct(*r));
        }
        println!();
    }
    Ok(())
}

fn check(sets: &std::path::Path, analyze: bool) -> Result<()> {
    let rows = set::load(sets, &[])?;
    let mut seen = HashSet::new();
    let mut bad = 0;
    for r in &rows {
        if !seen.insert(r.id.clone()) {
            println!("{}: duplicate id", r.id);
            bad += 1;
        }
        if let Err(c) = romaji::keys(&r.reading) {
            println!("{}: cannot type {c:?} in {}", r.id, r.reading);
            bad += 1;
        }
    }
    let mut per: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        *per.entry(&r.category).or_default() += 1;
    }
    println!(
        "{} rows {:?}; {bad} not typeable or duplicate",
        rows.len(),
        per
    );
    if !analyze {
        return Ok(());
    }
    use setsuna_core::engine_client::EngineClient;
    let cfg = setsuna_core::config::Config::load_default();
    if cfg.engine.analyzer_manifest.is_none() {
        bail!("no [engine] analyzer_manifest in the config");
    }
    let manifest = cfg
        .engine
        .model_manifest
        .clone()
        .context("model_manifest")?;
    let exe = cfg.engine.engine_exe.clone().context("engine_exe")?;
    let client = setsuna_core::child_daemon::ChildDaemonClient::spawn(
        &manifest,
        &exe,
        Box::new(|_| {}),
        &cfg.engine,
    )?;
    // the daemon up, or gone (a model it cannot load): not a wait forever
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while !client.is_ready() {
        if !client.is_alive() {
            bail!("the engine exited before it was ready (its stderr log says why)");
        }
        if std::time::Instant::now() >= deadline {
            bail!("the engine was not ready in 120 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    // hiragana only: katakana folded, everything else dropped
    let kana = |s: &str| -> String {
        s.chars()
            .map(|c| match c {
                'ァ'..='ヶ' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
                _ => c,
            })
            .filter(|c| matches!(c, 'ぁ'..='ゖ' | 'ー'))
            .collect()
    };
    // the analyzer loads its dictionary with the first request
    // (answered without words until it is in)
    let t0 = std::time::Instant::now();
    while client
        .analyze(&["今日".to_string()], Duration::from_secs(30))
        .is_none()
    {
        if t0.elapsed() > Duration::from_secs(120) {
            bail!("the analyzer did not answer");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let mut differ = 0;
    // the engine analyzes 16 texts a request at most
    for chunk in rows.chunks(16) {
        let texts: Vec<String> = chunk.iter().map(|r| r.answers[0].clone()).collect();
        let words = client
            .analyze(&texts, Duration::from_secs(30))
            .context("the analyzer did not answer")?;
        for (r, ws) in chunk.iter().zip(words) {
            let read: String = ws
                .iter()
                .map(|w| w.reading.clone().unwrap_or_else(|| w.surface.clone()))
                .collect();
            if kana(&read) != kana(&r.reading) {
                differ += 1;
                println!(
                    "{}\t{}\t{}\t{}",
                    r.id,
                    kana(&r.reading),
                    kana(&read),
                    r.answers[0]
                );
            }
        }
    }
    println!("{differ} rows whose answer the analyzer reads otherwise (review them)");
    Ok(())
}

fn score_runs(sets: &std::path::Path, runs: &[String]) -> Result<()> {
    let rows = set::load(sets, &[])?;
    let mut cats: Vec<String> = rows.iter().map(|r| r.category.clone()).collect();
    cats.dedup();
    cats.sort();
    cats.dedup();
    cats.push("all".into());
    print!("{:24}", "run");
    for c in &cats {
        print!("{:>14}", c);
    }
    println!(
        "{:>22}{:>22}",
        "Space shown p50/p95", "Space settled p50/p95"
    );
    for spec in runs {
        let (name, path) = spec
            .split_once('=')
            .unwrap_or((spec.as_str(), spec.as_str()));
        let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        let mut outputs = BTreeMap::new();
        let mut space = Vec::new();
        let mut first = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line)?;
            let id = v["id"].as_str().unwrap_or_default().to_string();
            outputs.insert(id, v["output"].as_str().unwrap_or_default().to_string());
            if let Some(ms) = v["space_ms"].as_f64() {
                space.push(ms);
            }
            if let Some(ms) = v["space_first_ms"].as_f64() {
                first.push(ms);
            }
        }
        let t = score::tally(&rows, &outputs);
        print!("{name:24}");
        for c in &cats {
            match t.get(c) {
                Some(x) if x.n > 0 => print!(
                    "{:>14}",
                    format!("{:.1} ({})", 100.0 * x.right as f64 / x.n as f64, x.n)
                ),
                _ => print!("{:>14}", "-"),
            }
        }
        let pair = |v: &mut Vec<f64>| {
            format!(
                "{:.0} / {:.0}",
                score::percentile(&mut v.clone(), 0.5),
                score::percentile(v, 0.95)
            )
        };
        println!("{:>22}{:>22}", pair(&mut first), pair(&mut space));
    }
    Ok(())
}
