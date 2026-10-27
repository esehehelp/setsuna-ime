//! Analyzer tools.
//!
//! `alternates <lex.csv> <matrix.def> <reading column> <out>`: compile the
//! other readings' table (`setsuna_analyzer::alternates`) from a trained
//! analyzer's lexicon and connection costs.
//! `analyze <manifest>`: JSON lines {"id", "text"} on stdin -> {"id",
//! "tokens"} on stdout.

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["alternates", lex, matrix, col, out] => {
            let t = Instant::now();
            let (keys, entries) = setsuna_analyzer::alternates::build(
                Path::new(lex),
                Path::new(matrix),
                col.parse().context("reading column")?,
                Path::new(out),
            )?;
            eprintln!(
                "{keys} surfaces, {entries} entries -> {out} in {:.1} s",
                t.elapsed().as_secs_f64()
            );
        }
        ["analyze", manifest] => {
            let t = Instant::now();
            let analyzer = setsuna_analyzer::Analyzer::from_manifest_path(Path::new(manifest))?;
            eprintln!("loaded in {:.2} s", t.elapsed().as_secs_f64());
            let stdout = std::io::stdout();
            let mut out = std::io::BufWriter::new(stdout.lock());
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                let text = v["text"].as_str().context("no text")?;
                let tokens = analyzer.analyze(text);
                serde_json::to_writer(&mut out, &serde_json::json!({"id": v["id"], "tokens": tokens}))?;
                out.write_all(b"\n")?;
            }
        }
        _ => bail!("usage: analyzer alternates <lex.csv> <matrix.def> <reading column> <out> | analyze <manifest>"),
    }
    Ok(())
}
