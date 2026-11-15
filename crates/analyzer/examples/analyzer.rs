//! Analyzer tools.
//!
//! `alternates <lex.csv> <matrix.def | bigram dir> <reading column> <out>`:
//! compile the other readings' table (`setsuna_analyzer::alternates`) from a
//! trained analyzer's lexicon and connection costs (the matrix, or the
//! model's bigram.right / .left / .cost the dictionary was built from).
//! `analyze <manifest>`: JSON lines {"id", "text"} on stdin -> {"id",
//! "tokens"} on stdout.
//! `usage <manifest> <texts.txt> <out.tsv>`: how often the analysis of each
//! line of `texts.txt` uses each word (`setsuna_analyzer::packed::count_usage`).
//! `build <lex.csv> <unk.def> <char.def> <matrix.def | bigram dir> <features>
//! <out dir> [<usage.tsv> <top>]`: a packed dictionary and its manifest
//! (`setsuna_analyzer::packed::build`); `features` names the columns after
//! the cost, comma-separated (pos1, pos2 and reading among them).
//!
//! The installed analyzer (2026-11-15): neo-ime-dict's nid-ma v2-600k-tb, its
//! lexicon and unk.def with the model's bigram ids and its bigram files (as
//! e1ea201's alternates take them); a packed build of all its words counts
//! their use over 4.0M sentences of the raw corpus (300,000 a source, no
//! document holding an evaluation text), and the 570,000 most used are
//! built: the most that keep the file under 10 MB.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["usage", manifest, texts, out] => {
            let analyzer = setsuna_analyzer::Analyzer::from_manifest_path(Path::new(manifest))?;
            let text = std::fs::read_to_string(texts).with_context(|| texts.to_string())?;
            let t = Instant::now();
            let n = setsuna_analyzer::packed::count_usage(
                &analyzer,
                text.lines().filter(|l| !l.trim().is_empty()),
                Path::new(out),
            )?;
            eprintln!("{n} words -> {out} in {:.1} s", t.elapsed().as_secs_f64());
        }
        ["build", lex, unk, char_def, connection, features, out, ref rest @ ..] => {
            let (usage, top) = match rest {
                [] => (None, None),
                [usage, top] => (
                    Some(PathBuf::from(usage)),
                    Some(top.parse().context("top")?),
                ),
                _ => bail!("build: <usage.tsv> and <top> go together"),
            };
            let t = Instant::now();
            let built = setsuna_analyzer::packed::build(
                &setsuna_analyzer::packed::BuildInput {
                    lex: lex.into(),
                    unk: unk.into(),
                    char_def: char_def.into(),
                    connection: connection.into(),
                    features: features.split(',').map(String::from).collect(),
                    usage,
                    top,
                },
                Path::new(out),
            )?;
            eprintln!(
                "{} words, {:.2} MB -> {} in {:.1} s",
                built.words,
                built.bytes as f64 / 1e6,
                built.manifest.display(),
                t.elapsed().as_secs_f64()
            );
        }
        ["alternates", lex, connection, col, out] => {
            let t = Instant::now();
            let (keys, entries) = setsuna_analyzer::alternates::build(
                Path::new(lex),
                Path::new(connection),
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
        _ => bail!("usage: analyzer alternates <lex.csv> <matrix.def | bigram dir> <reading column> <out> | analyze <manifest> | usage <manifest> <texts.txt> <out.tsv> | build <lex.csv> <unk.def> <char.def> <matrix.def | bigram dir> <features> <out dir> [<usage.tsv> <top>]"),
    }
    Ok(())
}
