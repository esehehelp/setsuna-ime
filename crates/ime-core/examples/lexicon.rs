//! The reading lexicon (`setsuna_core::lexicon`): compile its table, look
//! candidates up.
//!
//! ```text
//! lexicon build [--max-cost <nats>] <table.tsv> [connection.txt] <out.lexicon>
//! lexicon cost <file.lexicon> <reading> <surface>...
//! lexicon convert <file.lexicon> <reading>...
//! lexicon words <file.lexicon> <reading>...
//! lexicon split <file.lexicon> <reading> <text>
//! ```
//!
//! `build` compiles the table and the connection costs (the module docs say
//! what they hold) into the mapped file the IME reads (with `--max-cost`,
//! the words up to that cost only, the file marked partial); `cost` prints each
//! surface's cheapest reading as the reading, and the lexicon's entries for
//! it; `convert` the lexicon's own conversions of each reading; `words` the
//! words it lists for a reading of one word (`word_candidates`) and, for a
//! reading of one or two kana, its one-char words (`char_words`); `split`
//! where each char of a kanji run starts in its reading (`split_reading`).
//! `SETSUNA_CONNECTION_WEIGHT` weighs the connection costs otherwise.

use anyhow::{bail, Context, Result};
use setsuna_core::lexicon::{Lexicon, CLASSES};

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // build --max-cost <nats>: the commoner words only, a partial lexicon
    let mut max_cost: Option<f32> = None;
    if args.first().map(String::as_str) == Some("build")
        && args.get(1).map(String::as_str) == Some("--max-cost")
    {
        let v = args.get(2).context("--max-cost <nats>")?;
        max_cost = Some(v.parse().with_context(|| format!("--max-cost {v}"))?);
        args.drain(1..3);
    }
    match args.first().map(String::as_str) {
        Some("build") if args.len() == 3 || args.len() == 4 => {
            let table = std::fs::read_to_string(&args[1]).with_context(|| args[1].clone())?;
            let connection = match args.len() {
                4 => Some(std::fs::read_to_string(&args[2]).with_context(|| args[2].clone())?),
                _ => None,
            };
            let out = args.last().unwrap();
            let t = std::time::Instant::now();
            let bytes = match max_cost {
                Some(m) => Lexicon::build_partial(&table, connection.as_deref(), m)?,
                None => Lexicon::build(&table, connection.as_deref())?,
            };
            std::fs::write(out, &bytes).with_context(|| out.clone())?;
            let lex = Lexicon::open(std::path::Path::new(out))?;
            eprintln!(
                "{} entries{}, {:.1} MB, {:.1} s",
                lex.len(),
                if lex.partial() { " (partial)" } else { "" },
                bytes.len() as f64 / 1e6,
                t.elapsed().as_secs_f64()
            );
        }
        Some("cost") if args.len() >= 4 => {
            let lex = open(&args[1])?;
            let reading = &args[2];
            for surface in &args[3..] {
                let t = std::time::Instant::now();
                let cost = lex.cost(surface, reading);
                let us = t.elapsed().as_micros();
                let entries: Vec<String> = lex
                    .get(surface)
                    .iter()
                    .map(|e| {
                        let class = CLASSES[e.class as usize];
                        format!("{} {:.2} {class} {}/{}", e.reading, e.cost, e.lid, e.rid)
                    })
                    .collect();
                println!("{surface}: {cost:?} ({us} µs) {entries:?}");
            }
        }
        Some("convert") if args.len() >= 3 => {
            let lex = open(&args[1])?;
            for reading in &args[2..] {
                let t = std::time::Instant::now();
                let out = lex.convert(reading, 10);
                let us = t.elapsed().as_micros();
                let shown: Vec<String> = out.iter().map(|(s, c)| format!("{s} {c:.1}")).collect();
                println!("{reading} ({us} µs): {shown:?}");
            }
        }
        Some("words") if args.len() >= 3 => {
            let lex = open(&args[1])?;
            for reading in &args[2..] {
                let t = std::time::Instant::now();
                let out = lex.word_candidates(reading);
                let us = t.elapsed().as_micros();
                let shown: Vec<String> = out.iter().map(|(s, c)| format!("{s} {c:.1}")).collect();
                println!("{reading} ({us} µs): {shown:?}");
                let chars: Vec<String> = lex
                    .char_words(reading)
                    .iter()
                    .map(|(s, c)| format!("{s} {c:.1}"))
                    .collect();
                if !chars.is_empty() {
                    println!("  one-char words: {chars:?}");
                }
            }
        }
        Some("split") if args.len() == 4 => {
            let lex = open(&args[1])?;
            let starts = lex.split_reading(&args[3], &args[2]);
            println!("{} {}: {starts:?}", args[2], args[3]);
        }
        _ => bail!(
            "usage: lexicon build <table.tsv> [connection.txt] <out> | lexicon cost <lexicon> <reading> <surface>... \
             | lexicon convert <lexicon> <reading>... | lexicon words <lexicon> <reading>... \n             | lexicon split <lexicon> <reading> <text>"
        ),
    }
    Ok(())
}

fn open(path: &str) -> Result<Lexicon> {
    let mut lex = Lexicon::open(std::path::Path::new(path))?;
    if let Ok(w) = std::env::var("SETSUNA_CONNECTION_WEIGHT") {
        lex.set_connection_weight(w.parse().context("SETSUNA_CONNECTION_WEIGHT")?);
    }
    Ok(lex)
}
