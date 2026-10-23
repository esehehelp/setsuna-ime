//! The reading lexicon (`setsuna_core::lexicon`): compile its table, look
//! candidates up.
//!
//! ```text
//! lexicon build <table.tsv> <out.lexicon>
//! lexicon cost <file.lexicon> <reading> <surface>...
//! ```
//!
//! `build` compiles `surface<TAB>reading<TAB>cost<TAB>class` lines into the
//! mapped file the IME reads; `cost` prints each surface's cheapest reading
//! as the reading, and the lexicon's entries for it.

use anyhow::{bail, Context, Result};
use setsuna_core::lexicon::{Lexicon, CLASSES};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") if args.len() == 3 => {
            let table = std::fs::read_to_string(&args[1]).with_context(|| args[1].clone())?;
            let t = std::time::Instant::now();
            let bytes = Lexicon::build(&table)?;
            std::fs::write(&args[2], &bytes).with_context(|| args[2].clone())?;
            let lex = Lexicon::open(std::path::Path::new(&args[2]))?;
            eprintln!(
                "{} entries, {:.1} MB, {:.1} s",
                lex.len(),
                bytes.len() as f64 / 1e6,
                t.elapsed().as_secs_f64()
            );
        }
        Some("cost") if args.len() >= 4 => {
            let lex = Lexicon::open(std::path::Path::new(&args[1]))?;
            let reading = &args[2];
            for surface in &args[3..] {
                let t = std::time::Instant::now();
                let cost = lex.cost(surface, reading);
                let us = t.elapsed().as_micros();
                let entries: Vec<String> = lex
                    .get(surface)
                    .iter()
                    .map(|e| format!("{} {:.2} {}", e.reading, e.cost, CLASSES[e.class as usize]))
                    .collect();
                println!("{surface}: {cost:?} ({us} µs) {entries:?}");
            }
        }
        _ => bail!("usage: lexicon build <table.tsv> <out> | lexicon cost <lexicon> <reading> <surface>..."),
    }
    Ok(())
}
