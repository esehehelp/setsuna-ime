//! Segments of conversions: JSON lines {"reading", "text", "pos", "words"?}
//! (`pos` the engine's reading position of each char, `words` the engine's
//! analysis of the text) on stdin -> {"starts", "segments"} on stdout, each
//! segment [reading start, reading end, text start, text end]: by the
//! analysis (`setsuna_core::bunsetsu`) where it is given and fits, else by
//! the char alignment (`setsuna_core::segments`).

use std::io::{BufRead, Write};

use anyhow::{Context, Result};
use setsuna_core::engine_client::Word;
use setsuna_core::segments::{char_starts, segment};

fn main() -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let reading = v["reading"].as_str().context("reading")?;
        let text = v["text"].as_str().context("text")?;
        let pos: Option<Vec<u32>> = serde_json::from_value(v["pos"].clone()).ok();
        let words: Option<Vec<Word>> = serde_json::from_value(v["words"].clone()).ok();
        let mut starts = char_starts(reading, text, pos.as_deref());
        let segs = match (&starts, &words) {
            (Some(s), Some(w)) => setsuna_core::bunsetsu::cut(text, reading, w, s),
            _ => None,
        };
        let segs = match segs {
            Some((fixed, segs)) => {
                starts = Some(fixed);
                segs
            }
            None => starts
                .as_ref()
                .map(|s| segment(text, s, reading.chars().count()))
                .unwrap_or_default(),
        };
        let segments: Vec<[usize; 4]> = segs
            .iter()
            .map(|g| [g.reading.0, g.reading.1, g.text.0, g.text.1])
            .collect();
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"starts": starts, "segments": segments}),
        )?;
        out.write_all(b"\n")?;
    }
    Ok(())
}
