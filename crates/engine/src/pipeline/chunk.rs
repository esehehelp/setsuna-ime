//! Long readings: the model reads at most `max_seq_len` tokens ([CLS],
//! context, [SEP], reading), and 99% of its training readings fit in 88
//! chars. A reading of up to `MAX_CHUNK` chars goes whole, sentences and
//! all; a longer one is cut after each sentence end, and a sentence still
//! too long into chunks; they are converted one after another with the
//! output before each as its context.

use anyhow::Result;

use crate::candidate::Candidate;
use crate::pipeline::symsplit::{self, Seg, Vary};
use crate::pipeline::{digitsplit, symtoken};

/// Readings longer than this many chars are converted in chunks, and no
/// chunk is longer. 99% of the training readings fit in 88 chars.
pub const MAX_CHUNK: usize = 80;
/// Where a chunk with no comma to end at is cut.
const HARD_CUT: usize = 60;
/// A chunk ends at a comma only when it is at least this long.
const MIN_COMMA_CUT: usize = 20;

fn is_sentence_end(c: char) -> bool {
    matches!(c, '。' | '．' | '！' | '？' | '!' | '?')
}

/// What may follow a sentence end in the same chunk: more of them, and
/// closing brackets (`。」`).
fn closes_sentence(c: char) -> bool {
    is_sentence_end(c) || matches!(c, '」' | '』' | '）' | ')' | '】' | '〕')
}

fn is_comma(c: char) -> bool {
    matches!(c, '、' | '，')
}

/// Whether the sentence end at `i` goes on as one word with what follows
/// it: a decimal point (`３．５`), or an ASCII ! or ? before an ASCII char
/// (`?q=1` of a URL, `ok!ok`).
fn goes_on(chars: &[char], i: usize) -> bool {
    let digit =
        |c: Option<&char>| c.is_some_and(|c| c.is_ascii_digit() || ('０'..='９').contains(c));
    let next = chars.get(i + 1);
    match chars[i] {
        '．' => i > 0 && digit(chars.get(i - 1)) && digit(next),
        c if c.is_ascii() => next.is_some_and(|c| c.is_ascii_graphic()),
        _ => false,
    }
}

/// `reading` cut into chunks of at most `max` chars: whole when it fits,
/// else one per sentence (cut after 。！？ and the closing brackets after
/// it, but where it goes on as one word), a sentence still longer than
/// `max` cut after its last comma within `max`, else at `HARD_CUT`. Past
/// `max`, a chunk's cut depends only on the text up to it, so typing on
/// changes the last chunk alone.
///
/// A short reading of several sentences went a sentence after another
/// (2026-10-20 to 2026-10-24): each after the top 1 of the ones before, the
/// list varying the last. Whole, the beam varies every sentence: over
/// wiki_fresh_v2's 1,055 such lines with Shiran-v2 s7080, gold in the
/// engine's top 20 878 -> 924 of them, EM1 in the IME's pipeline over the
/// top 20 81.39 -> 82.09 (p = 8e-5), keeping the 3 best of each earlier
/// sentence 81.77, and the conversion 9.8 -> 7.4 ms at p50 in one pass.
pub fn split(reading: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = reading.chars().collect();
    let max = max.max(1);
    if chars.len() <= max {
        return vec![reading.to_string()];
    }
    let mut out = Vec::new();
    let mut start = 0;
    // in a sentence end and the closers after it
    let mut ending = false;
    for i in 0..chars.len() {
        ending = is_sentence_end(chars[i]) || (ending && closes_sentence(chars[i]));
        if ending && !chars.get(i + 1).is_some_and(|&c| closes_sentence(c)) && !goes_on(&chars, i) {
            split_long(&chars[start..=i], max, &mut out);
            start = i + 1;
            ending = false;
        }
    }
    split_long(&chars[start..], max, &mut out);
    out
}

/// A sentence as chunks of at most `max` chars.
fn split_long(mut rest: &[char], max: usize, out: &mut Vec<String>) {
    while rest.len() > max {
        let cut = rest[..max]
            .iter()
            .rposition(|&c| is_comma(c))
            .map(|i| i + 1)
            .filter(|&n| n >= MIN_COMMA_CUT)
            .unwrap_or(HARD_CUT.min(max));
        out.push(rest[..cut].iter().collect());
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        out.push(rest.iter().collect());
    }
}

/// Apply the model's token budget as well as the normal character cuts.
/// Byte-fallback characters stay whole; none of the reading is truncated.
/// With `symbol_runs` (symbol token), a run of symbols takes the slots of
/// one [`symtoken::SYMBOL_TOKEN`], as the model reads it, and is never cut
/// by the budget (its chars after the first take none). With
/// `digit_split`, a run of digits takes none: it never reaches the model in
/// the reading (`digitsplit`).
pub fn split_with_tokens(
    reading: &str,
    max_chars: usize,
    max_tokens: usize,
    symbol_runs: bool,
    digit_split: bool,
    token_len: impl Fn(char) -> usize,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for chunk in split(reading, max_chars) {
        // Runs as each chunk is tokenized on its own.
        let chars: Vec<char> = chunk.chars().collect();
        let mut text = String::new();
        let mut used = 0;
        for (i, &c) in chars.iter().enumerate() {
            let n = if digit_split && digitsplit::in_digit_run(&chars, i) {
                0
            } else if !(symbol_runs && symtoken::is_symbol_at(&chars, i)) {
                token_len(c)
            } else if i > 0 && symtoken::is_symbol_at(&chars, i - 1) {
                0
            } else {
                token_len(symtoken::SYMBOL_TOKEN)
            };
            anyhow::ensure!(n <= max_tokens, "model sequence cannot hold reading character {c:?} ({n} tokens, budget {max_tokens})");
            if used + n > max_tokens {
                out.push(std::mem::take(&mut text));
                used = 0;
            }
            text.push(c);
            used += n;
        }
        if !text.is_empty() || reading.is_empty() {
            out.push(text);
        }
    }
    Ok(out)
}

/// Convert `chunks` in order with `convert_chunk`, each after the top-1 of
/// the ones before it (on `context`); the candidates vary the last chunk.
pub fn convert_chunks(
    context: &str,
    chunks: &[String],
    pool_limit: usize,
    convert_chunk: impl FnMut(&str, &str) -> Result<Vec<Candidate>>,
) -> Result<Vec<Candidate>> {
    let segs: Vec<Seg> = chunks.iter().cloned().map(Seg::Kana).collect();
    symsplit::splice(context, &segs, pool_limit, Vary::Last, convert_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::CandidateSource;

    fn lens(chunks: &[String]) -> Vec<usize> {
        chunks.iter().map(|c| c.chars().count()).collect()
    }

    #[test]
    fn token_budget_preserves_every_character() {
        let reading = "ゕゖ".repeat(30);
        let chunks = split_with_tokens(&reading, 80, 126, false, false, |_| 3).unwrap();
        assert_eq!(lens(&chunks), [42, 18]);
        assert_eq!(chunks.concat(), reading);
        let chunks = split_with_tokens("a高橋b", 80, 4, false, false, |c| {
            if c.is_ascii() {
                1
            } else {
                3
            }
        })
        .unwrap();
        assert_eq!(chunks, ["a高", "橋b"]);
        assert!(split_with_tokens("高", 80, 2, false, false, |_| 3).is_err());
        assert_eq!(
            split_with_tokens("", 80, 0, false, false, |_| 1).unwrap(),
            [""]
        );
        // Spliced symbols take no neural input slots.
        assert_eq!(
            split_with_tokens("（）", 80, 0, false, false, |_| 0).unwrap(),
            ["（）"]
        );
    }

    #[test]
    fn a_run_of_symbols_takes_one_token_and_stays_whole() {
        // あ(())い: 6 tokens char by char, 3 as the model reads it (あ◇い).
        assert_eq!(
            split_with_tokens("あ(())い", 80, 4, true, false, |_| 1).unwrap(),
            ["あ(())い"]
        );
        assert_eq!(
            split_with_tokens("あ(())い", 80, 4, false, false, |_| 1).unwrap(),
            ["あ(()", ")い"]
        );
        // the budget never cuts inside a run; a number separator is no symbol
        assert_eq!(
            split_with_tokens("あい!!!!う", 80, 3, true, false, |_| 1).unwrap(),
            ["あい!!!!", "う"]
        );
        assert_eq!(
            split_with_tokens("1.5%%", 80, 3, true, false, |_| 1).unwrap(),
            ["1.5", "%%"]
        );
    }

    #[test]
    fn a_run_of_digits_takes_no_token_with_digit_split() {
        assert_eq!(
            split_with_tokens("あ1,000.5い", 80, 2, false, true, |_| 1).unwrap(),
            ["あ1,000.5い"]
        );
        assert_eq!(
            split_with_tokens("あ1,000.5い", 80, 2, false, false, |_| 1).unwrap(),
            ["あ1", ",0", "00", ".5", "い"]
        );
        // a trailing separator is no digit; a symbol after digits still counts
        assert_eq!(
            split_with_tokens("12.%う", 80, 2, true, true, |_| 1).unwrap(),
            ["12.%う"]
        );
        assert_eq!(
            split_with_tokens("12.。う", 80, 2, false, true, |_| 1).unwrap(),
            ["12.。", "う"]
        );
    }

    #[test]
    fn a_reading_that_fits_is_one_chunk() {
        assert_eq!(
            split("きょうは。いいてんき。", 80),
            ["きょうは。いいてんき。"]
        );
        assert_eq!(split("いいてんき", 80), ["いいてんき"]);
        assert_eq!(split("", 80), [""]);
    }

    #[test]
    fn past_the_limit_each_sentence_is_a_chunk() {
        assert_eq!(
            split("きょうは。いいてんき。", 8),
            ["きょうは。", "いいてんき。"]
        );
        assert_eq!(split("ほんとう？うん", 6), ["ほんとう？", "うん"]);
    }

    #[test]
    fn a_sentence_end_inside_a_word_is_no_cut() {
        assert_eq!(split("３．５ばい。そう", 7), ["３．５ばい。", "そう"]);
        assert_eq!(
            split("example.com/?q=1を。いい", 18),
            ["example.com/?q=1を。", "いい"]
        );
        assert_eq!(split("できた。2025ねん", 8), ["できた。", "2025ねん"]);
    }

    #[test]
    fn a_long_reading_is_cut_after_each_sentence() {
        let a = "あ".repeat(50) + "。";
        let b = "い".repeat(30) + "！？";
        let c = "う".repeat(10);
        let chunks = split(&format!("{a}{b}{c}"), 80);
        assert_eq!(chunks, [a, b, c]);
    }

    #[test]
    fn closing_brackets_stay_with_their_sentence() {
        let a = format!("「{}。」", "あ".repeat(50));
        let b = "い".repeat(40);
        assert_eq!(split(&format!("{a}{b}"), 80), [a, b]);
    }

    #[test]
    fn a_long_sentence_is_cut_after_its_last_comma_in_reach() {
        let a = "あ".repeat(30) + "、" + &"い".repeat(30) + "、";
        let b = "う".repeat(40) + "。";
        let chunks = split(&format!("{a}{b}"), 80);
        assert_eq!(chunks, [a, b]);
    }

    #[test]
    fn a_comma_too_early_is_not_a_cut() {
        let s = "あ".repeat(5) + "、" + &"い".repeat(120);
        assert_eq!(lens(&split(&s, 80)), [60, 66]);
    }

    #[test]
    fn no_chunk_is_longer_than_the_limit() {
        let s = "あ".repeat(300);
        let chunks = split(&s, 80);
        assert!(chunks.iter().all(|c| c.chars().count() <= 80));
        assert_eq!(chunks.concat(), s);
        // a model with a shorter sequence cuts at its own limit
        assert_eq!(lens(&split(&"あ".repeat(100), 40)), [40, 40, 20]);
    }

    #[test]
    fn typing_on_changes_only_the_last_chunk() {
        let base = "あ".repeat(50) + "。" + &"い".repeat(70) + "、" + &"う".repeat(20);
        let before = split(&base, 80);
        let after = split(&format!("{base}ええ"), 80);
        assert_eq!(before[..before.len() - 1], after[..before.len() - 1]);
        assert_eq!(after.len(), before.len());
    }

    fn cand(text: &str) -> Candidate {
        Candidate {
            text: text.to_string(),
            score: -1.0,
            source: CandidateSource::Beam,
            reading_pos: (0..text.chars().count() as u32).collect(),
        }
    }

    #[test]
    fn chunks_see_the_output_before_them_and_the_last_one_varies() {
        let mut calls = Vec::new();
        let chunks = ["あい。".to_string(), "うえ".to_string()];
        let out = convert_chunks("前", &chunks, 5, |context, reading| {
            calls.push((context.to_string(), reading.to_string()));
            Ok(match reading {
                "あい。" => vec![cand("愛。"), cand("藍。")],
                _ => vec![cand("植え"), cand("飢え")],
            })
        })
        .unwrap();
        assert_eq!(
            calls,
            [
                ("前".to_string(), "あい。".to_string()),
                ("前愛。".to_string(), "うえ".to_string())
            ]
        );
        let texts: Vec<_> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["愛。植え", "愛。飢え"]);
        // positions are shifted to each chunk's place in the reading
        assert_eq!(out[1].reading_pos, [0, 1, 3, 4]);
    }
}
