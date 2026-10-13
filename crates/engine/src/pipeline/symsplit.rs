//! Symbol split: bracket / annotation symbols, and every char the model's
//! vocabulary has no token for, are taken out of the reading so the model
//! converts only the spans it can read; they are spliced back verbatim.
//! Without the latter, a vocabulary without `. , - : 〜` fed `6.2`, `2,727`,
//! `UTF-8`, `10:30` to the model with an `[UNK]` in them.

use std::collections::HashSet;

use anyhow::Result;

use crate::candidate::{Candidate, CandidateSource};

/// Parenthetical / annotation brackets and emphatic punctuation the CTC model
/// fails to emit reliably: at these
/// positions (no kana reading of their own) it has ~zero probability mass and
/// produces geta/PUA junk (〓∞♨└§…). `symbol_split` pulls these out of the
/// reading so the neural model only converts the kana spans, then splices them
/// back verbatim.
///
/// Deliberately EXCLUDES corner / angle quotes (「」『』〈〉《》): unlike a
/// parenthetical aside, quoted speech is inline, so splitting there fragments
/// the sentence into spans too short to convert well in isolation (は→ハ,
/// であって→出会って were observed regressions). Also NOT 、。・ー… — the model
/// converts those correctly and they are useful conversion cues.
pub(crate) fn is_strip_symbol(c: char) -> bool {
    matches!(
        c,
        '(' | ')' | '\u{FF08}' | '\u{FF09}'        // ( ) （ ）
            | '\u{3010}' | '\u{3011}'              // 【 】
            | '\u{3014}' | '\u{3015}'              // 〔 〕
            | '\u{FF3B}' | '\u{FF3D}' | '[' | ']'  // ［ ］ [ ]
            | '{' | '}' | '\u{FF5B}' | '\u{FF5D}' // { } ｛ ｝
            | '!' | '?' | '\u{FF01}' | '\u{FF1F}' // ! ? ！ ？
            // Symbols typed as symbols are output as typed; the model is not
            // asked to produce them (it had learned ぱーせんと -> %, ど -> °).
            | '%' | '\u{FF05}' | '\u{00B0}'        // % ％ °
            | '&' | '\u{FF06}' | '#' | '\u{FF03}'  // & ＆ # ＃
            | '@' | '\u{FF20}' | '~' | '\u{FF5E}'  // @ ＠ ~ ～
            | '*' | '\u{FF0A}' | '=' | '\u{FF1D}'  // * ＊ = ＝
            | '+' | '\u{FF0B}' | '/' | '\u{FF0F}'  // + ＋ / ／
            | '\u{00A5}' | '\u{FFE5}' | '$' | '\u{FF04}' // ¥ ￥ $ ＄
    )
}

/// Split the reading at the chars `is_split` picks (`is_strip_symbol`, and
/// the ones out of the vocabulary), convert each other segment with
/// `convert_span` using the accumulated output as context, and splice the
/// split chars back verbatim at the boundaries. N-best is produced by
/// varying the largest converted segment (where homophone ambiguity usually
/// lives), keeping the others at their top-1.
pub fn convert_split(
    context: &str,
    reading: &str,
    pool_limit: usize,
    is_split: impl Fn(char) -> bool,
    convert_span: impl FnMut(&str, &str) -> Result<Vec<Candidate>>,
) -> Result<Vec<Candidate>> {
    // Partition into ordered, alternating symbol / non-symbol segments.
    let mut segs: Vec<Seg> = Vec::new();
    let mut buf = String::new();
    let mut cur_sym: Option<bool> = None;
    for c in reading.chars() {
        let is_sym = is_split(c);
        if cur_sym == Some(is_sym) {
            buf.push(c);
        } else {
            if let Some(s) = cur_sym {
                let t = std::mem::take(&mut buf);
                segs.push(if s { Seg::Sym(t) } else { Seg::Kana(t) });
            }
            buf.push(c);
            cur_sym = Some(is_sym);
        }
    }
    if let Some(s) = cur_sym {
        segs.push(if s { Seg::Sym(buf) } else { Seg::Kana(buf) });
    }
    splice(context, &segs, pool_limit, Vary::Longest, convert_span)
}

/// A piece of a split reading: spliced back verbatim, or converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Seg {
    Sym(String),
    Kana(String),
}

/// The kana segment whose N-best the spliced candidates vary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vary {
    /// The longest (where homophone ambiguity usually lives).
    Longest,
    /// The last (the one being typed).
    Last,
}

/// Convert each kana segment of `segs` with `convert_span`, the output so
/// far as its context, and splice the symbols back verbatim. The other
/// segments stay at their top-1 while the candidates vary the one `vary`
/// picks.
pub(crate) fn splice(
    context: &str,
    segs: &[Seg],
    pool_limit: usize,
    vary: Vary,
    mut convert_span: impl FnMut(&str, &str) -> Result<Vec<Candidate>>,
) -> Result<Vec<Candidate>> {
    // Step 1: top-1 conversion of each kana segment with running context.
    // Track the varied kana segment plus the output prefix that preceded
    // it, for N-best splicing. Reading positions are assembled alongside: a
    // spliced symbol sits at its own reading char, a span's candidate
    // positions are shifted by the span's reading offset. `None` once a
    // piece has no positions.
    let mut running_ctx = context.to_string();
    let mut prefix = String::new(); // accumulated top-1 output
    let mut prefix_pos: Option<Vec<u32>> = Some(Vec::new());
    let mut offset = 0u32;
    let mut main: Option<(usize, Vec<Candidate>, String, Option<Vec<u32>>, u32)> = None;
    for seg in segs {
        match seg {
            Seg::Sym(s) => {
                prefix.push_str(s);
                running_ctx.push_str(s);
                let n = s.chars().count() as u32;
                if let Some(p) = prefix_pos.as_mut() {
                    p.extend(offset..offset + n);
                }
                offset += n;
            }
            Seg::Kana(s) => {
                let prefix_before = prefix.clone();
                let pos_before = prefix_pos.clone();
                let cands = convert_span(&running_ctx, s)?;
                let n = s.chars().count() as u32;
                let (top1, top1_pos) = cands.first().map_or_else(
                    || (s.clone(), Some((offset..offset + n).collect())),
                    |c| (c.text.clone(), shifted(c, offset)),
                );
                prefix.push_str(&top1);
                running_ctx.push_str(&top1);
                match (prefix_pos.as_mut(), top1_pos) {
                    (Some(p), Some(t)) => p.extend(t),
                    _ => prefix_pos = None,
                }
                let rlen = n as usize;
                if vary == Vary::Last || main.as_ref().is_none_or(|(best, _, _, _, _)| rlen > *best)
                {
                    main = Some((rlen, cands, prefix_before, pos_before, offset));
                }
                offset += n;
            }
        }
    }

    // `prefix` is now the full top-1 output.
    let full = prefix;
    let full_pos = prefix_pos;
    let mut out: Vec<Candidate> = Vec::with_capacity(pool_limit);
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(full.clone());
    // The composite top-1 candidate carries the main segment's top-1 score
    // and source (only the text differs, being the spliced-together output):
    // that segment is where the neural evidence lives; the symbol splices
    // contribute no score of their own. No main segment (an all-symbol
    // reading) means no neural evidence at all.
    let mut primary = main
        .as_ref()
        .and_then(|(_, c, _, _, _)| c.first().cloned())
        .unwrap_or_else(|| Candidate {
            text: String::new(),
            score: 0.0,
            source: CandidateSource::Fallback,
            reading_pos: Vec::new(),
        });
    primary.text = full.clone();
    primary.reading_pos = full_pos.clone().unwrap_or_default();
    out.push(primary);

    // N-best: re-splice the main segment's rank-2.. candidates around the
    // fixed prefix/suffix taken from the top-1 assembly. Each spliced
    // candidate keeps the main segment candidate's score and source.
    if let Some((_, cands, prefix_before, pos_before, main_offset)) = main {
        let main_top1 = cands.first().map(|c| c.text.clone()).unwrap_or_default();
        let consumed = prefix_before.chars().count() + main_top1.chars().count();
        let suffix: String = full.chars().skip(consumed).collect();
        let suffix_pos: Option<Vec<u32>> = full_pos
            .as_ref()
            .map(|p| p.iter().skip(consumed).copied().collect());
        for cand in cands.iter().skip(1) {
            if out.len() >= pool_limit {
                break;
            }
            let text = format!("{}{}{}", prefix_before, cand.text, suffix);
            if seen.insert(text.clone()) {
                let mut spliced = cand.clone();
                spliced.text = text;
                spliced.reading_pos = match (&pos_before, shifted(cand, main_offset), &suffix_pos) {
                    (Some(b), Some(m), Some(s)) => [b.as_slice(), &m, s].concat(),
                    _ => Vec::new(),
                };
                out.push(spliced);
            }
        }
    }

    Ok(out)
}

/// A span candidate's reading positions moved to the span's offset in the
/// whole reading, or `None` when the candidate has none.
fn shifted(c: &Candidate, offset: u32) -> Option<Vec<u32>> {
    if c.text.is_empty() {
        return Some(Vec::new());
    }
    (c.reading_pos.len() == c.text.chars().count())
        .then(|| c.reading_pos.iter().map(|p| p + offset).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_symbols_include_emphatic_punctuation_not_sentence_punctuation() {
        for c in [
            '?', '!', '？', '！', '（', '）', '%', '％', '°', '&', '#', '@', '~', '～', '/', '＋',
            '￥', '$',
        ] {
            assert!(is_strip_symbol(c), "{c} should be split out");
        }
        for c in ['、', '。', '・', 'ー', '「', '」'] {
            assert!(!is_strip_symbol(c), "{c} should stay in model input");
        }
    }

    fn cand(text: &str, score: f32) -> Candidate {
        Candidate {
            text: text.to_string(),
            score,
            source: CandidateSource::Beam,
            // one position per char, all at the span's first reading char
            reading_pos: vec![0; text.chars().count()],
        }
    }

    #[test]
    fn reading_positions_are_assembled_across_spans() {
        let out = convert_split("", "かな（い）", 5, is_strip_symbol, |_, reading| {
            Ok(match reading {
                "かな" => vec![
                    Candidate {
                        reading_pos: vec![0, 1],
                        ..cand("仮名", -1.0)
                    },
                    Candidate {
                        reading_pos: vec![0],
                        ..cand("金", -2.0)
                    },
                ],
                _ => vec![cand("胃", -0.5)],
            })
        })
        .unwrap();
        // か な （ い ）
        assert_eq!(out[0].text, "仮名（胃）");
        assert_eq!(out[0].reading_pos, [0, 1, 2, 3, 4]);
        assert_eq!(out[1].text, "金（胃）");
        assert_eq!(out[1].reading_pos, [0, 2, 3, 4]);
    }

    #[test]
    fn splits_spans_and_resplices_the_longest_one() {
        let mut calls = Vec::new();
        let out = convert_split(
            "前",
            "かなかな（い）",
            5,
            is_strip_symbol,
            |context, reading| {
                calls.push((context.to_string(), reading.to_string()));
                Ok(match reading {
                    "かなかな" => vec![cand("金々", -1.0), cand("仮名々", -2.0)],
                    _ => vec![cand("胃", -0.5), cand("井", -0.7)],
                })
            },
        )
        .unwrap();
        // Each kana span sees the output so far as context.
        assert_eq!(
            calls,
            [
                ("前".to_string(), "かなかな".to_string()),
                ("前金々（".to_string(), "い".to_string())
            ]
        );
        let texts: Vec<_> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["金々（胃）", "仮名々（胃）"]);
        // Candidates keep the main span's score; only the text changes.
        assert_eq!(out[1].score, -2.0);
    }

    #[test]
    fn pool_limit_caps_the_resplice() {
        let out = convert_split("", "かな（）", 1, is_strip_symbol, |_, _| {
            Ok(vec![cand("金", -1.0), cand("仮名", -2.0)])
        })
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "金（）");
    }

    #[test]
    fn a_caller_picked_char_is_split_and_spliced_back() {
        // `.` out of the vocabulary: 6 and 2 are converted apart, the `.`
        // comes back verbatim.
        let mut calls = Vec::new();
        let out = convert_split(
            "",
            "6.2ぱーせんと",
            5,
            |c| is_strip_symbol(c) || c == '.',
            |_, reading| {
                calls.push(reading.to_string());
                Ok(match reading {
                    "6" => vec![cand("6", -0.1)],
                    _ => vec![cand("2パーセント", -0.2), cand("2ぱーせんと", -3.0)],
                })
            },
        )
        .unwrap();
        assert_eq!(calls, ["6", "2ぱーせんと"]);
        let texts: Vec<_> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["6.2パーセント", "6.2ぱーせんと"]);
    }

    #[test]
    fn all_symbol_reading_has_no_neural_evidence() {
        let out = convert_split("", "（）", 5, is_strip_symbol, |_, _| unreachable!()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "（）");
        assert_eq!(out[0].source, CandidateSource::Fallback);
    }
}
