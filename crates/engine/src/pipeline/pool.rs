//! Decoded hypotheses to candidates: detokenize, normalize (when asked), drop empty and
//! duplicate surfaces (first occurrence wins, so decode order is kept). An
//! empty result falls back to the proposal's greedy output, then to the raw
//! reading and its katakana form, so a host always gets something to show.

use std::collections::HashSet;

use crate::candidate::{identity_pos, Candidate, CandidateSource};
use crate::normalize::normalize_candidate;
use crate::pipeline::decode::BeamHypothesis;
use crate::pipeline::proposal::ProposalOutput;
use crate::tokenizer::SharedCharTokenizer;

/// The most hypotheses whose duplicates are found by comparing a text with
/// the ones kept (a beam's: no allocation, and faster than hashing that few);
/// more go through a hash set.
const DEDUP_BY_SCAN: usize = 32;

/// Whether `normalize_candidate` may change a char of `s`: an ASCII `,` `.`
/// `?` `!`, or a char of U+F000..=U+FFFF (the lead byte 0xEF: the full-width
/// forms it maps are U+FF01..=U+FF5E). A text without any comes back from it
/// as it went in, so it is not sent through (most candidates: kana and
/// kanji); the tests hold this to `normalize_candidate` over every char.
#[inline]
fn may_normalize(s: &str) -> bool {
    s.bytes()
        .any(|b| matches!(b, b',' | b'.' | b'?' | b'!' | 0xEF))
}

/// `raw` as a candidate's text: normalized with `normalize`, else as decoded;
/// a whitespace-only text is empty either way. The flag says whether the text
/// still has `raw`'s chars one for one (not when `normalize_candidate` changed
/// their number, or the text was emptied).
fn candidate_text(raw: String, normalize: bool) -> (String, bool) {
    if normalize && may_normalize(&raw) {
        let text = normalize_candidate(&raw);
        let same = text.chars().count() == raw.chars().count();
        (text, same)
    } else if raw.chars().all(char::is_whitespace) {
        (String::new(), raw.is_empty())
    } else {
        (raw, true)
    }
}

/// Text of `tokens` (normalized with `normalize`, else as decoded; a
/// whitespace-only text is empty either way) and each char's reading
/// position. `normalize_candidate` maps char for char, so the positions
/// carry over; they are dropped if it ever does not.
fn aligned_text(
    tokenizer: &SharedCharTokenizer,
    tokens: &[u32],
    frames: &[u32],
    frame_reading: &[u32],
    normalize: bool,
) -> (String, Vec<u32>) {
    // one frame a char of the decoded text
    let (raw, mut pos) = tokenizer.decode_aligned(tokens, frames);
    let (text, aligned) = candidate_text(raw, normalize);
    if aligned {
        let last = frame_reading.last().copied().unwrap_or(0);
        for p in pos.iter_mut() {
            *p = frame_reading.get(*p as usize).copied().unwrap_or(last);
        }
    } else {
        pos.clear();
    }
    (text, pos)
}

/// Candidates of one span, best first. `frame_reading` maps a proposal
/// frame to its reading char (`proposal::frame_reading`); `normalize`:
/// `normalize_candidate` on each text (the IME), else the decoded text.
pub fn candidates(
    hyps: Vec<BeamHypothesis>,
    greedy: bool,
    proposal: &ProposalOutput,
    reading: &str,
    tokenizer: &SharedCharTokenizer,
    frame_reading: &[u32],
    normalize: bool,
) -> Vec<Candidate> {
    let source = if greedy {
        CandidateSource::Greedy
    } else {
        CandidateSource::Beam
    };
    let scan = hyps.len() <= DEDUP_BY_SCAN;
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Candidate> = Vec::with_capacity(hyps.len());
    for h in hyps {
        let (text, reading_pos) =
            aligned_text(tokenizer, &h.tokens, &h.frames, frame_reading, normalize);
        if text.is_empty() {
            continue;
        }
        let new = if scan {
            !out.iter().any(|c| c.text == text)
        } else {
            seen.insert(text.clone())
        };
        if new {
            out.push(Candidate {
                text,
                score: h.score,
                source,
                reading_pos,
                parts: None,
            });
        }
    }
    if out.is_empty() {
        let (tokens, frames) = proposal.collapsed_with_frames(tokenizer.blank_id);
        let (greedy_text, reading_pos) =
            aligned_text(tokenizer, &tokens, &frames, frame_reading, normalize);
        if !greedy_text.is_empty() {
            out.push(Candidate {
                text: greedy_text,
                score: 0.0,
                source: CandidateSource::Greedy,
                reading_pos,
                parts: None,
            });
        }
    }
    if out.is_empty() {
        out = fallback(reading);
    }
    out
}

/// The candidates of a span decoding produced nothing for: the reading and
/// its katakana form (when it differs).
pub fn fallback(reading: &str) -> Vec<Candidate> {
    let mut out = vec![Candidate {
        text: reading.to_string(),
        score: 0.0,
        source: CandidateSource::Fallback,
        reading_pos: identity_pos(reading),
        parts: None,
    }];
    let kata = hira_to_kata(reading);
    if kata != reading {
        out.push(Candidate {
            reading_pos: identity_pos(&kata),
            text: kata,
            score: 0.0,
            source: CandidateSource::Fallback,
            parts: None,
        });
    }
    out
}

/// Text of an AR hypothesis's ids: decoded, normalized with `normalize`
/// (else as decoded; a whitespace-only text is empty either way).
pub fn ar_text(tokenizer: &SharedCharTokenizer, ids: &[u32], normalize: bool) -> String {
    candidate_text(tokenizer.decode(ids), normalize).0
}

fn hira_to_kata(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{3041}'..='\u{3096}' => char::from_u32(c as u32 + 0x60).unwrap_or(c),
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::proposal::ProposalFrame;

    fn tokenizer_of(names: &[&str]) -> SharedCharTokenizer {
        let id_to_token: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let token_to_id = id_to_token
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        SharedCharTokenizer {
            id_to_token,
            token_to_id,
            pad_id: 0,
            unk_id: 1,
            cls_id: 2,
            sep_id: 3,
            blank_id: 4,
            mask_id: 5,
            input_chars: None,
        }
    }

    fn tokenizer() -> SharedCharTokenizer {
        tokenizer_of(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "あ", "い", "亜",
        ])
    }

    fn proposal(top1: &[u32]) -> ProposalOutput {
        ProposalOutput {
            steps: Vec::new(),
            frames: top1
                .iter()
                .map(|&id| ProposalFrame {
                    top1_id: id,
                    top1_log_prob: 0.0,
                    top2_log_prob: 0.0,
                })
                .collect(),
            input_len: top1.len(),
        }
    }

    fn beam(tokens: &[u32], score: f32) -> BeamHypothesis {
        BeamHypothesis {
            tokens: tokens.to_vec(),
            score,
            frames: (0..tokens.len() as u32).collect(),
        }
    }

    /// One frame per reading char in these tests.
    const FRAMES: &[u32] = &[0, 1, 2, 3];

    #[test]
    fn candidates_carry_reading_positions() {
        let tok = tokenizer();
        let mut h = beam(&[8, 7], -1.0);
        h.frames = vec![0, 2];
        let out = candidates(
            vec![h],
            false,
            &proposal(&[4]),
            "ああい",
            &tok,
            FRAMES,
            false,
        );
        assert_eq!(
            (out[0].text.as_str(), out[0].reading_pos.clone()),
            ("亜い", vec![0, 2])
        );
        // fallbacks align char for char
        let out = candidates(
            Vec::new(),
            false,
            &proposal(&[4, 4]),
            "あい",
            &tok,
            FRAMES,
            false,
        );
        assert_eq!(out[0].reading_pos, [0, 1]);
        assert_eq!(out[1].reading_pos, [0, 1]);
    }

    #[test]
    fn hypotheses_are_deduped_in_decode_order() {
        let tok = tokenizer();
        let hyps = vec![beam(&[6, 7], -1.0), beam(&[6, 7], -2.0), beam(&[7], -3.0)];
        let out = candidates(hyps, false, &proposal(&[4]), "あい", &tok, FRAMES, false);
        let got: Vec<_> = out.iter().map(|c| (c.text.as_str(), c.score)).collect();
        assert_eq!(got, [("あい", -1.0), ("い", -3.0)]);
        assert!(out.iter().all(|c| c.source == CandidateSource::Beam));
    }

    /// A beam's worth of hypotheses (duplicates found by comparing) and more
    /// (by a hash set) keep the same: the first of each text, in decode order.
    #[test]
    fn many_hypotheses_are_deduped_as_few() {
        let tok = tokenizer();
        // the texts of 1..=3 tokens of あ い 亜, each several times over
        let all: Vec<BeamHypothesis> = (0..200u32)
            .map(|i| {
                let n = 1 + (i % 3) as usize;
                let tokens: Vec<u32> = (0..n as u32)
                    .map(|k| 6 + (i / 3 + k * (i % 5)) % 3)
                    .collect();
                beam(&tokens, -(i as f32))
            })
            .collect();
        for count in [1, 5, DEDUP_BY_SCAN, DEDUP_BY_SCAN + 1, 200] {
            let out = candidates(
                all[..count].to_vec(),
                false,
                &proposal(&[4]),
                "あい",
                &tok,
                FRAMES,
                false,
            );
            let mut want: Vec<(String, f32)> = Vec::new();
            for h in &all[..count] {
                let text = tok.decode(&h.tokens);
                if !want.iter().any(|w| w.0 == text) {
                    want.push((text, h.score));
                }
            }
            let got: Vec<(String, f32)> = out.iter().map(|c| (c.text.clone(), c.score)).collect();
            assert_eq!(got, want, "{count} hypotheses");
            assert!(count < 100 || got.len() < count);
        }
    }

    #[test]
    fn empty_decode_falls_back_to_greedy_then_reading() {
        let tok = tokenizer();
        let out = candidates(
            vec![beam(&[], -1.0)],
            false,
            &proposal(&[6, 4, 7]),
            "あい",
            &tok,
            FRAMES,
            false,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].text.as_str(), out[0].source),
            ("あい", CandidateSource::Greedy)
        );

        // An all-blank proposal decodes to nothing: raw reading + katakana.
        let out = candidates(
            vec![beam(&[], -1.0)],
            false,
            &proposal(&[4, 4]),
            "あい",
            &tok,
            FRAMES,
            false,
        );
        let texts: Vec<_> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["あい", "アイ"]);
        assert!(out.iter().all(|c| c.source == CandidateSource::Fallback));
    }

    #[test]
    fn greedy_decode_is_tagged_greedy() {
        let tok = tokenizer();
        let out = candidates(
            vec![beam(&[8], 0.0)],
            true,
            &proposal(&[8]),
            "あ",
            &tok,
            FRAMES,
            false,
        );
        assert_eq!(
            (out[0].text.as_str(), out[0].source),
            ("亜", CandidateSource::Greedy)
        );
    }

    /// Normalized or not, a text keeps its chars' reading positions, and a
    /// whitespace-only one is no candidate.
    #[test]
    fn normalized_texts_keep_their_reading_positions() {
        let tok = tokenizer_of(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "あ", ",", "Ａ", " ", "東京",
        ]);
        let frame_reading = [0, 0, 1, 2, 3];
        let text = |tokens: &[u32], normalize: bool| {
            let frames: Vec<u32> = (1..=tokens.len() as u32).collect();
            aligned_text(&tok, tokens, &frames, &frame_reading, normalize)
        };
        // nothing to normalize: the text as decoded, either way
        for normalize in [false, true] {
            assert_eq!(
                text(&[6, 10, 6], normalize),
                ("あ東京あ".to_string(), vec![0, 1, 1, 2])
            );
            assert_eq!(text(&[9, 9], normalize), (String::new(), Vec::new()));
            assert_eq!(text(&[], normalize), (String::new(), Vec::new()));
            // a frame past the reading's takes its last position
            assert_eq!(
                aligned_text(&tok, &[6], &[9], &frame_reading, normalize),
                ("あ".to_string(), vec![3])
            );
        }
        assert_eq!(
            text(&[6, 7, 8], false),
            ("あ,Ａ".to_string(), vec![0, 1, 2])
        );
        assert_eq!(text(&[6, 7, 8], true), ("あ、A".to_string(), vec![0, 1, 2]));
        assert_eq!(
            text(&[8, 7, 9, 6], true),
            ("A, あ".to_string(), vec![0, 1, 2, 3])
        );
        assert_eq!(ar_text(&tok, &[6, 7, 8], true), "あ、A");
        assert_eq!(ar_text(&tok, &[6, 7, 8], false), "あ,Ａ");
        assert_eq!(ar_text(&tok, &[6, 10], true), "あ東京");
        assert_eq!(ar_text(&tok, &[9], true), "");
        assert_eq!(ar_text(&tok, &[9], false), "");
    }

    /// `may_normalize` misses no char `normalize_candidate` changes: every
    /// char, alone and after a letter (the low ones and the full-width forms
    /// after a digit, a kana and a space too), comes back as it went in
    /// whenever the text is said to need no normalizing (and the chars it
    /// does change are said to).
    #[test]
    fn texts_left_alone_are_what_normalizing_leaves() {
        let mut changed = 0usize;
        for c in (0..=char::MAX as u32).filter_map(char::from_u32) {
            let near = (c as u32) < 0x3100 || (0xFF00..0xFFF0).contains(&(c as u32));
            let contexts: &[&str] = if near {
                &["", "a", "7", "あ", " "]
            } else {
                &["", "a"]
            };
            for &before in contexts {
                let mut s = String::with_capacity(8);
                s.push_str(before);
                s.push(c);
                let normalized = normalize_candidate(&s);
                if may_normalize(&s) {
                    changed += (normalized != s) as usize;
                    continue;
                }
                let want = if s.chars().all(char::is_whitespace) {
                    ""
                } else {
                    s.as_str()
                };
                assert_eq!(normalized, want, "{s:?}");
                assert_eq!(candidate_text(s.clone(), true), candidate_text(s, false));
            }
        }
        // `,` `.` `?` `!` and the full-width forms
        assert!(changed > 90 * 4);
    }
}
