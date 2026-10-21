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
    let (raw, char_frames) = tokenizer.decode_aligned(tokens, frames);
    let text = if normalize {
        normalize_candidate(&raw)
    } else if raw.chars().all(char::is_whitespace) {
        String::new()
    } else {
        raw
    };
    let last = frame_reading.last().copied().unwrap_or(0);
    let pos = if text.chars().count() == char_frames.len() {
        char_frames
            .iter()
            .map(|&f| frame_reading.get(f as usize).copied().unwrap_or(last))
            .collect()
    } else {
        Vec::new()
    };
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
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Candidate> = Vec::with_capacity(hyps.len());
    for h in hyps {
        let (text, reading_pos) =
            aligned_text(tokenizer, &h.tokens, &h.frames, frame_reading, normalize);
        if !text.is_empty() && seen.insert(text.clone()) {
            out.push(Candidate {
                text,
                score: h.score,
                source,
                reading_pos,
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
            });
        }
    }
    if out.is_empty() {
        out.push(Candidate {
            text: reading.to_string(),
            score: 0.0,
            source: CandidateSource::Fallback,
            reading_pos: identity_pos(reading),
        });
        let kata = hira_to_kata(reading);
        if kata != reading {
            out.push(Candidate {
                reading_pos: identity_pos(&kata),
                text: kata,
                score: 0.0,
                source: CandidateSource::Fallback,
            });
        }
    }
    out
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

    fn tokenizer() -> SharedCharTokenizer {
        let names = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "あ", "い", "亜",
        ];
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
}
