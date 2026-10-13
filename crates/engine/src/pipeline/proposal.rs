//! Proposal stage: the CTC-NAT graph's per-frame log-probabilities.
//!
//! The decoders only read, per frame, the best few tokens and the blank, so
//! the logits are reduced to that in one pass (log-sum-exp, then a top-k of
//! the log-probs) instead of materializing the whole (T, V) log-softmax.

use anyhow::Result;
use ort::session::Session;
use ort::value::{Tensor, ValueType};

use crate::pipeline::decode::{collapse_frame_ids, collapse_with_frames};
use crate::tokenizer::SharedCharTokenizer;

#[derive(Debug, Clone, Copy)]
pub struct ProposalFrame {
    pub top1_id: u32,
    pub top1_log_prob: f32,
    pub top2_log_prob: f32,
}

/// One frame as the decoders read it.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameLogProbs {
    /// The best (token id, log-prob), best first (ties: lower id first);
    /// blacklisted ids never appear.
    pub top: Vec<(u32, f32)>,
    /// The blank's log-prob; -inf when the blank is blacklisted.
    pub blank: f32,
}

#[derive(Debug)]
pub struct ProposalOutput {
    /// Per frame, the top-k (k = the decode's `top_k_per_step`, at least 2)
    /// and the blank.
    pub steps: Vec<FrameLogProbs>,
    pub frames: Vec<ProposalFrame>,
    pub input_len: usize,
}

/// How `log_probs_with` reduces each frame of the logits.
#[derive(Debug, Clone, Copy)]
pub struct FrameReduce {
    /// Row log-sum-exp: `lse::row` (vectorized) or `lse::exact` (the scalar
    /// loop, which the tests compare against).
    pub lse: fn(&[f32]) -> f32,
}

impl Default for FrameReduce {
    fn default() -> Self {
        Self {
            lse: crate::pipeline::lse::row,
        }
    }
}

impl ProposalOutput {
    pub fn collapsed_tokens(&self, blank_id: u32) -> Vec<u32> {
        collapse_frame_ids(
            &self.frames.iter().map(|f| f.top1_id).collect::<Vec<_>>(),
            blank_id,
        )
    }

    /// `collapsed_tokens` plus the frame each token was emitted at.
    pub fn collapsed_with_frames(&self, blank_id: u32) -> (Vec<u32>, Vec<u32>) {
        collapse_with_frames(
            &self.frames.iter().map(|f| f.top1_id).collect::<Vec<_>>(),
            blank_id,
        )
    }
}

fn fixed_input_seq_len(session: &Session) -> Option<usize> {
    let mut dims: Vec<usize> = Vec::new();
    for input in session.inputs() {
        if input.name() != "input_ids" && input.name() != "attention_mask" {
            continue;
        }
        match input.dtype() {
            ValueType::Tensor { shape, .. } if shape.len() >= 2 && shape[1] > 0 => {
                dims.push(shape[1] as usize);
            }
            _ => return None,
        }
    }
    if dims.is_empty() || dims.iter().any(|d| *d != dims[0]) {
        None
    } else {
        Some(dims[0])
    }
}

/// Build (input_ids, attention_mask, actual_len) for the proposal graph.
/// Layout: `[CLS] context[-max_context:] [SEP] reading`, padded to seq_len
/// with [PAD] / 0. (No trailing [SEP] after reading — matches the training
/// data tokenizer.)
pub fn encode(
    tokenizer: &SharedCharTokenizer,
    seq_len: usize,
    max_context: usize,
    context: &str,
    reading: &str,
) -> Result<(Vec<i64>, Vec<i64>, usize)> {
    anyhow::ensure!(seq_len >= 2, "model sequence must hold [CLS] and [SEP]");
    let ctx_chars: Vec<String> = chars(context);
    let take_ctx = ctx_chars.len().saturating_sub(max_context);
    let reading_chars = chars(reading);

    // Byte-level fallback for OOV chars, matching the Python tokenizer
    // used at train + eval time (`runtime.py:147`). Without this the
    // engine's context for rare kanji / cross-script chars drifts from
    // what the model was trained on (training: `<0xXX>` byte tokens;
    // engine was: single UNK).
    //
    // Layout mirrors the training encoder: reserve [CLS] + reading +
    // [SEP] first, then fill the remaining slot with context
    // (left-trimmed). Tail-truncating the whole [CLS]…[SEP]…reading
    // string would drop reading tokens whenever context + byte
    // fallback pushed total length past `seq_len`, putting the engine
    // in a state the model never saw at training time.
    let mut reading_ids: Vec<u32> = Vec::with_capacity(reading_chars.len());
    for ch in &reading_chars {
        reading_ids.extend(tokenizer.encode_with_byte_fallback(ch));
    }
    anyhow::ensure!(
        reading_ids.len() <= seq_len - 2,
        "reading exceeds model token budget: {} tokens, maximum {} (split the reading first)",
        reading_ids.len(),
        seq_len - 2
    );
    let ctx_iter = &ctx_chars[take_ctx..];
    let mut ctx_ids: Vec<u32> = Vec::with_capacity(ctx_iter.len());
    for ch in ctx_iter {
        ctx_ids.extend(tokenizer.encode_with_byte_fallback(ch));
    }
    // Left-trim context to fit the remaining budget after [CLS] +
    // reading + [SEP]. Byte fallback can produce more tokens than
    // chars, so we may still need to trim after expansion.
    let ctx_room = seq_len.saturating_sub(2 + reading_ids.len());
    if ctx_ids.len() > ctx_room {
        let drop = ctx_ids.len() - ctx_room;
        ctx_ids.drain(..drop);
    }
    let mut ids: Vec<i64> = Vec::with_capacity(seq_len);
    ids.push(tokenizer.cls_id as i64);
    for id in &ctx_ids {
        ids.push(*id as i64);
    }
    ids.push(tokenizer.sep_id as i64);
    for id in &reading_ids {
        ids.push(*id as i64);
    }
    let actual = ids.len();
    let mut mask: Vec<i64> = vec![0; seq_len];
    for slot in mask.iter_mut().take(actual) {
        *slot = 1;
    }
    while ids.len() < seq_len {
        ids.push(tokenizer.pad_id as i64);
    }
    Ok((ids, mask, actual))
}

/// For each frame of `encode`'s output (`ids[..actual]`), the index of the
/// reading char it holds; 0 for `[CLS]`, the context and `[SEP]`. A char
/// spelled in several byte-fallback tokens covers several frames.
pub fn frame_reading(
    tokenizer: &SharedCharTokenizer,
    ids: &[i64],
    actual: usize,
    reading: &str,
) -> Vec<u32> {
    let sep = ids[..actual]
        .iter()
        .skip(1)
        .position(|&id| id == tokenizer.sep_id as i64)
        .map_or(actual, |i| i + 2);
    let mut out = vec![0u32; sep.min(actual)];
    for (i, ch) in reading.chars().enumerate() {
        let mut buf = [0u8; 4];
        let n = tokenizer
            .encode_with_byte_fallback(ch.encode_utf8(&mut buf))
            .len();
        out.extend(std::iter::repeat_n(i as u32, n));
    }
    out.truncate(actual);
    out
}

fn chars(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

/// The proposal ONNX graph. Dynamic artifacts are fed at the actual input
/// length; fixed-shape ones at their declared length, sliced back after.
pub struct ProposalRunner {
    session: Session,
    fixed_seq_len: Option<usize>,
}

impl ProposalRunner {
    pub fn new(session: Session) -> Self {
        let fixed_seq_len = fixed_input_seq_len(&session);
        Self {
            session,
            fixed_seq_len,
        }
    }

    /// Forward proposal ONNX. Beam decoding keeps each frame's top-k
    /// log-probs and blank; greedy decoding needs only the argmax token ID.
    /// Returns input_len = T (no padding rows). Dynamic ONNX artifacts
    /// are fed at actual length. Legacy fixed-shape artifacts are fed at their
    /// declared sequence length, then sliced back to `actual` frames after
    /// forward.
    pub fn run(
        &mut self,
        tokenizer: &SharedCharTokenizer,
        encoded: (Vec<i64>, Vec<i64>, usize),
        vocab_blacklist: &[u32],
        top_k: usize,
        reduce: FrameReduce,
        greedy: bool,
    ) -> Result<ProposalOutput> {
        let (mut ids, mut mask, actual) = encoded;
        let forward_len = self.fixed_seq_len.unwrap_or(actual);
        ids.truncate(forward_len);
        mask.truncate(forward_len);
        // (shape, data) tensors: no dependency on ort's ndarray version.
        let ids_tensor =
            Tensor::from_array(([1usize, forward_len], ids)).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mask_tensor = Tensor::from_array(([1usize, forward_len], mask))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (shape, flat) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let dims = shape.as_ref();
        anyhow::ensure!(dims.len() == 3, "unexpected logits rank {}", dims.len());
        let vocab = dims[2] as usize;
        anyhow::ensure!(
            vocab == tokenizer.vocab_size(),
            "proposal logits vocab {vocab} != tokenizer vocab {}",
            tokenizer.vocab_size()
        );
        anyhow::ensure!(
            (tokenizer.blank_id as usize) < vocab,
            "blank_id {} out of range for vocab {vocab}",
            tokenizer.blank_id
        );
        anyhow::ensure!(
            flat.len() >= actual * vocab,
            "proposal logits buffer {} too small for {actual}x{vocab}",
            flat.len()
        );

        Ok(if greedy {
            greedy_from_logits(
                flat,
                actual,
                vocab,
                vocab_blacklist,
                tokenizer.blank_id,
                top_k,
                reduce,
            )
        } else {
            log_probs_with(
                flat,
                actual,
                vocab,
                vocab_blacklist,
                tokenizer.blank_id,
                top_k,
                reduce,
            )
        })
    }
}

/// Greedy CTC uses only the per-frame argmax, not probabilities or top-k.
/// Preserve the beam path for non-finite logits, near-ties (f32 subtraction
/// could collapse unequal logits to the same log-prob).
fn greedy_from_logits(
    flat: &[f32],
    actual: usize,
    vocab: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    top_k: usize,
    reduce: FrameReduce,
) -> ProposalOutput {
    let fallback = || {
        log_probs_with(
            flat,
            actual,
            vocab,
            vocab_blacklist,
            blank_id,
            top_k,
            reduce,
        )
    };
    let mut blocked = vec![false; vocab];
    for &id in vocab_blacklist {
        if let Some(slot) = blocked.get_mut(id as usize) {
            *slot = true;
        }
    }
    let mut frames = Vec::with_capacity(actual);
    for row in flat.chunks_exact(vocab).take(actual) {
        let mut best_id = None;
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for (id, &logit) in row.iter().enumerate() {
            // The full reducer includes blacklisted logits in its LSE, so
            // non-finite values there must also take the legacy path.
            if !logit.is_finite() {
                return fallback();
            }
            if blocked[id] {
                continue;
            }
            if best_id.is_none() || logit > best {
                second = best;
                best = logit;
                best_id = Some(id as u32);
            } else if logit > second {
                second = logit;
            }
        }
        let Some(top1_id) = best_id else {
            return fallback();
        };
        // Subtracting the row log-sum-exp can make two distinct f32 values
        // equal. The old top-k then chose the lower token ID. Only those
        // exceptionally close rows need the full log-softmax calculation.
        let gap = best - second;
        if gap > 0.0 && gap <= 4.0 * f32::EPSILON * (best.abs() + (vocab as f32).ln() + 1.0) {
            return fallback();
        }
        frames.push(ProposalFrame {
            top1_id,
            top1_log_prob: 0.0,
            top2_log_prob: f32::NEG_INFINITY,
        });
    }
    ProposalOutput {
        steps: Vec::new(),
        frames,
        input_len: actual,
    }
}

/// Log-softmax each frame of the proposal logits (blacklisted token ids
/// forced to -inf, their mass dropped rather than renormalized) and keep
/// what the decoders read: the top `top_k` (at least 2) and the blank, plus
/// the top-1 / top-2 metadata.
pub fn log_probs_from_logits(
    flat: &[f32],
    actual: usize,
    vocab: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    top_k: usize,
) -> ProposalOutput {
    log_probs_with(
        flat,
        actual,
        vocab,
        vocab_blacklist,
        blank_id,
        top_k,
        FrameReduce::default(),
    )
}

/// `log_probs_from_logits` with the frame reduction given.
pub fn log_probs_with(
    flat: &[f32],
    actual: usize,
    vocab: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    top_k: usize,
    reduce: FrameReduce,
) -> ProposalOutput {
    let k = top_k.max(2).min(vocab.max(1));
    let mut steps = Vec::with_capacity(actual);
    let mut frames = Vec::with_capacity(actual);
    // Blacklist mask: a bool table built once per proposal, avoiding a
    // per-(t, v) membership test in the hot loop.
    let mut is_blacklisted: Vec<bool> = vec![false; vocab];
    for &tok in vocab_blacklist {
        if (tok as usize) < vocab {
            is_blacklisted[tok as usize] = true;
        }
    }
    for t in 0..actual {
        let row_start = t * vocab;
        let row = &flat[row_start..row_start + vocab];
        let lse = (reduce.lse)(row);
        // Insertion top-k over the log-probs: an equal value goes after the
        // ones already kept, so ties keep the lower id first (as a strict
        // argmax does).
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (v, &x) in row.iter().enumerate() {
            if is_blacklisted[v] {
                continue;
            }
            let lp = x - lse;
            if top.len() == k && lp <= top[k - 1].1 {
                continue;
            }
            let at = top.partition_point(|&(_, y)| y >= lp);
            top.insert(at, (v as u32, lp));
            top.truncate(k);
        }
        let blank = if is_blacklisted
            .get(blank_id as usize)
            .copied()
            .unwrap_or(true)
        {
            f32::NEG_INFINITY
        } else {
            row[blank_id as usize] - lse
        };
        let (top1_id, top1_lp) = top.first().copied().unwrap_or((0, f32::NEG_INFINITY));
        let top2_lp = if vocab == 1 {
            top1_lp
        } else {
            top.get(1).map_or(f32::NEG_INFINITY, |e| e.1)
        };
        frames.push(ProposalFrame {
            top1_id,
            top1_log_prob: top1_lp,
            top2_log_prob: top2_lp,
        });
        steps.push(FrameLogProbs { top, blank });
    }

    ProposalOutput {
        steps,
        frames,
        input_len: actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn greedy_argmax_matches_full_reduction_with_blacklist_and_ties() {
        let logits = [
            0.0, 3.0, 3.0, 2.0, // tie: lower ID wins
            7.0, 2.0, 2.0, 0.0, // blacklisted argmax
            0.0, 1.0, 4.0, 3.0, // ordinary frame
        ];
        let full = log_probs_from_logits(&logits, 3, 4, &[0], 3, 16);
        let greedy = greedy_from_logits(&logits, 3, 4, &[0], 3, 16, FrameReduce::default());
        assert_eq!(
            greedy.frames.iter().map(|f| f.top1_id).collect::<Vec<_>>(),
            full.frames.iter().map(|f| f.top1_id).collect::<Vec<_>>()
        );
        assert_eq!(
            greedy.collapsed_with_frames(3),
            full.collapsed_with_frames(3)
        );
        assert!(greedy.steps.is_empty());
    }

    #[test]
    fn greedy_uses_full_reduction_for_nonfinite_or_near_ties() {
        for logits in [
            [1.0, f32::NAN, 0.0],
            [1.0, f32::INFINITY, 0.0],
            [1.0, 1.0 + f32::EPSILON, 0.0],
        ] {
            let greedy = greedy_from_logits(&logits, 1, 3, &[], 2, 16, FrameReduce::default());
            let full = log_probs_from_logits(&logits, 1, 3, &[], 2, 16);
            assert!(!greedy.steps.is_empty());
            assert_eq!(greedy.frames[0].top1_id, full.frames[0].top1_id);
        }
        let blacklisted_nan = [f32::NAN, 1.0, 0.0];
        let out = greedy_from_logits(&blacklisted_nan, 1, 3, &[0], 2, 16, FrameReduce::default());
        assert!(!out.steps.is_empty());
    }

    #[test]
    fn frame_reading_maps_frames_to_reading_chars() {
        let tok = tokenizer();
        let (ids, _, actual) = encode(&tok, 16, 64, "いい", "あい").unwrap();
        // [CLS] い い [SEP] あ い
        assert_eq!(
            frame_reading(&tok, &ids, actual, "あい"),
            [0, 0, 0, 0, 0, 1]
        );
    }

    #[test]
    fn encode_refuses_to_truncate_readings() {
        let tok = tokenizer();
        assert!(encode(&tok, 3, 64, "", "あい").is_err());
        assert!(encode(&tok, 1, 64, "", "").is_err());
    }

    #[test]
    fn encode_keeps_reading_and_left_trims_context() {
        let tok = tokenizer();
        // seq_len 6: [CLS] ctx.. [SEP] あい -> room for 2 context tokens.
        let (ids, mask, actual) = encode(&tok, 6, 64, "いいあ", "あい").unwrap();
        assert_eq!(ids, vec![2, 7, 6, 3, 6, 7]);
        assert_eq!(mask, vec![1; 6]);
        assert_eq!(actual, 6);
        // max_context trims before the seq_len budget does.
        let (ids, mask, actual) = encode(&tok, 8, 1, "いいあ", "あ").unwrap();
        assert_eq!(ids, vec![2, 6, 3, 6, 0, 0, 0, 0]);
        assert_eq!(mask, vec![1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(actual, 4);
    }

    #[test]
    fn log_probs_normalize_rows_and_mask_blacklist() {
        let flat = [0.0f32, 1.0, 5.0, 2.0, 2.0, -1.0];
        let out = log_probs_from_logits(&flat, 2, 3, &[2], 0, 3);
        assert_eq!(out.input_len, 2);
        for step in &out.steps {
            // the blacklisted id never shows up
            assert!(step.top.iter().all(|&(id, _)| id != 2));
            // The masked token's mass is dropped, not renormalized.
            let mass: f32 = step.top.iter().map(|&(_, lp)| lp.exp()).sum();
            assert!(mass < 1.0 + 1e-6);
            // best first
            assert!(step.top.windows(2).all(|w| w[0].1 >= w[1].1));
        }
        // the log-probs are the full row's log-softmax
        let lse0 = (0f32.exp() + 1f32.exp() + 5f32.exp()).ln();
        assert!((out.steps[0].blank - (0.0 - lse0)).abs() < 1e-6);
        assert_eq!(out.frames[0].top1_id, 1);
        assert!(out.frames[0].top1_log_prob > out.frames[0].top2_log_prob);
        // a tie keeps the lower id first
        assert_eq!(out.frames[1].top1_id, 0);
        assert_eq!(
            out.steps[1].top[..2]
                .iter()
                .map(|e| e.0)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(out.frames[1].top1_log_prob, out.frames[1].top2_log_prob);
        // a blacklisted blank is -inf
        let out = log_probs_from_logits(&flat, 2, 3, &[0], 0, 3);
        assert_eq!(out.steps[0].blank, f32::NEG_INFINITY);
    }
}
