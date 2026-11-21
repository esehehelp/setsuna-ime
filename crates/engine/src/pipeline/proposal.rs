//! Proposal stage: the CTC-NAT graph's per-frame log-probabilities.
//!
//! The decoders only read, per frame, the best few tokens and the blank, so
//! the logits are reduced to that in one pass (log-sum-exp, then a top-k of
//! the log-probs) instead of materializing the whole (T, V) log-softmax.

use std::time::Instant;

use anyhow::Result;

use crate::pipeline::ctc_score::TokenLogProbs;
use crate::pipeline::decode::{collapse_frame_ids, collapse_with_frames};
use crate::pipeline::native_encoder::NativeEncoder;
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
    /// Row log-sum-exp: `lse::peaked` (the default: vectorized, leaving out
    /// the chunks 24 nats below the frame's best, most of a CTC frame),
    /// `lse::row` (vectorized, every term) or `lse::exact` (the scalar loop,
    /// which the tests compare against).
    pub lse: fn(&[f32]) -> f32,
}

impl Default for FrameReduce {
    fn default() -> Self {
        Self {
            lse: crate::pipeline::lse::peaked,
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

/// Build (input_ids, attention_mask, actual_len) for the proposal graph.
/// Layout: `[CLS] context [SEP] reading`, padded to seq_len with [PAD] / 0
/// (no trailing [SEP] after reading), as the training rows are laid out
/// (tests/fixtures/model_input):
/// each char encoded as the rows read it
/// (`SharedCharTokenizer::encode_with_byte_fallback`), the context's last
/// `max_context` tokens (tokens, not chars: a byte-spelled char counts each
/// of its bytes), the reading reserved first and the context left-trimmed to
/// the room left in `seq_len`.
pub fn encode(
    tokenizer: &SharedCharTokenizer,
    seq_len: usize,
    max_context: usize,
    context: &str,
    reading: &str,
) -> Result<(Vec<i64>, Vec<i64>, usize)> {
    anyhow::ensure!(seq_len >= 2, "model sequence must hold [CLS] and [SEP]");
    let mut reading_ids: Vec<u32> = Vec::new();
    for ch in &chars(reading) {
        reading_ids.extend(tokenizer.encode_with_byte_fallback(ch));
    }
    anyhow::ensure!(
        reading_ids.len() <= seq_len - 2,
        "reading exceeds model token budget: {} tokens, maximum {} (split the reading first)",
        reading_ids.len(),
        seq_len - 2
    );
    // every char is at least one token: the last max_context chars hold the
    // last max_context tokens
    let ctx_chars = chars(context);
    let mut ctx_ids: Vec<u32> = Vec::new();
    for ch in &ctx_chars[ctx_chars.len().saturating_sub(max_context)..] {
        ctx_ids.extend(tokenizer.encode_with_byte_fallback(ch));
    }
    let keep = max_context.min(seq_len - 2 - reading_ids.len());
    if ctx_ids.len() > keep {
        ctx_ids.drain(..ctx_ids.len() - keep);
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

/// The proposal graph's computation: the engine's own encoder (`pipeline::native_encoder`, the manifest's
/// `proposal.native`), fed one row at its actual length.
pub struct ProposalRunner {
    encoder: Box<NativeEncoder>,
    /// The last `run`'s time in the encoder (its logits) and in reducing the logits to the frames' top-k, in ms (the
    /// engine's perf split of the proposal stage; `ort_ms` in the daemon's perf, the name it has always had).
    pub last_ort_ms: f64,
    pub last_reduce_ms: f64,
    /// The vocabulary id of each logit column (ascending, the blank among them) when the graph writes only those
    /// (manifest `proposal.logit_ids`; a reading corrector's only); none: a column an id.
    logit_ids: Option<Vec<u32>>,
}

impl ProposalRunner {
    pub fn new(encoder: Box<NativeEncoder>) -> Self {
        Self {
            encoder,
            last_ort_ms: 0.0,
            last_reduce_ms: 0.0,
            logit_ids: None,
        }
    }

    /// The runner of a graph whose logit columns are `ids` (see the field; checked against the vocabulary by the
    /// loader).
    pub fn with_logit_ids(mut self, ids: Option<Vec<u32>>) -> Self {
        self.logit_ids = ids;
        self
    }

    /// The vocabulary ids of the logit columns, when they are not the whole vocabulary.
    pub fn logit_ids(&self) -> Option<&[u32]> {
        self.logit_ids.as_deref()
    }

    /// The logits' width: the vocabulary's, or the columns'.
    fn width(&self, tokenizer: &SharedCharTokenizer) -> usize {
        self.logit_ids
            .as_ref()
            .map_or(tokenizer.vocab_size(), |c| c.len())
    }

    fn whole_vocabulary(&self) -> Result<()> {
        anyhow::ensure!(
            self.logit_ids.is_none(),
            "this graph's logits are a part of the vocabulary (manifest proposal.logit_ids): it runs only as a \
             reading corrector"
        );
        Ok(())
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
        self.whole_vocabulary()?;
        self.forward(
            tokenizer,
            encoded,
            tokenizer.vocab_size(),
            |flat, actual, vocab| {
                if greedy {
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
                }
            },
        )
    }

    /// The graph run as [`Self::run`] runs it, each frame reduced to the log-probs of the blank and of `ids` (sorted,
    /// distinct; blacklisted ones -inf) instead of its top-k: what a CTC forward over sequences of those ids reads
    /// (`ctc_score`). The reduce time is the gathering's.
    pub fn token_log_probs(
        &mut self,
        tokenizer: &SharedCharTokenizer,
        encoded: (Vec<i64>, Vec<i64>, usize),
        vocab_blacklist: &[u32],
        ids: &[u32],
        reduce: FrameReduce,
    ) -> Result<TokenLogProbs> {
        anyhow::ensure!(
            ids.iter().all(|&id| (id as usize) < tokenizer.vocab_size()),
            "a token id to score is out of the vocabulary ({} tokens)",
            tokenizer.vocab_size()
        );
        self.whole_vocabulary()?;
        self.forward(
            tokenizer,
            encoded,
            tokenizer.vocab_size(),
            |flat, actual, vocab| {
                gather_log_probs(
                    flat,
                    actual,
                    vocab,
                    vocab_blacklist,
                    tokenizer.blank_id,
                    ids,
                    reduce,
                )
            },
        )
    }

    /// One run of the graph for a reading corrector (`pipeline::correct`): [`greedy_scores`] of its logits. The
    /// reduce time is the argmax's and the gathering's.
    pub fn greedy_log_probs(
        &mut self,
        tokenizer: &SharedCharTokenizer,
        encoded: (Vec<i64>, Vec<i64>, usize),
        vocab_blacklist: &[u32],
        extra: &[u32],
    ) -> Result<(Vec<u32>, TokenLogProbs)> {
        anyhow::ensure!(
            extra
                .iter()
                .all(|&id| (id as usize) < tokenizer.vocab_size()),
            "a token id to score is out of the vocabulary ({} tokens)",
            tokenizer.vocab_size()
        );
        let blank_id = tokenizer.blank_id;
        // the columns lent to the reduction while the graph runs on `self`
        let width = self.width(tokenizer);
        let columns = self.logit_ids.take();
        let out = self.forward(tokenizer, encoded, width, |flat, actual, width| {
            greedy_scores(
                flat,
                actual,
                width,
                vocab_blacklist,
                blank_id,
                extra,
                columns.as_deref(),
            )
        });
        self.logit_ids = columns;
        out
    }

    /// The encoder over `encoded`, its logits checked (`width` columns: the vocabulary's), and `reduce` applied to
    /// them as (flat, frames, width); the times in the encoder and in `reduce` kept as `last_ort_ms` /
    /// `last_reduce_ms`.
    fn forward<R>(
        &mut self,
        tokenizer: &SharedCharTokenizer,
        encoded: (Vec<i64>, Vec<i64>, usize),
        width: usize,
        reduce: impl FnOnce(&[f32], usize, usize) -> R,
    ) -> Result<R> {
        let (mut ids, mask, actual) = encoded;
        let enc = &mut self.encoder;
        // one row, no padding: the encoder reads the ids alone
        ids.truncate(actual);
        anyhow::ensure!(
            mask.len() >= actual && mask[..actual].iter().all(|&m| m == 1),
            "the native encoder takes a row without padding"
        );
        // the logits a column a vocabulary id, or (a head cut to `logit_ids`) a column an id of those
        anyhow::ensure!(
            enc.config.vocab == tokenizer.vocab_size() && enc.logit_width() == Some(width),
            "native encoder logits {:?} wide over {} tokens, the tokenizer's vocabulary {} (columns {width})",
            enc.logit_width(),
            enc.config.vocab,
            tokenizer.vocab_size()
        );
        anyhow::ensure!(
            width != tokenizer.vocab_size() || (tokenizer.blank_id as usize) < width,
            "blank_id {} out of range for vocab {width}",
            tokenizer.blank_id
        );
        let t0 = Instant::now();
        let flat = enc.logits(&ids)?;
        anyhow::ensure!(
            flat.len() >= actual * width,
            "proposal logits buffer {} too small for {actual}x{width}",
            flat.len()
        );
        self.last_ort_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let reduce_t0 = Instant::now();
        let reduced = reduce(&flat, actual, width);
        self.last_reduce_ms = reduce_t0.elapsed().as_secs_f64() * 1000.0;
        Ok(reduced)
    }
}

/// Per frame, the log-softmax of the blank and of `ids` (sorted, distinct; each below `vocab`), the row normalized as
/// [`log_probs_with`] normalizes it (the blacklisted ids in its sum, themselves -inf).
pub fn gather_log_probs(
    flat: &[f32],
    actual: usize,
    vocab: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    ids: &[u32],
    reduce: FrameReduce,
) -> TokenLogProbs {
    gather_less(
        flat,
        actual,
        vocab,
        vocab_blacklist,
        blank_id,
        ids,
        |id| ((id as usize) < vocab).then_some(id as usize),
        |_, row| (reduce.lse)(row),
    )
}

/// Per frame t, the logits of the blank and of `ids` (sorted, distinct) less `norm(t, row)`, each id read from its
/// column `col(id)` of a row `width` wide; an id blacklisted or without a column -inf.
#[allow(clippy::too_many_arguments)]
fn gather_less(
    flat: &[f32],
    actual: usize,
    width: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    ids: &[u32],
    col: impl Fn(u32) -> Option<usize>,
    mut norm: impl FnMut(usize, &[f32]) -> f32,
) -> TokenLogProbs {
    let col = |id: u32| {
        if vocab_blacklist.contains(&id) {
            None
        } else {
            col(id)
        }
    };
    let blank_col = col(blank_id);
    let id_cols: Vec<Option<usize>> = ids.iter().map(|&id| col(id)).collect();
    let mut blank = Vec::with_capacity(actual);
    let mut cols = Vec::with_capacity(actual * ids.len());
    for (t, row) in flat.chunks_exact(width).take(actual).enumerate() {
        let less = norm(t, row);
        blank.push(blank_col.map_or(f32::NEG_INFINITY, |c| row[c] - less));
        for c in &id_cols {
            cols.push(c.map_or(f32::NEG_INFINITY, |c| row[c] - less));
        }
    }
    TokenLogProbs {
        ids: ids.to_vec(),
        blank,
        cols,
    }
}

/// A reading corrector's reduction of its logits (`pipeline::correct`): the greedy output (each frame's best token
/// but the blacklisted ones, the first of equal logits, the blank where every logit is -inf or NaN; collapsed:
/// repeats merged, blanks dropped) and, per frame, the logits of the blank and of the output's tokens and `extra`'s
/// (the reading as typed) less the frame's best logit, for `ctc_score`.
///
/// Those are the log-softmax's values plus a constant a frame (its log-sum-exp less its best logit), and every
/// alignment of every sequence passes each frame once, so a sequence's `ctc_score` log-likelihood is its
/// log-softmax one plus the same sum over the frames: the difference of two sequences' is the log-softmax's, without
/// a log-sum-exp over the vocabulary a frame.
///
/// `columns`: the vocabulary id of each of the `width` logit columns (ascending), when the graph writes only those
/// (manifest `proposal.logit_ids`); then the output is the best of those, and an id of `extra` without a column has
/// no alignment (-inf). Their logits are the whole head's (the same weights), so where a frame's best token is among
/// them the output and the margins are the whole head's.
pub fn greedy_scores(
    flat: &[f32],
    actual: usize,
    width: usize,
    vocab_blacklist: &[u32],
    blank_id: u32,
    extra: &[u32],
    columns: Option<&[u32]>,
) -> (Vec<u32>, TokenLogProbs) {
    let id_of = |c: usize| columns.map_or(c as u32, |ids| ids[c]);
    let col_of = |id: u32| match columns {
        None => ((id as usize) < width).then_some(id as usize),
        Some(ids) => ids.binary_search(&id).ok(),
    };
    let blocked_cols: Vec<usize> = vocab_blacklist
        .iter()
        .filter_map(|&id| col_of(id))
        .collect();
    let best: Vec<(u32, f32)> = if blocked_cols.is_empty() {
        flat.chunks_exact(width)
            .take(actual)
            .map(|row| {
                let max = crate::pipeline::lse::max(row);
                match row.iter().position(|&x| x == max) {
                    Some(c) if max > f32::NEG_INFINITY => (id_of(c), max),
                    _ => (blank_id, f32::NEG_INFINITY),
                }
            })
            .collect()
    } else {
        let mut blocked = vec![false; width];
        for c in blocked_cols {
            blocked[c] = true;
        }
        flat.chunks_exact(width)
            .take(actual)
            .map(|row| {
                let mut best = (None, f32::NEG_INFINITY);
                for (c, &logit) in row.iter().enumerate() {
                    // the first of equal logits, as an argmax over the row
                    if !blocked[c] && logit > best.1 {
                        best = (Some(c), logit);
                    }
                }
                (best.0.map_or(blank_id, id_of), best.1)
            })
            .collect()
    };
    let frame_ids: Vec<u32> = best.iter().map(|b| b.0).collect();
    let path = collapse_frame_ids(&frame_ids, blank_id);
    let mut ids: Vec<u32> = path.iter().chain(extra).copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let scores = gather_less(
        flat,
        actual,
        width,
        vocab_blacklist,
        blank_id,
        &ids,
        col_of,
        |t, _| {
            // a frame with nothing finite to read keeps its logits (its -inf stay -inf)
            if best[t].1.is_finite() {
                best[t].1
            } else {
                0.0
            }
        },
    );
    (path, scores)
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

/// The values of a row `log_probs_with`'s top-k tests at once once k are
/// kept (a chunk none of which would be kept is skipped whole).
const TOPK_CHUNK: usize = 32;

/// The start of the first chunk in `[at, full)` (chunks of `TOPK_CHUNK` from
/// `at`) holding a value the top-k would keep, `x - lse <= kth` false (NaN
/// included), else `full`. The test is a fixed-width loop of compares OR-ed
/// into a mask, compiled for AVX-512F and AVX2 and chosen at run time (the
/// fold over the chunk's bools did not vectorize: 17 ns a chunk of 32).
fn next_kept_chunk(row: &[f32], at: usize, full: usize, lse: f32, kth: f32) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::avx512() {
            // SAFETY: the CPU has AVX-512F
            return unsafe { next_kept_chunk_avx512(row, at, full, lse, kth) };
        }
        if crate::cpu::avx2() {
            // SAFETY: the CPU has AVX2
            return unsafe { next_kept_chunk_avx2(row, at, full, lse, kth) };
        }
        if crate::cpu::avx() {
            // SAFETY: the CPU has AVX
            return unsafe { next_kept_chunk_avx1(row, at, full, lse, kth) };
        }
    }
    next_kept_chunk_body(row, at, full, lse, kth)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn next_kept_chunk_avx512(row: &[f32], at: usize, full: usize, lse: f32, kth: f32) -> usize {
    next_kept_chunk_body(row, at, full, lse, kth)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn next_kept_chunk_avx2(row: &[f32], at: usize, full: usize, lse: f32, kth: f32) -> usize {
    next_kept_chunk_body(row, at, full, lse, kth)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn next_kept_chunk_avx1(row: &[f32], at: usize, full: usize, lse: f32, kth: f32) -> usize {
    next_kept_chunk_body(row, at, full, lse, kth)
}

#[inline(always)]
fn next_kept_chunk_body(row: &[f32], mut at: usize, full: usize, lse: f32, kth: f32) -> usize {
    while at < full {
        let c: &[f32; TOPK_CHUNK] = row[at..at + TOPK_CHUNK].try_into().expect("a full chunk");
        let mut kept = 0u32;
        for &x in c {
            kept |= u32::from(!(x - lse <= kth));
        }
        if kept != 0 {
            return at;
        }
        at += TOPK_CHUNK;
    }
    full
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
        // argmax does). Once k are kept, a chunk of the row none of whose
        // log-probs would be kept (each `lp <= the k-th`; a NaN stops the
        // skip, and the per-value test drops it) is skipped whole by one
        // vectorized test; the chunks with one go value by value as before,
        // so the result is the same.
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        let keep = |v: usize, x: f32, top: &mut Vec<(u32, f32)>| {
            let lp = x - lse;
            // a NaN (a broken model's output) is no log-prob: kept, it broke
            // the order the list is searched by and the beam sorted by
            if is_blacklisted[v] || lp.is_nan() {
                return;
            }
            if top.len() == k && lp <= top[k - 1].1 {
                return;
            }
            let at = top.partition_point(|&(_, y)| y >= lp);
            top.insert(at, (v as u32, lp));
            top.truncate(k);
        };
        let full = row.len() - row.len() % TOPK_CHUNK;
        let mut at = 0;
        while at < row.len() {
            if top.len() == k && at < full {
                at = next_kept_chunk(row, at, full, lse, top[k - 1].1);
            }
            let end = if at < full {
                at + TOPK_CHUNK
            } else {
                row.len()
            };
            for (i, &x) in row[at..end].iter().enumerate() {
                keep(at + i, x, &mut top);
            }
            at = end;
        }
        let blank = if is_blacklisted
            .get(blank_id as usize)
            .copied()
            .unwrap_or(true)
        {
            f32::NEG_INFINITY
        } else {
            // a NaN blank (a model gone wrong) is no path, as a NaN token is none
            let b = row[blank_id as usize] - lse;
            if b.is_nan() {
                f32::NEG_INFINITY
            } else {
                b
            }
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
            input_chars: None,
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

    /// The per-value insertion top-k `log_probs_with` had before it skipped
    /// chunks.
    fn top_k_per_value(row: &[f32], lse: f32, blacklist: &[bool], k: usize) -> Vec<(u32, f32)> {
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (v, &x) in row.iter().enumerate() {
            let lp = x - lse;
            if blacklist[v] || lp.is_nan() {
                continue;
            }
            if top.len() == k && lp <= top[k - 1].1 {
                continue;
            }
            let at = top.partition_point(|&(_, y)| y >= lp);
            top.insert(at, (v as u32, lp));
            top.truncate(k);
        }
        top
    }

    #[test]
    fn chunk_skipping_top_k_matches_the_per_value_one() {
        // Rows of a vocabulary not a multiple of the chunk, with many ties
        // (few distinct values), rising runs (the k-th moves inside a chunk),
        // a blacklist, k from 2 to 20.
        let vocab = 8003;
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut flat = Vec::with_capacity(6 * vocab);
        for t in 0..6 {
            for v in 0..vocab {
                flat.push(match t {
                    0 => (next() % 5) as f32,
                    1 => v as f32 * 1e-3,
                    2 => -(v as f32) * 1e-3,
                    _ => (next() % 100_000) as f32 * 1e-4 - 5.0,
                });
            }
        }
        // with the blacklist and without
        for ids in [&[3u32, 17, 4000, 8002][..], &[]] {
            let mut blacklist = vec![false; vocab];
            for &id in ids {
                blacklist[id as usize] = true;
            }
            for k in [2usize, 16, 20] {
                let out = log_probs_from_logits(&flat, 6, vocab, ids, 4, k);
                for t in 0..6 {
                    let row = &flat[t * vocab..(t + 1) * vocab];
                    let want =
                        top_k_per_value(row, crate::pipeline::lse::peaked(row), &blacklist, k);
                    assert_eq!(out.steps[t].top, want, "row {t}, k {k}, blacklist {ids:?}");
                }
            }
        }
        // a row with a NaN (no log-prob: dropped) and one with an infinity:
        // the per-value order holds, and no NaN is kept
        assert!(log_probs_from_logits(
            &{
                let mut r = flat[..vocab].to_vec();
                r[5] = f32::NAN;
                r
            },
            1,
            vocab,
            &[],
            4,
            16
        )
        .steps[0]
            .top
            .iter()
            .all(|&(_, lp)| !lp.is_nan()));
        // a NaN blank is no path either (the beam would carry the NaN on)
        let nan_blank = log_probs_from_logits(
            &{
                let mut r = flat[..vocab].to_vec();
                r[4] = f32::NAN;
                r
            },
            1,
            vocab,
            &[],
            4,
            16,
        );
        assert_eq!(nan_blank.steps[0].blank, f32::NEG_INFINITY);
        let mut odd = flat[..2 * vocab].to_vec();
        odd[100] = f32::NAN;
        odd[vocab + 200] = f32::INFINITY;
        let none = vec![false; vocab];
        let out = log_probs_from_logits(&odd, 2, vocab, &[], 4, 16);
        for t in 0..2 {
            let row = &odd[t * vocab..(t + 1) * vocab];
            let want = top_k_per_value(row, crate::pipeline::lse::peaked(row), &none, 16);
            let same = out.steps[t].top.len() == want.len()
                && out.steps[t]
                    .top
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits());
            assert!(same, "row {t} with a non-finite value");
        }
    }

    #[test]
    fn gathered_log_probs_are_the_reductions() {
        let flat = [0.0f32, 1.0, 5.0, 2.0, 2.0, -1.0, 0.5, 0.25];
        let full = log_probs_from_logits(&flat, 2, 4, &[2], 0, 4);
        let got = gather_log_probs(&flat, 2, 4, &[2], 0, &[1, 2, 3], FrameReduce::default());
        assert_eq!(got.ids, [1, 2, 3]);
        for t in 0..2 {
            assert_eq!(got.blank[t], full.steps[t].blank);
            for (k, id) in [1u32, 2, 3].into_iter().enumerate() {
                let want = full.steps[t]
                    .top
                    .iter()
                    .find(|e| e.0 == id)
                    .map_or(f32::NEG_INFINITY, |e| e.1);
                assert_eq!(got.cols[t * 3 + k], want, "frame {t}, id {id}");
            }
        }
    }

    #[test]
    fn greedy_scores_differ_from_log_probs_by_one_constant() {
        use crate::pipeline::ctc_score::Forward;
        let (vocab, frames, blank) = (300usize, 12usize, 4u32);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut flat: Vec<f32> = (0..frames * vocab)
            .map(|_| (next() % 100_000) as f32 * 2e-4 - 10.0)
            .collect();
        // a tie for the best (the first wins) and a NaN (skipped)
        flat[7] = 20.0;
        flat[9] = 20.0;
        flat[vocab + 3] = f32::NAN;
        let typed = [11u32, 12, 12, 40];
        for blacklist in [&[][..], &[12u32, 250][..]] {
            let (path, scores) =
                greedy_scores(&flat, frames, vocab, blacklist, blank, &typed, None);
            // the path: the scalar argmax over the frames, collapsed
            let mut blocked = vec![false; vocab];
            for &id in blacklist {
                blocked[id as usize] = true;
            }
            let argmax: Vec<u32> = flat
                .chunks_exact(vocab)
                .map(|row| {
                    let mut best = (blank, f32::NEG_INFINITY);
                    for (id, &x) in row.iter().enumerate() {
                        if !blocked[id] && x > best.1 {
                            best = (id as u32, x);
                        }
                    }
                    best.0
                })
                .collect();
            assert_eq!(argmax[0], 7, "the first of equal logits");
            assert_eq!(path, collapse_frame_ids(&argmax, blank));
            // the log-likelihoods: the log-softmax's plus one constant
            let lp = gather_log_probs(
                &flat,
                frames,
                vocab,
                blacklist,
                blank,
                &scores.ids,
                FrameReduce::default(),
            );
            let (f, g) = (Forward::new(&scores), Forward::new(&lp));
            let margin = |fw: &Forward| fw.log_likelihood(&path) - fw.log_likelihood(&typed);
            if blacklist.is_empty() {
                assert!(
                    (margin(&f) - margin(&g)).abs() < 1e-3,
                    "{} vs {}",
                    margin(&f),
                    margin(&g)
                );
            } else {
                // a blacklisted token of the reading as typed: no alignment
                assert_eq!(f.log_likelihood(&typed), f32::NEG_INFINITY);
                assert_eq!(g.log_likelihood(&typed), f32::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn greedy_scores_on_some_columns_are_the_whole_heads() {
        use crate::pipeline::ctc_score::Forward;
        let (vocab, frames, blank) = (300usize, 10usize, 4u32);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut flat: Vec<f32> = (0..frames * vocab)
            .map(|_| (next() % 100_000) as f32 * 2e-4 - 10.0)
            .collect();
        // the columns kept: every third id and the blank; each frame's best put on one of them (a tie on frame 0)
        let columns: Vec<u32> = (0..vocab as u32)
            .filter(|&id| id % 3 == 0 || id == blank)
            .collect();
        for t in 0..frames {
            flat[t * vocab + columns[(t * 7) % columns.len()] as usize] = 15.0;
        }
        flat[columns[5] as usize] = 15.0;
        let cut: Vec<f32> = flat
            .chunks_exact(vocab)
            .flat_map(|row| columns.iter().map(|&id| row[id as usize]))
            .collect();
        let typed = [columns[3], columns[9], columns[9]];
        let (path, whole) = greedy_scores(&flat, frames, vocab, &[], blank, &typed, None);
        let (path_cut, part) = greedy_scores(
            &cut,
            frames,
            columns.len(),
            &[],
            blank,
            &typed,
            Some(&columns),
        );
        assert_eq!(path, path_cut);
        assert_eq!(whole, part);
        let (f, g) = (Forward::new(&whole), Forward::new(&part));
        assert_eq!(f.log_likelihood(&typed), g.log_likelihood(&typed));
        // an id without a column: no alignment
        let (_, part) = greedy_scores(
            &cut,
            frames,
            columns.len(),
            &[],
            blank,
            &[1, 2],
            Some(&columns),
        );
        assert_eq!(
            Forward::new(&part).log_likelihood(&[1, 2]),
            f32::NEG_INFINITY
        );
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

#[cfg(test)]
mod timing {
    use super::*;
    use std::time::Instant;

    /// The 4M's reduction (T 30 frames, an 8k vocabulary, top-16): log-sum-exp of every row, the whole
    /// `log_probs_with`, and its top-k part alone (with the row's log-sum-exp given), median of 500.
    /// cargo test -p setsuna-engine --release --lib pipeline::proposal::timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn reduce_time() {
        let (t, v, k) = (30usize, 8192usize, 16usize);
        let mut seed = 11u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        // logits as a trained head gives them: most of the row low and spread, a few high
        let flat: Vec<f32> = (0..t * v)
            .map(|i| {
                let x = rnd() * 8.0 - 6.0;
                if i % 997 == 0 {
                    x + 14.0
                } else {
                    x
                }
            })
            .collect();
        let med = |mut s: Vec<f64>| {
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2] * 1e3
        };
        let mut a = Vec::new();
        let mut sink = 0.0f32;
        for _ in 0..500 {
            let t0 = Instant::now();
            for r in 0..t {
                sink += crate::pipeline::lse::row(&flat[r * v..(r + 1) * v]);
            }
            a.push(t0.elapsed().as_secs_f64());
        }
        let mut b = Vec::new();
        for _ in 0..500 {
            let t0 = Instant::now();
            let out = log_probs_from_logits(&flat, t, v, &[], 4, k);
            sink += out.steps[0].blank;
            b.push(t0.elapsed().as_secs_f64());
        }
        println!(
            "T {t}, V {v}, k {k}: lse rows {:.3} ms, log_probs_with {:.3} ms (sink {sink})",
            med(a),
            med(b)
        );
    }
}
