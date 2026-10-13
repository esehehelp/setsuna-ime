//! CTC decoding: greedy collapse and prefix beam search.
//!
//! Prefix beam search keeps, per prefix, the log-probability mass of the CTC
//! paths ending in blank (`pb`) and in a non-blank (`pnb`), merges paths with
//! logsumexp, collapses repeats only across a blank, and bounds each step by
//! the top-K tokens of that frame. A hypothesis' score is
//! `logsumexp(pb, pnb)`, the total CTC probability of its token sequence.
//!
//! Each hypothesis also carries the frame that emitted each of its tokens,
//! taken from the most probable path merged into it. The input is one frame
//! per input token, so the frames say which reading characters a token was
//! produced from (the alignment a host segments a conversion with). They
//! never change the scores or the order.
//!
//! Prefixes live in a trie (a prefix is a node id; extending it is a child
//! lookup) and a hypothesis' token frames in a persistent list (extending
//! it is one push), so a step copies no token sequence. Over the 1603 rows of
//! the research general dev set this returned the same top-20 lists, frames
//! included, as the `HashMap<Vec<u32>, _>` implementation it replaced
//! (kept as `tests::reference`), 2.5x faster.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::pipeline::proposal::{FrameLogProbs, ProposalOutput};

const NEG_INF: f32 = f32::NEG_INFINITY;

fn logsumexp(a: f32, b: f32) -> f32 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    /// logsumexp(pb, pnb): the CTC probability mass of `tokens`.
    pub score: f32,
    /// The frame that emitted each token (first frame of its run), along
    /// the most probable path; one per token.
    pub frames: Vec<u32>,
}

/// FxHash-style hasher for the trie's (parent, token) keys.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// Prefix trie: node 0 is the empty prefix.
struct Trie {
    parent: Vec<u32>,
    token: Vec<u32>,
    children: HashMap<u64, u32, BuildHasherDefault<FxHasher>>,
}

impl Trie {
    fn new() -> Self {
        Self {
            parent: vec![0],
            token: vec![u32::MAX],
            children: HashMap::default(),
        }
    }

    fn child(&mut self, node: u32, token: u32) -> u32 {
        let key = ((node as u64) << 32) | token as u64;
        if let Some(&c) = self.children.get(&key) {
            return c;
        }
        let id = self.parent.len() as u32;
        self.parent.push(node);
        self.token.push(token);
        self.children.insert(key, id);
        id
    }

    fn last(&self, node: u32) -> Option<u32> {
        (node != 0).then(|| self.token[node as usize])
    }

    fn tokens(&self, mut node: u32) -> Vec<u32> {
        let mut out = Vec::new();
        while node != 0 {
            out.push(self.token[node as usize]);
            node = self.parent[node as usize];
        }
        out.reverse();
        out
    }
}

/// A prefix in the beam: CTC masses ending in blank / non-blank, and its
/// token frames (`frames`: a node of the persistent frame list) from the most
/// probable single path merged into it (`frames_logp`).
#[derive(Debug, Clone, Copy)]
struct State {
    node: u32,
    pb: f32,
    pnb: f32,
    frames: u32,
    frames_logp: f32,
}

impl State {
    fn score(&self) -> f32 {
        logsumexp(self.pb, self.pnb)
    }
}

/// The states of one step, merged by prefix (a slot per trie node, stamped
/// with the step so it needs no clearing).
struct Step {
    states: Vec<State>,
    slot: Vec<u32>,
    stamp: Vec<u32>,
}

impl Step {
    /// Add a path's (pb, pnb) mass to `node`. Its frames (built only when
    /// needed) replace the state's when this single contribution is the most
    /// probable one seen for the prefix.
    fn merge(&mut self, mark: u32, node: u32, pb: f32, pnb: f32, frames: impl FnOnce() -> u32) {
        let contribution = logsumexp(pb, pnb);
        let n = node as usize;
        if n >= self.stamp.len() {
            self.stamp.resize(n + 1, 0);
            self.slot.resize(n + 1, 0);
        }
        if self.stamp[n] == mark {
            let s = &mut self.states[self.slot[n] as usize];
            s.pb = logsumexp(s.pb, pb);
            s.pnb = logsumexp(s.pnb, pnb);
            if contribution > s.frames_logp {
                s.frames = frames();
                s.frames_logp = contribution;
            }
        } else {
            self.stamp[n] = mark;
            self.slot[n] = self.states.len() as u32;
            self.states.push(State {
                node,
                pb,
                pnb,
                frames: frames(),
                frames_logp: contribution,
            });
        }
    }
}

/// Run prefix beam search over the proposal's frames. Returns up to
/// `beam_width` hypotheses, best first; ties break by token sequence.
pub fn prefix_beam_search(
    steps: &[FrameLogProbs],
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
) -> Vec<BeamHypothesis> {
    assert!(beam_width > 0, "beam_width must be positive");
    let top_k = top_k_per_step.max(1);
    let mut trie = Trie::new();
    // persistent frame lists: (frame, previous); 0 is the empty list
    let mut frame_list: Vec<(u32, u32)> = vec![(0, 0)];
    let mut beam: Vec<State> = vec![State {
        node: 0,
        pb: 0.0,
        pnb: NEG_INF,
        frames: 0,
        frames_logp: 0.0,
    }];
    let mut next = Step {
        states: Vec::new(),
        slot: vec![0],
        stamp: vec![0],
    };

    for (t, step) in steps.iter().enumerate() {
        let mark = t as u32 + 1;
        let frame = t as u32;
        let candidates = &step.top[..top_k.min(step.top.len())];
        next.states.clear();
        for s in &beam {
            let (pb, pnb) = (s.pb, s.pnb);
            // Emit blank: the prefix is unchanged.
            next.merge(
                mark,
                s.node,
                logsumexp(pb, pnb) + step.blank,
                NEG_INF,
                || s.frames,
            );
            let last = trie.last(s.node);
            for &(c, c_logp) in candidates {
                if c == blank_id {
                    continue;
                }
                let child = trie.child(s.node, c);
                let mut extended = || {
                    frame_list.push((frame, s.frames));
                    frame_list.len() as u32 - 1
                };
                if last == Some(c) {
                    // A new occurrence of `c` needs a blank in between.
                    next.merge(mark, child, NEG_INF, pb + c_logp, &mut extended);
                    // Repeat collapse: same prefix, more non-blank mass.
                    next.merge(mark, s.node, NEG_INF, pnb + c_logp, || s.frames);
                } else {
                    let new_pnb = logsumexp(pb, pnb) + c_logp;
                    next.merge(mark, child, NEG_INF, new_pnb, &mut extended);
                }
            }
        }

        let by_rank = |a: &State, b: &State| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| trie.tokens(a.node).cmp(&trie.tokens(b.node)))
        };
        if next.states.len() > beam_width {
            next.states.select_nth_unstable_by(beam_width - 1, by_rank);
            next.states.truncate(beam_width);
        }
        std::mem::swap(&mut beam, &mut next.states);
    }

    let mut hyps: Vec<BeamHypothesis> = beam
        .iter()
        .map(|s| {
            let mut frames = Vec::new();
            let mut at = s.frames;
            while at != 0 {
                frames.push(frame_list[at as usize].0);
                at = frame_list[at as usize].1;
            }
            frames.reverse();
            BeamHypothesis {
                tokens: trie.tokens(s.node),
                score: s.score(),
                frames,
            }
        })
        .collect();
    hyps.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.tokens.cmp(&b.tokens))
    });
    hyps
}

/// CTC greedy collapse: drop blanks, drop consecutive non-blank repeats.
/// Blank acts as a separator: [7, blank, 7] -> [7, 7] but [7, 7] -> [7].
pub fn collapse_frame_ids(frame_ids: &[u32], blank_id: u32) -> Vec<u32> {
    collapse_with_frames(frame_ids, blank_id).0
}

/// `collapse_frame_ids` plus the frame each kept token was emitted at (the
/// first frame of its run).
pub fn collapse_with_frames(frame_ids: &[u32], blank_id: u32) -> (Vec<u32>, Vec<u32>) {
    let mut decoded: Vec<u32> = Vec::with_capacity(frame_ids.len());
    let mut frames: Vec<u32> = Vec::with_capacity(frame_ids.len());
    let mut last: Option<u32> = None;
    for (t, &token) in frame_ids.iter().enumerate() {
        if token != blank_id && Some(token) != last {
            decoded.push(token);
            frames.push(t as u32);
        }
        last = Some(token);
    }
    (decoded, frames)
}

/// Decode the proposal: a true CTC greedy collapse when `beam_width <= 1`
/// (per-frame argmax, then collapse; a width-1 prefix beam is NOT
/// equivalent), prefix beam search otherwise. The greedy hypothesis scores 0.
pub fn hypotheses(
    proposal: &ProposalOutput,
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
) -> Vec<BeamHypothesis> {
    if beam_width <= 1 {
        let (tokens, frames) = proposal.collapsed_with_frames(blank_id);
        vec![BeamHypothesis {
            tokens,
            score: 0.0,
            frames,
        }]
    } else {
        prefix_beam_search(&proposal.steps, blank_id, beam_width, top_k_per_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::lse;
    use crate::pipeline::proposal::{log_probs_from_logits, log_probs_with, FrameReduce};
    use ndarray::{arr2, Array2};

    /// The `HashMap<Vec<u32>, _>` prefix beam search the trie version
    /// replaced, over a full (T, V) log-prob matrix: the reference the
    /// property tests hold the trie version to.
    mod reference {
        use super::super::{logsumexp, BeamHypothesis, NEG_INF};
        use ndarray::Array2;
        use std::collections::HashMap;

        struct Cell {
            pb: f32,
            pnb: f32,
            frames: Vec<u32>,
            frames_logp: f32,
        }

        fn extend(prefix: &[u32], c: u32) -> Vec<u32> {
            let mut extended = prefix.to_vec();
            extended.push(c);
            extended
        }

        fn merge(
            beam: &mut HashMap<Vec<u32>, Cell>,
            prefix: Vec<u32>,
            pb: f32,
            pnb: f32,
            frames: impl FnOnce() -> Vec<u32>,
        ) {
            let contribution = logsumexp(pb, pnb);
            match beam.get_mut(&prefix) {
                Some(cell) => {
                    cell.pb = logsumexp(cell.pb, pb);
                    cell.pnb = logsumexp(cell.pnb, pnb);
                    if contribution > cell.frames_logp {
                        cell.frames = frames();
                        cell.frames_logp = contribution;
                    }
                }
                None => {
                    beam.insert(
                        prefix,
                        Cell {
                            pb,
                            pnb,
                            frames: frames(),
                            frames_logp: contribution,
                        },
                    );
                }
            }
        }

        pub fn prefix_beam_search(
            log_probs: &Array2<f32>,
            blank_id: u32,
            beam_width: usize,
            top_k_per_step: usize,
        ) -> Vec<BeamHypothesis> {
            let (t_len, vocab) = (log_probs.shape()[0], log_probs.shape()[1]);
            let top_k = top_k_per_step.min(vocab).max(1);
            let mut step_topk: Vec<Vec<(u32, f32)>> = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let row = log_probs.row(t);
                let mut idx: Vec<usize> = (0..vocab).collect();
                idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
                    row[b]
                        .partial_cmp(&row[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut top: Vec<(u32, f32)> =
                    idx[..top_k].iter().map(|&i| (i as u32, row[i])).collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                step_topk.push(top);
            }
            let mut beam: HashMap<Vec<u32>, Cell> = HashMap::new();
            beam.insert(
                Vec::new(),
                Cell {
                    pb: 0.0,
                    pnb: NEG_INF,
                    frames: Vec::new(),
                    frames_logp: 0.0,
                },
            );
            for t in 0..t_len {
                let mut next_beam: HashMap<Vec<u32>, Cell> = HashMap::new();
                let blank_logp = log_probs[[t, blank_id as usize]];
                let frame = t as u32;
                for (prefix, cell) in &beam {
                    let (pb, pnb) = (cell.pb, cell.pnb);
                    merge(
                        &mut next_beam,
                        prefix.clone(),
                        logsumexp(pb, pnb) + blank_logp,
                        NEG_INF,
                        || cell.frames.clone(),
                    );
                    for &(c, c_logp) in &step_topk[t] {
                        if c == blank_id {
                            continue;
                        }
                        if prefix.last().copied() == Some(c) {
                            merge(
                                &mut next_beam,
                                extend(prefix, c),
                                NEG_INF,
                                pb + c_logp,
                                || extend(&cell.frames, frame),
                            );
                            merge(
                                &mut next_beam,
                                prefix.clone(),
                                NEG_INF,
                                pnb + c_logp,
                                || cell.frames.clone(),
                            );
                        } else {
                            let new_pnb = logsumexp(pb, pnb) + c_logp;
                            merge(&mut next_beam, extend(prefix, c), NEG_INF, new_pnb, || {
                                extend(&cell.frames, frame)
                            });
                        }
                    }
                }
                let mut scored: Vec<_> = next_beam
                    .into_iter()
                    .map(|(p, cell)| {
                        let score = logsumexp(cell.pb, cell.pnb);
                        (p, cell, score)
                    })
                    .collect();
                scored.sort_by(|a, b| {
                    b.2.partial_cmp(&a.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                scored.truncate(beam_width);
                beam = scored.into_iter().map(|(p, cell, _)| (p, cell)).collect();
            }
            let mut hyps: Vec<BeamHypothesis> = beam
                .into_iter()
                .map(|(tokens, cell)| BeamHypothesis {
                    tokens,
                    score: logsumexp(cell.pb, cell.pnb),
                    frames: cell.frames,
                })
                .collect();
            hyps.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.tokens.cmp(&b.tokens))
            });
            hyps
        }

        /// The full log-softmax the engine used to build: the row's
        /// log-sum-exp in the same order, blacklisted ids at -inf.
        pub fn log_probs(
            flat: &[f32],
            t_len: usize,
            vocab: usize,
            blacklist: &[u32],
        ) -> Array2<f32> {
            let mut out = Array2::<f32>::zeros((t_len, vocab));
            for t in 0..t_len {
                let row = &flat[t * vocab..(t + 1) * vocab];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for &v in row {
                    sum += (v - max).exp();
                }
                let lse = max + sum.ln();
                for v in 0..vocab {
                    out[[t, v]] = if blacklist.contains(&(v as u32)) {
                        f32::NEG_INFINITY
                    } else {
                        row[v] - lse
                    };
                }
            }
            out
        }
    }

    /// Frames from a log-prob matrix (blank id 0), as the proposal gives them.
    fn steps(log_probs: &Array2<f32>, k: usize) -> Vec<FrameLogProbs> {
        let (t_len, vocab) = (log_probs.shape()[0], log_probs.shape()[1]);
        let flat: Vec<f32> = log_probs.iter().copied().collect();
        log_probs_from_logits(&flat, t_len, vocab, &[], 0, k).steps
    }

    fn beam(log_probs: &Array2<f32>, beam_width: usize, k: usize) -> Vec<BeamHypothesis> {
        prefix_beam_search(&steps(log_probs, k), 0, beam_width, k)
    }

    #[test]
    fn repeats_collapse_without_a_blank() {
        // Peak at 1, 1 (repeat), 2: CTC collapses to [1, 2].
        let logits = arr2(&[
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -3.0, -0.1, -3.0],
        ]);
        let out = beam(&logits, 4, 4);
        assert_eq!(out[0].tokens, vec![1, 2]);
    }

    #[test]
    fn blank_separated_repeats_expand() {
        let logits = arr2(&[[-3.0, -0.1, -3.0], [-0.1, -3.0, -3.0], [-3.0, -0.1, -3.0]]);
        let out = beam(&logits, 4, 3);
        assert_eq!(out[0].tokens, vec![1, 1]);
    }

    #[test]
    fn scores_are_sorted_and_the_beam_is_bounded() {
        let logits = arr2(&[
            [-1.0, -1.2, -1.5, -2.0],
            [-1.1, -1.0, -1.4, -2.5],
            [-0.9, -1.3, -1.2, -2.2],
        ]);
        let out = beam(&logits, 3, 4);
        assert!(out.len() <= 3);
        assert!(out.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn zero_top_k_per_step_is_clamped() {
        let logits = arr2(&[[-3.0, -0.1], [-0.1, -3.0]]);
        let out = prefix_beam_search(&steps(&logits, 2), 0, 2, 0);
        assert!(!out.is_empty());
    }

    #[test]
    fn hypotheses_carry_the_frame_of_each_token() {
        // frames: 0 -> 1, 1 -> 1 (repeat), 2 -> blank, 3 -> 2
        let logits = arr2(&[
            [-3.0, -0.1, -3.0],
            [-3.0, -0.1, -3.0],
            [-0.1, -3.0, -3.0],
            [-3.0, -3.0, -0.1],
        ]);
        let out = beam(&logits, 4, 3);
        assert_eq!(out[0].tokens, vec![1, 2]);
        assert_eq!(out[0].frames, vec![0, 3]);
        assert!(out.iter().all(|h| h.frames.len() == h.tokens.len()));
        assert_eq!(
            collapse_with_frames(&[0, 1, 1, 0, 2, 2], 0),
            (vec![1, 2], vec![1, 4])
        );
    }

    #[test]
    fn collapse_frame_ids_drops_blanks_and_repeats() {
        assert_eq!(
            collapse_frame_ids(&[4, 7, 7, 9, 4, 9, 11, 11], 4),
            vec![7, 9, 9, 11]
        );
        assert_eq!(collapse_frame_ids(&[], 0), Vec::<u32>::new());
        assert_eq!(collapse_frame_ids(&[4, 4, 4], 4), Vec::<u32>::new());
        assert_eq!(collapse_frame_ids(&[7, 4, 7], 4), vec![7, 7]);
    }

    // ---- property tests against the reference implementation ----

    /// xorshift64*: deterministic, dependency-free.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    struct Case {
        logits: Vec<f32>,
        t_len: usize,
        vocab: usize,
        blank: u32,
        blacklist: Vec<u32>,
        beam_width: usize,
        k: usize,
    }

    fn run_both(c: &Case) -> (Vec<BeamHypothesis>, Vec<BeamHypothesis>) {
        let lp = reference::log_probs(&c.logits, c.t_len, c.vocab, &c.blacklist);
        let want = reference::prefix_beam_search(&lp, c.blank, c.beam_width, c.k);
        // the scalar log-sum-exp, as the reference: this compares the beams
        let proposal = log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: lse::exact },
        );
        let got = prefix_beam_search(&proposal.steps, c.blank, c.beam_width, c.k);
        // A blacklisted id is -inf in the reference's matrix and can fill a
        // top-k slot when fewer than k ids are allowed, making -inf
        // hypotheses; the proposal never lists blacklisted ids.
        let finite = |h: Vec<BeamHypothesis>| -> Vec<BeamHypothesis> {
            h.into_iter().filter(|h| h.score > NEG_INF).collect()
        };
        (finite(want), finite(got))
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + a.abs().max(b.abs()))
    }

    fn assert_same(c: &Case, want: &[BeamHypothesis], got: &[BeamHypothesis], seed: u64) {
        assert_eq!(want.len(), got.len(), "seed {seed}: lengths");
        for (w, g) in want.iter().zip(got) {
            assert_eq!(
                w.tokens, g.tokens,
                "seed {seed} T={} V={}",
                c.t_len, c.vocab
            );
            assert_eq!(w.frames, g.frames, "seed {seed}: frames of {:?}", w.tokens);
            assert!(
                close(w.score, g.score),
                "seed {seed}: {} vs {}",
                w.score,
                g.score
            );
        }
    }

    fn random_case(rng: &mut Rng, scale: f32) -> Case {
        let t_len = 1 + rng.below(24);
        let vocab = 3 + rng.below(40);
        let blank = rng.below(vocab) as u32;
        let mut logits: Vec<f32> = (0..t_len * vocab)
            .map(|_| (rng.unit() + rng.unit() + rng.unit() - 1.5) * scale)
            .collect();
        // CTC-like peaky frames: often the blank or one token dominates
        for t in 0..t_len {
            if rng.below(3) == 0 {
                let peak = if rng.below(2) == 0 {
                    blank as usize
                } else {
                    rng.below(vocab)
                };
                logits[t * vocab + peak] += 8.0 + 8.0 * rng.unit();
            }
        }
        let blacklist: Vec<u32> = (0..rng.below(3))
            .map(|_| rng.below(vocab) as u32)
            .filter(|&b| b != blank)
            .collect();
        Case {
            logits,
            t_len,
            vocab,
            blank,
            blacklist,
            beam_width: [2, 3, 5, 8, 20][rng.below(5)],
            k: [1, 2, 4, 8, 16][rng.below(5)],
        }
    }

    #[test]
    fn matches_the_reference_on_random_logits() {
        for seed in 1..=400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let c = random_case(&mut rng, 4.0);
            let (want, got) = run_both(&c);
            assert_same(&c, &want, &got, seed);
        }
    }

    #[test]
    fn matches_the_reference_on_extreme_logits() {
        for seed in 1..=150u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut c = random_case(&mut rng, 60.0);
            // saturated rows: one token at 1e4, others near -1e4, some -1e30
            for t in 0..c.t_len {
                if rng.below(2) == 0 {
                    for v in 0..c.vocab {
                        c.logits[t * c.vocab + v] = match rng.below(4) {
                            0 => -1e30,
                            1 => -1e4 + rng.unit(),
                            _ => rng.unit() * 10.0,
                        };
                    }
                    c.logits[t * c.vocab + rng.below(c.vocab)] = 1e4;
                }
            }
            let (want, got) = run_both(&c);
            assert_same(&c, &want, &got, seed);
        }
    }

    #[test]
    fn matches_the_reference_on_tied_logits() {
        // Values from a tiny set, so ranks and path masses tie exactly. With
        // top-k covering the vocabulary and a beam no step fills, neither the
        // top-k selection nor the truncation has to break a tie, and every
        // prefix's mass must agree. (Which of two equally probable paths
        // supplies the frames is merge-order dependent, so frames are not
        // compared.)
        for seed in 1..=200u64 {
            let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
            let t_len = 1 + rng.below(5);
            let vocab = 2 + rng.below(3);
            let logits: Vec<f32> = (0..t_len * vocab).map(|_| rng.below(3) as f32).collect();
            let c = Case {
                logits,
                t_len,
                vocab,
                blank: 0,
                blacklist: Vec::new(),
                beam_width: 4096,
                k: vocab,
            };
            let (want, got) = run_both(&c);
            assert_eq!(want.len(), got.len(), "seed {seed}");
            let by_tokens = |h: &[BeamHypothesis]| {
                let mut v: Vec<(Vec<u32>, f32)> =
                    h.iter().map(|h| (h.tokens.clone(), h.score)).collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            };
            for ((wt, ws), (gt, gs)) in by_tokens(&want).iter().zip(&by_tokens(&got)) {
                assert_eq!(wt, gt, "seed {seed}");
                assert!(close(*ws, *gs), "seed {seed}: {ws} vs {gs}");
            }
            // and the best one is the same whenever it is unique
            if want.len() > 1 && !close(want[0].score, want[1].score) {
                assert_eq!(want[0].tokens, got[0].tokens, "seed {seed}");
            }
        }
    }

    // ---- vectorized log-sum-exp against the scalar one ----

    /// Beam over `c` with the given row log-sum-exp.
    fn beam_with(c: &Case, row_lse: fn(&[f32]) -> f32) -> Vec<BeamHypothesis> {
        let p = log_probs_with(
            &c.logits,
            c.t_len,
            c.vocab,
            &c.blacklist,
            c.blank,
            c.k,
            FrameReduce { lse: row_lse },
        );
        prefix_beam_search(&p.steps, c.blank, c.beam_width, c.k)
    }

    /// The vectorized log-sum-exp may reorder only hypotheses whose scores
    /// are within `tol`: the same hypotheses in the same order otherwise,
    /// with scores within `tol`, and membership may differ only for those
    /// within `tol` of the beam's cutoff.
    fn assert_same_up_to_near_ties(want: &[BeamHypothesis], got: &[BeamHypothesis], seed: u64) {
        let tol = |a: f32| 2e-5 * (1.0 + a.abs());
        let near = |a: f32, b: f32| a == b || (a - b).abs() <= tol(a);
        let cutoff = want.last().map_or(NEG_INF, |h| h.score);
        let score_of =
            |h: &[BeamHypothesis], t: &[u32]| h.iter().find(|x| x.tokens == t).map(|x| x.score);
        for w in want {
            match score_of(got, &w.tokens) {
                Some(g) => assert!(near(g, w.score), "seed {seed}: {g} vs {}", w.score),
                None => assert!(
                    near(w.score, cutoff),
                    "seed {seed}: {:?} ({}) dropped away from the cutoff {cutoff}",
                    w.tokens,
                    w.score
                ),
            }
        }
        // order: any pair in both lists that swapped must be a near-tie
        for (i, a) in want.iter().enumerate() {
            for b in &want[i + 1..] {
                let (Some(ia), Some(ib)) = (
                    got.iter().position(|x| x.tokens == a.tokens),
                    got.iter().position(|x| x.tokens == b.tokens),
                ) else {
                    continue;
                };
                if ia > ib {
                    assert!(
                        near(a.score, b.score),
                        "seed {seed}: {:?} ({}) and {:?} ({}) swapped",
                        a.tokens,
                        a.score,
                        b.tokens,
                        b.score
                    );
                }
            }
        }
    }

    #[test]
    fn vectorized_lse_only_reorders_near_ties() {
        for seed in 1..=400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let c = random_case(&mut rng, 4.0);
            let want = beam_with(&c, lse::exact);
            let got = beam_with(&c, lse::row);
            assert_same_up_to_near_ties(&want, &got, seed);
        }
    }

    #[test]
    fn vectorized_lse_keeps_close_but_distinct_hypotheses_in_order() {
        // Two frames; tokens 1 and 2 each win one frame by `delta`, so the
        // hypotheses [1] and [2] differ by about `delta`. Down to a gap of
        // 1e-4 they keep the scalar path's order.
        for &delta in &[1e-1f32, 1e-2, 1e-3, 1e-4] {
            for &base in &[0.0f32, 5.0, 20.0] {
                let v = 64;
                let mut logits = vec![-3.0f32; 2 * v];
                logits[0] = base; // blank strong on frame 0
                logits[1] = base - 1.0;
                logits[2] = base - 1.0 - delta;
                logits[v] = base;
                logits[v + 1] = base - 1.0 - delta;
                logits[v + 2] = base - 1.0 + delta * 0.5;
                let c = Case {
                    logits,
                    t_len: 2,
                    vocab: v,
                    blank: 0,
                    blacklist: Vec::new(),
                    beam_width: 8,
                    k: 4,
                };
                let want = beam_with(&c, lse::exact);
                let got = beam_with(&c, lse::row);
                let order =
                    |h: &[BeamHypothesis]| h.iter().map(|x| x.tokens.clone()).collect::<Vec<_>>();
                assert_eq!(order(&want), order(&got), "delta {delta} base {base}");
            }
        }
    }
}
