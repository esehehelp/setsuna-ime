//! A token sequence's CTC log-likelihood: the log of the sum, over every alignment of the sequence to the frames
//! (blanks before, between and after its tokens, a token held over several frames, a blank between two equal
//! tokens), of the frames' probabilities. What a CTC model says of a candidate it did not propose (the session's
//! fusion of an AR model's candidates with a CTC model's scores); its prefix beam's score of a hypothesis is the same
//! sum over the alignments the beam kept, so the two agree where the beam kept them all.
//!
//! The forward recursion runs over the sequence's 2U + 1 states (a blank, then each token followed by a blank) in
//! probability space, each frame's row rescaled to sum to 1 and the log of the scale added up (no log or exp in the
//! inner loop), in f64 so that a frame's improbable tokens (e^-100) neither vanish nor flush the row.

const NEG_INF: f32 = f32::NEG_INFINITY;

/// Per frame, the log-probability of the blank and of each of a set of token ids (`ids`, sorted, distinct;
/// row-major, `ids.len()` a frame), as the CTC model's log-softmax gives them (`proposal::gather_log_probs`), or
/// those plus a constant a frame (`proposal::greedy_scores`: the log-likelihoods then differ from the log-softmax's
/// by one constant for every sequence).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogProbs {
    pub ids: Vec<u32>,
    pub blank: Vec<f32>,
    pub cols: Vec<f32>,
}

impl TokenLogProbs {
    pub fn frames(&self) -> usize {
        self.blank.len()
    }
}

/// The frames' probabilities ready for the forward recursion, built once and used for every sequence of a request.
pub struct Forward<'a> {
    ids: &'a [u32],
    frames: usize,
    /// e^log-prob: per frame, the blank first, then the columns of `ids`.
    probs: Vec<f64>,
}

impl<'a> Forward<'a> {
    pub fn new(lp: &'a TokenLogProbs) -> Self {
        debug_assert!(
            lp.ids.windows(2).all(|w| w[0] < w[1]),
            "ids sorted and distinct"
        );
        let n = lp.ids.len();
        let frames = lp.frames();
        let mut probs = Vec::with_capacity(frames * (n + 1));
        for t in 0..frames {
            probs.push((lp.blank[t] as f64).exp());
            probs.extend(
                lp.cols[t * n..(t + 1) * n]
                    .iter()
                    .map(|&x| (x as f64).exp()),
            );
        }
        Self {
            ids: &lp.ids,
            frames,
            probs,
        }
    }

    /// log P(`labels` | the frames); -inf when an id of `labels` is not among the columns or no alignment fits
    /// (fewer frames than the tokens and the blanks between equal neighbours take).
    pub fn log_likelihood(&self, labels: &[u32]) -> f32 {
        let row = self.ids.len() + 1;
        // the column of each label's token in a frame's row (1-based: 0 is the blank)
        let mut col = Vec::with_capacity(labels.len());
        for id in labels {
            match self.ids.binary_search(id) {
                Ok(i) => col.push(i + 1),
                Err(_) => return NEG_INF,
            }
        }
        let repeats = labels.windows(2).filter(|w| w[0] == w[1]).count();
        if labels.len() + repeats > self.frames {
            return NEG_INF;
        }
        let states = 2 * labels.len() + 1;
        // state s: even a blank, odd the token labels[s / 2]
        let emit = |frame: &[f64], s: usize| {
            if s % 2 == 0 {
                frame[0]
            } else {
                frame[col[s / 2]]
            }
        };
        let mut alpha = vec![0f64; states];
        let mut next = vec![0f64; states];
        let mut log_scale = 0f64;
        for t in 0..self.frames {
            let frame = &self.probs[t * row..(t + 1) * row];
            // the states an alignment can be in after frame t: at most 2t + 1 from the start, at least
            // states - 2 (frames - t) from the end
            let hi = (2 * t + 2).min(states);
            let lo = (states + 2 * t).saturating_sub(2 * self.frames);
            let mut sum = 0f64;
            for s in 0..states {
                let a = if s < lo || s >= hi {
                    0.0
                } else if t == 0 {
                    // the first frame starts on the leading blank or the first token
                    if s < 2 {
                        emit(frame, s)
                    } else {
                        0.0
                    }
                } else {
                    let mut a = alpha[s];
                    if s >= 1 {
                        a += alpha[s - 1];
                    }
                    // a token may follow the token before last, the blank between skipped, unless the two are equal
                    if s >= 2 && s % 2 == 1 && labels[s / 2] != labels[s / 2 - 1] {
                        a += alpha[s - 2];
                    }
                    a * emit(frame, s)
                };
                next[s] = a;
                sum += a;
            }
            if !(sum > 0.0) || !sum.is_finite() {
                return NEG_INF;
            }
            for a in next.iter_mut() {
                *a /= sum;
            }
            log_scale += sum.ln();
            std::mem::swap(&mut alpha, &mut next);
        }
        // the alignment ends on the last token or the blank after it
        let end = alpha[states - 1] + if states > 1 { alpha[states - 2] } else { 0.0 };
        if end > 0.0 {
            (log_scale + end.ln()) as f32
        } else {
            NEG_INF
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::decode::prefix_beam_search;
    use crate::pipeline::proposal::FrameLogProbs;

    /// A table of T frames over the blank (id 0) and the tokens 1..v, from a seed: each row a log-softmax.
    fn table(t: usize, v: usize, seed: u64) -> TokenLogProbs {
        let mut state = seed;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) as f32
        };
        let mut blank = Vec::new();
        let mut cols = Vec::new();
        for _ in 0..t {
            let row: Vec<f32> = (0..v).map(|_| rnd() * 6.0 - 3.0).collect();
            let lse = row.iter().map(|x| x.exp()).sum::<f32>().ln();
            blank.push(row[0] - lse);
            cols.extend(row[1..].iter().map(|x| x - lse));
        }
        TokenLogProbs {
            ids: (1..v as u32).collect(),
            blank,
            cols,
        }
    }

    /// The sum over every path of T frames that collapses to `labels`, by enumeration.
    fn brute(lp: &TokenLogProbs, labels: &[u32]) -> f64 {
        let v = lp.ids.len() + 1;
        let t = lp.frames();
        let mut total = 0f64;
        for code in 0..v.pow(t as u32) {
            let mut c = code;
            let mut path = Vec::with_capacity(t);
            for _ in 0..t {
                path.push((c % v) as u32);
                c /= v;
            }
            let mut out = Vec::new();
            let mut prev = 0u32;
            for &p in &path {
                if p != 0 && p != prev {
                    out.push(p);
                }
                prev = p;
            }
            if out == labels {
                let lp_path: f64 = path
                    .iter()
                    .enumerate()
                    .map(|(f, &p)| {
                        if p == 0 {
                            lp.blank[f] as f64
                        } else {
                            lp.cols[f * (v - 1) + p as usize - 1] as f64
                        }
                    })
                    .sum();
                total += lp_path.exp();
            }
        }
        total.ln()
    }

    #[test]
    fn forward_is_the_sum_over_every_alignment() {
        let sequences: [&[u32]; 9] = [
            &[],
            &[1],
            &[2],
            &[1, 2],
            &[1, 1],
            &[2, 1, 2],
            &[1, 1, 1],
            &[3, 3],
            &[1, 2, 3],
        ];
        for seed in 0..4u64 {
            for t in 1..=5 {
                let lp = table(t, 4, seed * 31 + t as u64);
                let f = Forward::new(&lp);
                for labels in sequences {
                    let want = brute(&lp, labels);
                    let got = f.log_likelihood(labels) as f64;
                    if want == f64::NEG_INFINITY {
                        assert_eq!(got, f64::NEG_INFINITY, "T {t}, {labels:?}");
                    } else {
                        assert!(
                            (got - want).abs() < 1e-4,
                            "T {t}, {labels:?}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn no_alignment_fits_too_few_frames_or_an_unknown_token() {
        let lp = table(2, 4, 5);
        let f = Forward::new(&lp);
        // two equal tokens take three frames
        assert_eq!(f.log_likelihood(&[1, 1]), NEG_INF);
        assert!(f.log_likelihood(&[1, 2]).is_finite());
        assert_eq!(f.log_likelihood(&[1, 2, 3]), NEG_INF);
        assert_eq!(f.log_likelihood(&[9]), NEG_INF);
    }

    #[test]
    fn a_beam_that_keeps_every_prefix_scores_as_the_forward() {
        // the prefix beam over the whole vocabulary and wide enough to prune nothing: its scores are the sums over
        // every alignment
        let (t, v) = (6usize, 4usize);
        for seed in 0..3u64 {
            let lp = table(t, v, 100 + seed);
            let steps: Vec<FrameLogProbs> = (0..t)
                .map(|f| {
                    let mut top: Vec<(u32, f32)> = (1..v as u32)
                        .map(|id| (id, lp.cols[f * (v - 1) + id as usize - 1]))
                        .collect();
                    top.push((0, lp.blank[f]));
                    top.sort_by(|a, b| b.1.total_cmp(&a.1));
                    FrameLogProbs {
                        top,
                        blank: lp.blank[f],
                    }
                })
                .collect();
            let hyps = prefix_beam_search(&steps, 0, 10_000, v);
            assert!(hyps.len() > 100);
            let f = Forward::new(&lp);
            for h in &hyps {
                let got = f.log_likelihood(&h.tokens);
                // the beam keeps prefixes no alignment fits (a repeat without the frames for its blanks) at -inf
                assert!(
                    (got == f32::NEG_INFINITY && h.score == f32::NEG_INFINITY)
                        || (got - h.score).abs() < 1e-4,
                    "{:?}: {got} vs {}",
                    h.tokens,
                    h.score
                );
            }
            assert!(hyps.iter().filter(|h| h.score.is_finite()).count() > 100);
        }
    }
}
