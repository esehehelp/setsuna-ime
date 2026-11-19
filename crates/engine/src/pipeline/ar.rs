//! AR decoding of a model with a shallow AR decoder (the research export's AR form).
//!
//! State: each hypothesis is the surface written so far (token ids, left to right) with the log-probability of its
//! tokens; the decoder's self-attention keys and values of its positions are the cache (layers, N, heads, S,
//! head_dim) the engine keeps and reorders to the surviving hypotheses after each position. The encoder graph runs
//! once per conversion and gives every AR layer's cross-attention keys and values (layers, 1, heads, T,
//! head_dim), shared by the hypotheses.
//!
//! The search ([`beam_search`]): at every position each live hypothesis is extended by its `top_k` best next tokens
//! (forbidden ids skipped); the end token closes a hypothesis (its score kept, the end not in its ids); the best
//! `beam` other extensions live on. It stops when nothing lives, at `max_len` positions, or once `want` hypotheses
//! are closed and none living scores above the `want`-th (log-probabilities only fall as tokens are added). Scores
//! are the plain sums of the log-probabilities; beam 1 is the greedy decode the export checks against the model.
//!
//! The step is the engine's own implementation (`ar_native`, the manifest's `ar_decoder.native`), which keeps its
//! cache its own way.

use std::time::Instant;

use anyhow::Result;

use crate::pipeline::ar_native::{NativeDecoder, NativeSearch};
use crate::pipeline::native_encoder::NativeEncoder;

/// One finished (or, if nothing finished, unfinished) hypothesis: its ids without the beginning and the end, and
/// the sum of their log-probabilities (the end's included when it finished).
#[derive(Debug, Clone, PartialEq)]
pub struct ArHypothesis {
    pub ids: Vec<u32>,
    pub score: f32,
}

/// The best next tokens of N hypotheses: row i holds hypothesis i's `k` best (ids and log-probabilities), best
/// first, row-major.
#[derive(Debug, Clone, Default)]
pub struct StepTop {
    pub k: usize,
    pub ids: Vec<u32>,
    pub log_probs: Vec<f32>,
}

/// A decoder that scores the next token of N hypotheses at a position.
pub trait StepModel {
    /// The best next tokens (at least the search's `top_k` a hypothesis, unless the vocabulary is smaller; tokens
    /// `forbidden` never among them) after `tokens[i]` (the last token of hypothesis i, the beginning at position 0)
    /// at `position`, hypothesis i continuing hypothesis `parents[i]` of the previous call (`parents` empty at
    /// position 0, where N is 1). `forbidden` is the same at every call of a search.
    fn step(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop>;
    /// Vocabulary size.
    fn vocab(&self) -> usize;
}

/// Search settings.
#[derive(Debug, Clone)]
pub struct ArSearch {
    pub bos: u32,
    pub eos: u32,
    /// Live hypotheses kept per position (1: greedy).
    pub beam: usize,
    /// Next tokens tried per live hypothesis.
    pub top_k: usize,
    /// Positions at most (the end included).
    pub max_len: usize,
    /// Closed hypotheses wanted before stopping early.
    pub want: usize,
    /// Token ids never written (the end is never forbidden).
    pub forbidden: Vec<u32>,
    /// Beam threshold: at each position the extensions scoring more than this below the position's best (ended or
    /// not) are dropped, so fewer hypotheses live (and run the step graph) where the model is sure; None keeps the
    /// full beam.
    pub margin: Option<f32>,
    /// With the script latent (`ArSpec::script`): added to the prior's logits of the classes other, kanji,
    /// hiragana, katakana at every position (a preference for a script where the model may choose one); zeros:
    /// none. Ignored for a model without the latent.
    pub script_bias: [f32; 4],
}

/// Beam search over `model` (see the module docs); closed hypotheses best first, at most `max(beam, want)`; when
/// none closed, the live ones.
pub fn beam_search(model: &mut impl StepModel, s: &ArSearch) -> Result<Vec<ArHypothesis>> {
    let vocab = model.vocab();
    let beam = s.beam.max(1);
    let top_k = s.top_k.max(1).min(vocab);
    let keep = beam.max(s.want.max(1));
    let mut forbidden = vec![false; vocab];
    for &f in &s.forbidden {
        if (f as usize) < vocab && f != s.eos {
            forbidden[f as usize] = true;
        }
    }
    let mut live: Vec<ArHypothesis> = vec![ArHypothesis {
        ids: Vec::new(),
        score: 0.0,
    }];
    let mut parents: Vec<usize> = Vec::new();
    let mut closed: Vec<ArHypothesis> = Vec::new();
    for position in 0..s.max_len {
        let tokens: Vec<u32> = live
            .iter()
            .map(|h| h.ids.last().copied().unwrap_or(s.bos))
            .collect();
        let top = model.step(&parents, &tokens, position, &forbidden)?;
        anyhow::ensure!(
            top.k > 0
                && top.ids.len() == live.len() * top.k
                && top.log_probs.len() == top.ids.len(),
            "the step gave {} / {} best tokens ({} a hypothesis) for {} hypotheses",
            top.ids.len(),
            top.log_probs.len(),
            top.k,
            live.len()
        );
        // every live hypothesis's top_k extensions (allowed, finite), then the best of them all
        let mut ext: Vec<(f32, usize, u32)> = Vec::with_capacity(live.len() * top_k);
        for (i, h) in live.iter().enumerate() {
            let row = i * top.k..(i + 1) * top.k;
            ext.extend(
                top.ids[row.clone()]
                    .iter()
                    .zip(&top.log_probs[row])
                    .filter(|&(&v, lp)| {
                        (v as usize) < vocab && !forbidden[v as usize] && lp.is_finite()
                    })
                    .take(top_k)
                    .map(|(&v, &lp)| (h.score + lp, i, v)),
            );
        }
        ext.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        if let (Some(m), Some(&(best, _, _))) = (s.margin, ext.first()) {
            ext.retain(|e| e.0 >= best - m);
        }
        let mut next: Vec<ArHypothesis> = Vec::with_capacity(beam);
        let mut next_parents: Vec<usize> = Vec::with_capacity(beam);
        for (score, i, v) in ext {
            if next.len() == beam {
                break;
            }
            if v == s.eos {
                closed.push(ArHypothesis {
                    ids: live[i].ids.clone(),
                    score,
                });
            } else {
                let mut ids = live[i].ids.clone();
                ids.push(v);
                next.push(ArHypothesis { ids, score });
                next_parents.push(i);
            }
        }
        closed.sort_by(|a, b| b.score.total_cmp(&a.score));
        closed.truncate(keep);
        let want = s.want.max(1);
        let settled = closed.len() >= want
            && next
                .first()
                .map_or(true, |best| best.score <= closed[want - 1].score);
        if next.is_empty() || settled {
            break;
        }
        live = next;
        parents = next_parents;
    }
    if closed.is_empty() {
        live.sort_by(|a, b| b.score.total_cmp(&a.score));
        return Ok(live);
    }
    Ok(closed)
}

/// The AR decoder's shapes and ids (the manifest's `ar_decoder`).
#[derive(Debug, Clone, Copy)]
pub struct ArSpec {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
    pub bos: u32,
    pub eos: u32,
    /// The best next tokens the step graph gives a hypothesis (the search tries at most this many).
    pub top_k: usize,
    /// The script latent at choice points (the manifest's `ar_decoder.script`): the step scores each token by its
    /// best class and takes `ArSearch::script_bias` (the step graph's `script_bias` input).
    pub script: bool,
}

/// The encoder and the step of an AR artifact.
pub struct ArRunner {
    encoder: Box<NativeEncoder>,
    step: NativeDecoder,
    pub spec: ArSpec,
    /// The last `run`'s time in the encoder and in the search (the step and the cache), in ms, and the
    /// positions it took.
    pub last_encoder_ms: f64,
    pub last_search_ms: f64,
    pub last_positions: usize,
    profile: Option<ArProfile>,
}

/// SETSUNA_AR_PROFILE=<path>: the native step's time by part (self-attention, cross-attention, FFN, output
/// projection, the best tokens' scan, the script latent's terms, and the search's own rest), the positions and the
/// hypotheses run, summed over the conversions and written to <path> as JSON every 256 conversions and when the
/// runner is dropped (a research measure).
struct ArProfile {
    path: std::path::PathBuf,
    ms: [f64; 7],
    conversions: usize,
    positions: usize,
    rows: usize,
}

const AR_PARTS: [&str; 7] = ["self", "cross", "ffn", "head", "top", "script", "search"];

impl ArProfile {
    fn from_env() -> Option<Self> {
        let path = std::env::var_os("SETSUNA_AR_PROFILE")?;
        Some(ArProfile {
            path: path.into(),
            ms: [0.0; 7],
            conversions: 0,
            positions: 0,
            rows: 0,
        })
    }

    fn add(&mut self, stage_ns: [u64; 6], search_ms: f64, positions: usize, rows: usize) {
        let mut parts = 0.0;
        for (m, ns) in self.ms.iter_mut().zip(stage_ns) {
            let ms = ns as f64 / 1e6;
            *m += ms;
            parts += ms;
        }
        self.ms[6] += (search_ms - parts).max(0.0);
        self.conversions += 1;
        self.positions += positions;
        self.rows += rows;
        if self.conversions % 256 == 0 {
            self.write();
        }
    }

    fn write(&self) {
        let parts: serde_json::Map<String, serde_json::Value> = AR_PARTS
            .iter()
            .zip(self.ms)
            .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect();
        let body = serde_json::json!({"conversions": self.conversions, "positions": self.positions,
                                      "rows": self.rows, "ms": parts});
        // a research file: a failed write loses the measure, not the conversion
        let _ = std::fs::write(&self.path, body.to_string());
    }
}

impl Drop for ArProfile {
    fn drop(&mut self) {
        self.write();
    }
}

impl ArRunner {
    pub fn new(encoder: Box<NativeEncoder>, step: NativeDecoder, spec: ArSpec) -> Self {
        Self {
            encoder,
            step,
            spec,
            last_encoder_ms: 0.0,
            last_search_ms: 0.0,
            last_positions: 0,
            profile: ArProfile::from_env(),
        }
    }

    /// The hypotheses of one encoded input (`proposal::encode`'s ids, mask and length). `search`'s `max_len` is
    /// capped at the decoder's positions less one.
    pub fn run(
        &mut self,
        encoded: (Vec<i64>, Vec<i64>, usize),
        search: &ArSearch,
    ) -> Result<Vec<ArHypothesis>> {
        let (mut ids, mut mask, actual) = encoded;
        ids.truncate(actual);
        mask.truncate(actual);
        let t0 = Instant::now();
        let sp = self.spec;
        let enc = &mut self.encoder;
        anyhow::ensure!(
            mask.iter().all(|&m| m == 1),
            "the native encoder takes a row without padding"
        );
        anyhow::ensure!(
            enc.ar_shape() == Some((sp.layers, sp.heads, sp.head_dim)),
            "native encoder's AR layers {:?}, the manifest's ({}, {}, {})",
            enc.ar_shape(),
            sp.layers,
            sp.heads,
            sp.head_dim
        );
        let (cross_k, cross_v) = enc.cross(&ids)?;
        self.last_encoder_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = Instant::now();
        let mut s = search.clone();
        s.max_len = s.max_len.min(sp.max_positions - 1);
        s.top_k = s.top_k.min(sp.top_k);
        let hyps = {
            {
                let dec = &mut self.step;
                anyhow::ensure!(
                    dec.script() == sp.script,
                    "the native AR step {} the script latent, the manifest {}",
                    if dec.script() { "has" } else { "lacks" },
                    if sp.script { "names it" } else { "does not" }
                );
                let mut model = NativeSearch::new(dec, &cross_k, cross_v, actual, s.top_k)?;
                model.set_script_bias(s.script_bias);
                let hyps = beam_search(&mut model, &s)?;
                self.last_positions = model.positions;
                if let Some(p) = self.profile.as_mut() {
                    p.add(
                        model.stage_ns,
                        t1.elapsed().as_secs_f64() * 1000.0,
                        model.positions,
                        model.rows,
                    );
                }
                hyps
            }
        };
        self.last_search_ms = t1.elapsed().as_secs_f64() * 1000.0;
        Ok(hyps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy decoder over 6 tokens (0 bos, 1 eos, 2..6 chars): the log-probs of the next token depend on the
    /// whole prefix through a fixed table; it records the parents it is given.
    struct Table {
        rows: Vec<(Vec<u32>, Vec<f32>)>,
        prefixes: Vec<Vec<u32>>,
        calls: Vec<Vec<usize>>,
    }

    impl Table {
        fn probs(&self, prefix: &[u32]) -> Vec<f32> {
            self.rows
                .iter()
                .find(|(p, _)| p == prefix)
                .map(|(_, r)| r.iter().map(|x| x.ln()).collect())
                .unwrap_or_else(|| {
                    let mut r = vec![0.01f32.ln(); 6];
                    r[1] = 0.95f32.ln();
                    r
                })
        }
    }

    impl StepModel for Table {
        fn vocab(&self) -> usize {
            6
        }
        fn step(
            &mut self,
            parents: &[usize],
            tokens: &[u32],
            position: usize,
            forbidden: &[bool],
        ) -> Result<StepTop> {
            self.calls.push(parents.to_vec());
            let prefixes: Vec<Vec<u32>> = if position == 0 {
                vec![Vec::new()]
            } else {
                parents
                    .iter()
                    .zip(tokens)
                    .map(|(&p, &t)| {
                        let mut v = self.prefixes[p].clone();
                        v.push(t);
                        v
                    })
                    .collect()
            };
            // each row's allowed tokens best first, as the step graph gives them (forbidden ones at the end, -inf)
            let mut out = StepTop {
                k: 6,
                ..StepTop::default()
            };
            for p in &prefixes {
                let mut row: Vec<(u32, f32)> = self
                    .probs(p)
                    .into_iter()
                    .enumerate()
                    .map(|(v, lp)| (v as u32, if forbidden[v] { f32::NEG_INFINITY } else { lp }))
                    .collect();
                row.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                out.ids.extend(row.iter().map(|r| r.0));
                out.log_probs.extend(row.iter().map(|r| r.1));
            }
            self.prefixes = prefixes;
            Ok(out)
        }
    }

    fn table() -> Table {
        // after nothing: 2 (0.5) or 3 (0.4); after 2: eos (0.3) or 4 (0.6); after 3: 5 (0.9); after 2 4: eos (0.9);
        // after 3 5: eos (0.9)
        let row = |pairs: &[(usize, f32)]| {
            let mut r = vec![0.001f32; 6];
            for &(i, p) in pairs {
                r[i] = p;
            }
            r
        };
        Table {
            rows: vec![
                (vec![], row(&[(2, 0.5), (3, 0.4)])),
                (vec![2], row(&[(1, 0.3), (4, 0.6)])),
                (vec![3], row(&[(5, 0.9)])),
                (vec![2, 4], row(&[(1, 0.9)])),
                (vec![3, 5], row(&[(1, 0.9)])),
            ],
            prefixes: Vec::new(),
            calls: Vec::new(),
        }
    }

    fn search(beam: usize, want: usize) -> ArSearch {
        ArSearch {
            bos: 0,
            eos: 1,
            beam,
            top_k: 3,
            max_len: 8,
            want,
            forbidden: vec![0],
            margin: None,
            script_bias: [0.0; 4],
        }
    }

    #[test]
    fn greedy_follows_the_best_token_and_stops_at_the_end() {
        let hyps = beam_search(&mut table(), &search(1, 1)).unwrap();
        assert_eq!(hyps[0].ids, vec![2, 4]);
        let want = 0.5f32.ln() + 0.6f32.ln() + 0.9f32.ln();
        assert!((hyps[0].score - want).abs() < 1e-5);
    }

    #[test]
    fn beam_finds_the_hypothesis_greedy_misses_and_orders_by_score() {
        // 3 5 end: 0.4 * 0.9 * 0.9 = 0.324 beats 2 4 end: 0.5 * 0.6 * 0.9 = 0.27; 2 end: 0.15
        let hyps = beam_search(&mut table(), &search(3, 3)).unwrap();
        let ids: Vec<Vec<u32>> = hyps.iter().map(|h| h.ids.clone()).collect();
        assert_eq!(ids[..3], [vec![3, 5], vec![2, 4], vec![2]]);
        assert!(hyps.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn parents_point_at_the_hypotheses_extended() {
        let mut t = table();
        beam_search(&mut t, &search(2, 2)).unwrap();
        assert!(t.calls[0].is_empty());
        // position 1 extends the two children of the root
        assert_eq!(t.calls[1], vec![0, 0]);
    }

    #[test]
    fn forbidden_tokens_are_never_written() {
        let mut s = search(3, 3);
        s.forbidden = vec![0, 2];
        let hyps = beam_search(&mut table(), &s).unwrap();
        assert!(hyps.iter().all(|h| !h.ids.contains(&2)));
        assert_eq!(hyps[0].ids, vec![3, 5]);
    }

    #[test]
    fn nothing_closed_returns_the_live_ones_best_first() {
        // one token per hypothesis and one position: the end is never tried
        let mut s = search(2, 2);
        s.top_k = 1;
        s.max_len = 1;
        let hyps = beam_search(&mut table(), &s).unwrap();
        assert_eq!(hyps.len(), 1);
        assert_eq!(hyps[0].ids, vec![2]);
    }

    #[test]
    fn the_margin_drops_unlikely_extensions_and_keeps_the_best() {
        // the root's four best extensions are 2, 3, the end (which closes) and a 0.001 one: without a margin three
        // live; with 1 nat only 2 and 3
        let mut wide = search(3, 3);
        wide.top_k = 4;
        let mut full = table();
        let a = beam_search(&mut full, &wide).unwrap();
        assert_eq!(full.calls[1].len(), 3);
        let mut cut = table();
        let mut s = wide.clone();
        s.margin = Some(1.0);
        let b = beam_search(&mut cut, &s).unwrap();
        assert_eq!(cut.calls[1].len(), 2);
        assert_eq!(a[0], b[0]);
        assert_eq!(b[0].ids, vec![3, 5]);
    }
}
