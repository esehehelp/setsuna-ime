//! Tree scoring with llama.cpp: one `llama_decode` per request, each trie
//! node a token carrying the sequence ids of the candidates through it. A
//! unified KV cache turns that into a tree attention mask.
//!
//! Two things stay in the cache between requests:
//!   * BOS and the context, which belong to every sequence: the next request
//!     with the same context decodes from where it differs.
//!   * The last trie, so that a request for the same candidates plus more
//!     (K grown while the user pauses) decodes only the new nodes.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use crate::trie::Trie;

/// Sequences (candidates) one request may hold.
pub const MAX_SEQ: usize = 32;
/// Tokens the cache holds: BOS, the context and every candidate's own part.
const N_CTX: u32 = 2048;

/// What a request cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Nodes taken from the cache (the context, or the last trie).
    pub cached: usize,
    /// Nodes decoded.
    pub decoded: usize,
    pub decode_ms: f64,
    pub lse_ms: f64,
}

/// The trie of the last request, for one with more candidates to extend.
struct Last {
    seqs: Vec<Vec<i32>>,
    start: usize,
    trie: Trie,
    owners: Vec<Vec<i32>>,
    /// Logits of the nodes that may get a scored child later.
    rows: HashMap<usize, Vec<f32>>,
    lse: HashMap<usize, f64>,
    /// log p of each scored node's token.
    logp: HashMap<usize, f64>,
}

pub struct TreeScorer {
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    n_vocab: usize,
    threads: usize,
    /// Tokens at positions 0.. that every sequence holds in the cache.
    cached: Vec<i32>,
    last: Option<Last>,
}

impl TreeScorer {
    pub fn new(model: &'static LlamaModel, backend: &LlamaBackend, threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(N_CTX)
            .with_n_ubatch(N_CTX)
            .with_n_seq_max(MAX_SEQ as u32)
            .with_kv_unified(true)
            .with_n_threads(threads as i32)
            .with_n_threads_batch(threads as i32);
        let ctx = model
            .new_context(backend, params)
            .context("create the llama.cpp context")?;
        Ok(Self {
            ctx,
            batch: LlamaBatch::new(N_CTX as usize, MAX_SEQ as i32),
            n_vocab: model.n_vocab() as usize,
            threads,
            cached: Vec::new(),
            last: None,
        })
    }

    /// log p of each sequence's tokens from `start` on (each `[BOS] +
    /// tokens(context + candidate)`; all of them share the first `start`).
    pub fn score(&mut self, seqs: &[Vec<i32>], start: usize) -> Result<(Vec<f64>, Stats)> {
        anyhow::ensure!(
            seqs.len() <= MAX_SEQ,
            "at most {MAX_SEQ} candidates, got {}",
            seqs.len()
        );
        anyhow::ensure!(start >= 1, "the scored part starts after BOS");
        let extends = self.last.as_ref().is_some_and(|l| {
            l.start == start && seqs.len() >= l.seqs.len() && seqs[..l.seqs.len()] == l.seqs[..]
        });
        if !extends {
            self.last = None;
        }
        match self.decode(seqs, start) {
            Ok(Some(out)) => Ok(out),
            // a continuation needed logits it had not kept: the whole trie again
            Ok(None) => {
                self.last = None;
                self.decode(seqs, start)?
                    .context("a new trie needs no kept logits")
            }
            Err(e) => {
                self.last = None;
                self.cached.clear();
                Err(e)
            }
        }
    }

    /// Decode what `seqs` need beyond the cache; `None` when extending the
    /// last trie needs logits that were not kept.
    fn decode(&mut self, seqs: &[Vec<i32>], start: usize) -> Result<Option<(Vec<f64>, Stats)>> {
        let trie = Trie::build(seqs);
        let owners = trie.owners();
        let n = trie.nodes.len();
        // the node whose output scores the first candidate token
        let fresh = start - 1;
        let mut last = self.last.take();
        let old_n = last.as_ref().map_or(0, |l| l.trie.nodes.len());
        let old_k = last.as_ref().map_or(0, |l| l.seqs.len());

        // scored now: the new nodes of the scored part
        let scored: Vec<usize> = (old_n..n)
            .filter(|&i| trie.nodes[i].depth >= start)
            .collect();
        let mut need = vec![false; n];
        for &j in &scored {
            if let Some(p) = trie.nodes[j].parent {
                need[p] = true;
            }
        }
        if let Some(l) = &last {
            if (0..old_n).any(|i| need[i] && !l.rows.contains_key(&i)) {
                return Ok(None);
            }
        }

        let mut keep = 0;
        let mem_err = |e| anyhow::anyhow!("llama.cpp KV cache: {e:?}");
        match &last {
            None => {
                // BOS and the context are the first `start` nodes (candidate
                // 0's path); keep the cached ones that match
                while keep < fresh
                    && keep < self.cached.len()
                    && self.cached[keep] == trie.nodes[keep].token
                {
                    keep += 1;
                }
                self.ctx
                    .clear_kv_cache_seq(None, Some(keep as u32), None)
                    .map_err(mem_err)?;
            }
            Some(_) => {
                // a new candidate takes the cells of its path's old part from
                // an old candidate through the deepest old node on it
                for c in old_k..seqs.len() {
                    let Some(&x) = trie.paths[c].iter().rev().find(|&&i| i < old_n) else {
                        continue;
                    };
                    let depth = trie.nodes[x].depth;
                    if depth > fresh {
                        let from = owners[x][0];
                        self.ctx
                            .copy_kv_cache_seq(from, c as i32, Some(0), Some(depth as u32 + 1))
                            .map_err(mem_err)?;
                    }
                }
            }
        }

        let all: Vec<i32> = (0..MAX_SEQ as i32).collect();
        let mut leaf = vec![false; n];
        for path in &trie.paths {
            if let Some(&end) = path.last() {
                leaf[end] = true;
            }
        }
        self.batch.clear();
        let mut row = vec![None; n];
        let mut with_logits = Vec::new();
        for i in old_n..n {
            let node = trie.nodes[i];
            if last.is_none() && i < keep {
                continue;
            }
            // BOS and the context belong to every sequence, so they serve any
            // candidate count later; deeper nodes to their candidates only
            let seq_ids: &[i32] = if node.depth <= fresh {
                &all
            } else {
                &owners[i]
            };
            // a leaf's logits are kept for candidates a later request may add below it
            let logits = need[i] || (leaf[i] && node.depth >= fresh);
            row[i] = Some(self.batch.n_tokens());
            self.batch
                .add(LlamaToken(node.token), node.depth as i32, seq_ids, logits)
                .context("llama.cpp batch")?;
            if logits {
                with_logits.push(i);
            }
        }
        let decoded = self.batch.n_tokens() as usize;
        let t0 = Instant::now();
        if decoded > 0 {
            self.ctx
                .decode(&mut self.batch)
                .context("llama.cpp decode")?;
        }
        let t1 = Instant::now();

        // log-sum-exp of each new row, the rows split across the threads
        let logits: Vec<&[f32]> = with_logits
            .iter()
            .map(|&i| self.ctx.get_logits_ith(row[i].expect("in the batch")))
            .collect();
        let mut lse = vec![0.0f64; logits.len()];
        let per = logits.len().div_ceil(self.threads).max(1);
        std::thread::scope(|s| {
            for (rows, out) in logits.chunks(per).zip(lse.chunks_mut(per)) {
                s.spawn(move || {
                    for (r, o) in rows.iter().zip(out) {
                        *o = log_sum_exp(r);
                    }
                });
            }
        });
        let mut l = last.take().unwrap_or_else(|| Last {
            seqs: Vec::new(),
            start,
            trie: Trie::default(),
            owners: Vec::new(),
            rows: HashMap::new(),
            lse: HashMap::new(),
            logp: HashMap::new(),
        });
        for ((&i, r), s) in with_logits.iter().zip(&logits).zip(&lse) {
            l.rows.insert(i, r[..self.n_vocab].to_vec());
            l.lse.insert(i, *s);
        }
        for &j in &scored {
            let node = trie.nodes[j];
            let p = node.parent.expect("a scored node is below BOS");
            l.logp
                .insert(j, f64::from(l.rows[&p][node.token as usize]) - l.lse[&p]);
        }
        let t2 = Instant::now();

        let scores = trie
            .paths
            .iter()
            .map(|path| path.iter().filter_map(|i| l.logp.get(i)).sum())
            .collect();
        if old_n == 0 {
            self.cached = trie.nodes[..start.min(n)]
                .iter()
                .map(|nd| nd.token)
                .collect();
        }
        let stats = Stats {
            cached: if old_n > 0 { old_n } else { keep },
            decoded,
            decode_ms: (t1 - t0).as_secs_f64() * 1000.0,
            lse_ms: (t2 - t1).as_secs_f64() * 1000.0,
        };
        l.seqs = seqs.to_vec();
        l.trie = trie;
        l.owners = owners;
        self.last = Some(l);
        Ok(Some((scores, stats)))
    }
}

fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
    f64::from(max) + f64::from(sum).ln()
}
