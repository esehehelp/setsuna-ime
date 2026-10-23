//! The transition table of a CTC-CRF (the research export's): a score
//! ψ(c, d) between consecutive output tokens on top of the CTC posteriors,
//! which the prefix beam search adds where it emits a new token.
//!
//! ψ is low rank and non-negative, ψ(c, d) = mean_j exp(a_cj + b_dj); row V
//! of `a` is the start symbol (before the first token). The manifest's `crf`
//! file holds `a` ((V+1) x r) then `b` (V x r), f32 little endian, row major.
//! Kept as exp(a) and exp(b) / r, so log ψ is the log of one dot product.

use std::path::Path;

use anyhow::{ensure, Context, Result};

pub struct CrfTables {
    rank: usize,
    vocab: usize,
    /// exp(a): (V+1) x r.
    ea: Vec<f32>,
    /// exp(b) / r: V x r.
    eb: Vec<f32>,
}

impl CrfTables {
    pub fn load(path: &Path, rank: usize, vocab: usize) -> Result<Self> {
        ensure!(rank > 0, "CRF rank must be positive");
        let bytes =
            std::fs::read(path).with_context(|| format!("read CRF table {}", path.display()))?;
        let want = ((vocab + 1) * rank + vocab * rank) * 4;
        ensure!(
            bytes.len() == want,
            "CRF table {} has {} bytes; rank {rank} over {vocab} tokens needs {want}",
            path.display(),
            bytes.len()
        );
        let mut values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let ea: Vec<f32> = values
            .by_ref()
            .take((vocab + 1) * rank)
            .map(f32::exp)
            .collect();
        let inv = 1.0 / rank as f32;
        let eb: Vec<f32> = values.map(|x| x.exp() * inv).collect();
        Ok(Self {
            rank,
            vocab,
            ea,
            eb,
        })
    }

    /// log ψ(last, c): `last` None is the start symbol.
    #[inline]
    pub fn log_psi(&self, last: Option<u32>, c: u32) -> f32 {
        let row = last.map_or(self.vocab, |x| x as usize);
        let a = &self.ea[row * self.rank..(row + 1) * self.rank];
        let b = &self.eb[c as usize * self.rank..(c as usize + 1) * self.rank];
        let mut acc = [0.0f32; 8];
        let mut i = 0;
        while i + 8 <= self.rank {
            for k in 0..8 {
                acc[k] += a[i + k] * b[i + k];
            }
            i += 8;
        }
        let mut sum: f32 = acc.iter().sum();
        while i < self.rank {
            sum += a[i] * b[i];
            i += 1;
        }
        sum.ln()
    }

    #[cfg(test)]
    pub fn from_tables(a: &[f32], b: &[f32], rank: usize, vocab: usize) -> Self {
        let inv = 1.0 / rank as f32;
        Self {
            rank,
            vocab,
            ea: a.iter().map(|x| x.exp()).collect(),
            eb: b.iter().map(|x| x.exp() * inv).collect(),
        }
    }
}
