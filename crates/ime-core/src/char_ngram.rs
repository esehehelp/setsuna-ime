//! Character n-gram scores of the engine's candidates (`[ngram]`).
//!
//! The model emits each char from its own frame, so a candidate can join
//! two words the reading allows into a sequence no text has (博紙 for
//! はくし, 博士 and 白紙 blended). Counts of adjacent chars in the training
//! text catch that: each candidate's score gets `bigram_weight * Σ min(0,
//! PMI(a, b))` over its adjacent chars and `trigram_weight * Σ min(0,
//! PMI(ab, c))` over its triples, the last char of the context included,
//! and the list is reordered by the sum. Only sequences rarer than chance
//! count, so a common word is never favoured over the model's choice, a
//! blend that never occurs is pushed down.
//!
//! The table (`CHNGRAM1`), little-endian, a 64-byte header (magic, version
//! u32 = 1, chars u32, bigrams u32, trigrams u32, total u64, rows u64,
//! min_bigram u32, min_trigram u32) and then, each padded to 8 bytes:
//! chars u32[chars] (code points, ascending: a char's id is its index),
//! left u64[chars] and right u64[chars] (bigrams starting / ending with the
//! char), bigram_start u32[chars + 1], bigram_next u16[bigrams] (ascending
//! per first char), bigram_count u32[bigrams], trigram_start u32[bigrams +
//! 1] (per bigram entry), trigram_next u16[trigrams], trigram_log
//! u8[trigrams] (round(14 ln(count + 0.5))). A trigram whose absence costs nothing (its unseen score
//! is 0) is left out of the table. The file is mapped, not read, so the
//! processes the IME runs in share one copy.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{ensure, Context, Result};

const MAGIC: &[u8; 8] = b"CHNGRAM1";
const HEADER: usize = 64;
/// Steps per nat of `trigram_log`.
const LOG_SCALE: f64 = 14.0;

enum Bytes {
    Mapped(memmap2::Mmap),
    #[cfg_attr(not(test), allow(dead_code))]
    Owned(Vec<u8>),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

/// A character bigram / trigram table.
pub struct CharNgram {
    bytes: Bytes,
    chars: usize,
    total: f64,
    /// Byte offsets of the sections, in the order the header lists them.
    at: [usize; 9],
}

#[derive(Clone, Copy)]
enum Sec {
    Chars,
    Left,
    Right,
    BigramStart,
    BigramNext,
    BigramCount,
    TrigramStart,
    TrigramNext,
    TrigramLog,
}

impl CharNgram {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the table is a read-only data file the IME installs; it is
        // not written while mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("map {}", path.display()))?;
        Self::parse(Bytes::Mapped(map))
            .with_context(|| format!("char n-gram table {}", path.display()))
    }

    #[cfg(test)]
    fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Bytes::Owned(bytes))
    }

    fn parse(bytes: Bytes) -> Result<Self> {
        ensure!(
            bytes.len() >= HEADER && &bytes[..8] == MAGIC,
            "not a CHNGRAM1 table"
        );
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        ensure!(u32_at(8) == 1, "version {} (1 expected)", u32_at(8));
        let (chars, bigrams, trigrams) = (u32_at(12), u32_at(16), u32_at(20));
        let total = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as f64;
        let sizes = [
            4 * chars,
            8 * chars,
            8 * chars,
            4 * (chars + 1),
            2 * bigrams,
            4 * bigrams,
            4 * (bigrams + 1),
            2 * trigrams,
            trigrams,
        ];
        let mut at = [0usize; 9];
        let mut end = HEADER;
        for (a, size) in at.iter_mut().zip(sizes) {
            *a = end;
            end = (end + size).div_ceil(8) * 8;
        }
        ensure!(
            bytes.len() >= end,
            "truncated: {} bytes, {end} expected",
            bytes.len()
        );
        let t = Self {
            bytes,
            chars,
            total,
            at,
        };
        ensure!(
            t.u32(Sec::BigramStart, chars) as usize == bigrams
                && t.u32(Sec::TrigramStart, bigrams) as usize == trigrams,
            "section counts disagree"
        );
        Ok(t)
    }

    fn u8(&self, s: Sec, i: usize) -> u8 {
        self.bytes[self.at[s as usize] + i]
    }

    fn u16(&self, s: Sec, i: usize) -> u16 {
        let p = self.at[s as usize] + 2 * i;
        u16::from_le_bytes(self.bytes[p..p + 2].try_into().unwrap())
    }

    fn u32(&self, s: Sec, i: usize) -> u32 {
        let p = self.at[s as usize] + 4 * i;
        u32::from_le_bytes(self.bytes[p..p + 4].try_into().unwrap())
    }

    fn u64(&self, s: Sec, i: usize) -> f64 {
        let p = self.at[s as usize] + 8 * i;
        u64::from_le_bytes(self.bytes[p..p + 8].try_into().unwrap()) as f64
    }

    /// The id of `c`, if the table has it.
    fn id(&self, c: char) -> Option<u16> {
        let c = c as u32;
        let (mut lo, mut hi) = (0, self.chars);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.u32(Sec::Chars, mid).cmp(&c) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u16),
            }
        }
        None
    }

    /// Index of `next` in `[lo, hi)` of the ascending section `s`.
    fn find(&self, s: Sec, lo: usize, hi: usize, next: u16) -> Option<usize> {
        let (mut lo, mut hi) = (lo, hi);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.u16(s, mid).cmp(&next) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The bigram entry of `ab` and its count.
    fn bigram(&self, a: u16, b: u16) -> (Option<usize>, f64) {
        let (lo, hi) = (
            self.u32(Sec::BigramStart, a as usize) as usize,
            self.u32(Sec::BigramStart, a as usize + 1) as usize,
        );
        match self.find(Sec::BigramNext, lo, hi, b) {
            Some(e) => (Some(e), self.u32(Sec::BigramCount, e) as f64),
            None => (None, 0.0),
        }
    }

    /// `Σ min(0, PMI2)` and `Σ min(0, PMI3)` over `text` after `prev` (the
    /// context's last char). A pair or triple with a char the table lacks
    /// counts nothing.
    pub fn terms(&self, prev: Option<char>, text: &str) -> (f64, f64) {
        let ids: Vec<Option<u16>> = prev
            .into_iter()
            .chain(text.chars())
            .map(|c| self.id(c))
            .collect();
        let (mut p2, mut p3) = (0.0, 0.0);
        // the pair before: its bigram entry and count
        let mut last: Option<(Option<usize>, f64)> = None;
        for i in 1..ids.len() {
            let (Some(a), Some(b)) = (ids[i - 1], ids[i]) else {
                last = None;
                continue;
            };
            let (entry, c_ab) = self.bigram(a, b);
            let (left, right) = (
                self.u64(Sec::Left, a as usize),
                self.u64(Sec::Right, b as usize),
            );
            p2 += ((c_ab + 0.5) * self.total / (left * right + 1.0))
                .ln()
                .min(0.0);
            // the triple ending here: (x, a) before, then b
            if let Some((prev_entry, c_xa)) = last {
                // ln(count + 0.5)
                let ln_n = prev_entry
                    .and_then(|e| {
                        let lo = self.u32(Sec::TrigramStart, e) as usize;
                        let hi = self.u32(Sec::TrigramStart, e + 1) as usize;
                        self.find(Sec::TrigramNext, lo, hi, b)
                    })
                    .map_or(0.5f64.ln(), |t| {
                        f64::from(self.u8(Sec::TrigramLog, t)) / LOG_SCALE
                    });
                let right_a = self.u64(Sec::Right, a as usize);
                p3 += (ln_n + ((right_a + 1.0) / ((c_xa + 0.5) * (c_ab + 0.5))).ln()).min(0.0);
            }
            last = Some((entry, c_ab));
        }
        (p2, p3)
    }

    /// The order of `candidates` by `score + bigram_weight * Σ min(0, PMI2)
    /// + trigram_weight * Σ min(0, PMI3)`, best first (a tie keeps the
    /// engine's order), with those sums.
    pub fn rescore(
        &self,
        context: &str,
        candidates: &[String],
        scores: &[f32],
        bigram_weight: f32,
        trigram_weight: f32,
    ) -> (Vec<usize>, Vec<f32>) {
        let prev = last_char(context);
        let total: Vec<f32> = candidates
            .iter()
            .zip(scores)
            .map(|(c, &s)| {
                let (p2, p3) = self.terms(prev, c);
                (f64::from(s) + f64::from(bigram_weight) * p2 + f64::from(trigram_weight) * p3)
                    as f32
            })
            .collect();
        let mut order: Vec<usize> = (0..total.len()).collect();
        order.sort_by(|&a, &b| total[b].total_cmp(&total[a]));
        (order, total)
    }
}

/// The context's last char as the model reads it (NFKC).
fn last_char(context: &str) -> Option<char> {
    use unicode_normalization::UnicodeNormalization;
    let n = context.chars().count();
    let tail: String = context.chars().skip(n.saturating_sub(4)).collect();
    tail.nfkc().last()
}

/// The table at `path`, mapped once per process (`None`, logged, when it
/// does not load).
pub fn shared(path: &str) -> Option<Arc<CharNgram>> {
    type Cache = HashMap<String, Option<Arc<CharNgram>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(path.to_string())
        .or_insert_with(|| match CharNgram::open(Path::new(path)) {
            Ok(t) => Some(Arc::new(t)),
            Err(e) => {
                crate::diagnostics::log(format!("ngram table: {e:#}"));
                None
            }
        })
        .clone()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A table of `text`'s bigrams and trigrams, as the writer makes it
    /// (every seen trigram kept).
    pub(crate) fn table(text: &str) -> Vec<u8> {
        let cs: Vec<char> = text.chars().collect();
        let mut chars: Vec<u32> = cs.iter().map(|&c| c as u32).collect();
        chars.sort_unstable();
        chars.dedup();
        let id = |c: char| chars.binary_search(&(c as u32)).unwrap();
        let v = chars.len();
        let mut big = std::collections::BTreeMap::<(usize, usize), u32>::new();
        let mut tri = std::collections::BTreeMap::<(usize, usize, usize), u32>::new();
        for w in cs.windows(2) {
            *big.entry((id(w[0]), id(w[1]))).or_default() += 1;
        }
        for w in cs.windows(3) {
            *tri.entry((id(w[0]), id(w[1]), id(w[2]))).or_default() += 1;
        }
        let (mut left, mut right) = (vec![0u64; v], vec![0u64; v]);
        for (&(a, b), &n) in &big {
            left[a] += n as u64;
            right[b] += n as u64;
        }
        let keys: Vec<(usize, usize)> = big.keys().copied().collect();
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for x in [1u32, v as u32, big.len() as u32, tri.len() as u32] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&left.iter().sum::<u64>().to_le_bytes());
        out.resize(HEADER, 0);
        let pad = |out: &mut Vec<u8>| out.resize(out.len().div_ceil(8) * 8, 0);
        chars
            .iter()
            .for_each(|c| out.extend_from_slice(&c.to_le_bytes()));
        pad(&mut out);
        left.iter()
            .for_each(|c| out.extend_from_slice(&c.to_le_bytes()));
        right
            .iter()
            .for_each(|c| out.extend_from_slice(&c.to_le_bytes()));
        for a in 0..=v {
            let n = keys.iter().filter(|k| k.0 < a).count() as u32;
            out.extend_from_slice(&n.to_le_bytes());
        }
        pad(&mut out);
        keys.iter()
            .for_each(|k| out.extend_from_slice(&(k.1 as u16).to_le_bytes()));
        pad(&mut out);
        big.values()
            .for_each(|n| out.extend_from_slice(&n.to_le_bytes()));
        pad(&mut out);
        for e in 0..=keys.len() {
            let n = tri
                .keys()
                .filter(|t| keys[..e].contains(&(t.0, t.1)))
                .count() as u32;
            out.extend_from_slice(&n.to_le_bytes());
        }
        pad(&mut out);
        tri.keys()
            .for_each(|t| out.extend_from_slice(&(t.2 as u16).to_le_bytes()));
        pad(&mut out);
        tri.values()
            .for_each(|&n| out.push(((n as f64 + 0.5).ln() * LOG_SCALE).round() as u8));
        pad(&mut out);
        out
    }

    /// 博, 紙 and が frequent enough that 博紙 and 士が白 are rarer than
    /// chance.
    fn text() -> String {
        "博士が来る。猫が白い。白紙に書く。".repeat(20)
    }

    #[test]
    fn a_blend_no_text_has_scores_below_the_words() {
        let t = CharNgram::from_bytes(table(&text())).unwrap();
        let (b2, b3) = t.terms(None, "博紙");
        let (w2, w3) = t.terms(None, "博士");
        assert!(b2 < w2 && w2 == 0.0, "{b2} {w2}");
        assert_eq!((b3, w3), (0.0, 0.0));
        // a triple: 士が and が白 are common, 士が白 is never seen, 猫が白 is
        let (_, unseen) = t.terms(None, "士が白");
        let (_, seen) = t.terms(None, "猫が白");
        assert!(unseen < seen && seen == 0.0, "{unseen} {seen}");
        // a char the table lacks counts nothing
        assert_eq!(t.terms(None, "博X紙").0, 0.0);
        // the context's last char is the first char's left neighbour
        assert!(t.terms(Some('白'), "士").0 < t.terms(Some('博'), "士").0);
    }

    #[test]
    fn rescore_pushes_the_blend_down_and_keeps_ties_in_order() {
        let t = CharNgram::from_bytes(table(&text())).unwrap();
        let cands: Vec<String> = ["博紙", "博士", "白紙"].map(String::from).into();
        let (order, total) = t.rescore("", &cands, &[-1.0, -1.2, -1.2], 0.7, 0.5);
        assert_eq!(order, [1, 2, 0]);
        assert!(total[0] < -1.2);
        let (order, _) = t.rescore("", &cands, &[-1.0, -1.2, -1.2], 0.0, 0.0);
        assert_eq!(order, [0, 1, 2]);
    }

    #[test]
    fn a_table_that_is_not_one_is_refused() {
        assert!(CharNgram::from_bytes(b"CHNGRAM2".repeat(10)).is_err());
        let mut short = table(&text());
        short.truncate(short.len() - 16);
        assert!(CharNgram::from_bytes(short).is_err());
    }
}
