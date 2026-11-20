//! The packed lexicon (`LEXICON1` version 4): the same words as version 3,
//! small enough to install with the model.
//!
//! The surfaces and the readings are each a table of the distinct strings
//! in order, front-coded: blocks of `BLOCK` strings, the first written
//! whole, each other as the chars it shares with the one before and the
//! rest, every char a code of the table's alphabet (its distinct chars in
//! order, so codes order as the chars do): a byte when it has 256 chars at
//! most, else the code's rank by how often the table's strings have it, its
//! 128 commonest in one byte and the others in two (`ranks` turns a rank
//! back into the code). A string is found by a binary search over the
//! blocks' first strings and a scan of one block. An entry (a form of a word)
//! holds, bit-packed, its reading's index, its (left id, right id, class)
//! triple's index and its cost in centinats (`COST_BITS`); the entries are
//! by surface, then cost, a bit vector marking each surface's first, and a
//! bit-packed list of them by reading, then cost, another bit vector
//! marking each reading's first. The connection costs are `CONN_BITS` each,
//! the index of the nearest of a codebook's values (centinats, picked for
//! the table's costs).
//!
//! Little-endian: the 64-byte header (magic, version u32 = 4, keys u32,
//! entries u32, readings u32, ids u32, flags u32, triples u32, then zeros)
//! and a directory, `SECTIONS` pairs of u32 (offset, length in bytes), of
//! the sections in `Sec` order, each padded to 8 bytes with 8 more after.

use super::{pad8, Entry, FLAG_PARTIAL, HEADER, MAGIC};
use anyhow::{ensure, Result};
use std::borrow::Cow;

pub(super) const VERSION: usize = 4;
/// Strings in a block of a front-coded table.
const BLOCK: usize = 16;
/// An entry's cost: centinats up to 2^12 - 1 (40.95 nats).
const COST_BITS: u32 = 12;
/// Bits a superblock of a bit vector counts (its words' ones before it).
const SUPER: usize = 512;
/// A connection cost: the index of one of 2^`CONN_BITS` values.
const CONN_BITS: u32 = 6;
/// Ranks of a two-byte table's chars written in one byte.
const ONE_BYTE: usize = 128;

#[derive(Clone, Copy)]
enum Sec {
    SurfAlphabet,
    SurfRanks,
    SurfBlocks,
    SurfData,
    ReadAlphabet,
    ReadRanks,
    ReadBlocks,
    ReadData,
    Triples,
    Entries,
    KeyBits,
    KeySuper,
    ByReading,
    ReadBits,
    ReadSuper,
    ConnBook,
    Connection,
}
const SECTIONS: usize = 17;

/// One row of the table, as `build` reads it: surface, reading, cost
/// (centinats), class, ids (lid << 16 | rid).
pub(super) type Row<'a> = (&'a str, &'a str, u16, u8, u32);

fn bits(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).max(1)
}

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

/// `width` bits at bit `i * width` of the bytes at `at` (`width` <= 57).
fn packed(b: &[u8], at: usize, width: u32, i: usize) -> u64 {
    let bit = i * width as usize;
    let o = at + bit / 8;
    let v = u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    (v >> (bit % 8)) & ((1u64 << width) - 1)
}

fn pack(values: impl Iterator<Item = u64>, width: u32) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let (mut acc, mut n) = (0u128, 0u32);
    for v in values {
        acc |= u128::from(v) << n;
        n += width;
        while n >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    if n > 0 {
        out.push(acc as u8);
    }
    out
}

/// A bit vector with rank and select.
#[derive(Clone, Copy)]
struct Bv {
    n: usize,
    words: usize,
    supers: usize,
}

impl Bv {
    fn word(&self, b: &[u8], w: usize) -> u64 {
        u64::from_le_bytes(
            b[self.words + 8 * w..self.words + 8 * w + 8]
                .try_into()
                .unwrap(),
        )
    }

    fn sup(&self, b: &[u8], s: usize) -> usize {
        u32_at(b, self.supers + 4 * s)
    }

    /// The ones before bit `i`.
    fn rank(&self, b: &[u8], i: usize) -> usize {
        let s = i / SUPER;
        let mut r = self.sup(b, s);
        let (w0, w1) = (s * SUPER / 64, i / 64);
        for w in w0..w1 {
            r += self.word(b, w).count_ones() as usize;
        }
        if i % 64 != 0 {
            r += (self.word(b, w1) & ((1u64 << (i % 64)) - 1)).count_ones() as usize;
        }
        r
    }

    /// Whether the superblocks count the ones before them, as `rank` and
    /// `select` take them to (their words and counts within the file);
    /// the ones in all.
    fn check(&self, b: &[u8]) -> Result<usize> {
        let mut ones = 0usize;
        for s in 0..=self.n.div_ceil(SUPER) {
            ensure!(
                self.sup(b, s) == ones,
                "lexicon: a bit vector's counts are off"
            );
            let w0 = s * SUPER / 64;
            let w1 = ((s + 1) * SUPER / 64).min(self.n.div_ceil(64));
            for w in w0..w1 {
                ones += self.word(b, w).count_ones() as usize;
            }
        }
        Ok(ones)
    }

    /// Where the `k`-th one is (from 0); `n` past the last.
    fn select(&self, b: &[u8], k: usize) -> usize {
        let supers = self.n.div_ceil(SUPER) + 1;
        // the last superblock with fewer than k + 1 ones before it
        let (mut lo, mut hi) = (0, supers);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.sup(b, mid) <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut left = k - self.sup(b, lo);
        let words = self.n.div_ceil(64);
        for w in lo * SUPER / 64..words {
            let mut v = self.word(b, w);
            let c = v.count_ones() as usize;
            if left < c {
                for _ in 0..left {
                    v &= v - 1;
                }
                return w * 64 + v.trailing_zeros() as usize;
            }
            left -= c;
        }
        self.n
    }

    fn build(set: &[bool]) -> (Vec<u8>, Vec<u8>) {
        let mut words = vec![0u64; set.len().div_ceil(64)];
        for (i, _) in set.iter().enumerate().filter(|(_, &s)| s) {
            words[i / 64] |= 1 << (i % 64);
        }
        let mut supers = Vec::new();
        let mut ones = 0u32;
        for (w, v) in words.iter().enumerate() {
            if w % (SUPER / 64) == 0 {
                supers.push(ones);
            }
            ones += v.count_ones();
        }
        // one past the end, for select
        supers.push(ones);
        (
            words.iter().flat_map(|v| v.to_le_bytes()).collect(),
            supers.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
}

/// A front-coded table of strings.
#[derive(Clone, Copy)]
struct Fct {
    n: usize,
    alphabet: usize,
    letters: usize,
    /// Codes by rank, for a table of more than 256 chars (0: a byte a code).
    ranks: usize,
    blocks: usize,
    data: usize,
}

impl Fct {
    /// The code of `c`, if the table has it.
    fn code(&self, b: &[u8], c: char) -> Option<u16> {
        let (mut lo, mut hi) = (0, self.letters);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let m = u32_at(b, self.alphabet + 4 * mid) as u32;
            match m.cmp(&(c as u32)) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u16),
            }
        }
        None
    }

    fn letter(&self, b: &[u8], code: u16) -> char {
        char::from_u32(u32_at(b, self.alphabet + 4 * code as usize) as u32).unwrap_or('\u{FFFD}')
    }

    /// `n` codes at `o`, onto `out`; where they end.
    fn codes_at(&self, b: &[u8], mut o: usize, n: usize, out: &mut Vec<u16>) -> usize {
        for _ in 0..n {
            if self.ranks == 0 {
                out.push(u16::from(b[o]));
                o += 1;
                continue;
            }
            let rank = if b[o] < ONE_BYTE as u8 {
                o += 1;
                usize::from(b[o - 1])
            } else {
                o += 2;
                ONE_BYTE + (usize::from(b[o - 2] & 0x7F) << 8 | usize::from(b[o - 1]))
            };
            let at = self.ranks + 2 * rank;
            out.push(u16::from_le_bytes([b[at], b[at + 1]]));
        }
        o
    }

    /// Decode the strings of block `blk` up to the one at `upto` (within
    /// it), calling `f` with each one's index and codes; stop when it says.
    fn scan(&self, b: &[u8], blk: usize, mut f: impl FnMut(usize, &[u16]) -> bool) {
        let mut o = self.data + u32_at(b, self.blocks + 4 * blk);
        let mut s: Vec<u16> = Vec::new();
        for k in 0..BLOCK {
            let i = blk * BLOCK + k;
            if i >= self.n {
                return;
            }
            if k == 0 {
                let len = b[o] as usize;
                o = self.codes_at(b, o + 1, len, &mut s);
            } else {
                let (shared, rest) = (b[o] as usize, b[o + 1] as usize);
                s.truncate(shared);
                o = self.codes_at(b, o + 2, rest, &mut s);
            }
            if !f(i, &s) {
                return;
            }
        }
    }

    fn get(&self, b: &[u8], i: usize) -> Vec<u16> {
        let mut out = Vec::new();
        self.scan(b, i / BLOCK, |k, s| {
            if k == i {
                out = s.to_vec();
                false
            } else {
                true
            }
        });
        out
    }

    fn text(&self, b: &[u8], i: usize) -> String {
        self.get(b, i).iter().map(|&c| self.letter(b, c)).collect()
    }

    /// Block `blk`'s first string against `probe`, decoded only as far as
    /// it takes.
    fn head_cmp(&self, b: &[u8], blk: usize, probe: &[u16]) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let mut o = self.data + u32_at(b, self.blocks + 4 * blk);
        let len = b[o] as usize;
        o += 1;
        let mut code = Vec::with_capacity(1);
        for (k, &p) in probe.iter().enumerate() {
            if k == len {
                return Ordering::Less;
            }
            code.clear();
            o = self.codes_at(b, o, 1, &mut code);
            match code[0].cmp(&p) {
                Ordering::Equal => {}
                other => return other,
            }
        }
        if len > probe.len() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// The first string not below `probe` among blocks `blo..bhi` (the
    /// first of `bhi` when none is), and that string when the scan saw it.
    fn lower_bound_in(
        &self,
        b: &[u8],
        probe: &[u16],
        blo: usize,
        bhi: usize,
    ) -> (usize, Option<Vec<u16>>) {
        // the last block whose first string is not above the probe
        let (mut lo, mut hi) = (blo, bhi);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.head_cmp(b, mid, probe).is_le() {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == blo {
            return ((blo * BLOCK).min(self.n), None);
        }
        let blk = lo - 1;
        let mut found = ((blk + 1) * BLOCK, None);
        self.scan(b, blk, |i, s| {
            if s >= probe {
                found = (i, Some(s.to_vec()));
                false
            } else {
                true
            }
        });
        (found.0.min(self.n), found.1)
    }

    /// Each prefix of `text` the table has, as the prefix's length in bytes
    /// and its index, shortest first; `max` chars at most. The blocks that
    /// may hold strings starting with the chars so far narrow down as it
    /// goes.
    fn prefixes(&self, b: &[u8], text: &str, max: usize, mut f: impl FnMut(usize, usize)) {
        let mut probe: Vec<u16> = Vec::new();
        let (mut blo, mut bhi) = (0, self.n.div_ceil(BLOCK));
        for (n, (i, c)) in text.char_indices().enumerate() {
            if n == max {
                return;
            }
            let Some(code) = self.code(b, c) else {
                return;
            };
            probe.push(code);
            let (at, seen) = self.lower_bound_in(b, &probe, blo, bhi);
            if at >= self.n {
                return;
            }
            let s = seen.unwrap_or_else(|| self.get(b, at));
            if !s.starts_with(&probe) {
                return;
            }
            if s == probe {
                f(i + c.len_utf8(), at);
            }
            // the strings past those that start with the probe
            let mut past = probe.clone();
            *past.last_mut().unwrap() += 1;
            let (end, _) = self.lower_bound_in(b, &past, at / BLOCK, bhi);
            blo = at / BLOCK;
            bhi = end.div_ceil(BLOCK).max(blo + 1).min(bhi);
        }
    }

    /// The table of `strings` (distinct, in order): alphabet, ranks (empty
    /// for a byte a code), block offsets and data.
    fn build(strings: &[&str]) -> Result<[Vec<u8>; 4]> {
        let mut letters: Vec<char> = strings.iter().flat_map(|s| s.chars()).collect();
        letters.sort_unstable();
        letters.dedup();
        ensure!(
            letters.len() <= ONE_BYTE + 0x8000,
            "{} distinct chars",
            letters.len()
        );
        let code = |c: char| letters.binary_search(&c).unwrap() as u16;
        // each code's rank by how often the strings have it
        let mut rank_of: Vec<usize> = (0..letters.len()).collect();
        let mut ranks: Vec<u16> = Vec::new();
        if letters.len() > 256 {
            let mut count = vec![0usize; letters.len()];
            for s in strings {
                for c in s.chars() {
                    count[code(c) as usize] += 1;
                }
            }
            ranks = (0..letters.len() as u16).collect();
            ranks.sort_by(|&a, &b| count[b as usize].cmp(&count[a as usize]).then(a.cmp(&b)));
            for (r, &c) in ranks.iter().enumerate() {
                rank_of[c as usize] = r;
            }
        }
        let two = !ranks.is_empty();
        let (mut blocks, mut data) = (Vec::new(), Vec::new());
        let mut prev: Vec<u16> = Vec::new();
        let put = |data: &mut Vec<u8>, codes: &[u16]| {
            for &c in codes {
                let r = rank_of[c as usize];
                if !two {
                    data.push(c as u8);
                } else if r < ONE_BYTE {
                    data.push(r as u8);
                } else {
                    let r = r - ONE_BYTE;
                    data.push(0x80 | (r >> 8) as u8);
                    data.push(r as u8);
                }
            }
        };
        for (i, s) in strings.iter().enumerate() {
            let codes: Vec<u16> = s.chars().map(code).collect();
            ensure!(codes.len() <= 255, "a string of {} chars", codes.len());
            if i % BLOCK == 0 {
                blocks.push(data.len() as u32);
                data.push(codes.len() as u8);
                put(&mut data, &codes);
            } else {
                let shared = prev.iter().zip(&codes).take_while(|(a, b)| a == b).count();
                data.push(shared as u8);
                data.push((codes.len() - shared) as u8);
                put(&mut data, &codes[shared..]);
            }
            prev = codes;
        }
        blocks.push(data.len() as u32);
        Ok([
            letters
                .iter()
                .flat_map(|&c| (c as u32).to_le_bytes())
                .collect(),
            ranks.iter().flat_map(|v| v.to_le_bytes()).collect(),
            blocks.iter().flat_map(|v| v.to_le_bytes()).collect(),
            data,
        ])
    }
}

pub(super) struct Packed {
    keys: usize,
    entries: usize,
    ids: usize,
    surf: Fct,
    read: Fct,
    triples: usize,
    entry_at: usize,
    read_bits: u32,
    triple_bits: u32,
    key_bv: Bv,
    by_reading: usize,
    entry_bits: u32,
    read_bv: Bv,
    conn: usize,
    conn_book: usize,
}

impl Packed {
    /// The packed lexicon in `b` (its header read: keys, entries, readings,
    /// ids, triples).
    pub(super) fn open(b: &[u8]) -> Result<Self> {
        let (keys, entries, readings, ids, triples) = (
            u32_at(b, 12),
            u32_at(b, 16),
            u32_at(b, 20),
            u32_at(b, 24),
            u32_at(b, 32),
        );
        ensure!(b.len() >= HEADER + 8 * SECTIONS, "lexicon truncated");
        let sec = |s: Sec| {
            let d = HEADER + 8 * s as usize;
            (u32_at(b, d), u32_at(b, d + 4))
        };
        // the sections start after the directory (a file of another layout
        // of version 4, without the readings' ranks, does not)
        ensure!(
            sec(Sec::SurfAlphabet).0 == pad8(HEADER + 8 * SECTIONS),
            "lexicon of an older layout: build it again"
        );
        // every section within the file, 8 bytes after it as the reads of
        // bit-packed values take, and as long as its counts make it
        let section = |s: Sec, bits_needed: Option<usize>| -> Result<()> {
            let (at, len) = sec(s);
            let end = at.checked_add(len).and_then(|e| e.checked_add(8));
            ensure!(
                end.is_some_and(|e| e <= b.len())
                    && bits_needed.is_some_and(|n| len.checked_mul(8).is_some_and(|l| l >= n)),
                "lexicon truncated"
            );
            Ok(())
        };
        let mul = |a: usize, b: usize| a.checked_mul(b);
        let entry_width = bits(readings) + bits(triples) + COST_BITS;
        ensure!(entry_width <= 57, "lexicon: an entry wider than 57 bits");
        let bv_words = mul(entries.div_ceil(64), 64);
        let bv_supers = mul(entries.div_ceil(SUPER) + 1, 32);
        for (s, need) in [
            (Sec::SurfAlphabet, Some(0)),
            (Sec::SurfRanks, Some(0)),
            (Sec::SurfBlocks, mul(keys.div_ceil(BLOCK), 32)),
            (Sec::SurfData, Some(0)),
            (Sec::ReadAlphabet, Some(0)),
            (Sec::ReadRanks, Some(0)),
            (Sec::ReadBlocks, mul(readings.div_ceil(BLOCK), 32)),
            (Sec::ReadData, Some(0)),
            (Sec::Triples, mul(triples, 64)),
            (Sec::Entries, mul(entries, entry_width as usize)),
            (Sec::KeyBits, bv_words),
            (Sec::KeySuper, bv_supers),
            (Sec::ByReading, mul(entries, bits(entries) as usize)),
            (Sec::ReadBits, bv_words),
            (Sec::ReadSuper, bv_supers),
            (Sec::ConnBook, Some(32 << CONN_BITS)),
            (
                Sec::Connection,
                mul(ids, ids).and_then(|n| n.checked_mul(CONN_BITS as usize)),
            ),
        ] {
            section(s, need)?;
        }
        let fct = |n: usize, a: Sec, r: Sec, k: Sec, d: Sec| Fct {
            n,
            alphabet: sec(a).0,
            letters: sec(a).1 / 4,
            ranks: if sec(r).1 > 0 { sec(r).0 } else { 0 },
            blocks: sec(k).0,
            data: sec(d).0,
        };
        Ok(Self {
            keys,
            entries,
            ids,
            surf: fct(
                keys,
                Sec::SurfAlphabet,
                Sec::SurfRanks,
                Sec::SurfBlocks,
                Sec::SurfData,
            ),
            read: fct(
                readings,
                Sec::ReadAlphabet,
                Sec::ReadRanks,
                Sec::ReadBlocks,
                Sec::ReadData,
            ),
            triples: sec(Sec::Triples).0,
            entry_at: sec(Sec::Entries).0,
            read_bits: bits(readings),
            triple_bits: bits(triples),
            key_bv: Bv {
                n: entries,
                words: sec(Sec::KeyBits).0,
                supers: sec(Sec::KeySuper).0,
            },
            by_reading: sec(Sec::ByReading).0,
            entry_bits: bits(entries),
            read_bv: Bv {
                n: entries,
                words: sec(Sec::ReadBits).0,
                supers: sec(Sec::ReadSuper).0,
            },
            conn: sec(Sec::Connection).0,
            conn_book: sec(Sec::ConnBook).0,
        })
        .and_then(|p| {
            p.check(b, readings, triples)?;
            Ok(p)
        })
    }

    /// Whether what the lookups read is all there: every string of both
    /// tables decodes within its data to codes of its alphabet, every
    /// entry's reading and triple are ones there are, the list by reading
    /// holds entries, and the bit vectors mark as many surfaces and
    /// readings as there are, the first entry the first's. A damaged file
    /// is refused here rather than taken down an app later.
    fn check(&self, b: &[u8], readings: usize, triples: usize) -> Result<()> {
        self.surf
            .check(b, &sec_len(b, Sec::SurfRanks), &sec_len(b, Sec::SurfData))?;
        self.read
            .check(b, &sec_len(b, Sec::ReadRanks), &sec_len(b, Sec::ReadData))?;
        let w = self.read_bits + self.triple_bits + COST_BITS;
        for e in 0..self.entries {
            let v = packed(b, self.entry_at, w, e);
            let reading = (v >> (self.triple_bits + COST_BITS)) as usize;
            let t = ((v >> COST_BITS) & ((1 << self.triple_bits) - 1)) as usize;
            ensure!(
                reading < readings && t < triples,
                "lexicon: an entry out of range"
            );
            let by = packed(b, self.by_reading, self.entry_bits, e) as usize;
            ensure!(
                by < self.entries,
                "lexicon: the list by reading out of range"
            );
        }
        let starts = |bv: &Bv| bv.word(b, 0) & 1 == 1;
        ensure!(
            self.key_bv.check(b)? == self.keys && self.read_bv.check(b)? == readings,
            "lexicon: the bit vectors mark other counts"
        );
        ensure!(
            self.entries == 0 || starts(&self.key_bv) && starts(&self.read_bv),
            "lexicon: the first entry starts no surface"
        );
        Ok(())
    }

    pub(super) fn ids(&self) -> usize {
        self.ids
    }

    /// The connection cost (centinats) of left id `l` after right id `r`.
    pub(super) fn connection(&self, b: &[u8], r: usize, l: usize) -> f32 {
        let k = packed(b, self.conn, CONN_BITS, r * self.ids + l) as usize;
        f32::from_bits(u32_at(b, self.conn_book + 4 * k) as u32)
    }

    /// Entry `e`, with its reading's index.
    fn raw(&self, b: &[u8], e: usize) -> (usize, Entry<'static>) {
        let w = self.read_bits + self.triple_bits + COST_BITS;
        let v = packed(b, self.entry_at, w, e);
        let reading = (v >> (self.triple_bits + COST_BITS)) as usize;
        let t = ((v >> COST_BITS) & ((1 << self.triple_bits) - 1)) as usize;
        let ids = u32_at(b, self.triples + 8 * t) as u32;
        let class = b[self.triples + 8 * t + 4];
        let cost = (v & ((1 << COST_BITS) - 1)) as f32 / 100.0;
        (
            reading,
            Entry {
                reading: Cow::Borrowed(""),
                cost,
                class,
                lid: (ids >> 16) as u16,
                rid: ids as u16,
            },
        )
    }

    fn entry(&self, b: &[u8], e: usize) -> Entry<'static> {
        let (r, mut entry) = self.raw(b, e);
        entry.reading = Cow::Owned(self.read.text(b, r));
        entry
    }

    /// The entries of key `k`.
    fn key_entries(&self, b: &[u8], k: usize) -> std::ops::Range<usize> {
        self.key_bv.select(b, k)..self.key_bv.select(b, k + 1)
    }

    pub(super) fn prefixes(
        &self,
        b: &[u8],
        text: &str,
        max: usize,
        mut f: impl FnMut(usize, Entry<'static>),
    ) {
        self.surf.prefixes(b, text, max, |len, k| {
            for e in self.key_entries(b, k) {
                f(len, self.entry(b, e));
            }
        });
    }

    pub(super) fn reading_prefixes(
        &self,
        b: &[u8],
        text: &str,
        max: usize,
        mut f: impl FnMut(usize, &str, Entry<'static>),
    ) {
        self.read.prefixes(b, text, max, |len, r| {
            let reading = &text[..len];
            for k in self.read_bv.select(b, r)..self.read_bv.select(b, r + 1) {
                let e = packed(b, self.by_reading, self.entry_bits, k) as usize;
                let (_, mut entry) = self.raw(b, e);
                entry.reading = Cow::Owned(reading.to_string());
                let key = self.key_bv.rank(b, e + 1) - 1;
                let surface = self.surf.text(b, key);
                f(len, &surface, entry);
            }
        });
    }

    pub(super) fn len(&self) -> usize {
        self.entries
    }

    pub(super) fn keys(&self) -> usize {
        self.keys
    }
}

/// The byte length of section `s` (its offset's checked by `open`).
fn sec_len(b: &[u8], s: Sec) -> usize {
    u32_at(b, HEADER + 8 * s as usize + 4)
}

impl Fct {
    /// Whether every string decodes within the data (`data` bytes) to
    /// codes of the alphabet, by ranks within the ranks (`ranks` bytes).
    fn check(&self, b: &[u8], ranks: &usize, data: &usize) -> Result<()> {
        let bad = || anyhow::anyhow!("lexicon: a string table out of range");
        let end = self.data + data;
        let byte = |o: usize| -> Result<u8> {
            if o < end {
                Ok(b[o])
            } else {
                Err(bad())
            }
        };
        let code = |o: &mut usize| -> Result<()> {
            let c = if self.ranks == 0 {
                *o += 1;
                usize::from(byte(*o - 1)?)
            } else {
                let first = byte(*o)?;
                let rank = if first < ONE_BYTE as u8 {
                    *o += 1;
                    usize::from(first)
                } else {
                    *o += 2;
                    ONE_BYTE + (usize::from(first & 0x7F) << 8 | usize::from(byte(*o - 1)?))
                };
                ensure!(2 * rank + 2 <= *ranks, bad());
                let at = self.ranks + 2 * rank;
                usize::from(u16::from_le_bytes([b[at], b[at + 1]]))
            };
            ensure!(c < self.letters, bad());
            Ok(())
        };
        for blk in 0..self.n.div_ceil(BLOCK) {
            let mut o = self.data + u32_at(b, self.blocks + 4 * blk);
            let mut len = 0usize;
            for k in 0..BLOCK.min(self.n - blk * BLOCK) {
                let rest = if k == 0 {
                    o += 1;
                    usize::from(byte(o - 1)?)
                } else {
                    let shared = usize::from(byte(o)?);
                    ensure!(shared <= len, bad());
                    o += 2;
                    len = shared;
                    usize::from(byte(o - 1)?)
                };
                for _ in 0..rest {
                    code(&mut o)?;
                }
                len += rest;
            }
        }
        Ok(())
    }
}

/// The bytes of a packed lexicon of `rows` (sorted by surface, then cost)
/// and the connection costs `costs` (centinats, `ids` squared).
pub(super) fn build(rows: &[Row<'_>], costs: &[i16], ids: usize, partial: bool) -> Result<Vec<u8>> {
    // the distinct surfaces and readings, in order
    let mut surfaces: Vec<&str> = rows.iter().map(|r| r.0).collect();
    surfaces.dedup();
    let mut readings: Vec<&str> = rows.iter().map(|r| r.1).collect();
    readings.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    readings.dedup();
    let mut triples: Vec<(u32, u8)> = rows.iter().map(|r| (r.4, r.3)).collect();
    triples.sort_unstable();
    triples.dedup();
    let (read_bits, triple_bits) = (bits(readings.len()), bits(triples.len()));
    let width = read_bits + triple_bits + COST_BITS;
    ensure!(width <= 57, "{width} bits an entry");
    let entries = pack(
        rows.iter().map(|r| {
            let reading = readings
                .binary_search_by(|x| x.as_bytes().cmp(r.1.as_bytes()))
                .unwrap() as u64;
            let t = triples.binary_search(&(r.4, r.3)).unwrap() as u64;
            let cost = u64::from(r.2.min((1 << COST_BITS) - 1));
            reading << (triple_bits + COST_BITS) | t << COST_BITS | cost
        }),
        width,
    );
    let starts: Vec<bool> = (0..rows.len())
        .map(|k| k == 0 || rows[k - 1].0 != rows[k].0)
        .collect();
    let (key_words, key_supers) = Bv::build(&starts);
    let mut by_reading: Vec<u32> = (0..rows.len() as u32).collect();
    by_reading.sort_by(|&a, &b| {
        let (a, b) = (&rows[a as usize], &rows[b as usize]);
        a.1.as_bytes().cmp(b.1.as_bytes()).then(a.2.cmp(&b.2))
    });
    let read_starts: Vec<bool> = (0..by_reading.len())
        .map(|k| k == 0 || rows[by_reading[k - 1] as usize].1 != rows[by_reading[k] as usize].1)
        .collect();
    let (read_words, read_supers) = Bv::build(&read_starts);
    let by_reading = pack(by_reading.iter().map(|&e| u64::from(e)), bits(rows.len()));
    let [sa, sr, sb, sd] = Fct::build(&surfaces)?;
    let [ra, rr, rb, rd] = Fct::build(&readings)?;
    let triple_bytes: Vec<u8> = triples
        .iter()
        .flat_map(|&(ids, class)| {
            let mut v = ids.to_le_bytes().to_vec();
            v.extend_from_slice(&[class, 0, 0, 0]);
            v
        })
        .collect();
    // the connection costs as codes of a codebook
    let book = codebook(costs, 1 << CONN_BITS);
    let nearest = |c: i16| -> u64 {
        let c = f32::from(c);
        let k = book.partition_point(|&v| v < c);
        let below = k.checked_sub(1);
        match (below, book.get(k)) {
            (Some(i), Some(&v)) if c - book[i] <= v - c => i as u64,
            (Some(i), None) => i as u64,
            _ => k as u64,
        }
    };
    let conn = pack(costs.iter().map(|&c| nearest(c)), CONN_BITS);
    let conn_book: Vec<u8> = book
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();

    let sections: [Vec<u8>; SECTIONS] = [
        sa,
        sr,
        sb,
        sd,
        ra,
        rr,
        rb,
        rd,
        triple_bytes,
        entries,
        key_words,
        key_supers,
        by_reading,
        read_words,
        read_supers,
        conn_book,
        conn,
    ];
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in [
        VERSION,
        surfaces.len(),
        rows.len(),
        readings.len(),
        ids,
        if partial { FLAG_PARTIAL } else { 0 },
        triples.len(),
    ] {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    out.resize(HEADER, 0);
    let dir = out.len();
    out.resize(dir + 8 * SECTIONS, 0);
    out.resize(pad8(out.len()), 0);
    for (k, s) in sections.iter().enumerate() {
        let at = out.len();
        out[dir + 8 * k..dir + 8 * k + 4].copy_from_slice(&(at as u32).to_le_bytes());
        out[dir + 8 * k + 4..dir + 8 * k + 8].copy_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s);
        out.resize(pad8(out.len()) + 8, 0);
    }
    Ok(out)
}

/// `n` values (ascending) that the costs are nearest to as a whole: Lloyd's
/// k-means over the costs' histogram, from their quantiles; a table of `n`
/// distinct costs or fewer keeps them all.
fn codebook(costs: &[i16], n: usize) -> Vec<f32> {
    let mut hist: std::collections::BTreeMap<i16, usize> = std::collections::BTreeMap::new();
    for &c in costs {
        *hist.entry(c).or_default() += 1;
    }
    let values: Vec<(f32, f64)> = hist
        .iter()
        .map(|(&c, &k)| (f32::from(c), k as f64))
        .collect();
    if values.len() <= n {
        let mut book: Vec<f32> = values.iter().map(|v| v.0).collect();
        book.resize(n, *book.last().unwrap_or(&0.0));
        return book;
    }
    let total: f64 = values.iter().map(|v| v.1).sum();
    let mut book: Vec<f32> = (0..n)
        .map(|k| {
            let want = (k as f64 + 0.5) / n as f64 * total;
            let mut acc = 0.0;
            values
                .iter()
                .find(|v| {
                    acc += v.1;
                    acc >= want
                })
                .map_or(0.0, |v| v.0)
        })
        .collect();
    for _ in 0..30 {
        book.sort_by(f32::total_cmp);
        book.dedup();
        let mut sum = vec![(0f64, 0f64); book.len()];
        for &(v, k) in &values {
            let i = book.partition_point(|&x| x < v);
            let i = match (i.checked_sub(1), book.get(i)) {
                (Some(j), Some(&x)) if v - book[j] <= x - v => j,
                (Some(j), None) => j,
                _ => i,
            };
            sum[i].0 += f64::from(v) * k;
            sum[i].1 += k;
        }
        for (b, (s, k)) in book.iter_mut().zip(&sum) {
            if *k > 0.0 {
                *b = (s / k) as f32;
            }
        }
    }
    book.sort_by(f32::total_cmp);
    book.resize(n, *book.last().unwrap_or(&0.0));
    book
}
