//! The mixed-input model's tables (`mixed::MixedModel`) as one block of
//! bytes read where it lies: the kana and the letter trigram models as open
//! addressing tables of packed char triples, and the Latin words sorted, with
//! their counts and running totals. Made from the model's TSV files once and
//! kept in the cache directory ([`crate::config::cache_dir`]), the block maps
//! into every process that loads the model (each app the IME runs in) as the
//! same pages; read into each process's heap, the model took 27 MB of each.
//!
//! The block is checked when it is opened (its sections within it, the
//! words' offsets in order), and read through bounds-checked slices: a block
//! that is not what it should be gives wrong costs, never a crash.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

/// The format; a cached block of another is made again.
const MAGIC: &[u8; 8] = b"SMIXTB01";
const HEADER: usize = 8 + 6 * 8;
/// An empty slot: no packed triple is all ones (a char is below 2^21).
const EMPTY: u64 = u64::MAX;
/// What a gram of one or two chars has for the chars before it.
pub(crate) const NONE: char = '\0';

fn pack3(a: char, b: char, c: char) -> u64 {
    ((a as u64) << 42) | ((b as u64) << 21) | c as u64
}

fn pack2(a: char, b: char) -> u64 {
    ((a as u64) << 21) | b as u64
}

fn slot(key: u64, bits: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
}

/// The counts of 1- to 3-grams a char trigram model (Witten–Bell
/// interpolation) is made of.
#[derive(Default)]
pub(crate) struct Counts {
    counts: HashMap<u64, u32>,
}

impl Counts {
    pub(crate) fn add(&mut self, gram: &str, count: u32) {
        let chars: Vec<char> = gram.chars().collect();
        self.add_chars(&chars, count);
    }

    pub(crate) fn add_chars(&mut self, chars: &[char], count: u32) {
        let key = match *chars {
            [c] => pack3(NONE, NONE, c),
            [b, c] => pack3(NONE, b, c),
            [a, b, c] => pack3(a, b, c),
            _ => return,
        };
        *self.counts.entry(key).or_insert(0) += count;
    }
}

/// A Latin word of the model's text and its counts in Japanese and in
/// English text.
pub(crate) struct Word {
    pub(crate) word: String,
    pub(crate) ja: u32,
    pub(crate) en: u32,
}

/// The block of the two models and the words (sorted here).
/// A word listed twice has the counts of its last line; the totals are of
/// every line, as the model counted them in maps.
pub(crate) fn write(kana: &Counts, letters: &Counts, mut words: Vec<Word>) -> Vec<u8> {
    let (mut ja_total, mut en_total) = (0.0f64, 0.0f64);
    for w in &words {
        ja_total += w.ja as f64;
        en_total += w.en as f64;
    }
    // stable: of equal words, the last line's counts stay
    words.sort_by(|a, b| a.word.cmp(&b.word));
    words.dedup_by(|later, kept| {
        let same = later.word == kept.word;
        if same {
            std::mem::swap(later, kept);
        }
        same
    });
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.resize(HEADER, 0);
    let kana_at = out.len();
    write_lm(&mut out, kana);
    let letters_at = out.len();
    write_lm(&mut out, letters);
    let words_at = out.len();
    write_words(&mut out, &words);
    let len = out.len();
    for (i, v) in [
        kana_at as u64,
        letters_at as u64,
        words_at as u64,
        ja_total.max(1.0).to_bits(),
        en_total.max(1.0).to_bits(),
        len as u64,
    ]
    .into_iter()
    .enumerate()
    {
        out[8 + 8 * i..16 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Slots for `n` keys, at most three quarters of them taken.
fn bits_for(n: usize) -> u32 {
    let want = (n + n / 3 + 1).max(2);
    want.next_power_of_two().trailing_zeros()
}

/// `[unigram total][vocabulary][count bits][context bits]`, the counts'
/// table (keys, counts), the contexts' (keys, totals, types).
fn write_lm(out: &mut Vec<u8>, lm: &Counts) {
    let (mut unigram_total, mut vocab) = (0u64, 0u64);
    let mut contexts: HashMap<u64, (u64, u32)> = HashMap::new();
    for (&key, &count) in &lm.counts {
        let (a, b) = (key >> 42, (key >> 21) & 0x1F_FFFF);
        if b == NONE as u64 {
            unigram_total += count as u64;
            vocab += 1;
        } else {
            let e = contexts.entry((a << 21) | b).or_insert((0, 0));
            e.0 += count as u64;
            e.1 += 1;
        }
    }
    let (cbits, xbits) = (bits_for(lm.counts.len()), bits_for(contexts.len()));
    for v in [unigram_total, vocab, cbits as u64, xbits as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let place = |keys: &mut Vec<u64>, bits: u32, key: u64| {
        let mask = (1usize << bits) - 1;
        let mut i = slot(key, bits);
        while keys[i] != EMPTY {
            i = (i + 1) & mask;
        }
        keys[i] = key;
        i
    };
    let mut keys = vec![EMPTY; 1 << cbits];
    let mut vals = vec![0u32; 1 << cbits];
    for (&key, &count) in &lm.counts {
        let i = place(&mut keys, cbits, key);
        vals[i] = count;
    }
    keys.iter()
        .for_each(|k| out.extend_from_slice(&k.to_le_bytes()));
    vals.iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    let mut keys = vec![EMPTY; 1 << xbits];
    let mut totals = vec![0u64; 1 << xbits];
    let mut types = vec![0u32; 1 << xbits];
    for (&key, &(total, n)) in &contexts {
        let i = place(&mut keys, xbits, key);
        totals[i] = total;
        types[i] = n;
    }
    keys.iter()
        .for_each(|k| out.extend_from_slice(&k.to_le_bytes()));
    totals
        .iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    types
        .iter()
        .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
}

/// `[n][text bytes]`, the offsets of the words' text (n + 1), the text,
/// the counts (Japanese, English), the running totals before each word
/// (n + 1 each).
fn write_words(out: &mut Vec<u8>, words: &[Word]) {
    let text: usize = words.iter().map(|w| w.word.len()).sum();
    out.extend_from_slice(&(words.len() as u64).to_le_bytes());
    out.extend_from_slice(&(text as u64).to_le_bytes());
    let mut at = 0u32;
    for w in words {
        out.extend_from_slice(&at.to_le_bytes());
        at += w.word.len() as u32;
    }
    out.extend_from_slice(&at.to_le_bytes());
    words
        .iter()
        .for_each(|w| out.extend_from_slice(w.word.as_bytes()));
    words
        .iter()
        .for_each(|w| out.extend_from_slice(&w.ja.to_le_bytes()));
    words
        .iter()
        .for_each(|w| out.extend_from_slice(&w.en.to_le_bytes()));
    for pick in [|w: &Word| w.ja, |w: &Word| w.en] {
        let mut run = 0u64;
        for w in words {
            out.extend_from_slice(&run.to_le_bytes());
            run += pick(w) as u64;
        }
        out.extend_from_slice(&run.to_le_bytes());
    }
}

enum Bytes {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped(m) => m,
        }
    }
}

#[derive(Clone, Copy)]
struct Table {
    bits: u32,
    keys: usize,
    vals: usize,
    /// The contexts' types (counts have none).
    types: usize,
}

#[derive(Clone, Copy)]
struct Lm {
    unigram_total: u64,
    vocab: u64,
    counts: Table,
    contexts: Table,
}

#[derive(Clone, Copy)]
struct Words {
    n: usize,
    offsets: usize,
    text: usize,
    ja: usize,
    en: usize,
    run_ja: usize,
    run_en: usize,
}

/// The tables, read in place.
pub(crate) struct Tables {
    bytes: Bytes,
    kana: Lm,
    letters: Lm,
    words: Words,
    ja_total: f64,
    en_total: f64,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    let mut a = [0; 4];
    a.copy_from_slice(&b[o..o + 4]);
    u32::from_le_bytes(a)
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut a = [0; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

/// `at` plus `n` items of `size` bytes, where it stays within `len`.
fn span(at: usize, n: usize, size: usize, len: usize) -> Result<usize> {
    let end = n
        .checked_mul(size)
        .and_then(|s| s.checked_add(at))
        .context("section past the end")?;
    ensure!(end <= len, "section past the end");
    Ok(end)
}

impl Tables {
    /// The tables of a block made by [`write`].
    pub(crate) fn from_vec(bytes: Vec<u8>) -> Result<Self> {
        Self::from_bytes(Bytes::Owned(bytes))
    }

    fn from_bytes(bytes: Bytes) -> Result<Self> {
        let len = bytes.len();
        ensure!(
            len >= HEADER && &bytes[..8] == MAGIC,
            "not a mixed tables block"
        );
        let h = |i: usize| u64_at(&bytes, 8 + 8 * i);
        ensure!(
            h(5) == len as u64,
            "block of {} bytes, header says {}",
            len,
            h(5)
        );
        let lm = |at: u64| -> Result<(Lm, usize)> {
            let at = usize::try_from(at)?;
            let fields = span(at, 4, 8, len)?;
            let (cbits, xbits) = (u64_at(&bytes, at + 16), u64_at(&bytes, at + 24));
            ensure!(
                (1..=30).contains(&cbits) && (1..=30).contains(&xbits),
                "table size"
            );
            let (cn, xn) = (1usize << cbits, 1usize << xbits);
            let ckeys = fields;
            let cvals = span(ckeys, cn, 8, len)?;
            let xkeys = span(cvals, cn, 4, len)?;
            let xtotals = span(xkeys, xn, 8, len)?;
            let xtypes = span(xtotals, xn, 8, len)?;
            let end = span(xtypes, xn, 4, len)?;
            Ok((
                Lm {
                    unigram_total: u64_at(&bytes, at),
                    vocab: u64_at(&bytes, at + 8),
                    counts: Table {
                        bits: cbits as u32,
                        keys: ckeys,
                        vals: cvals,
                        types: 0,
                    },
                    contexts: Table {
                        bits: xbits as u32,
                        keys: xkeys,
                        vals: xtotals,
                        types: xtypes,
                    },
                },
                end,
            ))
        };
        let (kana, _) = lm(h(0))?;
        let (letters, _) = lm(h(1))?;
        let at = usize::try_from(h(2))?;
        let fields = span(at, 2, 8, len)?;
        let n = usize::try_from(u64_at(&bytes, at))?;
        let text_len = usize::try_from(u64_at(&bytes, at + 8))?;
        let offsets = fields;
        let text = span(offsets, n.checked_add(1).context("words")?, 4, len)?;
        let ja = span(text, text_len, 1, len)?;
        let en = span(ja, n, 4, len)?;
        let run_ja = span(en, n, 4, len)?;
        let run_en = span(run_ja, n + 1, 8, len)?;
        span(run_en, n + 1, 8, len)?;
        // the words' offsets in order, the last at the text's end
        let mut last = 0;
        for i in 0..=n {
            let o = u32_at(&bytes, offsets + 4 * i) as usize;
            if o < last || o > text_len {
                bail!("word offsets out of order");
            }
            last = o;
        }
        ensure!(last == text_len, "word offsets short of the text");
        let words = Words {
            n,
            offsets,
            text,
            ja,
            en,
            run_ja,
            run_en,
        };
        Ok(Self {
            kana,
            letters,
            words,
            ja_total: f64::from_bits(h(3)),
            en_total: f64::from_bits(h(4)),
            bytes,
        })
    }

    fn find(&self, t: Table, key: u64) -> Option<usize> {
        let b = &self.bytes;
        let mask = (1usize << t.bits) - 1;
        let mut i = slot(key, t.bits);
        for _ in 0..=mask {
            match u64_at(b, t.keys + 8 * i) {
                k if k == key => return Some(i),
                EMPTY => return None,
                _ => i = (i + 1) & mask,
            }
        }
        None
    }

    /// −ln P(c | a b) under `lm`.
    fn cost(&self, lm: &Lm, a: char, b: char, c: char) -> f64 {
        let bytes = &self.bytes;
        let count = |key: u64| {
            self.find(lm.counts, key)
                .map_or(0.0, |i| u32_at(bytes, lm.counts.vals + 4 * i) as f64)
        };
        let p1 =
            (count(pack3(NONE, NONE, c)) + 1.0) / (lm.unigram_total as f64 + lm.vocab as f64 + 1.0);
        let interpolate = |ctx: u64, key: u64, lower: f64| match self.find(lm.contexts, ctx) {
            Some(i) => {
                let total = u64_at(bytes, lm.contexts.vals + 8 * i) as f64;
                let types = u32_at(bytes, lm.contexts.types + 4 * i) as f64;
                (count(key) + types * lower) / (total + types)
            }
            None => lower,
        };
        let p2 = interpolate(pack2(NONE, b), pack3(NONE, b, c), p1);
        let p3 = interpolate(pack2(a, b), pack3(a, b, c), p2);
        -p3.ln()
    }

    /// −ln P(c | a b) of the kana model.
    pub(crate) fn kana_cost(&self, a: char, b: char, c: char) -> f64 {
        self.cost(&self.kana, a, b, c)
    }

    /// −ln P(c | a b) of the letter model.
    pub(crate) fn letters_cost(&self, a: char, b: char, c: char) -> f64 {
        self.cost(&self.letters, a, b, c)
    }

    fn word_at(&self, i: usize) -> &[u8] {
        let w = self.words;
        let from = u32_at(&self.bytes, w.offsets + 4 * i) as usize;
        let to = u32_at(&self.bytes, w.offsets + 4 * i + 4) as usize;
        &self.bytes[w.text + from..w.text + to]
    }

    /// The first word index in `from..to` where `pred` turns false.
    fn partition_point(&self, from: usize, to: usize, pred: impl Fn(&[u8]) -> bool) -> usize {
        let (mut lo, mut hi) = (from, to);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.word_at(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The counts (Japanese, English) of `word`, if the model has it.
    pub(crate) fn word(&self, word: &str) -> Option<(u32, u32)> {
        let w = self.words;
        let i = self.partition_point(0, w.n, |x| x < word.as_bytes());
        (i < w.n && self.word_at(i) == word.as_bytes()).then(|| {
            (
                u32_at(&self.bytes, w.ja + 4 * i),
                u32_at(&self.bytes, w.en + 4 * i),
            )
        })
    }

    /// The counts (Japanese, English) of the words starting with `prefix`.
    pub(crate) fn prefix_counts(&self, prefix: &str) -> (u64, u64) {
        let w = self.words;
        let p = prefix.as_bytes();
        let from = self.partition_point(0, w.n, |x| x < p);
        let to = self.partition_point(from, w.n, |x| x.starts_with(p));
        let run = |at: usize, i: usize| u64_at(&self.bytes, at + 8 * i);
        (
            run(w.run_ja, to) - run(w.run_ja, from),
            run(w.run_en, to) - run(w.run_en, from),
        )
    }

    pub(crate) fn n_words(&self) -> usize {
        self.words.n
    }

    pub(crate) fn ja_total(&self) -> f64 {
        self.ja_total
    }

    pub(crate) fn en_total(&self) -> f64 {
        self.en_total
    }

    /// Whether the tables are a file's pages (shared), not this process's
    /// heap.
    #[cfg(test)]
    fn mapped(&self) -> bool {
        matches!(self.bytes, Bytes::Mapped(_))
    }
}

/// The tables of the model in `dir`: from the cache when they were made for
/// its files as they are, else made from them (`make`: the TSV texts of
/// kana.tsv and latin.tsv) and put there for the next process. Without a
/// cache ([`crate::config::cache_dir`]: an elevated process) or when it
/// cannot be written, in this process's heap.
pub(crate) fn load(dir: &Path, make: impl FnOnce(&str, &str) -> Vec<u8>) -> Result<Tables> {
    let read = |name: &str| {
        let path = dir.join(name);
        std::fs::read_to_string(&path).with_context(|| path.display().to_string())
    };
    let cached = crate::config::cache_dir().and_then(|cache| cache_file(&cache, dir));
    if let Some(path) = &cached {
        if let Ok(t) = open(path) {
            return Ok(t);
        }
    }
    let bytes = make(&read("kana.tsv")?, &read("latin.tsv")?);
    if let Some(path) = &cached {
        publish(path, &bytes);
        if let Ok(t) = open(path) {
            return Ok(t);
        }
    }
    Tables::from_vec(bytes)
}

/// The cache's file of the model in `dir`: named by the format, the
/// directory, and the size and time of its two files, so that another
/// model (or the same updated) has its own.
fn cache_file(cache: &Path, dir: &Path) -> Option<PathBuf> {
    // FNV-1a: stable across builds, which a std hasher is not
    let fnv = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut d: u64 = 0xcbf2_9ce4_8422_2325;
    fnv(&mut d, dir.to_string_lossy().as_bytes());
    let mut h = d;
    let mut add = |bytes: &[u8]| fnv(&mut h, bytes);
    add(MAGIC);
    for name in ["kana.tsv", "latin.tsv"] {
        let m = std::fs::metadata(dir.join(name)).ok()?;
        let at = m
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos();
        add(name.as_bytes());
        add(&m.len().to_le_bytes());
        add(&at.to_le_bytes());
    }
    Some(cache.join(format!("mixed-{:08x}-{h:016x}.bin", d as u32)))
}

fn open(path: &Path) -> Result<Tables> {
    let file = std::fs::File::open(path)?;
    // SAFETY: a file of the user's cache, written whole under another name
    // and moved to this one, never written again (a model of other files
    // has another name); read through bounds-checked slices, as the
    // lexicon is
    let map = unsafe { memmap2::Mmap::map(&file) }?;
    Tables::from_bytes(Bytes::Mapped(map))
}

/// Put `bytes` at `path` for the next process: written under a name of
/// this process's and moved there whole (another process may do the same
/// at once: one of the moves wins); the blocks made before for the same
/// directory go.
fn publish(path: &Path, bytes: &[u8]) {
    let Some(dir) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_err() || std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    // the blocks of the same directory's model before (`mixed-<dir>-`):
    // another model's (a checkout's, a test's) stay
    let Some(prefix) = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.get(..15))
        .map(str::to_string)
    else {
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        // one in use stays (mapped: it cannot be deleted), for its process
        if name.starts_with(&prefix) && name.ends_with(".bin") && e.path() != path {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(grams: &[(&str, u32)]) -> Counts {
        let mut c = Counts::default();
        for (g, n) in grams {
            c.add(g, *n);
        }
        c
    }

    fn words(list: &[(&str, u32, u32)]) -> Vec<Word> {
        list.iter()
            .map(|&(w, ja, en)| Word {
                word: w.to_string(),
                ja,
                en,
            })
            .collect()
    }

    /// Witten–Bell over plain maps, as the model computed it before its
    /// tables were a block.
    fn reference(grams: &[(&str, u32)], a: char, b: char, c: char) -> f64 {
        let mut counts: HashMap<(char, char, char), u32> = HashMap::new();
        for (g, n) in grams {
            let ch: Vec<char> = g.chars().collect();
            let key = match *ch {
                [c] => (NONE, NONE, c),
                [b, c] => (NONE, b, c),
                [a, b, c] => (a, b, c),
                _ => continue,
            };
            *counts.entry(key).or_insert(0) += n;
        }
        let mut contexts: HashMap<(char, char), (u64, u32)> = HashMap::new();
        let (mut total, mut vocab) = (0u64, 0u32);
        for (&(x, y, _), &n) in &counts {
            if y == NONE {
                total += n as u64;
                vocab += 1;
            } else {
                let e = contexts.entry((x, y)).or_insert((0, 0));
                e.0 += n as u64;
                e.1 += 1;
            }
        }
        let count = |k| *counts.get(&k).unwrap_or(&0) as f64;
        let p1 = (count((NONE, NONE, c)) + 1.0) / (total as f64 + vocab as f64 + 1.0);
        let interpolate = |ctx, key, lower: f64| match contexts.get(&ctx) {
            Some(&(t, n)) => (count(key) + n as f64 * lower) / (t as f64 + n as f64),
            None => lower,
        };
        let p2 = interpolate((NONE, b), (NONE, b, c), p1);
        -interpolate((a, b), (a, b, c), p2).ln()
    }

    #[test]
    fn the_blocks_costs_and_words_are_the_maps() {
        let grams = [
            ("あ", 10),
            ("い", 7),
            ("う", 2),
            ("あい", 4),
            ("いあ", 1),
            ("あいう", 3),
            ("@", 5),
            ("あ@", 2),
        ];
        let t = Tables::from_vec(write(
            &counts(&grams),
            &counts(&[("^", 3), ("a", 2)]),
            words(&[("zoo", 1, 4), ("apple", 3, 9), ("app", 2, 1), ("b", 0, 1)]),
        ))
        .unwrap();
        for (a, b, c) in [
            (NONE, NONE, 'あ'),
            (NONE, 'あ', 'い'),
            ('あ', 'い', 'う'),
            ('あ', 'い', 'え'),
            ('か', 'き', 'あ'),
            (NONE, 'あ', '@'),
        ] {
            assert_eq!(
                t.kana_cost(a, b, c),
                reference(&grams, a, b, c),
                "{a}{b}{c}"
            );
        }
        assert_eq!(t.word("apple"), Some((3, 9)));
        assert_eq!(t.word("appl"), None);
        assert_eq!(t.word("zzz"), None);
        assert_eq!(t.prefix_counts("app"), (5, 10));
        assert_eq!(t.prefix_counts("b"), (0, 1));
        assert_eq!(t.prefix_counts("q"), (0, 0));
        assert_eq!(t.n_words(), 4);
        assert_eq!((t.ja_total(), t.en_total()), (6.0, 15.0));
    }

    #[test]
    fn a_damaged_block_is_refused_not_read() {
        let block = write(&counts(&[("あ", 1)]), &counts(&[]), words(&[("a", 1, 1)]));
        assert!(Tables::from_vec(block.clone()).is_ok());
        for cut in [0, 8, HEADER, block.len() / 2, block.len() - 1] {
            assert!(Tables::from_vec(block[..cut].to_vec()).is_err(), "{cut}");
        }
        let mut bad = block.clone();
        bad[0] ^= 1;
        assert!(Tables::from_vec(bad).is_err());
    }

    #[test]
    fn the_cache_keeps_one_block_per_model_and_a_second_load_maps_it() {
        let root =
            std::env::temp_dir().join(format!("setsuna-mixed-tables-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (dir, cache) = (root.join("model"), root.join("cache"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("kana.tsv"), "あ\t3\n").unwrap();
        std::fs::write(dir.join("latin.tsv"), "go\t1\t2\n").unwrap();
        let path = cache_file(&cache, &dir).unwrap();
        let make =
            |_: &str, _: &str| write(&counts(&[("あ", 3)]), &counts(&[]), words(&[("go", 1, 2)]));
        publish(&path, &make("", ""));
        let t = open(&path).unwrap();
        assert!(t.mapped());
        assert_eq!(t.word("go"), Some((1, 2)));
        // the model's files changed: another name, and the old block goes
        // once it is not mapped
        drop(t);
        std::fs::write(dir.join("latin.tsv"), "go\t1\t2\nto\t5\t5\n").unwrap();
        let other = cache_file(&cache, &dir).unwrap();
        assert_ne!(other, path);
        // another directory's model, in the same cache, stays
        let elsewhere = root.join("other");
        std::fs::create_dir_all(&elsewhere).unwrap();
        for name in ["kana.tsv", "latin.tsv"] {
            std::fs::copy(dir.join(name), elsewhere.join(name)).unwrap();
        }
        let theirs = cache_file(&cache, &elsewhere).unwrap();
        publish(&theirs, &make("", ""));
        publish(&other, &make("", ""));
        assert!(!path.exists());
        assert!(other.exists());
        assert!(theirs.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
