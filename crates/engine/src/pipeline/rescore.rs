//! Rescoring of a conversion's candidates with outside statistics, after the
//! pipeline (the research side, 2026-10-23: on Shiran-v2's beam-20 candidates the two
//! terms together gave wiki-fresh EM1 +1.8pt and raw dev +1.2pt; most of it
//! undoes the beam's garbles, such as two words run into one). Each
//! candidate's score gains
//!
//!   pair_weight * Σ min(0, PMI(a, b))     over each two consecutive chars,
//!                                         the context's last char first
//!   word_weight * log P(words | reading)  the candidate aligned with the
//!                                         reading through a word table
//!
//! and the candidates are sorted again (ties keep their order). Only a
//! beam's candidates are rescored: greedy and fallback ones carry no score.
//!
//! PMI(a, b) = ln((c(a, b) + 0.5) N / (c(a ·) c(· b) + 1)) from two-character
//! counts ([`PairTable`]); a char without a count breaks the chain.
//!
//! log P(words | reading) is the best path over (candidate char, reading
//! char) whose steps are ([`WordTable`]):
//!   - a word of the table (at most [`MAX_WORD`] chars) read as one of its
//!     readings, ln((c(r, w) + E) / (c(r) + E));
//!   - else an unknown word, ln(E / (c(r) + E)) for the reading r it covers:
//!     a run of katakana read as itself in hiragana, a run of 1-4 kanji read
//!     as 1-8 reading chars, one hiragana read as itself;
//!   - a run of other chars (digits, Latin, symbols) read as 0-10 reading
//!     chars, or as itself, at no cost.
//!
//! A candidate no path aligns with its reading gets [`FLOOR`]. Both texts
//! are NFKC-normalized first.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use unicode_normalization::UnicodeNormalization;

use crate::candidate::{Candidate, CandidateSource};

/// The word term of a candidate no alignment explains.
pub const FLOOR: f64 = -60.0;
/// Longest table word a step takes.
pub const MAX_WORD: usize = 8;
/// Smoothing of the word table's counts.
const E: f64 = 0.01;
/// Stored two-character terms are in units of 1/PMI_UNIT.
const PMI_UNIT: f64 = 2048.0;
const PAIRS_MAGIC: &[u8; 8] = b"NIPAIRS1";

/// Two-character counts: a sparse matrix over chars, each row's counted
/// columns ascending. The term of a counted pair is kept clipped and
/// quantized (1/2048); a pair never counted is computed from the row and
/// column sums.
pub struct PairTable {
    ids: HashMap<char, u16>,
    starts: Vec<u32>,
    cols: Vec<u16>,
    /// min(0, PMI) of each counted pair, in 1/PMI_UNIT.
    terms: Vec<i16>,
    row_sum: Vec<f64>,
    col_sum: Vec<f64>,
    total: f64,
}

/// Values of a little-endian table read a block at a time, so a load holds
/// no second copy of the file.
fn read_values<R: Read, T>(
    r: &mut R,
    n: usize,
    width: usize,
    mut each: impl FnMut(&[u8]) -> T,
) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; width * 16384];
    let mut left = n;
    while left > 0 {
        let k = left.min(16384);
        r.read_exact(&mut buf[..k * width])?;
        out.extend(buf[..k * width].chunks_exact(width).map(&mut each));
        left -= k;
    }
    Ok(out)
}

impl PairTable {
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("read pair table {}", path.display()))?;
        Self::from_reader(BufReader::new(file))
            .with_context(|| format!("parse pair table {}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::from_reader(Cursor::new(bytes))
    }

    /// `NIPAIRS1`, u32 V, u32 0, u64 nnz, then V u32 chars (u32::MAX: an
    /// entry that is not one char, never matched), V+1 u32 row starts, nnz
    /// u16 columns and nnz u32 counts; little endian. The counts are read
    /// twice (their sums, then the terms) rather than held.
    pub fn from_reader<R: Read + Seek>(mut r: R) -> Result<Self> {
        let mut head = [0u8; 24];
        r.read_exact(&mut head).context("not a pair table")?;
        ensure!(&head[..8] == PAIRS_MAGIC, "not a pair table");
        let v = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let nnz = u64::from_le_bytes(head[16..24].try_into().unwrap()) as usize;
        ensure!(v <= u16::MAX as usize + 1, "{v} chars do not fit u16 ids");
        let want = 24 + 4 * v + 4 * (v + 1) + 2 * nnz + 4 * nnz;
        let len = r.seek(SeekFrom::End(0))? as usize;
        ensure!(
            len == want,
            "{len} bytes, {v} chars and {nnz} pairs need {want}"
        );
        r.seek(SeekFrom::Start(24))?;
        let u32_of = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let chars = read_values(&mut r, v, 4, u32_of)?;
        let ids: HashMap<char, u16> = chars
            .iter()
            .enumerate()
            .filter_map(|(k, &c)| char::from_u32(c).map(|c| (c, k as u16)))
            .collect();
        let starts = read_values(&mut r, v + 1, 4, u32_of)?;
        ensure!(
            starts[0] == 0 && starts[v] as usize == nnz && starts.windows(2).all(|w| w[0] <= w[1]),
            "row starts out of order"
        );
        let cols = read_values(&mut r, nnz, 2, |b| u16::from_le_bytes([b[0], b[1]]))?;
        for a in 0..v {
            let row = &cols[starts[a] as usize..starts[a + 1] as usize];
            ensure!(
                row.windows(2).all(|w| w[0] < w[1]),
                "row {a}'s columns out of order"
            );
            ensure!(
                row.last().is_none_or(|&c| (c as usize) < v),
                "row {a} has a column past {v} chars"
            );
        }
        // the counts' row and column sums, then the terms
        let counts_at = (24 + 4 * v + 4 * (v + 1) + 2 * nnz) as u64;
        let mut row_sum = vec![0.0f64; v];
        let mut col_sum = vec![0.0f64; v];
        let mut a = 0;
        let mut k = 0;
        read_values(&mut r, nnz, 4, |b| {
            while k == starts[a + 1] as usize {
                a += 1;
            }
            let c = u32_of(b) as f64;
            row_sum[a] += c;
            col_sum[cols[k] as usize] += c;
            k += 1;
        })?;
        let total: f64 = row_sum.iter().sum();
        r.seek(SeekFrom::Start(counts_at))?;
        let (mut a, mut k) = (0, 0);
        let terms = read_values(&mut r, nnz, 4, |b| {
            while k == starts[a + 1] as usize {
                a += 1;
            }
            let pmi = ((u32_of(b) as f64 + 0.5) * total
                / (row_sum[a] * col_sum[cols[k] as usize] + 1.0))
                .ln()
                .min(0.0);
            k += 1;
            (pmi * PMI_UNIT).round().max(i16::MIN as f64) as i16
        })?;
        Ok(Self {
            ids,
            starts,
            cols,
            terms,
            row_sum,
            col_sum,
            total,
        })
    }

    fn term(&self, a: u16, b: u16) -> f64 {
        let row = self.starts[a as usize] as usize..self.starts[a as usize + 1] as usize;
        match self.cols[row.clone()].binary_search(&b) {
            Ok(k) => self.terms[row.start + k] as f64 / PMI_UNIT,
            Err(_) => (0.5 * self.total
                / (self.row_sum[a as usize] * self.col_sum[b as usize] + 1.0))
                .ln()
                .min(0.0),
        }
    }

    /// Σ min(0, PMI) over `text`'s consecutive chars, `prev` before it.
    pub fn score(&self, prev: Option<char>, text: &str) -> f64 {
        let mut last = prev.and_then(|c| self.ids.get(&c).copied());
        let mut sum = 0.0;
        for c in text.chars() {
            let id = self.ids.get(&c).copied();
            if let (Some(a), Some(b)) = (last, id) {
                sum += self.term(a, b);
            }
            last = id;
        }
        sum
    }
}

/// Chars with a one-byte code, then chars with a two-byte code.
const ONE_BYTE: usize = 240;
const TWO_BYTE: usize = 15 * 256;

/// A compact code for the word table's strings, made from their own char
/// counts: the 240 most frequent chars one byte (0x00-0xEF), the next 3840
/// two (0xF0-0xFE, then one), any other 0xFF then its UTF-8. A prefix code,
/// so a coded text's byte prefixes are its char prefixes and equal codes are
/// equal texts (Shiran's readings at about one byte a char, against three).
struct Code {
    rank: HashMap<char, u16>,
}

impl Code {
    fn from_counts(counts: HashMap<char, u64>) -> Self {
        let mut by: Vec<(char, u64)> = counts.into_iter().collect();
        by.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let rank = by
            .into_iter()
            .take(ONE_BYTE + TWO_BYTE)
            .enumerate()
            .map(|(k, (c, _))| (c, k as u16))
            .collect();
        Self { rank }
    }

    fn push(&self, c: char, out: &mut Vec<u8>) {
        match self.rank.get(&c).map(|&r| r as usize) {
            Some(r) if r < ONE_BYTE => out.push(r as u8),
            Some(r) => {
                let r = r - ONE_BYTE;
                out.push(0xF0 + (r / 256) as u8);
                out.push((r % 256) as u8);
            }
            None => {
                out.push(0xFF);
                let mut b = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
            }
        }
    }

    fn encode(&self, text: &str, out: &mut Vec<u8>) {
        for c in text.chars() {
            self.push(c, out);
        }
    }

    /// `chars` coded, with the byte offset of each char and the end.
    fn encode_chars(&self, chars: &[char]) -> (Vec<u8>, Vec<usize>) {
        let mut out = Vec::with_capacity(chars.len() * 2);
        let mut starts = Vec::with_capacity(chars.len() + 1);
        for &c in chars {
            starts.push(out.len());
            self.push(c, &mut out);
        }
        starts.push(out.len());
        (out, starts)
    }
}

/// Coded strings sorted bytewise in one buffer.
#[derive(Default)]
struct Strings {
    text: Vec<u8>,
    ends: Vec<u32>,
}

impl Strings {
    fn len(&self) -> usize {
        self.ends.len()
    }

    fn push(&mut self, s: &[u8]) {
        self.text.extend_from_slice(s);
        self.ends.push(self.text.len() as u32);
    }

    fn get(&self, k: usize) -> &[u8] {
        let start = if k == 0 { 0 } else { self.ends[k - 1] as usize };
        &self.text[start..self.ends[k] as usize]
    }

    /// The first index of `range` where `pred` fails (`pred` true then false).
    fn partition(&self, range: Range<usize>, pred: impl Fn(&[u8]) -> bool) -> usize {
        let (mut lo, mut hi) = (range.start, range.end);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.get(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The block of `range` that starts with `prefix`.
    fn prefixed(&self, range: Range<usize>, prefix: &[u8]) -> Range<usize> {
        let lo = self.partition(range.clone(), |x| x < prefix);
        let hi = self.partition(lo..range.end, |x| x.starts_with(prefix));
        lo..hi
    }

    fn find(&self, s: &[u8]) -> Option<usize> {
        let k = self.partition(0..self.len(), |x| x < s);
        (k < self.len() && self.get(k) == s).then_some(k)
    }
}

/// (reading, word, count) counts: each word's readings with
/// ln((c(r, w) + E) / (c(r) + E)), and each reading's unknown-word term
/// ln(E / (c(r) + E)). Words longer than [`MAX_WORD`] count in their
/// readings' totals but are not kept (no step takes them).
pub struct WordTable {
    word_code: Code,
    words: Strings,
    /// The entries of word k: first[k]..first[k + 1].
    first: Vec<u32>,
    entry_reading: Vec<u32>,
    entry_term: Vec<f32>,
    reading_code: Code,
    readings: Strings,
    unknown: Vec<f32>,
}

/// A request's reading, NFKC, coded as the table's readings, with each of
/// its pieces' unknown-word terms.
pub struct Reading {
    chars: Vec<char>,
    coded: Vec<u8>,
    /// Coded offset of each char, and the end.
    starts: Vec<usize>,
    /// The char index at each coded offset (on a char boundary).
    char_at: Vec<u32>,
    /// unknown(chars[j..j + M]) at j * MAX_WORD + M - 1, M = 1..=MAX_WORD.
    short_unknown: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Hiragana,
    Katakana,
    Kanji,
    Other,
}

fn kind(c: char) -> Kind {
    match c {
        '\u{3041}'..='\u{309f}' => Kind::Hiragana,
        '\u{30a1}'..='\u{30ff}' => Kind::Katakana,
        '\u{3400}'..='\u{9fff}' | '々' | '〆' => Kind::Kanji,
        _ => Kind::Other,
    }
}

/// Katakana to hiragana as the research side reads it (jaconv): ァ-ヶ, ヽ, ヾ.
fn kata_to_hira(c: char) -> char {
    match c {
        '\u{30a1}'..='\u{30f6}' | 'ヽ' | 'ヾ' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        _ => c,
    }
}

/// One line of a word table: (reading, word, count).
fn row(line: &str, n: usize) -> Result<(&str, &str, u64)> {
    let mut f = line.split('\t');
    let (Some(r), Some(w), Some(c), None) = (f.next(), f.next(), f.next(), f.next()) else {
        bail!("line {n}: not reading \\t word \\t count");
    };
    let c = c
        .parse()
        .with_context(|| format!("line {n}: count {c:?}"))?;
    Ok((r, w, c))
}

/// A table row coded: its reading and word (None past MAX_WORD chars) in a
/// shared buffer, and its count.
struct Row {
    reading: (u32, u32),
    word: Option<(u32, u32)>,
    count: u64,
}

impl WordTable {
    pub fn load(path: &Path) -> Result<Self> {
        let lines = || -> Result<_> {
            let file =
                File::open(path).with_context(|| format!("read word table {}", path.display()))?;
            Ok(BufReader::new(file)
                .lines()
                .map(|l| l.map_err(anyhow::Error::from)))
        };
        Self::from_lines(lines).with_context(|| format!("parse word table {}", path.display()))
    }

    /// Lines `reading \t word \t count`; a pair given twice is summed.
    pub fn parse(text: &str) -> Result<Self> {
        Self::from_lines(|| Ok(text.lines().map(|l| Ok(l.to_string()))))
    }

    /// The table from its lines, read twice (the chars' counts for the codes,
    /// then the rows coded) rather than held as text.
    fn from_lines<I: Iterator<Item = Result<String>>>(
        lines: impl Fn() -> Result<I>,
    ) -> Result<Self> {
        let mut reading_chars: HashMap<char, u64> = HashMap::new();
        let mut word_chars: HashMap<char, u64> = HashMap::new();
        for (n, line) in lines()?.enumerate() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let (r, w, _) = row(&line, n + 1)?;
            for c in r.chars() {
                *reading_chars.entry(c).or_default() += 1;
            }
            if w.chars().count() <= MAX_WORD {
                for c in w.chars() {
                    *word_chars.entry(c).or_default() += 1;
                }
            }
        }
        let reading_code = Code::from_counts(reading_chars);
        let word_code = Code::from_counts(word_chars);
        let mut buf: Vec<u8> = Vec::new();
        let mut rows: Vec<Row> = Vec::new();
        for (n, line) in lines()?.enumerate() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let (r, w, count) = row(&line, n + 1)?;
            let start = buf.len() as u32;
            reading_code.encode(r, &mut buf);
            let reading = (start, buf.len() as u32);
            let word = (w.chars().count() <= MAX_WORD).then(|| {
                let start = buf.len() as u32;
                word_code.encode(w, &mut buf);
                (start, buf.len() as u32)
            });
            rows.push(Row {
                reading,
                word,
                count,
            });
        }
        let at = |(a, b): (u32, u32)| &buf[a as usize..b as usize];
        // each reading's total
        let mut order: Vec<u32> = (0..rows.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            at(rows[a as usize].reading).cmp(at(rows[b as usize].reading))
        });
        let mut readings = Strings::default();
        let mut totals: Vec<u64> = Vec::new();
        let mut reading_of = vec![0u32; rows.len()];
        for &k in &order {
            let row = &rows[k as usize];
            let r = at(row.reading);
            if readings.len() == 0 || readings.get(readings.len() - 1) != r {
                readings.push(r);
                totals.push(0);
            }
            *totals.last_mut().unwrap() += row.count;
            reading_of[k as usize] = readings.len() as u32 - 1;
        }
        let unknown: Vec<f32> = totals
            .iter()
            .map(|&c| (E / (c as f64 + E)).ln() as f32)
            .collect();
        // each word's readings
        order.retain(|&k| rows[k as usize].word.is_some());
        let key = |k: u32| (at(rows[k as usize].word.unwrap()), reading_of[k as usize]);
        order.sort_unstable_by(|&a, &b| key(a).cmp(&key(b)));
        let mut words = Strings::default();
        let mut first: Vec<u32> = Vec::new();
        let mut entry_reading: Vec<u32> = Vec::with_capacity(order.len());
        let mut entry_count: Vec<u64> = Vec::with_capacity(order.len());
        let mut last: Option<(&[u8], u32)> = None;
        for &k in &order {
            let (w, r) = key(k);
            if last == Some((w, r)) {
                *entry_count.last_mut().unwrap() += rows[k as usize].count;
                continue;
            }
            if last.is_none_or(|(lw, _)| lw != w) {
                words.push(w);
                first.push(entry_reading.len() as u32);
            }
            entry_reading.push(r);
            entry_count.push(rows[k as usize].count);
            last = Some((w, r));
        }
        first.push(entry_reading.len() as u32);
        let entry_term = entry_reading
            .iter()
            .zip(&entry_count)
            .map(|(&r, &c)| ((c as f64 + E) / (totals[r as usize] as f64 + E)).ln() as f32)
            .collect();
        for s in [&mut words, &mut readings] {
            s.text.shrink_to_fit();
            s.ends.shrink_to_fit();
        }
        Ok(Self {
            word_code,
            words,
            first,
            entry_reading,
            entry_term,
            reading_code,
            readings,
            unknown,
        })
    }

    /// `reading` prepared for [`Self::prior`].
    pub fn reading(&self, reading: &str) -> Reading {
        let chars: Vec<char> = reading.nfkc().collect();
        let (coded, starts) = self.reading_code.encode_chars(&chars);
        let mut char_at = vec![u32::MAX; coded.len() + 1];
        for (k, &b) in starts.iter().enumerate() {
            char_at[b] = k as u32;
        }
        let m = chars.len();
        let mut short_unknown = vec![0.0; m * MAX_WORD];
        for j in 0..m {
            let mut range = 0..self.readings.len();
            for len in 1..=MAX_WORD.min(m - j) {
                let piece = &coded[starts[j]..starts[j + len]];
                range = self.readings.prefixed(range, piece);
                if range.is_empty() {
                    break;
                }
                if self.readings.get(range.start) == piece {
                    short_unknown[j * MAX_WORD + len - 1] = self.unknown[range.start] as f64;
                }
            }
        }
        Reading {
            chars,
            coded,
            starts,
            char_at,
            short_unknown,
        }
    }

    /// The unknown-word term of the reading's chars j..j + len.
    fn unknown_at(&self, rd: &Reading, j: usize, len: usize) -> f64 {
        if len <= MAX_WORD {
            rd.short_unknown[j * MAX_WORD + len - 1]
        } else {
            self.readings
                .find(&rd.coded[rd.starts[j]..rd.starts[j + len]])
                .map_or(0.0, |k| self.unknown[k] as f64)
        }
    }

    /// The best path's log P(words | reading) of `text` (see the module), or
    /// [`FLOOR`].
    pub fn prior(&self, text: &str, rd: &Reading) -> f64 {
        let sc: Vec<char> = text.nfkc().collect();
        let (s, sb) = self.word_code.encode_chars(&sc);
        let (n, m) = (sc.len(), rd.chars.len());
        let w = m + 1;
        let mut best = vec![f64::NEG_INFINITY; (n + 1) * w];
        best[0] = 0.0;
        let step = |best: &mut [f64], i: usize, j: usize, v: f64| {
            let cell = &mut best[i * w + j];
            if *cell < v {
                *cell = v;
            }
        };
        let mut words: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            // the table's words at i: (end, word)
            words.clear();
            let mut range = 0..self.words.len();
            for len in 1..=MAX_WORD.min(n - i) {
                let piece = &s[sb[i]..sb[i + len]];
                range = self.words.prefixed(range, piece);
                if range.is_empty() {
                    break;
                }
                if self.words.get(range.start) == piece {
                    words.push((i + len, range.start));
                }
            }
            let k = kind(sc[i]);
            let run_end = |cap: usize| {
                let mut e = i;
                while e < n && e - i < cap && kind(sc[e]) == k {
                    e += 1;
                }
                e
            };
            let end = match k {
                Kind::Kanji => run_end(4),
                Kind::Katakana | Kind::Other => run_end(usize::MAX),
                Kind::Hiragana => i + 1,
            };
            for j in 0..=m {
                let here = best[i * w + j];
                if here == f64::NEG_INFINITY {
                    continue;
                }
                let rest = &rd.coded[rd.starts[j]..];
                for &(e, word) in &words {
                    for x in self.first[word] as usize..self.first[word + 1] as usize {
                        let r = self.readings.get(self.entry_reading[x] as usize);
                        if rest.starts_with(r) {
                            let j2 = rd.char_at[rd.starts[j] + r.len()] as usize;
                            step(&mut best, e, j2, here + self.entry_term[x] as f64);
                        }
                    }
                }
                match k {
                    Kind::Katakana => {
                        for t in 0..end - i {
                            if j + t >= m || rd.chars[j + t] != kata_to_hira(sc[i + t]) {
                                break;
                            }
                            let v = here + self.unknown_at(rd, j, t + 1);
                            step(&mut best, i + t + 1, j + t + 1, v);
                        }
                    }
                    Kind::Kanji => {
                        for e in i + 1..=end {
                            for len in 1..=MAX_WORD.min(m - j) {
                                let v = here + self.unknown_at(rd, j, len);
                                step(&mut best, e, j + len, v);
                            }
                        }
                    }
                    Kind::Hiragana => {
                        if j < m && rd.chars[j] == sc[i] {
                            let v = here + self.unknown_at(rd, j, 1);
                            step(&mut best, i + 1, j + 1, v);
                        }
                    }
                    Kind::Other => {
                        for len in 0..=10.min(m - j) {
                            step(&mut best, end, j + len, here);
                        }
                        // typed as written (Latin words, digits)
                        if rd.chars[j..].starts_with(&sc[i..end]) {
                            step(&mut best, end, j + (end - i), here);
                        }
                    }
                }
            }
        }
        let last = best[n * w + m];
        if last == f64::NEG_INFINITY {
            FLOOR
        } else {
            last
        }
    }
}

/// The rescoring a session applies: each table with its weight.
pub struct Rescorer {
    pairs: Option<(PairTable, f64)>,
    words: Option<(WordTable, f64)>,
}

impl Rescorer {
    /// The tables whose weight is not 0; `None` when neither is. A weight
    /// without its table is an error.
    pub fn load(
        pairs: Option<&Path>,
        pair_weight: f32,
        words: Option<&Path>,
        word_weight: f32,
    ) -> Result<Option<Self>> {
        fn table<'a>(path: Option<&'a Path>, weight: f32, what: &str) -> Result<Option<&'a Path>> {
            if weight == 0.0 {
                return Ok(None);
            }
            match path {
                Some(p) => Ok(Some(p)),
                None => {
                    bail!("{what}_weight is {weight} but the manifest has no rescore.{what}s table")
                }
            }
        }
        let pairs = table(pairs, pair_weight, "pair")?
            .map(|p| PairTable::load(p).map(|t| (t, pair_weight as f64)))
            .transpose()?;
        let words = table(words, word_weight, "word")?
            .map(|p| WordTable::load(p).map(|t| (t, word_weight as f64)))
            .transpose()?;
        Ok((pairs.is_some() || words.is_some()).then_some(Self { pairs, words }))
    }

    pub fn from_tables(pairs: Option<(PairTable, f64)>, words: Option<(WordTable, f64)>) -> Self {
        Self { pairs, words }
    }

    /// Add the weighted terms to each candidate's score and sort again, best
    /// first; candidates not all from the beam are left as they are.
    pub fn rescore(&self, context: &str, reading: &str, candidates: &mut [Candidate]) {
        if candidates.iter().any(|c| c.source != CandidateSource::Beam) {
            return;
        }
        let prev = context.chars().last();
        let rd = self.words.as_ref().map(|(t, _)| t.reading(reading));
        for c in candidates.iter_mut() {
            let mut add = 0.0;
            if let Some((t, w)) = &self.pairs {
                add += w * t.score(prev, &c.text);
            }
            if let (Some((t, w)), Some(rd)) = (&self.words, &rd) {
                add += w * t.prior(&c.text, rd);
            }
            c.score += add as f32;
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pair table of `chars` with the counts `pairs` ((a, b, count)).
    fn pair_bytes(chars: &[u32], pairs: &[(u16, u16, u32)]) -> Vec<u8> {
        let v = chars.len();
        let mut sorted = pairs.to_vec();
        sorted.sort();
        let mut out = Vec::new();
        out.extend_from_slice(PAIRS_MAGIC);
        out.extend_from_slice(&(v as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
        for &c in chars {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for a in 0..=v {
            let start = sorted.iter().filter(|p| (p.0 as usize) < a).count() as u32;
            out.extend_from_slice(&start.to_le_bytes());
        }
        for p in &sorted {
            out.extend_from_slice(&p.1.to_le_bytes());
        }
        for p in &sorted {
            out.extend_from_slice(&p.2.to_le_bytes());
        }
        out
    }

    #[test]
    fn pair_terms_follow_the_counts() {
        let chars = ['あ' as u32, 'い' as u32, 'う' as u32, u32::MAX];
        let t = PairTable::parse(&pair_bytes(
            &chars,
            &[(0, 1, 90), (1, 2, 5), (2, 0, 5), (3, 0, 100)],
        ))
        .unwrap();
        let (row, col, n) = ([90.0, 5.0, 5.0, 100.0], [105.0, 90.0, 5.0, 0.0], 200.0);
        let pmi =
            |c: f64, a: usize, b: usize| ((c + 0.5) * n / (row[a] * col[b] + 1.0f64)).ln().min(0.0);
        // a counted pair (quantized), then one never counted
        assert!((t.score(None, "あい") - pmi(90.0, 0, 1)).abs() < 1e-3);
        assert!((t.score(None, "いあ") - pmi(0.0, 1, 0)).abs() < 1e-12);
        // the context's last char comes first; an unknown char breaks the chain
        let two = t.score(Some('い'), "うあ");
        assert!((two - pmi(5.0, 1, 2) - pmi(5.0, 2, 0)).abs() < 1e-3);
        assert_eq!(t.score(Some('x'), "あxい"), 0.0);
        // a non-char entry is never matched but its counts are in the sums
        assert!(t.score(None, "う").abs() < 1e-12);
    }

    #[test]
    fn bad_pair_tables_are_refused() {
        let chars = ['あ' as u32, 'い' as u32];
        let good = pair_bytes(&chars, &[(0, 1, 3)]);
        assert!(PairTable::parse(&good[..good.len() - 1]).is_err());
        assert!(PairTable::parse(b"NIPAIRS0").is_err());
    }

    fn table() -> WordTable {
        WordTable::parse(
            "きょう\t今日\t30\nきょう\t京\t10\nきょう\t教\t1\nは\tは\t50\n\
             いい\t良い\t5\nてんき\t天気\t20\nてんき\t転機\t2\nいい\tいい\t20\n",
        )
        .unwrap()
    }

    fn lp(c: f64, total: f64) -> f64 {
        ((c + E) / (total + E)).ln()
    }

    /// Every piece of the reading あいは listed, so no unknown word is free.
    fn listed() -> WordTable {
        WordTable::parse(
            "あい	愛	30
あい	藍	10
あ	亜	50
い	胃	50
は	は	50
             いは	言葉	1
あいは	愛葉	1
",
        )
        .unwrap()
    }

    #[test]
    fn a_candidate_aligned_with_the_table() {
        let t = listed();
        let rd = t.reading("あいは");
        let got = t.prior("愛は", &rd);
        assert!((got - lp(30.0, 40.0) - lp(50.0, 50.0)).abs() < 1e-5);
        // the rarer word of the same reading scores lower; two words read alike
        assert!((t.prior("藍は", &rd) - lp(10.0, 40.0)).abs() < 1e-5);
        assert!(t.prior("亜胃は", &rd).abs() < 1e-5);
        // an unknown kanji word pays for the reading it covers
        assert!((t.prior("哀は", &rd) - (E / (40.0 + E)).ln()).abs() < 1e-5);
    }

    #[test]
    fn unknown_words_and_runs() {
        let t = table();
        // an unknown kanji word over a listed reading pays for that reading
        let rd = t.reading("きょう");
        assert!((t.prior("饗", &rd) - (E / (41.0 + E)).ln()).abs() < 1e-5);
        // over an unlisted reading it is free; katakana is read as itself
        let rd = t.reading("ぽてと");
        assert_eq!(t.prior("ポテト", &rd), 0.0);
        assert_eq!(t.prior("芋", &rd), 0.0);
        // a candidate no path reads gets the floor
        assert_eq!(t.prior("ポテ", &rd), FLOOR);
        assert_eq!(t.prior("ぽてとと", &rd), FLOOR);
        // digits and Latin as typed or read as up to 10 chars
        let rd = t.reading("abcは");
        assert!((t.prior("abcは", &rd) - lp(50.0, 50.0)).abs() < 1e-5);
        let rd = t.reading("にじゅうは");
        assert!((t.prior("20は", &rd) - lp(50.0, 50.0)).abs() < 1e-5);
        // full-width forms are read after NFKC
        let rd = t.reading("ａｂｃは");
        assert!((t.prior("abcは", &rd) - lp(50.0, 50.0)).abs() < 1e-5);
    }

    #[test]
    fn the_code_keeps_prefixes_at_every_width() {
        // 300 chars by falling counts: one byte, two bytes, and one left out
        let chars: Vec<char> = (0..300)
            .map(|k| char::from_u32(0x4e00 + k).unwrap())
            .collect();
        let counts = chars
            .iter()
            .enumerate()
            .map(|(k, &c)| (c, 1000 - k as u64))
            .collect();
        let mut code = Code::from_counts(counts);
        code.rank.remove(&chars[299]);
        let (one, two, out) = (chars[0], chars[250], chars[299]);
        let enc = |s: &[char]| code.encode_chars(s).0;
        assert_eq!(enc(&[one]).len(), 1);
        assert_eq!(enc(&[two]).len(), 2);
        assert_eq!(enc(&[out]).len(), 4);
        let texts = [
            vec![one, two, out],
            vec![one, two],
            vec![two, out],
            vec![out, one],
        ];
        for a in &texts {
            for b in &texts {
                assert_eq!(enc(a).starts_with(&enc(b)), a.starts_with(b), "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn a_word_past_the_longest_step_counts_only_in_its_reading_total() {
        let t = WordTable::parse("あい\t愛\t30\nあい\t愛愛愛愛愛愛愛愛愛\t10\n").unwrap();
        assert_eq!(t.words.len(), 1);
        let rd = t.reading("あい");
        assert!((t.prior("愛", &rd) - lp(30.0, 40.0)).abs() < 1e-5);
    }

    #[test]
    fn rescoring_reorders_beam_candidates_only() {
        let r = Rescorer::from_tables(None, Some((listed(), 1.0)));
        let cand = |text: &str, score: f32, source| Candidate {
            text: text.to_string(),
            score,
            source,
            reading_pos: Vec::new(),
        };
        let mut c = vec![
            cand("藍は", -1.0, CandidateSource::Beam),
            cand("愛は", -1.5, CandidateSource::Beam),
        ];
        r.rescore("", "あいは", &mut c);
        assert_eq!(c[0].text, "愛は");
        assert!((c[0].score as f64 - (-1.5 + lp(30.0, 40.0))).abs() < 1e-5);
        let mut f = vec![
            cand("きょうはいいてんき", 0.0, CandidateSource::Fallback),
            cand("キョウハイイテンキ", 0.0, CandidateSource::Fallback),
        ];
        r.rescore("", "きょうはいいてんき", &mut f);
        assert_eq!(f[0].score, 0.0);
    }

    /// The research script's terms on real candidates (its parity.tsv:
    /// prev, reading, candidate, PMI2, prior): SETSUNA_RESCORE_PAIRS,
    /// SETSUNA_RESCORE_WORDS and SETSUNA_RESCORE_CHECK name the files.
    #[test]
    fn matches_the_research_terms() {
        let (Ok(pairs), Ok(words), Ok(check)) = (
            std::env::var("SETSUNA_RESCORE_PAIRS"),
            std::env::var("SETSUNA_RESCORE_WORDS"),
            std::env::var("SETSUNA_RESCORE_CHECK"),
        ) else {
            return;
        };
        let p = PairTable::load(Path::new(&pairs)).unwrap();
        let w = WordTable::load(Path::new(&words)).unwrap();
        let text = std::fs::read_to_string(check).unwrap();
        let (mut n, mut worst_p, mut worst_w) = (0, 0.0f64, 0.0f64);
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            let prev = f[0].chars().next();
            let rd = w.reading(f[1]);
            let (want_p, want_w): (f64, f64) = (f[3].parse().unwrap(), f[4].parse().unwrap());
            worst_p = worst_p.max((p.score(prev, f[2]) - want_p).abs());
            let got_w = w.prior(f[2], &rd);
            worst_w = worst_w.max((got_w - want_w).abs());
            assert!((got_w - want_w).abs() < 1e-3, "{line}: word term {got_w}");
            n += 1;
        }
        eprintln!(
            "{n} candidates: worst |PMI2 diff| {worst_p:.2e}, worst |prior diff| {worst_w:.2e}"
        );
        assert!(worst_p < 1e-2);
    }
}
