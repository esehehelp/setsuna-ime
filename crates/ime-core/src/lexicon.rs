//! A reading lexicon: (surface, reading) pairs with a cost each, to tell the
//! model's near-misses from conversions text uses. The model emits a
//! candidate char by char, so its list holds surfaces no dictionary has
//! (続頭 for ぞくとう, き今日 for きょう); `cost` reads a candidate against
//! its reading as a lattice of the lexicon's words, the kana, digits and
//! symbols spelled as typed, and other chars at a high cost, and gives the
//! cheapest path.
//!
//! The source is a text table,
//! `surface<TAB>reading<TAB>cost<TAB>class<TAB>segments<TAB>lid<TAB>rid`
//! (cost in nats, the reading in hiragana; segments ignored; a pair may have
//! a line for each of its forms), where lid and rid are the context ids of
//! the word's left and right ends, and a connection table, the cost in nats
//! of each (right id, the next word's left id) pair: the number of ids and
//! then a cost a line. Mozc's ids and connection costs, which the table's
//! source maps onto its own scale, are such: a word's cost then depends on
//! the word before it (書い before た, not 書き). A table's cost is -ln of
//! how often text uses the word, which already holds how often its part of
//! speech comes, so `build` keeps of a connection cost how much dearer the
//! word is after that right id than after the right ids text has (each as
//! often as the table's costs say): cost(r, l) + ln sum_r' p(r')
//! exp(-cost(r', l)), below 0 for a pair likelier than so. `build` compiles
//! them into the file the IME maps (`LEXICON1`), little-endian: a 64-byte
//! header (magic, version u32 = 3, keys u32, entries u32, key_bytes u32,
//! reading_bytes u32, ids u32) and then, each padded to 8 bytes: key_off
//! u32[keys + 1] and key_blob u8[key_bytes] (the surfaces, UTF-8, ascending
//! by bytes), key_first u32[keys + 1] (each surface's entries), entry
//! u64[entries] (reading offset 32 bits, reading length 8, class 8, cost in
//! centinats 16), reading_blob u8[reading_bytes], by_reading u32[entries]
//! (the entries by reading, ascending by bytes, then by cost), context
//! u32[entries] (lid 16 bits, rid 16) and connection i16[ids * ids] (in
//! centinats, by right id then left id). Version 2 is the same without the
//! last two. The file is mapped, not read, so the processes the IME runs in
//! share one copy.
//!
//! `convert` goes the other way, as a dictionary IME does: the cheapest
//! sequences of the lexicon's words (and kana as typed) that read as a
//! reading, for the conversions the model does not find (大島村 for
//! おおしまむら, where it gave おおしま村).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, ensure, Context, Result};

/// What a candidate's score loses per nat of its lexicon cost, picked on the
/// distributed model's wiki_fresh_v2 conversions (Shiran-v2 s14000, its top
/// 20): EM1 74.73 -> 76.81 by the engine's score, 79.27 -> 79.95 with the
/// reranker at lambda 0.8; AJIMEE 72.0 -> 75.5 and 81.5 -> 83.0.
pub const WEIGHT: f32 = 0.4;
/// A candidate whose cost is this far above the cheapest of its list is
/// dropped, the engine's top kept: 59% / 54% of the candidates of
/// wiki_fresh_v2 / AJIMEE go, and none of their golds.
pub const MARGIN: f32 = 20.0;
/// The lexicon's own conversions (`convert`) tried for the engine's list:
/// a conversion no more than `ADDED_OVER` dearer than the engine's cheapest
/// candidate is added, scored `ADDED_BELOW` under its top, for the lexicon's
/// cost and the reranker to place. Picked on wiki_fresh_v2 (Shiran-v2 s14000, the
/// reranker at lambda 0.8 over the top 20) with the connection costs: golds
/// in the list 4339 -> 4494, EM1 79.95 -> 81.57, by the engine's score and
/// the lexicon 76.81 -> 76.95; AJIMEE 183 -> 186, 83.0 -> 84.5, 75.5 ->
/// 74.5. Adding every conversion found more golds (4563, EM1 up to 82.29)
/// and left more of the lexicon's own misses in the list (about 8
/// candidates after the rerank against 4.9), so only the cheaper ones went
/// in at first (`ADDED_OVER` 0). That left out a word's homophones where it
/// costs more than the word (検収担当者 24.0 for けんしゅうたんとうしゃ, the
/// model giving 研修担当者 18.7), which the user then could not reach (user
/// 2026-10-28). Up to 8 dearer, over wiki_fresh_v2 with Shiran-v2 s7080 (the
/// reranker over the top 20): golds in the list 4528 -> 4592, EM1 82.09 ->
/// 82.17, 4.7 -> 6.6 candidates after the rerank (at 4: 4580, 82.19, 5.8);
/// AJIMEE with Bletilla-v0.1-small 189 -> 192, 86.5 -> 87.0, 5.8 -> 8.2 (at
/// 4 the same, 7.3); by the engine's score and the lexicon EM1 is the same.
pub const CONVERSIONS: usize = 10;
pub const ADDED_OVER: f32 = 8.0;
pub const ADDED_BELOW: f32 = 8.0;
/// How much of the connection table's cost a path pays where one word
/// follows another: 0.5, 1 and 1.5 within 0.1 EM1 of each other on
/// wiki_fresh_v2, where the lexicon's top 10 conversions hold 225 of the 657
/// golds the engine's top 20 miss (192 without the connection costs).
pub const CONNECTION_WEIGHT: f32 = 1.0;
/// A prefix that ends the reading, which Mozc's connection costs do not
/// weigh (its end of sentence follows anything at 0): 神に各 for かみにかく
/// cost less than 紙に書く.
pub const PREFIX_LAST: f32 = 6.0;
/// The lexicon's words `word_candidates` lists for a reading of one word,
/// the chars of particle or okurigana it takes after the word, and how much
/// dearer than the cheapest a word may be (12: of ない's, 凪い in, 地震 out).
pub const WORDS: usize = 30;
pub const WORD_REST: usize = 3;
pub const WORD_MARGIN: f32 = 12.0;
/// Where the IME puts the words the model left out: after its first three
/// and the reading's kana forms (`kana_candidates`). Over wiki_fresh_v2's
/// golds cut into bunsetsu (21,842, each converted alone after the text
/// before it, as segment editing asks; Bletilla-v0.1-small), the gold is in
/// the segment's list 98.34% -> 98.90% and on its first page (9) 98.10%
/// -> 98.31% (98.10% with the words at the end of the list); AJIMEE's 999:
/// 98.40% -> 99.30%, first page 98.30% -> 99.00% (98.30%).
pub const WORDS_AT: usize = 5;
/// A reading of up to `CHAR_READING` kana lists up to `CHAR_WORDS` of the
/// lexicon's one-char words for it (`char_words`), however dear, after the
/// first page of the list (`CHAR_WORDS_AT`, the candidate window's 9): あ
/// listed 亜 亞 阿 有 在 合 and its kana, where a dictionary IME lists some
/// forty kanji (user 2026-10-30).
pub const CHAR_READING: usize = 2;
pub const CHAR_WORDS: usize = 50;
pub const CHAR_WORDS_AT: usize = 9;

const MAGIC: &[u8; 8] = b"LEXICON1";
const VERSION: usize = 3;
const HEADER: usize = 64;
/// The context id of what has none: a kana, digit or symbol spelled as
/// typed, a word of a table without ids. It follows and precedes anything
/// at no cost.
const NO_ID: u16 = u16::MAX;
/// The context id of the start of the reading (Mozc's BOS/EOS).
const BOS: u16 = 0;
/// The longest surface (and reading) looked up, in chars.
const MAX_KEY_CHARS: usize = 24;
/// The surfaces of one reading `convert` tries, the cheapest.
const PER_READING: usize = 8;
/// The surfaces `convert` keeps at each reading position, and the right ids
/// (forms of the word it ends with) each.
const BEAM: usize = 8;
const FORMS: usize = 4;

/// A kana spelled as typed with no word of the lexicon over it.
pub const KANA_COST: f32 = 9.0;
/// A digit, Latin letter or symbol spelled as typed.
pub const TYPED_COST: f32 = 1.0;
/// A char that neither the lexicon nor the reading spells (a kanji of no
/// word, a Latin letter the reading does not have), per char; it may take
/// 1 to 4 reading chars.
pub const UNKNOWN_COST: f32 = 30.0;

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

/// One reading of a surface, in one of its forms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry<'a> {
    pub reading: &'a str,
    pub cost: f32,
    pub class: u8,
    /// The context ids of its left and right ends (`NO_ID` for none).
    pub lid: u16,
    pub rid: u16,
}

impl Entry<'_> {
    /// What the word costs more when it ends the reading (`last`): a
    /// prefix, `PREFIX_LAST`.
    fn ending(&self, last: bool) -> f32 {
        if last && usize::from(self.class) == PREFIX {
            PREFIX_LAST
        } else {
            0.0
        }
    }
}

pub struct Lexicon {
    bytes: Bytes,
    keys: usize,
    entries: usize,
    /// The context ids the connection table has (0: none).
    ids: usize,
    connection_weight: f32,
    /// Byte offsets of key_off, key_blob, key_first, entry, reading_blob,
    /// by_reading, context, connection.
    at: [usize; 8],
}

/// The coarse parts of speech of the table's class column, by index.
pub const CLASSES: [&str; 15] = [
    "other", "noun", "propn", "verb", "adj", "adjn", "adv", "adnom", "conj", "intj", "part", "aux",
    "pref", "suff", "sym",
];
/// The class of a prefix.
const PREFIX: usize = 12;

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

impl Lexicon {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the lexicon is a read-only data file the IME installs; it
        // is not written while mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("map {}", path.display()))?;
        Self::from_bytes(Bytes::Mapped(map))
    }

    fn from_bytes(bytes: Bytes) -> Result<Self> {
        ensure!(
            bytes.len() >= HEADER && &bytes[..8] == MAGIC,
            "not a lexicon"
        );
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let version = u32_at(8);
        ensure!(
            version == 2 || version == VERSION,
            "lexicon version {version} (this IME reads 2 and {VERSION})"
        );
        let (keys, entries, key_bytes, reading_bytes) =
            (u32_at(12), u32_at(16), u32_at(20), u32_at(24));
        let ids = if version == 2 { 0 } else { u32_at(28) };
        let mut sizes = vec![
            4 * (keys + 1),
            key_bytes,
            4 * (keys + 1),
            8 * entries,
            reading_bytes,
            4 * entries,
        ];
        if version == VERSION {
            sizes.extend([4 * entries, 2 * ids * ids]);
        }
        let mut at = [0usize; 8];
        let mut o = HEADER;
        for (k, s) in sizes.iter().enumerate() {
            at[k] = o;
            o += pad8(*s);
        }
        ensure!(bytes.len() >= o, "lexicon truncated");
        Ok(Self {
            bytes,
            keys,
            entries,
            ids,
            connection_weight: CONNECTION_WEIGHT,
            at,
        })
    }

    /// Weigh the connection table's costs by `w` instead of
    /// `CONNECTION_WEIGHT` (for evaluations).
    pub fn set_connection_weight(&mut self, w: f32) {
        self.connection_weight = w;
    }

    /// The cost of a word with left id `lid` after one with right id `rid`.
    fn connection(&self, rid: u16, lid: u16) -> f32 {
        let (r, l) = (rid as usize, lid as usize);
        if r >= self.ids || l >= self.ids {
            return 0.0;
        }
        let o = self.at[7] + 2 * (r * self.ids + l);
        let c = i16::from_le_bytes(self.bytes[o..o + 2].try_into().unwrap());
        f32::from(c) / 100.0 * self.connection_weight
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    fn u32_in(&self, sec: usize, i: usize) -> usize {
        let o = self.at[sec] + 4 * i;
        u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap()) as usize
    }

    fn key(&self, i: usize) -> &[u8] {
        let (a, b) = (self.u32_in(0, i), self.u32_in(0, i + 1));
        &self.bytes[self.at[1] + a..self.at[1] + b]
    }

    fn entry(&self, i: usize) -> Entry<'_> {
        let o = self.at[3] + 8 * i;
        let v = u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap());
        let (off, len) = ((v >> 32) as usize, ((v >> 24) & 0xFF) as usize);
        let start = self.at[4] + off;
        let (lid, rid) = if self.ids == 0 {
            (NO_ID, NO_ID)
        } else {
            let o = self.at[6] + 4 * i;
            let ids = u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap());
            ((ids >> 16) as u16, ids as u16)
        };
        Entry {
            // the blob holds what `build` wrote from a &str
            reading: std::str::from_utf8(&self.bytes[start..start + len]).unwrap_or(""),
            class: ((v >> 16) & 0xFF) as u8,
            cost: (v & 0xFFFF) as f32 / 100.0,
            lid,
            rid,
        }
    }

    fn reading_of(&self, i: usize) -> &[u8] {
        let o = self.at[3] + 8 * i;
        let v = u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap());
        let (off, len) = ((v >> 32) as usize, ((v >> 24) & 0xFF) as usize);
        &self.bytes[self.at[4] + off..self.at[4] + off + len]
    }

    /// The surface entry `i` is of.
    fn surface_of(&self, i: usize) -> &str {
        // the last key whose first entry is not after `i`
        let (mut lo, mut hi) = (0, self.keys);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.u32_in(2, mid) <= i {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        std::str::from_utf8(self.key(lo)).unwrap_or("")
    }

    /// The first key in `lo..hi` not below `probe`.
    fn lower_bound(&self, mut lo: usize, mut hi: usize, probe: &[u8]) -> usize {
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key(mid) < probe {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Every surface of the lexicon that `text` starts with, as its length in
    /// bytes and its entries, shortest first.
    pub fn prefixes<'a>(&'a self, text: &str, mut f: impl FnMut(usize, Entry<'a>)) {
        let (mut lo, mut hi) = (0, self.keys);
        for (n, (i, c)) in text.char_indices().enumerate() {
            if n == MAX_KEY_CHARS || lo == hi {
                break;
            }
            let prefix = &text.as_bytes()[..i + c.len_utf8()];
            lo = self.lower_bound(lo, hi, prefix);
            // the keys that start with `prefix` end before its last byte + 1
            let mut upper = prefix.to_vec();
            *upper.last_mut().unwrap() += 1;
            hi = self.lower_bound(lo, hi, &upper);
            if lo < hi && self.key(lo) == prefix {
                for e in self.u32_in(2, lo)..self.u32_in(2, lo + 1) {
                    f(prefix.len(), self.entry(e));
                }
            }
        }
    }

    /// Every reading of the lexicon that `text` starts with, as its length
    /// in bytes and the surfaces read so with their entries, shortest first,
    /// each reading's cheapest first.
    pub fn reading_prefixes<'a>(
        &'a self,
        text: &str,
        mut f: impl FnMut(usize, &'a str, Entry<'a>),
    ) {
        let at = |k: usize| self.u32_in(5, k);
        let lower = |mut lo: usize, mut hi: usize, probe: &[u8]| {
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.reading_of(at(mid)) < probe {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let (mut lo, mut hi) = (0, self.entries);
        for (n, (i, c)) in text.char_indices().enumerate() {
            if n == MAX_KEY_CHARS || lo == hi {
                break;
            }
            let prefix = &text.as_bytes()[..i + c.len_utf8()];
            lo = lower(lo, hi, prefix);
            let mut upper = prefix.to_vec();
            *upper.last_mut().unwrap() += 1;
            hi = lower(lo, hi, &upper);
            let mut k = lo;
            while k < hi && self.reading_of(at(k)) == prefix {
                let e = at(k);
                f(prefix.len(), self.surface_of(e), self.entry(e));
                k += 1;
            }
        }
    }

    /// Up to `n` conversions of `reading` (hiragana, as typed) by the
    /// lexicon alone, cheapest first, with their costs: sequences of its
    /// words (each reading's `PER_READING` cheapest surfaces, in each of
    /// their forms), kana as typed (`KANA_COST`) and digits, letters and
    /// symbols as typed (`TYPED_COST`), with the connection costs between
    /// them, `BEAM` paths kept at each reading position.
    pub fn convert(&self, reading: &str, n: usize) -> Vec<(String, f32)> {
        let r: Vec<(usize, char)> = reading.char_indices().collect();
        let m = r.len();
        let byte = |j: usize| r.get(j).map_or(reading.len(), |&(b, _)| b);
        // paths ending at each reading position: (cost, surface, right id)
        let mut paths: Vec<Vec<(f32, String, u16)>> = vec![Vec::new(); m + 1];
        paths[0].push((0.0, String::new(), BOS));
        for j in 0..m {
            if paths[j].is_empty() {
                continue;
            }
            // the `BEAM` cheapest surfaces, each ending in its `FORMS`
            // cheapest right ids
            paths[j].sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut kept: HashMap<String, Vec<u16>> = HashMap::new();
            paths[j].retain(|p| {
                let full = kept.len() == BEAM;
                match kept.get_mut(&p.1) {
                    Some(rids) if rids.len() < FORMS && !rids.contains(&p.2) => rids.push(p.2),
                    None if !full => {
                        kept.insert(p.1.clone(), vec![p.2]);
                    }
                    _ => return false,
                }
                true
            });
            // (to, surface, cost, lid, rid)
            let mut steps: Vec<(usize, &str, f32, u16, u16)> = Vec::new();
            let mut per: (usize, Vec<&str>) = (usize::MAX, Vec::new());
            self.reading_prefixes(&reading[byte(j)..], |len, surface, e| {
                // each reading's cheapest few surfaces
                if per.0 != len {
                    per = (len, Vec::new());
                }
                if !per.1.contains(&surface) {
                    if per.1.len() == PER_READING {
                        return;
                    }
                    per.1.push(surface);
                }
                let chars = reading[byte(j)..byte(j) + len].chars().count();
                let last = j + chars == m;
                steps.push((j + chars, surface, e.cost + e.ending(last), e.lid, e.rid));
            });
            let c = r[j].1;
            let typed = if is_kana(c) { KANA_COST } else { TYPED_COST };
            let piece = &reading[byte(j)..byte(j + 1)];
            steps.push((j + 1, piece, typed, NO_ID, NO_ID));
            let here = std::mem::take(&mut paths[j]);
            for (cost, surface, rid) in &here {
                for &(to, piece, c, lid, next) in &steps {
                    let c = cost + self.connection(*rid, lid) + c;
                    paths[to].push((c, format!("{surface}{piece}"), next));
                }
            }
        }
        let mut done = std::mem::take(&mut paths[m]);
        done.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut out: Vec<(String, f32)> = Vec::new();
        for (cost, surface, _) in done {
            if out.len() == n {
                break;
            }
            if !out.iter().any(|(s, _)| *s == surface) {
                out.push((surface, cost));
            }
        }
        out
    }

    /// Up to `n` of the lexicon's surfaces read `reading` whole, cheapest
    /// first, with their costs.
    pub fn words(&self, reading: &str, n: usize) -> Vec<(&str, f32)> {
        let mut out: Vec<(&str, f32)> = Vec::new();
        self.reading_prefixes(reading, |len, surface, e| {
            if len == reading.len() && out.len() < n && !out.iter().any(|(s, _)| *s == surface) {
                out.push((surface, e.cost));
            }
        });
        out
    }

    /// The words a dictionary IME lists for a reading of one word, maybe
    /// with a particle or okurigana after it (けんしゅうを, なんい？): up to
    /// `WORDS` of the lexicon's surfaces read as the longest start of the
    /// reading that is a word of it (a noun, verb, ...: not a particle or an
    /// affix, unless it is the whole reading), each followed by the rest as
    /// typed, `WORD_REST` chars at most (kana or punctuation, not starting
    /// with a small kana, ん or ー: 見込みっと for みこみっと) and no more than
    /// the word's, those costing `WORD_MARGIN` more than the cheapest left
    /// out (地震 for ない). Empty for a longer reading: a sentence's list is
    /// the model's.
    pub fn word_candidates(&self, reading: &str) -> Vec<(String, f32)> {
        let m = reading.chars().count();
        let mut best: Option<usize> = None;
        self.reading_prefixes(reading, |len, _, e| {
            let whole = len == reading.len();
            let content = !matches!(
                CLASSES.get(usize::from(e.class)).copied(),
                Some("part" | "aux" | "suff" | "pref" | "sym")
            );
            if (whole || content) && best.is_none_or(|b| len > b) {
                best = Some(len);
            }
        });
        let Some(len) = best else {
            return Vec::new();
        };
        let rest = &reading[len..];
        let rest_chars = rest.chars().count();
        let leans = rest
            .chars()
            .next()
            .is_some_and(|c| "ぁぃぅぇぉっゃゅょゎんー".contains(c));
        if rest_chars > WORD_REST
            || rest_chars > m - rest_chars
            || leans
            || !rest
                .chars()
                .all(|c| is_kana(c) || (!c.is_alphanumeric() && !c.is_whitespace()))
        {
            return Vec::new();
        }
        let words = self.words(&reading[..len], WORDS);
        let floor = words.first().map_or(0.0, |w| w.1) + WORD_MARGIN;
        words
            .into_iter()
            .filter(|w| w.1 <= floor)
            .map(|(s, c)| (format!("{s}{rest}"), c))
            .collect()
    }

    /// The one-char words of a reading of one or two kana (`CHAR_READING`),
    /// as a dictionary IME lists them: up to `CHAR_WORDS` of the lexicon's
    /// surfaces of one char that is a kanji (whatever its class), read as
    /// the whole reading, cheapest first, however dear (亜 亞 啞 痾 在 婀 有
    /// 合 吾 唖 我 阿 for あ, where `word_candidates` keeps those up to
    /// `WORD_MARGIN` over the cheapest). Empty for a longer reading.
    pub fn char_words(&self, reading: &str) -> Vec<(String, f32)> {
        let m = reading.chars().count();
        if m == 0 || m > CHAR_READING || !reading.chars().all(is_kana) {
            return Vec::new();
        }
        let mut out: Vec<(String, f32)> = Vec::new();
        self.reading_prefixes(reading, |len, surface, e| {
            let mut chars = surface.chars();
            // a kanji, of any class (Mozc's single kanji are symbols: 阿 for
            // あ), not a symbol, a or 5 (the table has them for あ and い)
            let one = matches!(
                (chars.next(), chars.next()),
                (Some(c), None) if c.is_alphabetic()
                    && !c.is_ascii()
                    && !is_kana(c)
                    && !matches!(c, '\u{FF01}'..='\u{FF5E}')
            );
            if len == reading.len()
                && one
                && out.len() < CHAR_WORDS
                && !out.iter().any(|(s, _)| s == surface)
            {
                out.push((surface.to_string(), e.cost));
            }
        });
        out
    }

    /// Where each char of `text` (a run of kanji the engine placed as one,
    /// 東京 for とうきょう) starts in `reading`, by the readings the lexicon
    /// has for each char alone, voiced at its start (人 of 恋人, びと) or
    /// with a small っ at its end (学 of 学校, がっ), 々 reading as the char
    /// before it; the cheapest such split. `None` when there is none.
    pub fn split_reading(&self, text: &str, reading: &str) -> Option<Vec<usize>> {
        let t: Vec<char> = text.chars().collect();
        let r: Vec<char> = reading.chars().collect();
        let (n, m) = (t.len(), r.len());
        if n == 0 || m < n {
            return None;
        }
        // each char's readings (chars) and costs
        let mut options: Vec<Vec<(Vec<char>, f32)>> = Vec::with_capacity(n);
        for (i, &c) in t.iter().enumerate() {
            let mut o: Vec<(Vec<char>, f32)> = if c == '々' && i > 0 {
                options[i - 1].clone()
            } else {
                self.get(c.encode_utf8(&mut [0; 4]))
                    .iter()
                    .map(|e| (e.reading.chars().collect(), e.cost))
                    .collect()
            };
            let base = o.clone();
            for (w, cost) in &base {
                for v in w.first().map_or(Vec::new(), |&k| voicings(k)) {
                    let mut w2 = w.clone();
                    w2[0] = v;
                    o.push((w2, cost + 1.0));
                }
                if w.len() > 1 && matches!(w.last(), Some('つ' | 'く' | 'ち' | 'き')) {
                    let mut w2 = w.clone();
                    *w2.last_mut().unwrap() = 'っ';
                    o.push((w2, cost + 1.0));
                }
            }
            options.push(o);
        }
        // best[i][j]: the cheapest split of t[..i] over r[..j], with its starts
        let mut best: Vec<Vec<Option<(f32, Vec<usize>)>>> = vec![vec![None; m + 1]; n + 1];
        best[0][0] = Some((0.0, Vec::new()));
        for i in 0..n {
            for j in 0..m {
                let Some((cost, starts)) = best[i][j].clone() else {
                    continue;
                };
                for (w, c) in &options[i] {
                    let k = j + w.len();
                    if w.is_empty() || k > m || r[j..k] != w[..] {
                        continue;
                    }
                    if best[i + 1][k].as_ref().is_none_or(|b| cost + c < b.0) {
                        let mut s = starts.clone();
                        s.push(j);
                        best[i + 1][k] = Some((cost + c, s));
                    }
                }
            }
        }
        best[n][m].take().map(|(_, s)| s)
    }

    /// The entries of `surface`.
    pub fn get(&self, surface: &str) -> Vec<Entry<'_>> {
        let mut out = Vec::new();
        self.prefixes(surface, |len, e| {
            if len == surface.len() {
                out.push(e);
            }
        });
        out
    }

    /// The cheapest reading of `surface` as `reading` (hiragana, as typed):
    /// the lexicon's words with the connection costs between them, kana
    /// spelled as typed (`KANA_COST` each), digits, Latin letters and
    /// symbols the reading has (`TYPED_COST`), and any other char at
    /// `UNKNOWN_COST`, taking 1 to 4 reading chars. `None` when no path
    /// spells the reading out (a kana out of place, a reading char left
    /// over).
    pub fn cost(&self, surface: &str, reading: &str) -> Option<f32> {
        let s: Vec<(usize, char)> = surface.char_indices().collect();
        let r: Vec<char> = reading.chars().collect();
        let (n, m) = (s.len(), r.len());
        let at = |i: usize| s.get(i).map_or(surface.len(), |&(b, _)| b);
        // the lexicon's words starting at each char: (chars, reading chars, entry)
        let mut words: Vec<Vec<(usize, Vec<char>, Entry<'_>)>> = vec![Vec::new(); n];
        for (i, w) in words.iter_mut().enumerate() {
            let start = at(i);
            self.prefixes(&surface[start..], |len, e| {
                let chars = surface[start..start + len].chars().count();
                w.push((chars, e.reading.chars().collect(), e));
            });
        }
        // the cheapest path to each (surface char, reading char) by the right
        // id it ends with
        let mut best: Vec<Vec<(u16, f32)>> = vec![Vec::new(); (n + 1) * (m + 1)];
        let idx = |i: usize, j: usize| i * (m + 1) + j;
        best[idx(0, 0)].push((BOS, 0.0));
        let relax =
            |v: &mut Vec<(u16, f32)>, rid: u16, c: f32| match v.iter_mut().find(|x| x.0 == rid) {
                Some(x) => x.1 = x.1.min(c),
                None => v.push((rid, c)),
            };
        for i in 0..n {
            for j in 0..=m {
                // what leaves (i, j) arrives further on in the surface
                let here = std::mem::take(&mut best[idx(i, j)]);
                if here.is_empty() {
                    continue;
                }
                let plain = here.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
                for (chars, wr, e) in &words[i] {
                    if r.len() - j >= wr.len() && r[j..j + wr.len()] == wr[..] {
                        let c = here
                            .iter()
                            .map(|&(rid, c)| c + self.connection(rid, e.lid))
                            .fold(f32::INFINITY, f32::min);
                        let (i2, j2) = (i + chars, j + wr.len());
                        let c = c + e.cost + e.ending(i2 == n && j2 == m);
                        relax(&mut best[idx(i2, j2)], e.rid, c);
                    }
                }
                let c = s[i].1;
                let rc = r.get(j).copied();
                if is_kana(c) {
                    if rc.is_some_and(|rc| to_hira(c) == rc) {
                        relax(&mut best[idx(i + 1, j + 1)], NO_ID, plain + KANA_COST);
                    }
                    continue;
                }
                if rc == Some(c)
                    || (!c.is_alphanumeric()
                        && rc.is_some_and(|rc| !rc.is_alphanumeric() && !is_kana(rc)))
                {
                    relax(&mut best[idx(i + 1, j + 1)], NO_ID, plain + TYPED_COST);
                }
                if c.is_alphanumeric() {
                    for k in 1..=4.min(m - j) {
                        relax(&mut best[idx(i + 1, j + k)], NO_ID, plain + UNKNOWN_COST);
                    }
                }
            }
        }
        let total = best[idx(n, m)]
            .iter()
            .map(|x| x.1)
            .fold(f32::INFINITY, f32::min);
        total.is_finite().then_some(total)
    }

    /// Compile `table` and `connection` (see the module docs) into the bytes
    /// of a lexicon; without a connection table its words connect at no
    /// cost.
    pub fn build(table: &str, connection: Option<&str>) -> Result<Vec<u8>> {
        let mut raw: Vec<f64> = Vec::new();
        let mut ids = 0usize;
        if let Some(text) = connection {
            let mut lines = text.lines();
            ids = lines
                .next()
                .and_then(|l| l.trim().parse().ok())
                .context("connection: the number of ids wanted")?;
            ensure!(ids < NO_ID as usize, "connection: {ids} ids");
            for (k, l) in lines.enumerate() {
                raw.push(
                    l.trim()
                        .parse()
                        .with_context(|| format!("connection line {}: {l}", k + 2))?,
                );
            }
            ensure!(
                raw.len() == ids * ids,
                "connection: {} costs for {ids} ids",
                raw.len()
            );
        }
        let mut rows: Vec<(&str, &str, u16, u8, u32)> = Vec::new();
        for (n, line) in table.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut f = line.split('\t');
            let (Some(s), Some(r), Some(c)) = (f.next(), f.next(), f.next()) else {
                bail!("line {}: surface, reading and cost wanted", n + 1);
            };
            let cost: f32 = c
                .parse()
                .with_context(|| format!("line {}: cost {c}", n + 1))?;
            let class = f
                .next()
                .and_then(|k| CLASSES.iter().position(|&x| x == k))
                .unwrap_or(0) as u8;
            // after the segments, the context ids (none past the connection
            // table's)
            let _segments = f.next();
            let mut id = || {
                f.next()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&v| v < ids)
                    .map_or(NO_ID, |v| v as u16)
            };
            let (lid, rid) = (id(), id());
            if s.is_empty() || r.is_empty() || r.len() > 255 {
                continue;
            }
            rows.push((
                s,
                r,
                (cost.max(0.0) * 100.0).round().min(65535.0) as u16,
                class,
                u32::from(lid) << 16 | u32::from(rid),
            ));
        }
        // how often text has each right id, by the table's costs
        let mut p = vec![0f64; ids];
        for row in &rows {
            if let Some(x) = p.get_mut((row.4 & 0xFFFF) as usize) {
                *x += (-f64::from(row.2) / 100.0).exp();
            }
        }
        let sum: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= sum.max(f64::MIN_POSITIVE));
        // each left id's connection cost after those, on average
        let mut base = vec![0f64; ids];
        for (l, b) in base.iter_mut().enumerate() {
            let likely: f64 = (0..ids).map(|r| p[r] * (-raw[r * ids + l]).exp()).sum();
            *b = if likely > 0.0 { -likely.ln() } else { 0.0 };
        }
        let costs: Vec<i16> = raw
            .iter()
            .enumerate()
            .map(|(k, c)| {
                ((c - base[k % ids]) * 100.0)
                    .round()
                    .clamp(-32768.0, 32767.0) as i16
            })
            .collect();
        rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()).then(a.2.cmp(&b.2)));
        let (mut key_off, mut key_blob, mut key_first) = (vec![0u32], Vec::new(), vec![0u32]);
        let (mut entry, mut reading_blob) = (Vec::with_capacity(rows.len()), Vec::new());
        for (k, (s, r, c, class, _)) in rows.iter().enumerate() {
            if k == 0 || rows[k - 1].0 != *s {
                if k > 0 {
                    key_first.push(k as u32);
                }
                key_blob.extend_from_slice(s.as_bytes());
                key_off.push(key_blob.len() as u32);
            }
            let off = reading_blob.len() as u64;
            reading_blob.extend_from_slice(r.as_bytes());
            entry
                .push(off << 32 | (r.len() as u64) << 24 | u64::from(*class) << 16 | u64::from(*c));
        }
        key_first.push(rows.len() as u32);
        let keys = key_off.len() - 1;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        // the entries by reading, then cost
        let mut by_reading: Vec<u32> = (0..rows.len() as u32).collect();
        by_reading.sort_by(|&a, &b| {
            let (a, b) = (&rows[a as usize], &rows[b as usize]);
            a.1.as_bytes().cmp(b.1.as_bytes()).then(a.2.cmp(&b.2))
        });
        for v in [
            VERSION,
            keys,
            rows.len(),
            key_blob.len(),
            reading_blob.len(),
            ids,
        ] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        out.resize(HEADER, 0);
        let mut put = |bytes: &[u8]| {
            out.extend_from_slice(bytes);
            out.resize(pad8(out.len()), 0);
        };
        put(&key_off
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        put(&key_blob);
        put(&key_first
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        put(&entry
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        put(&reading_blob);
        put(&by_reading
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        put(&rows
            .iter()
            .flat_map(|row| row.4.to_le_bytes())
            .collect::<Vec<_>>());
        put(&costs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>());
        Ok(out)
    }

    #[cfg(test)]
    fn from_table(table: &str) -> Self {
        Self::from_bytes(Bytes::Owned(Self::build(table, None).unwrap())).unwrap()
    }
}

/// The lexicon at `path`, mapped once per process; `None` (logged) when it
/// cannot be opened.
pub fn shared(path: &str) -> Option<Arc<Lexicon>> {
    type Cache = HashMap<String, Option<Arc<Lexicon>>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(path.to_string())
        .or_insert_with(|| match Lexicon::open(Path::new(path)) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                crate::diagnostics::log(format!("lexicon: {e:#}"));
                None
            }
        })
        .clone()
}

fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | 'ー')
}

/// The voiced kana of `c` (か -> が, は -> ば and ぱ).
fn voicings(c: char) -> Vec<char> {
    let at = |k: u32| char::from_u32(c as u32 + k);
    match c {
        'か' | 'き' | 'く' | 'け' | 'こ' | 'さ' | 'し' | 'す' | 'せ' | 'そ' | 'た' | 'ち'
        | 'つ' | 'て' | 'と' => at(1).into_iter().collect(),
        'は' | 'ひ' | 'ふ' | 'へ' | 'ほ' => [at(1), at(2)].into_iter().flatten().collect(),
        _ => Vec::new(),
    }
}

fn to_hira(c: char) -> char {
    match c {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "続投\tぞくとう\t15.0\tnoun\n\
                         続\tぞく\t13.3\tpref\n\
                         頭\tとう\t11.0\tnoun\n\
                         頭\tあたま\t9.0\tnoun\n\
                         が\tが\t3.0\tpart\n\
                         決まっ\tきまっ\t10.0\tverb\n\
                         た\tた\t3.5\taux\n\
                         今日\tきょう\t8.7\tnoun\n\
                         今日\tこんにち\t11.9\tnoun\n";

    #[test]
    fn surfaces_are_found_by_prefix_with_their_readings() {
        let lex = Lexicon::from_table(TABLE);
        assert_eq!(lex.len(), 9);
        let mut got = Vec::new();
        lex.prefixes("続投が", |len, e| got.push((len, e.reading.to_string())));
        assert_eq!(got, [(3, "ぞく".to_string()), (6, "ぞくとう".to_string())]);
        let today = lex.get("今日");
        assert_eq!(today.len(), 2);
        assert_eq!((today[0].reading, today[0].class), ("きょう", 1));
        assert!((today[0].cost - 8.7).abs() < 1e-6);
        assert!(lex.get("今").is_empty());
    }

    #[test]
    fn a_near_miss_costs_more_than_the_word() {
        let lex = Lexicon::from_table(TABLE);
        let good = lex.cost("続投が決まった", "ぞくとうがきまった").unwrap();
        let bad = lex.cost("続頭が決まった", "ぞくとうがきまった").unwrap();
        assert!((good - (15.0 + 3.0 + 10.0 + 3.5)).abs() < 1e-4, "{good}");
        assert!(bad > good + 5.0, "{bad} vs {good}");
        // a kanji the reading cannot take next to a word that took it
        let extra = lex.cost("き今日", "きょう").unwrap();
        assert!(extra >= KANA_COST + UNKNOWN_COST, "{extra}");
        // a kana out of place spells nothing
        assert_eq!(lex.cost("続投を", "ぞくとうが"), None);
    }

    #[test]
    fn a_reading_converts_to_the_cheapest_words() {
        let table = format!(
            "{TABLE}大島\tおおしま\t12.0\tpropn\n村\tむら\t10.0\tnoun\n大島村\tおおしまむら\t18.3\tpropn\n\
             大\tおお\t14.0\tpref\n島\tしま\t11.0\tnoun\n"
        );
        let lex = Lexicon::from_table(&table);
        let mut got = Vec::new();
        lex.reading_prefixes("おおしまむら", |len, s, e| {
            got.push((len, s.to_string(), e.cost))
        });
        assert_eq!(got[0], (6, "大".to_string(), 14.0));
        assert!(got.iter().any(|(_, s, _)| s == "大島村"));
        let out = lex.convert("おおしまむら", 3);
        assert_eq!(out[0].0, "大島村");
        assert!((out[0].1 - 18.3).abs() < 1e-4);
        // 大島 + 村 is the same surface again, not repeated
        assert_eq!(out[1].0, "大島むら");
        // kana the lexicon has no word for come as typed
        let out = lex.convert("ぞくとうがきまった", 1);
        assert_eq!(out[0].0, "続投が決まった");
        assert_eq!(lex.convert("ん", 2)[0].0, "ん");
    }

    #[test]
    fn a_word_costs_by_the_word_before_it() {
        // ids: 0 the start, 1 a noun, 2 a verb's form before た, 3 its form
        // before ます, 4 た; た after 2 is cheap, after 1 or 3 dear
        let table = "貝\tかい\t8.5\tnoun\t\t1\t1\n\
                     書い\tかい\t9.0\tverb\t\t2\t2\n\
                     書き\tかき\t9.0\tverb\t\t3\t3\n\
                     柿\tかき\t8.5\tnoun\t\t1\t1\n\
                     た\tた\t3.5\taux\t\t4\t4\n";
        let mut connection = String::from("5\n");
        for r in 0..5 {
            for l in 0..5 {
                let c = if l == 4 && r != 2 { 6.0 } else { 1.0 };
                connection.push_str(&format!("{c}\n"));
            }
        }
        let bytes = Lexicon::build(table, Some(&connection)).unwrap();
        let lex = Lexicon::from_bytes(Bytes::Owned(bytes)).unwrap();
        let e = lex.get("書い");
        assert_eq!((e[0].lid, e[0].rid), (2, 2));
        let (verb, noun) = (
            lex.cost("書いた", "かいた").unwrap(),
            lex.cost("貝た", "かいた").unwrap(),
        );
        assert!(verb + 4.0 < noun, "{verb} vs {noun}");
        assert_eq!(lex.convert("かいた", 1)[0].0, "書いた");
        // alone, the cheaper word
        assert_eq!(lex.convert("かい", 1)[0].0, "貝");
        // a prefix does not end the reading if a word can
        assert_eq!(CLASSES[PREFIX], "pref");
        let more = format!("{table}各\tかく\t8.0\tpref\t\t1\t1\n書く\tかく\t9.0\tverb\t\t2\t2\n");
        let lex = Lexicon::from_bytes(Bytes::Owned(
            Lexicon::build(&more, Some(&connection)).unwrap(),
        ))
        .unwrap();
        assert_eq!(lex.convert("かく", 1)[0].0, "書く");
        let prefix = lex.cost("各", "かく").unwrap();
        assert!((prefix - lex.cost("書く", "かく").unwrap() - (PREFIX_LAST - 1.0)).abs() < 1e-4);
        // without the connection table, as before
        let flat = Lexicon::from_table(table);
        assert_eq!(flat.convert("かいた", 1)[0].0, "貝た");
        assert_eq!(flat.get("書い")[0].lid, NO_ID);
    }

    #[test]
    fn typed_digits_letters_and_symbols_spell_themselves() {
        let lex = Lexicon::from_table(TABLE);
        assert_eq!(lex.cost("3、", "3、"), Some(2.0 * TYPED_COST));
        assert!(lex.cost("今日。", "きょう。").is_some());
    }

    #[test]
    fn a_kanji_run_splits_by_its_chars_readings() {
        let table = "東\tとう\t10.0\tnoun\n京\tきょう\t10.0\tnoun\n学\tがく\t10.0\tnoun\n校\tこう\t10.0\tnoun\n人\tひと\t10.0\tnoun\n今日\tきょう\t8.0\tnoun\n";
        let lex = Lexicon::from_table(table);
        assert_eq!(lex.split_reading("東京", "とうきょう"), Some(vec![0, 2]));
        // a small っ at a char's end, a voiced start, 々 as the char before
        assert_eq!(lex.split_reading("学校", "がっこう"), Some(vec![0, 2]));
        assert_eq!(lex.split_reading("人々", "ひとびと"), Some(vec![0, 2]));
        // a word read whole (今日) has no reading of its chars
        assert_eq!(lex.split_reading("今日", "きょう"), None);
    }

    #[test]
    fn a_reading_of_one_word_lists_its_words() {
        let table = format!(
            "{TABLE}研修\tけんしゅう\t9.4\tnoun\n検収\tけんしゅう\t14.6\tnoun\n\
             謙\tけんしゅう\t25.7\tnoun\n見込み\tみこみ\t10.0\tnoun\n"
        );
        let lex = Lexicon::from_table(&table);
        let words = |r: &str| -> Vec<String> {
            lex.word_candidates(r).into_iter().map(|(w, _)| w).collect()
        };
        // cheapest first, those WORD_MARGIN dearer than the cheapest left out
        assert_eq!(words("けんしゅう"), ["研修", "検収"]);
        assert_eq!(lex.words("けんしゅう", 2), [("研修", 9.4), ("検収", 14.6)]);
        // a particle, okurigana or punctuation after the word, as typed
        assert_eq!(words("けんしゅうを"), ["研修を", "検収を"]);
        assert_eq!(words("きょう？"), ["今日？"]);
        // a particle alone is its own word; after a word it is the rest
        assert_eq!(words("が"), ["が"]);
        // more after the word than it has, or a rest that leans on it, is no
        // reading of one word
        assert!(words("ぞくとうがきまった").is_empty());
        assert!(words("みこみっと").is_empty());
        assert!(words("ぞくがきまった").is_empty());
        assert!(words("ほげ").is_empty());
    }

    #[test]
    fn a_reading_of_a_kana_or_two_lists_its_one_char_words() {
        let table = "亜\tあ\t12.7\tnoun\n阿\tあ\t23.7\tsym\nあ\tあ\t10.9\tintj\n\
                     a\tあ\t23.7\tnoun\n○\tまる\t9.0\tsym\n丸\tまる\t10.0\tnoun\n\
                     円\tまる\t15.0\tnoun\n丸い\tまるい\t9.0\tadj\n";
        let lex = Lexicon::from_table(table);
        let chars =
            |r: &str| -> Vec<String> { lex.char_words(r).into_iter().map(|(w, _)| w).collect() };
        // however dear, kanji only (阿 of the symbols' class, as Mozc has
        // single kanji): not the kana, a Latin letter or a symbol
        assert_eq!(chars("あ"), ["亜", "阿"]);
        assert_eq!(chars("まる"), ["丸", "円"]);
        // three kana are a word's reading, not a char's
        assert!(chars("まるい").is_empty());
    }
}
