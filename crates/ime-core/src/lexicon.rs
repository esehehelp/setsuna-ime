//! A reading lexicon: (surface, reading) pairs with a cost each, to tell the
//! model's near-misses from conversions text uses. The model emits a
//! candidate char by char, so its list holds surfaces no dictionary has
//! (続頭 for ぞくとう, き今日 for きょう); `cost` reads a candidate against
//! its reading as a lattice of the lexicon's words, the kana, digits and
//! symbols spelled as typed, and other chars at a high cost, and gives the
//! cheapest path.
//!
//! The source is a text table, `surface<TAB>reading<TAB>cost<TAB>class`
//! (more columns ignored; cost in nats, the reading in hiragana), which
//! `build` compiles into the file the IME maps (`LEXICON1`), little-endian:
//! a 64-byte header (magic, version u32 = 2, keys u32, entries u32,
//! key_bytes u32, reading_bytes u32) and then, each padded to 8 bytes:
//! key_off u32[keys + 1] and key_blob u8[key_bytes] (the surfaces, UTF-8,
//! ascending by bytes), key_first u32[keys + 1] (each surface's entries),
//! entry u64[entries] (reading offset 32 bits, reading length 8, class 8,
//! cost in centinats 16), reading_blob u8[reading_bytes] and by_reading
//! u32[entries] (the entries by reading, ascending by bytes, then by cost).
//! The file is mapped, not read, so the processes the IME runs in share one
//! copy.
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
/// a conversion `ADDED_UNDER` cheaper than every candidate of the engine is
/// added, scored `ADDED_BELOW` under its top, for the lexicon's cost and
/// the reranker to place. Picked on wiki_fresh_v2 (Shiran-v2 s14000, the
/// reranker at lambda 0.8 over the top 20): golds in the list 4339 -> 4417,
/// EM1 79.95 -> 81.05, by the engine's score and the lexicon 76.81 -> 76.83;
/// AJIMEE 183 -> 185, 83.0 -> 84.5, 75.5 -> 74.5. Adding every conversion
/// found more golds (4529) and put the unigram lexicon's own misses in the
/// list (紙に各 for かみにかく beside 紙に書く).
pub const CONVERSIONS: usize = 10;
pub const ADDED_UNDER: f32 = 3.0;
pub const ADDED_BELOW: f32 = 8.0;

const MAGIC: &[u8; 8] = b"LEXICON1";
const VERSION: usize = 2;
const HEADER: usize = 64;
/// The longest surface (and reading) looked up, in chars.
const MAX_KEY_CHARS: usize = 24;
/// The surfaces of one reading `convert` tries, the cheapest.
const PER_READING: usize = 8;
/// The paths `convert` keeps at each reading position.
const BEAM: usize = 8;

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

/// One reading of a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry<'a> {
    pub reading: &'a str,
    pub cost: f32,
    pub class: u8,
}

pub struct Lexicon {
    bytes: Bytes,
    keys: usize,
    entries: usize,
    /// Byte offsets of key_off, key_blob, key_first, entry, reading_blob,
    /// by_reading.
    at: [usize; 6],
}

/// The coarse parts of speech of the table's class column, by index.
pub const CLASSES: [&str; 15] = [
    "other", "noun", "propn", "verb", "adj", "adjn", "adv", "adnom", "conj", "intj", "part", "aux",
    "pref", "suff", "sym",
];

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
        ensure!(
            u32_at(8) == VERSION,
            "lexicon version {} (this IME reads {VERSION})",
            u32_at(8)
        );
        let (keys, entries, key_bytes, reading_bytes) =
            (u32_at(12), u32_at(16), u32_at(20), u32_at(24));
        let sizes = [
            4 * (keys + 1),
            key_bytes,
            4 * (keys + 1),
            8 * entries,
            reading_bytes,
            4 * entries,
        ];
        let mut at = [0usize; 6];
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
            at,
        })
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
        Entry {
            // the blob holds what `build` wrote from a &str
            reading: std::str::from_utf8(&self.bytes[start..start + len]).unwrap_or(""),
            class: ((v >> 16) & 0xFF) as u8,
            cost: (v & 0xFFFF) as f32 / 100.0,
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
    /// words (each reading's `PER_READING` cheapest), kana as typed
    /// (`KANA_COST`) and digits, letters and symbols as typed (`TYPED_COST`),
    /// `BEAM` paths kept at each reading position.
    pub fn convert(&self, reading: &str, n: usize) -> Vec<(String, f32)> {
        let r: Vec<(usize, char)> = reading.char_indices().collect();
        let m = r.len();
        let byte = |j: usize| r.get(j).map_or(reading.len(), |&(b, _)| b);
        // paths ending at each reading position: (cost, surface)
        let mut paths: Vec<Vec<(f32, String)>> = vec![Vec::new(); m + 1];
        paths[0].push((0.0, String::new()));
        for j in 0..m {
            if paths[j].is_empty() {
                continue;
            }
            paths[j].sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut seen = std::collections::HashSet::new();
            paths[j].retain(|p| seen.insert(p.1.clone()));
            paths[j].truncate(BEAM);
            let mut steps: Vec<(usize, String, f32)> = Vec::new();
            let mut per: (usize, usize) = (usize::MAX, 0);
            self.reading_prefixes(&reading[byte(j)..], |len, surface, e| {
                // each reading's cheapest few
                per = if per.0 == len {
                    (len, per.1 + 1)
                } else {
                    (len, 1)
                };
                if per.1 <= PER_READING {
                    let chars = reading[byte(j)..byte(j) + len].chars().count();
                    steps.push((j + chars, surface.to_string(), e.cost));
                }
            });
            let c = r[j].1;
            let typed = if is_kana(c) { KANA_COST } else { TYPED_COST };
            steps.push((j + 1, c.to_string(), typed));
            let here = std::mem::take(&mut paths[j]);
            for (cost, surface) in &here {
                for (to, piece, c) in &steps {
                    paths[*to].push((cost + c, format!("{surface}{piece}")));
                }
            }
        }
        let mut done = std::mem::take(&mut paths[m]);
        done.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut out: Vec<(String, f32)> = Vec::new();
        for (cost, surface) in done {
            if out.len() == n {
                break;
            }
            if !out.iter().any(|(s, _)| *s == surface) {
                out.push((surface, cost));
            }
        }
        out
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
    /// the lexicon's words, kana spelled as typed (`KANA_COST` each),
    /// digits, Latin letters and symbols the reading has (`TYPED_COST`), and
    /// any other char at `UNKNOWN_COST`, taking 1 to 4 reading chars. `None`
    /// when no path spells the reading out (a kana out of place, a reading
    /// char left over).
    pub fn cost(&self, surface: &str, reading: &str) -> Option<f32> {
        let s: Vec<(usize, char)> = surface.char_indices().collect();
        let r: Vec<char> = reading.chars().collect();
        let (n, m) = (s.len(), r.len());
        let at = |i: usize| s.get(i).map_or(surface.len(), |&(b, _)| b);
        // the lexicon's words starting at each char: (chars, reading chars, cost)
        let mut words: Vec<Vec<(usize, Vec<char>, f32)>> = vec![Vec::new(); n];
        for (i, w) in words.iter_mut().enumerate() {
            let start = at(i);
            self.prefixes(&surface[start..], |len, e| {
                let chars = surface[start..start + len].chars().count();
                w.push((chars, e.reading.chars().collect(), e.cost));
            });
        }
        let mut best = vec![f32::INFINITY; (n + 1) * (m + 1)];
        let idx = |i: usize, j: usize| i * (m + 1) + j;
        best[idx(0, 0)] = 0.0;
        for i in 0..n {
            for j in 0..=m {
                let here = best[idx(i, j)];
                if !here.is_finite() {
                    continue;
                }
                let mut relax = |i2: usize, j2: usize, c: f32| {
                    let k = idx(i2, j2);
                    if here + c < best[k] {
                        best[k] = here + c;
                    }
                };
                for (chars, wr, c) in &words[i] {
                    if r.len() - j >= wr.len() && r[j..j + wr.len()] == wr[..] {
                        relax(i + chars, j + wr.len(), *c);
                    }
                }
                let c = s[i].1;
                let rc = r.get(j).copied();
                if is_kana(c) {
                    if rc.is_some_and(|rc| to_hira(c) == rc) {
                        relax(i + 1, j + 1, KANA_COST);
                    }
                    continue;
                }
                if rc == Some(c)
                    || (!c.is_alphanumeric()
                        && rc.is_some_and(|rc| !rc.is_alphanumeric() && !is_kana(rc)))
                {
                    relax(i + 1, j + 1, TYPED_COST);
                }
                if c.is_alphanumeric() {
                    for k in 1..=4.min(m - j) {
                        relax(i + 1, j + k, UNKNOWN_COST);
                    }
                }
            }
        }
        let total = best[idx(n, m)];
        total.is_finite().then_some(total)
    }

    /// Compile `table` (see the module docs) into the bytes of a lexicon.
    pub fn build(table: &str) -> Result<Vec<u8>> {
        let mut rows: Vec<(&str, &str, u16, u8)> = Vec::new();
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
            if s.is_empty() || r.is_empty() || r.len() > 255 {
                continue;
            }
            rows.push((
                s,
                r,
                (cost.max(0.0) * 100.0).round().min(65535.0) as u16,
                class,
            ));
        }
        rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()).then(a.2.cmp(&b.2)));
        let (mut key_off, mut key_blob, mut key_first) = (vec![0u32], Vec::new(), vec![0u32]);
        let (mut entry, mut reading_blob) = (Vec::with_capacity(rows.len()), Vec::new());
        for (k, (s, r, c, class)) in rows.iter().enumerate() {
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
        Ok(out)
    }

    #[cfg(test)]
    fn from_table(table: &str) -> Self {
        Self::from_bytes(Bytes::Owned(Self::build(table).unwrap())).unwrap()
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
    fn typed_digits_letters_and_symbols_spell_themselves() {
        let lex = Lexicon::from_table(TABLE);
        assert_eq!(lex.cost("3、", "3、"), Some(2.0 * TYPED_COST));
        assert!(lex.cost("今日。", "きょう。").is_some());
    }
}
