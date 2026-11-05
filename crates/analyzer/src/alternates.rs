//! The other readings of a word: for each surface the dictionary reads more
//! than one way, its entries (left id, right id, word cost, reading), and
//! the connection costs, so that what a token would cost read another way in
//! its place (its word cost and its connections to the tokens around it) can
//! be weighed against the reading the analysis took.
//!
//! `build` compiles them from the analyzer's trained `lex.csv` (surface,
//! left id, right id, cost, then the features) and its connection costs into
//! the file the analyzer maps (`ALTERN01`), little-endian: a 64-byte header
//! (magic, version u32, num_right u32, num_left u32, keys u32, entries u32,
//! key_bytes u32, reading_bytes u32, and in version 2 templates u32 and pairs
//! u32) and then, each padded to 8 bytes: key_off u32[keys + 1] and key_blob
//! u8[key_bytes] (the surfaces, UTF-8, ascending by bytes), key_first
//! u32[keys + 1] (each surface's entries), entry u32[4 * entries] (left id |
//! right id << 16, the cost as i32, the reading's offset and length),
//! reading_blob u8[reading_bytes], and the connection costs:
//!
//! - version 1, from `matrix.def` (`right left` and then `right left cost` a
//!   line): connection i16[num_left * num_right] (by left id then right id);
//! - version 2, from the model's bigram files (`bigram.right`, `bigram.left`:
//!   `id<TAB>csv row` of a connection id's features, `bigram.cost`:
//!   `right feature/left feature<TAB>cost`), the costs vibrato's raw connector
//!   gives (the dictionary built from the same files): right_feats
//!   u32[num_right * templates] and left_feats u32[num_left * templates] (each
//!   id's features as indices into the cost file's, u32::MAX for none; id 0,
//!   a sentence's ends, the empty feature at every template), pair_key
//!   u64[pairs] (right feature << 32 | left feature, ascending) and pair_cost
//!   i32[pairs]. A connection costs the sum over the templates of its pair of
//!   features' cost, a few MB where the matrix takes num_left * num_right.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

const MAGIC: &[u8; 8] = b"ALTERN01";
const HEADER: usize = 64;
const NO_FEATURE: u32 = u32::MAX;

/// One entry of a surface: its context ids, word cost and reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry<'a> {
    pub left: u16,
    pub right: u16,
    pub cost: i32,
    pub reading: &'a str,
}

/// Where the connection costs are in the map.
enum Connection {
    Matrix(usize),
    Features {
        templates: usize,
        right_feats: usize,
        left_feats: usize,
        pairs: usize,
        pair_key: usize,
        pair_cost: usize,
    },
}

/// The mapped table.
pub struct Alternates {
    map: memmap2::Mmap,
    num_right: usize,
    num_left: usize,
    keys: usize,
    key_off: usize,
    key_blob: usize,
    key_first: usize,
    entry: usize,
    reading_blob: usize,
    connection: Connection,
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

impl Alternates {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is the installed table, not written while mapped
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("map {}", path.display()))?;
        ensure!(
            map.len() >= HEADER && &map[..8] == MAGIC,
            "{} is no alternates table",
            path.display()
        );
        let h = |i: usize| u32_at(&map, 8 + 4 * i) as usize;
        let version = h(0);
        ensure!(
            version == 1 || version == 2,
            "{}: version {version}, expected 1 or 2",
            path.display()
        );
        let (num_right, num_left, keys, entries, key_bytes, reading_bytes) =
            (h(1), h(2), h(3), h(4), h(5), h(6));
        let key_off = HEADER;
        let key_blob = key_off + pad8(4 * (keys + 1));
        let key_first = key_blob + pad8(key_bytes);
        let entry = key_first + pad8(4 * (keys + 1));
        let reading_blob = entry + 16 * entries;
        let costs = reading_blob + pad8(reading_bytes);
        let (connection, end) = if version == 1 {
            (
                Connection::Matrix(costs),
                costs + pad8(2 * num_left * num_right),
            )
        } else {
            let (templates, pairs) = (h(7), h(8));
            let right_feats = costs;
            let left_feats = right_feats + pad8(4 * num_right * templates);
            let pair_key = left_feats + pad8(4 * num_left * templates);
            let pair_cost = pair_key + 8 * pairs;
            (
                Connection::Features {
                    templates,
                    right_feats,
                    left_feats,
                    pairs,
                    pair_key,
                    pair_cost,
                },
                pair_cost + pad8(4 * pairs),
            )
        };
        ensure!(
            map.len() == end,
            "{}: {} bytes, the header says {end}",
            path.display(),
            map.len(),
        );
        Ok(Self {
            map,
            num_right,
            num_left,
            keys,
            key_off,
            key_blob,
            key_first,
            entry,
            reading_blob,
            connection,
        })
    }

    fn key(&self, i: usize) -> &[u8] {
        let a = u32_at(&self.map, self.key_off + 4 * i) as usize;
        let b = u32_at(&self.map, self.key_off + 4 * (i + 1)) as usize;
        &self.map[self.key_blob + a..self.key_blob + b]
    }

    /// The entries of `surface`; none when the dictionary reads it one way.
    pub fn entries(&self, surface: &str) -> impl Iterator<Item = Entry<'_>> {
        let (mut lo, mut hi) = (0, self.keys);
        let s = surface.as_bytes();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key(mid) < s {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let range = if lo < self.keys && self.key(lo) == s {
            let first = |i: usize| u32_at(&self.map, self.key_first + 4 * i) as usize;
            first(lo)..first(lo + 1)
        } else {
            0..0
        };
        range.map(move |i| {
            let e = |k: usize| u32_at(&self.map, self.entry + 16 * i + 4 * k);
            let ids = e(0);
            let (off, len) = (e(2) as usize, e(3) as usize);
            let at = self.reading_blob + off;
            Entry {
                left: ids as u16,
                right: (ids >> 16) as u16,
                cost: e(1) as i32,
                // written from a &str by `build`
                reading: std::str::from_utf8(&self.map[at..at + len]).unwrap_or(""),
            }
        })
    }

    /// The cost of a word ending with `right` followed by one starting with
    /// `left` (id 0: a sentence's ends).
    pub fn connection(&self, right: u16, left: u16) -> i32 {
        let (r, l) = (usize::from(right), usize::from(left));
        if r >= self.num_right || l >= self.num_left {
            return 0;
        }
        match self.connection {
            Connection::Matrix(at) => {
                let at = at + 2 * (l * self.num_right + r);
                i32::from(i16::from_le_bytes([self.map[at], self.map[at + 1]]))
            }
            Connection::Features {
                templates,
                right_feats,
                left_feats,
                pairs,
                pair_key,
                pair_cost,
            } => {
                let mut cost = 0;
                for t in 0..templates {
                    let rf = u32_at(&self.map, right_feats + 4 * (r * templates + t));
                    let lf = u32_at(&self.map, left_feats + 4 * (l * templates + t));
                    if rf == NO_FEATURE || lf == NO_FEATURE {
                        continue;
                    }
                    let key = u64::from(rf) << 32 | u64::from(lf);
                    let (mut lo, mut hi) = (0, pairs);
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if u64_at(&self.map, pair_key + 8 * mid) < key {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    if lo < pairs && u64_at(&self.map, pair_key + 8 * lo) == key {
                        cost += u32_at(&self.map, pair_cost + 4 * lo) as i32;
                    }
                }
                cost
            }
        }
    }
}

/// One `lex.csv` line: the surface and the fields after it.
fn split_lex_line(line: &str) -> Option<(String, &str)> {
    // a surface holding a comma is quoted ("1,000円",...); features hold none
    match line.strip_prefix('"') {
        Some(q) => {
            let end = q.find("\",")?;
            Some((q[..end].replace("\"\"", "\""), &q[end + 2..]))
        }
        None => {
            let (s, rest) = line.split_once(',')?;
            Some((s.to_string(), rest))
        }
    }
}

/// The cells of a CSV row (a quoted cell holds commas, `""` is a quote).
fn csv_cells(row: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = row.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cell.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => cells.push(std::mem::take(&mut cell)),
            _ => cell.push(c),
        }
    }
    cells.push(cell);
    cells
}

/// The connection costs a `build` writes after the entries.
enum Costs {
    Matrix {
        num_right: usize,
        num_left: usize,
        connection: Vec<i16>,
    },
    Features {
        right_feats: Vec<Vec<u32>>,
        left_feats: Vec<Vec<u32>>,
        templates: usize,
        pairs: Vec<(u64, i32)>,
    },
}

fn read_matrix(matrix: &Path) -> Result<Costs> {
    let mut lines =
        BufReader::new(File::open(matrix).with_context(|| format!("open {}", matrix.display()))?)
            .lines();
    let head = lines.next().context("empty matrix.def")??;
    let (r, l) = head.split_once(' ').context("matrix.def header")?;
    let (num_right, num_left): (usize, usize) = (r.trim().parse()?, l.trim().parse()?);
    let mut connection = vec![0i16; num_left * num_right];
    for line in lines {
        let line = line?;
        let mut it = line.split(' ');
        if let (Some(r), Some(l), Some(c)) = (it.next(), it.next(), it.next()) {
            let (r, l): (usize, usize) = (r.parse()?, l.parse()?);
            ensure!(r < num_right && l < num_left, "matrix.def: {line}");
            connection[l * num_right + r] = c.trim().parse()?;
        }
    }
    Ok(Costs::Matrix {
        num_right,
        num_left,
        connection,
    })
}

/// The bigram files in `dir`, their features numbered as the cost file
/// names them (vibrato's raw connector: the first cost line's features are
/// 0, which a sentence's ends take at every template).
fn read_bigram(dir: &Path) -> Result<Costs> {
    let open = |name: &str| -> Result<BufReader<File>> {
        let p = dir.join(name);
        Ok(BufReader::new(
            File::open(&p).with_context(|| format!("open {}", p.display()))?,
        ))
    };
    let (mut right_ids, mut left_ids) = (HashMap::new(), HashMap::new());
    let mut pairs = Vec::new();
    for line in open("bigram.cost")?.lines() {
        let line = line?;
        let (key, cost) = line.rsplit_once('\t').context("bigram.cost: no tab")?;
        let (r, l) = key
            .split_once('/')
            .with_context(|| format!("bigram.cost: {line}"))?;
        ensure!(!l.contains('/'), "bigram.cost: more than one '/': {line}");
        let n = right_ids.len() as u32;
        let r = *right_ids.entry(r.to_string()).or_insert(n);
        let n = left_ids.len() as u32;
        let l = *left_ids.entry(l.to_string()).or_insert(n);
        pairs.push((u64::from(r) << 32 | u64::from(l), cost.parse::<i32>()?));
    }
    pairs.sort_unstable();
    pairs.dedup_by_key(|p| p.0);
    let rows = |name: &str, ids: &HashMap<String, u32>| -> Result<Vec<Vec<u32>>> {
        let mut rows: Vec<Vec<u32>> = vec![Vec::new()];
        for line in open(name)?.lines() {
            let line = line?;
            let (id, row) = line
                .split_once('\t')
                .with_context(|| format!("{name}: {line}"))?;
            let id: usize = id.parse().with_context(|| format!("{name}: {line}"))?;
            if rows.len() <= id {
                rows.resize(id + 1, Vec::new());
            }
            rows[id] = csv_cells(row)
                .iter()
                .map(|f| ids.get(f).copied().unwrap_or(NO_FEATURE))
                .collect();
        }
        Ok(rows)
    };
    let mut right_feats = rows("bigram.right", &right_ids)?;
    let mut left_feats = rows("bigram.left", &left_ids)?;
    let templates = right_feats
        .iter()
        .chain(&left_feats)
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    for row in right_feats.iter_mut().chain(left_feats.iter_mut()) {
        row.resize(templates, NO_FEATURE);
    }
    right_feats[0] = vec![0; templates];
    left_feats[0] = vec![0; templates];
    Ok(Costs::Features {
        right_feats,
        left_feats,
        templates,
        pairs,
    })
}

/// Compile `lex` and the connection costs `connection` names (a
/// `matrix.def`, or a directory of bigram files) into `out`, the entries'
/// reading in feature column `reading_col`. Returns (surfaces, entries) kept.
pub fn build(
    lex: &Path,
    connection: &Path,
    reading_col: usize,
    out: &Path,
) -> Result<(usize, usize)> {
    let mut by_surface: BTreeMap<String, Vec<(u16, u16, i32, String)>> = BTreeMap::new();
    // surfaces with an entry of no reading (/ as a symbol): their one reading
    // is another than the analysis's
    let mut unread = BTreeSet::new();
    let lines = BufReader::new(File::open(lex).with_context(|| format!("open {}", lex.display()))?);
    for (n, line) in lines.lines().enumerate() {
        let line = line?;
        let Some((surface, rest)) = split_lex_line(&line) else {
            bail!("{}:{}: no surface", lex.display(), n + 1);
        };
        let f: Vec<&str> = rest.split(',').collect();
        // left, right, cost, then the features
        let Some(&reading) = f.get(3 + reading_col) else {
            bail!("{}:{}: no reading column", lex.display(), n + 1);
        };
        if reading == "*" {
            unread.insert(surface);
            continue;
        }
        let parse = |i: usize| -> Result<i64> {
            f[i].parse()
                .with_context(|| format!("{}:{}: field {i}", lex.display(), n + 1))
        };
        by_surface.entry(surface).or_default().push((
            parse(0)? as u16,
            parse(1)? as u16,
            parse(2)? as i32,
            reading.to_string(),
        ));
    }
    by_surface.retain(|s, e| unread.contains(s) || e.iter().any(|x| x.3 != e[0].3));

    let costs = if connection.is_dir() {
        read_bigram(connection)?
    } else {
        read_matrix(connection)?
    };

    let mut key_off = vec![0u32];
    let mut key_blob = Vec::new();
    let mut key_first = vec![0u32];
    let mut entry = Vec::new();
    let mut reading_blob = Vec::new();
    for (surface, entries) in &by_surface {
        key_blob.extend_from_slice(surface.as_bytes());
        key_off.push(key_blob.len() as u32);
        for (l, r, c, reading) in entries {
            entry.extend([
                u32::from(*l) | u32::from(*r) << 16,
                *c as u32,
                reading_blob.len() as u32,
                reading.len() as u32,
            ]);
            reading_blob.extend_from_slice(reading.as_bytes());
        }
        key_first.push((entry.len() / 4) as u32);
    }

    let mut w = std::io::BufWriter::new(
        File::create(out).with_context(|| format!("create {}", out.display()))?,
    );
    let mut header = Vec::with_capacity(HEADER);
    header.extend_from_slice(MAGIC);
    let (version, num_right, num_left, extra) = match &costs {
        Costs::Matrix {
            num_right,
            num_left,
            ..
        } => (1u32, *num_right, *num_left, vec![]),
        Costs::Features {
            right_feats,
            left_feats,
            templates,
            pairs,
        } => (
            2,
            right_feats.len(),
            left_feats.len(),
            vec![*templates as u32, pairs.len() as u32],
        ),
    };
    for v in [
        version,
        num_right as u32,
        num_left as u32,
        by_surface.len() as u32,
        (entry.len() / 4) as u32,
        key_blob.len() as u32,
        reading_blob.len() as u32,
    ]
    .into_iter()
    .chain(extra)
    {
        header.extend_from_slice(&v.to_le_bytes());
    }
    header.resize(HEADER, 0);
    w.write_all(&header)?;
    let mut padded = |bytes: Vec<u8>| -> Result<()> {
        let n = bytes.len();
        w.write_all(&bytes)?;
        w.write_all(&vec![0; pad8(n) - n])?;
        Ok(())
    };
    let le = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    padded(le(&key_off))?;
    padded(key_blob)?;
    padded(le(&key_first))?;
    padded(le(&entry))?;
    padded(reading_blob)?;
    match costs {
        Costs::Matrix { connection, .. } => {
            padded(connection.iter().flat_map(|x| x.to_le_bytes()).collect())?;
        }
        Costs::Features {
            right_feats,
            left_feats,
            pairs,
            ..
        } => {
            padded(le(&right_feats.concat()))?;
            padded(le(&left_feats.concat()))?;
            padded(pairs.iter().flat_map(|p| p.0.to_le_bytes()).collect())?;
            padded(pairs.iter().flat_map(|p| p.1.to_le_bytes()).collect())?;
        }
    }
    drop(padded);
    w.flush()?;
    Ok((by_surface.len(), entry.len() / 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEX: &str = "今日,1,2,100,名詞,普通名詞,*,*,きょう,今日\n\
                       今日,1,2,1500,名詞,普通名詞,*,*,こんにち,今日\n\
                       \"1,000\",3,3,50,名詞,数詞,*,*,せん,1000\n\
                       天気,1,2,200,名詞,普通名詞,*,*,てんき,天気\n\
                       謎,1,2,200,名詞,普通名詞,*,*,*,謎\n\
                       /,4,4,0,補助記号,一般,*,*,*,/\n\
                       /,1,2,900,補助記号,一般,*,*,すらっしゅ,/\n";

    fn dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("setsuna-alternates-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn builds_and_reads_back() {
        let dir = dir("matrix");
        let lex = dir.join("lex.csv");
        let matrix = dir.join("matrix.def");
        let out = dir.join("alternates.bin");
        std::fs::write(&lex, LEX).unwrap();
        std::fs::write(&matrix, "3 5\n0 1 -5\n2 0 7\n2 3 -300\n").unwrap();
        assert_eq!(build(&lex, &matrix, 4, &out).unwrap(), (2, 3));
        assert_eq!(Alternates::open(&out).unwrap().entries("/").count(), 1);
        let a = Alternates::open(&out).unwrap();
        let e: Vec<Entry> = a.entries("今日").collect();
        assert_eq!(e.len(), 2);
        assert_eq!(
            e[1],
            Entry {
                left: 1,
                right: 2,
                cost: 1500,
                reading: "こんにち"
            }
        );
        // one reading: not kept
        assert_eq!(a.entries("天気").count(), 0);
        assert_eq!(a.entries("今").count(), 0);
        assert_eq!(a.connection(0, 1), -5);
        assert_eq!(a.connection(2, 0), 7);
        assert_eq!(a.connection(2, 3), -300);
        assert_eq!(a.connection(1, 1), 0);
        assert_eq!(a.connection(9, 1), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn features_sum_per_template() {
        let dir = dir("bigram");
        let lex = dir.join("lex.csv");
        let out = dir.join("alternates.bin");
        std::fs::write(&lex, LEX).unwrap();
        // two templates; a feature with a comma is quoted
        std::fs::write(
            dir.join("bigram.right"),
            "1\tB1:名詞,\"B2:名詞,数詞\"\n2\tB1:助詞,∅NONE\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("bigram.left"),
            "1\t名詞,\"名詞,数詞\"\n2\t助詞,∅NONE\n3\t名詞,*\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("bigram.cost"),
            "/\t0\nB1:名詞/名詞\t10\nB2:名詞,数詞/名詞,数詞\t-3\nB1:助詞/名詞\t100\n/助詞\t-40\nB1:名詞/\t7\n",
        )
        .unwrap();
        build(&lex, &dir, 4, &out).unwrap();
        let a = Alternates::open(&out).unwrap();
        assert_eq!(a.entries("今日").count(), 2);
        assert_eq!(a.connection(1, 1), 7);
        assert_eq!(a.connection(2, 1), 100);
        assert_eq!(a.connection(1, 3), 10);
        // a sentence's start: the empty feature at every template
        assert_eq!(a.connection(0, 2), -40);
        assert_eq!(a.connection(1, 0), 7);
        assert_eq!(a.connection(2, 2), 0);
        assert_eq!(a.connection(9, 1), 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
