//! A SentencePiece Unigram tokenizer read from its `tokenizer.model`, for an
//! LM whose GGUF vocabulary llama.cpp segments otherwise: a Unigram model
//! converts to a llama-type (score-merge) vocabulary, which cuts Japanese
//! differently from the model's own segmentation and costs the reranker most
//! of its gain (2026-10-23: on one such LM, AJIMEE EM1 +2.5pt with llama.cpp's
//! tokens, about +6pt with SentencePiece's).
//!
//! The encoding is SentencePiece's `EncodeOptimized` (unigram_model.cc): the
//! best-scoring path over the pieces that match, a char no piece covers taken
//! as the unknown piece at the lowest normal score minus 10, then each unknown
//! piece written as the byte pieces of its UTF-8 bytes (byte fallback). Scores
//! add up in f32 and ties keep the first path found, as there. Only the
//! identity normalizer is read (no charsmap, no dummy prefix, no whitespace
//! folding; a space becomes U+2581 when the model escapes whitespace); any
//! other model is refused.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

const NORMAL: u64 = 1;
const UNKNOWN: u64 = 2;
const USER_DEFINED: u64 = 4;
const BYTE: u64 = 6;
const UNIGRAM: u64 = 1;
const UNK_PENALTY: f32 = 10.0;

pub struct SentencePiece {
    /// matchable piece (normal or user-defined) -> (id, score, user-defined)
    pieces: HashMap<Vec<u8>, (u32, f32, bool)>,
    /// longest matchable piece, in bytes
    max_len: usize,
    unk_id: u32,
    unk_score: f32,
    max_score: f32,
    /// byte -> its piece `<0xXX>`, with byte fallback
    bytes: Option<[u32; 256]>,
    escape_whitespaces: bool,
}

/// A protobuf reader for the few fields of `ModelProto` read here.
struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.at).context("truncated varint")?;
            self.at += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                return Ok(v);
            }
        }
        bail!("varint too long")
    }

    fn next(&mut self) -> Result<Option<(u64, Value<'a>)>> {
        if self.at >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                ensure!(self.at + 8 <= self.buf.len(), "truncated fixed64");
                self.at += 8;
                Value::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                ensure!(self.at + n <= self.buf.len(), "truncated field");
                let b = &self.buf[self.at..self.at + n];
                self.at += n;
                Value::Bytes(b)
            }
            5 => {
                ensure!(self.at + 4 <= self.buf.len(), "truncated fixed32");
                let b = &self.buf[self.at..self.at + 4];
                self.at += 4;
                Value::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            t => bail!("unsupported wire type {t}"),
        };
        Ok(Some((key >> 3, value)))
    }
}

fn fields(buf: &[u8]) -> Reader<'_> {
    Reader { buf, at: 0 }
}

impl SentencePiece {
    pub fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&buf).with_context(|| format!("SentencePiece model {}", path.display()))
    }

    pub fn parse(buf: &[u8]) -> Result<Self> {
        // ModelProto: 1 pieces, 2 trainer_spec, 3 normalizer_spec
        let mut list: Vec<(Vec<u8>, f32, u64)> = Vec::new();
        let (mut model_type, mut byte_fallback) = (UNIGRAM, false);
        // NormalizerSpec defaults (proto2): add_dummy_prefix, remove_extra_whitespaces and escape_whitespaces true
        let (mut charsmap, mut dummy, mut fold, mut escape) = (false, true, true, true);
        let mut r = fields(buf);
        while let Some((field, value)) = r.next()? {
            match (field, value) {
                (1, Value::Bytes(b)) => {
                    // SentencePiece: 1 piece, 2 score, 3 type (default NORMAL)
                    let (mut piece, mut score, mut kind) = (Vec::new(), 0.0f32, NORMAL);
                    let mut p = fields(b);
                    while let Some((f, v)) = p.next()? {
                        match (f, v) {
                            (1, Value::Bytes(s)) => piece = s.to_vec(),
                            (2, Value::Fixed32(x)) => score = f32::from_bits(x),
                            (3, Value::Varint(t)) => kind = t,
                            _ => {}
                        }
                    }
                    list.push((piece, score, kind));
                }
                (2, Value::Bytes(b)) => {
                    // TrainerSpec: 3 model_type, 35 byte_fallback
                    let mut t = fields(b);
                    while let Some((f, v)) = t.next()? {
                        match (f, v) {
                            (3, Value::Varint(x)) => model_type = x,
                            (35, Value::Varint(x)) => byte_fallback = x != 0,
                            _ => {}
                        }
                    }
                }
                (3, Value::Bytes(b)) => {
                    // NormalizerSpec: 2 precompiled_charsmap, 3 add_dummy_prefix, 4 remove_extra_whitespaces,
                    // 5 escape_whitespaces
                    let mut n = fields(b);
                    while let Some((f, v)) = n.next()? {
                        match (f, v) {
                            (2, Value::Bytes(m)) => charsmap = !m.is_empty(),
                            (3, Value::Varint(x)) => dummy = x != 0,
                            (4, Value::Varint(x)) => fold = x != 0,
                            (5, Value::Varint(x)) => escape = x != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        ensure!(
            model_type == UNIGRAM,
            "not a Unigram model (type {model_type})"
        );
        ensure!(
            !charsmap && !dummy && !fold,
            "only the identity normalizer is read (charsmap {charsmap}, dummy prefix {dummy}, whitespace folding {fold})"
        );
        let mut pieces = HashMap::new();
        let (mut min_score, mut max_score) = (f32::MAX, f32::MIN);
        let mut unk_id = None;
        let mut bytes = [u32::MAX; 256];
        for (id, (piece, score, kind)) in list.iter().enumerate() {
            let id = id as u32;
            match *kind {
                NORMAL => {
                    min_score = min_score.min(*score);
                    max_score = max_score.max(*score);
                    pieces.insert(piece.clone(), (id, *score, false));
                }
                USER_DEFINED => {
                    pieces.insert(piece.clone(), (id, 0.0, true));
                }
                UNKNOWN => unk_id = Some(id),
                BYTE => {
                    let hex = std::str::from_utf8(piece)
                        .ok()
                        .and_then(|s| s.strip_prefix("<0x")?.strip_suffix('>'))
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                        .with_context(|| {
                            format!("byte piece {:?}", String::from_utf8_lossy(piece))
                        })?;
                    bytes[hex as usize] = id;
                }
                _ => {}
            }
        }
        ensure!(!pieces.is_empty(), "no pieces");
        let unk_id = unk_id.context("no unknown piece")?;
        ensure!(
            !byte_fallback || bytes.iter().all(|&b| b != u32::MAX),
            "byte fallback without all 256 byte pieces"
        );
        let max_len = pieces.keys().map(Vec::len).max().unwrap_or(1);
        Ok(Self {
            pieces,
            max_len,
            unk_id,
            unk_score: min_score - UNK_PENALTY,
            max_score,
            bytes: byte_fallback.then_some(bytes),
            escape_whitespaces: escape,
        })
    }

    /// The ids of `text`, no BOS / EOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = if self.escape_whitespaces {
            text.replace(' ', "\u{2581}")
        } else {
            text.to_string()
        };
        let s = text.as_bytes();
        let n = s.len();
        // best path ending at each byte offset: (score, start, id); start usize::MAX = none yet
        let mut best: Vec<(f32, usize, u32)> = vec![(0.0, usize::MAX, 0); n + 1];
        best[0].1 = 0;
        let mut at = 0;
        while at < n {
            let here = best[at].0;
            let mblen = text[at..].chars().next().map_or(1, char::len_utf8);
            let mut single = false;
            let mut end = at;
            for ch in text[at..].chars() {
                end += ch.len_utf8();
                if end - at > self.max_len {
                    break;
                }
                if let Some(&(id, score, user)) = self.pieces.get(&s[at..end]) {
                    let score = if user {
                        (end - at) as f32 * self.max_score - 0.1
                    } else {
                        score
                    };
                    let cand = score + here;
                    let t = &mut best[end];
                    if t.1 == usize::MAX || cand > t.0 {
                        *t = (cand, at, id);
                    }
                    single |= end - at == mblen;
                }
            }
            if !single {
                let cand = self.unk_score + here;
                let t = &mut best[at + mblen];
                if t.1 == usize::MAX || cand > t.0 {
                    *t = (cand, at, self.unk_id);
                }
            }
            at += mblen;
        }
        let mut path: Vec<(usize, usize, u32)> = Vec::new();
        let mut end = n;
        while end > 0 {
            let (_, start, id) = best[end];
            path.push((start, end, id));
            end = start;
        }
        path.reverse();
        let mut ids = Vec::with_capacity(path.len());
        for (start, end, id) in path {
            match (&self.bytes, id == self.unk_id) {
                (Some(bytes), true) => ids.extend(s[start..end].iter().map(|&b| bytes[b as usize])),
                _ => ids.push(id),
            }
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ModelProto with these pieces (piece, score, type), the given trainer / normalizer flags.
    fn model(pieces: &[(&str, f32, u64)], byte_fallback: bool) -> Vec<u8> {
        fn varint(out: &mut Vec<u8>, mut v: u64) {
            while v >= 0x80 {
                out.push((v as u8) | 0x80);
                v >>= 7;
            }
            out.push(v as u8);
        }
        fn bytes_field(out: &mut Vec<u8>, field: u64, b: &[u8]) {
            varint(out, field << 3 | 2);
            varint(out, b.len() as u64);
            out.extend_from_slice(b);
        }
        let mut out = Vec::new();
        for (p, s, t) in pieces {
            let mut m = Vec::new();
            bytes_field(&mut m, 1, p.as_bytes());
            varint(&mut m, 2 << 3 | 5);
            m.extend_from_slice(&s.to_le_bytes());
            varint(&mut m, 3 << 3);
            varint(&mut m, *t);
            bytes_field(&mut out, 1, &m);
        }
        let mut t = Vec::new();
        varint(&mut t, 3 << 3);
        varint(&mut t, UNIGRAM);
        varint(&mut t, 35 << 3);
        varint(&mut t, byte_fallback as u64);
        bytes_field(&mut out, 2, &t);
        let mut n = Vec::new();
        for f in [3u64, 4] {
            varint(&mut n, f << 3);
            varint(&mut n, 0);
        }
        bytes_field(&mut out, 3, &n);
        out
    }

    fn with_bytes(mut pieces: Vec<(String, f32, u64)>) -> Vec<(String, f32, u64)> {
        pieces.extend((0..=255u32).map(|b| (format!("<0x{b:02X}>"), 0.0, BYTE)));
        pieces
    }

    fn sp(pieces: &[(String, f32, u64)], byte_fallback: bool) -> SentencePiece {
        let refs: Vec<(&str, f32, u64)> = pieces
            .iter()
            .map(|(p, s, t)| (p.as_str(), *s, *t))
            .collect();
        SentencePiece::parse(&model(&refs, byte_fallback)).unwrap()
    }

    #[test]
    fn takes_the_best_scoring_path() {
        let p: Vec<(String, f32, u64)> = [
            ("<unk>", 0.0, UNKNOWN),
            ("東", -5.0, NORMAL),
            ("京", -5.0, NORMAL),
            ("東京", -3.0, NORMAL),
            ("都", -4.0, NORMAL),
            ("京都", -2.5, NORMAL),
        ]
        .iter()
        .map(|(a, b, c)| (a.to_string(), *b, *c))
        .collect();
        let m = sp(&p, false);
        // 東京|都 (-7.0) over 東|京都 (-7.5) and 東|京|都
        assert_eq!(m.encode("東京都"), vec![3, 4]);
        assert_eq!(m.encode("京都"), vec![5]);
    }

    #[test]
    fn an_unknown_char_falls_back_to_its_bytes() {
        let p = with_bytes(vec![
            ("<unk>".into(), 0.0, UNKNOWN),
            ("あ".into(), -1.0, NORMAL),
        ]);
        let m = sp(&p, true);
        // 鵺 = E9 B5 BA; the byte pieces follow the two pieces above
        assert_eq!(m.encode("あ鵺"), vec![1, 2 + 0xE9, 2 + 0xB5, 2 + 0xBA]);
        let m = sp(&p, false);
        assert_eq!(m.encode("あ鵺"), vec![1, 0]);
    }

    #[test]
    fn a_space_is_escaped() {
        let p = with_bytes(vec![
            ("<unk>".into(), 0.0, UNKNOWN),
            ("a".into(), -1.0, NORMAL),
            ("\u{2581}".into(), -1.0, NORMAL),
        ]);
        assert_eq!(sp(&p, true).encode("a a"), vec![1, 2, 1]);
    }

    /// Against SentencePiece itself: SETSUNA_SPM_MODEL (a tokenizer.model) and SETSUNA_SPM_CHECK (lines
    /// `text<TAB>space-separated ids`, from Python's sentencepiece); skipped when unset.
    #[test]
    fn matches_sentencepiece_on_a_reference_file() {
        let (Some(model), Some(check)) = (
            std::env::var_os("SETSUNA_SPM_MODEL"),
            std::env::var_os("SETSUNA_SPM_CHECK"),
        ) else {
            eprintln!("skip: SETSUNA_SPM_MODEL / SETSUNA_SPM_CHECK unset");
            return;
        };
        let m = SentencePiece::load(Path::new(&model)).unwrap();
        let text = std::fs::read_to_string(&check).unwrap();
        let (mut n, mut bad) = (0, Vec::new());
        for line in text.lines() {
            let Some((t, ids)) = line.split_once('\t') else {
                continue;
            };
            let want: Vec<u32> = ids
                .split(' ')
                .filter(|x| !x.is_empty())
                .map(|x| x.parse().unwrap())
                .collect();
            n += 1;
            if m.encode(t) != want && bad.len() < 10 {
                bad.push(format!("{t:?}: {:?} want {want:?}", m.encode(t)));
            }
        }
        assert!(bad.is_empty(), "{n} lines, first mismatches: {bad:#?}");
        eprintln!("{n} lines match");
    }
}
