//! SharedCharTokenizer with `.vocab.hex.tsv` sidecar loader.
//!
//! Vocab format: one token per line, `id\thex(utf8_bytes)`, as written by the
//! export pipeline.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SharedCharTokenizer {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub pad_id: u32,
    pub unk_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub blank_id: u32,
    pub mask_id: u32,
    /// The chars of the vocabulary the training rows were written in
    /// (manifest `input_vocab`), when the model's keeps part of it.
    pub input_chars: Option<HashSet<String>>,
}

impl SharedCharTokenizer {
    pub fn load_vocab_hex_tsv(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let reader = BufReader::new(f);

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut max_id: i64 = -1;
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let (id_str, hex_str) = match line.split_once('\t') {
                Some(pair) => pair,
                None => continue,
            };
            let id: u32 = id_str.parse()?;
            let token = decode_hex_bytes(hex_str)?;
            token_to_id.insert(token, id);
            if id as i64 > max_id {
                max_id = id as i64;
            }
        }
        anyhow::ensure!(max_id >= 0, "no tokens parsed from {}", path.display());

        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (tok, id) in &token_to_id {
            id_to_token[*id as usize] = tok.clone();
        }

        let lookup =
            |name: &str, default: u32| -> u32 { token_to_id.get(name).copied().unwrap_or(default) };
        // Default IDs follow the SEP=2 < CLS=3 vocab convention. The
        // lookup-by-name covers the common case; defaults only kick in for
        // pathological vocab files missing one of the special tokens.
        Ok(Self {
            pad_id: lookup("[PAD]", 0),
            unk_id: lookup("[UNK]", 1),
            sep_id: lookup("[SEP]", 2),
            cls_id: lookup("[CLS]", 3),
            blank_id: lookup("[BLANK]", 4),
            mask_id: lookup("[MASK]", 5),
            id_to_token,
            token_to_id,
            input_chars: None,
        })
    }

    /// Read the input vocabulary (manifest `input_vocab`): its tokens other
    /// than specials and byte tokens, the chars a training row held as one
    /// token each.
    pub fn with_input_vocab(mut self, path: &Path) -> Result<Self> {
        let input = Self::load_vocab_hex_tsv(path)?;
        self.input_chars = Some(
            input
                .token_to_id
                .into_keys()
                .filter(|t| {
                    !(t.starts_with('[') && t.ends_with(']')) && parse_byte_token(t).is_none()
                })
                .collect(),
        );
        Ok(self)
    }

    /// Whether `c` is a token of its own. A char that is not would reach the
    /// model as `[UNK]` (or as byte-fallback tokens, when the vocab has them).
    pub fn has_char(&self, c: char) -> bool {
        let mut buf = [0u8; 4];
        self.token_to_id
            .contains_key(c.encode_utf8(&mut buf) as &str)
    }

    /// Whether `c` reaches the model as `[UNK]`: no token of its own and no
    /// byte-fallback tokens for all its bytes.
    pub fn is_unknown(&self, c: char) -> bool {
        let mut buf = [0u8; 4];
        self.encode_with_byte_fallback(c.encode_utf8(&mut buf))
            .contains(&self.unk_id)
    }

    pub fn encode_char(&self, ch: &str) -> u32 {
        self.token_to_id.get(ch).copied().unwrap_or(self.unk_id)
    }

    /// One char's token ids as the training rows read it
    /// (tests/fixtures/model_input): its own id
    /// when the vocabulary has it; else, with an input vocabulary (the rows
    /// written in a larger one, mapped to this one with `[UNK]` for what it
    /// lacks), one `[UNK]` for a char of the input vocabulary and one token
    /// per UTF-8 byte for any other (the row's byte tokens: a byte token of
    /// this vocabulary or `[UNK]`); without one, the byte tokens when this
    /// vocabulary has them, else one `[UNK]`.
    pub fn encode_with_byte_fallback(&self, ch: &str) -> Vec<u32> {
        if let Some(&id) = self.token_to_id.get(ch) {
            return vec![id];
        }
        let bytes = match &self.input_chars {
            Some(chars) => !chars.contains(ch),
            None => self.token_to_id.contains_key("<0x00>"),
        };
        if !bytes {
            return vec![self.unk_id];
        }
        let mut out = Vec::with_capacity(ch.len());
        for b in ch.as_bytes() {
            let key = format!("<0x{:02X}>", b);
            out.push(self.token_to_id.get(&key).copied().unwrap_or(self.unk_id));
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut byte_buffer: Vec<u8> = Vec::new();
        for &id in ids {
            if matches!(
                id,
                x if x == self.pad_id
                    || x == self.unk_id
                    || x == self.cls_id
                    || x == self.sep_id
                    || x == self.blank_id
                    || x == self.mask_id
            ) {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(id as usize) {
                if let Some(byte) = parse_byte_token(tok) {
                    byte_buffer.push(byte);
                    continue;
                }
                flush_byte_buffer(&mut out, &mut byte_buffer);
                out.push_str(tok);
            }
        }
        flush_byte_buffer(&mut out, &mut byte_buffer);
        out
    }

    /// `decode` plus, for each char of the text, the entry of `frames` (one
    /// per id) of the token it came from. A char spelled in byte-fallback
    /// tokens takes the frame of its first byte.
    pub fn decode_aligned(&self, ids: &[u32], frames: &[u32]) -> (String, Vec<u32>) {
        let mut out = String::new();
        let mut pos: Vec<u32> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        let mut byte_frames: Vec<u32> = Vec::new();
        let flush = |out: &mut String,
                     pos: &mut Vec<u32>,
                     bytes: &mut Vec<u8>,
                     byte_frames: &mut Vec<u32>| {
            match std::str::from_utf8(bytes) {
                Ok(text) => {
                    out.push_str(text);
                    pos.extend(text.char_indices().map(|(i, _)| byte_frames[i]));
                }
                Err(_) => {
                    // Match decode's invalid-run sentinel and its first byte.
                    out.push('\u{3013}');
                    pos.push(byte_frames.first().copied().unwrap_or(0));
                }
            }
            bytes.clear();
            byte_frames.clear();
        };
        for (i, &id) in ids.iter().enumerate() {
            let frame = frames.get(i).copied().unwrap_or(0);
            if [
                self.pad_id,
                self.unk_id,
                self.cls_id,
                self.sep_id,
                self.blank_id,
                self.mask_id,
            ]
            .contains(&id)
            {
                continue;
            }
            let Some(tok) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if let Some(byte) = parse_byte_token(tok) {
                bytes.push(byte);
                byte_frames.push(frame);
                continue;
            }
            flush(&mut out, &mut pos, &mut bytes, &mut byte_frames);
            out.push_str(tok);
            pos.extend(std::iter::repeat_n(frame, tok.chars().count()));
        }
        flush(&mut out, &mut pos, &mut bytes, &mut byte_frames);
        (out, pos)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn flush_byte_buffer(out: &mut String, byte_buffer: &mut Vec<u8>) {
    if byte_buffer.is_empty() {
        return;
    }
    match std::str::from_utf8(byte_buffer) {
        Ok(text) => out.push_str(text),
        Err(_) => out.push('\u{3013}'),
    }
    byte_buffer.clear();
}

fn decode_hex_bytes(hex: &str) -> Result<String> {
    anyhow::ensure!(hex.len().is_multiple_of(2), "odd-length hex");
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let b = hex.as_bytes();
    for i in (0..hex.len()).step_by(2) {
        let hi = hex_nibble(b[i])?;
        let lo = hex_nibble(b[i + 1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(String::from_utf8(bytes)?)
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => anyhow::bail!("bad hex byte {}", c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(tokens: &[&str]) -> SharedCharTokenizer {
        let id_to_token = tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let token_to_id = id_to_token
            .iter()
            .enumerate()
            .map(|(id, tok)| (tok.clone(), id as u32))
            .collect::<HashMap<_, _>>();
        SharedCharTokenizer {
            id_to_token,
            token_to_id,
            pad_id: 0,
            unk_id: 1,
            cls_id: 2,
            sep_id: 3,
            blank_id: 4,
            mask_id: 5,
            input_chars: None,
        }
    }

    #[test]
    fn has_char_is_a_single_token_lookup() {
        // byte fallback is on when the vocabulary has its byte tokens (as
        // the Python tokenizer: `<0x00>` present); `,` here lacks its own
        let t = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "あ", "<0x00>", "<0x2E>",
        ]);
        assert!(t.has_char('あ'));
        // a byte-fallback token does not make `.` a token of its own
        assert!(!t.has_char('.'));
        assert!(!t.has_char('['));
        // ... but it keeps it from being [UNK]
        assert!(!t.is_unknown('.'));
        assert!(t.is_unknown(','));
        assert!(!t.is_unknown('あ'));
        // without byte tokens an out-of-vocabulary char is one [UNK]
        let plain = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "あ",
        ]);
        assert_eq!(plain.encode_with_byte_fallback("頌"), [1]);
    }

    #[test]
    fn decode_aligned_gives_each_char_its_token_frame() {
        let tok = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "東京", "都", "<0xE9>",
            "<0xAB>", "<0x98>",
        ]);
        let (text, pos) = tok.decode_aligned(&[6, 4, 7, 8, 9, 10], &[1, 2, 3, 5, 6, 7]);
        assert_eq!(text, "東京都高");
        assert_eq!(pos, [1, 1, 3, 5]);
        assert_eq!(text, tok.decode(&[6, 4, 7, 8, 9, 10]));
    }

    #[test]
    fn adjacent_byte_characters_keep_their_first_byte_frames() {
        let tok = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "<0xE9>", "<0xAB>", "<0x98>",
            "<0xE6>", "<0xA9>", "<0x8B>", "A",
        ]);
        let ids = [6, 7, 8, 9, 10, 11, 12, 6];
        let (text, pos) = tok.decode_aligned(&ids, &[2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(text, tok.decode(&ids));
        assert_eq!(text, "高橋A〓");
        assert_eq!(pos, [2, 5, 8, 9]);
    }

    #[test]
    fn decode_byte_fallback_tokens_as_utf8() {
        let tok = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "<0x25>", "<0xE9>", "<0xAB>",
            "<0x98>",
        ]);
        assert_eq!(tok.decode(&[6, 7, 8, 9]), "%高");
    }

    #[test]
    fn decode_invalid_byte_run_as_sentinel_and_skips_unk() {
        let tok = tokenizer(&[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[BLANK]", "[MASK]", "<0xE9>", "A",
        ]);
        assert_eq!(tok.decode(&[1, 6, 7]), "\u{3013}A");
    }
}
