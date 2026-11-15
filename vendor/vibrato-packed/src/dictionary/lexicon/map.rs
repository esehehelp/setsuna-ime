pub mod posting;

use std::collections::BTreeMap;

use bincode::{
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};

use crate::dictionary::lexicon::map::posting::{Postings, PostingsBuilder};
use crate::errors::{Result, VibratoError};

/// A value with this bit set is an offset into the postings; without it, the
/// first word id << 8 | the number of ids, which follow one another.
const POSTED: u64 = 1 << 63;

/// The surfaces in an FST (setsuna-ime: upstream has a double-array trie
/// and a posting list for every surface), each mapped to its word ids: a run
/// of ids is the value itself (a lexicon sorted by surface has only those),
/// other ids are in the postings.
pub struct WordMap {
    fst: fst::raw::Fst<Vec<u8>>,
    postings: Postings,
}

fn fst_error(e: fst::Error) -> DecodeError {
    DecodeError::OtherString(e.to_string())
}

impl Encode for WordMap {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(self.fst.as_bytes(), encoder)?;
        Encode::encode(&self.postings, encoder)?;
        Ok(())
    }
}

impl<Context> Decode<Context> for WordMap {
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let data: Vec<u8> = Decode::decode(decoder)?;
        Ok(Self {
            fst: fst::raw::Fst::new(data).map_err(fst_error)?,
            postings: Decode::decode(decoder)?,
        })
    }
}

impl<'de, Context> BorrowDecode<'de, Context> for WordMap {
    fn borrow_decode<D: BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        Ok(Self {
            fst: fst::raw::Fst::new(data.to_vec()).map_err(fst_error)?,
            postings: BorrowDecode::borrow_decode(decoder)?,
        })
    }
}

impl WordMap {
    pub fn new<I, W>(words: I) -> Result<Self>
    where
        I: IntoIterator<Item = W>,
        W: AsRef<str>,
    {
        let mut b = WordMapBuilder::new();
        for (i, w) in words.into_iter().enumerate() {
            b.add_record(w.as_ref().to_string(), u32::try_from(i)?);
        }
        b.build()
    }

    /// The word ids of each surface `input` starts with, and the char the
    /// surface ends at.
    #[inline(always)]
    pub fn common_prefix_iterator<'a>(
        &'a self,
        input: &'a [char],
    ) -> impl Iterator<Item = (u32, usize)> + 'a {
        let mut found: Vec<(u32, usize)> = Vec::new();
        let mut node = self.fst.root();
        let mut acc = fst::raw::Output::zero();
        let mut buf = [0u8; 4];
        'chars: for (i, c) in input.iter().enumerate() {
            for &b in c.encode_utf8(&mut buf).as_bytes() {
                match node.find_input(b) {
                    Some(t) => {
                        let tr = node.transition(t);
                        acc = acc.cat(tr.out);
                        node = self.fst.node(tr.addr);
                    }
                    None => break 'chars,
                }
            }
            if node.is_final() {
                let v = acc.cat(node.final_output()).value();
                if v & POSTED != 0 {
                    let at = usize::try_from(v & !POSTED).unwrap();
                    found.extend(self.postings.ids(at).map(|id| (id, i + 1)));
                } else {
                    let first = u32::try_from(v >> 8).unwrap();
                    let n = u32::try_from(v & 0xff).unwrap();
                    found.extend((first..first + n).map(|id| (id, i + 1)));
                }
            }
        }
        found.into_iter()
    }
}

#[derive(Default)]
pub struct WordMapBuilder {
    map: BTreeMap<String, Vec<u32>>,
}

impl WordMapBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn add_record(&mut self, word: String, id: u32) {
        self.map.entry(word).or_default().push(id);
    }

    pub fn build(self) -> Result<WordMap> {
        let err = |e: fst::Error| VibratoError::invalid_argument("words", e.to_string());
        let mut postings = PostingsBuilder::new();
        let mut fst = fst::raw::Builder::memory();
        // a BTreeMap of Strings goes in byte order, as an FST takes its keys
        for (word, ids) in self.map {
            let run = ids.len() < 256 && ids.windows(2).all(|w| w[1] == w[0] + 1);
            let value = if run {
                u64::from(ids[0]) << 8 | ids.len() as u64
            } else {
                POSTED | u64::try_from(postings.push(&ids)?)?
            };
            fst.insert(word.as_bytes(), value).map_err(err)?;
        }
        let bytes = fst.into_inner().map_err(err)?;
        Ok(WordMap {
            fst: fst::raw::Fst::new(bytes).map_err(err)?,
            postings: postings.build(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::FromU32;

    #[test]
    fn surfaces_map_to_their_ids_runs_or_not() {
        // 東京 has ids 0 and 2 (posted), 京都 3 alone
        let map = WordMap::new(["東京", "東京都", "東京", "京都"]).unwrap();
        let input: Vec<char> = "東京都".chars().collect();
        let mut got: Vec<(u32, usize)> = map.common_prefix_iterator(&input).collect();
        got.sort();
        assert_eq!(got, [(0, 2), (1, 3), (2, 2)]);
        // sorted, as setsuna-ime builds its lexicons: runs only
        let map = WordMap::new(["京都", "東京", "東京", "東京都"]).unwrap();
        let mut got: Vec<(u32, usize)> = map.common_prefix_iterator(&input).collect();
        got.sort();
        assert_eq!(got, [(1, 2), (2, 2), (3, 3)]);
        assert_eq!(map.postings.len(), 0);
        let none: Vec<char> = "大阪".chars().collect();
        assert_eq!(map.common_prefix_iterator(&none).count(), 0);
        // FromU32 as upstream uses it
        assert_eq!(usize::from_u32(3), 3);
    }
}
