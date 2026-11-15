use bincode::{Decode, Encode};

use crate::errors::{Result, VibratoError};

/// Words per stored offset.
const STEP: usize = 64;

/// The words' features in one string (setsuna-ime: upstream has a `String`
/// per word): the offset of every `STEP`th word, and each word's end from
/// the offset of its block in two bytes.
#[derive(Default, Decode, Encode)]
pub struct WordFeatures {
    text: String,
    ends: Vec<u16>,
    marks: Vec<u32>,
}

impl WordFeatures {
    /// Fails where `STEP` words' features run over 64 KiB, or all of them
    /// over 4 GiB.
    pub fn new<I, S>(features: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut text = String::new();
        let mut ends = Vec::new();
        let mut marks = Vec::new();
        let mut block = 0;
        for (i, s) in features.into_iter().enumerate() {
            if i % STEP == 0 {
                block = text.len();
                marks.push(u32::try_from(block)?);
            }
            text.push_str(s.as_ref());
            ends.push(u16::try_from(text.len() - block).map_err(|_| {
                VibratoError::invalid_argument(
                    "features",
                    format!("over 64 KiB in words {i} and the {STEP} before"),
                )
            })?);
        }
        text.shrink_to_fit();
        ends.shrink_to_fit();
        marks.shrink_to_fit();
        Ok(Self { text, ends, marks })
    }

    #[inline(always)]
    pub fn get(&self, word_id: usize) -> &str {
        let base = self.marks[word_id / STEP] as usize;
        let start = if word_id % STEP == 0 {
            0
        } else {
            usize::from(self.ends[word_id - 1])
        };
        &self.text[base + start..base + usize::from(self.ends[word_id])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_come_back_by_word_id() {
        let words: Vec<String> = (0..200)
            .map(|i| format!("f{i},名詞{}", "x".repeat(i * 3)))
            .collect();
        let f = WordFeatures::new(&words).unwrap();
        for (i, w) in words.iter().enumerate() {
            assert_eq!(f.get(i), w);
        }
        assert!(WordFeatures::new(["x".repeat(70_000)]).is_err());
        assert_eq!(WordFeatures::new([""]).unwrap().get(0), "");
    }
}
