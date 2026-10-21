//! Dictionary-free alignment of a converted surface to its reading, shared
//! by the learning (word choices) and the segmentation of a conversion.
//!
//! A kana (either script) must match its reading char, a symbol matches any
//! reading symbol, and a run of other characters (kanji, digits, latin)
//! takes 1 to 4 reading chars per character.

/// Reading chars a run of `n` free characters may take at most.
const MAX_READING_PER_CHAR: usize = 4;

/// ヶ and ヵ (and their hiragana) are not: in 2ヶ月 and 鎌ヶ谷 they are
/// read か / が / こ, as a kanji is.
pub(crate) fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3094}' | '\u{30A1}'..='\u{30F4}' | '\u{30F7}'..='\u{30FA}' | 'ー')
}

pub(crate) fn to_hira(c: char) -> char {
    match c {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
        _ => c,
    }
}

/// Kanji, digits, latin: characters whose reading is not written out.
pub(crate) fn is_free(c: char) -> bool {
    !is_kana(c) && c.is_alphanumeric()
}

fn anchor_matches(s: char, r: char) -> bool {
    if is_kana(s) {
        // ケ is が in place names (栗ケ沢)
        is_kana(r) && (to_hira(s) == to_hira(r) || (s == 'ケ' && r == 'が'))
    } else {
        // A symbol matches any reading symbol: its width or style (、 / ，)
        // may differ after post-processing.
        !is_kana(r) && !r.is_alphanumeric()
    }
}

struct Aligner<'a> {
    s: &'a [char],
    r: &'a [char],
    memo: Vec<Option<bool>>,
    /// Reading end chosen for the free run starting at (i, j).
    choice: Vec<usize>,
}

impl Aligner<'_> {
    fn idx(&self, i: usize, j: usize) -> usize {
        i * (self.r.len() + 1) + j
    }

    fn go(&mut self, i: usize, j: usize) -> bool {
        if i == self.s.len() {
            return j == self.r.len();
        }
        let at = self.idx(i, j);
        if let Some(ok) = self.memo[at] {
            return ok;
        }
        let ok = if !is_free(self.s[i]) {
            j < self.r.len() && anchor_matches(self.s[i], self.r[j]) && self.go(i + 1, j + 1)
        } else {
            let k = (i..self.s.len())
                .find(|&k| !is_free(self.s[k]))
                .unwrap_or(self.s.len());
            let max = self.r.len().min(j + MAX_READING_PER_CHAR * (k - i));
            // Shortest reading first.
            let mut found = false;
            for l in j + 1..=max {
                if self.go(k, l) {
                    self.choice[at] = l;
                    found = true;
                    break;
                }
            }
            found
        };
        self.memo[at] = Some(ok);
        ok
    }
}

/// The reading range of every surface char (a free run's chars share the
/// run's range), or `None` when the surface does not fit the reading.
/// Whether `text` spells `reading` out: no kana out of place, no reading
/// char left with nothing emitted for it (ゼロプラモデル does not, for
/// ぜろせんぷらもでる).
pub(crate) fn spells_out(text: &str, reading: &[char]) -> bool {
    align(&text.chars().collect::<Vec<_>>(), reading).is_some()
}

pub(crate) fn align(s: &[char], r: &[char]) -> Option<Vec<(usize, usize)>> {
    let cells = (s.len() + 1) * (r.len() + 1);
    let mut a = Aligner {
        s,
        r,
        memo: vec![None; cells],
        choice: vec![0; cells],
    };
    if !a.go(0, 0) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len());
    let (mut i, mut j) = (0, 0);
    while i < s.len() {
        if !is_free(s[i]) {
            out.push((j, j + 1));
            i += 1;
            j += 1;
        } else {
            let k = (i..s.len()).find(|&k| !is_free(s[k])).unwrap_or(s.len());
            let l = a.choice[a.idx(i, j)];
            out.extend(std::iter::repeat_n((j, l), k - i));
            i = k;
            j = l;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spells(text: &str, reading: &str) -> bool {
        spells_out(text, &reading.chars().collect::<Vec<_>>())
    }

    #[test]
    fn a_lost_char_or_a_kana_out_of_place_does_not_spell_the_reading() {
        assert!(spells("ゼロ戦プラモデル", "ぜろせんぷらもでる"));
        assert!(!spells("ゼロプラモデル", "ぜろせんぷらもでる"));
        assert!(!spells("和えたたあん", "あえたたくあん"));
        // ヶ is read as a kanji is, ケ may be が
        assert!(spells("2ヶ月", "2かげつ"));
        assert!(spells("鎌ヶ谷", "かまがや"));
        assert!(spells("栗ケ沢", "くりがさわ"));
    }
}
