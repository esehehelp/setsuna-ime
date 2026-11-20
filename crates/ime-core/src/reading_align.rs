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

pub(crate) fn anchor_matches(s: char, r: char) -> bool {
    if is_kana(s) {
        // ケ is が in place names (栗ケ沢)
        is_kana(r) && (to_hira(s) == to_hira(r) || (s == 'ケ' && r == 'が'))
    } else {
        // A symbol matches any reading symbol: its width or style (、 / ，)
        // may differ after post-processing.
        !is_kana(r) && !r.is_alphanumeric()
    }
}

/// The table of a surface and a reading at most this big (chars of one
/// plus one, times of the other plus one): longer ones are not aligned.
const MAX_CELLS: usize = 1 << 22;

/// Whether the surface from char `i` on fits the reading from char `j` on,
/// for every (i, j), filled from the end (a table, not a recursion: a long
/// text ran the host app out of stack); and for a free run starting at
/// (i, j), the reading end it takes, the shortest that fits.
struct Aligner {
    width: usize,
    ok: Vec<bool>,
    choice: Vec<usize>,
}

impl Aligner {
    fn new(s: &[char], r: &[char]) -> Option<Self> {
        let width = r.len() + 1;
        let cells = (s.len() + 1).checked_mul(width)?;
        if cells > MAX_CELLS {
            return None;
        }
        let mut a = Aligner {
            width,
            ok: vec![false; cells],
            choice: vec![0; cells],
        };
        a.ok[s.len() * width + r.len()] = true;
        // where the free run at each char ends
        let mut run_end = vec![s.len(); s.len() + 1];
        for i in (0..s.len()).rev() {
            run_end[i] = if is_free(s[i]) { run_end[i + 1] } else { i };
        }
        for i in (0..s.len()).rev() {
            // only a free run's start is asked about
            if is_free(s[i]) && i > 0 && is_free(s[i - 1]) {
                continue;
            }
            for j in 0..=r.len() {
                let at = a.idx(i, j);
                if !is_free(s[i]) {
                    a.ok[at] =
                        j < r.len() && anchor_matches(s[i], r[j]) && a.ok[a.idx(i + 1, j + 1)];
                } else {
                    let k = run_end[i];
                    let max = r.len().min(j + MAX_READING_PER_CHAR * (k - i));
                    // Shortest reading first.
                    if let Some(l) = (j + 1..=max).find(|&l| a.ok[a.idx(k, l)]) {
                        a.ok[at] = true;
                        a.choice[at] = l;
                    }
                }
            }
        }
        Some(a)
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.width + j
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
    let a = Aligner::new(s, r)?;
    if !a.ok[0] {
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

    #[test]
    fn a_long_text_aligns_without_recursion_and_a_huge_one_not_at_all() {
        let text = "東京にいく".repeat(300);
        assert!(spells(&text, &"とうきょうにいく".repeat(300)));
        let huge = "あ".repeat(100_000);
        assert!(!spells(&huge, &huge));
    }
}
