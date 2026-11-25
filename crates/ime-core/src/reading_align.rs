//! Dictionary-free alignment of a converted surface to its reading, shared
//! by the learning (word choices) and the segmentation of a conversion.
//!
//! A kana (either script) must match its reading char, a symbol matches any
//! reading symbol, and a run of other characters (kanji, digits, latin)
//! takes 1 to 4 reading chars per character. A letter or digit the reading
//! has as typed is that char of it, where the surface fits so (`literals`):
//! the ID of 農園ID is the ID of のうえんID, not part of a run that reads
//! のうえんID as a whole.

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

/// For each surface char, whether it is a letter or digit typed in the
/// reading (`to_half` of it the same char there): each one matched to the
/// next such char of the reading, in order; `None` when none is.
fn literals(s: &[char], r: &[char]) -> Option<Vec<bool>> {
    let mut out = vec![false; s.len()];
    let mut j = 0;
    for (i, &c) in s.iter().enumerate() {
        let c = crate::postprocess::to_half(c);
        if !c.is_ascii_alphanumeric() {
            continue;
        }
        if let Some(k) = (j..r.len()).find(|&k| r[k] == c) {
            out[i] = true;
            j = k + 1;
        }
    }
    out.contains(&true).then_some(out)
}

impl Aligner {
    /// `lit`: the surface chars that are their reading char as typed
    /// (`literals`), anchors rather than free.
    fn new(s: &[char], r: &[char], lit: &[bool]) -> Option<Self> {
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
        let free = |i: usize| is_free(s[i]) && !lit[i];
        // where the free run at each char ends
        let mut run_end = vec![s.len(); s.len() + 1];
        for i in (0..s.len()).rev() {
            run_end[i] = if free(i) { run_end[i + 1] } else { i };
        }
        for i in (0..s.len()).rev() {
            // only a free run's start is asked about
            if free(i) && i > 0 && free(i - 1) {
                continue;
            }
            for j in 0..=r.len() {
                let at = a.idx(i, j);
                if lit[i] {
                    a.ok[at] = j < r.len()
                        && crate::postprocess::to_half(s[i]) == r[j]
                        && a.ok[a.idx(i + 1, j + 1)];
                } else if !free(i) {
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
    if let Some(lit) = literals(s, r) {
        if let Some(out) = align_with(s, r, &lit) {
            return Some(out);
        }
    }
    align_with(s, r, &vec![false; s.len()])
}

/// For each surface char, the reading char it is where it is a letter or
/// digit typed in the reading (`literals`) and the surface aligns so; `None`
/// when it does not.
pub(crate) fn typed_chars(s: &[char], r: &[char]) -> Option<Vec<Option<usize>>> {
    let lit = literals(s, r)?;
    let ranges = align_with(s, r, &lit)?;
    Some(
        lit.iter()
            .zip(&ranges)
            .map(|(&l, &(a, _))| l.then_some(a))
            .collect(),
    )
}

/// `align`, the chars `lit` says as their reading chars.
fn align_with(s: &[char], r: &[char], lit: &[bool]) -> Option<Vec<(usize, usize)>> {
    let a = Aligner::new(s, r, lit)?;
    if !a.ok[0] {
        return None;
    }
    let free = |i: usize| is_free(s[i]) && !lit[i];
    let mut out = Vec::with_capacity(s.len());
    let (mut i, mut j) = (0, 0);
    while i < s.len() {
        if !free(i) {
            out.push((j, j + 1));
            i += 1;
            j += 1;
        } else {
            let k = (i..s.len()).find(|&k| !free(k)).unwrap_or(s.len());
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
    fn letters_and_digits_typed_are_their_own_reading_chars() {
        let r: Vec<char> = "{のうえんID}に".chars().collect();
        let s: Vec<char> = "{農園ID}に".chars().collect();
        // 農園 reads のうえん, I and D themselves, not one run over のうえんID
        assert_eq!(
            align(&s, &r).unwrap(),
            [(0, 1), (1, 5), (1, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        );
        // in full width too
        let s: Vec<char> = "{農園ＩＤ}に".chars().collect();
        assert_eq!(align(&s, &r).unwrap()[3..5], [(5, 6), (6, 7)]);
        // a digit the model wrote for kana is no typed one: 2 for に
        let r: Vec<char> = "にこと2こ".chars().collect();
        assert!(align(&"2個と2個".chars().collect::<Vec<_>>(), &r).is_some());
    }

    #[test]
    fn a_long_text_aligns_without_recursion_and_a_huge_one_not_at_all() {
        let text = "東京にいく".repeat(300);
        assert!(spells(&text, &"とうきょうにいく".repeat(300)));
        let huge = "あ".repeat(100_000);
        assert!(!spells(&huge, &huge));
    }
}
