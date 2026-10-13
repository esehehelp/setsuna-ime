//! Symbol token: models trained with the symbol token see every run of
//! symbols in the reading (and write it in the surface) as one abstract
//! token `◇`. The reading is fed to the model with each run as one `◇`, in
//! one pass, and the k-th run is put back at the k-th `◇` of every
//! candidate. Contexts keep their symbols as typed.
//!
//! What counts as a symbol (`is_symbol`, `is_symbol_at`) must match the
//! rules the training rows were written with; the tests hold their cases.

use std::collections::HashMap;

use crate::candidate::Candidate;

/// The token a run of symbols becomes.
pub const SYMBOL_TOKEN: char = '\u{25C7}'; // ◇

/// Not kana, a kanji, ー 々 〆 〇, Japanese punctuation 、。・「」『』 or an
/// ASCII letter / digit.
pub fn is_symbol(c: char) -> bool {
    !(matches!(
        c,
        'ぁ'..='ゖ' | 'ァ'..='ヺ' | 'ー' | '㐀'..='䶿' | '一'..='鿿' | '豈'..='\u{faff}' | '々' | '〆' | '〇'
    ) || "、。・「」『』".contains(c)
        || c.is_ascii_alphanumeric())
}

/// `is_symbol`, except `.` and `,` between two ASCII digits, which belong to
/// the number (0.89, 17,000).
pub fn is_symbol_at(chars: &[char], i: usize) -> bool {
    let c = chars[i];
    if matches!(c, '.' | ',')
        && i > 0
        && i + 1 < chars.len()
        && chars[i - 1].is_ascii_digit()
        && chars[i + 1].is_ascii_digit()
    {
        return false;
    }
    is_symbol(c)
}

/// A reading with every run of symbols as one [`SYMBOL_TOKEN`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenized {
    /// What the model reads.
    pub text: String,
    /// The runs of symbols, in order.
    pub runs: Vec<String>,
    /// For each char of `text`, the index of the reading char it stands for
    /// (a token: the first char of its run).
    origin: Vec<u32>,
    /// Where each run starts in the reading.
    run_starts: Vec<u32>,
}

/// `reading` with every run of symbols as one [`SYMBOL_TOKEN`], and the runs.
pub fn tokenize(reading: &str) -> Tokenized {
    let chars: Vec<char> = reading.chars().collect();
    let mut out = Tokenized {
        text: String::with_capacity(reading.len()),
        runs: Vec::new(),
        origin: Vec::with_capacity(chars.len()),
        run_starts: Vec::new(),
    };
    for (i, &c) in chars.iter().enumerate() {
        if !is_symbol_at(&chars, i) {
            out.text.push(c);
            out.origin.push(i as u32);
        } else if i > 0 && is_symbol_at(&chars, i - 1) {
            out.runs.last_mut().expect("a run is open").push(c);
        } else {
            out.text.push(SYMBOL_TOKEN);
            out.origin.push(i as u32);
            out.runs.push(c.to_string());
            out.run_starts.push(i as u32);
        }
    }
    out
}

impl Tokenized {
    /// The reading char a char of `text` stands for.
    pub fn origin(&self, index: u32) -> Option<u32> {
        self.origin.get(index as usize).copied()
    }

    /// `output` with its k-th [`SYMBOL_TOKEN`] replaced by the k-th run; a
    /// token past the runs is dropped (nothing to put back).
    pub fn restore(&self, output: &str) -> String {
        let mut out = String::with_capacity(output.len());
        let mut k = 0;
        for c in output.chars() {
            if c == SYMBOL_TOKEN {
                if let Some(run) = self.runs.get(k) {
                    out.push_str(run);
                }
                k += 1;
            } else {
                out.push(c);
            }
        }
        out
    }

    /// A candidate of the tokenized reading as one of the reading: its text
    /// restored, and its reading positions (indices into `text`) moved to
    /// the reading, a restored run's chars each at its own reading char (as
    /// symbol split places a spliced symbol). Positions stay empty when the
    /// candidate has none.
    pub fn restore_candidate(&self, c: &Candidate) -> Candidate {
        let n = c.text.chars().count();
        let aligned = c.reading_pos.len() == n && n > 0;
        let last = self.origin.last().copied().unwrap_or(0);
        let mut text = String::with_capacity(c.text.len());
        let mut pos = Vec::with_capacity(n);
        let mut k = 0;
        for (j, ch) in c.text.chars().enumerate() {
            if ch == SYMBOL_TOKEN {
                if let (Some(run), Some(&start)) = (self.runs.get(k), self.run_starts.get(k)) {
                    text.push_str(run);
                    pos.extend(start..start + run.chars().count() as u32);
                }
                k += 1;
            } else {
                text.push(ch);
                if aligned {
                    pos.push(self.origin(c.reading_pos[j]).unwrap_or(last));
                }
            }
        }
        Candidate {
            text,
            score: c.score,
            source: c.source,
            reading_pos: if aligned { pos } else { Vec::new() },
        }
    }

    /// Candidates of the tokenized reading restored, best first: an empty
    /// one is dropped, and of the ones that restore to the same text the
    /// best-scored is kept (at the place of the first; the input is best
    /// first, so that is the first).
    pub fn restore_candidates(&self, candidates: &[Candidate]) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = Vec::with_capacity(candidates.len());
        let mut seen: HashMap<String, usize> = HashMap::new();
        for c in candidates {
            let r = self.restore_candidate(c);
            if r.text.is_empty() {
                continue;
            }
            match seen.get(&r.text) {
                Some(&i) => {
                    if r.score > out[i].score {
                        out[i] = r;
                    }
                }
                None => {
                    seen.insert(r.text.clone(), out.len());
                    out.push(r);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::CandidateSource;

    fn tok(s: &str) -> (String, Vec<String>) {
        let t = tokenize(s);
        (t.text, t.runs)
    }

    fn runs(r: &[&str]) -> Vec<String> {
        r.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn runs_of_symbols_become_one_token_each() {
        assert_eq!(
            tok("価格は1,000円(税込)です"),
            ("価格は1,000円◇税込◇です".to_string(), runs(&["(", ")"]))
        );
        assert_eq!(tok("F1は0.89、精度は2.5%").0, "F1は0.89、精度は2.5◇");
        assert_eq!(tok("1.、2,"), ("1◇、2◇".to_string(), runs(&[".", ","])));
        assert_eq!(
            tok("C# scriptを"),
            ("C◇scriptを".to_string(), runs(&["# "]))
        );
        assert_eq!(
            tok("「方」、何。・ー々"),
            ("「方」、何。・ー々".to_string(), runs(&[]))
        );
        assert!(is_symbol(SYMBOL_TOKEN) && is_symbol('ゝ') && !is_symbol('〆'));
    }

    #[test]
    fn a_run_is_cut_by_a_number_separator_only_between_digits() {
        // `.` after a symbol is in its run; `,` between digits is not.
        assert_eq!(tok("3.5%"), ("3.5◇".to_string(), runs(&["%"])));
        assert_eq!(tok("!?.1"), ("◇1".to_string(), runs(&["!?."])));
        assert_eq!(tok("1,2,3"), ("1,2,3".to_string(), runs(&[])));
    }

    #[test]
    fn restore_puts_each_run_back_and_drops_extra_tokens() {
        let text = "かかく(ぜいこみ)です";
        let t = tokenize(text);
        assert_eq!(t.restore(&t.text), text);
        assert_eq!(t.restore("価格◇税込◇です"), "価格(税込)です");
        assert_eq!(t.restore("価格◇税込です◇◇"), "価格(税込です)");
        assert_eq!(tokenize("").restore("記号なし"), "記号なし");
        assert_eq!(tokenize("きごう").restore("記◇号"), "記号");
    }

    fn cand(text: &str, score: f32, pos: &[u32]) -> Candidate {
        Candidate {
            text: text.to_string(),
            score,
            source: CandidateSource::Beam,
            reading_pos: pos.to_vec(),
        }
    }

    #[test]
    fn restored_candidates_carry_reading_positions_of_the_reading() {
        // reading か く ( ( ぜ い ) -> か く ◇ ぜ い ◇
        let t = tokenize("かく((ぜい)");
        assert_eq!(t.text, "かく◇ぜい◇");
        // positions in the tokenized text: 価 at く (1), 税 at ぜ (3)
        let c = t.restore_candidate(&cand("価◇税◇", -1.0, &[1, 2, 3, 5]));
        assert_eq!(c.text, "価((税)");
        assert_eq!(c.reading_pos, [1, 2, 3, 4, 6]);
        // unknown positions stay unknown
        let c = t.restore_candidate(&cand("価◇税◇", -1.0, &[]));
        assert!(c.reading_pos.is_empty());
        // the reading itself (a fallback) aligns char for char
        let c = t.restore_candidate(&cand(&t.text, 0.0, &[0, 1, 2, 3, 4, 5]));
        assert_eq!(c.text, "かく((ぜい)");
        assert_eq!(c.reading_pos, [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn duplicates_after_restore_keep_the_best_score_and_empty_ones_go() {
        let t = tokenize("あ!");
        let out = t.restore_candidates(&[
            cand("亜◇", -1.0, &[0, 1]),
            cand("◇", -1.5, &[1]),
            cand("亜◇◇", -2.0, &[0, 1, 1]),
            cand("阿◇", -3.0, &[0, 1]),
        ]);
        let got: Vec<_> = out.iter().map(|c| (c.text.as_str(), c.score)).collect();
        assert_eq!(got, [("亜!", -1.0), ("!", -1.5), ("阿!", -3.0)]);
        let t = tokenize("あ");
        let out = t.restore_candidates(&[cand("◇", -1.0, &[0]), cand("亜", -2.0, &[0])]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "亜");
    }
}
