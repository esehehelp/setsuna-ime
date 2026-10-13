//! State of a conversion being edited segment by segment (Converting phase
//! after ← / → / Shift+← / Shift+→): the segments over the reading, the
//! candidates and choice of each, and the focused one.
//!
//! Built from the whole-sentence list: the chosen sentence is cut into
//! segments (`segments::segment`) and every candidate of the list with an
//! alignment is cut at the same reading ranges, so a segment starts out with
//! the alternatives the sentence beam already had. Converting a segment's
//! own reading (with the segments before it as left context) adds more;
//! a segment whose range was resized starts over from its hiragana and waits
//! for that conversion (`stale`).

use std::collections::HashMap;

use crate::segments;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Seg {
    /// Reading chars `[start, end)`.
    pub start: usize,
    pub end: usize,
    pub cands: Vec<String>,
    pub selected: usize,
    /// Its own reading has been converted (its candidates include that).
    pub queried: bool,
    /// Resized: `cands` is only the hiragana until its conversion arrives.
    pub stale: bool,
    /// Continuations of its conversion asked for so far (`continue_beam`).
    pub more: u32,
    /// Its last continuation brought nothing new: the list goes round.
    pub exhausted: bool,
    /// Stepped past its last candidate: select the first new one when the
    /// continuation arrives.
    pub advance: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Segmented {
    pub reading: Vec<char>,
    pub segs: Vec<Seg>,
    pub focus: usize,
    /// The whole-sentence top shown before segment editing (what a changed
    /// commit is learned against).
    pub initial: String,
}

impl Segmented {
    /// Cut `list[selected]` into segments. `starts` gives the reading start
    /// of each char of a candidate (`segments::char_starts`); candidates
    /// without one contribute nothing to the segments' lists.
    pub fn build(
        reading: &str,
        list: &[String],
        selected: usize,
        starts: &HashMap<String, Vec<usize>>,
    ) -> Option<Self> {
        let chosen = list.get(selected)?;
        let reading_chars: Vec<char> = reading.chars().collect();
        let chosen_starts = match starts.get(chosen) {
            Some(s) => s.clone(),
            None => segments::char_starts(reading, chosen, None)?,
        };
        let cut = segments::segment(chosen, &chosen_starts, reading_chars.len());
        if cut.is_empty() {
            return None;
        }
        let ranges: Vec<(usize, usize)> = cut.iter().map(|s| s.reading).collect();
        let mut cands: Vec<Vec<String>> = segments::slice(chosen, &chosen_starts, &ranges)
            .into_iter()
            .map(|s| vec![s])
            .collect();
        // The whole reading's kana forms are not cut: each segment gets its
        // own hiragana / katakana (`kana_candidates::arrange`).
        let kana = [
            reading.to_string(),
            crate::kana_candidates::to_katakana(reading),
        ];
        for c in list {
            if c == chosen || kana.contains(c) {
                continue;
            }
            let Some(s) = starts.get(c) else {
                continue;
            };
            for (k, piece) in segments::slice(c, s, &ranges).into_iter().enumerate() {
                let seg_reading: String = reading_chars[ranges[k].0..ranges[k].1].iter().collect();
                if !piece.is_empty()
                    && !cands[k].contains(&piece)
                    && segments::fits(&seg_reading, &cands[k][0], &piece)
                {
                    cands[k].push(piece);
                }
            }
        }
        let segs = ranges
            .iter()
            .zip(cands)
            .map(|(&(start, end), cands)| Seg {
                start,
                end,
                cands,
                selected: 0,
                queried: false,
                stale: false,
                more: 0,
                exhausted: false,
                advance: false,
            })
            .collect();
        Some(Self {
            reading: reading_chars,
            segs,
            focus: 0,
            initial: list.first().cloned().unwrap_or_default(),
        })
    }

    pub fn reading_of(&self, k: usize) -> String {
        let s = &self.segs[k];
        self.reading[s.start..s.end].iter().collect()
    }

    pub fn full_reading(&self) -> String {
        self.reading.iter().collect()
    }

    /// The text shown for segment `k`: its choice, or its reading.
    pub fn text_of(&self, k: usize) -> String {
        let s = &self.segs[k];
        s.cands
            .get(s.selected)
            .cloned()
            .unwrap_or_else(|| self.reading_of(k))
    }

    pub fn texts(&self) -> Vec<String> {
        (0..self.segs.len()).map(|k| self.text_of(k)).collect()
    }

    pub fn text(&self) -> String {
        self.texts().concat()
    }

    /// The converted text before segment `k`.
    pub fn text_before(&self, k: usize) -> String {
        (0..k).map(|i| self.text_of(i)).collect()
    }

    /// UTF-16 offset of the focused segment in the shown text.
    pub fn focus_utf16_offset(&self) -> usize {
        self.text_before(self.focus).encode_utf16().count()
    }

    pub fn move_focus(&mut self, delta: i32) {
        let last = self.segs.len() as i32 - 1;
        self.focus = (self.focus as i32 + delta).clamp(0, last) as usize;
    }

    fn reset(&mut self, k: usize) {
        let hira = self.reading_of(k);
        let s = &mut self.segs[k];
        s.cands = vec![hira];
        s.selected = 0;
        s.queried = false;
        s.stale = true;
        s.more = 0;
        s.exhausted = false;
        s.advance = false;
    }

    /// Grow (`delta` > 0) or shrink the focused segment by one reading char,
    /// taking from or giving to the next one. Returns whether anything
    /// changed; the segments whose range changed become stale.
    pub fn resize(&mut self, delta: i32) -> bool {
        let k = self.focus;
        if delta > 0 {
            if k + 1 >= self.segs.len() {
                return false;
            }
            self.segs[k].end += 1;
            self.segs[k + 1].start += 1;
            if self.segs[k + 1].start >= self.segs[k + 1].end {
                self.segs.remove(k + 1);
            } else {
                self.reset(k + 1);
            }
        } else {
            if self.segs[k].end - self.segs[k].start <= 1 {
                return false;
            }
            self.segs[k].end -= 1;
            let end = self.segs[k].end;
            if k + 1 < self.segs.len() {
                self.segs[k + 1].start = end;
                self.reset(k + 1);
            } else {
                self.segs.push(Seg {
                    start: end,
                    end: self.reading.len(),
                    cands: Vec::new(),
                    selected: 0,
                    queried: false,
                    stale: false,
                    more: 0,
                    exhausted: false,
                    advance: false,
                });
                self.reset(k + 1);
            }
        }
        self.reset(k);
        true
    }

    /// The segment to convert next: the focused one if stale, else the
    /// first stale one.
    pub fn next_stale(&self) -> Option<usize> {
        if self.segs[self.focus].stale {
            return Some(self.focus);
        }
        self.segs.iter().position(|s| s.stale)
    }

    /// Take a conversion of segment `k`'s reading. A stale segment takes the
    /// list as is; otherwise its new candidates go after the ones it had,
    /// keeping the current choice.
    pub fn take(&mut self, k: usize, list: Vec<String>) {
        let s = &mut self.segs[k];
        if s.stale {
            if !list.is_empty() {
                s.cands = list;
            }
            s.selected = 0;
            s.stale = false;
        } else {
            for c in list {
                if !s.cands.contains(&c) {
                    s.cands.push(c);
                }
            }
        }
        s.queried = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    fn kisha() -> Segmented {
        let reading = "きしゃのきしゃ";
        let list = v(&["記者の記者", "汽車の記者", "記者の汽車"]);
        let mut starts = HashMap::new();
        for c in &list {
            starts.insert(c.clone(), vec![0, 2, 3, 4, 6]);
        }
        Segmented::build(reading, &list, 0, &starts).unwrap()
    }

    #[test]
    fn build_cuts_every_aligned_candidate_at_the_same_ranges() {
        let s = kisha();
        assert_eq!(s.segs.len(), 2);
        assert_eq!((s.segs[0].start, s.segs[0].end), (0, 4));
        assert_eq!(s.segs[0].cands, v(&["記者の", "汽車の"]));
        assert_eq!(s.segs[1].cands, v(&["記者", "汽車"]));
        assert_eq!(s.text(), "記者の記者");
        assert_eq!(s.initial, "記者の記者");
    }

    #[test]
    fn focus_and_offsets() {
        let mut s = kisha();
        s.move_focus(1);
        assert_eq!(s.focus, 1);
        assert_eq!(s.focus_utf16_offset(), 3);
        assert_eq!(s.text_before(1), "記者の");
        s.move_focus(5);
        assert_eq!(s.focus, 1);
        s.move_focus(-5);
        assert_eq!(s.focus, 0);
    }

    #[test]
    fn resizing_moves_a_reading_char_and_makes_both_stale() {
        let mut s = kisha();
        assert!(s.resize(-1));
        assert_eq!(s.reading_of(0), "きしゃ");
        assert_eq!(s.reading_of(1), "のきしゃ");
        assert_eq!(s.text(), "きしゃのきしゃ");
        assert_eq!(s.next_stale(), Some(0));
        s.take(0, v(&["記者", "汽車"]));
        assert_eq!(s.next_stale(), Some(1));
        s.take(1, v(&["の記者"]));
        assert_eq!(s.text(), "記者の記者");
        assert_eq!(s.next_stale(), None);
        // growing into the last char of the next segment removes it
        s.focus = 0;
        for _ in 0..4 {
            s.resize(1);
        }
        assert_eq!(s.segs.len(), 1);
        assert!(!s.resize(1));
    }

    #[test]
    fn shrinking_the_last_segment_adds_one() {
        let mut s = kisha();
        s.focus = 1;
        assert!(s.resize(-1));
        assert_eq!(s.segs.len(), 3);
        assert_eq!(s.reading_of(2), "ゃ");
    }

    #[test]
    fn a_new_conversion_extends_a_live_segment() {
        let mut s = kisha();
        s.segs[0].selected = 1;
        s.take(0, v(&["記者の", "帰社の"]));
        assert_eq!(s.segs[0].cands, v(&["記者の", "汽車の", "帰社の"]));
        assert_eq!(s.segs[0].selected, 1);
        assert!(s.segs[0].queried);
    }
}
