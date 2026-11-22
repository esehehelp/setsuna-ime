//! Segment editing (bunsetsu resize, focus movement, per-segment conversion
//! and continuation, and F6–F10 form conversion of the focused segment).

use chrono::Local;

use crate::char_forms::{self, Form};
use crate::engine_client::EngineResult;
use crate::segmented::Segmented;

use super::state::{consumed, not_consumed, Phase, ANALYSIS_WAIT};
use super::{Action, EngineBridge, KeyOutcome};

impl EngineBridge {
    /// Shift+← / Shift+→: resize the focused segment by one reading char
    /// (moving the cursor while composing).
    pub fn handle_resize(&mut self, delta: i32) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match &self.phase {
            Phase::Composing { .. } => self.handle_move_cursor(delta),
            Phase::Converting { .. } => {
                if !self.ensure_segmented() {
                    return consumed(Vec::new());
                }
                let changed = match &mut self.phase {
                    Phase::Converting { seg: Some(s), .. } => s.resize(delta),
                    _ => false,
                };
                if changed {
                    // Results for the old segments no longer fit.
                    self.comp.seg_requests.clear();
                    self.query_next_stale();
                }
                consumed(self.segment_actions())
            }
            _ => not_consumed(),
        }
    }

    /// Switch the Converting phase to segment editing unless it is already;
    /// false when the conversion cannot be segmented.
    pub(super) fn ensure_segmented(&mut self) -> bool {
        let date_now = self.comp.date_time_now.unwrap_or_else(Local::now);
        let date_time_enabled = self.cfg.date_time_enabled;
        let char_rules = self.cfg.char_rules.clone();
        let Phase::Converting {
            candidates,
            selected,
            seg,
        } = &mut self.phase
        else {
            return false;
        };
        if seg.is_some() {
            return true;
        }
        if self.comp.list_reading.is_empty() || self.comp.covers_typed {
            return false;
        }
        // a segment's own conversions take over from the whole sentence's rerank
        if self.comp.rerank.take().is_some() {
            if let Some(r) = &self.reranker {
                r.cancel();
            }
        }
        // the engine's analysis of the conversion cut, when it has an analyzer
        let words = candidates.get(*selected).and_then(|chosen| {
            let client = self.client.as_ref()?;
            client
                .analyze(std::slice::from_ref(chosen), ANALYSIS_WAIT)?
                .pop()
        });
        *seg = Segmented::build(
            &self.comp.list_reading,
            candidates,
            *selected,
            &self.comp.list_starts,
            words.as_deref(),
        );
        if let Some(s) = seg.as_mut() {
            // each segment offers its own hiragana / katakana, as a whole
            // conversion does
            for k in 0..s.segs.len() {
                let reading = s.reading_of(k);
                let cands = std::mem::take(&mut s.segs[k].cands);
                // the text the segment was cut to stays, even a hiragana other
                // than its reading's (a corrected reading's, ありがとう of ありがと)
                let cut = cands.first().cloned();
                let cands = with_romaji_kana(
                    &reading,
                    crate::kana_candidates::arrange(&reading, cut.as_deref(), cands),
                );
                let generated = if date_time_enabled {
                    char_rules.apply(crate::date_time::candidates(&reading, &date_now))
                } else {
                    Vec::new()
                };
                let chosen = cands.first().cloned();
                s.segs[k].cands = crate::date_time::insert(&reading, cands, generated);
                if let Some(chosen) = chosen {
                    s.segs[k].selected = s.segs[k]
                        .cands
                        .iter()
                        .position(|c| *c == chosen)
                        .unwrap_or(0);
                }
            }
        }
        self.comp.seg_requests.clear();
        let segments = self.segmented().map_or(0, |s| s.segs.len());
        for k in 0..segments {
            self.learn_segment(k, false);
        }
        segments > 0
    }

    /// The user's choices in segment `k`'s list (`Learning::rerank`): each
    /// candidate read as the sentence it makes, the other segments as
    /// chosen, over the whole reading, so that a sentence choice and a word
    /// choice whose context lies in the segments around apply as in a
    /// whole list. The candidate they rank over the chosen one goes right
    /// after it, the chosen one staying chosen: what is shown does not
    /// change by itself, and one ↓ or Space picks it. `fresh`: the list
    /// just replaced a resized segment's, whose first is chosen anyway: the
    /// candidate goes first.
    fn learn_segment(&mut self, k: usize, fresh: bool) {
        let Some(s) = self.segmented() else {
            return;
        };
        let Some(seg) = s.segs.get(k) else {
            return;
        };
        let reading = s.full_reading();
        let before = s.text_before(k);
        let after: String = s.texts()[k + 1..].concat();
        // the chosen one first: the one to outscore
        let order: Vec<usize> = std::iter::once(seg.selected)
            .chain((0..seg.cands.len()).filter(|&i| i != seg.selected))
            .filter(|&i| i < seg.cands.len())
            .collect();
        let sentences: Vec<String> = order
            .iter()
            .map(|&i| format!("{before}{}{after}", seg.cands[i]))
            .collect();
        let Some(learning) = self.learning.as_mut() else {
            return;
        };
        let Some(top) = learning
            .rerank(&reading, sentences.clone())
            .into_iter()
            .next()
        else {
            return;
        };
        let Some(best) = sentences.iter().position(|c| *c == top).filter(|&b| b > 0) else {
            return;
        };
        let Phase::Converting { seg: Some(s), .. } = &mut self.phase else {
            return;
        };
        let seg = &mut s.segs[k];
        let at = order[best];
        let c = seg.cands.remove(at);
        if fresh {
            seg.cands.insert(0, c);
            seg.selected = 0;
        } else {
            if at < seg.selected {
                seg.selected -= 1;
            }
            seg.cands.insert(seg.selected + 1, c);
        }
    }

    pub(super) fn segmented(&self) -> Option<&Segmented> {
        match &self.phase {
            Phase::Converting { seg: Some(s), .. } => Some(s),
            _ => None,
        }
    }

    /// The preedit and candidate window of segment editing.
    pub(super) fn segment_actions(&self) -> Vec<Action> {
        let Some(s) = self.segmented() else {
            return Vec::new();
        };
        let focused = &s.segs[s.focus];
        let mut texts = s.texts();
        if let Some(last) = texts.last_mut() {
            *last = self.with_pending(std::mem::take(last));
        }
        vec![
            Action::UpdateSegments {
                segments: texts,
                focus: s.focus,
            },
            Action::ShowSegmentCandidates {
                list: focused.cands.clone(),
                selected: focused.selected,
                at: s.focus_utf16_offset(),
            },
        ]
    }

    /// Convert segment `k`'s own reading, with the segments before it as
    /// left context; with `more` > 0, ask for more candidates of that
    /// conversion (the engine's `continue_beam`).
    fn query_segment(&mut self, k: usize, more: u32) {
        // Serialize segment jobs: the client's pending slot is replaceable.
        if !self.comp.seg_requests.is_empty() {
            return;
        }
        let Some(s) = self.segmented() else {
            return;
        };
        let reading = s.reading_of(k);
        let context = format!("{}{}", self.ctx.context, s.text_before(k));
        if let Some(client) = &self.client {
            let req_id = self.next_req_id();
            if more == 0 {
                client.dispatch(req_id, &context, &reading);
            } else {
                client.dispatch_more(req_id, &context, &reading, more);
            }
            self.comp.seg_requests.push((req_id, k, more, false));
        }
    }

    /// Convert segment `k`'s letters read as romaji (`romaji_to_kana`: an
    /// English segment's other reading, よろ for `yoro`), once, after its
    /// own conversion; its candidates join the segment's.
    fn query_segment_kana(&mut self, k: usize) {
        if !self.comp.seg_requests.is_empty() {
            return;
        }
        let Some(s) = self.segmented() else {
            return;
        };
        let kana = crate::composing_text::romaji_to_kana(&s.reading_of(k));
        let context = format!("{}{}", self.ctx.context, s.text_before(k));
        if let (Some(kana), Some(client)) = (kana, &self.client) {
            let req_id = self.next_req_id();
            client.dispatch(req_id, &context, &kana);
            self.comp.seg_requests.push((req_id, k, 0, true));
        }
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            s.segs[k].kana_queried = true;
        }
    }

    /// Past the focused segment's last candidate: ask for more of them
    /// instead of going round to the first, and move to the first new one
    /// when they arrive. False when the segment has no more to give (its
    /// last continuation brought nothing new, or it is waiting for its
    /// first conversion after a resize).
    pub(super) fn continue_focused(&mut self) -> bool {
        let Some(s) = self.segmented() else {
            return false;
        };
        let k = s.focus;
        let seg = &s.segs[k];
        if seg.exhausted || seg.stale || self.client.is_none() {
            return false;
        }
        let in_flight = !self.comp.seg_requests.is_empty();
        let more = (seg.queried && !in_flight).then_some(seg.more + 1);
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            s.segs[k].advance = true;
            if let Some(m) = more {
                s.segs[k].more = m;
            }
        }
        if !in_flight {
            // its own conversion first, if it has not had one
            self.query_segment(k, more.unwrap_or(0));
        }
        true
    }

    /// Convert the focused segment unless that was done or is under way,
    /// then (an English one) its letters read as romaji.
    pub(super) fn query_focused_segment(&mut self) {
        let Some(s) = self.segmented() else {
            return;
        };
        let k = s.focus;
        if s.segs[k].advance && s.segs[k].queried {
            self.continue_focused();
            return;
        }
        if !s.segs[k].queried && self.comp.seg_requests.is_empty() {
            self.query_segment(k, 0);
        } else if s.segs[k].queried && !s.segs[k].kana_queried && self.comp.seg_requests.is_empty()
        {
            self.query_segment_kana(k);
        }
    }

    fn query_next_stale(&mut self) {
        if let Some(k) = self.segmented().and_then(|s| s.next_stale()) {
            if !self.comp.seg_requests.iter().any(|&(_, i, _, _)| i == k) {
                self.query_segment(k, 0);
            }
        }
    }

    /// A result while editing segments: the conversion of one of them (put
    /// into its list), or a stale whole-sentence result (dropped). `None`
    /// when not editing segments.
    pub(super) fn apply_segment_result(&mut self, result: &EngineResult) -> Option<Vec<Action>> {
        self.segmented()?;
        let Some(at) = self
            .comp
            .seg_requests
            .iter()
            .position(|&(id, _, _, _)| id == result.req_id)
        else {
            return Some(Vec::new());
        };
        let (_, k, more, kana) = self.comp.seg_requests.remove(at);
        let reading = self.segmented()?.reading_of(k);
        if kana {
            // the letters read as romaji: its conversions after the
            // segment's own, the kana forms with them
            let Some(kana) =
                crate::composing_text::romaji_to_kana(&reading).filter(|r| *r == result.reading)
            else {
                return Some(Vec::new());
            };
            let list = self.finish_list(
                &kana,
                result.corrected_reading.as_deref(),
                result.candidates.clone(),
                false,
            );
            if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
                if !s.segs[k].stale {
                    s.take(k, with_romaji_kana(&reading, list));
                }
            }
            self.learn_segment(k, false);
            self.query_next_stale();
            self.query_focused_segment();
            return Some(self.segment_actions());
        }
        if reading != result.reading {
            return Some(Vec::new());
        }
        let mut list = self.finish_list(
            &reading,
            result.corrected_reading.as_deref(),
            result.candidates.clone(),
            false,
        );
        let dates = self
            .cfg
            .char_rules
            .apply(self.date_time_candidates(&reading));
        if let Some(s) = self.segmented() {
            let chosen = s.text_of(k);
            let stale = s.segs[k].stale;
            // a resized segment has no chosen text yet: judge by the list's top
            let chosen = if stale {
                list.first().cloned().unwrap_or(chosen)
            } else {
                chosen
            };
            list.retain(|c| dates.contains(c) || crate::segments::fits(&reading, &chosen, c));
        }
        let mut again = false;
        let mut fresh = false;
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            let before = s.segs[k].cands.len();
            fresh = s.segs[k].stale;
            s.take(k, with_romaji_kana(&reading, list));
            let seg = &mut s.segs[k];
            let added = seg.cands.len() > before;
            if more > 0 && !added {
                seg.exhausted = true;
            }
            if seg.advance {
                if added && s.focus == k {
                    seg.selected = before;
                    seg.advance = false;
                } else if s.focus == k && !seg.exhausted && more == 0 {
                    // its own conversion added nothing: ask for more
                    again = true;
                } else {
                    // nothing more: round to the first, as without continuing
                    if s.focus == k && seg.exhausted {
                        seg.selected = 0;
                    }
                    seg.advance = false;
                }
            }
        }
        self.learn_segment(k, fresh);
        if again {
            self.continue_focused();
        }
        self.query_next_stale();
        self.query_focused_segment();
        Some(self.segment_actions())
    }

    /// F6–F10 while editing segments: the focused segment only, in `form`
    /// (its letters read as romaji for the kana forms, its kana spelled as
    /// romaji for the alphanumeric ones), the others as they are. The text
    /// becomes the segment's choice, so ← / → go on editing; F9 / F10
    /// again cycle the case. The segments' readings stay as typed.
    pub(super) fn convert_segment_to(&mut self, form: Form) -> KeyOutcome {
        let Some(s) = self.segmented() else {
            return consumed(Vec::new());
        };
        let k = s.focus;
        let reading = s.reading_of(k);
        let shown = s.text_of(k);
        let kana =
            crate::composing_text::romaji_to_kana(&reading).unwrap_or_else(|| reading.clone());
        let romaji = char_forms::romaji_of(&reading);
        let case = match &self.comp.seg_form {
            Some((at, f, case, text)) if *at == k && *f == form && *text == shown => {
                if form.cycles_case() {
                    case + 1
                } else {
                    *case
                }
            }
            _ => 0,
        };
        let text = char_forms::convert(form, &kana, &romaji, case);
        if text.is_empty() {
            return consumed(Vec::new());
        }
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            let f = &mut s.segs[k];
            f.selected = match f.cands.iter().position(|c| *c == text) {
                Some(i) => i,
                None => {
                    f.cands.push(text.clone());
                    f.cands.len() - 1
                }
            };
            f.advance = false;
        }
        self.comp.seg_form = Some((k, form, case, text));
        consumed(self.segment_actions())
    }
}

/// A segment's candidates `cands` with, for a reading of letters (an
/// English word of mixed input), the hiragana and katakana of the letters
/// read as romaji (`yoro`: よろ, ヨロ) after them: a word taken for English
/// converts by the segment too, as F6 / F7 convert the whole input.
fn with_romaji_kana(reading: &str, mut cands: Vec<String>) -> Vec<String> {
    if cands.is_empty() {
        return cands;
    }
    if let Some(kana) = crate::composing_text::romaji_to_kana(reading) {
        let katakana = crate::kana_candidates::to_katakana(&kana);
        for form in [kana, katakana] {
            if !cands.contains(&form) {
                cands.push(form);
            }
        }
    }
    cands
}
