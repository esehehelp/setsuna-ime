//! Live-conversion dispatch, prefix preview cache, reading-alignment starts,
//! and shown-character Backspace handling.

use std::collections::HashMap;

use super::state::{consumed, Phase};
use super::{Action, EngineBridge, KeyOutcome};

impl EngineBridge {
    /// Fire an async convert job for the current full hiragana. Returns
    /// the candidate list currently held by the Composing phase (possibly
    /// stale — the keystroke renders at once with what it has, and the
    /// conversion's candidates land when the engine answers).
    pub(super) fn dispatch_live(&mut self) -> Vec<String> {
        self.dispatch_live_inner(false)
    }

    /// Same as `dispatch_live`, but bypasses the short-reading OOD guard.
    /// Used only for explicit conversion requests such as Space.
    pub(super) fn dispatch_live_forced(&mut self) -> Vec<String> {
        self.dispatch_live_inner(true)
    }

    fn dispatch_live_inner(&mut self, force: bool) -> Vec<String> {
        // the input changed or is converted: the kana Escape showed go
        self.comp.kana_shown = false;
        let hira = if force {
            self.composing.commit_reading()
        } else {
            self.composing.live_reading()
        };
        if self.comp.kept.as_ref().is_some_and(|(r, _, _)| *r != hira) {
            self.comp.kept = None;
        }
        if hira.is_empty() {
            self.comp.last_applied_reading.clear();
            self.comp.clear_preview_cache();
            self.comp.kept = None;
            return Vec::new();
        }
        // Skip OOD partial-input conversions during live typing.
        // Clear the cache so `live_preedit` falls through to raw hira;
        // we'll resume dispatching once the reading grows past the gate.
        if !force && hira.chars().count() < self.cfg.live_min_dispatch_chars {
            self.comp.last_applied_reading.clear();
            self.comp.clear_preview_cache();
            return Vec::new();
        }
        // Keep the cache even when the user backspaces — `live_preedit`
        // handles the shrink case by trimming the cached kanji tail,
        // which preserves visual continuity until the next async result
        // lands. We only drop the cache explicitly (in commit / escape /
        // empty-composition paths) elsewhere. Space may find the list in for
        // a reading the preview does not show (a pending n read as ん): it
        // takes that list, and its rerank goes on.
        let listed = force
            && hira == self.comp.list_reading
            && matches!(&self.phase, Phase::Composing { candidates } if !candidates.is_empty());
        if hira != self.comp.cached_prefix_reading && !listed {
            self.stop_rerank();
            if let Some(client) = &self.client {
                let req_id = self.next_req_id();
                client.dispatch(req_id, &self.ctx.context, &hira);
                self.comp.live_out = Some(hira.clone());
            }
        }
        if let Phase::Composing { candidates } = &self.phase {
            candidates.clone()
        } else {
            Vec::new()
        }
    }

    /// Preedit for the current composition. Three cases:
    ///
    /// * **Grow** (`hira.starts_with(cached_reading)`, user kept typing):
    ///   `cached_kanji + (hira - cached_reading) + pending`. Everything
    ///   already converted stays kanji; only the delta is raw kana.
    ///
    /// * **Shrink** (`cached_reading.starts_with(hira)`, user back-spaced):
    ///   trim trailing chars from `cached_kanji` by the delta count in
    ///   chars. This is an approximation (1 hiragana char ≠ 1 kanji char
    ///   in general), but it keeps the preedit visually continuous until
    ///   the async convert returns and fixes up the cache. Avoids the
    ///   `kanji → raw → kanji` flash that the naive "drop cache" version
    ///   caused.
    ///
    /// * **Diverge** (the end of the reading changed, mixed input reading
    ///   letters otherwise): the shrink of the cache to the reading they
    ///   share, then the rest as raw kana; raw `hira + pending` only when
    ///   they share nothing.
    ///
    /// `hira` is the kana typed, without the ん `live_reading` reads a
    /// pending n as: the n shows as typed (今日はn), and a conversion with
    /// that ん is never cached (`apply_async_result`).
    pub(super) fn live_preedit(&self) -> String {
        // editing mid-composition: the kana as typed, so the caret has
        // something to sit between
        if !self.composing.cursor_at_end() || !self.cfg.live_conversion {
            return self.composing.preedit();
        }
        let hira = self.composing.hiragana();
        let pending = self.composing.pending_romaji();
        if hira.is_empty() {
            return pending.to_string();
        }
        let cached_r = &self.comp.cached_prefix_reading;
        let cached_k = &self.comp.cached_prefix_kanji;
        // a shortcut word stands for its whole reading only
        if let Some(word) = self.comp.cached_shortcut.as_ref() {
            if hira == cached_r.as_str() {
                return format!("{word}{pending}");
            }
        }
        if !cached_r.is_empty() && !cached_k.is_empty() {
            let common = cached_r
                .chars()
                .zip(hira.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if common == cached_r.chars().count() {
                let tail: String = hira.chars().skip(common).collect();
                return format!("{cached_k}{tail}{pending}");
            }
            if common > 0 {
                if let Some((kept, read)) = self.converted_within(cached_r, cached_k, common) {
                    let tail: String = hira.chars().skip(read).collect();
                    return format!("{kept}{tail}{pending}");
                }
            }
        }
        // nothing converted yet: as typed (けっこn)
        format!("{hira}{pending}")
    }

    /// The chars of `text` (the conversion of `reading`) read wholly within
    /// the first `common` kana of the reading, and how many kana they read:
    /// what a shrunk or changed reading still shows of it, the kana after
    /// them shown as typed (会社 for かいしゃ, one kana deleted: 会 and し).
    /// The chars' reading starts are those Backspace kept, the list's
    /// (the engine's), else the char alignment, a kanji run split by the
    /// lexicon; `None` when they cannot be had, and the kana are shown: a
    /// char kept by counting kana hid the kana left of its reading
    /// (会 shown, かいし held: `a` gave 開始ア; HTC温度 and HTCondor gave
    /// HTondor).
    fn converted_within(
        &self,
        reading: &str,
        text: &str,
        common: usize,
    ) -> Option<(String, usize)> {
        let mut starts = self
            .comp
            .kept
            .as_ref()
            .filter(|(r, t, _)| r == reading && t == text)
            .map(|(_, _, s)| s.clone())
            .or_else(|| self.list_starts_for(reading, text))
            .or_else(|| crate::segments::char_starts(reading, text, None))?;
        let n = reading.chars().count();
        if !starts_fit(&starts, text, n) {
            return None;
        }
        if let Some(lex) = &self.lexicon {
            split_runs(lex, reading, text, &mut starts);
        }
        // char i reads starts[i] .. starts[i + 1] (the last, up to n); chars
        // sharing a start are one unit (今日, きょう), cut after its last
        let ends: Vec<usize> = starts.iter().skip(1).copied().chain([n]).collect();
        let mut k = 0;
        for c in 1..=ends.len() {
            if ends[c - 1] > common {
                break;
            }
            if c == ends.len() || starts[c] > starts[c - 1] {
                k = c;
            }
        }
        let read = if k == 0 { 0 } else { ends[k - 1] };
        Some((text.chars().take(k).collect(), read))
    }

    /// Cache the top of `list` as the conversion of `reading` shown while
    /// typing. A shortcut word of the user dictionary (短縮よみ: an address
    /// for じゅうしょ) is for the whole reading only: it is shown while the
    /// reading is that one, and the first other candidate stands for the
    /// reading as the start of a longer one, so the word does not stay on
    /// screen before the kana typed after it.
    pub(super) fn cache_conversion(&mut self, reading: &str, list: &[String]) {
        let dict = self.user_dict();
        let shortcut = |c: &String| dict.as_ref().is_some_and(|d| d.is_shortcut(reading, c));
        let top = list.first().cloned().unwrap_or_default();
        self.comp.cached_prefix_reading = reading.to_string();
        if shortcut(&top) {
            self.comp.cached_prefix_kanji = list
                .iter()
                .find(|c| !shortcut(c))
                .cloned()
                .unwrap_or_else(|| reading.to_string());
            self.comp.cached_shortcut = Some(top);
        } else {
            self.comp.cached_prefix_kanji = top;
            self.comp.cached_shortcut = None;
        }
    }

    /// Live conversion: a candidate of `list` ending in kana goes first in
    /// place of the model's first ending in a noun (明日え for 明日絵),
    /// when it keeps all that first has before its last unit and scores
    /// within `[frontend] live_noun_end_penalty` of it, so the end of what
    /// is being typed does not keep changing. `raw` and `scores` are the
    /// engine's candidates and their scores. A first of a single unit
    /// (東京), one the engine did not give (learned, the user
    /// dictionary's), one ending in punctuation (天気。) and what Backspace
    /// kept stay. Space puts the model's first back (`model_first`).
    pub(super) fn hold_noun_end(
        &mut self,
        reading: &str,
        list: &mut Vec<String>,
        raw: &[String],
        scores: &[f32],
        starts: &HashMap<String, Vec<usize>>,
    ) {
        self.comp.end_held = None;
        let penalty = self.cfg.noun_end_penalty;
        if !self.cfg.live_conversion
            || penalty <= 0.0
            || list.len() < 2
            || scores.len() != raw.len()
            || self
                .comp
                .kept
                .as_ref()
                .is_some_and(|(r, _, _)| r == reading)
        {
            return;
        }
        let score = |c: &str| raw.iter().position(|r| r == c).map(|i| scores[i]);
        let top = list[0].clone();
        let Some(top_score) = score(&top) else {
            return;
        };
        if !ends_in_noun(&top) {
            return;
        }
        let Some(before) = before_last_unit(reading, &top, starts.get(&top)) else {
            return;
        };
        if before.is_empty() {
            return;
        }
        let best = list
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, c)| c.len() > before.len() && c.starts_with(&before) && !ends_in_noun(c))
            .filter_map(|(i, c)| score(c).map(|s| (i, s)))
            .filter(|&(_, s)| s > top_score - penalty)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = best {
            let c = list.remove(i);
            list.insert(0, c);
            self.comp.end_held = Some((reading.to_string(), top));
        }
    }

    /// The live list of `reading` with the model's first back first, where
    /// a candidate ending in kana went before it (`hold_noun_end`): Space
    /// converts.
    pub(super) fn model_first(&mut self, reading: &str, mut list: Vec<String>) -> Vec<String> {
        if let Some((_, top)) = self.comp.end_held.take().filter(|(r, _)| r == reading) {
            if let Some(i) = list.iter().position(|c| *c == top) {
                let c = list.remove(i);
                list.insert(0, c);
            }
        }
        list
    }

    /// A provisional whole reading, never a stale candidate for a different
    /// reading. Only a matching converted prefix may replace its raw kana.
    pub(super) fn provisional_conversion(&self) -> String {
        let reading = self.composing.live_reading();
        if self.cfg.live_conversion
            && self.composing.cursor_at_end()
            && !self.comp.cached_prefix_reading.is_empty()
            && reading.starts_with(&self.comp.cached_prefix_reading)
        {
            format!(
                "{}{}",
                self.comp.cached_prefix_kanji,
                &reading[self.comp.cached_prefix_reading.len()..]
            )
        } else {
            reading.to_string()
        }
    }

    pub(super) fn with_pending(&self, mut text: String) -> String {
        // a prediction covers what is still being typed
        if self.form_shown().is_none() && self.comp.prediction.is_none() && !self.comp.covers_typed
        {
            text.push_str(self.composing.pending_romaji());
        }
        text
    }

    /// Live conversion, Backspace on a conversion shown whole: its last char
    /// goes with the reading under it (行 of 東京に行く takes い and く's
    /// く), the rest stays as shown (`kept`) and the shorter reading is
    /// converted for the list. The chars' reading starts are the engine's,
    /// or those Backspace kept; a kanji run placed as one (no engine
    /// positions: a candidate of the lexicon) is split by the lexicon's
    /// readings of its chars (東京 as とう|きょう). `None` where Backspace
    /// deletes the last kana typed: the setting off, the caret moved, romaji
    /// pending, a kana tail not converted yet, a text the reading does not
    /// align with, and a last char read only with the one before it (日 of
    /// 今日, きょう: user 2026-10-28, rather a kana than two chars).
    pub(super) fn backspace_shown(&mut self) -> Option<KeyOutcome> {
        if !self.cfg.backspace_deletes_shown
            || !self.cfg.live_conversion
            || !self.composing.cursor_at_end()
            || !self.composing.pending_romaji().is_empty()
        {
            return None;
        }
        let reading = self.composing.hiragana().to_string();
        if reading.is_empty() || self.comp.cached_prefix_reading != reading {
            return None;
        }
        let shown = self.live_preedit();
        let mut starts = self
            .comp
            .kept
            .as_ref()
            .filter(|(r, t, _)| *r == reading && *t == shown)
            .map(|(_, _, s)| s.clone())
            .or_else(|| self.list_starts_for(&reading, &shown))
            .or_else(|| crate::segments::char_starts(&reading, &shown, None))?;
        let n = reading.chars().count();
        if !starts_fit(&starts, &shown, n) {
            return None;
        }
        if let Some(lex) = &self.lexicon {
            split_runs(lex, &reading, &shown, &mut starts);
        }
        let last = *starts.last()?;
        if last >= n {
            return None;
        }
        let rest: String = shown
            .chars()
            .zip(&starts)
            .take_while(|&(_, &s)| s < last)
            .map(|(c, _)| c)
            .collect();
        if rest.chars().count() + 1 < starts.len() {
            return None;
        }
        let mut rest_starts = starts[..rest.chars().count()].to_vec();
        for _ in last..n {
            self.composing.delete_left();
        }
        self.invalidate_engine_results();
        self.stop_rerank();
        if self.composing.is_empty() {
            self.composing.reset();
            self.phase = Phase::Empty;
            self.clear_composition();
            return Some(consumed(vec![Action::EndComposition]));
        }
        let left = self.composing.hiragana().to_string();
        // A letter typed before the kana deleted went back to the romaji
        // pending (ふくめkなお less なお holds ふくめ, k pending): what is
        // kept is the chars read wholly within the kana left, then the kana
        // after them; the letter shows once, as pending (含めk, not 含めkk).
        let mut rest = rest;
        if !left.chars().eq(reading.chars().take(last)) {
            let common = left
                .chars()
                .zip(reading.chars())
                .take_while(|(a, b)| a == b)
                .count();
            let (within, read) = self
                .converted_within(&reading, &shown, common)
                .unwrap_or_default();
            rest_starts.truncate(within.chars().count());
            rest_starts.extend(read..left.chars().count());
            rest = within + &left.chars().skip(read).collect::<String>();
        }
        if rest.is_empty() {
            // a letter alone is left
            self.comp.kept = None;
            self.comp.clear_preview_cache();
            let candidates = self.dispatch_live();
            self.phase = Phase::Composing { candidates };
            return Some(consumed(vec![Action::UpdatePreedit {
                text: self.live_preedit(),
            }]));
        }
        self.comp.kept = Some((left.clone(), rest.clone(), rest_starts));
        let candidates = self.dispatch_live();
        self.comp.cached_prefix_reading = left;
        self.comp.cached_prefix_kanji = rest;
        self.comp.cached_shortcut = None;
        self.phase = Phase::Composing { candidates };
        Some(consumed(vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]))
    }

    /// The list's reading starts of the chars of `text`, where the list is
    /// `reading`'s: those of another reading (the list before a Backspace,
    /// or one not yet replaced) do not fit it.
    fn list_starts_for(&self, reading: &str, text: &str) -> Option<Vec<usize>> {
        if self.comp.list_reading != reading {
            return None;
        }
        self.comp.list_starts.get(text).cloned()
    }

    /// Reading starts of the chars of each candidate in `list`: from the
    /// engine's alignment of the raw candidate it was made from (when the
    /// post-processing kept its length), else from the kana anchors.
    pub(super) fn list_starts_of(
        &self,
        reading: &str,
        raw: &[String],
        reading_pos: &[Vec<u32>],
        list: &[String],
    ) -> HashMap<String, Vec<usize>> {
        let mut out: HashMap<String, Vec<usize>> = HashMap::new();
        for (r, pos) in raw.iter().zip(reading_pos) {
            let processed = crate::postprocess::apply(
                reading,
                vec![r.clone()],
                self.cfg.symbol_width,
                &self.ctx.context,
            );
            let processed = if self.cfg.drop_untyped_symbols {
                crate::postprocess::drop_untyped_symbols(reading, processed)
            } else {
                processed
            };
            let Some(one) = processed.first() else {
                continue;
            };
            let shown = self.cfg.char_rules.apply_one(one);
            if out.contains_key(&shown)
                || !list.contains(&shown)
                || shown.chars().count() != r.chars().count()
            {
                continue;
            }
            if let Some(starts) = crate::segments::char_starts(reading, &shown, Some(pos)) {
                out.insert(shown, starts);
            }
        }
        for c in self
            .cfg
            .char_rules
            .apply(self.date_time_candidates(reading))
        {
            if list.contains(&c) {
                out.insert(c.clone(), vec![0; c.chars().count()]);
            }
        }
        // what Backspace left, at the starts it had
        if let Some((_, text, starts)) = self.comp.kept.as_ref().filter(|(r, _, _)| r == reading) {
            if list.contains(text) {
                out.insert(text.clone(), starts.clone());
            }
        }
        for c in list {
            if !out.contains_key(c) {
                if let Some(starts) = crate::segments::char_starts(reading, c, None) {
                    out.insert(c.clone(), starts);
                }
            }
        }
        out
    }
}

/// Split each run of chars `starts` places at one reading char (a kanji
/// run without the engine's positions: 東京 both at the と of とう), and
/// each run of kanji, by the lexicon's readings of its chars
/// (`Lexicon::split_reading`). The engine's positions inside a kanji run are
/// the frames its tokens came out at, not where their readings start (会社
/// of かいしゃ: 会 at か, 社 at ゃ, which left し with 会 and hid it when
/// Backspace took 社). A kanji run the lexicon does not split goes as one
/// (今日, きょう).
fn split_runs(lex: &crate::lexicon::Lexicon, reading: &str, text: &str, starts: &mut [usize]) {
    let r: Vec<char> = reading.chars().collect();
    let t: Vec<char> = text.chars().collect();
    if !starts_fit(starts, text, r.len()) {
        return;
    }
    let mut i = 0;
    while i < t.len() {
        let mut j = i + 1;
        while j < t.len() && (starts[j] == starts[i] || is_kanji(t[j - 1]) && is_kanji(t[j])) {
            j += 1;
        }
        let end = if j < t.len() { starts[j] } else { r.len() };
        if j - i > 1 && starts[i] < end {
            let run: String = t[i..j].iter().collect();
            let read: String = r[starts[i]..end].iter().collect();
            match lex.split_reading(&run, &read).filter(|s| s.len() == j - i) {
                Some(split) => {
                    for (k, s) in split.into_iter().enumerate() {
                        starts[i + k] = starts[i] + s;
                    }
                }
                None => {
                    let s = starts[i];
                    starts[i..j].iter_mut().for_each(|x| *x = s);
                }
            }
        }
        i = j;
    }
}

/// Whether `starts` can be the reading starts of the chars of `text` in a
/// reading of `n` chars: one per char, in order, none past the end.
pub(super) fn starts_fit(starts: &[usize], text: &str, n: usize) -> bool {
    starts.len() == text.chars().count()
        && starts.windows(2).all(|w| w[0] <= w[1])
        && starts.last().is_none_or(|&s| s <= n)
}

/// A kanji, or an iteration or abbreviation mark read as one (々, 〆, ヶ).
fn is_kanji(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}' | '々' | '〆' | 'ヶ')
}

/// Whether `text` ends in a noun: a kanji, katakana, digit or letter last
/// (a long-vowel mark goes with the char before it, すげー ends in kana).
/// Hiragana, punctuation and symbols are no noun (天気。, 「天気」).
fn ends_in_noun(text: &str) -> bool {
    text.chars()
        .rev()
        .find(|c| !matches!(c, 'ー' | '〜' | '～'))
        .is_some_and(|c| c.is_alphanumeric() && !matches!(c, 'ぁ'..='ゖ' | 'ゝ' | 'ゞ'))
}

/// What `text` has before its last unit: the chars sharing its last
/// reading start (敗 of 今日敗), with the katakana before them where they
/// are katakana (a katakana word is placed a char a unit: コーヒー). The
/// starts are the list's, else the kana anchors; `None` when they cannot
/// be had.
fn before_last_unit(reading: &str, text: &str, starts: Option<&Vec<usize>>) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let starts = match starts {
        Some(s) => s.clone(),
        None => crate::segments::char_starts(reading, text, None)?,
    };
    if chars.is_empty() || !starts_fit(&starts, text, reading.chars().count()) {
        return None;
    }
    let last = *starts.last()?;
    let mut k = starts.iter().rposition(|&s| s != last).map_or(0, |i| i + 1);
    let katakana = |c: char| matches!(c, 'ァ'..='ヺ' | 'ー');
    if chars[k..].iter().all(|&c| katakana(c)) {
        while k > 0 && katakana(chars[k - 1]) {
            k -= 1;
        }
    }
    Some(chars[..k].iter().collect())
}

#[cfg(test)]
mod tests {
    use super::{before_last_unit, ends_in_noun};

    #[test]
    fn a_noun_end_is_a_kanji_katakana_digit_or_letter_last() {
        for t in ["明日絵", "会議の資料", "コーヒー", "100", "Windows", "人々"] {
            assert!(ends_in_noun(t), "{t}");
        }
        for t in [
            "明日え",
            "天気。",
            "天気、",
            "天気！",
            "天気？",
            "「天気」",
            "すげー",
            "",
        ] {
            assert!(!ends_in_noun(t), "{t}");
        }
    }

    #[test]
    fn the_last_unit_is_the_chars_sharing_the_last_start() {
        let unit = |r: &str, t: &str, s: &[usize]| before_last_unit(r, t, Some(&s.to_vec()));
        assert_eq!(
            unit("きょうはい", "今日敗", &[0, 0, 3]).as_deref(),
            Some("今日")
        );
        assert_eq!(unit("とうきょう", "東京", &[0, 0]).as_deref(), Some(""));
        assert_eq!(
            unit("きょうはこーひー", "今日はコーヒー", &[0, 0, 3, 4, 5, 6, 7]).as_deref(),
            Some("今日は")
        );
        assert_eq!(unit("あした", "明日", &[0, 9]), None);
    }
}
