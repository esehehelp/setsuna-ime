//! Candidate ordering, post-processing, local generators, and mixed-input list assembly.

use chrono::Local;

use crate::engine_client::EngineResult;

use super::state::{AltList, ALT_EARLY, ALT_MAX, NO_LEXICON_MARGIN};
use super::EngineBridge;

/// The IME's order of the engine's candidates (`lexicon` and `ngram` as
/// the bridge has them; public for evaluations): those that spell the
/// reading out first, by their scores with the character n-gram terms
/// added (`[ngram]`: those sums become their scores, which the LM reranker
/// then adds to) and, below the first of them, their lexicon costs taken
/// off (`[lexicon]`, which also adds its own conversions and drops those far
/// above the cheapest), then those that do not, in the
/// engine's order and scored no higher than the others. A candidate that
/// does not has lost a char or put a kana out of place (ゼロプラモデル for
/// ぜろせんぷらもでる): never the gold over wiki_fresh_v2 and AJIMEE,
/// and one the n-gram would favour, with fewer pairs to lose points on.
pub fn order_candidates(
    lexicon: Option<&crate::lexicon::Lexicon>,
    ngram: Option<(&crate::char_ngram::CharNgram, f32, f32)>,
    result: &mut EngineResult,
) {
    order_candidates_with(lexicon, ngram, result, true);
}

/// `order_candidates`, with the lexicon's costs and conversions
/// (`lexicon_costs`) or without them: the same first candidate either way
/// (the lexicon's step keeps the one before it first), the others in the
/// engine's order and none added or dropped. The lexicon's part is most of
/// the ordering's time (3 ms of a reading of 20 to 30 kana, 8 to 19 of a
/// longer one), which a live conversion's preview, showing the first only,
/// need not wait for.
pub(super) fn order_candidates_with(
    lexicon: Option<&crate::lexicon::Lexicon>,
    ngram: Option<(&crate::char_ngram::CharNgram, f32, f32)>,
    result: &mut EngineResult,
    lexicon_costs: bool,
) {
    use unicode_normalization::UnicodeNormalization;
    // the lexicon that judges the engine's candidates: not a partial one,
    // whose missing words are no slips (Lexicon::partial)
    let ranking = lexicon.filter(|l| !l.partial());
    if ranking.is_none() {
        drop_far_below_top(result);
    }
    let mut n = result.candidates.len();
    if n < 2 {
        return;
    }
    let reading: Vec<char> = result.reading.nfkc().collect();
    // the reading corrector's reading, when it changed the one typed: its
    // candidates spell that one out, and the lexicon costs them by it
    let corrected: Option<Vec<char>> = result
        .corrected_reading
        .as_ref()
        .map(|r| r.nfkc().collect());
    let (mut order, mut rest): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| {
        let c = &result.candidates[i];
        crate::reading_align::spells_out(c, &reading)
            || corrected
                .as_ref()
                .is_some_and(|r| crate::reading_align::spells_out(c, r))
    });
    if order.is_empty() {
        std::mem::swap(&mut order, &mut rest);
    }
    let scored = result.scores.len() == n;
    // numbers in digits and in kanji in one candidate (postprocess::mixes_numerals)
    let mixed: Vec<bool> = result
        .candidates
        .iter()
        .map(|c| scored && crate::postprocess::mixes_numerals(c))
        .collect();
    let any_mixed = mixed.iter().any(|&m| m);
    if rest.is_empty() && !any_mixed && !(scored && (ngram.is_some() || ranking.is_some())) {
        return;
    }
    let mut scores = result.scores.clone();
    for (s, &m) in scores.iter_mut().zip(&mixed) {
        if m {
            *s -= crate::postprocess::MIXED_NUMERALS;
        }
    }
    if any_mixed {
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    }
    if let (Some((table, w2, w3)), true) = (ngram, scored && order.len() >= 2) {
        let cands: Vec<String> = order
            .iter()
            .map(|&i| result.candidates[i].clone())
            .collect();
        let own: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
        let (by, total) = table.rescore(&result.context, &cands, &own, w2, w3);
        for (k, &i) in order.iter().enumerate() {
            scores[i] = total[k];
        }
        order = by.iter().map(|&k| order[k]).collect();
    }
    if let (Some(lex), true) = (ranking.filter(|_| lexicon_costs), scored) {
        // the model's top as ordered so far stays first: the lexicon's costs
        // read no context, and putting their cheapest first lost more than
        // it won (Cymbidium-v1.1-xsmall's top 20, EM1: AJIMEE v2-fix2 77.9
        // -> 74.8, its homophones 77.1 -> 73.7, probe_v6's 83.3 -> 75.0;
        // wiki_fresh_v4 82.3 -> 83.2). What the lexicon adds and orders below
        // it put more golds in the first 20 (AJIMEE 87.9 -> 91.2, its proper
        // nouns 78.3 -> 83.3) either way (user 2026-11-15: それでok).
        let model_top = order.first().copied();
        let r: String = reading.iter().collect();
        let fixed: Option<String> = corrected.as_ref().map(|r| r.iter().collect());
        let mut costs: Vec<Option<f32>> = result
            .candidates
            .iter()
            .map(|c| {
                let c: String = c.nfkc().collect();
                lex.cost(&c, &r)
                    .or_else(|| fixed.as_deref().and_then(|f| lex.cost(&c, f)))
            })
            .collect();
        // the lexicon's own conversions the engine missed (大島村 for
        // おおしまむら, where it gave おおしま村) and the homophones
        // of its words (検収担当者 for 研修担当者), those no more than
        // `ADDED_OVER` dearer than the engine's cheapest candidate,
        // scored `ADDED_BELOW` under its top: the lexicon's cost and the
        // reranker place them
        let top = result.scores[0];
        let cheapest = costs.iter().flatten().fold(f32::INFINITY, |a, &b| a.min(b));
        for (surface, cost) in lex.convert(&r, crate::lexicon::CONVERSIONS) {
            if cost > cheapest + crate::lexicon::ADDED_OVER
                || result
                    .candidates
                    .iter()
                    .any(|c| c.nfkc().eq(surface.chars()))
            {
                continue;
            }
            order.push(result.candidates.len());
            result.candidates.push(surface);
            scores.push(top - crate::lexicon::ADDED_BELOW);
            costs.push(Some(cost));
            if result.reading_pos.len() == n {
                result.reading_pos.push(Vec::new());
            }
            n += 1;
        }
        let best = costs.iter().flatten().fold(f32::INFINITY, |a, &b| a.min(b));
        if best.is_finite() {
            let worst = best + crate::lexicon::MARGIN;
            // the engine's top stays whatever it costs, and so does a
            // one-char word of a reading of a kana or two (`char_words`)
            let chars: Vec<String> = lex.char_words(&r).into_iter().map(|(w, _)| w).collect();
            let keep = |i: &usize| {
                *i == 0
                    || Some(*i) == model_top
                    || costs[*i].is_some_and(|c| c <= worst)
                    || chars
                        .iter()
                        .any(|w| result.candidates[*i].nfkc().eq(w.chars()))
            };
            order.retain(keep);
            rest.retain(keep);
            if order.is_empty() {
                std::mem::swap(&mut order, &mut rest);
            }
            for (i, c) in costs.iter().enumerate() {
                scores[i] -= crate::lexicon::WEIGHT * c.unwrap_or(worst);
            }
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            if let Some(pos) = model_top.and_then(|t| order.iter().position(|&i| i == t)) {
                let t = order.remove(pos);
                scores[t] = order.iter().map(|&i| scores[i]).fold(scores[t], f32::max);
                order.insert(0, t);
            }
        }
    }
    if scored {
        let floor = order
            .iter()
            .map(|&i| scores[i])
            .fold(f32::INFINITY, f32::min);
        for &i in &rest {
            scores[i] = scores[i].min(floor);
        }
    }
    order.extend(rest);
    fn permute<T>(v: &mut Vec<T>, order: &[usize]) {
        let mut old: Vec<Option<T>> = std::mem::take(v).into_iter().map(Some).collect();
        *v = order.iter().map(|&i| old[i].take().unwrap()).collect();
    }
    permute(&mut result.candidates, &order);
    if result.reading_pos.len() == n {
        permute(&mut result.reading_pos, &order);
    }
    if scored {
        result.scores = order.iter().map(|&i| scores[i]).collect();
    }
}

/// The engine's candidates more than `NO_LEXICON_MARGIN` below its top
/// dropped, the top kept (the reading and its katakana come after, from
/// `kana_candidates::arrange`).
fn drop_far_below_top(result: &mut EngineResult) {
    let n = result.candidates.len();
    if result.scores.len() != n || n < 2 {
        return;
    }
    let top = result
        .scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let keep: Vec<bool> = result
        .scores
        .iter()
        .enumerate()
        .map(|(i, &s)| i == 0 || s >= top - NO_LEXICON_MARGIN)
        .collect();
    let mut k = keep.iter();
    result.candidates.retain(|_| *k.next().unwrap());
    let mut k = keep.iter();
    result.scores.retain(|_| *k.next().unwrap());
    if result.reading_pos.len() == n {
        let mut k = keep.iter();
        result.reading_pos.retain(|_| *k.next().unwrap());
    }
}

/// The first candidates by the engine's score plus lambda times the LM's,
/// those that spell the reading out above those that do not (the LM lifted
/// おしま村 for おおしまむら over おおしま村; kept below, EM1 over the top 20
/// of wiki_fresh_v2 79.21 -> 79.27, AJIMEE 81.0 -> 81.5).
pub(super) fn rerank_order(
    scores: &[f32],
    logp: &[f64],
    lambda: f32,
    spelled: &[bool],
) -> Vec<usize> {
    let total = |i: usize| rerank_total(scores[i], logp[i], lambda);
    let first = |i: usize| spelled.get(i).copied().unwrap_or(true);
    let mut order: Vec<usize> = (0..scores.len().min(logp.len())).collect();
    order.sort_by(|&a, &b| first(b).cmp(&first(a)).then(total(b).total_cmp(&total(a))));
    order
}

/// `converter + lambda * LM`, or minus infinity where that is not a finite
/// number (a score the converter or the LM could not give), so that it
/// sorts last and the order is total.
pub(super) fn rerank_total(score: f32, logp: f64, lambda: f32) -> f64 {
    let t = f64::from(score) + f64::from(lambda) * logp;
    if t.is_finite() {
        t
    } else {
        f64::NEG_INFINITY
    }
}

impl EngineBridge {
    /// The engine's candidates in the IME's order (`order_candidates`),
    /// with this bridge's lexicon and n-gram.
    pub(super) fn order_engine_result(&self, result: &mut EngineResult) {
        self.order_engine_result_with(result, true);
    }

    /// `order_engine_result`, with or without the lexicon's costs
    /// (`order_candidates_with`).
    pub(super) fn order_engine_result_with(&self, result: &mut EngineResult, lexicon_costs: bool) {
        let ngram = self
            .ngram
            .as_ref()
            .map(|(t, w2, w3)| (t.as_ref(), *w2, *w3));
        order_candidates_with(self.lexicon.as_deref(), ngram, result, lexicon_costs);
    }

    /// Whether the lexicon's costs would change the order of a result: a
    /// whole lexicon is loaded (`order_candidates`).
    pub(super) fn lexicon_ranks(&self) -> bool {
        self.lexicon.as_deref().is_some_and(|l| !l.partial())
    }

    /// Space: the other reading to offer, or `None`: with English in the
    /// composition, a Shift released a key late put right
    /// (`ComposingText::shift_slip_alt`), or else every letter as romaji; a
    /// typo's correction; ず and づ the other way
    /// (`ComposingText::voicing_alt`).
    pub(super) fn alt_for_space(&self) -> Option<AltList> {
        // a typo: the reading one edit away, when a letter reads as no kana
        // and nothing was read as English
        let typo = || {
            // typed in capitals, English was meant (mixed input reads a
            // lowercase word it does not know as English too: kais + が)
            let shifted = self
                .composing
                .romaji()
                .chars()
                .any(|c| c.is_ascii_uppercase());
            if !self.cfg.typo_correction || shifted {
                return None;
            }
            crate::typo::correct(
                &self.composing.romaji(),
                self.composing.mixed_model().as_deref(),
            )
            .or_else(|| self.composing.voicing_alt())
        };
        if !self.composing.is_mixed() {
            return typo().map(|reading| AltList {
                reading: Some(reading),
                req_id: None,
                awaited: false,
                candidates: Vec::new(),
                main: Vec::new(),
                ends_english: false,
                typed: String::new(),
                typed_early: false,
                user_words: Vec::new(),
            });
        }
        // a Shift released a key late (IMENo for IMEの): the capitals before
        // it were English, so it goes before every letter as romaji (いめの)
        let slip = || {
            self.cfg
                .typo_correction
                .then(|| self.composing.shift_slip_alt())
                .flatten()
        };
        let reading = self.composing.kana_reading();
        let mixed = self.composing.full_reading();
        let other = reading != mixed && !reading.chars().any(|c| c.is_ascii_alphabetic());
        Some(AltList {
            reading: slip().or_else(|| if other { Some(reading) } else { typo() }),
            req_id: None,
            awaited: false,
            candidates: Vec::new(),
            main: Vec::new(),
            ends_english: self.composing.ends_in_english(),
            typed: self.composing.romaji(),
            typed_early: self.composing.is_known_word(),
            user_words: Vec::new(),
        })
    }

    /// The list to show for the engine's list `main` of the reading open:
    /// with mixed input's other candidates, and the rewrite first.
    pub(super) fn shown_list(&mut self, main: Vec<String>) -> Vec<String> {
        let mut list = self.mixed_list(main);
        if let Some(text) = self.rewrite_shown() {
            list.retain(|c| *c != text);
            list.insert(0, text);
        }
        list
    }

    /// The rewrite of the reading typed, once it is in.
    pub(super) fn rewrite_shown(&self) -> Option<String> {
        self.comp
            .rewrite
            .as_ref()
            .filter(|j| j.reading == self.composing.hiragana())
            .and_then(|j| j.text.clone())
    }

    /// The list to show for the engine's list `main` of the mixed reading:
    /// with the other reading's candidates and the input as typed (see
    /// `AltList`). Unchanged without one.
    fn mixed_list(&mut self, main: Vec<String>) -> Vec<String> {
        let Some(alt) = self.comp.alt.as_mut() else {
            return main;
        };
        alt.main = main.clone();
        // the user's words of the other reading first, all of them (よろ:
        // ten registered, read yoro after English: all ten offered)
        let (mine, rest): (Vec<&String>, Vec<&String>) = alt
            .candidates
            .iter()
            .partition(|c| alt.user_words.contains(c));
        let other: Vec<String> = mine
            .into_iter()
            .chain(rest.into_iter().take(ALT_MAX))
            .cloned()
            .collect();
        let (other_early, other_late) = other.split_at(other.len().min(ALT_EARLY));
        let mut list: Vec<String> = Vec::new();
        if alt.ends_english {
            list.extend(main.first().map(|top| format!("{top} ")));
            list.extend(other_early.iter().cloned());
            list.extend(main);
        } else {
            let (early, late) = main.split_at(main.len().min(ALT_EARLY));
            list.extend(early.iter().cloned());
            list.extend(other_early.iter().cloned());
            list.extend(late.iter().cloned());
        }
        list.extend(other_late.iter().cloned());
        if alt.typed_early {
            list.insert(list.len().min(2), alt.typed.clone());
        } else {
            list.push(alt.typed.clone());
        }
        let mut seen = std::collections::HashSet::new();
        list.retain(|c| !c.is_empty() && seen.insert(c.clone()));
        // the user's choices over the whole list: one of the other
        // reading's chosen before (IMEの for IMENo) comes first as one of
        // the mixed reading's would, which `finish_list` reranked alone
        let reading = self.composing.hiragana().to_string();
        list = self.rerank_learned(&reading, list);
        // what Backspace left of the conversion shown stays first
        if let Some((_, text, _)) = self.comp.kept.as_ref().filter(|(r, _, _)| *r == reading) {
            if let Some(i) = list.iter().position(|c| c == text) {
                let kept = list.remove(i);
                list.insert(0, kept);
            }
        }
        list
    }

    /// The engine's candidates for `reading` as the IME lists them:
    /// post-processing, kana forms, symbols, char rules and (for a whole
    /// reading) the learned reranking.
    pub(super) fn finish_list(
        &mut self,
        reading: &str,
        corrected: Option<&str>,
        raw: Vec<String>,
        rerank: bool,
    ) -> Vec<String> {
        let processed =
            crate::postprocess::apply(reading, raw, self.cfg.symbol_width, &self.ctx.context);
        let processed = if self.cfg.drop_untyped_symbols {
            crate::postprocess::drop_untyped_symbols(reading, processed)
        } else {
            processed
        };
        let processed = crate::postprocess::drop_changed_digits(reading, processed);
        let processed = crate::postprocess::drop_broken_numbers(reading, processed);
        let processed = crate::postprocess::drop_misread_digits(reading, processed);
        let processed = crate::postprocess::drop_moved_sentence_ends(reading, processed);
        let processed = crate::postprocess::drop_script_flips(processed);
        let processed = crate::postprocess::drop_lone_katakana(processed);
        // The English words of a mixed reading stay as typed.
        let processed = if self.composing.is_mixed() {
            processed
                .into_iter()
                .filter(|c| crate::mixed::keeps_latin(reading, c))
                .collect()
        } else {
            processed
        };
        let mut arranged = crate::kana_candidates::arrange(reading, corrected, processed);
        // a reading of one word: the lexicon's words for it that the model
        // left out (検収 for けんしゅう), after its first candidates; not
        // for a reading with English in it (`is_mixed` says only that the
        // mixed-input model is loaded, as it is by default)
        let english = reading
            .chars()
            .any(|c| crate::postprocess::to_half(c).is_ascii_alphabetic());
        if let (Some(lex), false) = (&self.lexicon, english) {
            use unicode_normalization::UnicodeNormalization;
            let words: Vec<String> = lex
                .word_candidates(reading)
                .into_iter()
                .map(|(w, _)| w)
                .filter(|w| !arranged.iter().any(|c| c.nfkc().eq(w.nfkc())))
                .collect();
            let at = crate::lexicon::WORDS_AT.min(arranged.len());
            arranged.splice(at..at, words);
            // a reading of a kana or two: its one-char words however dear
            // (阿 吾 唖 for あ), after the first page
            let chars: Vec<String> = lex
                .char_words(reading)
                .into_iter()
                .map(|(w, _)| w)
                .filter(|w| !arranged.iter().any(|c| c.nfkc().eq(w.nfkc())))
                .collect();
            let at = crate::lexicon::CHAR_WORDS_AT.min(arranged.len());
            arranged.splice(at..at, chars);
        }
        if let Some(dict) = &self.symbol_dict {
            arranged = dict.insert(reading, arranged);
        }
        if self.cfg.date_time_enabled && crate::date_time::is_reading(reading) {
            self.comp.date_time_now.get_or_insert_with(Local::now);
            arranged =
                crate::date_time::insert(reading, arranged, self.date_time_candidates(reading));
        }
        if self.cfg.numbers_enabled {
            arranged =
                crate::numbers::insert(reading, arranged, crate::numbers::candidates(reading));
        }
        let mut list = self.cfg.char_rules.apply(arranged);
        // after the symbol widths: a kaomoji stays as it is drawn
        if self.cfg.emoji_enabled {
            list = crate::emoji::insert(reading, list);
        }
        if let Some(dict) = self.user_dict() {
            list = dict.apply(reading, list);
        }
        if rerank {
            list = self.rerank_learned(reading, list);
        }
        if let Some(hidden) = self.hidden.as_mut() {
            list = hidden.filter(reading, list);
        }
        // what Backspace left of the conversion shown stays first
        if let Some((_, text, _)) = self.comp.kept.as_ref().filter(|(r, _, _)| r == reading) {
            list.retain(|c| c != text);
            list.insert(0, text.clone());
        }
        list
    }

    pub(super) fn date_time_candidates(&self, reading: &str) -> Vec<String> {
        if !self.cfg.date_time_enabled || !crate::date_time::is_reading(reading) {
            return Vec::new();
        }
        crate::date_time::candidates(reading, &self.comp.date_time_now.unwrap_or_else(Local::now))
    }

    /// `list` reranked by the learned choices, except that they never move a
    /// date or a time the reading generates first: those read as another one
    /// each day (a date chosen once for きょう then came first as soon as
    /// きょう was typed), and they are not learned (`commit_converting`).
    pub(super) fn rerank_learned(&mut self, reading: &str, list: Vec<String>) -> Vec<String> {
        let dates = self.date_time_candidates(reading);
        let Some(learning) = self.learning.as_mut() else {
            return list;
        };
        let out = learning.rerank(reading, list.clone());
        if out.first() != list.first() && out.first().is_some_and(|t| dates.contains(t)) {
            list
        } else {
            out
        }
    }
}
