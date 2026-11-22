//! Async result handling (engine conversion, alternate reading, LM rerank,
//! LLM rewrite), deferred preview ticks, and composition invalidation.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::engine_client::EngineResult;

use super::ranking::{rerank_order, rerank_total};
use super::state::{
    rewriting, Deferred, Phase, RerankJob, RewriteJob, ALT_WAIT, ENTER_WAIT, RERANK_GAP,
};
use super::{Action, EngineBridge};

impl EngineBridge {
    /// Called on the UI thread when the engine client notifies a completed
    /// convert. We accept the result as long as its `reading`
    /// is still a **prefix** of the current hiragana — if the user kept
    /// typing while the engine was converting, the newly-typed tail will
    /// show as raw kana after the cached prefix (no whole-string
    /// flicker). If the user back-spaced past the cached prefix the
    /// result is dropped.
    pub fn apply_async_result(&mut self) -> Vec<Action> {
        let mut actions = self.apply_engine_result();
        actions.extend(self.apply_rerank_result());
        actions.extend(self.apply_rewrite_result());
        actions
    }

    fn apply_engine_result(&mut self) -> Vec<Action> {
        let Some(client) = self.client.as_ref() else {
            return Vec::new();
        };
        let Some(mut result) = client.take_latest() else {
            return Vec::new();
        };
        if result.req_id < self.result_floor {
            return Vec::new();
        }
        // live typing: the preview shows the first candidate, which the
        // lexicon's costs never change; the whole list, which waits for them,
        // is made when it is wanted (`finish_list_now`): at the frontend's
        // next idle (`tick`), or at a key but one typed into the reading
        let quick = self.cfg.live_conversion
            && matches!(self.phase, Phase::Composing { .. })
            && self.reranker.is_none()
            && self.segmented().is_none()
            && self.lexicon_ranks()
            && !self
                .comp
                .alt
                .as_ref()
                .is_some_and(|alt| alt.req_id == Some(result.req_id));
        let unordered = quick.then(|| result.clone());
        self.order_engine_result_with(&mut result, !quick);
        if self
            .comp
            .alt
            .as_ref()
            .is_some_and(|alt| alt.req_id == Some(result.req_id))
        {
            return self.apply_alt_result(result);
        }
        if let Some(actions) = self.apply_segment_result(&result) {
            return actions;
        }
        let raw = std::mem::take(&mut result.candidates);
        let list = self.finish_list(
            &result.reading,
            result.corrected_reading.as_deref(),
            raw.clone(),
            true,
        );
        let starts = self.list_starts_of(&result.reading, &raw, &result.reading_pos, &list);
        result.candidates = list;
        let current_hira = self.composing.live_reading();

        // Live typing: accept the result while its reading is still a prefix
        // of the current reading; the freshly-typed tail then shows as raw
        // kana after the cached prefix.
        if matches!(self.phase, Phase::Composing { .. }) {
            if !current_hira.starts_with(&result.reading) {
                return Vec::new();
            }
            let top = result.candidates.first().cloned().unwrap_or_default();
            if top.is_empty() {
                return Vec::new();
            }
            if let Phase::Composing { candidates } = &mut self.phase {
                *candidates = result.candidates.clone();
            }
            self.comp.list_reading = result.reading.clone();
            self.comp.list_starts = starts;
            self.comp.unfinished = unordered;
            // the conversion of all that is typed: rerank it once typing pauses
            let whole = result.reading == current_hira;
            // one that reads a pending n as ん is what Enter commits; the
            // preview keeps the conversion before it and shows the n, as
            // the next key may make it な (共犯 for きょうはn, then the whole
            // in kana while きょうはな was converted)
            let n_read = whole && self.composing.n_pending();
            if whole {
                self.start_rerank(
                    &result.reading,
                    result.corrected_reading.as_deref(),
                    &raw,
                    &result.scores,
                    &result.reading_pos,
                    false,
                );
            }
            let reranking = self
                .comp
                .rerank
                .as_ref()
                .is_some_and(|j| j.reading == result.reading);
            if n_read {
                self.comp.deferred = None;
            } else if whole && reranking && self.cfg.rerank_cfg.defer_ms > 0 {
                // shown once the reranked top is in, or at the deadline: the
                // preedit changes once (what was shown stays meanwhile)
                self.comp.deferred = Some(Deferred {
                    reading: result.reading.clone(),
                    top,
                    until: Instant::now()
                        + Duration::from_millis(
                            self.cfg.rerank_cfg.idle_ms + self.cfg.rerank_cfg.defer_ms,
                        ),
                });
            } else {
                self.comp.deferred = None;
                self.cache_conversion(&result.reading, &result.candidates);
            }
            self.comp.last_applied_reading = current_hira;
            return vec![Action::UpdatePreedit {
                text: self.live_preedit(),
            }];
        }

        // Candidate window open: the reading is frozen, so accept only an
        // exact match and swap the provisional list for the real one.
        if self.form_shown().is_some() {
            return Vec::new();
        }
        let (selected, chosen) = match &self.phase {
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => (
                *selected,
                (*selected > 0)
                    .then(|| candidates.get(*selected).cloned())
                    .flatten(),
            ),
            _ => return Vec::new(),
        };
        if current_hira != result.reading || result.candidates.is_empty() {
            return Vec::new();
        }
        // the list opens with the other reading's
        self.request_alt();
        self.await_alt();
        let candidates = self.shown_list(result.candidates.clone());
        // Locally generated candidates may move when the model arrives.
        // Keep a deliberate choice on its text rather than its old index.
        let sel = chosen
            .and_then(|c| candidates.iter().position(|t| *t == c))
            .unwrap_or(if selected < candidates.len() {
                selected
            } else {
                0
            });
        let Some(text) = candidates.get(sel).cloned() else {
            return Vec::new();
        };
        let text = self.with_pending(text);
        let list = candidates;
        // the candidate window is open: rerank at once
        self.start_rerank(
            &result.reading,
            result.corrected_reading.as_deref(),
            &raw,
            &result.scores,
            &result.reading_pos,
            true,
        );
        self.comp.list_reading = result.reading;
        self.comp.list_starts = starts;
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        vec![
            Action::UpdatePreedit { text },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
        ]
    }

    /// Ask the engine for the other reading's list, once the mixed one is in.
    pub(super) fn request_alt(&mut self) {
        let Some(alt) = self.comp.alt.as_mut() else {
            return;
        };
        let Some(reading) = &alt.reading else {
            return;
        };
        if alt.req_id.is_some() {
            return;
        }
        if let Some(client) = &self.client {
            let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
            client.dispatch(req_id, &self.ctx.context, reading);
            alt.req_id = Some(req_id);
        }
    }

    /// Space: wait, once, for the other reading's list just asked for
    /// (`request_alt`), at most `ALT_WAIT`, so that the list opens with it
    /// and with the user's choices over the whole (`mixed_list`) rather than
    /// changing under the user when it comes. Results of earlier requests
    /// met on the way are stale: the other reading was asked for after
    /// them.
    pub(super) fn await_alt(&mut self) {
        match self.comp.alt.as_mut() {
            Some(alt) if alt.req_id.is_some() && !alt.awaited => alt.awaited = true,
            _ => return,
        }
        let deadline = Instant::now() + ALT_WAIT;
        loop {
            let Some(client) = self.client.as_ref() else {
                return;
            };
            if !client.is_ready() || !client.is_alive() {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !client.wait_latest(left) {
                crate::diagnostics::log("space: the other reading did not come in time");
                return;
            }
            let Some(mut result) = client.take_latest() else {
                continue;
            };
            if result.req_id < self.result_floor {
                continue;
            }
            self.order_engine_result(&mut result);
            if self
                .comp
                .alt
                .as_ref()
                .is_some_and(|a| a.req_id == Some(result.req_id))
            {
                self.store_alt_result(result);
                return;
            }
        }
    }

    /// Keep the other reading's list for the list shown (`mixed_list`);
    /// `false` when `result` is not of the other reading.
    fn store_alt_result(&mut self, result: EngineResult) -> bool {
        let Some(reading) = self
            .comp
            .alt
            .as_ref()
            .and_then(|alt| alt.reading.clone())
            .filter(|r| *r == result.reading)
        else {
            return false;
        };
        let candidates = self.finish_list(
            &reading,
            result.corrected_reading.as_deref(),
            result.candidates,
            false,
        );
        let user_words = self
            .user_dict()
            .map(|d| d.words(&reading))
            .unwrap_or_default();
        if let Some(alt) = self.comp.alt.as_mut() {
            alt.candidates = candidates;
            alt.user_words = user_words;
        }
        true
    }

    /// The other reading's list arrived after the list opened (it did not
    /// come within `ALT_WAIT`): add it to the open list, the selection
    /// staying on its candidate.
    fn apply_alt_result(&mut self, result: EngineResult) -> Vec<Action> {
        let Some(main) = self.comp.alt.as_ref().map(|alt| alt.main.clone()) else {
            return Vec::new();
        };
        if !self.store_alt_result(result) {
            return Vec::new();
        }
        // an F6–F10 form shown stays: the list is there again when it is left
        if self.form_shown().is_some() {
            return Vec::new();
        }
        let shown = match &self.phase {
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => candidates.get(*selected).cloned(),
            _ => return Vec::new(),
        };
        let list = self.shown_list(main);
        let sel = shown
            .and_then(|t| list.iter().position(|c| *c == t))
            .unwrap_or(0);
        let Some(text) = list.get(sel).cloned() else {
            return Vec::new();
        };
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        vec![
            Action::UpdatePreedit {
                text: self.with_pending(text),
            },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
        ]
    }

    /// Rerank the whole-reading conversion just shown: its first candidates
    /// once typing pauses (`now`: at once, the candidate window being open),
    /// then more while the user stays idle. The same conversion again only
    /// hurries a request still waiting.
    fn start_rerank(
        &mut self,
        reading: &str,
        corrected: Option<&str>,
        raw: &[String],
        scores: &[f32],
        reading_pos: &[Vec<u32>],
        now: bool,
    ) {
        if self.reranker.is_none() || raw.len() < 2 || scores.len() != raw.len() {
            return;
        }
        if let Some(job) = &self.comp.rerank {
            if job.reading == reading && job.raw == raw {
                if now {
                    self.rerank_now();
                }
                return;
            }
        }
        let max = self.rerank_max(raw.len());
        let k = self
            .cfg
            .rerank_cfg
            .first_candidates
            .clamp(2, max.max(2))
            .min(max);
        let due = if now {
            Instant::now()
        } else {
            Instant::now() + Duration::from_millis(self.cfg.rerank_cfg.idle_ms)
        };
        // a candidate of the corrected reading spells it, not the one typed:
        // it is no less spelled out for that
        let spelled = {
            use unicode_normalization::UnicodeNormalization;
            let chars = |s: &str| s.nfkc().collect::<Vec<char>>();
            let r = chars(reading);
            let fixed = corrected.map(chars);
            raw.iter()
                .map(|c| {
                    crate::reading_align::spells_out(c, &r)
                        || fixed
                            .as_ref()
                            .is_some_and(|f| crate::reading_align::spells_out(c, f))
                })
                .collect()
        };
        self.comp.rerank = Some(RerankJob {
            reading: reading.to_string(),
            corrected_reading: corrected.map(str::to_string),
            context: self.ctx.context.clone(),
            raw: raw.to_vec(),
            scores: scores.to_vec(),
            spelled,
            reading_pos: reading_pos.to_vec(),
            k,
            req_id: 0,
            due,
        });
        self.send_rerank(due);
    }

    /// Candidates of a list of `n` reranked at most.
    fn rerank_max(&self, n: usize) -> usize {
        let info = self.reranker.as_ref().and_then(|r| r.info());
        let mut limit = self
            .cfg
            .rerank_cfg
            .max_candidates
            .or(info.map(|i| i.max_candidates))
            .unwrap_or(20);
        if let Some(i) = info {
            limit = limit.min(i.max_candidates);
        }
        limit.min(n)
    }

    /// The LM's weight: the config's, else the reranker's, else 0.3; a
    /// weight that is not a finite number is none.
    fn rerank_lambda(&self) -> f32 {
        let finite = |l: &f32| l.is_finite();
        self.cfg
            .rerank_cfg
            .lambda
            .filter(finite)
            .or(self
                .reranker
                .as_ref()
                .and_then(|r| r.info())
                .map(|i| i.lambda)
                .filter(finite))
            .unwrap_or(0.3)
    }

    fn send_rerank(&mut self, due: Instant) {
        let req_id = self.next_req_id();
        let (Some(client), Some(job)) = (self.reranker.as_ref(), self.comp.rerank.as_mut()) else {
            return;
        };
        job.req_id = req_id;
        job.due = due;
        client.dispatch(req_id, &job.context, &job.raw[..job.k], due);
    }

    /// The conversion key: a rerank waiting for the typing pause goes now.
    pub(super) fn rerank_now(&mut self) {
        if self
            .comp
            .rerank
            .as_ref()
            .is_some_and(|j| j.due > Instant::now())
        {
            self.send_rerank(Instant::now());
        }
    }

    /// The conversion being reranked is gone (typed over, committed,
    /// dropped): no more requests for it.
    pub(super) fn stop_rerank(&mut self) {
        self.comp.deferred = None;
        if self.comp.rerank.take().is_some() {
            if let Some(r) = &self.reranker {
                r.cancel();
            }
        }
    }

    /// Rewrite mode, a conversion just opened: ask for the rewrite of
    /// `reading`'s conversion `text` (once per reading) and say so.
    pub(super) fn start_rewrite(&mut self, reading: &str, text: &str) -> Option<Action> {
        // nothing sensitive leaves the machine: no rewrite of a token, a
        // URL or a password, and no context that holds one
        if !self.cfg.rewrite_mode
            || self.ctx.private_field
            || text.trim().is_empty()
            || crate::privacy::is_sensitive(reading)
            || crate::privacy::is_sensitive(text)
        {
            return None;
        }
        if let Some(job) = self.comp.rewrite.as_ref().filter(|j| j.reading == reading) {
            // asked already: in, or still on its way
            return job.text.is_none().then(rewriting);
        }
        let client = self.rewriter.as_ref()?;
        let req_id = self.next_req_id();
        let context = if crate::privacy::is_sensitive(&self.ctx.raw_context) {
            ""
        } else {
            &self.ctx.raw_context
        };
        client.dispatch(req_id, context, reading, text.trim_end());
        self.comp.rewrite = Some(RewriteJob {
            req_id,
            reading: reading.to_string(),
            text: None,
        });
        Some(rewriting())
    }

    /// The conversion is committed or dropped: no rewrite for it.
    pub(super) fn stop_rewrite(&mut self) {
        if self.comp.rewrite.take().is_some() {
            if let Some(r) = &self.rewriter {
                r.cancel();
            }
        }
    }

    /// A rewrite came back: first in the list open for its reading,
    /// selected unless the user moved off the top; or why it failed.
    fn apply_rewrite_result(&mut self) -> Vec<Action> {
        let Some(result) = self.rewriter.as_ref().and_then(|r| r.take_latest()) else {
            return Vec::new();
        };
        let Some(reading) = self
            .comp
            .rewrite
            .as_ref()
            .filter(|j| j.req_id == result.req_id)
            .map(|j| j.reading.clone())
        else {
            return Vec::new();
        };
        let open = self.composing.hiragana() == reading
            && self.form_shown().is_none()
            && matches!(self.phase, Phase::Converting { seg: None, .. });
        let Some(text) = result.text.filter(|t| !t.is_empty()) else {
            self.comp.rewrite = None;
            let code = result.code.unwrap_or_default();
            return if open {
                vec![Action::Status {
                    text: crate::rewrite_client::failure_message(&code).to_string(),
                }]
            } else {
                Vec::new()
            };
        };
        if let Some(job) = self.comp.rewrite.as_mut() {
            job.text = Some(text.clone());
        }
        crate::diagnostics::log(format!("rewrite applied ms={:.0}", result.ms));
        if !open {
            return Vec::new();
        }
        // segment editing cuts the rewrite too, as best the kana allow
        if let Some(starts) = crate::segments::char_starts(&reading, &text, None) {
            self.comp.list_starts.insert(text.clone(), starts);
        }
        let Phase::Converting {
            candidates,
            selected,
            ..
        } = &self.phase
        else {
            return Vec::new();
        };
        let shown = candidates.get(*selected).cloned();
        let at_top = *selected == 0;
        let mut list = candidates.clone();
        list.retain(|c| *c != text);
        list.insert(0, text);
        let sel = if at_top {
            0
        } else {
            shown
                .and_then(|t| list.iter().position(|c| *c == t))
                .unwrap_or(0)
        };
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: sel,
            seg: None,
        };
        vec![
            Action::UpdatePreedit {
                text: self.with_pending(list[sel].clone()),
            },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
            Action::Status {
                text: String::new(),
            },
        ]
    }

    /// A rerank result: the scored candidates reordered by `converter +
    /// lambda * LM` (the rest kept after them), the list rebuilt and shown,
    /// then more candidates asked for while the user stays idle.
    fn apply_rerank_result(&mut self) -> Vec<Action> {
        let Some(result) = self.reranker.as_ref().and_then(|r| r.take_latest()) else {
            return Vec::new();
        };
        let Some(job) = self.comp.rerank.as_ref() else {
            return Vec::new();
        };
        if result.req_id != job.req_id || result.logp.len() != job.k {
            return Vec::new();
        }
        let lambda = self.rerank_lambda();
        let order = rerank_order(
            &job.scores[..job.k],
            &result.logp,
            lambda,
            &job.spelled[..job.k],
        );
        if order.is_empty() {
            return Vec::new();
        }
        // the scored candidates far below the best go (`RERANK_GAP`); the
        // best stays, and with no finite best (scores that are not
        // numbers) none goes
        let total = |i: usize| rerank_total(job.scores[i], result.logp[i], lambda);
        let best = total(order[0]);
        let order: Vec<usize> = order
            .into_iter()
            .enumerate()
            .filter(|&(at, i)| at == 0 || !best.is_finite() || best - total(i) <= RERANK_GAP)
            .map(|(_, i)| i)
            .collect();
        let reorder = |v: &[String]| -> Vec<String> {
            order
                .iter()
                .map(|&i| v[i].clone())
                .chain(v[job.k..].iter().cloned())
                .collect()
        };
        let raw = reorder(&job.raw);
        let reading_pos: Vec<Vec<u32>> = if job.reading_pos.len() == job.raw.len() {
            order
                .iter()
                .map(|&i| job.reading_pos[i].clone())
                .chain(job.reading_pos[job.k..].iter().cloned())
                .collect()
        } else {
            Vec::new()
        };
        let reading = job.reading.clone();
        let corrected = job.corrected_reading.clone();
        let (k, n) = (job.k, job.raw.len());
        crate::diagnostics::log(if crate::diagnostics::log_typed_text() {
            format!(
                "rerank applied k={k} ms={:.1} top={:?} (was {:?})",
                result.score_ms, raw[0], job.raw[0]
            )
        } else {
            format!(
                "rerank applied k={k} ms={:.1} top changed={}",
                result.score_ms,
                raw[0] != job.raw[0]
            )
        });
        let list = self.finish_list(&reading, corrected.as_deref(), raw.clone(), true);
        let starts = self.list_starts_of(&reading, &raw, &reading_pos, &list);
        let actions = self.show_reranked(&reading, list, starts);
        let max = self.rerank_max(n);
        if k < max {
            if let Some(job) = self.comp.rerank.as_mut() {
                job.k = (k * 2).min(max);
            }
            self.send_rerank(Instant::now());
        }
        actions
    }

    /// Show a reranked list of `reading` where its conversion is shown: the
    /// typing preedit, or the open window (the selection kept on the
    /// candidate it was on, unless that was the top).
    fn show_reranked(
        &mut self,
        reading: &str,
        list: Vec<String>,
        starts: HashMap<String, Vec<usize>>,
    ) -> Vec<Action> {
        if self.comp.list_reading != reading || list.is_empty() || self.form_shown().is_some() {
            return Vec::new();
        }
        match &mut self.phase {
            Phase::Composing { candidates } if self.composing.live_reading() == reading => {
                *candidates = list.clone();
            }
            Phase::Converting {
                candidates,
                selected,
                seg: None,
            } => {
                let shown = candidates.get(*selected).cloned();
                let at_top = *selected == 0;
                let list = self.shown_list(list);
                let sel = if at_top {
                    0
                } else {
                    shown
                        .and_then(|t| list.iter().position(|c| *c == t))
                        .unwrap_or(0)
                };
                let Some(text) = list.get(sel).cloned() else {
                    return Vec::new();
                };
                if let Phase::Converting {
                    candidates,
                    selected,
                    ..
                } = &mut self.phase
                {
                    *candidates = list.clone();
                    *selected = sel;
                }
                self.comp.list_starts = starts;
                return vec![
                    Action::UpdatePreedit {
                        text: self.with_pending(text),
                    },
                    Action::ShowCandidates {
                        list,
                        selected: sel,
                    },
                ];
            }
            _ => return Vec::new(),
        }
        self.comp.list_starts = starts;
        // the reranked top is what the deferred preedit waited for
        let waited = self
            .comp
            .deferred
            .as_ref()
            .is_some_and(|d| d.reading == reading);
        if waited {
            self.comp.deferred = None;
        }
        if waited || self.comp.cached_prefix_reading == reading {
            self.cache_conversion(reading, &list);
        }
        vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]
    }

    /// Enter on a live conversion whose reading is still being converted
    /// (typed after the last result came in): wait for its conversion, at
    /// most `ENTER_WAIT`, so that Enter commits it rather than the
    /// conversion before with the kana typed since (the whole in kana when
    /// none was in). Results for a shorter reading are taken on the way.
    /// Not when Enter commits the kana anyway (the caret moved, Escape, a
    /// reading too short to convert, mixed input reading the whole
    /// otherwise, unless `done`: that reading is converted now, typing done)
    /// or the engine is not serving.
    pub(super) fn await_live_conversion(&mut self, done: bool) {
        if !self.cfg.live_conversion
            || !matches!(self.phase, Phase::Composing { .. })
            || !self.composing.cursor_at_end()
            || self.comp.kana_shown
            || !done && self.composing.reads_otherwise_when_done()
        {
            return;
        }
        let reading = self.composing.live_reading();
        if reading.chars().count() < self.cfg.live_min_dispatch_chars {
            return;
        }
        let deadline = Instant::now() + ENTER_WAIT;
        while self.comp.list_reading != reading {
            let Some(client) = self.client.as_ref() else {
                return;
            };
            if !client.is_ready() || !client.is_alive() {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !client.wait_latest(left) {
                crate::diagnostics::log("enter: the conversion did not come in time");
                return;
            }
            // what it shows is committed next: its actions go
            let _ = self.apply_engine_result();
        }
    }

    /// The frontend's timer (`deferred_for`): the list the preview went
    /// without (`finish_list_now`), and the deferred preedit once its
    /// deadline passed: the engine's top, without the reranked one.
    pub fn tick(&mut self) -> Vec<Action> {
        let finished = self.finish_list_now(false);
        if !finished.is_empty() {
            return finished;
        }
        let Some(d) = self.comp.deferred.take() else {
            return Vec::new();
        };
        if Instant::now() < d.until {
            self.comp.deferred = Some(d);
            return Vec::new();
        }
        let list = match &self.phase {
            Phase::Composing { candidates } if self.composing.live_reading() == d.reading => {
                if candidates.first() == Some(&d.top) {
                    candidates.clone()
                } else {
                    vec![d.top]
                }
            }
            _ => return Vec::new(),
        };
        self.cache_conversion(&d.reading, &list);
        vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]
    }

    /// The whole list of the result the live preview went without it for
    /// (`unfinished`): the lexicon's costs and conversions, and what
    /// `finish_list` does after them. `keep_shown`: a key came before the
    /// frontend's idle, and the conversion on screen, which it acts on,
    /// stays first; else the preview takes the list's first when that is
    /// another one (a candidate the lexicon added, learned for the reading),
    /// and these are its actions.
    pub(super) fn finish_list_now(&mut self, keep_shown: bool) -> Vec<Action> {
        let Some(mut result) = self.comp.unfinished.take() else {
            return Vec::new();
        };
        let shown = match &self.phase {
            Phase::Composing { candidates } if self.comp.list_reading == result.reading => {
                candidates.first().cloned()
            }
            _ => return Vec::new(),
        };
        self.order_engine_result(&mut result);
        let raw = std::mem::take(&mut result.candidates);
        let mut list = self.finish_list(
            &result.reading,
            result.corrected_reading.as_deref(),
            raw.clone(),
            true,
        );
        let mut starts = self.list_starts_of(&result.reading, &raw, &result.reading_pos, &list);
        let other = !list.is_empty() && list.first() != shown.as_ref();
        if let (true, true, Some(top)) = (other, keep_shown, &shown) {
            list.retain(|c| c != top);
            list.insert(0, top.clone());
            if let Some(s) = self.comp.list_starts.get(top) {
                starts.entry(top.clone()).or_insert_with(|| s.clone());
            }
        }
        if let Phase::Composing { candidates } = &mut self.phase {
            *candidates = list.clone();
        }
        self.comp.list_starts = starts;
        // the preview's conversion of that reading, where it shows it (not
        // one kept back for a pending n)
        if !other || keep_shown || self.comp.cached_prefix_reading != result.reading {
            return Vec::new();
        }
        self.cache_conversion(&result.reading, &list);
        vec![Action::UpdatePreedit {
            text: self.live_preedit(),
        }]
    }

    /// Forget what belonged to the composition that ended (committed,
    /// dropped, deleted to nothing): its list and the reading it is of, the
    /// conversion cached for the preview, the F6–F10 form, the other
    /// reading's list, the prediction open, and the rerank and the rewrite
    /// asked for it; so that none of it is taken for the next one's (the
    /// same reading typed again and Enter at once committed the kana, not
    /// waiting for its conversion; a form kept refused Space's list).
    pub(super) fn clear_composition(&mut self) {
        self.stop_rerank();
        self.stop_rewrite();
        self.comp.clear();
    }

    pub(super) fn invalidate_engine_results(&mut self) {
        self.result_floor = self.req_counter.load(Ordering::Relaxed) + 1;
        self.comp.date_time_now = None;
    }
}
