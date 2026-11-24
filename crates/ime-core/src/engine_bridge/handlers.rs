//! Key handlers, candidate window navigation, predictions, and commit transitions.

use chrono::Local;

use crate::char_forms::{self, Form};

use super::state::{consumed, not_consumed, FormShown, Phase};
use super::{Action, EngineBridge, KeyOutcome};

impl EngineBridge {
    /// Ctrl+Delete on the candidate selected (Mozc's
    /// DeleteSelectedCandidate): what was learned to put it up is forgotten;
    /// when nothing was, it is hidden for this reading from now on
    /// (`hidden.jsonl`). Either way it leaves the list open. The kana of the
    /// reading and a list's last candidate stay in it (and are not hidden).
    /// On the predictions open (Tab while typing), the prediction is
    /// forgotten instead (`forget_prediction`).
    pub fn handle_forget_candidate(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        let Some(reading) = self.shown_reading() else {
            return not_consumed();
        };
        if self.form_shown().is_some() {
            return consumed(Vec::new());
        }
        if self.comp.prediction.is_some() {
            return self.forget_prediction();
        }
        let (cand, count) = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &s.segs[s.focus];
                (f.cands.get(f.selected).cloned(), f.cands.len())
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => (candidates.get(*selected).cloned(), candidates.len()),
            _ => return not_consumed(),
        };
        let Some(cand) = cand else {
            return consumed(Vec::new());
        };
        let forgot = self
            .learning
            .as_mut()
            .is_some_and(|l| l.forget(&reading, &cand).unwrap_or(false));
        // a list's last candidate and the kana stay, learned or not
        let stays =
            count <= 1 || cand == reading || cand == crate::kana_candidates::to_katakana(&reading);
        let message = if forgot {
            "学習を取り消しました"
        } else if stays {
            return consumed(vec![Action::Status {
                text: "この候補は非表示にできません".to_string(),
            }]);
        } else if self.ctx.private_field {
            // a field that asks for no learning keeps no history, the
            // candidates hidden there neither
            return consumed(vec![Action::Status {
                text: "この欄では候補を非表示にできません".to_string(),
            }]);
        } else {
            match self.hidden.as_mut().map(|h| h.hide(&reading, &cand)) {
                Some(Ok(())) => "候補を非表示にしました",
                _ => return consumed(Vec::new()),
            }
        };
        let status = Action::Status {
            text: message.to_string(),
        };
        if stays {
            return consumed(vec![status]);
        }
        // out of the list open
        let mut actions = match &mut self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &mut s.segs[s.focus];
                f.cands.retain(|c| c != &cand);
                f.selected = f.selected.min(f.cands.len().saturating_sub(1));
                self.segment_actions()
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => {
                candidates.retain(|c| c != &cand);
                *selected = (*selected).min(candidates.len().saturating_sub(1));
                let (list, sel) = (candidates.clone(), *selected);
                match list.get(sel) {
                    Some(top) => vec![
                        Action::UpdatePreedit {
                            text: self.with_pending(top.clone()),
                        },
                        Action::ShowCandidates {
                            list,
                            selected: sel,
                        },
                    ],
                    // nothing left to show: back to typing
                    None => return consumed(self.close_list()),
                }
            }
            _ => Vec::new(),
        };
        actions.push(status);
        consumed(actions)
    }

    /// Ctrl+Delete on a prediction open: its commits are forgotten
    /// (`Predictor::forget`), so it is predicted no more, and it leaves the
    /// list; with none left, typing goes on. The learning and the hidden
    /// candidates stay as they are: a prediction is no conversion of what
    /// is typed.
    fn forget_prediction(&mut self) -> KeyOutcome {
        let Phase::Converting {
            candidates,
            selected,
            seg: None,
        } = &self.phase
        else {
            return consumed(Vec::new());
        };
        let Some(surface) = candidates.get(*selected).cloned() else {
            return consumed(self.close_list());
        };
        let reading = self
            .comp
            .prediction
            .iter()
            .flatten()
            .find(|p| p.surface == surface)
            .map(|p| p.reading.clone())
            .unwrap_or_default();
        let forgot = self.predictor.as_ref().is_some_and(|p| {
            p.borrow_mut()
                .forget(&reading, &surface)
                .unwrap_or_else(|e| {
                    tracing::warn!("prediction forget failed: {e}");
                    false
                })
        });
        if !forgot {
            return consumed(Vec::new());
        }
        if let Some(p) = self.comp.prediction.as_mut() {
            p.retain(|x| x.surface != surface);
        }
        let Phase::Converting {
            candidates,
            selected,
            ..
        } = &mut self.phase
        else {
            return consumed(Vec::new());
        };
        candidates.retain(|c| *c != surface);
        *selected = (*selected).min(candidates.len().saturating_sub(1));
        let (list, sel) = (candidates.clone(), *selected);
        let Some(top) = list.get(sel).cloned() else {
            return consumed(self.close_list());
        };
        consumed(vec![
            Action::UpdatePreedit { text: top },
            Action::ShowCandidates {
                list,
                selected: sel,
            },
            Action::Status {
                text: "予測候補を削除しました".to_string(),
            },
        ])
    }

    /// Close the candidate window for typing (a list emptied).
    fn close_list(&mut self) -> Vec<Action> {
        self.back_to_composing();
        vec![
            Action::HideCandidates,
            Action::UpdatePreedit {
                text: self.live_preedit(),
            },
        ]
    }

    /// Shift+letter: enter (or extend) the temporary `Alphabet` phase.
    /// Commits any pending hiragana/kanji composition first so the raw
    /// ASCII buffer is cleanly separated from what came before.
    pub fn handle_shifted_alpha(&mut self, c: char) -> KeyOutcome {
        if !c.is_ascii_alphabetic() {
            return not_consumed();
        }
        // kana input: Shift+letter is a kana key (Shift+E: ぃ)
        if crate::romaji_table::active().is_some_and(|t| t.kana_keys) {
            return self.handle_char(c);
        }
        // Mixed input: the letter joins the composition, its case marking
        // it English.
        if self.composing.is_mixed() {
            return self.handle_char(c);
        }
        let mut actions = match &self.phase {
            Phase::Empty => Vec::new(),
            Phase::Composing { .. } => {
                let outcome = self.handle_enter();
                outcome.actions
            }
            Phase::Converting { .. } => {
                let mut a = self.commit_converting();
                a.push(Action::HideCandidates);
                a
            }
            Phase::Alphabet { buffer } => {
                // Already in alphabet mode — just append.
                let mut buf = buffer.clone();
                buf.push(c);
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                return KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                };
            }
        };
        let buffer = c.to_string();
        let preedit = buffer.clone();
        self.phase = Phase::Alphabet { buffer };
        actions.push(Action::UpdatePreedit { text: preedit });
        KeyOutcome {
            consumed: true,
            actions,
        }
    }

    /// Whether `c` goes to the app as part of a command's name being typed
    /// (`pass_through_commands`).
    fn passes(&mut self, c: char) -> bool {
        if !self.ctx.passing || !matches!(self.phase, Phase::Empty) {
            self.ctx.passing = false;
            return false;
        }
        self.ctx.passing = c.is_ascii_alphanumeric() || "-_:./".contains(c);
        self.ctx.passing
    }

    pub fn handle_char(&mut self, c: char) -> KeyOutcome {
        if self.passes(c) {
            return not_consumed();
        }
        // a key the table reads, or a kana typed by a key of its own
        let table_key =
            crate::romaji_table::active().is_some_and(|t| t.uses(c) || ('ぁ'..='ゖ').contains(&c));
        if !c.is_ascii_alphabetic() && !c.is_ascii_digit() && c != '-' && !table_key {
            return not_consumed();
        }
        // In Alphabet phase, letters / digits / hyphen append verbatim
        // (case preserved) to the buffer.
        if let Phase::Alphabet { buffer } = &self.phase {
            let mut buf = buffer.clone();
            buf.push(c);
            let preedit = buf.clone();
            self.phase = Phase::Alphabet { buffer: buf };
            return KeyOutcome {
                consumed: true,
                actions: vec![Action::UpdatePreedit { text: preedit }],
            };
        }
        if self.cfg.digit_select && c.is_ascii_digit() && self.window_open() {
            return self.pick_by_digit(c);
        }
        self.type_char(c)
    }

    /// `c` typed into the composition, committing a conversion first.
    fn type_char(&mut self, c: char) -> KeyOutcome {
        if matches!(self.phase, Phase::Empty | Phase::Converting { .. }) {
            self.invalidate_engine_results();
        }
        match &self.phase {
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.composing.reset();
                self.composing.input_char(c);
                self.phase = Phase::Composing {
                    candidates: Vec::new(),
                };
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            _ => {
                self.composing.input_char(c);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_backspace(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.pop();
                if buf.is_empty() {
                    self.phase = Phase::Empty;
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                self.back_to_composing();
                let preedit = self.live_preedit();
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. } => {
                if let Some(out) = self.backspace_shown() {
                    return out;
                }
                self.composing.delete_left();
                if self.composing.is_empty() {
                    self.invalidate_engine_results();
                    self.phase = Phase::Empty;
                    self.clear_composition();
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_space(&mut self) -> KeyOutcome {
        let composing = matches!(self.phase, Phase::Composing { .. });
        let last = composing && self.cfg.space_last_segment;
        let out = self.convert_typed(true, last);
        if last {
            return self.on_last_segment(out);
        }
        out
    }

    /// `out`, the whole sentence converted (`convert_typed`), as segment
    /// editing on its last segment: the segments before it stay as
    /// converted, the list is that segment's. `out` as it is when the
    /// conversion cannot be cut (no list of its reading yet).
    fn on_last_segment(&mut self, out: KeyOutcome) -> KeyOutcome {
        if !out.consumed || !matches!(self.phase, Phase::Converting { seg: None, .. }) {
            return out;
        }
        if !self.ensure_segmented() {
            // the engine's list of the reading not in yet: cut once it is
            self.comp.last_segment_due = !self.comp.covers_typed;
            return out;
        }
        // the sentence's preedit and list give way to the segments'; what
        // else the conversion did stays
        let mut actions = self.focus_last_segment();
        actions.extend(out.actions.into_iter().filter(|a| {
            !matches!(
                a,
                Action::UpdatePreedit { .. } | Action::ShowCandidates { .. }
            )
        }));
        consumed(actions)
    }

    /// Segment editing on the last segment of the conversion cut, its list
    /// asked for: the segments' preedit and list.
    pub(super) fn focus_last_segment(&mut self) -> Vec<Action> {
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            s.focus = s.segs.len().saturating_sub(1);
        }
        self.query_focused_segment();
        self.segment_actions()
    }

    /// Live conversion shows the conversion of all that is typed (not the
    /// kana Escape brought back, nor one of a reading still being typed).
    fn shows_whole_conversion(&self) -> bool {
        self.cfg.live_conversion
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.cursor_at_end()
            && self.composing.pending_romaji().is_empty()
            && !self.comp.kana_shown
            && !self.comp.cached_prefix_kanji.is_empty()
            && self.comp.cached_prefix_reading == self.composing.live_reading()
    }

    /// Convert the reading typed (the whole sentence's list), or the next
    /// candidate of a conversion open: by Space (`space`: after English it
    /// types a space), by ← on the conversion live conversion shows or by a
    /// suggestion clicked (no space: the English ones among the segments or
    /// the sentence). `segments`: the conversion opens on its segments
    /// (`on_last_segment`), where a rewrite of the sentence would not show:
    /// none is asked for.
    fn convert_typed(&mut self, space: bool, segments: bool) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        // a command's name ends
        self.ctx.passing = false;
        if self.form_shown().is_some() {
            // Space after F6–F10 converts the reading as usual.
            self.back_to_composing();
        }
        // letters or symbols the user dictionary has as a reading (`w`)
        if let Some(out) = self.open_typed_words() {
            return out;
        }
        // Mixed input: Space after an English word, or a lone `i` (`i
        // think`), types a space (`Hello world`); Space again, or after
        // Japanese, converts.
        if space
            && self.composing.is_mixed()
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.cursor_at_end()
        {
            if self.composing.ends_with_space() {
                self.composing.trim_end_spaces();
            } else if self.composing.ends_in_english() || self.composing.ends_in_lone_i() {
                return self.type_char(' ');
            }
        }
        let mut previous_dates = Vec::new();
        if matches!(self.phase, Phase::Composing { .. }) {
            self.comp.alt = self.alt_for_space();
            if !space {
                // the top with a space after the English is Space's offer
                if let Some(alt) = self.comp.alt.as_mut() {
                    alt.ends_english = false;
                }
            }
            if self.cfg.date_time_enabled {
                previous_dates = self
                    .cfg
                    .char_rules
                    .apply(self.date_time_candidates(&self.composing.full_reading()));
                // Each explicit conversion gets a fresh clock; its later
                // model/rerank/segment results reuse this exact instant.
                self.comp.date_time_now = Some(Local::now());
            }
        }
        match &self.phase {
            Phase::Empty if self.cfg.full_width_space => self.commit_space(),
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.push(' ');
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => self.handle_down(),
            Phase::Composing { .. } => {
                // Async model: the live path already dispatches on each
                // keystroke for longer readings. Space is an explicit
                // conversion request, so it must dispatch even for 1-2
                // character readings; otherwise dictionary-backed short
                // conversions never reach the daemon and the UI is stuck
                // with a single provisional candidate.
                let candidates = if self.cfg.force_convert_on_space {
                    self.dispatch_live_forced()
                } else {
                    // typing is done for this conversion, as in the forced
                    // dispatch (a pending n is ん)
                    self.composing.commit_reading();
                    self.dispatch_live()
                };
                // the engine has the list of this reading already
                let reading = self.composing.hiragana().to_string();
                let ready = reading == self.comp.list_reading;
                let dynamic = self.cfg.date_time_enabled && crate::date_time::is_reading(&reading)
                    // digits or a calculation: their forms need no model,
                    // so the list opens whole before the engine answers
                    || self.cfg.numbers_enabled && crate::numbers::stands_alone(&reading);
                // A prefix list is not a conversion of the frozen whole reading.
                let candidates = if ready && dynamic {
                    let normal = candidates
                        .into_iter()
                        .filter(|c| !previous_dates.contains(c))
                        .collect();
                    crate::date_time::insert(
                        &reading,
                        normal,
                        self.cfg
                            .char_rules
                            .apply(self.date_time_candidates(&reading)),
                    )
                } else if ready {
                    // converting: the model's first, also where a candidate
                    // ending in kana was shown before it
                    self.model_first(&reading, candidates)
                } else {
                    Vec::new()
                };
                if candidates.is_empty() {
                    let provisional = self.provisional_conversion();
                    if provisional.is_empty() {
                        return self.handle_enter();
                    }
                    // This also works without a model or while it loads.
                    let mut list = if dynamic {
                        self.finish_list(&reading, None, vec![reading.clone()], true)
                    } else {
                        Vec::new()
                    };
                    // (the user dictionary may suppress them all)
                    if list.is_empty() {
                        list.push(provisional);
                    }
                    self.comp.list_starts = self.list_starts_of(&reading, &[], &[], &list);
                    self.comp.list_reading = if dynamic {
                        reading.clone()
                    } else {
                        String::new()
                    };
                    let source = list[0].clone();
                    let list = self.shown_list(list);
                    let first = self.with_pending(list.first().cloned().unwrap_or_default());
                    self.phase = Phase::Converting {
                        candidates: list.clone(),
                        selected: 0,
                        seg: None,
                    };
                    let mut actions = vec![
                        Action::UpdatePreedit { text: first },
                        Action::ShowCandidates { list, selected: 0 },
                    ];
                    if !segments {
                        actions.extend(self.start_rewrite(&reading, &source));
                    }
                    return consumed(actions);
                }
                if dynamic && self.comp.cached_prefix_reading == reading {
                    self.cache_conversion(&reading, &candidates);
                }
                // Preserve the model's alignments while adding starts for
                // fresh local candidates (which each cover the whole reading).
                for (text, starts) in self.list_starts_of(&reading, &[], &[], &candidates) {
                    self.comp.list_starts.entry(text).or_insert(starts);
                }
                let source = candidates.first().cloned().unwrap_or_default();
                if ready {
                    // the list opens with the other reading's
                    self.request_alt();
                    self.await_alt();
                }
                let candidates = self.shown_list(candidates);
                let first = self.with_pending(candidates.first().cloned().unwrap_or_default());
                self.phase = Phase::Converting {
                    candidates: candidates.clone(),
                    selected: 0,
                    seg: None,
                };
                // the conversion key: a rerank waiting for the typing pause goes now
                self.rerank_now();
                let mut actions = vec![
                    Action::UpdatePreedit { text: first },
                    Action::ShowCandidates {
                        list: candidates,
                        selected: 0,
                    },
                ];
                if !segments {
                    actions.extend(self.start_rewrite(&reading, &source));
                }
                consumed(actions)
            }
        }
    }

    /// Space on what was typed when the user dictionary has it as a reading
    /// of letters or symbols (`w` for (笑), as MS-IME takes one): the list
    /// of its words and the input as typed, which no conversion makes (`w`
    /// is no kana; mixed input would type a space after it). `None` when
    /// the dictionary has no such reading.
    fn open_typed_words(&mut self) -> Option<KeyOutcome> {
        if !matches!(self.phase, Phase::Composing { .. }) || !self.composing.cursor_at_end() {
            return None;
        }
        let typed = self.composing.romaji();
        let key = typed.to_ascii_lowercase();
        if !crate::user_dict::is_typed_reading(&key) {
            return None;
        }
        let dict = self.user_dict()?;
        if dict.words(&key).is_empty() {
            return None;
        }
        let mut list = dict.apply(&key, vec![typed]);
        if let Some(learning) = self.learning.as_mut() {
            list = learning.rerank(&key, list);
        }
        if let Some(hidden) = self.hidden.as_mut() {
            list = hidden.filter(&key, list);
        }
        let first = list.first().cloned()?;
        self.invalidate_engine_results();
        self.stop_rerank();
        self.comp.alt = None;
        self.comp.kana_shown = false;
        self.comp.covers_typed = true;
        self.comp.list_reading = key;
        self.comp.list_starts.clear();
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected: 0,
            seg: None,
        };
        Some(consumed(vec![
            Action::UpdatePreedit { text: first },
            Action::ShowCandidates { list, selected: 0 },
        ]))
    }

    pub fn handle_enter(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.commit_composition(true)
    }

    /// Commit what is currently shown without waiting for a pending live
    /// conversion (focus out, input method switch): never blocks the host's
    /// event loop, and keeps the commit aligned with what was displayed.
    pub fn flush_shown(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.commit_composition(false)
    }

    fn commit_composition(&mut self, wait: bool) -> KeyOutcome {
        self.ctx.passing = false;
        // mixed input reading the whole otherwise once typing is done
        // (げんだいぶんめいok: お and a pending k while typed, ok when done):
        // that reading is converted, as Space does (its kana went
        // unconverted); one with no kana is committed as it reads (`ok`: OK)
        let done_otherwise = self.composing.reads_otherwise_when_done();
        let convert_done = wait
            && done_otherwise
            && self.cfg.live_conversion
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.cursor_at_end()
            && !self.comp.kana_shown
            && self
                .composing
                .full_reading()
                .chars()
                .any(|c| matches!(c, 'ぁ'..='ゖ' | 'ァ'..='ヺ'));
        if convert_done {
            self.composing.commit_reading();
            self.dispatch_live();
        }
        if wait {
            self.await_live_conversion(convert_done);
        }
        self.invalidate_engine_results();
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let text = buffer.clone();
                self.phase = Phase::Empty;
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.phase = Phase::Empty;
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            Phase::Composing { candidates } => {
                // with the caret moved the preedit is the kana, and so is
                // the commit; so it is when mixed input reads the whole
                // otherwise than what was typed so far and has no kana to
                // convert (`ok`: OK, not お and k)
                let done_otherwise = done_otherwise && !convert_done;
                // after Escape the kana shown is what goes
                let kana = std::mem::take(&mut self.comp.kana_shown);
                let reading = self.composing.full_reading();
                let deferred = self.comp.deferred.is_some();
                let cand_top = candidates
                    .first()
                    .filter(|_| {
                        self.cfg.live_conversion
                            && self.composing.cursor_at_end()
                            && self.comp.list_reading == self.composing.live_reading()
                            && !done_otherwise
                            && !kana
                            && (wait || !deferred)
                    })
                    .cloned();
                let text = match cand_top {
                    _ if done_otherwise || kana => self.composing.commit(),
                    Some(t) if !t.is_empty() => format!("{t}{}", self.composing.commit_live_tail()),
                    _ if self.cfg.live_conversion && self.composing.cursor_at_end() => {
                        let text = self.provisional_conversion();
                        format!("{text}{}", self.composing.commit_live_tail())
                    }
                    _ => self.composing.commit(),
                };
                self.phase = Phase::Empty;
                self.clear_composition();
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.record_commit(&reading, &text);
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
        }
    }

    /// Escape steps back, as MS-IME's does (`escape_to_kana`): the
    /// candidate window, or a conversion shown live, gives way to the kana
    /// typed, and Escape on the kana clears the composition. Without it the
    /// window goes back to the live conversion, and Escape there clears.
    pub fn handle_escape(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.ctx.passing = false;
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { .. } => {
                self.phase = Phase::Empty;
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::EndComposition],
                }
            }
            Phase::Converting { .. } => {
                self.back_to_composing();
                if self.cfg.escape_to_kana {
                    self.show_kana();
                }
                let preedit = self.live_preedit();
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. }
                if self.cfg.escape_to_kana
                    && !self.comp.kana_shown
                    && self.live_preedit() != self.composing.preedit() =>
            {
                self.show_kana();
                consumed(vec![Action::UpdatePreedit {
                    text: self.live_preedit(),
                }])
            }
            Phase::Composing { .. } => consumed(self.cancel()),
        }
    }

    /// Show the kana typed in place of the conversion while typing (the
    /// first Escape): the conversion shown, a rerank and a result on their
    /// way are dropped; the candidates stay for Space.
    fn show_kana(&mut self) {
        self.invalidate_engine_results();
        self.stop_rerank();
        self.comp.deferred = None;
        self.comp.clear_preview_cache();
        self.comp.kept = None;
        self.comp.kana_shown = true;
    }

    pub fn handle_move_cursor(&mut self, delta: i32) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        // ← on the conversion live conversion shows: its segments to edit,
        // the last one first, as Space on it (`on_last_segment`); the
        // conversion stays, where moving the caret into it brought the kana
        // typed back
        if delta < 0 && self.shows_whole_conversion() {
            // the segments of the conversion shown, a noun end held too
            self.comp.end_held = None;
            let out = self.convert_typed(false, true);
            return self.on_last_segment(out);
        }
        match &self.phase {
            Phase::Composing { .. } => {
                self.composing.move_cursor(delta);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { seg, .. } => {
                // The first ← / → switches to segment editing on the first
                // segment; later ones move the focus.
                let entering = seg.is_none();
                if !self.ensure_segmented() {
                    return consumed(Vec::new());
                }
                if !entering {
                    if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
                        s.move_focus(delta);
                    }
                }
                self.query_focused_segment();
                consumed(self.segment_actions())
            }
            _ => not_consumed(),
        }
    }

    pub fn handle_up(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.move_selection(-1)
    }

    /// ↓ steps down the list; while typing it opens the conversion, as
    /// Tab does.
    pub fn handle_down(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        if matches!(self.phase, Phase::Composing { .. }) {
            return self.handle_space();
        }
        self.move_selection(1)
    }

    pub fn handle_page_up(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.move_page(-1)
    }

    pub fn handle_page_down(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        self.move_page(1)
    }

    /// Shift+Space: Space while composing; with nothing composed, the space
    /// width plain Space does not type.
    pub fn handle_shift_space(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match self.phase {
            Phase::Empty if !self.cfg.full_width_space => self.commit_space(),
            Phase::Empty => not_consumed(),
            _ => self.handle_space(),
        }
    }

    fn commit_space(&mut self) -> KeyOutcome {
        let text = "\u{3000}".to_string();
        self.append_context(&text);
        consumed(vec![Action::Commit { text }])
    }

    /// Tab opens the candidates like Space, then steps down (Shift+Tab up).
    pub fn handle_tab(&mut self, back: bool) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { .. } => consumed(Vec::new()),
            Phase::Converting { .. } if self.form_shown().is_none() => {
                self.move_selection(if back { -1 } else { 1 })
            }
            // typing with predictions: Tab opens them (as Mozc's)
            Phase::Composing { .. } => match self.open_predictions(0) {
                Some(out) => out,
                None => self.handle_space(),
            },
            _ => self.handle_space(),
        }
    }

    /// Delete: remove the kana right of the caret.
    pub fn handle_delete(&mut self) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Composing { .. } => {
                self.composing.delete_right();
                if self.composing.is_empty() {
                    return self.handle_escape();
                }
                let candidates = self.dispatch_live();
                self.phase = Phase::Composing { candidates };
                consumed(vec![Action::UpdatePreedit {
                    text: self.live_preedit(),
                }])
            }
            _ => consumed(Vec::new()),
        }
    }

    /// Home / End: the caret to the start / end of the composition, or the
    /// first / last candidate of the open list.
    pub fn handle_home_end(&mut self, end: bool) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        match self.phase {
            Phase::Empty => not_consumed(),
            Phase::Composing { .. } => {
                self.composing.move_cursor_to_edge(end);
                let candidates = self.dispatch_live();
                self.phase = Phase::Composing { candidates };
                consumed(vec![Action::UpdatePreedit {
                    text: self.live_preedit(),
                }])
            }
            Phase::Converting { .. } if self.window_open() => {
                let Some((selected, n)) = self.selection() else {
                    return consumed(Vec::new());
                };
                let target = if end { n - 1 } else { 0 };
                self.move_selection(target as i32 - selected as i32)
            }
            _ => consumed(Vec::new()),
        }
    }

    /// F6–F10: show the whole composition in `form`. Pressing F9 / F10
    /// again cycles the case (`tokyo` → `TOKYO` → `Tokyo`). Enter commits
    /// it, Space converts the reading as usual, Escape / Backspace go back
    /// to typing.
    pub fn handle_convert_to(&mut self, form: Form) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        let mut actions = Vec::new();
        match &self.phase {
            Phase::Empty => return not_consumed(),
            Phase::Converting { seg: Some(_), .. } => return self.convert_segment_to(form),
            Phase::Alphabet { buffer } => {
                // the typed letters become the composition
                let buffer = buffer.clone();
                self.composing.reset();
                for c in buffer.chars() {
                    self.composing.input_char(c);
                }
                self.phase = Phase::Composing {
                    candidates: Vec::new(),
                };
            }
            _ => {}
        }
        // every letter as romaji: a word taken for English converts too
        let reading = self.composing.kana_reading();
        if reading.is_empty() {
            return consumed(actions);
        }
        let (case, list) = match self.form_shown() {
            Some(f) => (
                if f.form == form && form.cycles_case() {
                    f.case + 1
                } else {
                    0
                },
                f.list.clone(),
            ),
            None => (0, self.current_list()),
        };
        let text = char_forms::convert(form, &reading, &self.composing.romaji(), case);
        self.comp.seg_requests.clear();
        self.comp.form = Some(FormShown {
            form,
            case,
            text: text.clone(),
            list,
        });
        self.phase = Phase::Converting {
            candidates: vec![text.clone()],
            selected: 0,
            seg: None,
        };
        actions.push(Action::HideCandidates);
        actions.push(Action::UpdatePreedit { text });
        consumed(actions)
    }

    /// The candidates of the whole reading the screen holds: the typing
    /// phase's, or the open list's when it is not provisional.
    pub(super) fn current_list(&self) -> Vec<String> {
        match &self.phase {
            Phase::Composing { candidates } => candidates.clone(),
            Phase::Converting { candidates, .. } if !self.comp.list_reading.is_empty() => {
                candidates.clone()
            }
            _ => Vec::new(),
        }
    }

    /// Leave Converting for typing, keeping the reading's candidates (Enter
    /// then commits the conversion the preedit shows).
    fn back_to_composing(&mut self) {
        // the predictions are no conversion of what is typed, nor are the
        // words of letters typed
        let predicted =
            self.comp.prediction.take().is_some() | std::mem::take(&mut self.comp.covers_typed);
        self.comp.last_segment_due = false;
        if predicted {
            self.comp.list_reading.clear();
        }
        self.comp.seg_form = None;
        let mut list = match self.form_shown() {
            _ if predicted => Vec::new(),
            Some(f) => f.list.clone(),
            None => self.current_list(),
        };
        // typing shows (and Enter commits) the engine's conversion; the
        // rewrite stays for the next conversion of the same reading
        if let Some(text) = self.rewrite_shown() {
            list.retain(|c| *c != text);
        }
        self.comp.form = None;
        self.comp.seg_requests.clear();
        self.phase = Phase::Composing {
            candidates: Vec::new(),
        };
        let dispatched = self.dispatch_live();
        self.phase = Phase::Composing {
            candidates: if list.is_empty() { dispatched } else { list },
        };
    }

    /// Whether the candidate window is open (Converting, not an F6–F10
    /// form).
    pub(super) fn window_open(&self) -> bool {
        self.is_converting() && self.form_shown().is_none()
    }

    /// (selected index, list length) of the open list: the focused
    /// segment's while editing segments.
    fn selection(&self) -> Option<(usize, usize)> {
        let (selected, n) = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => {
                let f = &s.segs[s.focus];
                (f.selected, f.cands.len())
            }
            Phase::Converting {
                candidates,
                selected,
                ..
            } => (*selected, candidates.len()),
            _ => return None,
        };
        (n > 0).then_some((selected, n))
    }

    /// PageUp / PageDown: the same slot on the previous / next page,
    /// wrapping around (past a segment's last page: its continuation).
    fn move_page(&mut self, dir: i32) -> KeyOutcome {
        let Some((selected, n)) = self.selection() else {
            return not_consumed();
        };
        let ps = self.cfg.page_size;
        let pages = n.div_ceil(ps) as i32;
        let page = (selected / ps) as i32;
        if dir > 0 && page + 1 >= pages && self.continue_focused() {
            return consumed(self.segment_actions());
        }
        let target_page = (page + dir).rem_euclid(pages) as usize;
        let target = (target_page * ps + selected % ps).min(n - 1);
        self.move_selection(target as i32 - selected as i32)
    }

    /// Digit `c` with the window open: pick that candidate of the page and
    /// commit it, or, editing segments, fix the focused segment's and move
    /// to the next one (committing after the last).
    fn pick_by_digit(&mut self, c: char) -> KeyOutcome {
        let slot = match c.to_digit(10) {
            Some(0) => 10,
            Some(d) => d as usize,
            None => return not_consumed(),
        };
        let Some((selected, n)) = self.selection() else {
            return consumed(Vec::new());
        };
        let index = selected / self.cfg.page_size * self.cfg.page_size + slot - 1;
        if slot > self.cfg.page_size || index >= n {
            return consumed(Vec::new());
        }
        self.handle_pick(index)
    }

    /// Pick candidate `index` of the open list (or suggestions) and accept it:
    /// commits it, or, while editing segments, fixes the focused segment and
    /// moves to the next one (committing after the last), matching digit keys.
    pub fn handle_pick(&mut self, index: usize) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        let picked = self.handle_select(index);
        if !picked.consumed {
            return picked;
        }
        let mut actions = picked.actions;
        let last_segment = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => s.focus + 1 == s.segs.len(),
            _ => true,
        };
        let next = if last_segment {
            self.handle_enter()
        } else {
            self.handle_move_cursor(1)
        };
        actions.extend(next.actions);
        consumed(actions)
    }

    /// Pick candidate `index` of the open list (a click in the candidate
    /// window): the same as moving the selection there with ↑ / ↓.
    /// While typing it picks from [`Self::suggestions`]: the conversion
    /// opens on that candidate.
    pub fn handle_select(&mut self, index: usize) -> KeyOutcome {
        // the list the preview went without, whole for this key
        self.finish_list_now(true);
        if matches!(self.phase, Phase::Composing { .. }) {
            if let Some(want) = self.suggestions().get(index).cloned() {
                // the suggestion picked, where the whole sentence's list has
                // it (mixed input's puts the other reading's in: it is not
                // the suggestions' order); a sentence, not the last segment
                // Space may open, nor a space after English
                let mut actions = self.convert_typed(false, false).actions;
                let at = match &self.phase {
                    Phase::Converting {
                        candidates,
                        seg: None,
                        ..
                    } => candidates.iter().position(|c| *c == want),
                    _ => None,
                };
                actions.extend(self.handle_select(at.unwrap_or(index)).actions);
                return consumed(actions);
            }
        }
        let current = match &self.phase {
            Phase::Converting { seg: Some(s), .. } => s.segs[s.focus].selected,
            Phase::Converting { selected, .. } => *selected,
            _ => return not_consumed(),
        };
        self.move_selection(index as i32 - current as i32)
    }

    fn move_selection(&mut self, delta: i32) -> KeyOutcome {
        if let Some(s) = self.segmented() {
            let focused = &s.segs[s.focus];
            // past the last candidate: more of them rather than round to the first
            if delta > 0
                && focused.selected + delta as usize >= focused.cands.len()
                && self.continue_focused()
            {
                return consumed(self.segment_actions());
            }
        }
        if let Phase::Converting { seg: Some(s), .. } = &mut self.phase {
            let focused = &mut s.segs[s.focus];
            let n = focused.cands.len().max(1) as i32;
            focused.selected = (focused.selected as i32 + delta).rem_euclid(n) as usize;
            self.query_focused_segment();
            return consumed(self.segment_actions());
        }
        let Phase::Converting {
            candidates,
            selected,
            ..
        } = &mut self.phase
        else {
            return not_consumed();
        };
        let n = candidates.len();
        if n == 0 {
            return not_consumed();
        }
        let cur = *selected as i32;
        let new_sel = ((cur + delta).rem_euclid(n as i32)) as usize;
        *selected = new_sel;
        let text = candidates[new_sel].clone();
        let list_snap = candidates.clone();
        let text = self.with_pending(text);
        KeyOutcome {
            consumed: true,
            actions: vec![
                Action::UpdatePreedit { text },
                Action::ShowCandidates {
                    list: list_snap,
                    selected: new_sel,
                },
            ],
        }
    }

    pub fn handle_punctuation(&mut self, kana: &str) -> KeyOutcome {
        // `/command`, `$skill`, `@file` at a line's start or after a space:
        // to the app as typed, its name too
        let ascii = |c: char| match c {
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            '・' => '/',
            c => c,
        };
        let typed: Vec<char> = kana.chars().map(ascii).collect();
        if let [c] = typed[..] {
            if self.passes(c) {
                return not_consumed();
            }
            if self.cfg.pass_commands
                && matches!(self.phase, Phase::Empty)
                && self.ctx.at_word_start == Some(true)
                && matches!(c, '/' | '$' | '@')
            {
                self.ctx.passing = true;
                return not_consumed();
            }
        }
        // a key the romaji table reads (AZIK's `;`, kana input's `,`) goes
        // into the composition
        if let Some(table) = crate::romaji_table::active() {
            let ascii: Vec<char> = kana
                .chars()
                .map(|c| match c {
                    '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                    '、' => ',',
                    '。' => '.',
                    '・' => '/',
                    '「' => '[',
                    '」' => ']',
                    '￥' => '¥',
                    c => c,
                })
                .collect();
            if let [c] = ascii[..] {
                if table.uses(c) && !matches!(self.phase, Phase::Alphabet { .. }) {
                    return self.handle_char(c);
                }
            }
        }
        // Sentence punctuation right after English, in a sentence with no
        // Japanese (`mixed::english_after`): as typed, not 、。！. After the
        // English typed with Shift it goes on with it; after an English text
        // committed it is committed. Mixed input decides it per span.
        let sentence_punct = match typed[..] {
            ['。'] => Some('.'),
            ['、'] => Some(','),
            [c @ ('.' | ',' | '!' | '?' | ';' | ':')] => Some(c),
            _ => None,
        };
        if let Some(c) = sentence_punct {
            match &self.phase {
                _ if !self.cfg.sentence_case => {}
                Phase::Alphabet { buffer }
                    if crate::mixed::english_after(&format!("{}{buffer}", self.ctx.context)) =>
                {
                    let buf = format!("{buffer}{c}");
                    self.phase = Phase::Alphabet {
                        buffer: buf.clone(),
                    };
                    return consumed(vec![Action::UpdatePreedit { text: buf }]);
                }
                Phase::Empty if crate::mixed::english_after(&self.ctx.context) => {
                    let text = c.to_string();
                    self.append_context(&text);
                    return consumed(vec![Action::Commit { text }]);
                }
                _ => {}
            }
        }
        // The period key right after a digit types `.` (3.5, and 3. before
        // the 5): a 。 there is next to never meant. Mixed input shows it so
        // itself (`mixed::render_parts`).
        let kana = if kana == "。"
            && !self.composing.is_mixed()
            && matches!(self.phase, Phase::Composing { .. })
            && self.composing.digit_before_cursor()
        {
            "."
        } else {
            kana
        };
        if matches!(
            self.phase,
            Phase::Empty | Phase::Alphabet { .. } | Phase::Converting { .. }
        ) {
            self.invalidate_engine_results();
        }
        // Mixed input: the symbol goes in as typed and shows per span
        // (`example.com`, 。 after Japanese).
        if self.composing.is_mixed() && !matches!(self.phase, Phase::Alphabet { .. }) {
            // A symbol with no name in the config comes full width.
            let typed = self.cfg.mixed_symbols.get(kana).copied().or_else(|| {
                let mut chars = kana.chars();
                match (chars.next(), chars.next()) {
                    (Some(c @ '\u{FF01}'..='\u{FF5E}'), None) => {
                        char::from_u32(c as u32 - 0xFF01 + 0x21)
                    }
                    _ => None,
                }
            });
            if let Some(c) = typed {
                return self.type_char(c);
            }
        }
        // In Alphabet mode punctuation commits the buffer (same as Enter)
        // and then starts a fresh punctuation composition. Matches MS-IME:
        // typing `!` while in temp-alphabet mode ends it.
        if let Phase::Alphabet { buffer } = &self.phase {
            let text = buffer.clone();
            self.phase = Phase::Empty;
            let mut actions = Vec::new();
            if !text.is_empty() {
                self.append_context(&text);
                actions.push(Action::Commit { text });
            }
            self.composing.reset();
            self.composing.flush_romaji_and_insert(kana);
            let candidates = self.dispatch_live();
            let preedit = self.live_preedit();
            self.phase = Phase::Composing { candidates };
            actions.push(Action::UpdatePreedit { text: preedit });
            return KeyOutcome {
                consumed: true,
                actions,
            };
        }
        match &self.phase {
            Phase::Alphabet { .. } => unreachable!(), // handled above
            Phase::Empty | Phase::Composing { .. } => {
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                self.composing.reset();
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                self.phase = Phase::Composing { candidates };
                actions.push(Action::HideCandidates);
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
        }
    }

    /// Record a commit for prediction (and as the commit before the next).
    fn record_commit(&mut self, reading: &str, text: &str) {
        if self.ctx.private_field {
            return;
        }
        if let Some(p) = self.predictor.as_ref() {
            if let Err(e) = p.borrow_mut().record(reading, text, &self.ctx.last_commit) {
                tracing::warn!("prediction record failed: {e}");
            }
        }
        self.ctx.last_commit = text.to_string();
    }

    /// Open the predictions as the candidate list, `selected` chosen: Tab
    /// while typing.
    fn open_predictions(&mut self, selected: usize) -> Option<KeyOutcome> {
        let preds = self.predictions();
        if preds.is_empty() {
            return None;
        }
        let list: Vec<String> = preds.iter().map(|p| p.surface.clone()).collect();
        let selected = selected.min(list.len() - 1);
        self.stop_rerank();
        self.stop_rewrite();
        self.comp.prediction = Some(preds);
        self.comp.list_reading.clear();
        self.phase = Phase::Converting {
            candidates: list.clone(),
            selected,
            seg: None,
        };
        Some(consumed(vec![
            Action::UpdatePreedit {
                text: list[selected].clone(),
            },
            Action::ShowCandidates { list, selected },
        ]))
    }

    fn commit_converting(&mut self) -> Vec<Action> {
        let Phase::Converting {
            candidates,
            selected,
            seg,
        } = &self.phase
        else {
            return vec![];
        };
        // What is committed, and the choice to learn: the sentence the user
        // ended up with over the one shown first.
        let (text, learn) = match seg {
            Some(s) => {
                let text = s.text();
                let learn = (text != s.initial).then(|| (s.full_reading(), s.initial.clone()));
                (text, learn)
            }
            None => {
                let text = candidates.get(*selected).cloned().unwrap_or_default();
                // over a rewrite first, the engine's top is what was passed over
                let top = match self.rewrite_shown() {
                    Some(r) if candidates.first() == Some(&r) => 1,
                    _ => 0,
                };
                let learn = (*selected > top && !self.comp.list_reading.is_empty())
                    .then(|| (self.comp.list_reading.clone(), candidates[top].clone()));
                (text, learn)
            }
        };
        // a prediction: no choice to learn, but its use to record
        let prediction = self.comp.prediction.take().map(|p| {
            let reading = p
                .iter()
                .find(|x| x.surface == text)
                .map(|x| x.reading.clone())
                .unwrap_or_default();
            (reading, text.clone())
        });
        // nor a date or a time the reading generated, another one each day
        let learn = learn.filter(|(reading, _)| {
            !self.ctx.private_field
                && prediction.is_none()
                && !self.date_time_candidates(reading).contains(&text)
        });
        if let (Some((reading, rejected)), Some(learning)) = (learn, self.learning.as_mut()) {
            if let Err(e) = learning.record(&reading, &text, &rejected) {
                tracing::warn!("learning record failed: {e}");
            }
        }
        let reading = match &prediction {
            Some((r, _)) => r.clone(),
            None if !self.comp.list_reading.is_empty() && seg.is_none() => {
                self.comp.list_reading.clone()
            }
            None => self.composing.full_reading(),
        };
        let covers_typed = std::mem::take(&mut self.comp.covers_typed);
        let tail = if self.form_shown().is_none() && prediction.is_none() && !covers_typed {
            self.composing.commit_tail()
        } else {
            // a prediction covers what was still being typed
            self.composing.reset();
            String::new()
        };
        self.record_commit(&reading, &text);
        let text = format!("{text}{tail}");
        self.invalidate_engine_results();
        self.clear_composition();
        if !text.is_empty() {
            self.append_context(&text);
            vec![Action::Commit { text }]
        } else {
            vec![Action::EndComposition]
        }
    }
}
