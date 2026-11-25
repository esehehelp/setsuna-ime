use super::live_cache::starts_fit;
use super::ranking::rerank_order;
use super::state::*;
use super::*;
use crate::engine_client::EngineResult;
use chrono::Local;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct MockState {
    dispatched: RefCell<Vec<(u64, String, String)>>,
    /// (request id, continue_beam) of the `dispatch_more` requests.
    continued: RefCell<Vec<(u64, u32)>>,
    next: RefCell<Option<EngineResult>>,
    /// The backend has exited.
    gone: std::cell::Cell<bool>,
    /// What `analyze` answers for a text (none: no analyzer).
    analyses: RefCell<HashMap<String, Vec<crate::engine_client::Word>>>,
    /// Readings answered as soon as they are dispatched, within the
    /// wait of Enter (`wait_latest`).
    answers: RefCell<HashMap<String, Vec<String>>>,
}

/// In-memory `EngineClient` for phase-machine tests. Cloning shares the
/// same state, so a test can keep a handle, hand a clone to the bridge,
/// and still inspect dispatches / inject results.
#[derive(Clone, Default)]
struct MockClient {
    state: Rc<MockState>,
}

impl MockClient {
    fn set_result(&self, reading: &str, candidates: &[&str]) {
        *self.state.next.borrow_mut() = Some(EngineResult {
            req_id: self
                .state
                .dispatched
                .borrow()
                .iter()
                .rev()
                .find(|(_, _, r)| r == reading)
                .map_or(1, |r| r.0),
            reading: reading.to_string(),
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
            reading_pos: Vec::new(),
            ..Default::default()
        });
    }
    fn set_aligned(&self, reading: &str, candidates: &[&str], pos: &[&[u32]]) {
        *self.state.next.borrow_mut() = Some(EngineResult {
            req_id: self
                .state
                .dispatched
                .borrow()
                .iter()
                .rev()
                .find(|(_, _, r)| r == reading)
                .map_or(1, |r| r.0),
            reading: reading.to_string(),
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
            reading_pos: pos.iter().map(|p| p.to_vec()).collect(),
            ..Default::default()
        });
    }

    /// Answer the last dispatched request (its id and reading).
    fn answer_last(&self, candidates: &[&str]) -> (String, String) {
        let (req_id, context, reading) = self.state.dispatched.borrow().last().cloned().unwrap();
        *self.state.next.borrow_mut() = Some(EngineResult {
            req_id,
            reading: reading.clone(),
            candidates: candidates.iter().map(|s| s.to_string()).collect(),
            reading_pos: Vec::new(),
            ..Default::default()
        });
        (context, reading)
    }

    fn dispatched_readings(&self) -> Vec<String> {
        self.state
            .dispatched
            .borrow()
            .iter()
            .map(|(_, _, r)| r.clone())
            .collect()
    }
}

impl EngineClient for MockClient {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
        self.state
            .dispatched
            .borrow_mut()
            .push((req_id, context.to_string(), reading.to_string()));
        if let Some(candidates) = self.state.answers.borrow().get(reading) {
            *self.state.next.borrow_mut() = Some(EngineResult {
                req_id,
                reading: reading.to_string(),
                candidates: candidates.clone(),
                ..Default::default()
            });
        }
    }
    fn wait_latest(&self, _timeout: std::time::Duration) -> bool {
        !self.state.answers.borrow().is_empty() && self.state.next.borrow().is_some()
    }
    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
        self.dispatch(req_id, context, reading);
        self.state
            .continued
            .borrow_mut()
            .push((req_id, continue_beam));
    }
    fn take_latest(&self) -> Option<EngineResult> {
        self.state.next.borrow_mut().take()
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn is_alive(&self) -> bool {
        !self.state.gone.get()
    }
    fn analyze(
        &self,
        texts: &[String],
        _timeout: std::time::Duration,
    ) -> Option<Vec<Vec<crate::engine_client::Word>>> {
        let a = self.state.analyses.borrow();
        texts.iter().map(|t| a.get(t).cloned()).collect()
    }
}

#[derive(Default)]
struct RerankState {
    /// (request id, context, candidates, due)
    dispatched: RefCell<Vec<(u64, String, Vec<String>, Instant)>>,
    cancels: RefCell<usize>,
    next: RefCell<Option<crate::rerank_client::RerankResult>>,
}

#[derive(Clone, Default)]
struct MockReranker {
    state: Rc<RerankState>,
}

impl MockReranker {
    /// Answer the last request with these log-probabilities.
    fn answer(&self, logp: &[f64]) {
        let req_id = self.state.dispatched.borrow().last().unwrap().0;
        *self.state.next.borrow_mut() = Some(crate::rerank_client::RerankResult {
            req_id,
            logp: logp.to_vec(),
            score_ms: 1.0,
        });
    }
    fn last(&self) -> (Vec<String>, Instant) {
        let d = self.state.dispatched.borrow();
        let (_, _, c, due) = d.last().unwrap();
        (c.clone(), *due)
    }
}

impl RerankClient for MockReranker {
    fn dispatch(&self, req_id: u64, context: &str, candidates: &[String], due: Instant) {
        self.state.dispatched.borrow_mut().push((
            req_id,
            context.to_string(),
            candidates.to_vec(),
            due,
        ));
    }
    fn cancel(&self) {
        *self.state.cancels.borrow_mut() += 1;
    }
    fn take_latest(&self) -> Option<crate::rerank_client::RerankResult> {
        self.state.next.borrow_mut().take()
    }
    fn info(&self) -> Option<crate::rerank_client::RerankInfo> {
        None
    }
}

/// A bridge with a reranker that scores the first 2 candidates first.
fn bridge_with_reranker() -> (EngineBridge, MockClient, MockReranker) {
    let (mut bridge, mock) = bridge_with_mock();
    let reranker = MockReranker::default();
    let cfg = crate::config::RerankCfg {
        first_candidates: 2,
        lambda: Some(0.5),
        ..Default::default()
    };
    bridge.set_reranker(Some(Box::new(reranker.clone())), &cfg);
    (bridge, mock, reranker)
}

#[derive(Default)]
struct RewriteState {
    /// (request id, context, reading, text)
    dispatched: RefCell<Vec<(u64, String, String, String)>>,
    next: RefCell<Option<crate::rewrite_client::RewriteResult>>,
}

#[derive(Clone, Default)]
struct MockRewriter {
    state: Rc<RewriteState>,
}

impl MockRewriter {
    /// Answer the last request with `text`, or fail it with `code`.
    fn answer(&self, text: Result<&str, &str>) {
        let req_id = self.state.dispatched.borrow().last().unwrap().0;
        *self.state.next.borrow_mut() = Some(crate::rewrite_client::RewriteResult {
            req_id,
            text: text.ok().map(str::to_string),
            code: text.err().map(str::to_string),
            ms: 1.0,
        });
    }
    fn requests(&self) -> usize {
        self.state.dispatched.borrow().len()
    }
}

impl RewriteClient for MockRewriter {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str, text: &str) {
        self.state.dispatched.borrow_mut().push((
            req_id,
            context.to_string(),
            reading.to_string(),
            text.to_string(),
        ));
    }
    fn cancel(&self) {}
    fn take_latest(&self) -> Option<crate::rewrite_client::RewriteResult> {
        self.state.next.borrow_mut().take()
    }
}

/// `kasano` typed and converted by the engine, with a rewriter in
/// rewrite mode.
fn rewriting_bridge() -> (EngineBridge, MockClient, MockRewriter) {
    let (mut bridge, mock) = bridge_with_mock();
    let rewriter = MockRewriter::default();
    bridge.set_rewriter(Some(Box::new(rewriter.clone())));
    assert!(bridge.set_rewrite_mode(true).is_empty());
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    (bridge, mock, rewriter)
}

fn status(actions: &[Action]) -> Option<&str> {
    actions.iter().find_map(|a| match a {
        Action::Status { text } => Some(text.as_str()),
        _ => None,
    })
}

fn shown(actions: &[Action]) -> Option<(Vec<String>, usize)> {
    actions.iter().rev().find_map(|a| match a {
        Action::ShowCandidates { list, selected } => Some((list.clone(), *selected)),
        _ => None,
    })
}

#[test]
fn the_models_read_the_context_after_anything_sensitive() {
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.set_app_context(Some("詳細はhttps://example.com/aを見て、"));
    assert_eq!(bridge.context(), "を見て、");
    bridge.set_app_context(Some("今日は晴れ。"));
    assert_eq!(bridge.context(), "今日は晴れ。");
}

#[test]
fn a_private_field_has_no_context_and_learns_nothing() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_private_field(true);
    bridge.set_app_context(Some("今日は"));
    assert_eq!(bridge.context(), "");
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_space();
    bridge.handle_down();
    bridge.handle_enter();
    // the commit is no context either
    assert_eq!(bridge.context(), "");
    bridge.set_private_field(false);
    bridge.set_app_context(Some("今日は"));
    assert_eq!(bridge.context(), "今日は");
}

#[test]
fn the_rewriter_gets_no_sensitive_context_nor_text() {
    let (mut bridge, mock) = bridge_with_mock();
    let rewriter = MockRewriter::default();
    bridge.set_rewriter(Some(Box::new(rewriter.clone())));
    bridge.set_rewrite_mode(true);
    // the context holds a password word: none of it is sent, not even
    // the text after it
    bridge.set_app_context(Some("パスワードは"));
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    bridge.apply_async_result();
    bridge.handle_space();
    assert_eq!(rewriter.state.dispatched.borrow()[0].1, "");
    bridge.handle_escape();
    bridge.handle_escape();
    bridge.set_app_context(Some("今日は"));
    type_str(&mut bridge, "kasano");
    bridge.handle_space();
    assert_eq!(
        rewriter.state.dispatched.borrow().last().unwrap().1,
        "今日は"
    );
    // a conversion that is a token itself is not sent at all
    bridge.handle_escape();
    bridge.handle_escape();
    let before = rewriter.state.dispatched.borrow().len();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["AKIAIOSFODNN7EXAMPLE"]);
    bridge.apply_async_result();
    bridge.handle_space();
    assert_eq!(rewriter.state.dispatched.borrow().len(), before);
}

#[test]
fn rewrite_mode_needs_a_rewriter() {
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.set_rewrite_mode(true);
    assert!(!bridge.rewrite_mode());
}

#[test]
fn the_conversion_key_asks_for_a_rewrite_that_comes_first() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    let out = bridge.handle_space();
    assert_eq!(status(&out.actions), Some("書き直し中…"));
    {
        let sent = rewriter.state.dispatched.borrow();
        assert_eq!((sent[0].2.as_str(), sent[0].3.as_str()), ("かさの", "傘の"));
    }
    rewriter.answer(Ok("傘の"));
    // the same text as the engine's top: it moves first, once
    let actions = bridge.apply_async_result();
    let (list, selected) = shown(&actions).unwrap();
    assert_eq!(
        (list[0].as_str(), list[1].as_str(), selected),
        ("傘の", "笠の", 0)
    );
    assert_eq!(status(&actions), Some(""));
}

#[test]
fn a_conversion_opened_on_its_segments_asks_for_no_rewrite() {
    // Space on the last segment
    let (mut bridge, mock, rewriter) = rewriting_bridge();
    bridge.cfg.space_last_segment = true;
    let pos: &[u32] = &[0, 2];
    mock.set_aligned("かさの", &["傘の", "笠の"], &[pos, pos]);
    bridge.apply_async_result();
    let out = bridge.handle_space();
    assert_eq!(status(&out.actions), None, "{:?}", out.actions);
    assert!(rewriter.state.dispatched.borrow().is_empty());
    // ← on the live conversion
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    let out = bridge.handle_move_cursor(-1);
    assert!(bridge.is_converting());
    assert_eq!(status(&out.actions), None, "{:?}", out.actions);
    assert!(rewriter.state.dispatched.borrow().is_empty());
}

#[test]
fn a_rewrite_keeps_a_selection_off_the_top() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.handle_space();
    bridge.handle_down();
    rewriter.answer(Ok("傘をさして"));
    let actions = bridge.apply_async_result();
    let (list, selected) = shown(&actions).unwrap();
    assert_eq!(list[0], "傘をさして");
    assert_eq!(list[selected], "笠の");
    assert!(actions.contains(&Action::UpdatePreedit {
        text: "笠の".into()
    }));
    // Enter commits the selection, the context grows by it
    let out = bridge.handle_enter();
    assert!(out.actions.contains(&Action::Commit {
        text: "笠の".into()
    }));
}

#[test]
fn enter_commits_the_rewrite() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.handle_space();
    rewriter.answer(Ok("傘をさして"));
    bridge.apply_async_result();
    let out = bridge.handle_enter();
    assert!(out.actions.contains(&Action::Commit {
        text: "傘をさして".into()
    }));
    assert_eq!(bridge.context(), "傘をさして");
}

#[test]
fn escape_leaves_the_rewrite_out_of_typing() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.cfg.escape_to_kana = false;
    bridge.handle_space();
    rewriter.answer(Ok("傘をさして"));
    bridge.apply_async_result();
    // back to typing: Enter commits the engine's conversion
    bridge.handle_escape();
    let out = bridge.handle_enter();
    assert!(out.actions.contains(&Action::Commit {
        text: "傘の".into()
    }));
}

#[test]
fn the_same_reading_is_rewritten_once() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.handle_space();
    rewriter.answer(Ok("傘をさして"));
    bridge.apply_async_result();
    bridge.handle_escape();
    let out = bridge.handle_space();
    assert_eq!(rewriter.requests(), 1);
    assert_eq!(status(&out.actions), None);
    assert_eq!(shown(&out.actions).unwrap().0[0], "傘をさして");
}

#[test]
fn a_failed_rewrite_says_why() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.handle_space();
    rewriter.answer(Err("usage_limit"));
    let actions = bridge.apply_async_result();
    assert_eq!(
        status(&actions),
        Some(crate::rewrite_client::failure_message("usage_limit"))
    );
    assert!(shown(&actions).is_none());
}

#[test]
fn rewrite_mode_turned_on_over_an_open_conversion_asks_at_once() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    bridge.set_rewrite_mode(false);
    bridge.handle_space();
    assert_eq!(rewriter.requests(), 0);
    let actions = bridge.set_rewrite_mode(true);
    assert_eq!(status(&actions), Some("書き直し中…"));
    assert_eq!(rewriter.requests(), 1);
}

#[test]
fn user_words_come_first_and_inside_longer_readings() {
    let (mut bridge, mock) = bridge_with_mock();
    let (entries, _) = crate::user_dict::parse("かさ\t笠\t名詞\n");
    bridge.set_user_dict(Some(crate::user_dict::UserDict::new(&entries)));
    type_str(&mut bridge, "kasa");
    mock.set_result("かさ", &["傘", "嵩"]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "笠");
    bridge.handle_escape();
    bridge.handle_escape();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "嵩の"]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "笠の");
}

#[test]
fn ctrl_delete_forgets_a_learned_candidate_or_hides_it() {
    let dir = std::env::temp_dir().join(format!("setsuna-bridge-forget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (mut bridge, mock) = bridge_with_mock();
    let cfg = crate::config::LearningCfg::default();
    bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
    bridge.set_hidden(Some(crate::hidden::Hidden::at(dir.join("hidden.jsonl"))));
    let convert = |bridge: &mut EngineBridge| {
        type_str(bridge, "kasano");
        mock.set_result("かさの", &["傘の", "笠の", "嵩の"]);
        bridge.apply_async_result();
        bridge.handle_space()
    };
    // nothing learned for 笠の: hidden from now on
    convert(&mut bridge);
    bridge.handle_down();
    let out = bridge.handle_forget_candidate();
    assert_eq!(status(&out.actions), Some("候補を非表示にしました"));
    assert!(!shown(&out.actions).unwrap().0.contains(&"笠の".to_string()));
    bridge.handle_escape();
    bridge.handle_escape();
    let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
    assert!(!list.contains(&"笠の".to_string()));
    // 嵩の chosen, so learned first; Ctrl+Delete forgets that
    for _ in 0..list.iter().position(|c| c == "嵩の").unwrap() {
        bridge.handle_down();
    }
    bridge.handle_enter();
    let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
    assert_eq!(list[0], "嵩の");
    let out = bridge.handle_forget_candidate();
    assert_eq!(status(&out.actions), Some("学習を取り消しました"));
    bridge.handle_escape();
    bridge.handle_escape();
    let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
    assert_eq!(list[0], "傘の");
    // the kana stay
    let i = list.iter().position(|c| c == "かさの").unwrap();
    for _ in 0..i {
        bridge.handle_down();
    }
    let out = bridge.handle_forget_candidate();
    assert_eq!(status(&out.actions), Some("この候補は非表示にできません"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_private_field_hides_no_candidate() {
    let dir = std::env::temp_dir().join(format!(
        "setsuna-bridge-hide-private-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_hidden(Some(crate::hidden::Hidden::at(dir.join("hidden.jsonl"))));
    bridge.set_private_field(true);
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の", "嵩の"]);
    bridge.apply_async_result();
    bridge.handle_space();
    bridge.handle_down();
    let out = bridge.handle_forget_candidate();
    assert_eq!(
        status(&out.actions),
        Some("この欄では候補を非表示にできません")
    );
    assert!(!dir.join("hidden.jsonl").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn picking_the_engines_second_over_a_rewrite_learns_against_its_top() {
    let (mut bridge, _mock, rewriter) = rewriting_bridge();
    let dir = std::env::temp_dir().join(format!(
        "setsuna-bridge-rewrite-learn-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = crate::config::LearningCfg::default();
    bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
    bridge.handle_space();
    rewriter.answer(Ok("傘をさして"));
    bridge.apply_async_result();
    // the rewrite, 傘の, 笠の: pick 笠の
    bridge.handle_down();
    bridge.handle_down();
    bridge.handle_enter();
    let log = std::fs::read_to_string(dir.join("learning.jsonl")).unwrap();
    assert!(log.contains("笠の") && log.contains("傘の"), "{log}");
    assert!(!log.contains("傘をさして"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The engine's answer to the last request, with converter scores.
fn answer_scored(mock: &MockClient, candidates: &[&str], scores: &[f32]) {
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: candidates.iter().map(|s| s.to_string()).collect(),
        scores: scores.to_vec(),
        ..Default::default()
    });
}

/// A bridge on a mock engine. Emoji are off: the lists the tests
/// compare are the engine's (`emoji_join_the_list` has them on).
fn bridge_with_mock() -> (EngineBridge, MockClient) {
    let mock = MockClient::default();
    let mut bridge = EngineBridge::new(Some(Box::new(mock.clone())));
    bridge.cfg.emoji_enabled = false;
    (bridge, mock)
}

#[test]
fn commands_typed_at_a_word_start_go_to_the_app() {
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.cfg.pass_commands = true;
    bridge.set_app_context(Some("hello "));
    assert!(!bridge.handle_punctuation("・").consumed);
    for c in "review-pr".chars() {
        assert!(!bridge.handle_char(c).consumed, "{c}");
    }
    assert!(!bridge.would_consume(&KeyKind::Char('x')));
    // the space ends it: romaji again
    bridge.handle_space();
    assert!(bridge.handle_char('k').consumed);
    bridge.handle_escape();
    // not at a word's start, or the text unknown: a symbol as usual
    bridge.set_app_context(Some("今日は"));
    assert!(bridge.handle_punctuation("・").consumed);
    bridge.handle_escape();
    bridge.forget_context();
    assert!(bridge.handle_punctuation("＠").consumed);
}

#[test]
fn a_deferred_preedit_changes_once() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    bridge.cfg.rerank_cfg.defer_ms = 250;
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
    // the engine's top waits for the reranked one: the kana meanwhile
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("かさの")
    );
    assert!(bridge.deferred_for().is_some());
    assert!(bridge.tick().is_empty());
    rr.answer(&[-30.0, -20.0]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("笠の")
    );
    assert!(bridge.deferred_for().is_none());
    // none by the deadline: the engine's top
    bridge.handle_escape();
    bridge.handle_escape();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
    bridge.apply_async_result();
    bridge.comp.deferred.as_mut().unwrap().until = Instant::now();
    assert_eq!(preedit(&bridge.tick()).as_deref(), Some("傘の"));
}

#[test]
fn commits_are_predicted_and_tab_opens_them() {
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-predict-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    bridge.set_predictor(Some(crate::predict::Predictor::at(path.clone())));
    type_str(&mut bridge, "kasanoshita");
    mock.set_result("かさのした", &["傘の下"]);
    bridge.apply_async_result();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の下"]);
    // its start typed: predicted, but no window comes up by itself;
    // Tab opens them
    type_str(&mut bridge, "kasanos");
    assert_eq!(bridge.predictions().len(), 1);
    assert!(bridge.suggestions().is_empty());
    let out = bridge.handle_tab(false);
    assert_eq!(preedit(&out.actions).as_deref(), Some("傘の下"));
    // the s still being typed is no part of it
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の下"]);
    // Escape from them goes back to typing
    type_str(&mut bridge, "kasano");
    bridge.handle_tab(false);
    bridge.handle_escape();
    assert!(!bridge.is_converting() && bridge.is_active());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn emoji_join_the_list() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.emoji_enabled = true;
    type_str(&mut bridge, "nikoniko");
    mock.set_result("にこにこ", &["ニコニコ"]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert!(list.contains(&"😊".to_string()));
    assert!(list.last().unwrap().contains('^'));
}

#[test]
fn a_client_whose_daemon_exited_counts_as_none() {
    // the frontend starts a new daemon when `has_client` is false; a dead
    // one kept counting as a client left the IME converting nothing
    let (bridge, mock) = bridge_with_mock();
    assert!(bridge.has_client());
    mock.state.gone.set(true);
    assert!(!bridge.has_client());
    assert!(!EngineBridge::new(None).has_client());
}

fn type_str(bridge: &mut EngineBridge, s: &str) {
    for c in s.chars() {
        bridge.handle_char(c);
    }
}

/// A bridge with the small mixed-input model of the `mixed` tests.
fn mixed_bridge() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    let model = std::sync::Arc::new(crate::mixed::tests::model());
    bridge.set_mixed(
        Some(model),
        &crate::config::default_symbol_map(),
        crate::mixed::Casing::default(),
    );
    (bridge, mock)
}

#[test]
fn mixed_shift_letters_type_into_the_composition() {
    let (mut bridge, _mock) = mixed_bridge();
    for c in "GitHub".chars() {
        if c.is_ascii_uppercase() {
            bridge.handle_shifted_alpha(c);
        } else {
            bridge.handle_char(c);
        }
    }
    let out = {
        type_str(&mut bridge, "ni");
        bridge.handle_enter()
    };
    assert_eq!(commits(&out.actions), vec!["GitHubに"]);
}

#[test]
fn mixed_space_after_english_types_a_space() {
    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "hello");
    let out = bridge.handle_space();
    assert_eq!(preedit(&out.actions).as_deref(), Some("hello "));
    assert!(!bridge.is_converting());
    type_str(&mut bridge, "google");
    // two words: an English sentence, with its capital
    assert_eq!(
        preedit(&bridge.handle_space().actions).as_deref(),
        Some("Hello google ")
    );
    assert_eq!(
        commits(&bridge.handle_enter().actions),
        vec!["Hello google "]
    );
}

#[test]
fn mixed_space_twice_after_english_offers_it_with_a_space() {
    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "hello");
    bridge.handle_space();
    let out = bridge.handle_space();
    assert!(bridge.is_converting());
    assert_eq!(preedit(&out.actions).as_deref(), Some("hello "));
    // the next key commits it, as between English words
    assert_eq!(commits(&bridge.handle_char('w').actions), vec!["hello "]);
}

#[test]
fn live_conversion_reads_a_pending_n_as_n_kana() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "kekkon");
    // the engine converts けっこん, not けっこ, for Enter; the preview
    // shows the n as typed
    assert_eq!(mock.dispatched_readings().last().unwrap(), "けっこん");
    mock.set_result("けっこん", &["結婚"]);
    let out = bridge.apply_async_result();
    assert_eq!(preedit(&out).as_deref(), Some("けっこn"));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["結婚"]);
    // a vowel after it converts again
    type_str(&mut bridge, "kekkona");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "けっこな");
    // typed to its end (nn), the conversion of けっこん is shown
    bridge.handle_escape();
    type_str(&mut bridge, "kekkonn");
    mock.set_result("けっこん", &["結婚"]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("結婚")
    );
}

#[test]
fn a_pending_n_shows_as_typed_after_the_conversion_before_it() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "kyouha");
    mock.set_result("きょうは", &["今日は"]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("今日は")
    );
    let out = bridge.handle_char('n');
    assert_eq!(preedit(&out.actions).as_deref(), Some("今日はn"));
    // the conversion of きょうはん does not take its place
    mock.set_result("きょうはん", &["共犯"]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("今日はn")
    );
    // nor does the whole go back to kana when the n becomes な
    let out = bridge.handle_char('a');
    assert_eq!(preedit(&out.actions).as_deref(), Some("今日はな"));
    mock.set_result("きょうはな", &["今日花"]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("今日花")
    );
    // Enter on a pending n commits the conversion with ん
    bridge.handle_escape();
    bridge.handle_escape();
    type_str(&mut bridge, "kyouhan");
    mock.set_result("きょうはん", &["共犯"]);
    bridge.apply_async_result();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["共犯"]);
    // and Space shows it, not asking the engine again
    type_str(&mut bridge, "kyouhan");
    mock.set_result("きょうはん", &["共犯"]);
    bridge.apply_async_result();
    let sent = mock.dispatched_readings().len();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "共犯");
    assert_eq!(mock.dispatched_readings().len(), sent);
}

#[test]
fn a_reading_whose_end_changed_keeps_the_conversion_it_shares() {
    // mixed input reading letters otherwise, a kana replaced: the shared
    // start stays converted while the new end is
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "watashiha");
    mock.set_result("わたしは", &["私は"]);
    bridge.apply_async_result();
    bridge.comp.cached_prefix_reading = "わたしわ".into();
    bridge.comp.cached_prefix_kanji = "私わ".into();
    assert_eq!(bridge.live_preedit(), "私は");
}

#[test]
fn mixed_enter_commits_what_the_whole_reads_as() {
    // the shipped model knows ok
    let (mut bridge, mock) = bridge_with_mock();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
    bridge.set_mixed(
        Some(std::sync::Arc::new(
            crate::mixed::MixedModel::load(&dir).unwrap(),
        )),
        &crate::config::default_symbol_map(),
        crate::mixed::Casing::default(),
    );
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "ok");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["OK"]);
    // with kana in it, that reading is converted, not committed as kana
    // (げんだいぶんめいok: お and a pending k while typed)
    mock.state
        .answers
        .borrow_mut()
        .insert("げんだいぶんめいOK".into(), vec!["現代文明OK".into()]);
    type_str(&mut bridge, "genndaibunnmeiok");
    assert_eq!(bridge.live_preedit(), "げんだいぶんめいおk");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["現代文明OK"]);
}

#[test]
fn a_key_can_wait_for_its_conversion_and_show_it_at_once() {
    let (mut bridge, mock) = bridge_with_mock();
    mock.state
        .answers
        .borrow_mut()
        .insert("きしゃ".into(), vec!["記者".into()]);
    let mut last = Vec::new();
    for c in "kisha".chars() {
        last = bridge.handle_char(c).actions;
    }
    // the key shows the kana; its conversion, in at once, stands for them
    assert_eq!(preedit(&last).as_deref(), Some("きしゃ"));
    let wait = std::time::Duration::from_millis(4);
    assert_eq!(preedit(&bridge.settle_live(wait)).as_deref(), Some("記者"));
    // nothing out: nothing to wait for
    assert!(bridge.settle_live(wait).is_empty());
    // an engine slower than the wait lately: no wait, its kana as before
    let (mut bridge, mock) = bridge_with_mock();
    mock.state
        .answers
        .borrow_mut()
        .insert("きしゃ".into(), vec!["記者".into()]);
    bridge.live_rtt_ms = 9.0;
    for c in "kisha".chars() {
        bridge.handle_char(c);
    }
    assert!(bridge.settle_live(wait).is_empty());
    assert_eq!(bridge.live_preedit(), "きしゃ");
}

#[test]
fn mixed_enter_and_space_convert_the_reading_shown() {
    // the shipped model shows Nandinaha as Nandina + は; read again when
    // done it was Nand + いなは, and Enter committed Nand否は; Space froze
    // it as Nandinaは, which read Nand + いなは again
    let bridge = || {
        let (mut bridge, mock) = bridge_with_mock();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
        bridge.set_mixed(
            Some(std::sync::Arc::new(
                crate::mixed::MixedModel::load(&dir).unwrap(),
            )),
            &crate::config::default_symbol_map(),
            crate::mixed::Casing::default(),
        );
        bridge.cfg.live_conversion = true;
        for (reading, text) in [("Nandinaは", "Nandina葉"), ("Nandいなは", "Nand否は")] {
            mock.state
                .answers
                .borrow_mut()
                .insert(reading.into(), vec![text.into()]);
        }
        type_str(&mut bridge, "Nandinaha");
        assert_eq!(bridge.composing.preedit(), "Nandinaは");
        (bridge, mock)
    };
    let (mut b, _mock) = bridge();
    assert_eq!(commits(&b.handle_enter().actions), vec!["Nandina葉"]);
    for force in [true, false] {
        let (mut b, mock) = bridge();
        b.cfg.force_convert_on_space = force;
        b.handle_space();
        assert_eq!(b.composing.hiragana(), "Nandinaは");
        assert_eq!(mock.dispatched_readings().last().unwrap(), "Nandinaは");
        mock.set_result("Nandinaは", &["Nandina葉"]);
        b.apply_async_result();
        assert_eq!(commits(&b.handle_enter().actions), vec!["Nandina葉"]);
    }
}

#[test]
fn mixed_a_slip_shown_as_english_reads_again_when_deleted() {
    // the shipped model reads sp as an acronym; deleting the p leaves
    // the s typed, which reads as romaji again (frozen as SP, the
    // capitals kept it English)
    let (mut bridge, _mock) = bridge_with_mock();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
    bridge.set_mixed(
        Some(std::sync::Arc::new(
            crate::mixed::MixedModel::load(&dir).unwrap(),
        )),
        &crate::config::default_symbol_map(),
        crate::mixed::Casing::default(),
    );
    type_str(&mut bridge, "nanidemosp");
    assert_eq!(bridge.composing.preedit(), "なにでもSP");
    bridge.handle_backspace();
    assert_eq!(bridge.composing.preedit(), "なにでもs");
    type_str(&mut bridge, "u");
    assert_eq!(bridge.composing.preedit(), "なにでもす");
}

#[test]
fn mixed_space_after_a_lone_i_waits_for_the_next_word() {
    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "i");
    bridge.handle_space();
    assert!(!bridge.is_converting());
    type_str(&mut bridge, "hello");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["I hello"]);
    // Space again converts い
    type_str(&mut bridge, "i");
    bridge.handle_space();
    bridge.handle_space();
    assert!(bridge.is_converting());
}

#[test]
fn period_after_a_digit_types_a_point() {
    let (mut bridge, _mock) = bridge_with_mock();
    type_str(&mut bridge, "3");
    let out = bridge.handle_punctuation("。");
    assert_eq!(preedit(&out.actions).as_deref(), Some("3."));
    type_str(&mut bridge, "desu");
    let out = bridge.handle_punctuation("。");
    assert_eq!(preedit(&out.actions).as_deref(), Some("3.です。"));

    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "3");
    let out = bridge.handle_punctuation("。");
    assert_eq!(preedit(&out.actions).as_deref(), Some("3."));
}

#[test]
fn mixed_conversion_offers_the_reading_without_english() {
    let (mut bridge, mock) = mixed_bridge();
    type_str(&mut bridge, "hello");
    // the first Space types a space
    bridge.handle_space();
    bridge.handle_space();
    mock.answer_last(&["hello"]);
    bridge.apply_async_result();
    // with the English list in, the reading with every letter as romaji
    assert_eq!(mock.dispatched_readings().last().unwrap(), "へっぉ");
    mock.answer_last(&["減っぉ"]);
    let out = bridge.apply_async_result();
    let list = out
        .iter()
        .find_map(|a| match a {
            Action::ShowCandidates { list, .. } => Some(list.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(list[0], "hello ");
    assert_eq!(list[1], "減っぉ");
    assert!(list.contains(&"hello".to_string()));
    // Space again goes on to it
    assert_eq!(
        preedit(&bridge.handle_space().actions).as_deref(),
        Some("減っぉ")
    );
}

#[test]
fn mixed_kana_forms_read_every_letter_as_romaji() {
    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "hello");
    let out = bridge.handle_convert_to(Form::Hiragana);
    assert_eq!(preedit(&out.actions).as_deref(), Some("へっぉ"));
}

#[test]
fn mixed_symbols_show_per_span() {
    let (mut bridge, _mock) = mixed_bridge();
    type_str(&mut bridge, "example");
    bridge.handle_punctuation("。");
    type_str(&mut bridge, "com");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["example.com"]);
    type_str(&mut bridge, "desu");
    bridge.handle_punctuation("。");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["です。"]);
}

#[test]
fn mixed_input_types_the_symbols_in_the_width_chosen() {
    let model = std::sync::Arc::new(crate::mixed::tests::model());
    for (full, paren, question) in [(false, "(", "?"), (true, "（", "？")] {
        let symbols = crate::config::symbols_in_width(&crate::config::default_symbol_map(), full);
        let (mut bridge, _mock) = bridge_with_mock();
        bridge.set_mixed(
            Some(model.clone()),
            &symbols,
            crate::mixed::Casing::default(),
        );
        // what the frontends hand over for the keys: the table's symbol
        let typed = |c: char| crate::config::typed_symbol(c, false, &symbols);
        type_str(&mut bridge, "desu");
        bridge.handle_punctuation(&typed('('));
        type_str(&mut bridge, "ka");
        bridge.handle_punctuation(&typed('?'));
        bridge.handle_punctuation(&typed('.'));
        assert_eq!(
            commits(&bridge.handle_enter().actions),
            vec![format!("です{paren}か{question}。")]
        );
    }
}

#[test]
fn punctuation_after_an_english_sentence_is_english() {
    let (mut bridge, _mock) = bridge_with_mock();
    // typed with Shift: it goes on with the English
    bridge.handle_shifted_alpha('T');
    type_str(&mut bridge, "his");
    bridge.handle_space();
    type_str(&mut bridge, "works");
    let out = bridge.handle_punctuation("！");
    assert_eq!(preedit(&out.actions).as_deref(), Some("This works!"));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["This works!"]);
    // after English committed: committed as typed
    bridge.set_app_context(Some("This is a pen"));
    let out = bridge.handle_punctuation("。");
    assert_eq!(commits(&out.actions), vec!["."]);
    // English inside Japanese: Japanese punctuation
    bridge.set_app_context(Some("これはGoogle"));
    let out = bridge.handle_punctuation("。");
    assert_eq!(preedit(&out.actions).as_deref(), Some("。"));
}

#[test]
fn mixed_punctuation_follows_the_sentence() {
    let (mut bridge, mock) = mixed_bridge();
    type_str(&mut bridge, "hello");
    bridge.handle_punctuation("、");
    // Space after English punctuation types a space, as after a word
    let out = bridge.handle_space();
    assert_eq!(preedit(&out.actions).as_deref(), Some("hello, "));
    type_str(&mut bridge, "google");
    bridge.handle_punctuation("。");
    assert_eq!(
        commits(&bridge.handle_enter().actions),
        vec!["Hello, google."]
    );
    // a candidate that turned it into 。 is a misconversion
    type_str(&mut bridge, "korehagoogle");
    bridge.handle_punctuation("。");
    mock.set_result("これはgoogle。", &["これはgoogle。"]);
    bridge.apply_async_result();
    assert_eq!(
        commits(&bridge.handle_enter().actions),
        vec!["これはgoogle。"]
    );
    assert!(crate::mixed::keeps_latin("hello, world.", "hello, world."));
    assert!(!crate::mixed::keeps_latin(
        "hello, world.",
        "hello、world。"
    ));
}

#[test]
fn left_on_a_live_conversion_ending_in_english_types_no_space() {
    let (mut bridge, mock) = mixed_bridge();
    type_str(&mut bridge, "kyouhagoogle");
    mock.set_result("きょうはgoogle", &["今日はgoogle", "きょうはgoogle"]);
    bridge.apply_async_result();
    let out = bridge.handle_move_cursor(-1);
    assert!(out.consumed);
    // the conversion's segments to edit, no space after the English (as
    // Space there types)
    assert!(bridge.is_converting(), "{:?}", out.actions);
    assert!(
        !out.actions
            .iter()
            .any(|a| matches!(a, Action::UpdatePreedit { text } if text.ends_with(' '))),
        "{:?}",
        out.actions
    );
}

#[test]
fn mixed_candidates_keep_the_english_as_typed() {
    let (mut bridge, mock) = mixed_bridge();
    type_str(&mut bridge, "googlede");
    mock.set_result("googleで", &["Googleで", "googleで"]);
    bridge.apply_async_result();
    let out = bridge.handle_space();
    assert!(out.actions.iter().any(|a| matches!(
        a,
        Action::ShowCandidates { list, .. } if list[0] == "googleで" && !list.iter().any(|c| c == "Googleで")
    )));
}

#[test]
fn char_enters_composing_with_raw_preedit() {
    let (mut bridge, _mock) = bridge_with_mock();
    let outcome = bridge.handle_char('k');
    assert!(outcome.consumed);
    assert_eq!(
        outcome.actions,
        vec![Action::UpdatePreedit { text: "k".into() }]
    );
    assert!(bridge.is_active());
}

#[test]
fn completed_kana_dispatches_convert() {
    let (mut bridge, mock) = bridge_with_mock();
    // 3+ char reading required: dispatch_live skips OOD short partials
    // (see MIN_DISPATCH_CHARS). The guard only fires when hira chars < 3,
    // so we type 3 chars to verify dispatch in the normal case.
    type_str(&mut bridge, "kasa");
    // type "ka"->"か" (1 char, no dispatch), "sa"->"かさ" (2 chars, no
    // dispatch) — actually we need 3 chars, so type one more:
    type_str(&mut bridge, "no");
    assert_eq!(
        mock.dispatched_readings().last(),
        Some(&"かさの".to_string())
    );
}

#[test]
fn short_reading_skips_dispatch() {
    // Regression test for the dispatch guard: 1- and 2-char readings
    // must NOT trigger a daemon dispatch — they are OOD on the
    // sentence-trained CTC-NAT and produce nonsense flicker. The
    // preedit still shows raw hira; conversion fires once the reading
    // grows past MIN_DISPATCH_CHARS.
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    assert!(mock.dispatched_readings().is_empty(), "1-char skipped");
    type_str(&mut bridge, "sa");
    assert!(mock.dispatched_readings().is_empty(), "2-char skipped");
    type_str(&mut bridge, "no");
    // Now ≥3 chars → dispatch.
    assert_eq!(
        mock.dispatched_readings().last(),
        Some(&"かさの".to_string())
    );
}

#[test]
fn space_forces_dispatch_for_short_reading() {
    // Live typing still skips 1-char readings, but explicit Space must
    // ask the daemon so dictionary/model fallback can return more than
    // the provisional single candidate.
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    assert!(mock.dispatched_readings().is_empty(), "live skip remains");

    let outcome = bridge.handle_space();
    assert!(outcome.consumed);
    assert_eq!(mock.dispatched_readings().last(), Some(&"か".to_string()));
    assert_eq!(
        outcome.actions,
        vec![
            Action::UpdatePreedit { text: "か".into() },
            Action::ShowCandidates {
                list: vec!["か".into()],
                selected: 0,
            },
        ]
    );
}

#[test]
fn space_flushes_pending_n_before_dispatch() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "shin");
    assert!(mock.dispatched_readings().is_empty());

    let outcome = bridge.handle_space();
    assert!(outcome.consumed);
    assert_eq!(mock.dispatched_readings().last(), Some(&"しん".to_string()));
    assert_eq!(
        outcome.actions,
        vec![
            Action::UpdatePreedit {
                text: "しん".into()
            },
            Action::ShowCandidates {
                list: vec!["しん".into()],
                selected: 0,
            },
        ]
    );
}

#[test]
fn async_result_swaps_preedit_to_kanji() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    mock.set_result("か", &["可", "蚊"]);
    let actions = bridge.apply_async_result();
    assert_eq!(actions, vec![Action::UpdatePreedit { text: "可".into() }]);
}

#[test]
fn a_typed_number_stays_as_typed_when_the_model_merges_its_digits() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "19911225");
    mock.set_result("19911225", &["199125", "1991125"]);
    let actions = bridge.apply_async_result();
    assert_eq!(
        actions,
        vec![Action::UpdatePreedit {
            text: "19911225".into()
        }]
    );
    let outcome = bridge.handle_space();
    let Some(Action::ShowCandidates { list, .. }) = outcome.actions.last() else {
        panic!("no candidate window: {:?}", outcome.actions);
    };
    assert_eq!(list[0], "19911225");
    assert!(!list.iter().any(|c| c == "199125" || c == "1991125"));
}

#[test]
fn space_opens_candidate_window_with_async_list() {
    let (mut bridge, mock) = bridge_with_mock();
    // ≥3 char reading so dispatch_live actually fires (otherwise the
    // MIN_DISPATCH_CHARS guard short-circuits and the mock daemon
    // never gets the reading).
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    let outcome = bridge.handle_space();
    assert!(outcome.consumed);
    assert_eq!(
        outcome.actions,
        vec![
            Action::UpdatePreedit {
                text: "傘の".into()
            },
            // The reading and its katakana are always offered last.
            Action::ShowCandidates {
                list: vec![
                    "傘の".into(),
                    "笠の".into(),
                    "かさの".into(),
                    "カサノ".into()
                ],
                selected: 0,
            },
        ]
    );
}

#[test]
fn space_before_result_shows_provisional_then_async_swaps() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    // No result applied yet: Space opens a provisional 1-item window.
    let outcome = bridge.handle_space();
    assert_eq!(
        outcome.actions,
        vec![
            Action::UpdatePreedit { text: "か".into() },
            Action::ShowCandidates {
                list: vec!["か".into()],
                selected: 0,
            },
        ]
    );
    // The async result then swaps in the real candidate list.
    mock.set_result("か", &["可", "蚊"]);
    let actions = bridge.apply_async_result();
    assert_eq!(
        actions,
        vec![
            Action::UpdatePreedit { text: "可".into() },
            Action::ShowCandidates {
                list: vec!["可".into(), "蚊".into(), "か".into(), "カ".into()],
                selected: 0,
            },
        ]
    );
}

#[test]
fn wrong_hiragana_candidate_is_replaced_by_the_reading() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "moji");
    mock.set_result("もじ", &["文字", "もし"]);
    bridge.apply_async_result();
    let outcome = bridge.handle_space();
    assert_eq!(
        outcome.actions[1],
        Action::ShowCandidates {
            list: vec!["文字".into(), "もじ".into(), "モジ".into()],
            selected: 0,
        }
    );
}

fn converted_kasano() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    (bridge, mock)
}

#[test]
fn enter_keeps_the_unconverted_completed_tail() {
    let (mut bridge, _) = converted_kasano();
    type_str(&mut bridge, "ame");
    assert_eq!(
        bridge.handle_enter().actions,
        vec![Action::Commit {
            text: "傘のあめ".into()
        }]
    );
}

#[test]
fn backspace_deletes_the_last_char_shown_and_keeps_the_rest() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "toukyouniiku");
    mock.set_result("とうきょうにいく", &["東京に行く", "東京二行く"]);
    bridge.apply_async_result();
    assert_eq!(bridge.live_preedit(), "東京に行く");
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "東京に行"
    );
    // the shorter reading is converted for its list
    assert_eq!(mock.dispatched_readings().last().unwrap(), "とうきょうにい");
    // its conversion does not change what is shown, and lists it first
    mock.set_result("とうきょうにい", &["東京に胃", "東京二位"]);
    bridge.apply_async_result();
    assert_eq!(bridge.live_preedit(), "東京に行");
    // 行 goes with its い, に with its に
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "東京に"
    );
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東京");
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "東京");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["東京"]);
}

#[test]
fn backspace_deletes_a_kanji_with_all_its_reading() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "toukyou");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: vec!["東京".into()],
        // the engine's positions: 東 at と, 京 at き
        reading_pos: vec![vec![0, 2]],
        ..Default::default()
    });
    bridge.apply_async_result();
    // 京 takes きょう
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東");
    assert_eq!(bridge.composing.hiragana(), "とう");
    // and the last char goes with the composition
    let out = bridge.handle_backspace();
    assert!(
        out.actions.contains(&Action::EndComposition),
        "{:?}",
        out.actions
    );
    assert!(!bridge.is_active());
    // without the engine's positions a kanji run is placed as one: the
    // lexicon's readings of its chars split it
    let table = "東\tとう\t10.0\tnoun\n京\tきょう\t10.0\tnoun\n今日\tきょう\t8.0\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-split-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let lexicon = crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    };
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&lexicon);
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "toukyouno");
    mock.set_result("とうきょうの", &["東京の"]);
    bridge.apply_async_result();
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東京");
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東");
    // one its chars do not split (今日, read きょう whole): a kana of
    // the reading goes, not both chars
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&lexicon);
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "kyouha");
    mock.set_result("きょうは", &["今日は"]);
    bridge.apply_async_result();
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "今日");
    bridge.handle_backspace();
    assert_eq!(bridge.composing.hiragana(), "きょ");
    assert!(bridge.is_active());
    let _ = std::fs::remove_file(&path);
}

/// The engine's positions inside a kanji run are where its tokens came
/// out, not where their readings start (社 of 会社 at the ゃ of かいしゃ):
/// the lexicon's readings of the chars place them, so Backspace takes
/// 社 with しゃ and what is shown is what is held (会 shown with かいし
/// held gave 開始ア for an `a` typed after, on the phone).
#[test]
fn backspace_takes_a_kanji_s_whole_reading_where_the_engine_put_it_late() {
    let table = "会\tかい\t10.0\tnoun\n社\tしゃ\t10.0\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-late-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "kaisha");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: vec!["会社".into()],
        reading_pos: vec![vec![0, 3]],
        ..Default::default()
    });
    bridge.apply_async_result();
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "会");
    assert_eq!(bridge.composing.hiragana(), "かい");
    let _ = std::fs::remove_file(&path);
}

/// A conversion shown while its reading shrinks keeps the chars read
/// within what is left and shows the kana after them as typed: a char
/// kept by counting kana hid the kana left of its reading (会 for かいし).
#[test]
fn a_shrunk_reading_shows_its_kana_not_a_char_cut_by_count() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    // a kana deleted, not the char shown
    bridge.cfg.backspace_deletes_shown = false;
    type_str(&mut bridge, "kaisha");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: vec!["会社".into()],
        reading_pos: vec![vec![0, 2]],
        ..Default::default()
    });
    bridge.apply_async_result();
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "会し");
    // where the chars' readings are not known: the kana
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    bridge.cfg.backspace_deletes_shown = false;
    type_str(&mut bridge, "kaisha");
    mock.set_result("かいしゃ", &["会社"]);
    bridge.apply_async_result();
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "かいし"
    );
}

/// A letter typed before a kana stays in the reading as it is (`kna` is
/// kな) and goes back to romaji pending when the kana after it goes:
/// what Backspace keeps shown leaves it to the pending romaji (含め and
/// k showed 含めkk, and Enter committed it, on this PC: "含めk直せる",
/// "含めkk").
#[test]
fn backspace_leaves_a_letter_gone_back_to_romaji_to_it() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "fukumeknao");
    assert_eq!(bridge.composing.hiragana(), "ふくめkなお");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: vec!["含めk治".into()],
        reading_pos: vec![vec![0, 2, 3, 4]],
        ..Default::default()
    });
    bridge.apply_async_result();
    assert_eq!(bridge.live_preedit(), "含めk治");
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "含めk"
    );
    assert_eq!(bridge.composing.hiragana(), "ふくめ");
    assert_eq!(bridge.composing.pending_romaji(), "k");
    mock.set_result("ふくめ", &["含め", "服め"]);
    bridge.apply_async_result();
    assert_eq!(bridge.live_preedit(), "含めk");
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["含めk"]);
}

#[test]
fn backspace_again_takes_the_reading_starts_the_engine_gave() {
    // 東京都庁 placed by the engine; what is left keeps its starts
    // though the shorter reading's list has none
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "toukyoutotyou");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: vec!["東京都庁".into()],
        reading_pos: vec![vec![0, 2, 5, 6]],
        ..Default::default()
    });
    bridge.apply_async_result();
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "東京都"
    );
    mock.set_result("とうきょうと", &["東京と"]);
    bridge.apply_async_result();
    assert_eq!(bridge.live_preedit(), "東京都");
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東京");
    assert_eq!(preedit(&bridge.handle_backspace().actions).unwrap(), "東");
    assert_eq!(bridge.composing.hiragana(), "とう");
}

#[test]
fn backspace_deletes_a_kana_not_converted_yet_or_romaji_as_typed() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.live_conversion = true;
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    bridge.apply_async_result();
    type_str(&mut bridge, "amek");
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "傘のあめ"
    );
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).unwrap(),
        "傘のあ"
    );
}

#[test]
fn enter_does_not_commit_a_deleted_or_edited_reading() {
    // Backspace deleting the last kana typed: the conversion of the
    // longer reading is not committed
    let (mut bridge, _) = converted_kasano();
    bridge.cfg.backspace_deletes_shown = false;
    bridge.handle_backspace();
    assert_eq!(
        bridge.handle_enter().actions,
        vec![Action::Commit {
            text: "かさ".into()
        }]
    );
    // deleting the last char shown: what is left of it is
    let (mut bridge, _) = converted_kasano();
    bridge.handle_backspace();
    assert_eq!(
        bridge.handle_enter().actions,
        vec![Action::Commit { text: "傘".into() }]
    );
    let (mut bridge, _) = converted_kasano();
    bridge.handle_home_end(false);
    type_str(&mut bridge, "a");
    bridge.handle_home_end(true);
    assert_eq!(
        bridge.handle_enter().actions,
        vec![Action::Commit {
            text: "あかさの".into()
        }]
    );
}

#[test]
fn space_uses_the_whole_reading_while_its_conversion_is_pending() {
    let (mut bridge, _) = converted_kasano();
    type_str(&mut bridge, "ame");
    let out = bridge.handle_space();
    assert_eq!(
        out.actions[0],
        Action::UpdatePreedit {
            text: "傘のあめ".into()
        }
    );
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "傘のあめ".into()
        }
    );
}

#[test]
fn conversion_keeps_pending_romaji_with_or_without_a_result() {
    for answer in [false, true] {
        let (mut bridge, mock) = converted_kasano();
        type_str(&mut bridge, "k");
        assert_eq!(
            bridge.handle_space().actions[0],
            Action::UpdatePreedit {
                text: "傘のk".into()
            }
        );
        if answer {
            mock.set_result("かさの", &["傘の", "笠の"]);
            assert_eq!(
                bridge.apply_async_result()[0],
                Action::UpdatePreedit {
                    text: "傘のk".into()
                }
            );
            assert_eq!(
                bridge.handle_down().actions[0],
                Action::UpdatePreedit {
                    text: "笠のk".into()
                }
            );
        }
        let text = if answer { "笠のk" } else { "傘のk" };
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit { text: text.into() }
        );
    }
}

#[test]
fn provisional_pending_romaji_is_not_appended_twice() {
    let (mut bridge, _) = bridge_with_mock();
    type_str(&mut bridge, "nekok");
    assert_eq!(
        bridge.handle_space().actions[0],
        Action::UpdatePreedit {
            text: "ねこk".into()
        }
    );
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "ねこk".into()
        }
    );
}

#[test]
fn late_results_do_not_cross_composition_boundaries() {
    for end in 0..5 {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kasano");
        mock.set_result("かさの", &["笠の"]);
        let old = mock.state.next.borrow_mut().take().unwrap();
        match end {
            0 => {
                bridge.discard();
            }
            1 => {
                bridge.handle_enter();
            }
            2 => {
                bridge.handle_escape();
            }
            3 => {
                bridge.reset_preserving_client(&crate::config::FrontendCfg::default());
            }
            _ => {
                for _ in 0..3 {
                    bridge.handle_backspace();
                }
                assert!(!bridge.is_active());
            }
        }
        bridge.set_app_context(Some("別の文脈"));
        type_str(&mut bridge, "kasano");
        assert!(mock.state.dispatched.borrow().last().unwrap().0 > old.req_id);
        *mock.state.next.borrow_mut() = Some(old);
        assert!(bridge.apply_async_result().is_empty());
        mock.answer_last(&["傘の"]);
        assert_eq!(
            bridge.apply_async_result(),
            vec![Action::UpdatePreedit {
                text: "傘の".into()
            }]
        );
    }
}

#[test]
fn date_time_candidates_work_without_a_model_including_short_readings() {
    for (typed, reading) in [
        ("kinou", "きのう"),
        ("kyou", "きょう"),
        ("ima", "いま"),
        ("raishuu", "らいしゅう"),
        ("suiyoubi", "すいようび"),
        ("1jikango", "1じかんご"),
        ("30punmae", "30ぷんまえ"),
        ("raishuunosuiyoubi", "らいしゅうのすいようび"),
    ] {
        let mut bridge = EngineBridge::new(None);
        type_str(&mut bridge, typed);
        let out = bridge.handle_space();
        let expected = crate::date_time::candidates(reading, &bridge.comp.date_time_now.unwrap());
        let list = match &out.actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!("no candidate list"),
        };
        assert!(list.contains(&reading.to_string()), "{typed}");
        assert!(expected.iter().all(|c| list.contains(c)), "{typed}");
        let chosen = expected[0].clone();
        bridge.handle_select(list.iter().position(|c| *c == chosen).unwrap());
        assert_eq!(
            bridge.handle_enter().actions[0],
            Action::Commit { text: chosen }
        );
    }
}

#[test]
fn local_date_choice_survives_a_delayed_model_and_segment_result() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kyou");
    let out = bridge.handle_space();
    let now = bridge.comp.date_time_now.unwrap();
    let date = crate::date_time::candidates("きょう", &now)[0].clone();
    let list = match &out.actions[1] {
        Action::ShowCandidates { list, .. } => list,
        _ => panic!(),
    };
    bridge.handle_select(list.iter().position(|c| *c == date).unwrap());
    mock.answer_last(&["今日", "京"]);
    let actions = bridge.apply_async_result();
    assert_eq!(preedit(&actions), Some(date.clone()));
    assert_eq!(bridge.comp.date_time_now, Some(now));
    let list = match &actions[1] {
        Action::ShowCandidates { list, selected } => {
            assert_eq!(list[*selected], date);
            list
        }
        _ => panic!(),
    };
    assert!(list.contains(&"今日".to_string()));
    assert_eq!(
        bridge.comp.list_starts[&date],
        vec![0; date.chars().count()]
    );
    let actions = bridge.handle_move_cursor(1).actions;
    assert_eq!(segments_shown(&actions).0, vec![date.clone()]);
    mock.answer_last(&["今日", "京"]);
    let actions = bridge.apply_async_result();
    assert_eq!(segments_shown(&actions).0, vec![date.clone()]);
    let (list, selected, _) = segment_list(&actions);
    assert_eq!(list[selected], date);
    assert!(crate::date_time::candidates("きょう", &now)
        .iter()
        .all(|c| list.contains(c)));
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit { text: date }
    );
}

#[test]
fn ngram_scores_reorder_the_engines_candidates_for_the_reranker_too() {
    let path = std::env::temp_dir().join(format!(
        "setsuna-ngram-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    let text = "博士が来る。猫が白い。白紙に書く。".repeat(20);
    std::fs::write(&path, crate::char_ngram::tests::table(&text)).unwrap();
    let (mut bridge, mock, _rr) = bridge_with_reranker();
    bridge.set_ngram(&crate::config::NgramCfg {
        table: Some(path.to_string_lossy().into_owned()),
        ..Default::default()
    });
    type_str(&mut bridge, "hakusi");
    bridge.handle_space();
    // 博紙, the model's top, joins two chars no text has together
    answer_scored(&mock, &["博紙", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
    let actions = bridge.apply_async_result();
    let list = match &actions[1] {
        Action::ShowCandidates { list, .. } => list.clone(),
        _ => panic!(),
    };
    assert_eq!(list[..2], ["博士", "白紙"]);
    let pos = |c: &str| list.iter().position(|x| x == c).unwrap();
    assert!(pos("博紙") > pos("白紙"), "{list:?}");
    let job = bridge.comp.rerank.as_ref().unwrap();
    assert_eq!(job.raw, ["博士", "白紙", "博紙"]);
    assert_eq!(job.scores[..2], [-1.2, -1.3]);
    assert!(job.scores[2] < -1.3);
    // off: the engine's order
    bridge.handle_escape();
    bridge.handle_escape();
    bridge.set_ngram(&crate::config::NgramCfg::default());
    type_str(&mut bridge, "hakusi");
    bridge.handle_space();
    answer_scored(&mock, &["博紙", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
    bridge.apply_async_result();
    assert_eq!(
        bridge.comp.rerank.as_ref().unwrap().raw,
        ["博紙", "博士", "白紙"]
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn candidates_that_leave_reading_out_go_last() {
    let (mut bridge, mock, _rr) = bridge_with_reranker();
    type_str(&mut bridge, "hakusi");
    bridge.handle_space();
    // はし leaves く out
    answer_scored(&mock, &["はし", "博士", "白紙"], &[-1.0, -1.2, -1.3]);
    bridge.apply_async_result();
    let job = bridge.comp.rerank.as_ref().unwrap();
    assert_eq!(job.raw, ["博士", "白紙", "はし"]);
    assert_eq!(job.scores, [-1.2, -1.3, -1.3]);
    // and the n-gram does not lift one: し has no pair to lose points on
    let path = std::env::temp_dir().join(format!(
        "setsuna-ngram-spell-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    let text = "博士が来る。猫が白い。白紙に書く。".repeat(20);
    std::fs::write(&path, crate::char_ngram::tests::table(&text)).unwrap();
    bridge.handle_escape();
    bridge.handle_escape();
    bridge.set_ngram(&crate::config::NgramCfg {
        table: Some(path.to_string_lossy().into_owned()),
        ..Default::default()
    });
    type_str(&mut bridge, "hakusi");
    bridge.handle_space();
    answer_scored(&mock, &["博紙", "し", "博士"], &[-1.0, -1.1, -1.2]);
    bridge.apply_async_result();
    let job = bridge.comp.rerank.as_ref().unwrap();
    assert_eq!(job.raw, ["博士", "博紙", "し"]);
    assert!(job.scores[2] <= job.scores[1], "{:?}", job.scores);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn time_snapshot_and_selection_survive_reranking() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "ima");
    bridge.handle_space();
    answer_scored(&mock, &["今", "居間"], &[-1.0, -1.5]);
    let actions = bridge.apply_async_result();
    let now = bridge.comp.date_time_now.unwrap();
    let time = crate::date_time::candidates("いま", &now)[1].clone();
    let list = match &actions[1] {
        Action::ShowCandidates { list, .. } => list,
        _ => panic!(),
    };
    bridge.handle_select(list.iter().position(|c| *c == time).unwrap());
    rr.answer(&[-30.0, -20.0]);
    let actions = bridge.apply_async_result();
    assert_eq!(bridge.comp.date_time_now, Some(now));
    assert_eq!(preedit(&actions), Some(time.clone()));
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit { text: time }
    );
}

#[test]
fn explicit_conversion_refreshes_old_dates_without_reapplying_character_rules() {
    let mock = MockClient::default();
    let frontend = crate::config::FrontendCfg {
        replacements: vec![crate::config::Replacement {
            from: "今日".into(),
            to: "今日今日".into(),
            enabled: true,
        }],
        ..Default::default()
    };
    let mut bridge = EngineBridge::with_frontend(Some(Box::new(mock.clone())), &frontend);
    type_str(&mut bridge, "kyou");
    let old = Local::now() - chrono::Duration::days(1);
    bridge.comp.date_time_now = Some(old);
    mock.answer_last(&["今日", "京"]);
    bridge.apply_async_result();
    let old_date = crate::date_time::candidates("きょう", &old)[0].clone();
    let out = bridge.handle_space();
    let now = bridge.comp.date_time_now.unwrap();
    let new_date = crate::date_time::candidates("きょう", &now)[0].clone();
    assert_ne!(old_date, new_date);
    let list = match &out.actions[1] {
        Action::ShowCandidates { list, .. } => list,
        _ => panic!(),
    };
    assert!(!list.contains(&old_date));
    assert!(list.contains(&new_date));
    assert_eq!(list[0], "今日今日");
    bridge.handle_escape();
    let out = bridge.handle_space();
    let list = match &out.actions[1] {
        Action::ShowCandidates { list, .. } => list,
        _ => panic!(),
    };
    assert!(!list.contains(&old_date));
    assert_eq!(list[0], "今日今日");
}

#[test]
fn local_date_reconversion_keeps_the_normal_kana_candidates() {
    let mut bridge = EngineBridge::new(None);
    type_str(&mut bridge, "ima");
    for _ in 0..3 {
        let out = bridge.handle_space();
        let list = match &out.actions[1] {
            Action::ShowCandidates { list, .. } => list,
            _ => panic!(),
        };
        assert_eq!(&list[..2], ["いま", "イマ"]);
        assert!(list.len() > 2);
        bridge.handle_escape();
    }
}

#[test]
fn date_time_candidates_can_be_disabled_and_do_not_match_sentence_prefixes() {
    let frontend = crate::config::FrontendCfg {
        date_time_candidates: false,
        ..Default::default()
    };
    let mut bridge = EngineBridge::with_frontend(None, &frontend);
    type_str(&mut bridge, "kyou");
    assert_eq!(
        bridge.handle_space().actions[1],
        Action::ShowCandidates {
            list: v(&["きょう"]),
            selected: 0
        }
    );
    let mut bridge = EngineBridge::new(None);
    type_str(&mut bridge, "kyouha");
    assert_eq!(
        bridge.handle_space().actions[1],
        Action::ShowCandidates {
            list: v(&["きょうは"]),
            selected: 0
        }
    );
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kyouha");
    mock.answer_last(&["今日は"]);
    bridge.apply_async_result();
    assert!(bridge.current_list().iter().all(|c| !c.contains('/')));
}

#[test]
fn enter_commits_top_candidate() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    mock.set_result("か", &["可", "蚊"]);
    bridge.apply_async_result();
    let outcome = bridge.handle_enter();
    assert_eq!(outcome.actions, vec![Action::Commit { text: "可".into() }]);
    assert!(!bridge.is_active());
}

#[test]
fn enter_on_a_reading_typed_again_waits_for_its_conversion() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_enter();
    // the same reading again, its conversion not in yet: Enter waits
    // for it rather than taking the list of the one committed for its
    mock.state
        .answers
        .borrow_mut()
        .insert("かさの".into(), vec!["嵩の".into()]);
    type_str(&mut bridge, "kasano");
    let outcome = bridge.handle_enter();
    assert_eq!(
        outcome.actions,
        vec![Action::Commit {
            text: "嵩の".into()
        }]
    );
}

#[test]
fn a_form_committed_leaves_the_next_conversion_of_its_reading_alone() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    bridge.handle_convert_to(Form::Hiragana);
    bridge.handle_enter();
    type_str(&mut bridge, "kasano");
    bridge.handle_space();
    mock.answer_last(&["傘の", "笠の"]);
    let (list, _) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!(list[0], "傘の");
}

#[test]
fn deleting_the_composition_drops_its_rerank() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の"], &[-1.0, -1.5]);
    bridge.apply_async_result();
    assert!(bridge.comp.rerank.is_some());
    bridge.cfg.backspace_deletes_shown = false;
    for _ in 0..3 {
        bridge.handle_backspace();
    }
    assert!(!bridge.is_active());
    assert!(bridge.comp.rerank.is_none());
    assert_eq!(*rr.state.cancels.borrow(), 1);
}

#[test]
fn escape_discards_composition() {
    let (mut bridge, _mock) = bridge_with_mock();
    type_str(&mut bridge, "ka");
    let outcome = bridge.handle_escape();
    assert_eq!(outcome.actions, vec![Action::EndComposition]);
    assert!(!bridge.is_active());
}

#[test]
fn punctuation_rules_apply_to_the_candidate_list() {
    let mock = MockClient::default();
    let frontend = crate::config::FrontendCfg {
        punctuation: crate::config::PunctuationStyle::CommaPeriod,
        ..crate::config::FrontendCfg::default()
    };
    let mut bridge = EngineBridge::with_frontend(Some(Box::new(mock.clone())), &frontend);
    type_str(&mut bridge, "hai");
    bridge.handle_punctuation("、");
    mock.set_result("はい、", &["はい、", "肺、"]);
    bridge.apply_async_result();
    let outcome = bridge.handle_space();
    assert_eq!(
        outcome.actions[1],
        Action::ShowCandidates {
            list: vec!["はい，".into(), "肺，".into(), "ハイ，".into()],
            selected: 0,
        }
    );
}

#[test]
fn a_picked_candidate_is_learned_and_ranked_first_next_time() {
    let dir = std::env::temp_dir().join(format!("setsuna-bridge-learn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let learning = || {
        Some(Learning::at(
            dir.join("learning.jsonl"),
            &crate::config::LearningCfg::default(),
        ))
    };
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_learning(learning());
    type_str(&mut bridge, "kami");
    mock.set_result("かみ", &["神", "紙", "髪"]);
    bridge.apply_async_result();
    bridge.handle_space();
    bridge.handle_down();
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit { text: "紙".into() }
    );

    // Another bridge (another application) sees the choice.
    let (mut other, mock) = bridge_with_mock();
    other.set_learning(learning());
    type_str(&mut other, "kami");
    mock.set_result("かみ", &["神", "紙", "髪"]);
    assert_eq!(
        other.apply_async_result(),
        vec![Action::UpdatePreedit { text: "紙".into() }]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn committing_the_top_candidate_learns_nothing() {
    let dir = std::env::temp_dir().join(format!("setsuna-bridge-top-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("learning.jsonl");
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_learning(Some(Learning::at(
        path.clone(),
        &crate::config::LearningCfg::default(),
    )));
    type_str(&mut bridge, "kami");
    mock.set_result("かみ", &["神", "紙"]);
    bridge.apply_async_result();
    bridge.handle_space();
    bridge.handle_enter();
    assert!(!path.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn committed_text_is_the_next_context_until_forgotten() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    bridge.apply_async_result();
    bridge.handle_enter();
    assert_eq!(bridge.context(), "傘の");
    type_str(&mut bridge, "sitade");
    let last = mock.state.dispatched.borrow().last().cloned().unwrap();
    assert_eq!((last.1.as_str(), last.2.as_str()), ("傘の", "したで"));
    bridge.handle_escape();
    bridge.forget_context();
    type_str(&mut bridge, "sitade");
    let last = mock.state.dispatched.borrow().last().cloned().unwrap();
    assert_eq!(last.1, "");
}

#[test]
fn the_apps_text_replaces_the_kept_context() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.handle_punctuation("。");
    bridge.handle_enter();
    assert_eq!(bridge.context(), "。");
    // line breaks and embedded objects are dropped, the tail kept
    let long = format!("{}\r\n三相\u{FFFC}交流の", "あ".repeat(100));
    bridge.set_app_context(Some(&long));
    assert_eq!(bridge.context().chars().count(), CONTEXT_WINDOW);
    assert!(bridge.context().ends_with("あ三相交流の"));
    type_str(&mut bridge, "teiden");
    let last = mock.state.dispatched.borrow().last().cloned().unwrap();
    assert!(last.1.ends_with("三相交流の"));
    // the next read replaces it again
    bridge.handle_escape();
    bridge.set_app_context(Some("ビー"));
    assert_eq!(bridge.context(), "ビー");
}

#[test]
fn unreadable_app_text_keeps_the_kept_context() {
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.handle_punctuation("。");
    bridge.handle_enter();
    bridge.set_app_context(None);
    assert_eq!(bridge.context(), "。");
    bridge.set_app_context(Some(""));
    assert_eq!(bridge.context(), "。");
    bridge.set_app_context(Some("\r\n"));
    assert_eq!(bridge.context(), "。");
}

/// きしゃのきしゃ converted, Space pressed: the whole-sentence list.
fn converting_kisha() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kishanokisha");
    let pos: &[u32] = &[0, 2, 3, 4, 6];
    mock.set_aligned(
        "きしゃのきしゃ",
        &["記者の記者", "汽車の記者", "記者の汽車"],
        &[pos, pos, pos],
    );
    bridge.apply_async_result();
    bridge.handle_space();
    (bridge, mock)
}

fn segments_shown(actions: &[Action]) -> (Vec<String>, usize) {
    actions
        .iter()
        .find_map(|a| match a {
            Action::UpdateSegments { segments, focus } => Some((segments.clone(), *focus)),
            _ => None,
        })
        .expect("UpdateSegments")
}

fn segment_list(actions: &[Action]) -> (Vec<String>, usize, usize) {
    actions
        .iter()
        .find_map(|a| match a {
            Action::ShowSegmentCandidates { list, selected, at } => {
                Some((list.clone(), *selected, *at))
            }
            _ => None,
        })
        .expect("ShowSegmentCandidates")
}

fn v(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

/// きしゃのきしゃ typed and converted live (記者の記者 shown), nothing pressed.
fn live_kisha() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kishanokisha");
    let pos: &[u32] = &[0, 2, 3, 4, 6];
    mock.set_aligned(
        "きしゃのきしゃ",
        &["記者の記者", "汽車の記者", "記者の汽車"],
        &[pos, pos, pos],
    );
    bridge.apply_async_result();
    (bridge, mock)
}

#[test]
fn space_can_open_the_last_segments_list_only() {
    let (mut bridge, _mock) = live_kisha();
    bridge.cfg.space_last_segment = true;
    let out = bridge.handle_space();
    // the sentence converted, the last segment the one being edited
    assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 1));
    let (list, selected, at) = segment_list(&out.actions);
    assert_eq!((list[..2].to_vec(), selected), (v(&["記者", "汽車"]), 0));
    assert_eq!(at, "記者の".encode_utf16().count());
    // Space again: that segment's next candidate, the one before as it was
    let out = bridge.handle_space();
    assert_eq!(segments_shown(&out.actions), (v(&["記者の", "汽車"]), 1));
    // ← to the segment before
    let out = bridge.handle_move_cursor(-1);
    assert_eq!(segments_shown(&out.actions).1, 0);
    // off (the default): the whole sentence's list
    let (mut bridge, _mock) = live_kisha();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "記者の記者");
}

#[test]
fn space_before_the_list_came_opens_the_last_segment_once_it_comes() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.space_last_segment = true;
    type_str(&mut bridge, "kishanokisha");
    bridge.handle_space();
    let pos: &[u32] = &[0, 2, 3, 4, 6];
    mock.set_aligned(
        "きしゃのきしゃ",
        &["記者の記者", "汽車の記者", "記者の汽車"],
        &[pos, pos, pos],
    );
    let out = bridge.apply_async_result();
    assert_eq!(segments_shown(&out), (v(&["記者の", "記者"]), 1), "{out:?}");
}

#[test]
fn left_on_a_live_conversion_edits_its_segments_instead_of_bringing_the_kana_back() {
    let (mut bridge, _mock) = live_kisha();
    let out = bridge.handle_move_cursor(-1);
    assert!(out.consumed);
    // the conversion stays, its last segment to edit
    assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 1));
    assert!(bridge.is_converting());
    // → at the end of it: nothing to go to, the conversion as it is
    let (mut bridge, _mock) = live_kisha();
    let out = bridge.handle_move_cursor(1);
    assert!(out.consumed);
    assert!(!bridge.is_converting());
    // with the kana shown (Escape), ← moves the caret in them
    let (mut bridge, _mock) = live_kisha();
    bridge.handle_escape();
    let out = bridge.handle_move_cursor(-1);
    assert!(!bridge.is_converting());
    assert!(
        matches!(out.actions.last(), Some(Action::UpdatePreedit { text }) if text == "きしゃのきしゃ"),
        "{:?}",
        out.actions
    );
    assert_eq!(bridge.caret(), Some(6));
}

/// `ashitae` typed and its conversion in, the chars of each candidate at
/// あした and え, with these scores.
fn live_ashitae(scores: &[f32]) -> (EngineBridge, Vec<Action>) {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "ashitae");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: v(&["明日絵", "明日え", "明日エ"]),
        reading_pos: vec![vec![0, 0, 3]; 3],
        scores: scores.to_vec(),
        ..Default::default()
    });
    let actions = bridge.apply_async_result();
    (bridge, actions)
}

#[test]
fn live_conversion_holds_a_noun_end_back_for_a_kana_one_close_to_it() {
    // 明日え within the penalty of 明日絵: shown, and what Enter commits
    let close = [-0.87, -1.23, -2.2];
    let (mut bridge, actions) = live_ashitae(&close);
    assert_eq!(preedit(&actions).as_deref(), Some("明日え"));
    let out = bridge.handle_enter();
    assert!(
        out.actions
            .iter()
            .any(|a| matches!(a, Action::Commit { text } if text == "明日え")),
        "{:?}",
        out.actions
    );
    // Space converts: the model's first
    let (mut bridge, _) = live_ashitae(&close);
    let out = bridge.handle_space();
    assert_eq!(preedit(&out.actions).as_deref(), Some("明日絵"));
    // ← edits the conversion shown
    let (mut bridge, _) = live_ashitae(&close);
    let out = bridge.handle_move_cursor(-1);
    assert_eq!(segments_shown(&out.actions).0.concat(), "明日え");
    // the model surer of its noun than that: it stays
    let (_, actions) = live_ashitae(&[-0.1, -4.0, -5.0]);
    assert_eq!(preedit(&actions).as_deref(), Some("明日絵"));
    // off
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.noun_end_penalty = 0.0;
    type_str(&mut bridge, "ashitae");
    let (req_id, _, reading) = mock.state.dispatched.borrow().last().cloned().unwrap();
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id,
        reading,
        candidates: v(&["明日絵", "明日え"]),
        reading_pos: vec![vec![0, 0, 3]; 2],
        scores: close[..2].to_vec(),
        ..Default::default()
    });
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("明日絵")
    );
}

#[test]
fn discard_drops_the_composition_and_a_late_result() {
    let (mut bridge, mock) = converting_kisha();
    assert_eq!(
        bridge.discard(),
        vec![Action::HideCandidates, Action::EndComposition]
    );
    assert!(!bridge.is_active());
    assert_eq!(bridge.context(), "");
    mock.set_result("きしゃのきしゃ", &["記者の記者"]);
    assert!(bridge.apply_async_result().is_empty());
    assert!(bridge.discard().is_empty());
    let out = bridge.handle_char('a');
    assert_eq!(
        out.actions,
        vec![Action::UpdatePreedit { text: "あ".into() }]
    );
}

#[test]
fn select_picks_a_candidate_of_the_open_list() {
    let (mut bridge, _mock) = converting_kisha();
    let out = bridge.handle_select(4);
    assert!(out.consumed);
    assert!(out.actions.contains(&Action::UpdatePreedit {
        text: "記者の汽車".into()
    }));
    // in segment editing it picks the focused segment's candidate (the
    // rest keep the whole candidate picked before, 記者の汽車)
    bridge.handle_move_cursor(1);
    let out = bridge.handle_select(1);
    assert_eq!(segments_shown(&out.actions), (v(&["汽車の", "汽車"]), 0));
    // nothing to pick without a list
    let (mut idle, _mock) = bridge_with_mock();
    assert!(!idle.handle_select(0).consumed);
}

#[test]
fn a_suggestion_clicked_is_the_whole_sentence_with_space_on_the_last_segment() {
    let (mut bridge, _mock) = live_kisha();
    bridge.cfg.live_conversion = false;
    bridge.cfg.space_last_segment = true;
    let at = bridge
        .suggestions()
        .iter()
        .position(|c| c == "汽車の記者")
        .unwrap();
    let out = bridge.handle_select(at);
    assert!(out.consumed);
    assert_eq!(
        commits(&bridge.handle_enter().actions),
        vec!["汽車の記者"],
        "{:?}",
        out.actions
    );
}

#[test]
fn arrows_edit_the_conversion_segment_by_segment() {
    let (mut bridge, mock) = converting_kisha();
    // → enters segment editing on the first segment
    let out = bridge.handle_move_cursor(1);
    assert!(out.consumed);
    assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 0));
    let (list, selected, at) = segment_list(&out.actions);
    assert_eq!(list, v(&["記者の", "汽車の", "きしゃの", "キシャノ"]));
    assert_eq!((selected, at), (0, 0));
    // the focused segment's own reading is converted
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きしゃの");
    // → again focuses the second; its window sits after 記者の
    let out = bridge.handle_move_cursor(1);
    assert_eq!(segments_shown(&out.actions), (v(&["記者の", "記者"]), 1));
    let (list, _, at) = segment_list(&out.actions);
    assert_eq!(list, v(&["記者", "汽車", "きしゃ", "キシャ"]));
    assert_eq!(at, 3);
    // Segment jobs are serialized; finishing the first starts the
    // now-focused second, with the first as its left context.
    assert_eq!(mock.answer_last(&["記者の"]).1, "きしゃの");
    bridge.apply_async_result();
    let (context, reading) = mock.answer_last(&["記者", "帰社"]);
    assert_eq!((context.as_str(), reading.as_str()), ("記者の", "きしゃ"));
    let out = bridge.apply_async_result();
    let (list, _, _) = segment_list(&out);
    assert_eq!(list, v(&["記者", "汽車", "きしゃ", "キシャ", "帰社"]));
    // ↓ picks 汽車, Enter commits both segments
    bridge.handle_down();
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "記者の汽車".into()
        }
    );
}

#[test]
fn segment_editing_cuts_by_the_engines_analysis() {
    let segments_of = |words: Option<Vec<crate::engine_client::Word>>| {
        let (mut bridge, mock) = bridge_with_mock();
        if let Some(w) = words {
            mock.state
                .analyses
                .borrow_mut()
                .insert("申し出る".into(), w);
        }
        type_str(&mut bridge, "moushideru");
        mock.set_aligned("もうしでる", &["申し出る"], &[&[0, 2, 3, 4]]);
        bridge.apply_async_result();
        bridge.handle_space();
        segments_shown(&bridge.handle_move_cursor(1).actions).0
    };
    // the char alignment cuts the compound verb
    assert_eq!(segments_of(None), v(&["申し", "出る"]));
    let word = crate::engine_client::Word {
        surface: "申し出る".into(),
        start: 0,
        end: 4,
        reading: Some("もうしでる".into()),
        pos: "動詞".into(),
        pos2: "一般".into(),
        ..Default::default()
    };
    assert_eq!(segments_of(Some(vec![word])), v(&["申し出る"]));
}

#[test]
fn rapid_segment_moves_do_not_queue_overwritable_jobs() {
    let (mut bridge, mock) = converting_kisha();
    bridge.handle_move_cursor(1);
    let count = mock.state.dispatched.borrow().len();
    for _ in 0..3 {
        bridge.handle_move_cursor(1);
        bridge.handle_move_cursor(-1);
    }
    bridge.handle_move_cursor(1);
    assert_eq!(mock.state.dispatched.borrow().len(), count);
    assert_eq!(bridge.comp.seg_requests.len(), 1);
    // Finishing the first job dispatches the currently focused segment.
    assert_eq!(mock.answer_last(&["記者の"]).1, "きしゃの");
    bridge.apply_async_result();
    assert_eq!(mock.answer_last(&["記者"]).1, "きしゃ");
    bridge.apply_async_result();
    assert!(bridge.comp.seg_requests.is_empty());
}

#[test]
fn a_deferred_segment_continuation_is_dispatched_after_the_current_job() {
    let (mut bridge, mock) = converting_kisha();
    bridge.handle_move_cursor(1);
    mock.answer_last(&["記者の"]);
    bridge.apply_async_result();
    bridge.handle_move_cursor(1); // second segment's conversion is running
    bridge.handle_move_cursor(-1);
    bridge.handle_home_end(true);
    bridge.handle_down(); // defer the first segment's continuation
    assert!(mock.state.continued.borrow().is_empty());
    assert_eq!(mock.answer_last(&["記者"]).1, "きしゃ");
    bridge.apply_async_result();
    assert_eq!(mock.state.continued.borrow().last().unwrap().1, 1);
    assert_eq!(mock.answer_last(&["記者の", "貴社の"]).1, "きしゃの");
    let (list, selected, _) = segment_list(&bridge.apply_async_result());
    assert_eq!(list[selected], "貴社の");
}

#[test]
fn segment_editing_and_forms_keep_the_pending_tail_once() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kishanokishak");
    mock.set_aligned("きしゃのきしゃ", &["記者の記者"], &[&[0, 2, 3, 4, 6]]);
    bridge.apply_async_result();
    bridge.handle_space();
    assert_eq!(
        segments_shown(&bridge.handle_move_cursor(1).actions).0,
        v(&["記者の", "記者k"])
    );
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "記者の記者k".into()
        }
    );
    let (mut bridge, _) = converted_kasano();
    type_str(&mut bridge, "k");
    bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "kasanok".into()
        }
    );
}

#[test]
fn past_a_segments_last_candidate_come_more_not_the_first() {
    let (mut bridge, mock) = converting_kisha();
    bridge.handle_move_cursor(1);
    mock.answer_last(&["記者の", "帰社の"]);
    let (list, _, _) = segment_list(&bridge.apply_async_result());
    assert_eq!(list.len(), 5);
    for _ in 0..4 {
        bridge.handle_down();
    }
    // ↓ on the last asks for more of the segment's conversion and waits there
    let out = bridge.handle_down();
    assert_eq!(segment_list(&out.actions).1, 4);
    let (req, more) = *mock.state.continued.borrow().last().unwrap();
    assert_eq!(more, 1);
    let (context, reading) = mock.answer_last(&["記者の", "帰社の", "貴社の"]);
    assert_eq!((context.as_str(), reading.as_str()), ("", "きしゃの"));
    assert_eq!(mock.state.dispatched.borrow().last().unwrap().0, req);
    // the first new candidate is selected when it arrives
    let (list, selected, _) = segment_list(&bridge.apply_async_result());
    assert_eq!((list[5].as_str(), selected), ("貴社の", 5));
    // a continuation with nothing new: round to the first, as before
    bridge.handle_down();
    assert_eq!(mock.state.continued.borrow().last().unwrap().1, 2);
    mock.answer_last(&["記者の"]);
    assert_eq!(segment_list(&bridge.apply_async_result()).1, 0);
    // and from then on without asking again
    for _ in 0..5 {
        bridge.handle_down();
    }
    assert_eq!(segment_list(&bridge.handle_down().actions).1, 0);
    assert_eq!(mock.state.continued.borrow().len(), 2);
}

#[test]
fn a_whole_sentence_list_still_goes_round() {
    let (mut bridge, mock) = converting_kisha();
    for _ in 0..5 {
        bridge.handle_down();
    }
    assert!(mock.state.continued.borrow().is_empty());
    assert!(bridge
        .handle_down()
        .actions
        .contains(&Action::UpdatePreedit {
            text: "汽車の記者".into()
        }));
}

#[test]
fn rerank_order_adds_the_weighted_lm_score() {
    // 0: -1 - 0.5*30 = -16, 1: -2 - 0.5*20 = -12, 2: -3 - 0.5*18 = -12
    assert_eq!(
        rerank_order(&[-1.0, -2.0, -3.0], &[-30.0, -20.0, -18.0], 0.5, &[true; 3]),
        [1, 2, 0]
    );
    assert_eq!(
        rerank_order(&[-1.0, -2.0], &[-30.0, -20.0], 0.0, &[true; 2]),
        [0, 1]
    );
    // one that does not spell the reading stays below one that does
    assert_eq!(
        rerank_order(&[-1.0, -2.0], &[-30.0, -10.0], 0.5, &[true, false]),
        [0, 1]
    );
}

#[test]
fn a_typing_pause_reranks_the_preedit_then_more_candidates() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
    let t0 = Instant::now();
    assert_eq!(
        bridge.apply_async_result(),
        vec![Action::UpdatePreedit {
            text: "傘の".into()
        }]
    );
    // the first 2, due once typing has paused
    let (cands, due) = rr.last();
    assert_eq!(cands, v(&["傘の", "笠の"]));
    assert!(due >= t0 + Duration::from_millis(100));
    // the LM prefers 笠の: -1.5 - 0.5*20 > -1.0 - 0.5*30
    rr.answer(&[-30.0, -20.0]);
    assert_eq!(
        bridge.apply_async_result(),
        vec![Action::UpdatePreedit {
            text: "笠の".into()
        }]
    );
    // then all three, at once
    let (cands, due) = rr.last();
    assert_eq!(cands, v(&["傘の", "笠の", "嵩の"]));
    assert!(due <= Instant::now());
    // Enter commits the reranked top
    assert_eq!(
        bridge.handle_enter().actions,
        vec![Action::Commit {
            text: "笠の".into()
        }]
    );
}

#[test]
fn the_lexicon_scores_the_candidates_and_drops_the_near_misses() {
    let table = "続投\tぞくとう\t15.0\tnoun\n続\tぞく\t13.3\tpref\n頭\tとう\t11.0\tnoun\n\
                 続騰\tぞくとう\t17.0\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-lexicon-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "zokutou");
    // the engine liked 続頭 and a kanji no word reads (続当) best
    answer_scored(
        &mock,
        &["続頭", "続当", "続投", "続騰"],
        &[-0.5, -0.6, -1.0, -1.5],
    );
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    // 続投 15.0 and 続騰 17.0 by words, 続頭 24.3 by a prefix and a noun:
    // the engine's top 続頭 stays first, 続投 comes next; 続当 (30 for
    // 当) is more than MARGIN above 15.0 and goes
    assert_eq!(list[0], "続頭");
    assert_eq!(list[1], "続投");
    assert!(!list.contains(&"続当".to_string()), "{list:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_live_preview_goes_without_the_lexicons_costs_until_the_idle_makes_the_list() {
    let table = "続投\tぞくとう\t15.0\tnoun\n続\tぞく\t13.3\tpref\n頭\tとう\t11.0\tnoun\n\
                 続騰\tぞくとう\t17.0\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-live-lexicon-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_live_conversion(true);
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "zokutou");
    answer_scored(
        &mock,
        &["続頭", "続当", "続投", "続騰"],
        &[-0.5, -0.6, -1.0, -1.5],
    );
    let actions = bridge.apply_async_result();
    // the preview: the engine's first, which the lexicon's order keeps first
    assert!(
        matches!(actions.last(), Some(Action::UpdatePreedit { text }) if text == "続頭"),
        "{actions:?}"
    );
    // the list as the engine ordered it (続当, which the lexicon drops, in),
    // the frontend's timer due at once
    let Phase::Composing { candidates } = &bridge.phase else {
        panic!("not composing")
    };
    assert!(candidates.contains(&"続当".to_string()), "{candidates:?}");
    assert_eq!(bridge.deferred_for(), Some(Duration::ZERO));
    // the idle: the whole list, the preview as it was
    assert!(bridge.tick().is_empty());
    assert_eq!(bridge.deferred_for(), None);
    let Phase::Composing { candidates } = &bridge.phase else {
        panic!("not composing")
    };
    assert_eq!(candidates[..2], ["続頭", "続投"]);
    assert!(!candidates.contains(&"続当".to_string()), "{candidates:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_lexicon_conversion_learned_first_shows_at_once() {
    // the learned sentence goes into the list itself: the preview need not
    // wait for the lexicon's conversions at the idle
    let table =
        "大島村\tおおしまむら\t10.0\tnoun\n大島\tおおしま\t8.0\tnoun\n村\tむら\t6.0\tnoun\n";
    let dir = std::env::temp_dir().join(format!(
        "setsuna-bridge-live-learned-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("x.lexicon");
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let cfg = crate::config::LearningCfg::default();
    let mut learning = Learning::at(dir.join("learning.jsonl"), &cfg);
    learning
        .record("おおしまむら", "大島村", "おおしま村")
        .unwrap();
    let convert = |learning: Learning| {
        let (mut bridge, mock) = bridge_with_mock();
        bridge.set_live_conversion(true);
        bridge.set_lexicon(&crate::config::LexiconCfg {
            file: Some(path.to_string_lossy().into_owned()),
        });
        bridge.set_learning(Some(learning));
        type_str(&mut bridge, "oosimamura");
        // the engine gave no 大島村
        answer_scored(&mock, &["おおしま村", "大しま村"], &[-0.5, -1.0]);
        let actions = bridge.apply_async_result();
        assert!(
            matches!(actions.last(), Some(Action::UpdatePreedit { text }) if text == "大島村"),
            "{actions:?}"
        );
        bridge
    };
    // the idle leaves it
    let mut bridge = convert(Learning::at(dir.join("learning.jsonl"), &cfg));
    let actions = bridge.tick();
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::UpdatePreedit { text } if text != "大島村")),
        "{actions:?}"
    );
    // Space before the idle: the conversion on screen first
    let mut bridge = convert(learning);
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "大島村");
    assert!(list.contains(&"おおしま村".to_string()), "{list:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_reading_of_one_word_lists_the_lexicons_words_the_model_left_out() {
    let table = "研修\tけんしゅう\t9.4\tnoun\n検収\tけんしゅう\t14.6\tnoun\n\
                 献酬\tけんしゅう\t20.2\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-words-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "kensyuu");
    answer_scored(&mock, &["研修", "献酬"], &[-0.5, -3.0]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    // after the model's and the kana forms, the word it did not give;
    // 献酬 once, where the model put it
    assert_eq!(list, ["研修", "献酬", "けんしゅう", "ケンシュウ", "検収"]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dictionary_words_are_listed_with_the_mixed_input_model_loaded() {
    // the installer loads the mixed-input model by default: a reading
    // with no English in it still gets the lexicon's words
    let table = "研修\tけんしゅう\t9.4\tnoun\n検収\tけんしゅう\t14.6\tnoun\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-words-mixed-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = mixed_bridge();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "kensyuu");
    answer_scored(&mock, &["研修"], &[-0.5]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert!(list.contains(&"検収".to_string()), "{list:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_reading_of_one_kana_lists_its_one_char_words() {
    let mut table = String::from("亜\tあ\t12.7\tnoun\n亞\tあ\t16.7\tnoun\n");
    // dearer than WORD_MARGIN above 亜, and a kanji the engine gave
    // that costs more than MARGIN above it
    for (w, cost) in [("阿", 23.0), ("吾", 23.5), ("唖", 24.0), ("我", 30.0)] {
        table.push_str(&format!("{w}\tあ\t{cost}\tnoun\n"));
    }
    table.push_str("娃\tあ\t40.0\tnoun\na\tあ\t26.0\tnoun\n");
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-chars-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(&table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "a");
    bridge.handle_space();
    answer_scored(&mock, &["亜", "娃"], &[-0.5, -1.0]);
    let list = bridge
        .apply_async_result()
        .iter()
        .find_map(|a| match a {
            Action::ShowCandidates { list, .. } => Some(list.clone()),
            _ => None,
        })
        .unwrap();
    // 娃 is a word of あ and stays though it costs more than MARGIN over
    // 亜; 亞 is in WORD_MARGIN (`word_candidates`), the other words come
    // after it cheapest first, a Latin letter not
    assert_eq!(
        list,
        ["亜", "あ", "ア", "娃", "亞", "阿", "吾", "唖", "我"],
        "{list:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_homophone_of_a_word_in_the_engines_top_is_listed() {
    let table = "研修\tけんしゅう\t9.4\tnoun\n検収\tけんしゅう\t14.6\tnoun\n\
                 担当\tたんとう\t8.0\tnoun\n者\tしゃ\t5.0\tsuff\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-homophone-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "kensyuutantousya");
    answer_scored(&mock, &["研修担当者", "研修担当社"], &[-0.5, -3.0]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    // 検収担当者 costs 5.2 more than 研修担当者, within ADDED_OVER
    assert_eq!(list[0], "研修担当者");
    assert!(list.contains(&"検収担当者".to_string()), "{list:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_candidate_with_numbers_in_digits_and_kanji_goes_below_one_in_digits() {
    let (mut bridge, mock, _rr) = bridge_with_reranker();
    type_str(&mut bridge, "itinitijuujikan");
    bridge.handle_space();
    answer_scored(
        &mock,
        &["一日10時間", "1日10時間", "一日十時間"],
        &[-1.0, -1.5, -2.0],
    );
    let actions = bridge.apply_async_result();
    let list = match &actions[1] {
        Action::ShowCandidates { list, .. } => list.clone(),
        _ => panic!(),
    };
    assert_eq!(list[..2], ["1日10時間", "一日十時間"]);
    let pos = |c: &str| list.iter().position(|x| x == c).unwrap();
    assert!(pos("一日10時間") > pos("一日十時間"), "{list:?}");
    let job = bridge.comp.rerank.as_ref().unwrap();
    assert_eq!(job.raw[0], "1日10時間");
    assert!(job.scores[2] < -2.0, "{:?}", job.scores);
}

#[test]
fn a_candidate_the_reranker_scores_far_below_the_best_goes() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "傘ノ", "嵩の"], &[-1.0, -1.5, -4.0]);
    bridge.apply_async_result();
    bridge.handle_space();
    // 傘ノ: -1.5 - 0.5*70 is 21 below -1.0 - 0.5*30; 嵩の is not scored
    rr.answer(&[-30.0, -70.0]);
    bridge.apply_async_result();
    rr.answer(&[-30.0, -70.0, -35.0]);
    let (list, _) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!(list, v(&["傘の", "嵩の", "かさの", "カサノ"]));
}

#[test]
fn a_score_that_is_not_a_number_ranks_last_and_drops_nothing() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の"], &[-1.0, -1.5]);
    bridge.apply_async_result();
    bridge.handle_space();
    rr.answer(&[f64::NAN, f64::NAN]);
    let (list, _) = shown(&bridge.apply_async_result()).unwrap();
    assert!(list.starts_with(&v(&["傘の", "笠の"])), "{list:?}");
    assert_eq!(
        rerank_order(
            &[-1.0, -2.0, -3.0],
            &[f64::NAN, -1.0, f64::INFINITY],
            0.5,
            &[true; 3]
        ),
        [1, 0, 2]
    );
}

#[test]
fn reading_starts_fit_only_their_text_and_reading() {
    assert!(starts_fit(&[0, 2, 2], "東京に", 4));
    assert!(!starts_fit(&[0, 2], "東京に", 4));
    assert!(!starts_fit(&[0, 3, 2], "東京に", 4));
    assert!(!starts_fit(&[0, 2, 5], "東京に", 4));
}

#[test]
fn typing_on_drops_the_rerank_and_a_late_result() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の"], &[-1.0, -1.5]);
    bridge.apply_async_result();
    type_str(&mut bridge, "si");
    assert_eq!(*rr.state.cancels.borrow(), 1);
    rr.answer(&[-30.0, -20.0]);
    assert!(bridge.apply_async_result().is_empty());
}

#[test]
fn the_conversion_key_reranks_at_once_and_keeps_a_moved_selection() {
    let (mut bridge, mock, rr) = bridge_with_reranker();
    type_str(&mut bridge, "kasano");
    answer_scored(&mock, &["傘の", "笠の", "嵩の"], &[-1.0, -1.5, -4.0]);
    bridge.apply_async_result();
    bridge.handle_space();
    let (_, due) = rr.last();
    assert!(due <= Instant::now());
    // the user moved to かさの before the result came: it stays selected
    bridge.handle_down();
    bridge.handle_down();
    rr.answer(&[-30.0, -20.0]);
    let out = bridge.apply_async_result();
    assert_eq!(
        out,
        vec![
            Action::UpdatePreedit {
                text: "かさの".into()
            },
            Action::ShowCandidates {
                list: v(&["笠の", "傘の", "かさの", "カサノ", "嵩の"]),
                selected: 2,
            }
        ]
    );
}

#[test]
fn shift_arrows_resize_and_reconvert() {
    let (mut bridge, mock) = converting_kisha();
    bridge.handle_move_cursor(1);
    // Shift+← gives の to the next segment: both start over as hiragana
    let out = bridge.handle_resize(-1);
    assert_eq!(
        segments_shown(&out.actions),
        (v(&["きしゃ", "のきしゃ"]), 0)
    );
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きしゃ");
    mock.answer_last(&["汽車", "記者"]);
    let out = bridge.apply_async_result();
    assert_eq!(segments_shown(&out).0, v(&["汽車", "のきしゃ"]));
    // then the next stale segment, with the new first one as context
    let (context, reading) = mock.answer_last(&["の記者"]);
    assert_eq!((context.as_str(), reading.as_str()), ("汽車", "のきしゃ"));
    let out = bridge.apply_async_result();
    assert_eq!(segments_shown(&out).0, v(&["汽車", "の記者"]));
}

#[test]
fn a_whole_sentence_result_is_ignored_while_editing_segments() {
    let (mut bridge, mock) = converting_kisha();
    bridge.handle_move_cursor(1);
    mock.set_result("きしゃのきしゃ", &["帰社の記者"]);
    assert!(bridge.apply_async_result().is_empty());
    assert_eq!(
        bridge.handle_enter().actions[0],
        Action::Commit {
            text: "記者の記者".into()
        }
    );
}

#[test]
fn a_segment_edit_is_learned_as_the_sentence_choice() {
    let dir = std::env::temp_dir().join(format!("setsuna-bridge-seg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("learning.jsonl");
    let (mut bridge, _mock) = converting_kisha();
    bridge.set_learning(Some(Learning::at(
        path.clone(),
        &crate::config::LearningCfg::default(),
    )));
    bridge.handle_move_cursor(1);
    bridge.handle_move_cursor(1);
    bridge.handle_down();
    bridge.handle_enter();
    let log = std::fs::read_to_string(&path).unwrap();
    assert!(log.contains("\"chosen\":\"記者の汽車\""), "{log}");
    assert!(log.contains("\"rejected\":\"記者の記者\""), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_segments_list_offers_what_the_user_chose_before() {
    // a choice learned for the sentence applies in a segment's list,
    // each candidate read as the sentence it makes
    let dir =
        std::env::temp_dir().join(format!("setsuna-bridge-seg-learned-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let learned = |chosen: &str, tag: &str| {
        let mut learning = Learning::at(
            dir.join(format!("{tag}.jsonl")),
            &crate::config::LearningCfg::default(),
        );
        learning
            .record("きしゃのきしゃ", chosen, "記者の記者")
            .unwrap();
        learning
    };
    // 帰社 right after the chosen 記者, which stays chosen
    let (mut bridge, mock) = converting_kisha();
    bridge.set_learning(Some(learned("記者の帰社", "after")));
    bridge.handle_move_cursor(1);
    bridge.handle_move_cursor(1);
    mock.answer_last(&["記者の"]);
    bridge.apply_async_result();
    mock.answer_last(&["記者", "帰社"]);
    let out = bridge.apply_async_result();
    let (list, selected, _) = segment_list(&out);
    assert_eq!(list[..3], v(&["記者", "帰社", "汽車"])[..]);
    assert_eq!(selected, 0);
    assert_eq!(segments_shown(&out).0, v(&["記者の", "記者"]));
    bridge.handle_down();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["記者の帰社"]);
    // one its own conversion lacks goes in all the same
    let (mut bridge, mock) = converting_kisha();
    bridge.set_learning(Some(learned("記者の帰社", "lacking")));
    bridge.handle_move_cursor(1);
    bridge.handle_move_cursor(1);
    mock.answer_last(&["記者の"]);
    bridge.apply_async_result();
    mock.answer_last(&["記者", "汽車"]);
    let out = bridge.apply_async_result();
    let (list, selected, _) = segment_list(&out);
    assert_eq!(list[..3], v(&["記者", "帰社", "汽車"])[..]);
    assert_eq!(selected, 0);
    // a resized segment's new list takes it first
    let (mut bridge, mock) = converting_kisha();
    bridge.set_learning(Some(learned("帰社の記者", "fresh")));
    bridge.handle_move_cursor(1);
    bridge.handle_resize(-1);
    mock.answer_last(&["汽車", "帰社", "記者"]);
    let out = bridge.apply_async_result();
    assert_eq!(segments_shown(&out).0, v(&["帰社", "のきしゃ"]));
    let _ = std::fs::remove_dir_all(&dir);
}

fn commits(actions: &[Action]) -> Vec<String> {
    actions
        .iter()
        .filter_map(|a| match a {
            Action::Commit { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn preedit(actions: &[Action]) -> Option<String> {
    actions.iter().rev().find_map(|a| match a {
        Action::UpdatePreedit { text } => Some(text.clone()),
        _ => None,
    })
}

/// `kasano` typed and converted, with `n` candidates (候補No1, …: a
/// digit by a Latin letter, which `drop_misread_digits` leaves).
fn converting_many(n: usize) -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    let cands: Vec<String> = (0..n).map(|i| format!("候補No{}", i + 1)).collect();
    let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
    mock.set_result("かさの", &refs);
    bridge.apply_async_result();
    bridge.handle_space();
    (bridge, mock)
}

#[test]
fn function_keys_convert_the_composition_to_a_form() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "tokyo");
    mock.set_result("ときょ", &["時許"]);
    bridge.apply_async_result();
    let out = bridge.handle_convert_to(Form::Katakana);
    assert_eq!(preedit(&out.actions).as_deref(), Some("トキョ"));
    assert!(out.actions.contains(&Action::HideCandidates));
    let out = bridge.handle_convert_to(Form::HalfKatakana);
    assert_eq!(preedit(&out.actions).as_deref(), Some("ﾄｷｮ"));
    // F10 again and again cycles the case
    let out = bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(preedit(&out.actions).as_deref(), Some("tokyo"));
    let out = bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(preedit(&out.actions).as_deref(), Some("TOKYO"));
    let out = bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(preedit(&out.actions).as_deref(), Some("Tokyo"));
    let out = bridge.handle_convert_to(Form::FullAlnum);
    assert_eq!(preedit(&out.actions).as_deref(), Some("ｔｏｋｙｏ"));
    // a late result of the reading does not replace the form
    mock.set_result("ときょ", &["時許"]);
    assert!(bridge.apply_async_result().is_empty());
    // Enter commits it, unlearned
    let out = bridge.handle_enter();
    assert_eq!(commits(&out.actions), vec!["ｔｏｋｙｏ"]);
    assert!(!bridge.is_active());
    // nothing composed: left to the app
    assert!(!bridge.handle_convert_to(Form::Katakana).consumed);
}

#[test]
fn escape_and_space_leave_a_form_for_the_conversion() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_convert_to(Form::Katakana);
    // Escape: back to typing, the kana typed shown
    let out = bridge.handle_escape();
    assert_eq!(preedit(&out.actions).as_deref(), Some("かさの"));
    bridge.handle_convert_to(Form::Katakana);
    // Space: the usual candidate window
    let out = bridge.handle_space();
    assert!(out.actions.iter().any(|a| matches!(
        a,
        Action::ShowCandidates { list, .. } if list[0] == "傘の"
    )));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の"]);
    // escape_to_kana off: Escape goes back to the conversion, still what
    // Enter commits
    let (mut bridge, mock) = bridge_with_mock();
    bridge.cfg.escape_to_kana = false;
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_convert_to(Form::Katakana);
    let out = bridge.handle_escape();
    assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の"]);
}

#[test]
fn shift_letters_convert_with_function_keys_too() {
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.handle_shifted_alpha('N');
    type_str(&mut bridge, "eko");
    let out = bridge.handle_convert_to(Form::Hiragana);
    assert_eq!(preedit(&out.actions).as_deref(), Some("ねこ"));
}

#[test]
fn digits_pick_from_the_open_page_when_enabled() {
    let (mut bridge, _mock) = converting_many(15);
    // off by default: a digit commits and starts typing
    let out = bridge.handle_char('2');
    assert_eq!(commits(&out.actions), vec!["候補No1"]);
    bridge.handle_escape();

    let (mut bridge, _mock) = converting_many(15);
    bridge.set_candidate_paging(9, true);
    // the reading and its katakana are 3rd and 4th
    let out = bridge.handle_char('5');
    assert_eq!(commits(&out.actions), vec!["候補No3"]);
    assert!(!bridge.is_active());

    // on the second page, 1 is the page's first
    let (mut bridge, _mock) = converting_many(15);
    bridge.set_candidate_paging(9, true);
    bridge.handle_page_down();
    let out = bridge.handle_char('1');
    assert_eq!(commits(&out.actions), vec!["候補No8"]);

    // past the list: nothing
    let (mut bridge, _mock) = converting_many(3);
    bridge.set_candidate_paging(9, true);
    let out = bridge.handle_char('6');
    assert!(out.consumed && out.actions.is_empty());
    // typing digits is unaffected
    let (mut bridge, _mock) = bridge_with_mock();
    bridge.set_candidate_paging(9, true);
    type_str(&mut bridge, "1");
    assert!(bridge.is_active());
}

#[test]
fn page_keys_step_by_the_window_page() {
    // 候補No1, 候補No2, かさの, カサノ, 候補No3, …, 候補No12: 14 candidates
    let (mut bridge, _mock) = converting_many(12);
    bridge.set_candidate_paging(5, false);
    bridge.handle_down();
    let out = bridge.handle_page_down();
    assert_eq!(preedit(&out.actions).as_deref(), Some("候補No5"));
    // the last page is short: its last candidate
    bridge.handle_down();
    bridge.handle_down();
    bridge.handle_down();
    let out = bridge.handle_page_down();
    assert_eq!(preedit(&out.actions).as_deref(), Some("候補No12"));
    // and it wraps
    let out = bridge.handle_page_down();
    assert_eq!(preedit(&out.actions).as_deref(), Some("カサノ"));
    let out = bridge.handle_home_end(true);
    assert_eq!(preedit(&out.actions).as_deref(), Some("候補No12"));
    let out = bridge.handle_home_end(false);
    assert_eq!(preedit(&out.actions).as_deref(), Some("候補No1"));
}

#[test]
fn tab_opens_then_steps_through_the_candidates() {
    let (mut bridge, mock) = bridge_with_mock();
    assert!(!bridge.handle_tab(false).consumed);
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    let out = bridge.handle_tab(false);
    assert!(bridge.is_converting());
    assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
    let out = bridge.handle_tab(false);
    assert_eq!(preedit(&out.actions).as_deref(), Some("笠の"));
    let out = bridge.handle_tab(true);
    assert_eq!(preedit(&out.actions).as_deref(), Some("傘の"));
}

#[test]
fn caret_edits_show_and_commit_the_kana() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    bridge.apply_async_result();
    assert_eq!(bridge.caret(), None);
    let out = bridge.handle_home_end(false);
    assert_eq!(preedit(&out.actions).as_deref(), Some("かさの"));
    assert_eq!(bridge.caret(), Some(0));
    let out = bridge.handle_delete();
    assert_eq!(preedit(&out.actions).as_deref(), Some("さの"));
    bridge.handle_move_cursor(1);
    assert_eq!(bridge.caret(), Some(1));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["さの"]);
    // Delete of the last char ends the composition
    type_str(&mut bridge, "a");
    bridge.handle_move_cursor(-1);
    let out = bridge.handle_delete();
    assert_eq!(out.actions, vec![Action::EndComposition]);
    assert!(!bridge.is_active());
}

#[test]
fn space_widths() {
    let (mut bridge, _mock) = bridge_with_mock();
    assert!(!bridge.handle_space().consumed);
    assert_eq!(
        commits(&bridge.handle_shift_space().actions),
        vec!["\u{3000}"]
    );
    let cfg = crate::config::FrontendCfg {
        full_width_space: true,
        ..Default::default()
    };
    let mut bridge = EngineBridge::with_frontend(None, &cfg);
    assert_eq!(commits(&bridge.handle_space().actions), vec!["\u{3000}"]);
    assert!(!bridge.handle_shift_space().consumed);
    // composing: Shift+Space converts like Space
    type_str(&mut bridge, "ka");
    bridge.handle_shift_space();
    assert!(bridge.is_converting());
}

#[test]
fn without_live_conversion_the_kana_stays_and_conversions_are_suggested() {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_live_conversion(false);
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    let actions = bridge.apply_async_result();
    assert_eq!(preedit(&actions).as_deref(), Some("かさの"));
    assert_eq!(bridge.suggestions()[..2], ["傘の", "笠の"]);
    // a stale list (for a shorter reading) is not suggested
    type_str(&mut bridge, "k");
    assert!(
        !bridge.suggestions().is_empty(),
        "pending romaji keeps them"
    );
    type_str(&mut bridge, "a");
    assert!(bridge.suggestions().is_empty());
    bridge.handle_backspace();
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    // Enter commits the kana shown
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["かさの"]);

    // a click on a suggestion opens the conversion on it
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    let out = bridge.handle_select(1);
    assert!(bridge.is_converting());
    assert_eq!(preedit(&out.actions).as_deref(), Some("笠の"));
    assert!(bridge.suggestions().is_empty());
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["笠の"]);

    // ↓ opens it too
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_down();
    assert!(bridge.is_converting());
}

#[test]
fn live_conversion_suggests_nothing() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    let actions = bridge.apply_async_result();
    assert_eq!(preedit(&actions).as_deref(), Some("傘の"));
    assert!(bridge.suggestions().is_empty());
}

/// A fresh directory under the temp dir for a test's files.
fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("setsuna-bridge-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn ctrl_delete_on_a_prediction_forgets_it_not_the_learning() {
    // the crash: 傘の下 learned over 笠の下 and predicted, Tab on かさの,
    // Ctrl+Delete; the learned word 傘 made the learning's forget succeed
    // and the one-prediction list was emptied, then indexed
    let dir = temp_dir("forget-prediction");
    let (mut bridge, mock) = bridge_with_mock();
    let cfg = crate::config::LearningCfg::default();
    bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
    bridge.set_hidden(Some(crate::hidden::Hidden::at(dir.join("hidden.jsonl"))));
    bridge.set_predictor(Some(crate::predict::Predictor::at(
        dir.join("prediction.jsonl"),
    )));
    let convert = |bridge: &mut EngineBridge| {
        type_str(bridge, "kasanoshita");
        mock.set_result("かさのした", &["笠の下", "傘の下"]);
        bridge.apply_async_result();
        bridge.handle_space()
    };
    for _ in 0..2 {
        let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
        for _ in 0..list.iter().position(|c| c == "傘の下").unwrap() {
            bridge.handle_down();
        }
        assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘の下"]);
    }
    type_str(&mut bridge, "kasano");
    let out = bridge.handle_tab(false);
    assert_eq!(shown(&out.actions).unwrap().0, v(&["傘の下"]));
    let out = bridge.handle_forget_candidate();
    assert!(out.consumed);
    assert!(out.actions.contains(&Action::HideCandidates));
    assert_eq!(preedit(&out.actions).as_deref(), Some("かさの"));
    assert!(!bridge.is_converting() && bridge.is_active());
    // predicted no more, here and in a new predictor of the file
    assert!(bridge.predictions().is_empty());
    let mut again = crate::predict::Predictor::at(dir.join("prediction.jsonl"));
    assert!(again.complete("かさの", "", 3).is_empty());
    // the learning and the hidden candidates are as they were
    assert!(!dir.join("hidden.jsonl").exists());
    bridge.handle_escape();
    let (list, _) = shown(&convert(&mut bridge).actions).unwrap();
    assert_eq!(list[0], "傘の下");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ctrl_delete_keeps_a_lists_last_candidate_learned_or_not() {
    let dir = temp_dir("forget-one");
    let (mut bridge, _mock) = bridge_with_mock();
    let cfg = crate::config::LearningCfg::default();
    let mut learning = Learning::at(dir.join("learning.jsonl"), &cfg);
    learning.record("かさの", "傘の", "笠の").unwrap();
    bridge.set_learning(Some(learning));
    bridge.set_hidden(Some(crate::hidden::Hidden::at(dir.join("hidden.jsonl"))));
    type_str(&mut bridge, "kasano");
    let one = |bridge: &mut EngineBridge, text: &str| {
        bridge.comp.list_reading = "かさの".into();
        bridge.phase = Phase::Converting {
            candidates: v(&[text]),
            selected: 0,
            seg: None,
        };
    };
    // learned: the learning goes, the candidate stays shown
    one(&mut bridge, "傘の");
    let out = bridge.handle_forget_candidate();
    assert_eq!(status(&out.actions), Some("学習を取り消しました"));
    assert!(bridge.is_converting());
    assert_eq!(bridge.current_list(), v(&["傘の"]));
    // not learned: it cannot go
    one(&mut bridge, "笠の");
    let out = bridge.handle_forget_candidate();
    assert_eq!(status(&out.actions), Some("この候補は非表示にできません"));
    assert_eq!(bridge.current_list(), v(&["笠の"]));
    assert!(!dir.join("hidden.jsonl").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn escape_shows_the_kana_then_clears() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    // the conversion shown: the kana typed
    let out = bridge.handle_escape();
    assert_eq!(
        out.actions,
        vec![Action::UpdatePreedit {
            text: "かさの".into()
        }]
    );
    assert!(bridge.is_active());
    // a late result does not bring the conversion back
    mock.set_result("かさの", &["傘の"]);
    assert!(bridge.apply_async_result().is_empty());
    // Escape again clears
    assert_eq!(bridge.handle_escape().actions, vec![Action::EndComposition]);
    assert!(!bridge.is_active());

    // Enter on the kana commits the kana
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_escape();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["かさの"]);

    // Space converts again, at once
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_escape();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "傘の");
    // from the candidate window: the kana, then cleared
    let out = bridge.handle_escape();
    assert!(out.actions.contains(&Action::HideCandidates));
    assert_eq!(preedit(&out.actions).as_deref(), Some("かさの"));
    assert_eq!(bridge.handle_escape().actions, vec![Action::EndComposition]);

    // typing on converts again
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の", "笠の"]);
    bridge.apply_async_result();
    bridge.handle_escape();
    type_str(&mut bridge, "de");
    mock.set_result("かさので", &["傘ので"]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("傘ので")
    );
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘ので"]);

    // the kana shown already: one Escape clears
    type_str(&mut bridge, "ka");
    assert_eq!(bridge.handle_escape().actions, vec![Action::EndComposition]);
}

#[test]
fn the_app_ending_the_composition_drops_it_whole() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kasano");
    mock.set_result("かさの", &["傘の"]);
    bridge.apply_async_result();
    bridge.handle_space();
    let actions = bridge.cancel();
    assert_eq!(
        actions,
        vec![Action::HideCandidates, Action::EndComposition]
    );
    assert!(!bridge.is_active());
    assert!(bridge.cancel().is_empty());
}

#[test]
fn space_on_letters_the_user_dictionary_reads_lists_its_words() {
    let (entries, skipped) =
        crate::user_dict::parse("w\t(笑)\t名詞\nｗ\t（ｗ）\t名詞\nわ\t輪\t名詞\n");
    assert!(skipped.is_empty());
    let dict = || Some(crate::user_dict::UserDict::new(&entries));
    // romaji input: w is pending romaji, no kana
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_user_dict(dict());
    bridge.handle_char('w');
    let out = bridge.handle_space();
    assert_eq!(shown(&out.actions).unwrap().0, v(&["(笑)", "（ｗ）", "w"]));
    assert_eq!(preedit(&out.actions).as_deref(), Some("(笑)"));
    // nothing typed is left after it
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["(笑)"]);
    assert!(!bridge.is_active());
    // the input as typed is offered too
    bridge.handle_char('w');
    bridge.handle_space();
    bridge.handle_down();
    bridge.handle_down();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["w"]);
    // a reading of one kana still converts as one
    type_str(&mut bridge, "wa");
    bridge.handle_space();
    mock.set_result("わ", &["わ", "和"]);
    let (list, _) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!(list[0], "輪");
    bridge.handle_escape();
    bridge.handle_escape();
    // letters it has no reading for: as before
    bridge.handle_char('x');
    assert_eq!(commits(&bridge.handle_space().actions), vec!["x"]);

    // mixed input: no space typed after it
    let (mut bridge, _mock) = mixed_bridge();
    bridge.set_user_dict(dict());
    bridge.handle_char('w');
    let out = bridge.handle_space();
    assert!(bridge.is_converting());
    assert_eq!(shown(&out.actions).unwrap().0[0], "(笑)");
}

#[test]
fn every_user_word_of_the_other_reading_is_offered() {
    // yoro read as English (after an English sentence): the dictionary's
    // ten words for よろ come with the reading every letter as romaji,
    // past the five candidates that reading gives otherwise
    let (mut bridge, mock) = mixed_bridge();
    let words: Vec<String> = (0..10).map(|i| format!("語{i}")).collect();
    let text: String = words.iter().map(|w| format!("よろ\t{w}\t名詞\n")).collect();
    let (entries, _) = crate::user_dict::parse(&text);
    bridge.set_user_dict(Some(crate::user_dict::UserDict::new(&entries)));
    bridge.phase = Phase::Converting {
        candidates: v(&["yoro"]),
        selected: 0,
        seg: None,
    };
    bridge.comp.alt = Some(AltList {
        reading: Some("よろ".into()),
        req_id: Some(7),
        awaited: false,
        candidates: Vec::new(),
        main: v(&["yoro"]),
        ends_english: false,
        typed: "yoro".into(),
        typed_early: false,
        user_words: Vec::new(),
    });
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id: 7,
        reading: "よろ".into(),
        candidates: v(&["余呂", "世路", "夜路", "四路", "代ろ", "与呂"]),
        ..Default::default()
    });
    let (list, _) = shown(&bridge.apply_async_result()).unwrap();
    for w in &words {
        assert!(list.contains(w), "{w} not in {list:?}");
    }
    // they come first of the other reading's, right after the main top
    assert_eq!(list[..4], v(&["yoro", "語0", "語1", "語2"])[..]);
    // and the others are cut as before
    let others = list
        .iter()
        .filter(|c| !words.contains(c) && *c != "yoro")
        .count();
    assert_eq!(others, ALT_MAX);
}

#[test]
fn the_other_readings_late_list_leaves_a_form_shown() {
    // the other reading's list came after its wait and F7 was pressed
    // first: the katakana form stays what Enter commits
    let (mut bridge, mock) = mixed_bridge();
    type_str(&mut bridge, "yoro");
    bridge.phase = Phase::Converting {
        candidates: v(&["yoro"]),
        selected: 0,
        seg: None,
    };
    bridge.comp.alt = Some(AltList {
        reading: Some("よろ".into()),
        req_id: Some(7),
        awaited: false,
        candidates: Vec::new(),
        main: v(&["yoro"]),
        ends_english: false,
        typed: "yoro".into(),
        typed_early: false,
        user_words: Vec::new(),
    });
    let out = bridge.handle_convert_to(Form::Katakana);
    let form = preedit(&out.actions).expect("a form shown");
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id: 7,
        reading: "よろ".into(),
        candidates: v(&["余呂", "世路"]),
        ..Default::default()
    });
    assert!(shown(&bridge.apply_async_result()).is_none());
    assert_eq!(commits(&bridge.handle_enter().actions), vec![form]);
}

#[test]
fn a_shortcut_word_stays_out_of_the_preedit_of_a_longer_reading() {
    let (mut bridge, mock) = bridge_with_mock();
    let (entries, _) = crate::user_dict::parse("じゅうしょ\t東京都千代田区1-1\t短縮よみ\n");
    bridge.set_user_dict(Some(crate::user_dict::UserDict::new(&entries)));
    type_str(&mut bridge, "juusho");
    mock.set_result("じゅうしょ", &["住所", "じゅうしょ"]);
    // the whole reading shows it, and Enter would commit it
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("東京都千代田区1-1")
    );
    // typed on, the conversion before the kana is the model's, not it
    // (while a letter is pending the reading is still its own)
    bridge.handle_char('w');
    assert_eq!(
        preedit(&bridge.handle_char('o').actions).as_deref(),
        Some("住所を")
    );
    // deleted back to it, it shows again
    assert_eq!(
        preedit(&bridge.handle_backspace().actions).as_deref(),
        Some("東京都千代田区1-1")
    );
    assert_eq!(
        commits(&bridge.handle_enter().actions),
        vec!["東京都千代田区1-1"]
    );
}

/// `傘をyoro` converting, `yoro` the English of mixed input, cut at
/// the を: two segments, the first focused.
fn editing_an_english_segment() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    bridge.comp.list_reading = "かさをyoro".into();
    bridge
        .comp
        .list_starts
        .insert("傘をyoro".into(), vec![0, 2, 3, 4, 5, 6]);
    bridge.phase = Phase::Converting {
        candidates: v(&["傘をyoro"]),
        selected: 0,
        seg: None,
    };
    let (texts, _) = segments_shown(&bridge.handle_move_cursor(1).actions);
    assert_eq!(texts, v(&["傘を", "yoro"]));
    (bridge, mock)
}

#[test]
fn an_english_segment_offers_its_letters_read_as_romaji() {
    let (mut bridge, mock) = editing_an_english_segment();
    let (list, _, _) = segment_list(&bridge.handle_move_cursor(1).actions);
    assert_eq!(list, v(&["yoro", "よろ", "ヨロ"]));
    // the first segment's conversion, then the English one's, then
    // that of its letters as romaji
    mock.answer_last(&["傘を", "笠を"]);
    bridge.apply_async_result();
    assert_eq!(mock.dispatched_readings().last().unwrap(), "yoro");
    mock.answer_last(&["yoro"]);
    bridge.apply_async_result();
    let (context, reading) = mock.answer_last(&["余呂", "よろ"]);
    assert_eq!((context.as_str(), reading.as_str()), ("傘を", "よろ"));
    let (list, selected, _) = segment_list(&bridge.apply_async_result());
    assert_eq!(list, v(&["yoro", "よろ", "ヨロ", "余呂"]));
    assert_eq!(selected, 0);
    bridge.handle_down();
    bridge.handle_down();
    bridge.handle_down();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["傘を余呂"]);
}

#[test]
fn function_keys_convert_the_focused_segment_only() {
    let (mut bridge, _mock) = editing_an_english_segment();
    bridge.handle_move_cursor(1);
    let out = bridge.handle_convert_to(Form::Katakana);
    assert_eq!(segments_shown(&out.actions).0, v(&["傘を", "ヨロ"]));
    assert!(bridge.is_converting());
    // F10 again and again cycles the case
    let out = bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(segments_shown(&out.actions).0, v(&["傘を", "yoro"]));
    let out = bridge.handle_convert_to(Form::HalfAlnum);
    assert_eq!(segments_shown(&out.actions).0, v(&["傘を", "YORO"]));
    // the other segment by its kana and romaji
    bridge.handle_move_cursor(-1);
    let out = bridge.handle_convert_to(Form::FullAlnum);
    assert_eq!(segments_shown(&out.actions).0, v(&["ｋａｓａｗｏ", "YORO"]));
    let out = bridge.handle_convert_to(Form::Hiragana);
    assert_eq!(segments_shown(&out.actions).0, v(&["かさを", "YORO"]));
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["かさをYORO"]);
}

#[test]
fn no_client_degrades_to_raw_hiragana() {
    // With no backend the bridge must still accept input and render a
    // raw-kana preedit rather than panic.
    let mut bridge = EngineBridge::new(None);
    type_str(&mut bridge, "ka");
    let outcome = bridge.handle_char('n');
    assert!(outcome.consumed);
    assert_eq!(
        outcome.actions,
        vec![Action::UpdatePreedit {
            text: "かn".into()
        }]
    );
}

#[test]
fn space_on_digits_lists_their_dates_first_before_the_engine_answers() {
    // user 2026-11-10: 1001 converts to 10月1日; the digits stay first
    let (mut bridge, _mock) = bridge_with_mock();
    type_str(&mut bridge, "1001");
    let (list, selected) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(selected, 0);
    assert_eq!(
        list[..6],
        v(&["1001", "10月1日", "10/1", "10/01", "10:01", "10時1分"])[..]
    );
    assert!(list.contains(&"千一".to_string()));
    // two digits: their forms at once, no date
    let (mut bridge, _mock) = bridge_with_mock();
    type_str(&mut bridge, "10");
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[..3], v(&["10", "１０", "十"])[..]);
}

fn voicing_bridge() -> (EngineBridge, MockClient) {
    let (mut bridge, mock) = bridge_with_mock();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mixed");
    bridge.set_mixed(
        Some(std::sync::Arc::new(
            crate::mixed::MixedModel::load(&dir).unwrap(),
        )),
        &crate::config::default_symbol_map(),
        crate::mixed::Casing::default(),
    );
    (bridge, mock)
}

#[test]
fn zu_for_du_converts_as_du_and_the_reading_typed_comes_too() {
    let (mut bridge, mock) = voicing_bridge();
    type_str(&mut bridge, "kizuitara");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きづいたら");
    mock.set_result("きづいたら", &["気づいたら", "気付いたら"]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "気づいたら");
    // the reading as typed, its candidates after the first
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きずいたら");
    mock.set_result("きずいたら", &["築いたら", "きずいたら"]);
    let (list, selected) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!((list[0].as_str(), selected), ("気づいたら", 0));
    assert!(list.contains(&"築いたら".to_string()), "{list:?}");

    // no word as typed (つずく): converted as つづく, alone
    let (mut bridge, mock) = voicing_bridge();
    type_str(&mut bridge, "tuzuku");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "つづく");
    mock.set_result("つづく", &["続く", "つづく"]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "続く");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "つづく");

    // both in use (築く, 気づく): converted as typed, the other offered
    let (mut bridge, mock) = voicing_bridge();
    type_str(&mut bridge, "kizuku");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きずく");
    mock.set_result("きずく", &["築く"]);
    bridge.apply_async_result();
    bridge.handle_space();
    assert_eq!(mock.dispatched_readings().last().unwrap(), "きづく");

    // off with typo correction
    let (mut bridge, mock) = voicing_bridge();
    bridge.composing.set_voicing(false);
    type_str(&mut bridge, "tuzuku");
    assert_eq!(mock.dispatched_readings().last().unwrap(), "つずく");
}

#[test]
fn a_shift_released_a_key_late_offers_the_reading_put_right() {
    // issue #6: IMENo stays English as typed (capitals are), and the
    // reading with the N as romaji is converted next to it
    let (mut bridge, mock) = voicing_bridge();
    type_str(&mut bridge, "IMENotsukaikata");
    assert_eq!(
        mock.dispatched_readings().last().unwrap(),
        "IMENおつかいかた"
    );
    mock.set_result(
        "IMENおつかいかた",
        &[
            "IMENお使い方",
            "IMENお使いかた",
            "IMEN御使い方",
            "IMENおつかいかた",
        ],
    );
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[0], "IMENお使い方");
    assert_eq!(
        mock.dispatched_readings().last().unwrap(),
        "IMEのつかいかた"
    );
    mock.set_result("IMEのつかいかた", &["IMEの使い方", "IMEの使いかた"]);
    // its first right after the first three as typed, which stays chosen
    let (list, selected) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!((list[0].as_str(), selected), ("IMENお使い方", 0));
    assert_eq!(list[3], "IMEの使い方", "{list:?}");

    // ending in English: Space types a space, Space again converts,
    // the reading put right before every letter as romaji (いめの)
    let (mut bridge, mock) = voicing_bridge();
    type_str(&mut bridge, "IMENo");
    bridge.handle_space();
    bridge.handle_space();
    mock.answer_last(&["IMENo", "IMENO"]);
    bridge.apply_async_result();
    assert_eq!(mock.dispatched_readings().last().unwrap(), "IMEの");
    // second, after the English with a space, which stays chosen
    mock.answer_last(&["IMEの", "IMEノ"]);
    let (list, selected) = shown(&bridge.apply_async_result()).unwrap();
    assert_eq!((list[0].as_str(), selected), ("IMENo ", 0));
    assert_eq!(list[1], "IMEの", "{list:?}");

    // off with typo correction: every letter as romaji, as before
    let (mut bridge, mock) = voicing_bridge();
    bridge.cfg.typo_correction = false;
    type_str(&mut bridge, "IMENo");
    bridge.handle_space();
    bridge.handle_space();
    mock.answer_last(&["IMENo"]);
    bridge.apply_async_result();
    assert_eq!(mock.dispatched_readings().last().unwrap(), "いめの");
}

#[test]
fn a_slip_put_right_chosen_once_comes_first_after() {
    // issue #6: Space waits for the other reading's list, so the list
    // opens with it and the user's choices apply over the whole: IMEの
    // chosen over IMENo once is first the next time
    let dir =
        std::env::temp_dir().join(format!("setsuna-bridge-slip-learn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (mut bridge, mock) = voicing_bridge();
    let cfg = crate::config::LearningCfg::default();
    bridge.set_learning(Some(Learning::at(dir.join("learning.jsonl"), &cfg)));
    // the engine answers the reading put right as soon as it is asked
    mock.state
        .answers
        .borrow_mut()
        .insert("IMEの".into(), v(&["IMEの", "IMEノ"]));
    let convert = |bridge: &mut EngineBridge| {
        type_str(bridge, "IMENo");
        bridge.handle_space();
        bridge.handle_space();
        mock.answer_last(&["IMENo", "IMENO"]);
        shown(&bridge.apply_async_result()).unwrap()
    };
    let (list, selected) = convert(&mut bridge);
    assert_eq!(list[..3], v(&["IMENo ", "IMEの", "IMEノ"])[..]);
    assert_eq!(selected, 0);
    bridge.handle_down();
    assert_eq!(commits(&bridge.handle_enter().actions), vec!["IMEの"]);
    let (list, selected) = convert(&mut bridge);
    assert_eq!((list[0].as_str(), selected), ("IMEの", 0));
    assert!(list.contains(&"IMENo ".to_string()), "{list:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_a_lexicon_the_candidates_far_below_the_top_go() {
    // user 2026-11-10, a phone without the dictionaries: 今日 日 今陽 昨日 今ウ 況日 京日
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kyou");
    *mock.state.next.borrow_mut() = Some(EngineResult {
        req_id: mock.state.dispatched.borrow().last().map_or(1, |d| d.0),
        reading: "きょう".into(),
        candidates: v(&["今日", "京", "日", "今陽", "況日"]),
        scores: vec![-0.1, -4.0, -9.9, -12.4, -13.0],
        ..Default::default()
    });
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[..2], v(&["今日", "京"])[..]);
    for gone in ["日", "今陽", "況日"] {
        assert!(!list.contains(&gone.to_string()), "{gone} in {list:?}");
    }
}

#[test]
fn a_partial_lexicon_lists_its_words_but_does_not_judge_the_engine() {
    // the commoner words only (Lexicon::build_partial): 京 listed for
    // きょう, the engine's order kept (its 今陽 unknown to the lexicon is
    // no reason to drop 今日's neighbour), what is far below the top gone
    let table = "今日\tきょう\t8.7\tnoun\n京\tきょう\t12.7\tpropn\n\
                 京都\tきょうと\t9.0\tpropn\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-partial-{}.lexicon",
        std::process::id()
    ));
    let bytes = crate::lexicon::Lexicon::build_partial(table, None, 15.0).unwrap();
    std::fs::write(&path, bytes).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "kyou");
    answer_scored(&mock, &["今日", "今陽", "況日"], &[-0.1, -3.0, -9.0]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    assert_eq!(list[..2], v(&["今日", "今陽"])[..]);
    assert!(list.contains(&"京".to_string()), "{list:?}");
    assert!(!list.contains(&"況日".to_string()), "{list:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_corrected_readings_kana_candidate_stays_in_the_list_and_its_segment() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "arigato");
    // the engine's reading corrector made ありがとう of ありがと, both
    // readings' candidates merged, positions on the reading typed
    mock.set_aligned(
        "ありがと",
        &["ありがとう", "有難う", "ありがど"],
        &[&[0, 1, 2, 3, 3], &[0, 2, 3], &[0, 1, 2, 3]],
    );
    mock.state
        .next
        .borrow_mut()
        .as_mut()
        .unwrap()
        .corrected_reading = Some("ありがとう".into());
    bridge.apply_async_result();
    let out = bridge.handle_space();
    let list = out
        .actions
        .iter()
        .find_map(|a| match a {
            Action::ShowCandidates { list, .. } => Some(list.clone()),
            _ => None,
        })
        .expect("ShowCandidates");
    assert_eq!(
        list[..4],
        v(&["ありがとう", "有難う", "ありがと", "アリガト"])[..]
    );
    // segment editing keeps the text it was cut from
    let out = bridge.handle_move_cursor(1);
    assert_eq!(segments_shown(&out.actions), (v(&["ありがとう"]), 0));
    let (list, selected, _) = segment_list(&out.actions);
    assert_eq!(list[selected], "ありがとう");
}

#[test]
fn a_date_is_never_learned_nor_put_first_by_learning() {
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-date-learning-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let cfg = crate::config::LearningCfg::default();
    // chosen from the list: not learned
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_learning(Some(Learning::at(path.clone(), &cfg)));
    type_str(&mut bridge, "kyou");
    answer_scored(&mock, &["今日", "京"], &[-0.1, -2.0]);
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    let date =
        crate::date_time::candidates("きょう", &bridge.comp.date_time_now.unwrap())[0].clone();
    bridge.handle_select(list.iter().position(|c| *c == date).unwrap());
    assert_eq!(commits(&bridge.handle_enter().actions), vec![date.clone()]);
    assert_eq!(bridge.learning.as_mut().unwrap().len(), 0);
    // learned before (the phone's case): きょう still converts to 今日
    Learning::at(path.clone(), &cfg)
        .record("きょう", &date, "今日")
        .unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_learning(Some(Learning::at(path.clone(), &cfg)));
    type_str(&mut bridge, "kyou");
    answer_scored(&mock, &["今日", "京"], &[-0.1, -2.0]);
    assert_eq!(
        preedit(&bridge.apply_async_result()).as_deref(),
        Some("今日")
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn space_before_the_engine_answers_does_not_convert_to_the_date() {
    let (mut bridge, mock) = bridge_with_mock();
    type_str(&mut bridge, "kyou");
    let out = bridge.handle_space();
    let shown_first = shown(&out.actions).map(|(l, _)| l[0].clone());
    let pre = preedit(&out.actions);
    let date =
        crate::date_time::candidates("きょう", &bridge.comp.date_time_now.unwrap())[0].clone();
    assert_ne!(
        shown_first.as_deref(),
        Some(date.as_str()),
        "{:?}",
        out.actions
    );
    assert_ne!(pre.as_deref(), Some(date.as_str()), "{:?}", out.actions);
    let _ = mock;
}

#[test]
fn a_date_reading_converts_live_to_its_word_not_to_the_date() {
    for engine in [
        &["今日", "京", "教"][..],
        &["今日"][..],
        &["きょう", "今日"][..],
    ] {
        let (mut bridge, mock) = bridge_with_mock();
        type_str(&mut bridge, "kyou");
        let scores: Vec<f32> = (0..engine.len()).map(|i| -(i as f32)).collect();
        answer_scored(&mock, engine, &scores);
        let shown = preedit(&bridge.apply_async_result());
        assert_eq!(shown.as_deref(), Some(engine[0]), "{engine:?}");
    }
}

#[test]
fn the_lexicon_keeps_the_corrected_readings_candidates() {
    // what the lexicon costs by the reading typed alone, the corrected
    // reading's words cost nothing and went (ごめんなさい for
    // ごめんなさあい): they are costed by the corrected reading
    let table = "有難う\tありがとう\t9.0\tinterj\n有り難う\tありがとう\t11.0\tinterj\n\
                 有り\tあり\t8.0\tverb\n";
    let path = std::env::temp_dir().join(format!(
        "setsuna-bridge-corrected-{}.lexicon",
        std::process::id()
    ));
    std::fs::write(&path, crate::lexicon::Lexicon::build(table, None).unwrap()).unwrap();
    let (mut bridge, mock) = bridge_with_mock();
    bridge.set_lexicon(&crate::config::LexiconCfg {
        file: Some(path.to_string_lossy().into_owned()),
    });
    type_str(&mut bridge, "arigato");
    // the typed reading's top first, the corrected reading's after it
    answer_scored(
        &mock,
        &["有りがと", "有難う", "有り難う"],
        &[-0.5, -0.7, -1.2],
    );
    mock.state
        .next
        .borrow_mut()
        .as_mut()
        .unwrap()
        .corrected_reading = Some("ありがとう".into());
    bridge.apply_async_result();
    let (list, _) = shown(&bridge.handle_space().actions).unwrap();
    // the kana of the reading typed after the first two, as always
    assert_eq!(list[..2], v(&["有りがと", "有難う"])[..]);
    assert!(list.contains(&"有り難う".to_string()), "{list:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn composition_state_clear_resets_all_transient_fields() {
    let mut comp = CompositionState {
        last_applied_reading: "きょう".into(),
        cached_prefix_reading: "きょう".into(),
        cached_prefix_kanji: "今日".into(),
        cached_shortcut: Some("短縮".into()),
        date_time_now: Some(chrono::Local::now()),
        prediction: Some(Vec::new()),
        covers_typed: true,
        deferred: Some(Deferred {
            reading: "きょう".into(),
            top: "今日".into(),
            until: Instant::now(),
        }),
        list_reading: "きょう".into(),
        list_starts: HashMap::from([("今日".into(), vec![0, 0])]),
        seg_requests: vec![(1, 0, 0, false)],
        kana_shown: true,
        kept: Some(("きょう".into(), "今日".into(), vec![0, 0])),
        form: Some(FormShown {
            form: Form::Katakana,
            case: 1,
            text: "キョウ".into(),
            list: vec!["今日".into()],
        }),
        seg_form: Some((0, Form::Katakana, 1, "キョウ".into())),
        rerank: Some(RerankJob {
            reading: "きょう".into(),
            corrected_reading: None,
            context: String::new(),
            raw: vec!["今日".into()],
            scores: vec![0.0],
            spelled: vec![true],
            reading_pos: Vec::new(),
            k: 1,
            req_id: 1,
            due: Instant::now(),
        }),
        alt: Some(AltList {
            reading: Some("きょう".into()),
            req_id: Some(1),
            awaited: true,
            candidates: vec!["今日".into()],
            main: vec!["今日".into()],
            ends_english: true,
            typed: "kyou".into(),
            typed_early: true,
            user_words: vec!["今日".into()],
        }),
        rewrite: Some(RewriteJob {
            req_id: 1,
            reading: "きょう".into(),
            text: Some("本日".into()),
        }),
        unfinished: Some(crate::engine_client::EngineResult {
            reading: "きょう".into(),
            ..Default::default()
        }),
        end_held: Some(("きょう".into(), "今日".into())),
        live_out: Some("きょう".into()),
        last_segment_due: true,
    };
    comp.clear();
    assert!(comp.last_applied_reading.is_empty());
    assert!(comp.cached_prefix_reading.is_empty());
    assert!(comp.cached_prefix_kanji.is_empty());
    assert!(comp.cached_shortcut.is_none());
    assert!(comp.date_time_now.is_none());
    assert!(comp.prediction.is_none());
    assert!(!comp.covers_typed);
    assert!(comp.deferred.is_none());
    assert!(comp.list_reading.is_empty());
    assert!(comp.list_starts.is_empty());
    assert!(comp.seg_requests.is_empty());
    assert!(!comp.kana_shown);
    assert!(comp.kept.is_none());
    assert!(comp.form.is_none());
    assert!(comp.seg_form.is_none());
    assert!(comp.rerank.is_none());
    assert!(comp.alt.is_none());
    assert!(comp.rewrite.is_none());
    assert!(comp.unfinished.is_none());
    assert!(comp.end_held.is_none());
    assert!(comp.live_out.is_none());
    assert!(!comp.last_segment_due);
}
