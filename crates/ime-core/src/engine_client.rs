//! Conversion-backend abstraction.
//!
//! `EngineBridge` talks to the engine only through `EngineClient`, so no
//! frontend links `setsuna-engine` directly. The production implementations
//! are `child_daemon::ChildDaemonClient` (a child `setsuna-engine daemon`
//! process) and, on Windows, `child_daemon::PipeEngineClient` (the login's
//! shared `setsuna-engine serve`); tests use an in-memory mock.

/// Called from a backend thread when a conversion result is ready, with the
/// request id. The frontend wakes its UI thread with it (a posted window
/// message, an event-loop wakeup), which then calls
/// `EngineBridge::apply_async_result`.
pub type Notify = Box<dyn Fn(u64) + Send + 'static>;

/// A completed conversion delivered by the backend.
#[derive(Debug, Clone, Default)]
pub struct EngineResult {
    /// Request id this result answers (monotonic, assigned at dispatch).
    pub req_id: u64,
    /// The reading (full hiragana) the candidates were produced for. Lets
    /// the bridge check the result is still a prefix of what the user has
    /// typed before applying it.
    pub reading: String,
    /// The left context the candidates were produced after, as the engine
    /// read it (NFKC); empty when not known.
    pub context: String,
    /// Ranked candidate surfaces, best-first.
    pub candidates: Vec<String>,
    /// Per candidate, for each char the index of the reading char it was
    /// emitted at (the engine's CTC alignment); empty when not given.
    pub reading_pos: Vec<Vec<u32>>,
    /// Per candidate, the engine's log-probability score; empty when not
    /// given.
    pub scores: Vec<f32>,
    /// Time the engine spent on the conversion, when it reports it.
    pub engine_ms: Option<f64>,
    /// From dispatch to the result arriving, queueing and pacing included.
    pub round_trip_ms: f64,
    /// The model's output frame by frame, as the engine sent it, when asked
    /// for (`EngineCfg::emit_frames`).
    pub frames: Option<serde_json::Value>,
}

/// A word of the engine's analysis of a text (`--analyzer`).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
pub struct Word {
    pub surface: String,
    /// The surface's chars `[start, end)` in the text.
    pub start: usize,
    pub end: usize,
    /// Hiragana; none for a word the analyzer's dictionary does not hold.
    #[serde(default)]
    pub reading: Option<String>,
    /// The part of speech and its subdivision (名詞 / 普通名詞, 動詞 /
    /// 非自立可能).
    #[serde(default)]
    pub pos: String,
    #[serde(default)]
    pub pos2: String,
    /// Other readings and how much more the analysis costs with each.
    #[serde(default)]
    pub alts: Vec<(String, i32)>,
    /// The analysis's cost from the text's start through this word.
    #[serde(default)]
    pub cost: i32,
}

/// Backend the `EngineBridge` dispatches conversions to.
///
/// All methods take `&self`: the production client owns background threads
/// and uses interior mutability, so a shared `&` is enough for the UI
/// thread to drive it.
pub trait EngineClient {
    /// Queue a conversion. Only the latest queued request matters — a burst
    /// of keystrokes collapses to one backend run. Cheap and non-blocking;
    /// the result arrives later via `take_latest` (signalled by the
    /// backend's [`Notify`]).
    fn dispatch(&self, req_id: u64, context: &str, reading: &str);

    /// Queue a request for more candidates of a conversion already made
    /// (`continue_beam` = n > 0: the engine's beam widened n steps). A
    /// backend without it converts as usual, which brings no new
    /// candidates.
    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
        let _ = continue_beam;
        self.dispatch(req_id, context, reading);
    }

    /// Take the most recent completed result, if any. Single-slot: older
    /// results are overwritten before they are taken.
    fn take_latest(&self) -> Option<EngineResult>;

    /// Whether the backend has finished loading and is serving requests.
    /// Dispatches before this is true are still accepted (buffered).
    fn is_ready(&self) -> bool;

    /// False once the backend has gone (its process exited or was killed):
    /// nothing dispatched is answered any more, and the frontend starts a
    /// new one on its next activation.
    fn is_alive(&self) -> bool {
        true
    }

    /// The engine's analysis of each of `texts`, waiting at most `timeout`
    /// (it goes before any queued conversion). `None` when the engine has
    /// no analyzer, is not ready, or does not answer in time.
    fn analyze(&self, texts: &[String], timeout: std::time::Duration) -> Option<Vec<Vec<Word>>> {
        let _ = (texts, timeout);
        None
    }
}
