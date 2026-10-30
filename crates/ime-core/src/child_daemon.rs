//! Child `setsuna-engine daemon` process client.
//!
//! The frontend spawns the engine as a child process and talks the stdio
//! JSONL protocol (`crates/engine/src/bin/main.rs`): the model load happens
//! in the child, so activating the IME never blocks on it.
//!
//! Threading:
//!   * writer thread — owns the child's stdin; sends one request at a time.
//!   * reader thread — owns the child's stdout; parses responses, paces them,
//!     and calls the frontend's [`Notify`] (which wakes its UI thread).
//!
//! Clobbering: only the latest dispatched request matters. `dispatch`
//! overwrites a single `pending` slot, and the writer holds at most one
//! request in flight, so a burst of keystrokes collapses to one backend run.
//! An analysis (`EngineClient::analyze`) has a slot of its own that goes
//! first, and its caller waits for the answer.
//!
//!   - pipe: the same protocol with the login's shared `setsuna-engine
//!     serve` (Windows), which `spawn_with_config` prefers to a child

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::engine_client::{EngineClient, EngineResult, Notify, Word};

#[cfg(windows)]
mod pipe;
#[cfg(windows)]
pub use pipe::PipeEngineClient;

/// File name of the engine binary on this platform.
const ENGINE_EXE: &str = if cfg!(windows) {
    "setsuna-engine.exe"
} else {
    "setsuna-engine"
};

/// Dispatch→post delays a result is padded to, so the user always perceives
/// raw hiragana first and the kanji rendering after a predictable beat: the
/// shortest one the conversion has not already overrun. A conversion slower
/// than 20 ms lands at 50 ms rather than whenever it finishes; one slower
/// than the last step is posted as soon as it arrives.
const PACING_STEPS: [Duration; 2] = [Duration::from_millis(20), Duration::from_millis(50)];

/// How long to wait before posting a result `elapsed` after its dispatch.
fn pacing_wait(elapsed: Duration) -> Duration {
    PACING_STEPS
        .iter()
        .find(|step| **step >= elapsed)
        .map_or(Duration::ZERO, |step| *step - elapsed)
}

#[derive(Serialize)]
struct DaemonRequest<'a> {
    id: u64,
    context: &'a str,
    reading: &'a str,
    #[serde(skip_serializing_if = "is_zero")]
    continue_beam: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    analyze: Option<&'a [String]>,
}

/// Ids of analyses: above the bridge's conversion ids.
const ANALYZE_IDS: u64 = 1 << 62;

fn is_zero(n: &u32) -> bool {
    *n == 0
}

#[derive(Deserialize)]
struct DaemonResponse {
    #[serde(default)]
    id: u64,
    #[serde(default)]
    candidates: Vec<String>,
    #[serde(default)]
    candidate_reading_pos: Vec<Vec<u32>>,
    #[serde(default)]
    candidate_scores: Vec<f32>,
    #[serde(default)]
    engine_ms: Option<f64>,
    #[serde(default)]
    frames: Option<serde_json::Value>,
    /// An analysis's words, each text's.
    #[serde(default)]
    analyses: Option<Vec<Vec<Word>>>,
    #[serde(default)]
    error: Option<String>,
}

/// The latest request the UI queued but the writer has not sent yet.
struct Pending {
    req_id: u64,
    context: String,
    reading: String,
    continue_beam: u32,
    /// The texts of an analysis; empty for a conversion.
    analyze: Vec<String>,
    dispatched_at: Instant,
}

/// A request the writer has sent and is awaiting a response for.
struct InFlight {
    reading: String,
    /// The context as sent (NFKC).
    context: String,
    dispatched_at: Instant,
    analyze: bool,
}

#[derive(Default)]
struct State {
    pending: Option<Pending>,
    /// An analysis its caller waits for: sent before `pending`.
    analysis: Option<Pending>,
    in_flight: Option<InFlight>,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    latest: Mutex<Option<EngineResult>>,
    ready: AtomicBool,
    alive: AtomicBool,
    /// The engine was started with an analyzer.
    has_analyzer: bool,
    /// The last analysis answered: its id and words (none on an error).
    analyzed: Mutex<Option<(u64, Option<Vec<Vec<Word>>>)>>,
    analyzed_cv: Condvar,
    next_analysis: AtomicU64,
}

impl Shared {
    fn new(cfg: &crate::config::EngineCfg) -> Self {
        Self {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            ready: AtomicBool::new(false),
            alive: AtomicBool::new(true),
            has_analyzer: analyzer_of(cfg).is_some(),
            analyzed: Mutex::new(None),
            analyzed_cv: Condvar::new(),
            next_analysis: AtomicU64::new(ANALYZE_IDS),
        }
    }

    /// Queue an analysis of `texts` before any conversion and wait for it,
    /// at most `timeout`.
    fn analyze(&self, texts: &[String], timeout: Duration) -> Option<Vec<Vec<Word>>> {
        if !self.has_analyzer
            || texts.is_empty()
            || !self.ready.load(Ordering::SeqCst)
            || !self.alive.load(Ordering::SeqCst)
        {
            return None;
        }
        let id = self.next_analysis.fetch_add(1, Ordering::SeqCst);
        let deadline = Instant::now() + timeout;
        self.state.lock().unwrap().analysis = Some(Pending {
            req_id: id,
            context: String::new(),
            reading: String::new(),
            continue_beam: 0,
            analyze: texts.to_vec(),
            dispatched_at: Instant::now(),
        });
        self.cv.notify_all();
        let mut slot = self.analyzed.lock().unwrap();
        loop {
            if let Some((answered, _)) = slot.as_ref() {
                if *answered == id {
                    let words = slot.take().and_then(|(_, w)| w);
                    crate::diagnostics::log(format!(
                        "analysis id={id} texts={} words={:?} in {:.1} ms",
                        texts.len(),
                        words
                            .as_ref()
                            .map(|w| w.iter().map(Vec::len).sum::<usize>()),
                        (timeout - deadline.saturating_duration_since(Instant::now()))
                            .as_secs_f64()
                            * 1000.0
                    ));
                    // a list for each text sent
                    return words.filter(|w| w.len() == texts.len());
                }
            }
            let now = Instant::now();
            if now >= deadline || !self.alive.load(Ordering::SeqCst) {
                break;
            }
            slot = self
                .analyzed_cv
                .wait_timeout(slot, deadline - now)
                .unwrap()
                .0;
        }
        drop(slot);
        // not sent yet: it goes no more
        let mut st = self.state.lock().unwrap();
        if st.analysis.as_ref().is_some_and(|p| p.req_id == id) {
            st.analysis = None;
        }
        crate::diagnostics::log(format!("analysis id={id} unanswered in {timeout:?}"));
        None
    }
}

pub struct ChildDaemonClient {
    shared: Arc<Shared>,
    child: Option<Child>,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
}

impl ChildDaemonClient {
    /// Spawn `setsuna-engine daemon --manifest <manifest>` and start the I/O
    /// threads. Returns immediately — does NOT wait for the daemon's `Ready`.
    pub fn spawn(
        manifest: &Path,
        engine_exe: &Path,
        notify: Notify,
        engine_cfg: &crate::config::EngineCfg,
    ) -> std::io::Result<Self> {
        let mut cmd = Command::new(engine_exe);
        let args = daemon_args(manifest, engine_cfg);
        crate::diagnostics::log(format!(
            "spawn engine={} args={:?}",
            engine_exe.display(),
            args
        ));
        cmd.args(args);

        let stderr = crate::diagnostics::log_path("engine-stderr.log")
            .and_then(|path| {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok()
            })
            .map(Stdio::from)
            .unwrap_or_else(Stdio::null);

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // CREATE_NO_WINDOW: keep the daemon console off-screen.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn()?;
        crate::diagnostics::log(format!("engine child pid={}", child.id()));

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("child stdin not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("child stdout not piped"))?;

        let shared = Arc::new(Shared::new(engine_cfg));

        let writer = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-writer".into())
                .spawn(move || {
                    crate::diagnostics::log("writer thread start");
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        writer_loop(shared, stdin)
                    }));
                    if result.is_err() {
                        crate::diagnostics::log("writer thread panic");
                    }
                    crate::diagnostics::log("writer thread exit");
                })?
        };
        let reader = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-reader".into())
                .spawn(move || {
                    crate::diagnostics::log("reader thread start");
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        reader_loop(shared, stdout, notify)
                    }));
                    if result.is_err() {
                        crate::diagnostics::log("reader thread panic");
                    }
                    crate::diagnostics::log("reader thread exit");
                })?
        };

        Ok(Self {
            shared,
            child: Some(child),
            writer: Some(writer),
            reader: Some(reader),
        })
    }
}

impl EngineClient for ChildDaemonClient {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
        self.dispatch_more(req_id, context, reading, 0);
    }

    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
        if reading.is_empty() {
            return;
        }
        let context_len = context.chars().count();
        let context_tail: String = context
            .chars()
            .skip(context_len.saturating_sub(8))
            .collect();
        crate::diagnostics::log(format!(
            "dispatch req_id={req_id} context_len={context_len} context_tail={context_tail:?} reading={reading} continue_beam={continue_beam}"
        ));
        {
            let mut st = self.shared.state.lock().unwrap();
            st.pending = Some(Pending {
                req_id,
                context: context.to_string(),
                reading: reading.to_string(),
                continue_beam,
                analyze: Vec::new(),
                dispatched_at: Instant::now(),
            });
        }
        self.shared.cv.notify_all();
    }

    fn take_latest(&self) -> Option<EngineResult> {
        self.shared.latest.lock().ok()?.take()
    }

    fn analyze(&self, texts: &[String], timeout: Duration) -> Option<Vec<Vec<Word>>> {
        self.shared.analyze(texts, timeout)
    }

    fn is_ready(&self) -> bool {
        self.shared.ready.load(Ordering::SeqCst)
    }

    fn is_alive(&self) -> bool {
        self.shared.alive.load(Ordering::SeqCst)
    }
}

impl Drop for ChildDaemonClient {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::SeqCst);
        self.shared.cv.notify_all();
        // Joining the writer drops the child's stdin handle → the daemon
        // sees EOF on stdin and exits cleanly.
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        if let Some(mut child) = self.child.take() {
            // Backstop: the daemon should already be exiting on stdin EOF.
            let _ = child.kill();
            let _ = child.wait();
        }
        // The daemon's stdout is now closed → the reader's read_line ends.
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

/// Wait for a request to send (one queued, none in flight) and put it in
/// flight; `None` once the client is gone.
fn next_request(shared: &Shared) -> Option<Pending> {
    let mut st = shared.state.lock().unwrap();
    while shared.alive.load(Ordering::SeqCst)
        && !((st.pending.is_some() || st.analysis.is_some()) && st.in_flight.is_none())
    {
        st = shared.cv.wait(st).unwrap();
    }
    if !shared.alive.load(Ordering::SeqCst) {
        return None;
    }
    let p = st.analysis.take().or_else(|| st.pending.take()).unwrap();
    st.in_flight = Some(InFlight {
        reading: p.reading.clone(),
        context: p.context.nfkc().collect(),
        dispatched_at: p.dispatched_at,
        analyze: !p.analyze.is_empty(),
    });
    Some(p)
}

/// The request's line (no newline); `None`, with nothing left in flight,
/// when it cannot be written.
fn request_line(shared: &Shared, pending: &Pending) -> Option<String> {
    // The model is trained on NFKC text; the typed forms of symbols are
    // restored on the way back (postprocess), keyed on the original
    // reading kept in `in_flight`.
    let context: String = pending.context.nfkc().collect();
    let reading: String = pending.reading.nfkc().collect();
    let req = DaemonRequest {
        id: pending.req_id,
        context: &context,
        reading: &reading,
        continue_beam: pending.continue_beam,
        analyze: (!pending.analyze.is_empty()).then_some(&pending.analyze[..]),
    };
    crate::diagnostics::log(format!(
        "writer send req_id={} reading={}",
        pending.req_id, pending.reading
    ));
    match serde_json::to_string(&req) {
        Ok(l) => Some(l),
        Err(_) => {
            // Clear in_flight so the writer doesn't deadlock waiting
            // for a response that will never come.
            shared.state.lock().unwrap().in_flight = None;
            None
        }
    }
}

fn writer_loop(shared: Arc<Shared>, mut stdin: ChildStdin) {
    while let Some(pending) = next_request(&shared) {
        let Some(line) = request_line(&shared, &pending) else {
            continue;
        };
        if writeln!(stdin, "{line}").is_err() || stdin.flush().is_err() {
            crate::diagnostics::log("writer failed: stdin closed");
            shared.alive.store(false, Ordering::SeqCst);
            shared.cv.notify_all();
            return;
        }
    }
}

fn reader_loop(shared: Arc<Shared>, stdout: ChildStdout, notify: Notify) {
    reader_pump(&shared, stdout, &notify);
    // The daemon's stdout has closed — it has exited (crash, model-load
    // failure, or our own kill on Drop). Release the writer thread: if a
    // request was in flight when the daemon died, the writer is parked
    // waiting on a response that will never arrive, and without this it
    // stays stuck (and conversion silently stops) until the client drops.
    if shared.alive.load(Ordering::SeqCst) {
        tracing::warn!("Setsuna engine daemon exited unexpectedly");
    }
    shared.alive.store(false, Ordering::SeqCst);
    shared.cv.notify_all();
    crate::diagnostics::log("reader loop marked client dead");
}

fn reader_pump(shared: &Shared, stdout: ChildStdout, notify: &Notify) {
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    // The daemon's first non-empty stdout line is its `Ready` handshake.
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                crate::diagnostics::log("reader handshake EOF");
                return;
            }
            Err(e) => {
                crate::diagnostics::log(format!("reader handshake failed: {e}"));
                return;
            }
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        shared.ready.store(true, Ordering::SeqCst);
        crate::diagnostics::log(format!("reader ready: {}", line.trim()));
        break;
    }

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                crate::diagnostics::log("reader EOF");
                return;
            }
            Err(e) => {
                crate::diagnostics::log(format!("reader failed: {e}"));
                return;
            }
            Ok(_) => {}
        }
        if let Some((resp, inflight)) = take_response(shared, &line) {
            deliver(shared, resp, inflight, notify);
        }
    }
}

/// A response line read: the request it answers taken out of flight, so
/// that the next one goes at once (the engine overlaps it with this one's
/// pacing). `None` for a blank line, one that does not parse, or an error.
fn take_response(shared: &Shared, line: &str) -> Option<(DaemonResponse, Option<InFlight>)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let resp: DaemonResponse = match serde_json::from_str(trimmed) {
        Ok(r) => r,
        Err(e) => {
            crate::diagnostics::log(format!("reader parse failed: {e}; line={trimmed}"));
            return None;
        }
    };
    crate::diagnostics::log(format!(
        "reader response id={} candidates={} error={:?}",
        resp.id,
        resp.candidates.len(),
        resp.error
    ));
    let inflight = shared.state.lock().unwrap().in_flight.take();
    shared.cv.notify_all();
    if inflight.as_ref().is_some_and(|f| f.analyze) {
        // to its caller, not the frontend
        let words = resp.analyses.filter(|_| resp.error.is_none());
        *shared.analyzed.lock().unwrap() = Some((resp.id, words));
        shared.analyzed_cv.notify_all();
        return None;
    }
    if resp.error.is_some() {
        return None;
    }
    Some((resp, inflight))
}

/// Pace a result (`pacing_wait`), make it the latest and tell the frontend.
fn deliver(shared: &Shared, resp: DaemonResponse, inflight: Option<InFlight>, notify: &Notify) {
    let (reading, context, dispatched_at) = match inflight {
        Some(f) => (f.reading, f.context, f.dispatched_at),
        None => (String::new(), String::new(), Instant::now()),
    };
    let wait = pacing_wait(dispatched_at.elapsed());
    if !wait.is_zero() {
        thread::sleep(wait);
    }
    {
        let mut slot = shared.latest.lock().unwrap();
        *slot = Some(EngineResult {
            req_id: resp.id,
            reading,
            context,
            candidates: resp.candidates,
            reading_pos: resp.candidate_reading_pos,
            scores: resp.candidate_scores,
            engine_ms: resp.engine_ms,
            frames: resp.frames,
            round_trip_ms: dispatched_at.elapsed().as_secs_f64() * 1000.0,
        });
    }
    if shared.alive.load(Ordering::SeqCst) {
        crate::diagnostics::log(format!("notify req_id={}", resp.id));
        notify(resp.id);
    } else {
        crate::diagnostics::log(format!("notify skipped: client dropped req_id={}", resp.id));
    }
}

/// The engine client for the model and binary `cfg` names or discovery
/// finds. `module_dir` is the directory the frontend module is installed in
/// (the TSF DLL's), searched last for the engine and models. On Windows it
/// connects to the login's shared server (`setsuna-engine serve`, one model
/// in memory for every app), starting it when there is none, in the
/// background: this returns at once; an elevated app has a child of its own
/// (a server it started would run elevated and serve the login's other apps
/// across that boundary), as does every app elsewhere. An app in an
/// AppContainer (a Store app, the Start menu's search) only connects to the
/// server another app started: a server it started would run in the
/// container where the others cannot reach it, and a child would load the
/// model once per sandbox; the installer lets AppContainers read the
/// install directory and `config.toml`, so it computes the same pipe name.
/// Returns `None` (→ raw-hiragana passthrough) when no model is found or
/// the client fails to start. `notify` is called from a client thread for
/// each result.
pub fn spawn_with_config(
    cfg: &crate::config::Config,
    module_dir: Option<&Path>,
    notify: Notify,
) -> Option<Box<dyn EngineClient>> {
    let manifest = match resolve_manifest(&cfg.engine, module_dir, |key| {
        std::env::var_os(key).map(PathBuf::from)
    }) {
        Ok(Some(p)) => p,
        Ok(None) => {
            tracing::warn!("no model manifest found; engine disabled");
            crate::diagnostics::log("engine disabled: no model manifest");
            return None;
        }
        Err(e) => {
            tracing::warn!("model discovery failed: {e}; engine disabled");
            crate::diagnostics::log(format!("engine disabled: {e}"));
            return None;
        }
    };
    let engine_exe = cfg
        .engine
        .engine_exe
        .as_ref()
        .filter(|p| p.is_file())
        .cloned()
        .or_else(|| find_engine_exe(module_dir));
    let engine_exe = match engine_exe {
        Some(p) => p,
        None => {
            tracing::warn!("{ENGINE_EXE} not found; engine disabled");
            crate::diagnostics::log(format!("engine disabled: {ENGINE_EXE} not found"));
            return None;
        }
    };
    #[cfg(windows)]
    let notify = {
        use setsuna_pipe::win;
        if win::is_elevated().unwrap_or(false) {
            crate::diagnostics::log("engine: elevated, a child of this app's own");
            notify
        } else if win::is_app_container().unwrap_or(false) {
            return match pipe::PipeEngineClient::connect_only(
                &manifest,
                &engine_exe,
                notify,
                &cfg.engine,
            ) {
                Ok(c) => {
                    crate::diagnostics::log("engine: in an AppContainer, the shared engine only");
                    Some(Box::new(c))
                }
                Err((e, _)) => {
                    crate::diagnostics::log(format!(
                        "engine disabled: in an AppContainer, the shared engine cannot be named: {e}"
                    ));
                    None
                }
            };
        } else {
            match pipe::PipeEngineClient::start(&manifest, &engine_exe, notify, &cfg.engine) {
                Ok(c) => return Some(Box::new(c)),
                Err((e, notify)) => {
                    crate::diagnostics::log(format!(
                        "engine pipe client failed: {e}; a child of this app's own"
                    ));
                    notify
                }
            }
        }
    };
    match ChildDaemonClient::spawn(&manifest, &engine_exe, notify, &cfg.engine) {
        Ok(c) => {
            tracing::info!(
                "engine daemon spawned: {} (manifest {})",
                engine_exe.display(),
                manifest.display(),
            );
            Some(Box::new(c))
        }
        Err(e) => {
            tracing::warn!("engine daemon spawn failed: {}", e);
            crate::diagnostics::log(format!("engine daemon spawn failed: {e}"));
            None
        }
    }
}

/// Locate the model manifest. Order: config `model_manifest`, config
/// `model_dir`, `SETSUNA_MANIFEST`, `SETSUNA_MODEL_DIR`, then the model dirs
/// next to the frontend module. A directory must hold exactly one `*.manifest.json`.
fn resolve_manifest(
    cfg: &crate::config::EngineCfg,
    module_dir: Option<&Path>,
    env: impl Fn(&str) -> Option<PathBuf>,
) -> Result<Option<PathBuf>, String> {
    let file = |p: &Path, what: &str| {
        if p.is_file() {
            Ok(Some(p.to_path_buf()))
        } else {
            Err(format!("{what} {} does not exist", p.display()))
        }
    };
    if let Some(p) = &cfg.model_manifest {
        return file(p, "model_manifest");
    }
    if let Some(d) = &cfg.model_dir {
        return manifest_in_dir(d)?
            .map(Some)
            .ok_or_else(|| format!("model_dir {} holds no *.manifest.json", d.display()));
    }
    if let Some(p) = env("SETSUNA_MANIFEST") {
        return file(&p, "SETSUNA_MANIFEST");
    }
    if let Some(d) = env("SETSUNA_MODEL_DIR") {
        return manifest_in_dir(&d)?
            .map(Some)
            .ok_or_else(|| format!("SETSUNA_MODEL_DIR {} holds no *.manifest.json", d.display()));
    }
    let Some(dir) = module_dir else {
        return Ok(None);
    };
    let root = dir.join("..").join("..");
    let dirs = [
        // Installed layout (the installer puts models alongside the module).
        dir.join("models").join("onnx"),
        dir.join("onnx"),
        // Dev/cargo layout (the module in `target/release`, models in `repo/local/models/onnx`).
        root.join("local").join("models").join("onnx"),
        root.join("models").join("onnx"),
    ];
    for d in &dirs {
        if let Some(p) = manifest_in_dir(d)? {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

/// The single `*.manifest.json` in `dir`: `Ok(None)` when there is none or
/// the directory does not exist, an error when there are several.
fn manifest_in_dir(dir: &Path) -> Result<Option<PathBuf>, String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("manifest.json"))
        })
        .collect();
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.pop()),
        n => {
            found.sort();
            Err(format!(
                "{} holds {n} manifests ({}); set model_manifest to choose one",
                dir.display(),
                found
                    .iter()
                    .filter_map(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }
}

/// Locate the `setsuna-engine` daemon binary. It is a sibling of the
/// frontend module (the TSF DLL in the cargo target dir);
/// `SETSUNA_ENGINE_EXE` overrides.
fn find_engine_exe(module_dir: Option<&Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SETSUNA_ENGINE_EXE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    module_dir
        .map(|d| d.join(ENGINE_EXE))
        .filter(|p| p.is_file())
}

/// Daemon command line derived from the engine config. Only knobs the
/// config sets are passed; the rest come from the manifest's
/// `decode_defaults` and the engine defaults. Values use the `--flag=value`
/// form so clap never reads a negative number as a short flag.
fn daemon_args(manifest: &Path, engine_cfg: &crate::config::EngineCfg) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec![
        "daemon".into(),
        "--manifest".into(),
        manifest.as_os_str().to_owned(),
    ];
    args.extend(decode_flags(engine_cfg).into_iter().map(Into::into));
    args
}

/// The analyzer manifest `cfg` names, when it is set.
fn analyzer_of(cfg: &crate::config::EngineCfg) -> Option<&Path> {
    cfg.analyzer_manifest
        .as_deref()
        .filter(|p| !p.as_os_str().is_empty())
}

/// The flags after the manifest, `daemon`'s and `serve`'s: what changes the
/// engine's answers, and so what a shared server is named by.
fn decode_flags(engine_cfg: &crate::config::EngineCfg) -> Vec<String> {
    let mut args: Vec<String> = vec![
        // segment editing cuts conversions by the CTC alignment
        "--emit-alignment".into(),
        // the IME shows punctuation and full-width ASCII normalized (the
        // engine's default is the model's output as decoded)
        "--normalize-candidates".into(),
    ];
    let c = engine_cfg;
    if let Some(p) = &c.ort_dylib {
        args.push(format!("--ort-dylib={}", p.display()));
    }
    let mut flag = |name: &str, value: Option<String>| {
        if let Some(v) = value {
            args.push(format!("--{name}={v}"));
        }
    };
    flag("beam-width", c.beam_width.map(|v| v.to_string()));
    flag("top-k", c.top_k.map(|v| v.to_string()));
    flag("top-k-per-step", c.top_k_per_step.map(|v| v.to_string()));
    flag("max-context", c.max_context.map(|v| v.to_string()));
    flag("symbol-split", c.symbol_split.map(|v| v.to_string()));
    flag("symbol-token", c.symbol_token.map(|v| v.to_string()));
    flag("digit-split", c.digit_split.map(|v| v.to_string()));
    flag("emit-frames", c.emit_frames.map(|v| v.to_string()));
    flag("analyzer", analyzer_of(c).map(|p| p.display().to_string()));
    match &c.vocab_blacklist {
        // The bare flag clears a manifest blacklist.
        Some(ids) if ids.is_empty() => args.push("--vocab-blacklist".to_string()),
        Some(ids) => {
            let csv = ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push(format!("--vocab-blacklist={csv}"));
        }
        None => {}
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineCfg;

    fn args_of(cfg: &EngineCfg) -> Vec<String> {
        daemon_args(Path::new("m.manifest.json"), cfg)
            .into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn pacing_pads_to_20ms_then_50ms() {
        let ms = Duration::from_millis;
        assert_eq!(pacing_wait(ms(0)), ms(20));
        assert_eq!(pacing_wait(ms(12)), ms(8));
        assert_eq!(pacing_wait(ms(20)), ms(0));
        assert_eq!(pacing_wait(ms(21)), ms(29));
        assert_eq!(pacing_wait(ms(50)), ms(0));
        assert_eq!(pacing_wait(ms(80)), ms(0));
    }

    #[test]
    fn daemon_args_pass_only_set_knobs() {
        assert_eq!(
            args_of(&EngineCfg::default()),
            [
                "daemon",
                "--manifest",
                "m.manifest.json",
                "--emit-alignment",
                "--normalize-candidates"
            ]
        );
        let cfg = EngineCfg {
            ort_dylib: Some(PathBuf::from("ort/onnxruntime.dll")),
            beam_width: Some(20),
            symbol_split: Some(true),
            symbol_token: Some(true),
            digit_split: Some(true),
            vocab_blacklist: Some(vec![724, 730]),
            ..EngineCfg::default()
        };
        assert_eq!(
            args_of(&cfg),
            [
                "daemon",
                "--manifest",
                "m.manifest.json",
                "--emit-alignment",
                "--normalize-candidates",
                "--ort-dylib=ort/onnxruntime.dll",
                "--beam-width=20",
                "--symbol-split=true",
                "--symbol-token=true",
                "--digit-split=true",
                "--vocab-blacklist=724,730",
            ]
        );
        let cleared = EngineCfg {
            vocab_blacklist: Some(Vec::new()),
            ..EngineCfg::default()
        };
        assert_eq!(args_of(&cleared).last().unwrap(), "--vocab-blacklist");
        let analyzer = EngineCfg {
            analyzer_manifest: Some(PathBuf::from("a/analyzer.manifest.json")),
            ..EngineCfg::default()
        };
        assert_eq!(
            args_of(&analyzer).last().unwrap(),
            &format!(
                "--analyzer={}",
                Path::new("a/analyzer.manifest.json").display()
            )
        );
        let empty = EngineCfg {
            analyzer_manifest: Some(PathBuf::new()),
            ..EngineCfg::default()
        };
        assert_eq!(args_of(&empty), args_of(&EngineCfg::default()));
    }

    #[test]
    fn an_analysis_goes_first_and_returns_to_its_caller() {
        let cfg = EngineCfg {
            analyzer_manifest: Some(PathBuf::from("a.json")),
            ..EngineCfg::default()
        };
        let shared = Arc::new(Shared::new(&cfg));
        shared.ready.store(true, Ordering::SeqCst);
        shared.state.lock().unwrap().pending = Some(Pending {
            req_id: 7,
            context: String::new(),
            reading: "きょう".into(),
            continue_beam: 0,
            analyze: Vec::new(),
            dispatched_at: Instant::now(),
        });
        // the engine's side: the analysis is sent first, then the conversion
        let engine = {
            let shared = shared.clone();
            thread::spawn(move || {
                let first = next_request(&shared).unwrap();
                let line = request_line(&shared, &first).unwrap();
                let v: serde_json::Value = serde_json::from_str(&line).unwrap();
                assert_eq!(v["analyze"][0], "今日");
                let answer = format!(
                    r#"{{"id":{},"analyses":[[{{"surface":"今日","start":0,"end":2,"reading":"きょう","pos":"名詞","pos2":"普通名詞","cost":100}}]],"engine_ms":0.1}}"#,
                    first.req_id
                );
                assert!(take_response(&shared, &answer).is_none());
                let second = next_request(&shared).unwrap();
                assert_eq!((second.req_id, second.analyze.len()), (7, 0));
            })
        };
        let words = shared
            .analyze(&["今日".to_string()], Duration::from_secs(5))
            .unwrap();
        engine.join().unwrap();
        assert_eq!(words[0][0].reading.as_deref(), Some("きょう"));
        assert_eq!(words[0][0].pos2, "普通名詞");
        // no analyzer, or not ready: nothing asked
        let none = Shared::new(&EngineCfg::default());
        none.ready.store(true, Ordering::SeqCst);
        assert!(none
            .analyze(&["今日".to_string()], Duration::from_secs(1))
            .is_none());
        assert!(none.state.lock().unwrap().analysis.is_none());
        // unanswered in time: taken back
        shared.state.lock().unwrap().in_flight = Some(InFlight {
            reading: String::new(),
            context: String::new(),
            dispatched_at: Instant::now(),
            analyze: false,
        });
        assert!(shared
            .analyze(&["今日".to_string()], Duration::from_millis(20))
            .is_none());
        assert!(shared.state.lock().unwrap().analysis.is_none());
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("setsuna-discovery-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn touch(&self, rel: &str) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "{}").unwrap();
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn no_env(_: &str) -> Option<PathBuf> {
        None
    }

    #[test]
    fn explicit_manifest_and_model_dir() {
        let t = TempDir::new("explicit");
        let m = t.touch("a/model.int4.manifest.json");
        t.touch("a/model.int4.onnx");
        let cfg = EngineCfg {
            model_manifest: Some(m.clone()),
            model_dir: Some(t.0.join("elsewhere")),
            ..EngineCfg::default()
        };
        assert_eq!(resolve_manifest(&cfg, None, no_env), Ok(Some(m.clone())));
        let cfg = EngineCfg {
            model_dir: Some(t.0.join("a")),
            ..EngineCfg::default()
        };
        assert_eq!(resolve_manifest(&cfg, None, no_env), Ok(Some(m)));
        let missing = EngineCfg {
            model_manifest: Some(t.0.join("absent.manifest.json")),
            ..EngineCfg::default()
        };
        assert!(resolve_manifest(&missing, None, no_env).is_err());
    }

    #[test]
    fn model_dir_needs_exactly_one_manifest() {
        let t = TempDir::new("dir");
        let cfg = EngineCfg {
            model_dir: Some(t.0.clone()),
            ..EngineCfg::default()
        };
        assert!(resolve_manifest(&cfg, None, no_env).is_err());
        t.touch("model.fp32.manifest.json");
        assert!(resolve_manifest(&cfg, None, no_env).unwrap().is_some());
        t.touch("model.int4.manifest.json");
        let err = resolve_manifest(&cfg, None, no_env).unwrap_err();
        assert!(err.contains("2 manifests"), "{err}");
    }

    #[test]
    fn env_and_dll_relative_discovery() {
        let t = TempDir::new("env");
        let m = t.touch("env/model.manifest.json");
        let env = |key: &str| (key == "SETSUNA_MANIFEST").then(|| m.clone());
        assert_eq!(
            resolve_manifest(&EngineCfg::default(), None, env),
            Ok(Some(m.clone()))
        );
        let dir_env = |key: &str| (key == "SETSUNA_MODEL_DIR").then(|| t.0.join("env"));
        assert_eq!(
            resolve_manifest(&EngineCfg::default(), None, dir_env),
            Ok(Some(m))
        );
        let installed = t.touch("dll/models/onnx/model.int4.manifest.json");
        assert_eq!(
            resolve_manifest(&EngineCfg::default(), Some(&t.0.join("dll")), no_env),
            Ok(Some(installed))
        );
        assert_eq!(
            resolve_manifest(&EngineCfg::default(), Some(&t.0.join("empty")), no_env),
            Ok(None)
        );
    }
}
