//! Child `setsuna-rerank daemon` process client: a causal LM's log-probability
//! of each candidate after the left context (`crates/rerank`).
//!
//! As with the engine client, a writer thread sends one request at a time and
//! a reader thread delivers results through the frontend's [`Notify`]. A
//! request waits until its `not_before` time: the bridge sends one when a
//! conversion lands, due once typing has paused, and a newer request or
//! [`RerankClient::cancel`] replaces it before then.
//!
//! The daemon starts with the first request, not with the client (most apps
//! the IME is active in never convert), and stops after [`IDLE_STOP`]
//! without one, to start again with the next: each holds a few hundred MB
//! of its own, and every app the IME runs in has a client.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::engine_client::Notify;

/// File name of the reranker binary on this platform.
const RERANK_EXE: &str = if cfg!(windows) {
    "setsuna-rerank.exe"
} else {
    "setsuna-rerank"
};

/// How long the daemon stays up without a request.
pub const IDLE_STOP: Duration = Duration::from_secs(300);

/// A completed rerank: the LM log-probability of each candidate sent.
#[derive(Debug, Clone, Default)]
pub struct RerankResult {
    pub req_id: u64,
    pub logp: Vec<f64>,
    pub score_ms: f64,
}

/// What the reranker reported when it came up.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RerankInfo {
    pub max_candidates: usize,
    pub lambda: f32,
}

/// Reranking backend the `EngineBridge` sends candidate lists to.
pub trait RerankClient {
    /// Queue a request, sent no earlier than `not_before`. It replaces one
    /// not sent yet.
    fn dispatch(&self, req_id: u64, context: &str, candidates: &[String], not_before: Instant);
    /// Drop the request not sent yet, if any.
    fn cancel(&self);
    /// Take the most recent completed result, if any.
    fn take_latest(&self) -> Option<RerankResult>;
    /// The reranker's limits, once it has been up.
    fn info(&self) -> Option<RerankInfo>;
    /// False once the backend has gone; see `EngineClient::is_alive`.
    fn is_alive(&self) -> bool {
        true
    }
}

#[derive(Serialize)]
struct Request<'a> {
    id: u64,
    context: &'a str,
    candidates: &'a [String],
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    id: u64,
    #[serde(default)]
    logp: Vec<f64>,
    #[serde(default)]
    score_ms: f64,
    #[serde(default)]
    error: Option<String>,
}

struct Pending {
    req_id: u64,
    context: String,
    candidates: Vec<String>,
    not_before: Instant,
}

#[derive(Default)]
struct State {
    pending: Option<Pending>,
    in_flight: bool,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    latest: Mutex<Option<RerankResult>>,
    info: Mutex<Option<RerankInfo>>,
    /// False once the client is dropped or its daemon failed.
    alive: AtomicBool,
}

/// How the daemon is started.
struct Launch {
    exe: PathBuf,
    args: Vec<OsString>,
}

/// A running daemon.
struct Conn {
    child: Child,
    stdin: ChildStdin,
    reader: JoinHandle<()>,
    /// Set before the writer stops it, so its reader does not take the
    /// exit for a failure.
    stopping: Arc<AtomicBool>,
}

pub struct ChildRerankClient {
    shared: Arc<Shared>,
    writer: Option<JoinHandle<()>>,
}

impl ChildRerankClient {
    /// The client of `setsuna-rerank daemon --manifest <manifest>`, started
    /// with the first request. Returns immediately.
    pub fn spawn(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::RerankCfg,
    ) -> std::io::Result<Self> {
        let mut args: Vec<OsString> = vec![
            "daemon".into(),
            "--manifest".into(),
            manifest.as_os_str().to_owned(),
        ];
        if let Some(t) = cfg.threads {
            args.push(format!("--threads={t}").into());
        }
        if let Some(k) = cfg.max_candidates {
            args.push(format!("--max-candidates={k}").into());
        }
        let launch = Launch {
            exe: exe.to_path_buf(),
            args,
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            info: Mutex::new(None),
            alive: AtomicBool::new(true),
        });
        let writer = {
            let shared = shared.clone();
            let notify = Arc::new(Mutex::new(notify));
            thread::Builder::new()
                .name("setsuna-rerank-writer".into())
                .spawn(move || writer_loop(&shared, &launch, &notify))?
        };
        Ok(Self {
            shared,
            writer: Some(writer),
        })
    }
}

impl RerankClient for ChildRerankClient {
    fn dispatch(&self, req_id: u64, context: &str, candidates: &[String], not_before: Instant) {
        if candidates.is_empty() {
            return;
        }
        crate::diagnostics::log(format!(
            "rerank dispatch req_id={req_id} n={} in_ms={}",
            candidates.len(),
            not_before
                .saturating_duration_since(Instant::now())
                .as_millis()
        ));
        self.shared.state.lock().unwrap().pending = Some(Pending {
            req_id,
            context: context.to_string(),
            candidates: candidates.to_vec(),
            not_before,
        });
        self.shared.cv.notify_all();
    }

    fn cancel(&self) {
        if self.shared.state.lock().unwrap().pending.take().is_some() {
            self.shared.cv.notify_all();
        }
    }

    fn take_latest(&self) -> Option<RerankResult> {
        self.shared.latest.lock().ok()?.take()
    }

    fn info(&self) -> Option<RerankInfo> {
        *self.shared.info.lock().ok()?
    }

    fn is_alive(&self) -> bool {
        self.shared.alive.load(Ordering::SeqCst)
    }
}

impl Drop for ChildRerankClient {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::SeqCst);
        self.shared.cv.notify_all();
        // the writer stops the daemon on its way out
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

/// What the writer does next.
#[derive(Debug, PartialEq)]
enum Step {
    /// Start the daemon.
    Start,
    /// Send the request waiting.
    Send,
    /// Stop the daemon.
    Stop,
    /// Wait for a request, a response or the time given.
    Wait(Option<Duration>),
    /// The client is gone or its daemon failed.
    Exit,
}

/// The writer's next step at `now`, with a daemon `running` or not, the
/// last request sent at `last_sent`. A request queued starts the daemon at
/// once, so that its load overlaps the typing pause the request waits for.
fn next_step(alive: bool, st: &State, running: bool, last_sent: Instant, now: Instant) -> Step {
    if !alive {
        return Step::Exit;
    }
    let due = |at: Instant, then: Step| match at.checked_duration_since(now) {
        Some(wait) if !wait.is_zero() => Step::Wait(Some(wait)),
        _ => then,
    };
    match &st.pending {
        Some(_) if !running => Step::Start,
        Some(p) if !st.in_flight => due(p.not_before, Step::Send),
        None if running && !st.in_flight => due(last_sent + IDLE_STOP, Step::Stop),
        _ => Step::Wait(None),
    }
}

fn writer_loop(shared: &Arc<Shared>, launch: &Launch, notify: &Arc<Mutex<Notify>>) {
    let mut conn: Option<Conn> = None;
    let mut last_sent = Instant::now();
    loop {
        let step = {
            let mut st = shared.state.lock().unwrap();
            loop {
                let alive = shared.alive.load(Ordering::SeqCst);
                match next_step(alive, &st, conn.is_some(), last_sent, Instant::now()) {
                    Step::Wait(Some(wait)) => st = shared.cv.wait_timeout(st, wait).unwrap().0,
                    Step::Wait(None) => st = shared.cv.wait(st).unwrap(),
                    Step::Send => {
                        st.in_flight = true;
                        break st.pending.take();
                    }
                    Step::Start => {
                        drop(st);
                        match start(shared, launch, notify) {
                            Ok(c) => conn = Some(c),
                            Err(e) => {
                                crate::diagnostics::log(format!("rerank spawn failed: {e}"));
                                shared.alive.store(false, Ordering::SeqCst);
                            }
                        }
                        break None;
                    }
                    Step::Stop => {
                        drop(st);
                        crate::diagnostics::log("rerank idle: stop the daemon");
                        stop(conn.take());
                        break None;
                    }
                    Step::Exit => {
                        drop(st);
                        stop(conn.take());
                        return;
                    }
                }
            }
        };
        let (Some(pending), Some(c)) = (step, conn.as_mut()) else {
            continue;
        };
        // the engine converts NFKC text: score the context it saw
        let context: String = pending.context.nfkc().collect();
        let line = serde_json::to_string(&Request {
            id: pending.req_id,
            context: &context,
            candidates: &pending.candidates,
        });
        let Ok(line) = line else {
            shared.state.lock().unwrap().in_flight = false;
            continue;
        };
        crate::diagnostics::log(format!(
            "rerank send req_id={} n={}",
            pending.req_id,
            pending.candidates.len()
        ));
        last_sent = Instant::now();
        if writeln!(c.stdin, "{line}").is_err() || c.stdin.flush().is_err() {
            crate::diagnostics::log("rerank writer failed: stdin closed");
            shared.alive.store(false, Ordering::SeqCst);
        }
    }
}

/// Start the daemon and its reader.
fn start(
    shared: &Arc<Shared>,
    launch: &Launch,
    notify: &Arc<Mutex<Notify>>,
) -> std::io::Result<Conn> {
    crate::diagnostics::log(format!(
        "spawn rerank={} args={:?}",
        launch.exe.display(),
        launch.args
    ));
    let mut cmd = Command::new(&launch.exe);
    cmd.args(&launch.args);
    let stderr = crate::diagnostics::log_path("rerank-stderr.log")
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
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn()?;
    crate::diagnostics::log(format!("rerank child pid={}", child.id()));
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(std::io::Error::other("child stdio not piped"));
    };
    let stopping = Arc::new(AtomicBool::new(false));
    let reader = {
        let shared = shared.clone();
        let notify = notify.clone();
        let stopping = stopping.clone();
        thread::Builder::new()
            .name("setsuna-rerank-reader".into())
            .spawn(move || {
                reader_loop(&shared, stdout, &notify);
                if !stopping.load(Ordering::SeqCst) {
                    shared.alive.store(false, Ordering::SeqCst);
                    crate::diagnostics::log("rerank reader exit");
                }
                shared.state.lock().unwrap().in_flight = false;
                shared.cv.notify_all();
            })
    };
    match reader {
        Ok(reader) => Ok(Conn {
            child,
            stdin,
            reader,
            stopping,
        }),
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            Err(e)
        }
    }
}

/// Stop the daemon, if one runs, and wait for its reader.
fn stop(conn: Option<Conn>) {
    let Some(mut c) = conn else {
        return;
    };
    c.stopping.store(true, Ordering::SeqCst);
    let _ = c.child.kill();
    let _ = c.child.wait();
    let _ = c.reader.join();
}

fn reader_loop(shared: &Shared, stdout: ChildStdout, notify: &Mutex<Notify>) {
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut ready = false;
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !ready {
            ready = true;
            crate::diagnostics::log(format!("rerank ready: {trimmed}"));
            *shared.info.lock().unwrap() = serde_json::from_str(trimmed).ok();
            continue;
        }
        let resp: Response = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                crate::diagnostics::log(format!("rerank parse failed: {e}; line={trimmed}"));
                continue;
            }
        };
        shared.state.lock().unwrap().in_flight = false;
        shared.cv.notify_all();
        crate::diagnostics::log(format!(
            "rerank response id={} n={} ms={:.1} error={:?}",
            resp.id,
            resp.logp.len(),
            resp.score_ms,
            resp.error
        ));
        if resp.error.is_some() {
            continue;
        }
        *shared.latest.lock().unwrap() = Some(RerankResult {
            req_id: resp.id,
            logp: resp.logp,
            score_ms: resp.score_ms,
        });
        if shared.alive.load(Ordering::SeqCst) {
            (notify.lock().unwrap())(resp.id);
        }
    }
}

/// The client of the reranker `cfg` names, or `None` when reranking is off
/// (no manifest), the binary is missing or the client fails to start.
/// `module_dir` is the frontend module's directory, searched for the binary.
pub fn spawn_with_config(
    cfg: &crate::config::Config,
    module_dir: Option<&Path>,
    notify: Notify,
) -> Option<ChildRerankClient> {
    let c = &cfg.rerank;
    let manifest = c.manifest.as_ref()?;
    if !manifest.is_file() {
        crate::diagnostics::log(format!(
            "rerank disabled: manifest {} not found",
            manifest.display()
        ));
        return None;
    }
    let exe = c
        .exe
        .clone()
        .filter(|p| p.is_file())
        .or_else(|| find_exe(module_dir));
    let Some(exe) = exe else {
        crate::diagnostics::log(format!("rerank disabled: {RERANK_EXE} not found"));
        return None;
    };
    match ChildRerankClient::spawn(manifest, &exe, notify, c) {
        Ok(client) => Some(client),
        Err(e) => {
            crate::diagnostics::log(format!("rerank spawn failed: {e}"));
            None
        }
    }
}

fn find_exe(module_dir: Option<&Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SETSUNA_RERANK_EXE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    module_dir
        .map(|d| d.join(RERANK_EXE))
        .filter(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(not_before: Instant) -> State {
        State {
            pending: Some(Pending {
                req_id: 1,
                context: String::new(),
                candidates: vec!["a".into(), "b".into()],
                not_before,
            }),
            in_flight: false,
        }
    }

    #[test]
    fn the_daemon_starts_with_a_request_and_stops_when_idle() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let idle = State::default();
        // none runs until a request comes, then it starts before the request is due
        assert_eq!(next_step(true, &idle, false, t0, t0), Step::Wait(None));
        let queued = pending(t0 + ms(120));
        assert_eq!(next_step(true, &queued, false, t0, t0), Step::Start);
        assert_eq!(
            next_step(true, &queued, true, t0, t0),
            Step::Wait(Some(ms(120)))
        );
        assert_eq!(next_step(true, &queued, true, t0, t0 + ms(120)), Step::Send);
        // one at a time
        let busy = State {
            in_flight: true,
            ..pending(t0)
        };
        assert_eq!(next_step(true, &busy, true, t0, t0), Step::Wait(None));
        // with nothing to send, it stays up for IDLE_STOP after the last request
        let sent = t0 + ms(120);
        assert_eq!(
            next_step(true, &idle, true, sent, sent + ms(1000)),
            Step::Wait(Some(IDLE_STOP - ms(1000)))
        );
        assert_eq!(
            next_step(true, &idle, true, sent, sent + IDLE_STOP),
            Step::Stop
        );
        // never while a response is due
        let waiting = State {
            pending: None,
            in_flight: true,
        };
        assert_eq!(
            next_step(true, &waiting, true, sent, sent + IDLE_STOP),
            Step::Wait(None)
        );
        assert_eq!(next_step(false, &queued, true, t0, t0), Step::Exit);
    }
}
