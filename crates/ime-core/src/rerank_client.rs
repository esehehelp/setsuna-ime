//! Child `setsuna-rerank daemon` process client: a causal LM's log-probability
//! of each candidate after the left context (`crates/rerank`).
//!
//! As with the engine client, a writer thread sends one request at a time and
//! a reader thread delivers results through the frontend's [`Notify`]. A
//! request waits until its `not_before` time: the bridge sends one when a
//! conversion lands, due once typing has paused, and a newer request or
//! [`RerankClient::cancel`] replaces it before then.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::engine_client::Notify;

/// File name of the reranker binary on this platform.
const RERANK_EXE: &str = if cfg!(windows) {
    "setsuna-rerank.exe"
} else {
    "setsuna-rerank"
};

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
    /// The reranker's limits, once it is up.
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
    alive: AtomicBool,
}

pub struct ChildRerankClient {
    shared: Arc<Shared>,
    child: Option<Child>,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
}

impl ChildRerankClient {
    /// Spawn `setsuna-rerank daemon --manifest <manifest>` and start the I/O
    /// threads. Returns immediately; the model loads in the child.
    pub fn spawn(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::RerankCfg,
    ) -> std::io::Result<Self> {
        let mut cmd = Command::new(exe);
        let mut args: Vec<std::ffi::OsString> = vec![
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
        crate::diagnostics::log(format!("spawn rerank={} args={:?}", exe.display(), args));
        cmd.args(args);
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
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("child stdin not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("child stdout not piped"))?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            info: Mutex::new(None),
            alive: AtomicBool::new(true),
        });
        let writer = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-rerank-writer".into())
                .spawn(move || writer_loop(shared, stdin))?
        };
        let reader = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-rerank-reader".into())
                .spawn(move || {
                    reader_loop(&shared, stdout, &notify);
                    shared.alive.store(false, Ordering::SeqCst);
                    shared.cv.notify_all();
                    crate::diagnostics::log("rerank reader exit");
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
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

fn writer_loop(shared: Arc<Shared>, mut stdin: ChildStdin) {
    loop {
        let pending = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if !shared.alive.load(Ordering::SeqCst) {
                    return;
                }
                match &st.pending {
                    Some(p) if !st.in_flight => {
                        let now = Instant::now();
                        if p.not_before <= now {
                            break;
                        }
                        let wait = p.not_before - now;
                        st = shared.cv.wait_timeout(st, wait).unwrap().0;
                    }
                    _ => st = shared.cv.wait(st).unwrap(),
                }
            }
            st.in_flight = true;
            st.pending.take().unwrap()
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
        if writeln!(stdin, "{line}").is_err() || stdin.flush().is_err() {
            crate::diagnostics::log("rerank writer failed: stdin closed");
            shared.alive.store(false, Ordering::SeqCst);
            return;
        }
    }
}

fn reader_loop(shared: &Shared, stdout: ChildStdout, notify: &Notify) {
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
            notify(resp.id);
        }
    }
}

/// Spawn the reranker `cfg` names, or `None` when reranking is off (no
/// manifest), the binary is missing or it fails to start. `module_dir` is
/// the frontend module's directory, searched for the binary.
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
