//! Client of the login's shared `setsuna-rerank serve` process (Windows):
//! one reranker for the IME of every app instead of a child per app.
//!
//! The pipe's name comes from the user, the manifest, the binary and the
//! options (`setsuna_pipe::rerank_pipe_name`), as the server computes it. One
//! I/O thread connects, starting the server when the pipe does not exist, then
//! sends each request due and reads its response on the same synchronous
//! handle (one request in flight at most). A pipe that broke (the server
//! killed or gone) is connected again for the next request; failing that,
//! reranking waits a while before trying again, never falling back to a child
//! of its own. In an AppContainer it never starts the server either
//! ([`PipeRerankClient::connect_only`]): one started there would run in the
//! sandbox, out of the other apps' reach.

use std::ffi::OsString;
use std::io::{self, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use setsuna_pipe::win::{self, Pipe};
use setsuna_pipe::{exe_identity, rerank_pipe_name, RerankKey};

use crate::daemon_io::lock;
use crate::daemon_io::pipe::{self as shared_pipe, PipeTarget};

use super::{
    next_due, request_line, stderr_log_path, Pending, RerankClient, RerankInfo, RerankResult,
    Response, Shared, State,
};
use crate::engine_client::Notify;

/// Wait after a failed connection before the next try, doubling up to the max.
const RETRY_MIN: Duration = Duration::from_secs(5);
const RETRY_MAX: Duration = Duration::from_secs(300);

pub struct PipeRerankClient {
    shared: Arc<Shared>,
    /// The server's pid while connected, else 0.
    server: Arc<AtomicU32>,
    io: Option<JoinHandle<()>>,
}

impl PipeRerankClient {
    /// Start the I/O thread, which connects to the server for `manifest`
    /// (starting `exe serve` when there is none). Returns immediately.
    pub fn start(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::RerankCfg,
    ) -> io::Result<Self> {
        Self::open(manifest, exe, notify, cfg, true)
    }

    /// The client of a server some other app started, never starting one.
    pub fn connect_only(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::RerankCfg,
    ) -> io::Result<Self> {
        Self::open(manifest, exe, notify, cfg, false)
    }

    fn open(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::RerankCfg,
        start: bool,
    ) -> io::Result<Self> {
        let user = win::user_sid()?;
        let canonical = std::fs::canonicalize(manifest)?;
        let name = rerank_pipe_name(&RerankKey {
            user: &user,
            manifest: &canonical,
            exe: &exe_identity(exe)?,
            threads: cfg.threads,
            max_candidates: cfg.max_candidates,
            max_context_chars: None,
            lambda: None,
        });
        let mut args: Vec<OsString> = vec![
            "serve".into(),
            "--manifest".into(),
            manifest.as_os_str().to_owned(),
        ];
        if let Some(t) = cfg.threads {
            args.push(format!("--threads={t}").into());
        }
        if let Some(k) = cfg.max_candidates {
            args.push(format!("--max-candidates={k}").into());
        }
        if let Some(log) = stderr_log_path() {
            args.push("--log".into());
            args.push(log.into_os_string());
        }
        crate::diagnostics::log(format!("rerank pipe={name} exe={}", exe.display()));
        let target = PipeTarget {
            name,
            user,
            exe: exe.to_path_buf(),
            args,
            start,
            label: "rerank",
        };
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            info: Mutex::new(None),
            alive: true.into(),
        });
        let server = Arc::new(AtomicU32::new(0));
        let io = {
            let (shared, server) = (shared.clone(), server.clone());
            thread::Builder::new()
                .name("setsuna-rerank-pipe".into())
                .spawn(move || {
                    io_loop(&shared, &server, &target, &notify);
                    crate::diagnostics::log("rerank pipe thread exit");
                })?
        };
        Ok(Self {
            shared,
            server,
            io: Some(io),
        })
    }

    /// The pid of the server last connected to, while the connection lasts.
    pub fn server_pid(&self) -> Option<u32> {
        Some(self.server.load(Ordering::SeqCst)).filter(|&pid| pid != 0)
    }
}

impl RerankClient for PipeRerankClient {
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
        lock(&self.shared.state).pending = Some(Pending {
            req_id,
            context: context.to_string(),
            candidates: candidates.to_vec(),
            not_before,
        });
        self.shared.cv.notify_all();
    }

    fn cancel(&self) {
        if lock(&self.shared.state).pending.take().is_some() {
            self.shared.cv.notify_all();
        }
    }

    fn take_latest(&self) -> Option<RerankResult> {
        lock(&self.shared.latest).take()
    }

    fn info(&self) -> Option<RerankInfo> {
        *lock(&self.shared.info)
    }
}

impl Drop for PipeRerankClient {
    fn drop(&mut self) {
        self.shared.stop();
        if let Some(io) = self.io.take() {
            shared_pipe::stop_io_thread(
                io,
                || {
                    drop(self.shared.state.lock());
                    self.shared.cv.notify_all();
                },
                "rerank",
            );
        }
    }
}

fn io_loop(shared: &Shared, server: &AtomicU32, target: &PipeTarget, notify: &Notify) {
    let mut conn = None;
    let mut retry_at = Instant::now();
    let mut retry_wait = RETRY_MIN;
    // connect at once: the server (and its model) comes up while the user
    // types, and info() is known before the first rerank
    let mut try_connect = |conn: &mut Option<BufReader<Pipe>>| {
        if Instant::now() < retry_at {
            return;
        }
        match connect(shared, target) {
            Ok((reader, info)) => {
                let pid = reader.get_ref().server_pid().unwrap_or(0);
                server.store(pid, Ordering::SeqCst);
                crate::diagnostics::log(format!(
                    "rerank pipe connected: server pid={pid} {info:?}"
                ));
                if info.is_some() {
                    *lock(&shared.info) = info;
                }
                *conn = Some(reader);
                retry_wait = RETRY_MIN;
            }
            Err(e) => {
                if shared.alive.load(Ordering::SeqCst) {
                    crate::diagnostics::log(format!(
                        "rerank pipe connect failed: {e}; next try in {retry_wait:?}"
                    ));
                }
                retry_at = Instant::now() + retry_wait;
                retry_wait = (retry_wait * 2).min(RETRY_MAX);
            }
        }
    };
    try_connect(&mut conn);
    loop {
        let Some(pending) = next_due(shared) else {
            return;
        };
        let response = match request_line(&pending) {
            Ok(mut line) => {
                line.push('\n');
                crate::diagnostics::log(format!(
                    "rerank send req_id={} n={}",
                    pending.req_id,
                    pending.candidates.len()
                ));
                // a broken pipe is tried once more on a new connection
                let mut response = None;
                for _ in 0..2 {
                    if conn.is_none() {
                        try_connect(&mut conn);
                    }
                    let Some(reader) = conn.as_mut() else {
                        crate::diagnostics::log("rerank skipped: no server");
                        break;
                    };
                    match exchange(reader, &line, pending.req_id) {
                        Ok(r) => {
                            response = Some(r);
                            break;
                        }
                        Err(e) => {
                            conn = None;
                            server.store(0, Ordering::SeqCst);
                            if !shared.alive.load(Ordering::SeqCst) {
                                break;
                            }
                            crate::diagnostics::log(format!("rerank pipe lost: {e}"));
                        }
                    }
                }
                response
            }
            Err(_) => None,
        };
        lock(&shared.state).in_flight = false;
        shared.cv.notify_all();
        let Some(resp) = response else {
            continue;
        };
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
        *lock(&shared.latest) = Some(RerankResult {
            req_id: resp.id,
            logp: resp.logp,
            score_ms: resp.score_ms,
        });
        if shared.alive.load(Ordering::SeqCst) {
            notify(resp.id);
        }
    }
}

/// Open the pipe, starting the server when it does not exist, and read its
/// ready line: the connection and what the server reported.
fn connect(
    shared: &Shared,
    target: &PipeTarget,
) -> io::Result<(BufReader<Pipe>, Option<RerankInfo>)> {
    let (reader, line) = shared_pipe::connect(&shared.alive, target)?;
    Ok((reader, serde_json::from_str(line.trim()).ok()))
}

/// Send one request line and read its response.
fn exchange(reader: &mut BufReader<Pipe>, line: &str, id: u64) -> io::Result<Response> {
    shared_pipe::exchange(reader, line, "rerank", |text| {
        match serde_json::from_str::<Response>(text) {
            // id 0: the server could not read the request
            Ok(resp) if resp.id == id || resp.id == 0 => Some(resp),
            Ok(resp) => {
                crate::diagnostics::log(format!("rerank stale response id={}", resp.id));
                None
            }
            Err(e) => {
                crate::diagnostics::log(format!(
                    "rerank parse failed: {e}; {}",
                    crate::diagnostics::text_field("line", text)
                ));
                None
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::ChildRerankClient;
    use super::*;
    use std::path::PathBuf;
    use std::sync::mpsc;

    fn waiting_client(
        start: impl FnOnce(Notify) -> io::Result<Box<dyn RerankClient>>,
    ) -> (Box<dyn RerankClient>, mpsc::Receiver<u64>) {
        let (tx, rx) = mpsc::channel();
        let notify: Notify = Box::new(move |id| {
            let _ = tx.send(id);
        });
        (start(notify).unwrap(), rx)
    }

    fn score(
        client: &dyn RerankClient,
        done: &mpsc::Receiver<u64>,
        id: u64,
        context: &str,
        candidates: &[String],
    ) -> Vec<f64> {
        client.dispatch(id, context, candidates, Instant::now());
        let got = done
            .recv_timeout(Duration::from_secs(60))
            .expect("a result");
        assert_eq!(got, id);
        let result = client.take_latest().expect("the result");
        assert_eq!(result.req_id, id);
        result.logp
    }

    /// Two clients of one server score as the stdio daemon does, their
    /// requests interleaved (each other's cache never leaks into a score).
    #[test]
    #[ignore = "needs the reranker and a model: SETSUNA_RERANK_EXE, SETSUNA_RERANK_MANIFEST"]
    fn two_clients_share_one_server() {
        let exe =
            PathBuf::from(std::env::var_os("SETSUNA_RERANK_EXE").expect("SETSUNA_RERANK_EXE"));
        let manifest = PathBuf::from(
            std::env::var_os("SETSUNA_RERANK_MANIFEST").expect("SETSUNA_RERANK_MANIFEST"),
        );
        let cfg = crate::config::RerankCfg {
            threads: Some(2),
            ..Default::default()
        };
        let pipe = |n: Notify| {
            PipeRerankClient::start(&manifest, &exe, n, &cfg)
                .map(|c| Box::new(c) as Box<dyn RerankClient>)
        };
        let a = PipeRerankClient::start(&manifest, &exe, Box::new(|_| {}), &cfg).unwrap();
        let b = PipeRerankClient::start(&manifest, &exe, Box::new(|_| {}), &cfg).unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        while (a.server_pid().is_none() || b.server_pid().is_none()) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert!(a.server_pid().is_some(), "a connected");
        assert_eq!(a.server_pid(), b.server_pid(), "one server");
        drop((a, b));

        let (a, a_done) = waiting_client(pipe);
        let (b, b_done) = waiting_client(pipe);
        let (child, child_done) = waiting_client(|n| {
            ChildRerankClient::spawn(&manifest, &exe, n, &cfg)
                .map(|c| Box::new(c) as Box<dyn RerankClient>)
        });
        let s = |v: &[&str]| v.iter().map(|c| c.to_string()).collect::<Vec<_>>();
        let requests = [
            ("今日はいい", s(&["天気", "転機", "電気", "天気です"])),
            ("会議の", s(&["資料", "飼料", "試料", "史料", "思料"])),
            (
                "今日はいい",
                s(&["天気", "転機", "電気", "天気です", "転記", "伝記"]),
            ),
            ("", s(&["こんにちは", "今日は"])),
            ("会議の", s(&["資料", "飼料"])),
        ];
        let mut id = 0;
        let mut worst = 0.0f64;
        for (context, candidates) in &requests {
            id += 1;
            let want = score(child.as_ref(), &child_done, id, context, candidates);
            for (client, done) in [(&a, &a_done), (&b, &b_done)] {
                id += 1;
                let got = score(client.as_ref(), done, id, context, candidates);
                assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    worst = worst.max((g - w).abs());
                }
            }
        }
        eprintln!("largest |pipe - daemon| logp difference: {worst:e}");
        assert!(worst < 1e-3, "{worst}");
    }
}
