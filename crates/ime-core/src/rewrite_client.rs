//! Child `setsuna-rewrite daemon` process client: a model rewriting the
//! conversion into natural Japanese (`crates/rewrite`).
//!
//! As with the reranker, a writer thread sends one request at a time (a
//! newer request replaces one not sent yet) and a reader thread delivers
//! results, failures included, through the frontend's [`Notify`].

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use serde::{Deserialize, Serialize};

use crate::config::RewriterCfg;
use crate::engine_client::Notify;

/// File name of the rewriter binary on this platform.
const REWRITE_EXE: &str = if cfg!(windows) {
    "setsuna-rewrite.exe"
} else {
    "setsuna-rewrite"
};

/// A finished rewrite: the text, or why there is none.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RewriteResult {
    pub req_id: u64,
    pub text: Option<String>,
    /// The failure's code (`usage_limit`, `not_signed_in`, ...).
    pub code: Option<String>,
    pub ms: f64,
}

/// Rewriting backend the `EngineBridge` sends conversions to.
pub trait RewriteClient {
    /// Queue a request; it replaces one not sent yet.
    fn dispatch(&self, req_id: u64, context: &str, reading: &str, text: &str);
    /// Drop the request not sent yet, if any.
    fn cancel(&self);
    /// Take the most recent finished rewrite, if any.
    fn take_latest(&self) -> Option<RewriteResult>;
}

#[derive(Serialize)]
struct Request<'a> {
    id: u64,
    context: &'a str,
    reading: &'a str,
    text: &'a str,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    id: u64,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    ms: f64,
}

struct Pending {
    req_id: u64,
    context: String,
    reading: String,
    text: String,
}

#[derive(Default)]
struct State {
    pending: Option<Pending>,
    in_flight: bool,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    latest: Mutex<Option<RewriteResult>>,
    alive: AtomicBool,
}

pub struct ChildRewriteClient {
    shared: Arc<Shared>,
    send_context: bool,
    child: Option<Child>,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
}

/// The daemon's command line for `cfg`, its credentials in `credentials_dir`.
fn daemon_args(cfg: &RewriterCfg, credentials_dir: &Path) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec![
        "daemon".into(),
        format!("--provider={}", cfg.provider.as_str()).into(),
        format!("--model={}", cfg.model).into(),
        format!("--timeout-ms={}", cfg.timeout_ms).into(),
        "--credentials-dir".into(),
        credentials_dir.as_os_str().to_owned(),
    ];
    if let Some(u) = cfg.base_url.as_deref().filter(|u| !u.is_empty()) {
        args.push(format!("--base-url={u}").into());
    }
    if let Some(a) = cfg.api {
        args.push(format!("--api={}", a.as_str()).into());
    }
    if let Some(i) = cfg.instructions.as_deref().filter(|i| !i.trim().is_empty()) {
        args.push(format!("--instructions={i}").into());
    }
    if let Some(e) = cfg.reasoning_effort.as_deref().filter(|e| !e.is_empty()) {
        args.push(format!("--reasoning-effort={e}").into());
    }
    args
}

impl ChildRewriteClient {
    /// Spawn `setsuna-rewrite daemon` and start the I/O threads.
    pub fn spawn(
        cfg: &RewriterCfg,
        exe: &Path,
        credentials_dir: &Path,
        notify: Notify,
    ) -> std::io::Result<Self> {
        let mut cmd = Command::new(exe);
        let args = daemon_args(cfg, credentials_dir);
        crate::diagnostics::log(format!(
            "spawn rewrite={} provider={} model={}",
            exe.display(),
            cfg.provider.as_str(),
            cfg.model
        ));
        cmd.args(args);
        let stderr = crate::diagnostics::log_path("rewrite-stderr.log")
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
        crate::diagnostics::log(format!("rewrite child pid={}", child.id()));
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
            alive: AtomicBool::new(true),
        });
        let writer = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-rewrite-writer".into())
                .spawn(move || writer_loop(shared, stdin))?
        };
        let reader = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("setsuna-rewrite-reader".into())
                .spawn(move || {
                    reader_loop(&shared, stdout, &notify);
                    shared.alive.store(false, Ordering::SeqCst);
                    shared.cv.notify_all();
                    crate::diagnostics::log("rewrite reader exit");
                })?
        };
        Ok(Self {
            shared,
            send_context: cfg.send_context,
            child: Some(child),
            writer: Some(writer),
            reader: Some(reader),
        })
    }
}

impl RewriteClient for ChildRewriteClient {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str, text: &str) {
        if text.is_empty() {
            return;
        }
        crate::diagnostics::log(format!("rewrite dispatch req_id={req_id}"));
        self.shared.state.lock().unwrap().pending = Some(Pending {
            req_id,
            context: if self.send_context {
                context.to_string()
            } else {
                String::new()
            },
            reading: reading.to_string(),
            text: text.to_string(),
        });
        self.shared.cv.notify_all();
    }

    fn cancel(&self) {
        if self.shared.state.lock().unwrap().pending.take().is_some() {
            self.shared.cv.notify_all();
        }
    }

    fn take_latest(&self) -> Option<RewriteResult> {
        self.shared.latest.lock().ok()?.take()
    }
}

impl Drop for ChildRewriteClient {
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
            // wait for a request, and for the one before it to be answered
            while shared.alive.load(Ordering::SeqCst) && (st.pending.is_none() || st.in_flight) {
                st = shared.cv.wait(st).unwrap();
            }
            if !shared.alive.load(Ordering::SeqCst) {
                return;
            }
            st.in_flight = true;
            st.pending.take().unwrap()
        };
        let line = serde_json::to_string(&Request {
            id: pending.req_id,
            context: &pending.context,
            reading: &pending.reading,
            text: &pending.text,
        });
        let Ok(line) = line else {
            shared.state.lock().unwrap().in_flight = false;
            continue;
        };
        if writeln!(stdin, "{line}").is_err() || stdin.flush().is_err() {
            crate::diagnostics::log("rewrite writer failed: stdin closed");
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
            crate::diagnostics::log(format!("rewrite ready: {trimmed}"));
            continue;
        }
        let resp: Response = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                crate::diagnostics::log(format!("rewrite parse failed: {e}"));
                continue;
            }
        };
        shared.state.lock().unwrap().in_flight = false;
        shared.cv.notify_all();
        // the text is the user's: only the outcome goes to the log
        crate::diagnostics::log(format!(
            "rewrite response id={} ms={:.0} code={:?} error={:?}",
            resp.id, resp.ms, resp.code, resp.error
        ));
        let code = match (&resp.text, resp.code) {
            (Some(_), _) => None,
            (None, c) => Some(c.unwrap_or_else(|| "error".into())),
        };
        *shared.latest.lock().unwrap() = Some(RewriteResult {
            req_id: resp.id,
            text: resp.text,
            code,
            ms: resp.ms,
        });
        if shared.alive.load(Ordering::SeqCst) {
            notify(resp.id);
        }
    }
}

/// Spawn the rewriter `cfg` describes, or `None` when it is off, the binary
/// is missing or it fails to start. `module_dir` is the frontend module's
/// directory, searched for the binary.
pub fn spawn_with_config(
    cfg: &crate::config::Config,
    module_dir: Option<&Path>,
    notify: Notify,
) -> Option<ChildRewriteClient> {
    let c = &cfg.rewriter;
    if !c.enabled {
        return None;
    }
    let Some(dir) = crate::config::config_dir() else {
        crate::diagnostics::log("rewrite disabled: no config directory");
        return None;
    };
    let exe = c
        .exe
        .clone()
        .filter(|p| p.is_file())
        .or_else(|| find_exe(module_dir));
    let Some(exe) = exe else {
        crate::diagnostics::log(format!("rewrite disabled: {REWRITE_EXE} not found"));
        return None;
    };
    match ChildRewriteClient::spawn(c, &exe, &dir, notify) {
        Ok(client) => Some(client),
        Err(e) => {
            crate::diagnostics::log(format!("rewrite spawn failed: {e}"));
            None
        }
    }
}

fn find_exe(module_dir: Option<&Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SETSUNA_REWRITE_EXE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    module_dir
        .map(|d| d.join(REWRITE_EXE))
        .filter(|p| p.is_file())
}

/// What the IME shows for a failed rewrite's `code`.
pub fn failure_message(code: &str) -> &'static str {
    match code {
        "not_signed_in" | "signed_out" => "書き直し: ChatGPT にサインインしてください（設定）",
        "no_plan_usage" => "書き直し: ChatGPT プランの利用が許可されていません",
        "no_api_key" => "書き直し: API キーが設定されていません",
        "no_model" => "書き直し: モデルが選ばれていません（設定）",
        "usage_limit" => "書き直し: 利用上限に達しました（ChatGPT の設定 > 使用量）",
        "not_eligible" => "書き直し: このアカウントではプランを利用できません",
        "unavailable" => "書き直し: 一時的に利用できません",
        "unsupported" | "not_found" => "書き直し: このモデルでは使えません",
        "auth" | "forbidden" => "書き直し: 認証に失敗しました",
        "network" => "書き直し: 接続できませんでした",
        "incomplete" => "書き直し: 応答が途中で終わりました",
        _ => "書き直しに失敗しました",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RewriteApi, RewriteProvider};

    #[test]
    fn daemon_args_pass_only_what_is_set() {
        let args = |cfg: &RewriterCfg| -> Vec<String> {
            daemon_args(cfg, Path::new("C:/cfg"))
                .into_iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect()
        };
        let mut cfg = RewriterCfg {
            model: "gpt-x".into(),
            ..RewriterCfg::default()
        };
        assert_eq!(
            args(&cfg),
            [
                "daemon",
                "--provider=chatgpt",
                "--model=gpt-x",
                "--timeout-ms=20000",
                "--credentials-dir",
                "C:/cfg"
            ]
        );
        cfg.provider = RewriteProvider::Compatible;
        cfg.base_url = Some("http://127.0.0.1:8080/v1".into());
        cfg.api = Some(RewriteApi::Responses);
        cfg.instructions = Some("  ".into());
        cfg.reasoning_effort = Some("low".into());
        let a = args(&cfg);
        assert!(a.contains(&"--base-url=http://127.0.0.1:8080/v1".to_string()));
        assert!(a.contains(&"--api=responses".to_string()));
        assert!(a.contains(&"--reasoning-effort=low".to_string()));
        assert!(!a.iter().any(|x| x.starts_with("--instructions")));
    }
}
