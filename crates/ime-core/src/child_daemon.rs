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

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::engine_client::{EngineClient, EngineResult, Notify};

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
}

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
    #[serde(default)]
    error: Option<String>,
}

/// The latest request the UI queued but the writer has not sent yet.
struct Pending {
    req_id: u64,
    context: String,
    reading: String,
    continue_beam: u32,
    dispatched_at: Instant,
}

/// A request the writer has sent and is awaiting a response for.
struct InFlight {
    reading: String,
    dispatched_at: Instant,
}

#[derive(Default)]
struct State {
    pending: Option<Pending>,
    in_flight: Option<InFlight>,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    latest: Mutex<Option<EngineResult>>,
    ready: AtomicBool,
    alive: AtomicBool,
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

        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            ready: AtomicBool::new(false),
            alive: AtomicBool::new(true),
        });

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
                dispatched_at: Instant::now(),
            });
        }
        self.shared.cv.notify_all();
    }

    fn take_latest(&self) -> Option<EngineResult> {
        self.shared.latest.lock().ok()?.take()
    }

    fn is_ready(&self) -> bool {
        self.shared.ready.load(Ordering::SeqCst)
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

fn writer_loop(shared: Arc<Shared>, mut stdin: ChildStdin) {
    loop {
        let pending = {
            let mut st = shared.state.lock().unwrap();
            while shared.alive.load(Ordering::SeqCst)
                && !(st.pending.is_some() && st.in_flight.is_none())
            {
                st = shared.cv.wait(st).unwrap();
            }
            if !shared.alive.load(Ordering::SeqCst) {
                return;
            }
            let p = st.pending.take().unwrap();
            st.in_flight = Some(InFlight {
                reading: p.reading.clone(),
                dispatched_at: p.dispatched_at,
            });
            p
        };

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
        };
        crate::diagnostics::log(format!(
            "writer send req_id={} reading={}",
            pending.req_id, pending.reading
        ));
        let line = match serde_json::to_string(&req) {
            Ok(l) => l,
            Err(_) => {
                // Clear in_flight so the writer doesn't deadlock waiting
                // for a response that will never come.
                shared.state.lock().unwrap().in_flight = None;
                continue;
            }
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
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp: DaemonResponse = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                crate::diagnostics::log(format!("reader parse failed: {e}; line={trimmed}"));
                continue;
            }
        };
        crate::diagnostics::log(format!(
            "reader response id={} candidates={} error={:?}",
            resp.id,
            resp.candidates.len(),
            resp.error
        ));

        // Clear the in-flight slot and wake the writer so it can send the
        // next pending request immediately — before we pace + post, so the
        // daemon overlaps the next conversion with our pacing delay.
        let inflight = shared.state.lock().unwrap().in_flight.take();
        shared.cv.notify_all();

        if resp.error.is_some() {
            continue;
        }
        let (reading, dispatched_at) = match inflight {
            Some(f) => (f.reading, f.dispatched_at),
            None => (String::new(), Instant::now()),
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
}

/// Spawn the engine daemon with the model and binary `cfg` names or
/// discovery finds. `module_dir` is the directory the frontend module is
/// installed in (the TSF DLL's), searched last for the engine and models.
/// Returns `None` (→ raw-hiragana passthrough) when no model is found or
/// the child fails to start. `notify` is called from the reader thread for
/// each result.
pub fn spawn_with_config(
    cfg: &crate::config::Config,
    module_dir: Option<&Path>,
    notify: Notify,
) -> Option<ChildDaemonClient> {
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
    match ChildDaemonClient::spawn(&manifest, &engine_exe, notify, &cfg.engine) {
        Ok(c) => {
            tracing::info!(
                "engine daemon spawned: {} (manifest {})",
                engine_exe.display(),
                manifest.display(),
            );
            Some(c)
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
        // segment editing cuts conversions by the CTC alignment
        "--emit-alignment".into(),
    ];
    let c = engine_cfg;
    if let Some(p) = &c.ort_dylib {
        args.push(format!("--ort-dylib={}", p.display()).into());
    }
    let mut flag = |name: &str, value: Option<String>| {
        if let Some(v) = value {
            args.push(format!("--{name}={v}").into());
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
    match &c.vocab_blacklist {
        // The bare flag clears a manifest blacklist.
        Some(ids) if ids.is_empty() => args.push("--vocab-blacklist".into()),
        Some(ids) => {
            let csv = ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push(format!("--vocab-blacklist={csv}").into());
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
                "--emit-alignment"
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
