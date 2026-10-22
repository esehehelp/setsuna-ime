//! Client of the login's shared `setsuna-engine serve` process (Windows):
//! one conversion model in memory for the IME of every app instead of a
//! child per app.
//!
//! The pipe's name comes from the user, the manifest, the binary and the
//! decode flags (`setsuna_pipe::engine_pipe_name`); the server is started
//! with the same flags and that name. One I/O thread connects at once
//! (starting the server when the pipe does not exist), then sends each
//! request and reads its response on the same synchronous handle, one in
//! flight at most, as the child's writer and reader do; a second thread
//! paces and posts the results, so the next request goes while one waits.
//! A pipe that broke (the server killed or gone) is connected again, the
//! server started again, and the request sent again; when that fails, the
//! client starts a child of this app's own and passes it everything from
//! then on, so conversion never stops for want of the server.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use setsuna_pipe::win::{self, Pipe};
use setsuna_pipe::{engine_pipe_name, exe_identity};

use super::{
    decode_flags, deliver, next_request, request_line, take_response, ChildDaemonClient,
    DaemonResponse, InFlight, Pending, Shared, State,
};
use crate::engine_client::{EngineClient, EngineResult, Notify};

/// How long a connection may take, the server's start and model load included.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The server to connect to, and how to start it.
struct Target {
    name: String,
    user: String,
    exe: PathBuf,
    args: Vec<OsString>,
}

/// What the I/O thread needs to start the child it falls back to.
struct Fallback {
    manifest: PathBuf,
    exe: PathBuf,
    cfg: crate::config::EngineCfg,
    notify: Arc<Mutex<Notify>>,
    /// The child, once started: every call goes to it.
    child: Mutex<Option<ChildDaemonClient>>,
}

pub struct PipeEngineClient {
    shared: Arc<Shared>,
    fallback: Arc<Fallback>,
    /// The server's pid while connected, else 0.
    server: Arc<AtomicU32>,
    io: Option<JoinHandle<()>>,
    poster: Option<JoinHandle<()>>,
}

impl PipeEngineClient {
    /// Start the I/O thread, which connects to the server for `manifest`
    /// (starting `exe serve` when there is none). Returns immediately; on
    /// an error before any thread runs, `notify` comes back with it.
    pub fn start(
        manifest: &Path,
        exe: &Path,
        notify: Notify,
        cfg: &crate::config::EngineCfg,
    ) -> Result<Self, (io::Error, Notify)> {
        let target = match target(manifest, exe, cfg) {
            Ok(t) => t,
            Err(e) => return Err((e, notify)),
        };
        crate::diagnostics::log(format!("engine pipe={} exe={}", target.name, exe.display()));
        let notify = Arc::new(Mutex::new(notify));
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
            latest: Mutex::new(None),
            ready: false.into(),
            alive: true.into(),
        });
        let fallback = Arc::new(Fallback {
            manifest: manifest.to_path_buf(),
            exe: exe.to_path_buf(),
            cfg: cfg.clone(),
            notify: notify.clone(),
            child: Mutex::new(None),
        });
        let server = Arc::new(AtomicU32::new(0));
        let (post, posts) = mpsc::channel::<(DaemonResponse, Option<InFlight>)>();
        let spawn = |name: &str, f: Box<dyn FnOnce() + Send>| {
            thread::Builder::new().name(name.into()).spawn(f)
        };
        let poster = {
            let shared = shared.clone();
            let notify: Notify = Box::new(move |id| (notify.lock().unwrap())(id));
            spawn(
                "setsuna-engine-post",
                Box::new(move || {
                    for (resp, inflight) in posts {
                        deliver(&shared, resp, inflight, &notify);
                    }
                }),
            )
        };
        let poster = match poster {
            Ok(p) => p,
            Err(e) => {
                let notify = take_notify(fallback);
                return Err((e, notify));
            }
        };
        let io = {
            let (shared, fallback, server) = (shared.clone(), fallback.clone(), server.clone());
            spawn(
                "setsuna-engine-pipe",
                Box::new(move || {
                    io_loop(&shared, &server, &target, &post, &fallback);
                    crate::diagnostics::log("engine pipe thread exit");
                }),
            )
        };
        let io = match io {
            Ok(io) => io,
            Err(e) => {
                // the poster ends with the channel, whose sender the closure held
                let _ = poster.join();
                return Err((e, take_notify(fallback)));
            }
        };
        Ok(Self {
            shared,
            fallback,
            server,
            io: Some(io),
            poster: Some(poster),
        })
    }

    /// The pid of the server connected to, while the connection lasts.
    pub fn server_pid(&self) -> Option<u32> {
        Some(self.server.load(Ordering::SeqCst)).filter(|&pid| pid != 0)
    }
}

/// The frontend's callback back out of a fallback no thread holds any more.
fn take_notify(fallback: Arc<Fallback>) -> Notify {
    let notify = Arc::try_unwrap(fallback)
        .ok()
        .and_then(|f| Arc::try_unwrap(f.notify).ok())
        .and_then(|n| n.into_inner().ok());
    notify.unwrap_or_else(|| Box::new(|_| {}))
}

/// The pipe's name and the server's command line for this config.
fn target(manifest: &Path, exe: &Path, cfg: &crate::config::EngineCfg) -> io::Result<Target> {
    let user = win::user_sid()?;
    let flags = decode_flags(cfg);
    let name = engine_pipe_name(
        &user,
        &std::fs::canonicalize(manifest)?,
        &exe_identity(exe)?,
        &flags,
    );
    let mut args: Vec<OsString> = vec![
        "serve".into(),
        "--manifest".into(),
        manifest.as_os_str().to_owned(),
    ];
    args.extend(flags.into_iter().map(OsString::from));
    args.push("--pipe".into());
    args.push(name.clone().into());
    if let Some(log) = crate::diagnostics::log_path("engine-stderr.log") {
        args.push("--log".into());
        args.push(log.into_os_string());
    }
    Ok(Target {
        name,
        user,
        exe: exe.to_path_buf(),
        args,
    })
}

impl EngineClient for PipeEngineClient {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
        self.dispatch_more(req_id, context, reading, 0);
    }

    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, continue_beam: u32) {
        if reading.is_empty() {
            return;
        }
        // the fallback's lock first: the I/O thread hands the queued
        // request to the child under it
        let child = self.fallback.child.lock().unwrap();
        if let Some(child) = child.as_ref() {
            return child.dispatch_more(req_id, context, reading, continue_beam);
        }
        crate::diagnostics::log(format!(
            "dispatch req_id={req_id} context_len={} reading={reading} continue_beam={continue_beam}",
            context.chars().count()
        ));
        self.shared.state.lock().unwrap().pending = Some(Pending {
            req_id,
            context: context.to_string(),
            reading: reading.to_string(),
            continue_beam,
            dispatched_at: Instant::now(),
        });
        drop(child);
        self.shared.cv.notify_all();
    }

    fn take_latest(&self) -> Option<EngineResult> {
        let ours = self.shared.latest.lock().ok()?.take();
        ours.or_else(|| self.fallback.child.lock().ok()?.as_ref()?.take_latest())
    }

    fn is_ready(&self) -> bool {
        match self.fallback.child.lock().unwrap().as_ref() {
            Some(child) => child.is_ready(),
            None => self.shared.ready.load(Ordering::SeqCst),
        }
    }

    fn is_alive(&self) -> bool {
        match self.fallback.child.lock().unwrap().as_ref() {
            Some(child) => child.is_alive(),
            None => self.shared.alive.load(Ordering::SeqCst),
        }
    }
}

impl Drop for PipeEngineClient {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::SeqCst);
        let mut io_done = true;
        if let Some(io) = self.io.take() {
            // the thread may be waiting for a request, reading a response or
            // waiting for the server to come up: wake it and abort its read
            let deadline = Instant::now() + Duration::from_secs(2);
            while !io.is_finished() && Instant::now() < deadline {
                drop(self.shared.state.lock());
                self.shared.cv.notify_all();
                win::cancel_io(&io);
                thread::sleep(Duration::from_millis(5));
            }
            io_done = io.is_finished();
            if io_done {
                let _ = io.join();
            } else {
                crate::diagnostics::log("engine pipe thread still busy, left behind");
            }
        }
        // the I/O thread gone, the channel is closed: the poster ends after
        // the result it is pacing, if any
        if let Some(poster) = self.poster.take() {
            if io_done {
                let _ = poster.join();
            }
        }
        // the child, if any, stops with its own Drop
        self.fallback.child.lock().unwrap().take();
    }
}

fn io_loop(
    shared: &Shared,
    server: &AtomicU32,
    target: &Target,
    post: &mpsc::Sender<(DaemonResponse, Option<InFlight>)>,
    fallback: &Fallback,
) {
    // connect at once: the server (and its model) comes up while the user
    // starts typing
    let mut conn = match connect(shared, server, target) {
        Ok(c) => Some(c),
        Err(e) => return give_up(shared, fallback, &e),
    };
    while let Some(pending) = next_request(shared) {
        let Some(mut line) = request_line(shared, &pending) else {
            continue;
        };
        line.push('\n');
        // a broken pipe is tried once more on a new connection
        let mut response = None;
        for attempt in 0..2 {
            if conn.is_none() {
                match connect(shared, server, target) {
                    Ok(c) => conn = Some(c),
                    Err(e) => {
                        // what was queued since goes to the child too
                        shared.state.lock().unwrap().pending.get_or_insert(pending);
                        return give_up(shared, fallback, &e);
                    }
                }
            }
            let reader = conn.as_mut().expect("connected above");
            match exchange(reader, &line, pending.req_id) {
                Ok(r) => {
                    response = Some(r);
                    break;
                }
                Err(e) => {
                    conn = None;
                    server.store(0, Ordering::SeqCst);
                    if !shared.alive.load(Ordering::SeqCst) {
                        return;
                    }
                    crate::diagnostics::log(format!("engine pipe lost: {e} (attempt {attempt})"));
                }
            }
        }
        match response.and_then(|line| take_response(shared, &line)) {
            Some(r) => {
                if post.send(r).is_err() {
                    return;
                }
            }
            None => {
                // nothing to post (an error, or the pipe lost twice): the
                // next request may go
                shared.state.lock().unwrap().in_flight = None;
                shared.cv.notify_all();
            }
        }
    }
}

/// The server cannot be had: start a child of this app's own and pass it
/// the queued request; every call goes to it from then on.
fn give_up(shared: &Shared, fallback: &Fallback, e: &io::Error) {
    if !shared.alive.load(Ordering::SeqCst) {
        return;
    }
    crate::diagnostics::log(format!(
        "engine pipe unusable: {e}; a child of this app's own"
    ));
    let mut child = fallback.child.lock().unwrap();
    let notify = fallback.notify.clone();
    let notify: Notify = Box::new(move |id| (notify.lock().unwrap())(id));
    match ChildDaemonClient::spawn(&fallback.manifest, &fallback.exe, notify, &fallback.cfg) {
        Ok(c) => {
            let queued = {
                let mut st = shared.state.lock().unwrap();
                st.in_flight = None;
                st.pending.take()
            };
            if let Some(p) = queued {
                c.dispatch_more(p.req_id, &p.context, &p.reading, p.continue_beam);
            }
            *child = Some(c);
        }
        Err(e) => {
            crate::diagnostics::log(format!("engine daemon spawn failed: {e}"));
            // the frontend starts a new client on its next activation
            shared.alive.store(false, Ordering::SeqCst);
            shared.cv.notify_all();
        }
    }
}

/// Open the pipe, starting the server when it does not exist, and read its
/// ready line.
fn connect(shared: &Shared, server: &AtomicU32, target: &Target) -> io::Result<BufReader<Pipe>> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let mut started = None;
    let mut nap = Duration::from_millis(20);
    loop {
        if !shared.alive.load(Ordering::SeqCst) {
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }
        match Pipe::open(&target.name) {
            Ok(pipe) => {
                // a pipe of this name owned by someone else is not our server
                let owner = pipe.owner_sid()?;
                if owner != target.user {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("{} is owned by {owner}", target.name),
                    ));
                }
                // the server sends it once its model is up
                let mut reader = BufReader::new(pipe);
                let mut line = String::new();
                if reader.read_line(&mut line)? > 0 {
                    let pid = reader.get_ref().server_pid().unwrap_or(0);
                    server.store(pid, Ordering::SeqCst);
                    shared.ready.store(true, Ordering::SeqCst);
                    crate::diagnostics::log(format!(
                        "engine pipe connected: server pid={pid} {}",
                        line.trim()
                    ));
                    return Ok(reader);
                }
                // it left before (exiting idle as we came): again
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => match started {
                None => {
                    let cwd = target.exe.parent();
                    let args: Vec<&std::ffi::OsStr> =
                        target.args.iter().map(OsString::as_os_str).collect();
                    let (pid, own_job) = win::spawn_detached(&target.exe, &args, cwd)?;
                    crate::diagnostics::log(format!(
                        "engine server started pid={pid} out_of_job={own_job} args={:?}",
                        target.args
                    ));
                    started = Some(pid);
                }
                // gone with no pipe: it could not load (or serve) the model
                Some(pid) if !win::is_running(pid) => {
                    return Err(io::Error::other(format!(
                        "the server (pid {pid}) exited without its pipe"
                    )));
                }
                Some(_) => {}
            },
            Err(e) if win::is_busy(&e) => {
                if Instant::now() >= deadline {
                    return Err(io::Error::from(io::ErrorKind::TimedOut));
                }
                win::wait(&target.name, 250);
                continue;
            }
            Err(e) => return Err(e),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::from(io::ErrorKind::TimedOut));
        }
        thread::sleep(nap);
        nap = (nap * 2).min(Duration::from_millis(250));
    }
}

/// Send one request line and read its response line.
fn exchange(reader: &mut BufReader<Pipe>, line: &str, id: u64) -> io::Result<String> {
    let mut out: &Pipe = reader.get_ref();
    out.write_all(line.as_bytes())?;
    let mut buf = String::new();
    loop {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        let text = buf.trim();
        if text.is_empty() {
            continue;
        }
        // id 0: the server could not read the request
        match serde_json::from_str::<serde_json::Value>(text) {
            Ok(v) if v["id"] == id || v["id"] == 0 => return Ok(buf),
            Ok(v) => crate::diagnostics::log(format!("engine stale response id={}", v["id"])),
            Err(e) => crate::diagnostics::log(format!("engine parse failed: {e}; line={text}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waiting_client(
        start: impl FnOnce(Notify) -> Box<dyn EngineClient>,
    ) -> (Box<dyn EngineClient>, mpsc::Receiver<u64>) {
        let (tx, rx) = mpsc::channel();
        let notify: Notify = Box::new(move |id| {
            let _ = tx.send(id);
        });
        (start(notify), rx)
    }

    fn convert(
        client: &dyn EngineClient,
        done: &mpsc::Receiver<u64>,
        id: u64,
        context: &str,
        reading: &str,
    ) -> (Vec<String>, Vec<f32>) {
        client.dispatch(id, context, reading);
        let got = done
            .recv_timeout(Duration::from_secs(60))
            .expect("a result");
        assert_eq!(got, id);
        let result = client.take_latest().expect("the result");
        assert_eq!(result.req_id, id);
        (result.candidates, result.scores)
    }

    /// Two clients of one server convert as a child daemon does, their
    /// requests interleaved; a killed server is started again.
    #[test]
    #[ignore = "needs the engine and a model: SETSUNA_ENGINE_EXE, SETSUNA_MANIFEST, SETSUNA_ORT_DYLIB"]
    fn two_clients_share_one_server() {
        let env = |k: &str| PathBuf::from(std::env::var_os(k).unwrap_or_else(|| panic!("{k}")));
        let (exe, manifest) = (env("SETSUNA_ENGINE_EXE"), env("SETSUNA_MANIFEST"));
        let cfg = crate::config::EngineCfg {
            ort_dylib: Some(env("SETSUNA_ORT_DYLIB")),
            ..Default::default()
        };
        let pipe = |n: Notify| -> Box<dyn EngineClient> {
            Box::new(
                PipeEngineClient::start(&manifest, &exe, n, &cfg)
                    .map_err(|(e, _)| e)
                    .unwrap(),
            )
        };
        let a = PipeEngineClient::start(&manifest, &exe, Box::new(|_| {}), &cfg)
            .map_err(|(e, _)| e)
            .unwrap();
        let b = PipeEngineClient::start(&manifest, &exe, Box::new(|_| {}), &cfg)
            .map_err(|(e, _)| e)
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        while (a.server_pid().is_none() || b.server_pid().is_none()) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert!(a.server_pid().is_some(), "a connected");
        assert_eq!(a.server_pid(), b.server_pid(), "one server");
        let first = a.server_pid().unwrap();
        drop((a, b));

        let (a, a_done) = waiting_client(pipe);
        let (b, b_done) = waiting_client(pipe);
        let (child, child_done) = waiting_client(|n| {
            Box::new(ChildDaemonClient::spawn(&manifest, &exe, n, &cfg).unwrap())
        });
        let requests = [
            ("", "きょうはいいてんき"),
            ("会議の", "しりょう"),
            ("今日は", "いいてんきですね。あしたは"),
            ("", "とうきょうとっきょきょかきょく"),
        ];
        let mut id = 0;
        for (context, reading) in requests {
            id += 1;
            let want = convert(child.as_ref(), &child_done, id, context, reading);
            for (client, done) in [(&a, &a_done), (&b, &b_done)] {
                id += 1;
                assert_eq!(convert(client.as_ref(), done, id, context, reading), want);
            }
        }
        // the server killed: the next request starts another one
        std::process::Command::new("taskkill")
            .args(["/F", "/PID", &first.to_string()])
            .output()
            .unwrap();
        id += 1;
        let (context, reading) = requests[0];
        let want = convert(child.as_ref(), &child_done, id, context, reading);
        id += 1;
        assert_eq!(convert(a.as_ref(), &a_done, id, context, reading), want);
    }
}
