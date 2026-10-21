//! `setsuna-rerank serve`: one reranker for every app of a Windows login,
//! on a named pipe (`setsuna_rerank_pipe`), instead of a child per app each
//! holding its own llama.cpp context.
//!
//! The first instance of the pipe claims its name before the model loads, so
//! of the servers started at once one stays and the others exit. Each client
//! gets a thread reading its requests; one worker thread owns the reranker
//! and scores the requests of all of them in turn. Two clients interleaving
//! make its cache miss, never mix them up (what it keeps is matched token by
//! token). Idle for `idle_release`, the llama.cpp context is freed and the
//! next request makes it again; with no client for `exit_after`, the server
//! exits.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use setsuna_rerank_pipe::win::{self, Pipe};

use crate::{protocol, Reranker};

pub struct ServeOptions {
    /// From `setsuna_rerank_pipe::pipe_name`.
    pub pipe_name: String,
    /// The user the pipe is open to.
    pub user_sid: String,
    pub idle_release: Duration,
    pub exit_after: Duration,
    /// File the server's messages are appended to; else stderr (a server
    /// the IME starts has none).
    pub log: Option<PathBuf>,
}

/// One request line and where its response goes.
struct Job {
    line: String,
    reply: mpsc::Sender<protocol::Response>,
}

struct Clients {
    connected: usize,
    /// When the last one left (or the server came up).
    since: Instant,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

fn say(message: impl std::fmt::Display) {
    let line = format!("setsuna-rerank serve pid={}: {message}", std::process::id());
    match LOG.get().and_then(Option::as_ref) {
        Some(file) => {
            let _ = writeln!(lock(file), "{line}");
        }
        None => eprintln!("{line}"),
    }
}

/// Claim the pipe, load the reranker with `load` and serve its clients until
/// none has been connected for `exit_after`. Returns at once when another
/// server holds the pipe.
pub fn serve(
    opts: ServeOptions,
    load: impl FnOnce() -> Result<Reranker> + Send + 'static,
) -> Result<()> {
    let mut instance = match Pipe::create(&opts.pipe_name, true, &opts.user_sid) {
        Ok(p) => p,
        Err(e) if win::is_taken(&e) => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("create {}", opts.pipe_name)),
    };
    let log = opts.log.as_ref().and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });
    let _ = LOG.set(log.map(Mutex::new));
    let clients = Arc::new(Mutex::new(Clients {
        connected: 0,
        since: Instant::now(),
    }));
    let (jobs, queue) = mpsc::channel::<Job>();
    let (up, ready) = mpsc::channel::<Result<String>>();
    {
        let clients = clients.clone();
        let (idle_release, exit_after) = (opts.idle_release, opts.exit_after);
        thread::Builder::new()
            .name("setsuna-rerank-worker".into())
            .spawn(move || {
                let reranker = load().and_then(|r| {
                    let line = serde_json::to_string(&protocol::ready(&r.config))?;
                    Ok((r, line))
                });
                match reranker {
                    Ok((reranker, line)) => {
                        let _ = up.send(Ok(line));
                        work(reranker, &queue, &clients, idle_release, exit_after);
                    }
                    Err(e) => {
                        let _ = up.send(Err(e));
                    }
                }
            })?;
    }
    let ready: Arc<str> = match ready.recv().context("the worker thread died") {
        Ok(Ok(line)) => line.into(),
        Ok(Err(e)) | Err(e) => {
            say(format!("{e:#}"));
            return Err(e);
        }
    };
    say(format!("{} up: {ready}", opts.pipe_name));
    loop {
        match instance.accept() {
            Ok(()) => {
                lock(&clients).connected += 1;
                let (clients, jobs, ready) = (clients.clone(), jobs.clone(), ready.clone());
                thread::Builder::new()
                    .name("setsuna-rerank-client".into())
                    .spawn(move || {
                        if let Err(e) = serve_client(&instance, &ready, &jobs) {
                            say(format!("client: {e}"));
                        }
                        drop(instance);
                        let mut c = lock(&clients);
                        c.connected -= 1;
                        c.since = Instant::now();
                    })?;
            }
            Err(e) => say(format!("accept: {e}")),
        }
        // the next client's instance; the name stays ours while one is open
        instance = loop {
            match Pipe::create(&opts.pipe_name, false, &opts.user_sid) {
                Ok(p) => break p,
                Err(e) => {
                    say(format!("create an instance: {e}"));
                    thread::sleep(Duration::from_secs(1));
                }
            }
        };
    }
}

/// The ready line, then a response to each request line until the client
/// leaves.
fn serve_client(pipe: &Pipe, ready: &str, jobs: &mpsc::Sender<Job>) -> std::io::Result<()> {
    let mut out = pipe;
    out.write_all(format!("{ready}\n").as_bytes())?;
    let mut reader = BufReader::new(pipe);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        if line.trim().is_empty() {
            continue;
        }
        let (reply, response) = mpsc::channel();
        let job = Job {
            line: line.trim().to_string(),
            reply,
        };
        let gone = || std::io::Error::other("the worker thread died");
        jobs.send(job).map_err(|_| gone())?;
        let response = response.recv().map_err(|_| gone())?;
        let mut text = serde_json::to_string(&response).map_err(std::io::Error::other)?;
        text.push('\n');
        out.write_all(text.as_bytes())?;
    }
}

/// Score the clients' requests in turn; free the llama.cpp context when
/// idle, exit when no client is left.
fn work(
    mut reranker: Reranker,
    queue: &mpsc::Receiver<Job>,
    clients: &Mutex<Clients>,
    idle_release: Duration,
    exit_after: Duration,
) {
    let tick = idle_release
        .min(exit_after)
        .div_f32(2.0)
        .clamp(Duration::from_millis(50), Duration::from_secs(1));
    let mut last_request = Instant::now();
    loop {
        match queue.recv_timeout(tick) {
            Ok(job) => {
                let _ = job.reply.send(protocol::handle(&mut reranker, &job.line));
                last_request = Instant::now();
                continue;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        if last_request.elapsed() >= idle_release && reranker.release() {
            say("idle, the llama.cpp context released");
        }
        let c = lock(clients);
        if c.connected == 0 && c.since.elapsed() >= exit_after {
            // a client that connects now finds the pipe gone and starts a
            // new server
            say(format!("no client for {exit_after:?}, exiting"));
            std::process::exit(0);
        }
    }
}
