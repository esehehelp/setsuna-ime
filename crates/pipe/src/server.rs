//! A server on a pipe: one process for every app of a Windows login, the
//! model loaded once.
//!
//! The first instance of the pipe claims its name before the model loads, so
//! of the servers started at once one stays and the others exit. Each client
//! gets a thread reading its request lines; one worker thread owns the
//! [`Service`] (made by the `load` given to [`serve`], in that thread) and
//! answers the requests of all of them in turn. Idle for `idle_after`, the
//! service may free what it can make again; with no client for
//! `exit_after`, the server exits.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::win::{self, Pipe};

/// What a server answers with.
pub trait Service {
    /// The response line to one request line (no newline).
    fn handle(&mut self, line: &str) -> String;

    /// Called while no request has come for `idle_after`: free what the next
    /// request can make again. A message when something was freed.
    fn idle(&mut self) -> Option<String> {
        None
    }
}

pub struct ServeOptions {
    /// The pipe's name (`crate::rerank_pipe_name`, `crate::engine_pipe_name`).
    pub pipe_name: String,
    /// The user the pipe is open to.
    pub user_sid: String,
    pub idle_after: Duration,
    pub exit_after: Duration,
    /// File the server's messages are appended to; else stderr (a server
    /// the IME starts has none).
    pub log: Option<PathBuf>,
    /// What the messages start with, e.g. `setsuna-rerank serve`.
    pub label: &'static str,
}

/// One request line and where its response goes.
struct Job {
    line: String,
    reply: mpsc::Sender<String>,
}

struct Clients {
    connected: usize,
    /// When the last one left (or the server came up).
    since: Instant,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

static LOG: OnceLock<(&'static str, Option<Mutex<std::fs::File>>)> = OnceLock::new();

/// A message of the server: to its log file, else stderr.
pub fn say(message: impl std::fmt::Display) {
    let (label, file) = match LOG.get() {
        Some((label, file)) => (*label, file.as_ref()),
        None => ("serve", None),
    };
    let line = format!("{label} pid={}: {message}", std::process::id());
    match file {
        Some(file) => {
            let _ = writeln!(lock(file), "{line}");
        }
        None => eprintln!("{line}"),
    }
}

/// Claim the pipe, make the service with `load` (it returns the service and
/// the ready line each client is sent first) and serve the clients until
/// none has been connected for `exit_after`. Returns at once when another
/// server holds the pipe.
pub fn serve<S: Service + 'static>(
    opts: ServeOptions,
    load: impl FnOnce() -> Result<(S, String)> + Send + 'static,
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
    let _ = LOG.set((opts.label, log.map(Mutex::new)));
    let clients = Arc::new(Mutex::new(Clients {
        connected: 0,
        since: Instant::now(),
    }));
    let (jobs, queue) = mpsc::channel::<Job>();
    let (up, ready) = mpsc::channel::<Result<String>>();
    {
        let clients = clients.clone();
        let (idle_after, exit_after) = (opts.idle_after, opts.exit_after);
        thread::Builder::new()
            .name("setsuna-serve-worker".into())
            .spawn(move || match load() {
                Ok((service, line)) => {
                    let _ = up.send(Ok(line));
                    work(service, &queue, &clients, idle_after, exit_after);
                }
                Err(e) => {
                    let _ = up.send(Err(e));
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
                    .name("setsuna-serve-client".into())
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
        let mut text = response.recv().map_err(|_| gone())?;
        text.push('\n');
        out.write_all(text.as_bytes())?;
    }
}

/// Answer the clients' requests in turn; let the service free what it can
/// when idle, exit when no client is left.
fn work<S: Service>(
    mut service: S,
    queue: &mpsc::Receiver<Job>,
    clients: &Mutex<Clients>,
    idle_after: Duration,
    exit_after: Duration,
) {
    let tick = idle_after
        .min(exit_after)
        .div_f32(2.0)
        .clamp(Duration::from_millis(50), Duration::from_secs(1));
    let mut last_request = Instant::now();
    loop {
        match queue.recv_timeout(tick) {
            Ok(job) => {
                let _ = job.reply.send(service.handle(&job.line));
                last_request = Instant::now();
                continue;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        if last_request.elapsed() >= idle_after {
            if let Some(message) = service.idle() {
                say(message);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Upper-cases its requests; counts them.
    struct Upper(usize);

    impl Service for Upper {
        fn handle(&mut self, line: &str) -> String {
            self.0 += 1;
            format!("{} {}", self.0, line.to_uppercase())
        }
    }

    #[test]
    fn one_worker_answers_every_client_and_a_second_server_steps_aside() {
        let name = format!(r"\\.\pipe\setsuna-pipe-serve-test-{}", std::process::id());
        let sid = win::user_sid().unwrap();
        let opts = |name: &str| ServeOptions {
            pipe_name: name.to_string(),
            user_sid: sid.clone(),
            idle_after: Duration::from_secs(60),
            exit_after: Duration::from_secs(600),
            log: None,
            label: "test serve",
        };
        let server = {
            let opts = opts(&name);
            thread::spawn(move || serve(opts, || Ok((Upper(0), "{\"ready\":true}".to_string()))))
        };
        let connect = || loop {
            match Pipe::open(&name) {
                Ok(p) => break BufReader::new(p),
                Err(_) => thread::sleep(Duration::from_millis(10)),
            }
        };
        let (mut a, mut b) = (connect(), connect());
        let mut line = String::new();
        for r in [&mut a, &mut b] {
            line.clear();
            r.read_line(&mut line).unwrap();
            assert_eq!(line, "{\"ready\":true}\n");
        }
        let ask = |r: &mut BufReader<Pipe>, text: &str| {
            (&*r.get_ref())
                .write_all(format!("{text}\n").as_bytes())
                .unwrap();
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            line
        };
        assert_eq!(ask(&mut a, "a"), "1 A\n");
        assert_eq!(ask(&mut b, "b"), "2 B\n");
        assert_eq!(ask(&mut a, "c"), "3 C\n");
        // a second server of the name returns at once, the first one serving on
        assert!(serve(opts(&name), || -> Result<(Upper, String)> {
            unreachable!()
        })
        .is_ok());
        assert_eq!(ask(&mut b, "d"), "4 D\n");
        assert!(!server.is_finished());
    }
}
