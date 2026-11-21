//! Shared Windows named-pipe helpers for the shared engine and reranker
//! server clients (`child_daemon::pipe` and `rerank_client::pipe`).

use std::ffi::OsString;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use setsuna_pipe::win::{self, Pipe};

use super::{read_line, Line};

/// How long a pipe connection may take, server start and model load included.
pub(crate) const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long `Drop` retries cancelling an I/O thread before abandoning it.
const CANCEL_TIMEOUT: Duration = Duration::from_secs(2);

/// Target shared pipe server and how to start it when absent.
pub(crate) struct PipeTarget {
    pub name: String,
    pub user: String,
    pub exe: PathBuf,
    pub args: Vec<OsString>,
    /// Start the server when the pipe does not exist (`false`: connect only).
    pub start: bool,
    pub label: &'static str,
}

/// Open the named pipe, starting the server when absent (if `target.start`),
/// verify its owner SID and binary identity, and read its ready line.
pub(crate) fn connect(
    alive: &AtomicBool,
    target: &PipeTarget,
) -> io::Result<(BufReader<Pipe>, String)> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let mut started: Option<u32> = None;
    let mut nap = Duration::from_millis(20);
    loop {
        if !alive.load(Ordering::SeqCst) {
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
                // any process of the user can add an instance of the name:
                // the server must be the exe this client starts, not a
                // low-integrity one
                win::trust_server(&pipe, &target.exe)?;
                // the server sends its ready line once its model is up
                let mut reader = BufReader::new(pipe);
                let mut line = String::new();
                if read_line(&mut reader, &mut line)? == Line::Read {
                    return Ok((reader, line));
                }
                // it left before (exiting idle as we came): try again
            }
            // no server, and this client may not start one
            Err(e) if e.kind() == io::ErrorKind::NotFound && !target.start => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no server on {}", target.name),
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => match started {
                None => {
                    let cwd = target.exe.parent();
                    let args: Vec<&std::ffi::OsStr> =
                        target.args.iter().map(OsString::as_os_str).collect();
                    let env = crate::config::helper_env();
                    let (pid, own_job) =
                        win::spawn_detached(&target.exe, &args, cwd, env.as_deref())?;
                    crate::diagnostics::log(format!(
                        "{} server started pid={pid} out_of_job={own_job} args={:?}",
                        target.label, target.args
                    ));
                    started = Some(pid);
                }
                // gone with no pipe: it could not load (or serve) the model
                Some(pid) if !win::is_running(pid) => {
                    return Err(io::Error::other(format!(
                        "the {} server (pid {pid}) exited without its pipe",
                        target.label
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

/// Write one JSONL request line and read response lines until `parse` returns
/// `Some(response)` for the matching request.
pub(crate) fn exchange<T>(
    reader: &mut BufReader<Pipe>,
    line: &str,
    label: &str,
    mut parse: impl FnMut(&str) -> Option<T>,
) -> io::Result<T> {
    let mut out: &Pipe = reader.get_ref();
    out.write_all(line.as_bytes())?;
    let mut buf = String::new();
    loop {
        match read_line(reader, &mut buf)? {
            Line::Eof => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Line::TooLong => {
                crate::diagnostics::log(format!("{label} response line too long, skipped"));
                continue;
            }
            Line::Read => {}
        }
        let text = buf.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(parsed) = parse(text) {
            return Ok(parsed);
        }
    }
}

/// Wake and cancel a blocked pipe I/O thread during `Drop`, joining it if it
/// finishes within [`CANCEL_TIMEOUT`]. Returns whether the thread finished.
pub(crate) fn stop_io_thread(io: JoinHandle<()>, wake: impl Fn(), label: &str) -> bool {
    let deadline = Instant::now() + CANCEL_TIMEOUT;
    while !io.is_finished() && Instant::now() < deadline {
        wake();
        win::cancel_io(&io);
        thread::sleep(Duration::from_millis(5));
    }
    if io.is_finished() {
        let _ = io.join();
        true
    } else {
        crate::diagnostics::log(format!("{label} pipe thread still busy, left behind"));
        false
    }
}
