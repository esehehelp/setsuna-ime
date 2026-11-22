//! Shared transport, process, thread and framing helpers for the helper
//! daemon clients (engine, reranker, rewriter).

#[cfg(windows)]
pub(crate) mod pipe;

use std::ffi::OsString;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The longest line read from a daemon: a longer one (a daemon gone wrong)
/// is skipped, not held whole in memory.
pub(crate) const MAX_LINE: usize = 16 << 20;

/// How long a client's drop waits for its writer before killing the child
/// (or leaving a blocked writer behind).
pub(crate) const WRITER_GRACE: Duration = Duration::from_millis(500);

/// A line read from a daemon.
#[derive(Debug, PartialEq)]
pub(crate) enum Line {
    /// The daemon's output ended.
    Eof,
    /// A line, in the buffer given.
    Read,
    /// A line over [`MAX_LINE`] bytes, skipped to its end.
    TooLong,
}

/// Read a line into `line` (cleared first), at most `MAX_LINE` bytes of it.
pub(crate) fn read_line(reader: &mut impl BufRead, line: &mut String) -> io::Result<Line> {
    read_line_max(reader, line, MAX_LINE)
}

fn read_line_max(reader: &mut impl BufRead, line: &mut String, max: usize) -> io::Result<Line> {
    line.clear();
    let mut buf = Vec::new();
    let n = reader
        .by_ref()
        .take(max as u64 + 1)
        .read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(Line::Eof);
    }
    if buf.len() <= max || buf.last() == Some(&b'\n') {
        *line =
            String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        return Ok(Line::Read);
    }
    // the rest of the line, a chunk at a time
    loop {
        buf.clear();
        let n = reader.by_ref().take(64 << 10).read_until(b'\n', &mut buf)?;
        if n == 0 || buf.last() == Some(&b'\n') {
            return Ok(Line::TooLong);
        }
    }
}

/// Whether `line` is a daemon's ready line (`{"ready":true,...}`): anything
/// else it prints first (a library's banner) is not.
pub(crate) fn is_ready(line: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(line.trim())
        .is_ok_and(|v| v.get("ready") == Some(&serde_json::Value::Bool(true)))
}

/// Lock `m`, whatever a thread that panicked holding it left: the state is
/// plain data a panic leaves whole, and the frontend's thread must not
/// panic over another's.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Locate a helper executable via `env_var` first, then next to `module_dir`.
pub(crate) fn find_exe(
    env_var: &str,
    module_dir: Option<&Path>,
    exe_name: &str,
) -> Option<PathBuf> {
    if let Some(p) = crate::config::env_path(env_var) {
        if p.is_file() {
            return Some(p);
        }
    }
    module_dir.map(|d| d.join(exe_name)).filter(|p| p.is_file())
}

/// Log path for a helper daemon's stderr, ensuring its parent directory exists.
pub(crate) fn stderr_log_path(name: &str) -> Option<PathBuf> {
    let path = crate::diagnostics::log_path(name)?;
    if let Some(parent) = path.parent() {
        let _ = crate::private_file::create_dir_all(parent);
    }
    Some(path)
}

/// A spawned helper daemon with its piped stdin/stdout extracted.
pub(crate) struct SpawnedChild {
    pub child: Child,
    pub stdin: ChildStdin,
    pub stdout: ChildStdout,
}

/// Spawn a helper daemon with scrubbed environment, piped stdio, appended
/// stderr log, and hidden console window on Windows. Kills and reaps the child
/// if its pipes cannot be acquired.
pub(crate) fn spawn_child(
    exe: &Path,
    args: &[OsString],
    stderr_log: &str,
    label: &str,
) -> io::Result<SpawnedChild> {
    let mut cmd = Command::new(exe);
    crate::config::scrub_env(&mut cmd);
    cmd.args(args);
    let stderr = crate::diagnostics::log_path(stderr_log)
        .and_then(|path| crate::diagnostics::open_log(&path))
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
    crate::diagnostics::log(format!("{label} child pid={}", child.id()));
    #[cfg(windows)]
    match setsuna_pipe::win::place_on_largest_l3(child.id()) {
        Ok(Some(mask)) => crate::diagnostics::log(format!(
            "{label} child kept to the processors {mask:#x} (the largest L3)"
        )),
        Ok(None) => {}
        Err(e) => crate::diagnostics::log(format!("{label} child: its processors not set: {e}")),
    }
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        kill_and_wait(&mut child);
        return Err(io::Error::other("child stdio not piped"));
    };
    Ok(SpawnedChild {
        child,
        stdin,
        stdout,
    })
}

/// Kill `child` and wait for it to exit, ignoring already-exited errors.
pub(crate) fn kill_and_wait(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Spawn a named worker thread wrapped in `catch_unwind` so panics are logged
/// and `on_exit` always runs to unblock waiters.
pub(crate) fn spawn_guarded(
    thread_name: &str,
    label: &'static str,
    body: impl FnOnce() + Send + 'static,
    on_exit: impl FnOnce() + Send + 'static,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
            if res.is_err() {
                crate::diagnostics::log(format!("{label} panic"));
            }
            on_exit();
        })
}

/// Wait up to `grace` for `handle` to finish, returning whether it finished.
pub(crate) fn wait_for_thread(handle: &JoinHandle<()>, grace: Duration) -> bool {
    let deadline = Instant::now() + grace;
    while !handle.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(2));
    }
    handle.is_finished()
}

/// Gracefully stop a stdio child daemon after signalling its `alive` flag off:
/// wait up to [`WRITER_GRACE`] for the writer to drop stdin, then kill/wait the
/// child and join both I/O threads.
pub(crate) fn stop_stdio_child(
    writer: &mut Option<JoinHandle<()>>,
    child: &mut Option<Child>,
    reader: &mut Option<JoinHandle<()>>,
) {
    if let Some(w) = writer.as_ref() {
        wait_for_thread(w, WRITER_GRACE);
    }
    if let Some(mut c) = child.take() {
        kill_and_wait(&mut c);
    }
    if let Some(w) = writer.take() {
        let _ = w.join();
    }
    if let Some(r) = reader.take() {
        let _ = r.join();
    }
}

/// Write one JSONL request line and flush `stdin`.
pub(crate) fn write_json_line(stdin: &mut ChildStdin, line: &str) -> bool {
    writeln!(stdin, "{line}").is_ok() && stdin.flush().is_ok()
}

/// Read a JSONL daemon stdout stream: wait for the ready line (`is_ready`),
/// then feed each non-empty response line to `on_line`, notifying `on_too_long`
/// when an over-length post-ready line is skipped.
pub(crate) fn read_json_stream<R: BufRead>(
    reader: &mut R,
    label: &str,
    mut on_ready: impl FnMut(&str),
    mut on_too_long: impl FnMut(),
    mut on_line: impl FnMut(&str),
) {
    let mut line = String::new();
    let mut ready = false;
    loop {
        match read_line(reader, &mut line) {
            Ok(Line::Eof) => {
                crate::diagnostics::log(format!(
                    "{label} {}",
                    if ready { "EOF" } else { "handshake EOF" }
                ));
                return;
            }
            Err(e) => {
                crate::diagnostics::log(format!(
                    "{label} {} failed: {e}",
                    if ready { "read" } else { "handshake" }
                ));
                return;
            }
            Ok(Line::TooLong) => {
                crate::diagnostics::log(format!("{label} line too long, skipped"));
                if ready {
                    on_too_long();
                }
                continue;
            }
            Ok(Line::Read) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !ready {
            if is_ready(trimmed) {
                ready = true;
                // ready before the log line: a waiter must not wait on the log
                on_ready(trimmed);
                crate::diagnostics::log(format!("{label} ready: {trimmed}"));
            } else {
                crate::diagnostics::log(format!("{label} skipped a line before the ready line"));
            }
            continue;
        }
        on_line(trimmed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_too_long_is_skipped_and_the_next_one_read() {
        let text = format!("{}\nnext\n{}", "x".repeat(100), "y".repeat(10));
        let mut r = io::BufReader::new(text.as_bytes());
        let mut line = String::new();
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::TooLong);
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
        assert_eq!(line, "next\n");
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
        assert_eq!(line, "y".repeat(10));
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Eof);
        // a line of exactly the limit, its newline past it
        let text = format!("{}\n", "z".repeat(50));
        let mut r = io::BufReader::new(text.as_bytes());
        assert_eq!(read_line_max(&mut r, &mut line, 50).unwrap(), Line::Read);
    }

    #[test]
    fn only_a_ready_line_is_one() {
        assert!(is_ready("{\"ready\":true,\"version\":\"1\"}\n"));
        assert!(!is_ready("OpenMP: 16 threads\n"));
        assert!(!is_ready("{\"ready\":false}"));
        assert!(!is_ready("{\"id\":1}"));
    }

    #[test]
    fn json_stream_skips_banner_until_ready_then_delivers_lines() {
        let input = "OpenMP banner\n\n{\"ready\":true,\"v\":1}\n{\"id\":1}\n{\"id\":2}\n";
        let mut reader = io::BufReader::new(input.as_bytes());
        let mut ready_line = String::new();
        let mut lines = Vec::new();
        read_json_stream(
            &mut reader,
            "test",
            |r| ready_line = r.to_string(),
            || {},
            |l| lines.push(l.to_string()),
        );
        assert_eq!(ready_line, "{\"ready\":true,\"v\":1}");
        assert_eq!(lines, ["{\"id\":1}", "{\"id\":2}"]);
    }
}
