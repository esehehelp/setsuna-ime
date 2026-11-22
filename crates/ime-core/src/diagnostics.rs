//! Append-only diagnostic log of the frontend (key events, engine traffic).
//!
//! Off until the frontend names its log file with [`init`]: the TSF DLL
//! writes `tsf.log`, the fcitx5 addon `fcitx5.log`. Logs live in
//! `%APPDATA%\setsuna-ime` on Windows and `$XDG_STATE_HOME/setsuna-ime` (or
//! `~/.local/state/setsuna-ime`) elsewhere.
//!
//! Every app the IME runs in appends to the same file: a line carries the
//! local time and the process id and is written whole, in one write. Past
//! [`LOG_LIMIT`] a log moves to `<name>.1` (the one there before goes) and
//! starts again, so a log takes at most twice that.
//!
//! What the user types does not go in: readings, preedits, candidates and
//! contexts are logged by their length, character keys without their code.
//! `[diagnostics] log_typed_text` (off unless a config.toml sets it, for
//! troubleshooting on one's own machine) logs them as they are
//! ([`set_log_typed_text`], [`text_field`]); character keys stay without
//! their code even so, and a password field composes nothing, so what is
//! typed there stays out still.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static LOG_NAME: OnceLock<&'static str> = OnceLock::new();
/// Cached resolved log file path for `LOG_NAME` (its parent directory ensured
/// once on initialization) so `log()` avoids per-keystroke `env::var_os` and
/// `create_dir_all` calls.
static LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static LOG_TYPED_TEXT: AtomicBool = AtomicBool::new(false);

/// Log what the user types as it is (`[diagnostics] log_typed_text`).
pub fn set_log_typed_text(on: bool) {
    LOG_TYPED_TEXT.store(on, Ordering::Relaxed);
}

/// Whether what the user types is logged as it is.
pub fn log_typed_text() -> bool {
    LOG_TYPED_TEXT.load(Ordering::Relaxed)
}

/// `name=<text>` (quoted) when what the user types is logged, else
/// `name_len=<chars>`.
pub fn text_field(name: &str, text: &str) -> String {
    field_of(name, text, log_typed_text())
}

fn field_of(name: &str, text: &str, typed: bool) -> String {
    if typed {
        format!("{name}={text:?}")
    } else {
        format!("{name}_len={}", text.chars().count())
    }
}

/// The size past which a log moves to `<name>.1`.
pub const LOG_LIMIT: u64 = 4 << 20;

/// Turn the log on, writing to `name` in the log directory. Only the first
/// call counts.
pub fn init(name: &'static str) {
    let _ = LOG_NAME.set(name);
}

fn cached_log_path() -> Option<&'static Path> {
    let name = *LOG_NAME.get()?;
    LOG_PATH
        .get_or_init(|| {
            let path = log_path(name)?;
            if let Some(parent) = path.parent() {
                let _ = crate::private_file::create_dir_all(parent);
            }
            Some(path)
        })
        .as_deref()
}

/// The log as this process has it open: opening it for each line took a
/// quarter of a millisecond, a dozen times between a key and its
/// conversion shown; a write to it open, some microseconds. Opened again
/// (`open_capped_inner`: its size checked, moved aside past the limit) a
/// second on, so that a file another process moved to `.1` is written to
/// that long at most, and once this process's lines took it past the limit.
struct OpenLog {
    file: File,
    opened: Instant,
    len: u64,
}

static OPEN_LOG: Mutex<Option<OpenLog>> = Mutex::new(None);

const REOPEN_LOG_AFTER: Duration = Duration::from_secs(1);

pub fn log(message: impl AsRef<str>) {
    let Some(path) = cached_log_path() else {
        return;
    };
    let line = format!(
        "{} [{}] {}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
        std::process::id(),
        message.as_ref()
    );
    let mut open = OPEN_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let stale = match open.as_ref() {
        Some(o) => o.opened.elapsed() >= REOPEN_LOG_AFTER || o.len > LOG_LIMIT,
        None => true,
    };
    if stale {
        *open = open_capped_inner(path, LOG_LIMIT, false).map(|file| OpenLog {
            len: file.metadata().map_or(0, |m| m.len()),
            file,
            opened: Instant::now(),
        });
    }
    if let Some(o) = open.as_mut() {
        if o.file.write_all(line.as_bytes()).is_ok() {
            o.len += line.len() as u64;
        }
    }
}

/// `path` opened to append, its directory made; moved to `<path>.1` first
/// when past [`LOG_LIMIT`]. For the logs of the helper processes too (their
/// stderr), which grow only while they run.
pub fn open_log(path: &Path) -> Option<File> {
    open_capped(path, LOG_LIMIT)
}

fn open_capped(path: &Path, limit: u64) -> Option<File> {
    open_capped_inner(path, limit, true)
}

fn open_capped_inner(path: &Path, limit: u64, ensure_dir: bool) -> Option<File> {
    if ensure_dir {
        if let Some(parent) = path.parent() {
            let _ = crate::private_file::create_dir_all(parent);
        }
    }
    // the user's alone, as what is typed may be in it (`log_typed_text`)
    let open = || match crate::private_file::append(path) {
        Ok(f) => Some(f),
        Err(e) if !ensure_dir && e.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                let _ = crate::private_file::create_dir_all(parent);
            }
            crate::private_file::append(path).ok()
        }
        Err(_) => None,
    };
    let file = open()?;
    if file.metadata().is_ok_and(|m| m.len() > limit) {
        drop(file);
        // two processes may move it at once: the second moves the few lines
        // written since over the first's, a log's loss; one that cannot be
        // moved (open in a viewer) is appended to as it is
        let mut old = path.as_os_str().to_owned();
        old.push(".1");
        let _ = std::fs::rename(path, &old);
        return open();
    }
    Some(file)
}

/// Where log `name` goes; none in an elevated process (Windows: an
/// administrator's full token), which writes nothing into the user's
/// profile: the user's other processes can change what is there (a
/// junction put in the log's place), and the helpers it starts log nothing
/// either (their stderr goes nowhere).
pub fn log_path(name: &str) -> Option<PathBuf> {
    if crate::config::process_is_elevated() {
        return None;
    }
    if let Some(dirs) = crate::config::app_dirs() {
        return Some(dirs.state_home.join("setsuna-ime").join(name));
    }
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        env("APPDATA")
    } else {
        env("XDG_STATE_HOME").or_else(|| env("HOME").map(|h| h.join(".local").join("state")))
    };
    base.map(|b| b.join("setsuna-ime").join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_past_the_limit_moves_aside_and_starts_again() {
        let dir = std::env::temp_dir().join(format!("setsuna-diag-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("x.log");
        let write = |s: &str| {
            open_capped(&path, 10)
                .unwrap()
                .write_all(s.as_bytes())
                .unwrap()
        };
        write("0123456789\n");
        write("a\n");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\n");
        assert_eq!(
            std::fs::read_to_string(dir.join("x.log.1")).unwrap(),
            "0123456789\n"
        );
        write("b\n");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn typed_text_goes_in_only_when_asked() {
        assert_eq!(field_of("reading", "きょう", false), "reading_len=3");
        assert_eq!(field_of("reading", "きょう", true), "reading=\"きょう\"");
    }
}
