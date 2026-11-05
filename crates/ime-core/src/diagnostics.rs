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
//! What the user types never goes in: readings, preedits, candidates and
//! contexts are logged by their length, character keys without their code.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static LOG_NAME: OnceLock<&'static str> = OnceLock::new();

/// The size past which a log moves to `<name>.1`.
pub const LOG_LIMIT: u64 = 4 << 20;

/// Turn the log on, writing to `name` in the log directory. Only the first
/// call counts.
pub fn init(name: &'static str) {
    let _ = LOG_NAME.set(name);
}

pub fn log(message: impl AsRef<str>) {
    let Some(path) = LOG_NAME.get().and_then(|name| log_path(name)) else {
        return;
    };
    let line = format!(
        "{} [{}] {}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
        std::process::id(),
        message.as_ref()
    );
    if let Some(mut file) = open_log(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// `path` opened to append, its directory made; moved to `<path>.1` first
/// when past [`LOG_LIMIT`]. For the logs of the helper processes too (their
/// stderr), which grow only while they run.
pub fn open_log(path: &Path) -> Option<File> {
    open_capped(path, LOG_LIMIT)
}

fn open_capped(path: &Path, limit: u64) -> Option<File> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let open = || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
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

pub fn log_path(name: &str) -> Option<PathBuf> {
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
}
