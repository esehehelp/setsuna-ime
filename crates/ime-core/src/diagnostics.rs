//! Append-only diagnostic log of the frontend (key events, engine traffic).
//!
//! Off until the frontend names its log file with [`init`]: the TSF DLL
//! writes `tsf.log`, the fcitx5 addon `fcitx5.log`. Logs live in
//! `%APPDATA%\setsuna-ime` on Windows and `$XDG_STATE_HOME/setsuna-ime` (or
//! `~/.local/state/setsuna-ime`) elsewhere.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

static LOG_NAME: OnceLock<&'static str> = OnceLock::new();

/// Turn the log on, writing to `name` in the log directory. Only the first
/// call counts.
pub fn init(name: &'static str) {
    let _ = LOG_NAME.set(name);
}

pub fn log(message: impl AsRef<str>) {
    let Some(path) = LOG_NAME.get().and_then(|name| log_path(name)) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", message.as_ref());
    }
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
