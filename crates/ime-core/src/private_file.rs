//! The files of what the user typed or chose (the learning, the prediction
//! history, the hidden candidates, the user dictionary) are the user's
//! alone: on unix 0600 when made, whatever the umask would let others read
//! (0644 by default), and an older one made so is narrowed when written
//! again. On Windows they stay as the profile directory's ACL makes them
//! (the user, SYSTEM and Administrators).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

#[cfg(unix)]
fn narrow(file: &File) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = file.metadata()?.permissions().mode();
    if mode & 0o077 != 0 {
        file.set_permissions(std::fs::Permissions::from_mode(mode & 0o700))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn narrow(_: &File) -> std::io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn options() -> OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;
    let mut o = OpenOptions::new();
    o.mode(0o600);
    o
}

#[cfg(not(unix))]
fn options() -> OpenOptions {
    OpenOptions::new()
}

/// `path` opened to append to, made if missing.
pub fn append(path: &Path) -> std::io::Result<File> {
    let file = options().create(true).append(true).open(path)?;
    narrow(&file)?;
    Ok(file)
}

/// The lock of `path` (`<name>.lock` beside it), which every process
/// appending to `path` or replacing it takes.
fn lock_file(path: &Path) -> std::io::Result<File> {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".lock");
    options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path.with_file_name(name))
}

/// `line` appended to `path` (made if missing) under its lock, so that no
/// compaction replaces the file between another's check and its move.
pub fn append_locked(path: &Path, line: &[u8]) -> std::io::Result<()> {
    let lock = lock_file(path)?;
    lock.lock()?;
    let written = append(path).and_then(|mut f| f.write_all(line));
    let _ = lock.unlock();
    written
}

/// `f` run under `path`'s lock when no other process holds it (`None`
/// when one does): a compaction's check that nothing was appended and its
/// replacing the file are then one step to every appender.
pub fn try_locked<T>(
    path: &Path,
    f: impl FnOnce() -> std::io::Result<T>,
) -> std::io::Result<Option<T>> {
    let lock = lock_file(path)?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
        Err(std::fs::TryLockError::Error(e)) => return Err(e),
    }
    let done = f();
    let _ = lock.unlock();
    done.map(Some)
}

/// `bytes` written to `path`, made or replaced at once: written whole to a
/// file of its own next to it, then moved over it, so that a reader (the
/// IME of another app reading the user dictionary as the settings page
/// saves it) never takes a half-written or empty file for it, and a crash
/// while writing loses nothing.
pub fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static WRITES: AtomicU32 = AtomicU32::new(0);
    let n = WRITES.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}-{n}.tmp", std::process::id()));
    let tmp = path.with_file_name(name);
    let written = (|| {
        // a new file: not one put there to be written through
        let mut file = options().create_new(true).write(true).open(&tmp)?;
        narrow(&file)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_files_are_the_users_alone() {
        let dir = std::env::temp_dir().join(format!("setsuna-private-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, w) = (dir.join("a.jsonl"), dir.join("w.txt"));
        super::append(&a).unwrap().write_all(b"x\n").unwrap();
        super::write(&w, b"y").unwrap();
        assert_eq!(std::fs::read_to_string(&w).unwrap(), "y");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for p in [&a, &w] {
                let mode = std::fs::metadata(p).unwrap().permissions().mode();
                assert_eq!(mode & 0o077, 0, "{p:?} {mode:o}");
            }
            // one made readable by others before is narrowed
            std::fs::set_permissions(&a, std::fs::Permissions::from_mode(0o644)).unwrap();
            super::append(&a).unwrap();
            let mode = std::fs::metadata(&a).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "{mode:o}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    use std::io::Write;
}
