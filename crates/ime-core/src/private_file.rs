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

/// `bytes` written to `path`, made or replaced.
pub fn write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = options()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    narrow(&file)?;
    file.write_all(bytes)
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
