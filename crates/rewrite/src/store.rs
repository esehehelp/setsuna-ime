//! The credentials: ChatGPT sign-ins and API keys, in one file next to
//! `config.toml` that only this user can read (encrypted with DPAPI on
//! Windows, mode 0600 elsewhere). Every IME process runs its own rewriter,
//! and a refresh token is replaced on each refresh, so a change happens
//! under an exclusive lock on a sibling lock file, on the file as it is then.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(windows)]
const FILE_NAME: &str = "credentials.bin";
#[cfg(not(windows))]
const FILE_NAME: &str = "credentials.json";
const LOCK_NAME: &str = "credentials.lock";

/// What the credentials file holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Credentials {
    /// This machine's `ext_agent_host_id` (`urn:uuid:...`), kept from the
    /// first sign-in on.
    pub host_id: Option<String>,
    /// One per ChatGPT registration (issued client ID and account). A
    /// sign-out clears the tokens and keeps the rest for the next sign-in.
    pub chatgpt: Vec<ChatGptAccount>,
    /// The issued client ID of the account requests use.
    pub active: Option<String>,
    /// API keys by [`api_key_slot`].
    pub api_keys: BTreeMap<String, String>,
    /// The first-sign-in notice about plan usage was acknowledged.
    pub plan_notice_shown: bool,
}

/// A ChatGPT registration: the client ID issued for this user and
/// workspace, the verified identity, and the tokens while signed in.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatGptAccount {
    pub client_id: String,
    pub sub: String,
    pub email: Option<String>,
    pub name: Option<String>,
    /// Sent back as `id_token_hint` when signing in again.
    pub id_token: Option<String>,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    /// Unix seconds.
    pub expires_at: i64,
    pub scope: String,
    pub saved_at: i64,
}

impl ChatGptAccount {
    pub fn signed_in(&self) -> bool {
        self.refresh_token.is_some() || self.access_token.is_some()
    }

    /// The grant lets requests use the ChatGPT plan.
    pub fn plan_usage(&self) -> bool {
        self.scope
            .split_whitespace()
            .any(|s| s == crate::oauth::PLAN_SCOPE)
    }

    /// A label for the account picker.
    pub fn label(&self) -> String {
        self.email
            .clone()
            .or_else(|| self.name.clone())
            .unwrap_or_else(|| self.sub.clone())
    }
}

impl Credentials {
    pub fn active_account(&self) -> Option<&ChatGptAccount> {
        let id = self.active.as_deref()?;
        self.chatgpt.iter().find(|a| a.client_id == id)
    }

    pub fn active_account_mut(&mut self) -> Option<&mut ChatGptAccount> {
        let id = self.active.clone()?;
        self.chatgpt.iter_mut().find(|a| a.client_id == id)
    }
}

/// Where an API key for `base_url` is kept: one slot per server.
pub fn api_key_slot(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_string()
}

/// The credentials file in a directory (the config directory).
#[derive(Debug, Clone)]
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn path(&self) -> PathBuf {
        self.dir.join(FILE_NAME)
    }

    /// The credentials as saved; empty when there is no file yet.
    pub fn load(&self) -> Result<Credentials> {
        let path = self.path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Credentials::default()),
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let plain = unprotect(&bytes).with_context(|| format!("decrypt {}", path.display()))?;
        serde_json::from_slice(&plain).with_context(|| format!("parse {}", path.display()))
    }

    /// Run `f` on the credentials under the lock and save them when it
    /// changed them (whether it returns `Ok` or not).
    pub fn update<T>(&self, f: impl FnOnce(&mut Credentials) -> Result<T>) -> Result<T> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("create {}", self.dir.display()))?;
        let lock = self.lock()?;
        let before = self.load()?;
        let mut creds = before.clone();
        let out = f(&mut creds);
        if creds != before {
            self.save(&creds)?;
        }
        drop(lock);
        out
    }

    fn lock(&self) -> Result<File> {
        let path = self.dir.join(LOCK_NAME);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        file.lock()
            .with_context(|| format!("lock {}", path.display()))?;
        Ok(file)
    }

    fn save(&self, creds: &Credentials) -> Result<()> {
        let path = self.path();
        let plain = serde_json::to_vec(creds).context("serialize credentials")?;
        let bytes = protect(&plain).context("encrypt credentials")?;
        // write then rename, so a reader never sees half a file
        let tmp = path.with_extension("tmp");
        write_private(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path).with_context(|| format!("replace {}", path.display()))?;
        Ok(())
    }
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    f.write_all(bytes)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

#[cfg(windows)]
fn protect(plain: &[u8]) -> Result<Vec<u8>> {
    dpapi::protect(plain)
}

#[cfg(windows)]
fn unprotect(bytes: &[u8]) -> Result<Vec<u8>> {
    dpapi::unprotect(bytes)
}

#[cfg(not(windows))]
fn protect(plain: &[u8]) -> Result<Vec<u8>> {
    Ok(plain.to_vec())
}

#[cfg(not(windows))]
fn unprotect(bytes: &[u8]) -> Result<Vec<u8>> {
    Ok(bytes.to_vec())
}

/// The current user's DPAPI: only this user on this machine decrypts.
#[cfg(windows)]
mod dpapi {
    use anyhow::Result;
    use windows::core::w;
    use windows::Win32::Foundation::{LocalFree, HLOCAL};
    use windows::Win32::Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    fn blob(data: &[u8]) -> CRYPT_INTEGER_BLOB {
        CRYPT_INTEGER_BLOB {
            cbData: data.len() as u32,
            pbData: data.as_ptr() as *mut u8,
        }
    }

    unsafe fn take(out: CRYPT_INTEGER_BLOB) -> Vec<u8> {
        let v = unsafe { std::slice::from_raw_parts(out.pbData, out.cbData as usize) }.to_vec();
        unsafe {
            let _ = LocalFree(HLOCAL(out.pbData as *mut core::ffi::c_void));
        }
        v
    }

    pub fn protect(plain: &[u8]) -> Result<Vec<u8>> {
        let input = blob(plain);
        let mut out = CRYPT_INTEGER_BLOB::default();
        unsafe {
            CryptProtectData(
                &input,
                w!("Setsuna IME credentials"),
                None,
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut out,
            )?;
            Ok(take(out))
        }
    }

    pub fn unprotect(bytes: &[u8]) -> Result<Vec<u8>> {
        let input = blob(bytes);
        let mut out = CRYPT_INTEGER_BLOB::default();
        unsafe {
            CryptUnprotectData(
                &input,
                None,
                None,
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut out,
            )?;
            Ok(take(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "setsuna-rewrite-store-{tag}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn credentials_round_trip_through_the_file() {
        let t = TempDir::new("round-trip");
        let store = Store::at(&t.0);
        assert_eq!(store.load().unwrap(), Credentials::default());
        store
            .update(|c| {
                c.host_id = Some("urn:uuid:x".into());
                c.api_keys
                    .insert(api_key_slot("https://api.openai.com/v1/"), "sk-1".into());
                c.chatgpt.push(ChatGptAccount {
                    client_id: "oaiapp_1".into(),
                    sub: "user-1".into(),
                    refresh_token: Some("r".into()),
                    scope: "openid chatgpt.tokens.use.direct".into(),
                    ..ChatGptAccount::default()
                });
                c.active = Some("oaiapp_1".into());
                Ok(())
            })
            .unwrap();
        let back = store.load().unwrap();
        assert_eq!(back.api_keys["https://api.openai.com/v1"], "sk-1");
        let account = back.active_account().unwrap();
        assert!(account.signed_in() && account.plan_usage());
        // the key is not stored in the clear
        let raw = std::fs::read(store.path()).unwrap();
        assert_eq!(
            cfg!(windows),
            !raw.windows(4).any(|w| w == b"sk-1"),
            "encrypted on Windows only"
        );
    }

    #[test]
    fn a_failed_update_still_saves_what_it_changed() {
        let t = TempDir::new("failed");
        let store = Store::at(&t.0);
        let r: Result<()> = store.update(|c| {
            c.plan_notice_shown = true;
            anyhow::bail!("refresh failed")
        });
        assert!(r.is_err());
        assert!(store.load().unwrap().plan_notice_shown);
    }
}
