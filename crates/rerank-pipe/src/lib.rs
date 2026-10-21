//! The named pipe of the shared reranker: one `setsuna-rerank serve`
//! process per login answers the IME in every app, instead of a reranker
//! child per app (each holding its own llama.cpp context).
//!
//! The pipe's name is a hash of everything the clients must agree on to
//! share a server: the user, the manifest, the reranker binary and the
//! options that change its scores or limits. Both ends compute it with
//! [`pipe_name`]; the server claims it, the client opens it.
//!
//!   - win: the Win32 calls of both ends (Windows only)

use std::path::Path;

#[cfg(windows)]
pub mod win;

/// What a server is shared over: clients with the same key use one server.
#[derive(Debug, Clone, Copy)]
pub struct PipeKey<'a> {
    /// The user: the token's SID.
    pub user: &'a str,
    /// The LM scorer manifest, canonical.
    pub manifest: &'a Path,
    /// The reranker binary: [`exe_identity`].
    pub exe: &'a str,
    pub threads: Option<usize>,
    pub max_candidates: Option<usize>,
    pub max_context_chars: Option<usize>,
    pub lambda: Option<f32>,
}

/// `\\.\pipe\setsuna-rerank-<128-bit hex>` of `key`.
pub fn pipe_name(key: &PipeKey) -> String {
    let n = |v: Option<usize>| v.map_or_else(|| "-".to_string(), |v| v.to_string());
    let text = [
        "setsuna-rerank-pipe",
        env!("CARGO_PKG_VERSION"),
        key.user,
        &key.manifest.to_string_lossy().to_lowercase(),
        key.exe,
        &n(key.threads),
        &n(key.max_candidates),
        &n(key.max_context_chars),
        &key.lambda
            .map_or_else(|| "-".to_string(), |l| format!("{:08x}", l.to_bits())),
    ]
    .join("\n");
    format!(
        r"\\.\pipe\setsuna-rerank-{:016x}{:016x}",
        fnv1a(0xcbf2_9ce4_8422_2325, &text),
        fnv1a(0x6c62_272e_07bb_0142, &text)
    )
}

/// The binary as the pipe name sees it: its canonical path, size and
/// modification time, so a rebuilt or reinstalled reranker gets a server of
/// its own while the old one runs out.
pub fn exe_identity(exe: &Path) -> std::io::Result<String> {
    let path = std::fs::canonicalize(exe)?;
    let meta = std::fs::metadata(&path)?;
    let modified = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    Ok(format!(
        "{}|{}|{modified}",
        path.to_string_lossy().to_lowercase(),
        meta.len()
    ))
}

/// FNV-1a, 64 bits, from `basis`: stable across builds and platforms.
fn fnv1a(basis: u64, text: &str) -> u64 {
    text.bytes().fold(basis, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> PipeKey<'static> {
        PipeKey {
            user: "S-1-5-21-1-2-3-1001",
            manifest: Path::new(r"C:\models\rerank\lm.manifest.json"),
            exe: r"c:\setsuna-ime\rerank\setsuna-rerank.exe|1040384|1760000000000000000",
            threads: None,
            max_candidates: Some(20),
            max_context_chars: None,
            lambda: None,
        }
    }

    #[test]
    fn name_is_pinned() {
        // the server and every client build compute this: a change here
        // splits servers across versions, so make it on purpose only
        assert_eq!(
            pipe_name(&key()),
            r"\\.\pipe\setsuna-rerank-c3a284b618595feb0231c0a308bb3446"
        );
    }

    #[test]
    fn every_field_counts() {
        let base = pipe_name(&key());
        let variants = [
            PipeKey {
                user: "S-1-5-21-1-2-3-1002",
                ..key()
            },
            PipeKey {
                manifest: Path::new(r"C:\models\rerank\other.manifest.json"),
                ..key()
            },
            PipeKey {
                exe: r"c:\setsuna-ime\rerank\setsuna-rerank.exe|1040384|1760000000000000001",
                ..key()
            },
            PipeKey {
                threads: Some(2),
                ..key()
            },
            PipeKey {
                max_candidates: None,
                ..key()
            },
            PipeKey {
                max_context_chars: Some(40),
                ..key()
            },
            PipeKey {
                lambda: Some(0.3),
                ..key()
            },
        ];
        for v in &variants {
            assert_ne!(pipe_name(v), base, "{v:?}");
        }
        // the manifest path compares as Windows does, ignoring case
        let upper = PipeKey {
            manifest: Path::new(r"C:\MODELS\rerank\LM.manifest.json"),
            ..key()
        };
        assert_eq!(pipe_name(&upper), base);
    }

    #[test]
    fn exe_identity_reads_the_file() {
        let exe = std::env::current_exe().unwrap();
        let id = exe_identity(&exe).unwrap();
        assert_eq!(id, exe_identity(&exe).unwrap());
        assert!(id.contains(&std::fs::metadata(&exe).unwrap().len().to_string()));
        assert!(exe_identity(Path::new("no/such/setsuna-rerank.exe")).is_err());
    }
}
