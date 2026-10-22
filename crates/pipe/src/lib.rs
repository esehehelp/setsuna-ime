//! The named pipes of the shared servers: one `setsuna-engine serve` and one
//! `setsuna-rerank serve` process per login answer the IME in every app,
//! instead of an engine and a reranker child per app (each holding its own
//! copy of the model).
//!
//! A pipe's name is a hash of everything the clients must agree on to share
//! a server: the user, the manifest, the binary and the options that change
//! its answers. The reranker's both ends compute it with
//! [`rerank_pipe_name`]; the engine's client computes it with
//! [`engine_pipe_name`] and passes it to the server it starts.
//!
//!   - win: the Win32 calls of both ends (Windows only)
//!   - server: the server's loop, a worker thread owning the model (Windows only)

use std::path::Path;

#[cfg(windows)]
pub mod server;
#[cfg(windows)]
pub mod win;

/// What a reranker server is shared over: clients with the same key use one
/// server.
#[derive(Debug, Clone, Copy)]
pub struct RerankKey<'a> {
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
pub fn rerank_pipe_name(key: &RerankKey) -> String {
    let n = |v: Option<usize>| v.map_or_else(|| "-".to_string(), |v| v.to_string());
    let options = [
        n(key.threads),
        n(key.max_candidates),
        n(key.max_context_chars),
        key.lambda
            .map_or_else(|| "-".to_string(), |l| format!("{:08x}", l.to_bits())),
    ];
    name_of("setsuna-rerank", key.user, key.manifest, key.exe, &options)
}

/// `\\.\pipe\setsuna-engine-<128-bit hex>` of the user, the conversion
/// model's manifest (canonical), the engine binary ([`exe_identity`]) and
/// the decode flags the server is started with (each as passed, in order).
pub fn engine_pipe_name(user: &str, manifest: &Path, exe: &str, flags: &[String]) -> String {
    name_of("setsuna-engine", user, manifest, exe, flags)
}

fn name_of(service: &str, user: &str, manifest: &Path, exe: &str, options: &[String]) -> String {
    let mut text = vec![
        format!("{service}-pipe"),
        env!("CARGO_PKG_VERSION").to_string(),
        user.to_string(),
        manifest.to_string_lossy().to_lowercase(),
        exe.to_string(),
    ];
    text.extend(options.iter().cloned());
    let text = text.join("\n");
    format!(
        r"\\.\pipe\{service}-{:016x}{:016x}",
        fnv1a(0xcbf2_9ce4_8422_2325, &text),
        fnv1a(0x6c62_272e_07bb_0142, &text)
    )
}

/// The binary as the pipe name sees it: its canonical path, size and
/// modification time, so a rebuilt or reinstalled binary gets a server of
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

    fn key() -> RerankKey<'static> {
        RerankKey {
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
            rerank_pipe_name(&key()),
            r"\\.\pipe\setsuna-rerank-c3a284b618595feb0231c0a308bb3446"
        );
    }

    #[test]
    fn every_field_counts() {
        let base = rerank_pipe_name(&key());
        let variants = [
            RerankKey {
                user: "S-1-5-21-1-2-3-1002",
                ..key()
            },
            RerankKey {
                manifest: Path::new(r"C:\models\rerank\other.manifest.json"),
                ..key()
            },
            RerankKey {
                exe: r"c:\setsuna-ime\rerank\setsuna-rerank.exe|1040384|1760000000000000001",
                ..key()
            },
            RerankKey {
                threads: Some(2),
                ..key()
            },
            RerankKey {
                max_candidates: None,
                ..key()
            },
            RerankKey {
                max_context_chars: Some(40),
                ..key()
            },
            RerankKey {
                lambda: Some(0.3),
                ..key()
            },
        ];
        for v in &variants {
            assert_ne!(rerank_pipe_name(v), base, "{v:?}");
        }
        // the manifest path compares as Windows does, ignoring case
        let upper = RerankKey {
            manifest: Path::new(r"C:\MODELS\rerank\LM.manifest.json"),
            ..key()
        };
        assert_eq!(rerank_pipe_name(&upper), base);
    }

    #[test]
    fn engine_name_counts_the_flags_and_differs_from_the_reranker() {
        let (user, manifest) = (
            "S-1-5-21-1-2-3-1001",
            Path::new(r"C:\models\onnx\m.manifest.json"),
        );
        let exe = r"c:\setsuna-ime\setsuna-engine.exe|1460736|1760000000000000000";
        let flags = |f: &[&str]| f.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let base = engine_pipe_name(user, manifest, exe, &flags(&["--emit-alignment"]));
        // pinned as the reranker's is: the IME's clients in every app compute it
        assert_eq!(
            base,
            r"\\.\pipe\setsuna-engine-0d8b5a77b2f80471999a54fdea85d64c"
        );
        for other in [
            engine_pipe_name(user, manifest, exe, &flags(&[])),
            engine_pipe_name(
                user,
                manifest,
                exe,
                &flags(&["--emit-alignment", "--beam-width=20"]),
            ),
            engine_pipe_name(
                "S-1-5-21-1-2-3-1002",
                manifest,
                exe,
                &flags(&["--emit-alignment"]),
            ),
        ] {
            assert_ne!(other, base);
        }
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
