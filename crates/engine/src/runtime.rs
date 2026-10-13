//! The ONNX Runtime the engine runs on: Microsoft's official build, loaded
//! from an explicit path.
//!
//! The `ort` crate's bundled (pyke) build was measurably slower than the
//! official build of the same version: on Aster-v1.2-small int8, T=58, 4
//! threads, identical Rust code and session options, 6.35 ms vs 4.02 ms per
//! session.run. So the engine links no runtime and loads `onnxruntime.dll`
//! (`libonnxruntime.so`, `libonnxruntime.dylib`) itself, from, in order:
//!
//! 1. an explicit path (`--ort-dylib`),
//! 2. `SETSUNA_ORT_DYLIB`,
//! 3. the directory of the engine executable.
//!
//! Never a bare library name: Windows keeps an old `onnxruntime.dll` in
//! System32 that the default search would find first. `fetch-onnxruntime.ps1`
//! puts the pinned official build next to the engine.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};

#[cfg(target_os = "windows")]
pub const DYLIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
pub const DYLIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub const DYLIB_NAME: &str = "libonnxruntime.so";

static LOADED: OnceLock<PathBuf> = OnceLock::new();

/// Where the runtime is loaded from, before any file check.
fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("SETSUNA_ORT_DYLIB").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().context("locate the engine executable")?;
    let dir = exe
        .parent()
        .context("the engine executable has no directory")?;
    Ok(dir.join(DYLIB_NAME))
}

/// Load the runtime once per process. Later calls return the path loaded
/// first, whatever they are given (a process can hold one runtime).
pub fn init(explicit: Option<&Path>) -> Result<&'static Path> {
    if let Some(p) = LOADED.get() {
        return Ok(p);
    }
    let path = resolve(explicit)?;
    anyhow::ensure!(
        path.is_file(),
        "ONNX Runtime not found at {}: put Microsoft's official {DYLIB_NAME} next to the engine \
         (crates/engine/fetch-onnxruntime.ps1), or pass --ort-dylib / set SETSUNA_ORT_DYLIB",
        path.display()
    );
    let absolute = std::path::absolute(&path).unwrap_or(path);
    ort::init_from(&absolute)
        .map_err(|e| anyhow::anyhow!("load ONNX Runtime {}: {e}", absolute.display()))?
        .commit();
    Ok(LOADED.get_or_init(|| absolute))
}

/// The loaded runtime's build description (version, commit), for logs.
pub fn build_info() -> &'static str {
    ort::info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_path_wins_and_a_missing_file_is_an_error() {
        let p = Path::new("definitely/not/here/onnxruntime.dll");
        assert_eq!(resolve(Some(p)).unwrap(), p);
        // never falls back to a bare name for the system search
        assert!(resolve(None).unwrap().is_absolute());
        let err = init(Some(p)).unwrap_err().to_string();
        assert!(err.contains("ONNX Runtime not found"), "{err}");
    }
}
