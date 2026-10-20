//! Put llama.cpp's shared libraries next to the executable: `llama.dll` and
//! `ggml*.dll`, which it loads at start, and the CPU backends
//! (`ggml-cpu-*.dll`, one per x86 level), which it loads at run time from
//! its own directory (`setsuna_rerank::load_backends`). The installer copies
//! the same set.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Linux looks for libllama.so.0 and the rest next to the executable only
    // when it says so (Windows always does): an RPATH, not a RUNPATH, so
    // that libllama.so.0 finds libggml-base.so.0 there too.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN,--disable-new-dtags");
    }
    let Some(backends) = std::env::var_os("DEP_LLAMA_BACKENDS_DIR").map(PathBuf::from) else {
        return;
    };
    // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    // (dir, only the names the executable asks for): the backends, the
    // Windows DLLs, and on Linux lib/ (also holding the unversioned names)
    let mut sources = vec![(backends.clone(), false)];
    if let Some(sys_out) = backends.parent() {
        sources.push((sys_out.join("bin"), false));
        sources.push((sys_out.join("lib"), true));
    }
    for (dir, sonames_only) in sources {
        copy_libraries(&dir, profile_dir, sonames_only);
    }
}

fn copy_libraries(from: &Path, to: &Path, sonames_only: bool) {
    let Ok(entries) = std::fs::read_dir(from) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // and on Linux the names the executable asks for (libllama.so.0),
        // symlinks to the versioned files: copied as files
        let soname = entry
            .file_name()
            .to_str()
            .and_then(|n| n.rsplit_once(".so."))
            .is_some_and(|(_, v)| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()));
        let is_lib = soname
            || !sonames_only
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e, "dll" | "so" | "dylib"));
        if !is_lib {
            continue;
        }
        let dst = to.join(entry.file_name());
        let same = std::fs::metadata(&dst)
            .ok()
            .zip(entry.metadata().ok())
            .is_some_and(|(d, s)| d.len() == s.len() && d.modified().ok() >= s.modified().ok());
        if !same {
            // a running reranker holds the old one: keep it, the next build retries
            if let Err(e) = std::fs::copy(&path, &dst) {
                println!(
                    "cargo:warning=copy {} -> {}: {e}",
                    path.display(),
                    dst.display()
                );
            }
        }
    }
}
