//! Put llama.cpp's shared libraries next to the executable: `llama.dll` and
//! `ggml*.dll`, which it loads at start, and the CPU backends
//! (`ggml-cpu-*.dll`, one per x86 level), which it loads at run time from
//! its own directory (`setsuna_rerank::load_backends`). The installer copies
//! the same set.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let Some(backends) = std::env::var_os("DEP_LLAMA_BACKENDS_DIR").map(PathBuf::from) else {
        return;
    };
    // OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    let mut sources = vec![backends.clone()];
    if let Some(sys_out) = backends.parent() {
        sources.push(sys_out.join("bin"));
    }
    for dir in sources {
        copy_libraries(&dir, profile_dir);
    }
}

fn copy_libraries(from: &Path, to: &Path) {
    let Ok(entries) = std::fs::read_dir(from) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_lib = path
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
