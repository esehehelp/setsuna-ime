//! The commit a build is of, for the settings page's version (user
//! 2026-11-11: "設定画面にバージョン書いたら？コミットでもビルド番号でもいい"):
//! `SETSUNA_COMMIT`, the environment's when it is set (a build outside a
//! clone), else `git rev-parse --short=7 HEAD`, else "unknown". Run again
//! when HEAD or its branch moves.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SETSUNA_COMMIT");
    let commit = match std::env::var("SETSUNA_COMMIT") {
        Ok(c) if !c.is_empty() => c,
        _ => {
            let mut watch = vec!["HEAD".to_string(), "packed-refs".to_string()];
            watch.extend(git(&["symbolic-ref", "-q", "HEAD"]));
            for w in watch {
                if let Some(p) = git(&["rev-parse", "--path-format=absolute", "--git-path", &w]) {
                    println!("cargo:rerun-if-changed={p}");
                }
            }
            git(&["rev-parse", "--short=7", "HEAD"]).unwrap_or_else(|| "unknown".into())
        }
    };
    println!("cargo:rustc-env=SETSUNA_COMMIT={commit}");
}
