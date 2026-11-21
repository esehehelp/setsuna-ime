//! The rows typed into the IME session (setsuna-core), as a frontend drives
//! it, without a desktop: each row in a field of its own (the context of the
//! rows before forgotten, the row's context as the text before the caret),
//! its keys one after another (or `key_ms` apart), Space once the view is
//! still, Enter after a look (`look_ms`) once it is still again, as the TSF
//! runner does. The user's config (or `--config`'s, as a Kairo task gives
//! it) with learning, prediction and the rewriter off, so nothing is
//! recorded; an engine child of its own and the login's reranker.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{ensure, Result};
use serde::Serialize;
use setsuna_core::child_daemon::ChildDaemonClient;
use setsuna_core::config::Config;
use setsuna_core::engine_client::{EngineClient, EngineResult, Word};
use setsuna_core::keysym::{Key, MOD_SHIFT};
use setsuna_core::session::{Field, Session};

use crate::set::Row;

const LEFT: u32 = 0xff51;
const RETURN: u32 = 0xff0d;
const SPACE: u32 = 0x20;

pub struct Options {
    /// Settings from this file (`load_config`) instead of the user's
    /// config.toml.
    pub config: Option<std::path::PathBuf>,
    pub rerank: bool,
    pub lexicon: bool,
    /// Keep the context of the rows before (an IME that cannot read the
    /// field and an app that empties it unseen), as a real text box under
    /// CUAS does.
    pub carry: bool,
    pub manifest: Option<std::path::PathBuf>,
    /// Keys this far apart; 0: one after another without waiting.
    pub key_ms: u64,
    /// The view still this long counts as settled.
    pub quiet_ms: u64,
    /// Enter at the earliest this long after Space (a look at the
    /// conversion), then once the view is still for `quiet_ms`.
    pub look_ms: u64,
    pub limit: usize,
}

#[derive(Serialize)]
pub struct Typed {
    pub id: String,
    pub category: String,
    pub output: String,
    /// Space to the first change of the view (the conversion shown).
    pub space_first_ms: f64,
    /// Space to the last change of the view (reranked, settled).
    pub space_ms: f64,
    /// Typing done to the view settled.
    pub typed_ms: f64,
}

/// The engine client with each request's time to its result.
struct Timed {
    inner: ChildDaemonClient,
    sent: Arc<Mutex<HashMap<u64, Instant>>>,
}

impl EngineClient for Timed {
    fn dispatch(&self, req_id: u64, context: &str, reading: &str) {
        self.sent.lock().unwrap().insert(req_id, Instant::now());
        self.inner.dispatch(req_id, context, reading)
    }
    fn dispatch_more(&self, req_id: u64, context: &str, reading: &str, n: u32) {
        self.sent.lock().unwrap().insert(req_id, Instant::now());
        self.inner.dispatch_more(req_id, context, reading, n)
    }
    fn take_latest(&self) -> Option<EngineResult> {
        self.inner.take_latest()
    }
    fn wait_latest(&self, timeout: Duration) -> bool {
        self.inner.wait_latest(timeout)
    }
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }
    fn analyze(&self, texts: &[String], timeout: Duration) -> Option<Vec<Vec<Word>>> {
        self.inner.analyze(texts, timeout)
    }
}

pub struct Harness {
    s: Session,
    opts: Options,
    /// Engine request latencies (ms), dispatch to result.
    pub engine_ms: Arc<Mutex<Vec<f64>>>,
}

fn view_key(s: &Session) -> String {
    let v = s.view();
    format!(
        "{}\u{1}{}\u{1}{}",
        v.preedit.concat(),
        v.candidates.join("\u{2}"),
        v.commits.concat()
    )
}

/// Settings from `path`, every `${KAIRO_*}` in it replaced by that
/// environment variable (`expand_kairo`): a run under Kairo names the
/// directories it is handed (kairo.toml).
pub fn load_config(path: &std::path::Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let text = expand_kairo(&text, |name| std::env::var(name).ok())?;
    Config::load_from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))
}

/// `text` with each `${NAME}` replaced by `var(NAME)`, backslashes made
/// forward slashes (the value goes into a TOML string). Only the
/// directories Kairo hands a task (`KAIRO_INPUT_*`, `KAIRO_DATA_*`,
/// `KAIRO_OUTPUT_DIR`, `KAIRO_SCRATCH`, `KAIRO_CACHE_DIR`): no other
/// variable (a token Kairo or the shell holds) goes into a config, whose
/// values may end up in a log; and each must be set: a path left as written
/// would be read as a relative one.
pub fn expand_kairo(text: &str, var: impl Fn(&str) -> Option<String>) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| anyhow::anyhow!("unclosed ${{ in the config"))?;
        let name = &after[..end];
        let suffix = |prefix: &str| {
            name.strip_prefix(prefix).is_some_and(|s| {
                !s.is_empty()
                    && s.chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            })
        };
        ensure!(
            suffix("KAIRO_INPUT_")
                || suffix("KAIRO_DATA_")
                || matches!(
                    name,
                    "KAIRO_OUTPUT_DIR" | "KAIRO_SCRATCH" | "KAIRO_CACHE_DIR"
                ),
            "only Kairo's directories (${{KAIRO_INPUT_*}}, ${{KAIRO_DATA_*}}, \
             ${{KAIRO_OUTPUT_DIR}}, ${{KAIRO_SCRATCH}}, ${{KAIRO_CACHE_DIR}}) are expanded, \
             not ${{{name}}}"
        );
        let value = var(name).ok_or_else(|| anyhow::anyhow!("{name} is not set"))?;
        out.push_str(&value.replace('\\', "/"));
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

impl Harness {
    pub fn new(opts: Options) -> Result<Self> {
        let mut cfg = match &opts.config {
            Some(path) => load_config(path)?,
            None => Config::load_default(),
        };
        cfg.learning.enabled = false;
        cfg.frontend.prediction = false;
        cfg.rewriter.enabled = false;
        if !opts.lexicon {
            cfg.lexicon.file = None;
        }
        if let Some(m) = &opts.manifest {
            cfg.engine.model_manifest = Some(m.clone());
        }
        let manifest = cfg
            .engine
            .model_manifest
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the config names no [engine] model_manifest"))?;
        let exe = cfg
            .engine
            .engine_exe
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the config names no [engine] engine_exe"))?;
        let sent: Arc<Mutex<HashMap<u64, Instant>>> = Arc::default();
        let engine_ms: Arc<Mutex<Vec<f64>>> = Arc::default();
        let notify = {
            let (sent, engine_ms) = (sent.clone(), engine_ms.clone());
            Box::new(move |id: u64| {
                if let Some(t) = sent.lock().unwrap().remove(&id) {
                    engine_ms
                        .lock()
                        .unwrap()
                        .push(t.elapsed().as_secs_f64() * 1e3);
                }
            })
        };
        let inner = ChildDaemonClient::spawn(&manifest, &exe, notify, &cfg.engine)?;
        let t0 = Instant::now();
        while !inner.is_ready() {
            ensure!(t0.elapsed() < Duration::from_secs(120), "engine not ready");
            std::thread::sleep(Duration::from_millis(50));
        }
        let client: Box<dyn EngineClient> = Box::new(Timed { inner, sent });
        let mut s = Session::new(&cfg, Some(client));
        if opts.rerank {
            let r = setsuna_core::rerank_client::spawn_with_config(&cfg, None, Box::new(|_| {}));
            ensure!(r.is_some(), "no reranker in the config");
            s.set_reranker(r, &cfg);
        }
        let mut h = Self { s, opts, engine_ms };
        // warm-up: the servers up, nothing of it kept
        h.type_row("", "korehatesutodesu.")?;
        h.engine_ms.lock().unwrap().clear();
        Ok(h)
    }

    fn press(&mut self, sym: u32, before: Option<&str>, out: &mut String) {
        let mods = if (b'A' as u32..=b'Z' as u32).contains(&sym) {
            MOD_SHIFT
        } else {
            0
        };
        self.s
            .key_in(&Key { sym, code: 0, mods }, Field::Normal, before);
        out.push_str(&self.s.view().commits.concat());
        self.s.clear_commits();
    }

    /// Pump the session until its view has been still for `quiet` (at most
    /// `cap`); the times from the start to its first and last change.
    fn settle(&mut self, quiet: Duration, cap: Duration, out: &mut String) -> (Duration, Duration) {
        self.settle_after(Duration::ZERO, quiet, cap, out)
    }

    /// As `settle`, but for `at_least` in any case.
    fn settle_after(
        &mut self,
        at_least: Duration,
        quiet: Duration,
        cap: Duration,
        out: &mut String,
    ) -> (Duration, Duration) {
        let start = Instant::now();
        let mut last = view_key(&self.s);
        let mut changed = start;
        let mut first = None;
        loop {
            self.s.apply_async_result();
            self.s.tick();
            out.push_str(&self.s.view().commits.concat());
            self.s.clear_commits();
            let now = view_key(&self.s);
            if now != last {
                last = now;
                changed = Instant::now();
                first.get_or_insert(changed);
            }
            if (changed.elapsed() >= quiet && start.elapsed() >= at_least) || start.elapsed() >= cap
            {
                return (first.unwrap_or(start) - start, changed - start);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Type `keys` after `context` in a field of its own; what was committed,
    /// Space to shown and to settled, typing done to settled.
    fn type_row(&mut self, context: &str, keys: &str) -> Result<(String, f64, f64, f64)> {
        let mut out = String::new();
        if !self.opts.carry {
            self.press(LEFT, None, &mut out);
        }
        // a box under CUAS shows TSF no text at all: the IME reads "" and keeps its own commits
        let before = if self.opts.carry {
            Some("")
        } else {
            Some(context)
        };
        for c in keys.chars() {
            self.press(c as u32, before, &mut out);
            if self.opts.key_ms == 0 {
                continue;
            }
            let end = Instant::now() + Duration::from_millis(self.opts.key_ms);
            while Instant::now() < end {
                self.s.apply_async_result();
                self.s.tick();
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let quiet = Duration::from_millis(self.opts.quiet_ms);
        let (_, typed) = self.settle(quiet, Duration::from_secs(3), &mut out);
        let shown = view_key(&self.s);
        let t = Instant::now();
        self.press(SPACE, before, &mut out);
        let pressed = t.elapsed();
        let by_key = view_key(&self.s) != shown;
        let (first, space) = self.settle_after(
            Duration::from_millis(self.opts.look_ms),
            quiet,
            Duration::from_secs(5),
            &mut out,
        );
        self.press(RETURN, before, &mut out);
        self.settle(Duration::from_millis(50), Duration::from_secs(2), &mut out);
        // a conversion the key itself showed counts from the key
        let first = if by_key { pressed } else { pressed + first };
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        Ok((out, ms(first), ms(pressed + space), ms(typed)))
    }

    /// Type `rows`, `each` told of every row typed; the run stops at a row
    /// `each` answers false for (its output cannot be written).
    pub fn run(
        &mut self,
        rows: &[Row],
        mut each: impl FnMut(&Typed) -> bool,
    ) -> Result<Vec<Typed>> {
        let mut done = Vec::new();
        for row in rows.iter().take(self.opts.limit) {
            let keys = match crate::romaji::keys(&row.reading) {
                Ok(keys) => keys,
                Err(c) => {
                    eprintln!("{}: cannot type {c:?}, left out", row.id);
                    continue;
                }
            };
            let (output, space_first_ms, space_ms, typed_ms) =
                self.type_row(&row.context, &keys)?;
            let t = Typed {
                id: row.id.clone(),
                category: row.category.clone(),
                output: output.replace(['\r', '\n'], ""),
                space_first_ms,
                space_ms,
                typed_ms,
            };
            let go_on = each(&t);
            done.push(t);
            if !go_on {
                break;
            }
        }
        Ok(done)
    }
}

#[cfg(test)]
mod tests {
    use super::expand_kairo;

    #[test]
    fn a_config_names_the_directories_kairo_hands_the_run() {
        let var = |name: &str| {
            (name == "KAIRO_INPUT_BUILD").then(|| r"C:\store\outputs\exe_1".to_string())
        };
        assert_eq!(
            expand_kairo(r#"engine_exe = "${KAIRO_INPUT_BUILD}/bin/e.exe""#, var).unwrap(),
            r#"engine_exe = "C:/store/outputs/exe_1/bin/e.exe""#
        );
        // unset, not KAIRO_*, unclosed: refused rather than read as a path
        assert!(expand_kairo("${KAIRO_INPUT_MODEL}", var).is_err());
        assert!(expand_kairo("${HOME}", var).is_err());
        // Kairo's other variables (a token) are no directory
        let token = |_: &str| Some("secret".to_string());
        assert!(expand_kairo("${KAIRO_TOKEN}", token).is_err());
        assert!(expand_kairo("${KAIRO_INPUT_}", token).is_err());
        assert!(expand_kairo("${KAIRO_OUTPUT_DIR}", token).is_ok());
        assert!(expand_kairo("${KAIRO_INPUT_BUILD", var).is_err());
        assert_eq!(expand_kairo("no variables", var).unwrap(), "no variables");
    }

    #[test]
    fn the_kairo_bench_config_is_a_config() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kairo/bench/harness.toml"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let text = expand_kairo(&text, |name| Some(format!("C:/inputs/{name}"))).unwrap();
        let cfg = setsuna_core::config::Config::load_from_str(&text).unwrap();
        assert!(cfg.engine.engine_exe.is_some() && cfg.engine.model_manifest.is_some());
        assert!(!cfg.learning.enabled && !cfg.rewriter.enabled);
    }
}
