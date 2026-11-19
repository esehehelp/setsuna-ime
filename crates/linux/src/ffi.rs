//! The C ABI the fcitx5 addon calls (`include/setsuna.h`).
//!
//! Every function runs on the addon's event-loop thread except the notify
//! callback, which the engine client calls from its reader thread. Panics
//! never cross the boundary: each entry point catches them and returns its
//! "nothing happened" value.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use setsuna_core::child_daemon;
use setsuna_core::config::{AppStart, Config, RerankCfg, RewriterCfg};
use setsuna_core::engine_client::Notify;
use setsuna_core::rerank_client;
use setsuna_core::rewrite_client::{self, RewriteClient};

use crate::keys::Key;
use crate::session::{Field, Session};

/// App-specific initial mode from `[[apps]]` (`setsuna_session_configure_app`).
pub const SETSUNA_APP_START_KEEP: c_int = 0;
pub const SETSUNA_APP_START_ON: c_int = 1;
pub const SETSUNA_APP_START_OFF: c_int = 2;

/// Minimum interval between `stat(config.toml)` checks on idle key presses.
const CONFIG_CHECK_EVERY: Duration = Duration::from_secs(1);
/// Cooldown before retrying an engine spawn that failed (no model, bad path).
const ENGINE_RETRY_EVERY: Duration = Duration::from_secs(30);

fn config_stamp(path: Option<&Path>) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path?).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

fn rerank_launch_key(c: &RerankCfg) -> Option<String> {
    c.manifest.as_ref()?;
    Some(format!(
        "{:?}|{:?}|{:?}|{:?}",
        c.manifest, c.exe, c.threads, c.max_candidates
    ))
}

fn app_start_code(start: AppStart) -> c_int {
    match start {
        AppStart::Keep => SETSUNA_APP_START_KEEP,
        AppStart::On => SETSUNA_APP_START_ON,
        AppStart::Off => SETSUNA_APP_START_OFF,
    }
}

/// Wakes the addon's event loop; called from a backend thread with the
/// `user` pointer given to `setsuna_session_new`.
pub type SetsunaNotify = extern "C" fn(user: *mut c_void);

/// The view as C strings; pointers valid until the next call on the session.
#[repr(C)]
pub struct SetsunaView {
    pub commits: *const *const c_char,
    pub n_commits: usize,
    pub preedit: *const *const c_char,
    pub n_preedit: usize,
    /// Index of the focused preedit segment, -1 for none.
    pub focus: isize,
    /// Byte offset of the caret in the concatenated preedit.
    pub cursor: usize,
    pub candidates: *const *const c_char,
    /// A note per candidate (`[全] カタカナ`), "" for none.
    pub comments: *const *const c_char,
    pub n_candidates: usize,
    pub selected: usize,
    /// The reading being converted, "" while typing.
    pub reading: *const c_char,
    /// The F6–F10 form shown and its key ("全角カタカナ", "F7"), "" for none.
    pub form: *const c_char,
    pub form_key: *const c_char,
    /// Conversions of what is typed, for a suggestion window; none while
    /// converting. A click picks one like a candidate (by its index).
    pub suggestions: *const *const c_char,
    pub suggestion_comments: *const *const c_char,
    pub n_suggestions: usize,
    /// A line under the open candidate list (書き直し中…), "" for none.
    pub status: *const c_char,
}

/// C strings of a list, and the pointer array handed out.
#[derive(Default)]
struct CList {
    strings: Vec<CString>,
    ptrs: Vec<*const c_char>,
}

impl CList {
    fn set(&mut self, items: &[String]) {
        self.strings = items
            .iter()
            .map(|s| CString::new(s.replace('\0', "")).unwrap_or_default())
            .collect();
        self.ptrs = self.strings.iter().map(|s| s.as_ptr()).collect();
    }
}

pub struct SetsunaSession {
    session: Session,
    /// Where the engine and models are looked for last (see `with_module_defaults`).
    module_dir: Option<PathBuf>,
    /// What the daemons wake the addon with.
    notify: SetsunaNotify,
    user: UserPtr,
    /// Path to `config.toml` watched for changes (`mtime`, `len`).
    config_path: Option<PathBuf>,
    config_stamp: Option<(SystemTime, u64)>,
    config_checked: Option<Instant>,
    /// Last successfully loaded `config.toml` (with `module_dir` defaults).
    base_config: Config,
    /// Last active application program name (for idle key-time reloads).
    last_program: String,
    /// Effective `Config` debug key last applied to `session`.
    applied_key: Option<String>,
    /// `(engine config key, spawn succeeded)` of the last engine launch attempt.
    engine_started: Option<(String, bool)>,
    engine_retry_at: Option<Instant>,
    /// `(reranker launch key, spawn succeeded)` of the last reranker attempt.
    reranker_started: Option<(String, bool)>,
    /// `(rewriter cfg, spawn succeeded)` of the last rewriter attempt.
    rewriter_started: Option<(RewriterCfg, bool)>,
    commits: CList,
    preedit: CList,
    candidates: CList,
    comments: CList,
    /// reading, form, form key, status
    texts: CList,
    suggestions: CList,
    suggestion_comments: CList,
    view: SetsunaView,
}

impl SetsunaSession {
    /// Rebuild the C view from the session's.
    fn refresh(&mut self) {
        let v = self.session.view();
        self.commits.set(&v.commits);
        self.preedit.set(&v.preedit);
        self.candidates.set(&v.candidates);
        self.comments.set(&v.comments);
        self.suggestions.set(&v.suggestions);
        self.suggestion_comments.set(&v.suggestion_comments);
        let (form, key) = v.form.unwrap_or(("", ""));
        self.texts.set(&[
            v.reading.clone(),
            form.to_string(),
            key.to_string(),
            v.status.clone(),
        ]);
        self.view = SetsunaView {
            commits: self.commits.ptrs.as_ptr(),
            n_commits: self.commits.ptrs.len(),
            preedit: self.preedit.ptrs.as_ptr(),
            n_preedit: self.preedit.ptrs.len(),
            focus: v.focus.map_or(-1, |f| f as isize),
            cursor: v.cursor(),
            candidates: self.candidates.ptrs.as_ptr(),
            comments: self.comments.ptrs.as_ptr(),
            n_candidates: self.candidates.ptrs.len(),
            selected: v.selected,
            reading: self.texts.ptrs[0],
            form: self.texts.ptrs[1],
            form_key: self.texts.ptrs[2],
            suggestions: self.suggestions.ptrs.as_ptr(),
            suggestion_comments: self.suggestion_comments.ptrs.as_ptr(),
            n_suggestions: self.suggestions.ptrs.len(),
            status: self.texts.ptrs[3],
        };
    }

    /// A notify for a daemon's reader thread: wakes the addon, which then
    /// takes the result (`setsuna_session_apply_async`).
    fn wake(&self) -> Notify {
        let (notify, user) = (self.notify, self.user);
        Box::new(move |_req_id: u64| {
            let user = &user;
            notify(user.0)
        })
    }

    /// Re-read `config.toml` when its `(mtime, size)` changed (throttled when
    /// `!force`). Keeps the previous `base_config` on parse errors so a half-
    /// written or broken file never wipes live settings.
    fn reload_base_config(&mut self, force: bool) -> bool {
        let now = Instant::now();
        if !force
            && self
                .config_checked
                .is_some_and(|t| now.duration_since(t) < CONFIG_CHECK_EVERY)
        {
            return false;
        }
        self.config_checked = Some(now);
        let stamp = config_stamp(self.config_path.as_deref());
        if stamp == self.config_stamp && self.applied_key.is_some() {
            return false;
        }
        self.config_stamp = stamp;
        let loaded = match self.config_path.as_deref() {
            Some(p) if p.exists() => match Config::load_from_path(p) {
                Ok(c) => c,
                Err(e) => {
                    setsuna_core::diagnostics::log(format!(
                        "fcitx5 config.toml not applied: {e:#}"
                    ));
                    return false;
                }
            },
            _ => Config::default(),
        };
        self.base_config = with_module_defaults(loaded, self.module_dir.as_deref());
        self.engine_retry_at = None;
        true
    }

    /// Apply `base_config` + `[[apps]]` for `program` only when the config
    /// file, effective app settings, or daemon liveness changed. Returns the
    /// app's configured `AppStart` without overwriting `session.enabled()`.
    fn sync_config(&mut self, program: &str, force: bool) -> AppStart {
        let file_changed = self.reload_base_config(force);
        let (cfg, app) = self.base_config.for_app(program);
        let start = app.as_ref().map_or(AppStart::Keep, |a| a.start);
        let effective_key = format!("{cfg:?}");
        let engine_key = format!("{:?}", cfg.engine);
        let rerank_key = rerank_launch_key(&cfg.rerank);
        let now = Instant::now();

        let engine_changed = self
            .engine_started
            .as_ref()
            .is_none_or(|(k, _)| k != &engine_key);
        let engine_died =
            self.engine_started.as_ref().is_some_and(|(_, ok)| *ok) && !self.session.has_client();
        let engine_retry =
            !self.session.has_client() && self.engine_retry_at.is_none_or(|t| now >= t);
        let needs_engine = engine_changed || engine_died || engine_retry;

        let needs_reranker = needs_engine
            || match (&rerank_key, &self.reranker_started) {
                (None, Some(_)) => true,
                (None, None) => self.session.has_reranker(),
                (Some(k), Some((prev, ok))) => {
                    k != prev || ((*ok || file_changed) && !self.session.has_reranker())
                }
                (Some(_), None) => true,
            };

        let needs_rewriter = needs_engine
            || if cfg.rewriter.enabled {
                match &self.rewriter_started {
                    Some((prev, ok)) => {
                        prev != &cfg.rewriter
                            || ((*ok || file_changed) && !self.session.has_rewriter())
                    }
                    None => true,
                }
            } else {
                self.session.has_rewriter() || self.rewriter_started.is_some()
            };

        let cfg_changed = file_changed || self.applied_key.as_ref() != Some(&effective_key);
        self.last_program = program.to_string();
        if !cfg_changed && !needs_engine && !needs_reranker && !needs_rewriter {
            return start;
        }

        let module_dir = self.module_dir.clone();
        let module_dir = module_dir.as_deref();
        if needs_engine {
            let client = child_daemon::spawn_with_config(&cfg, module_dir, self.wake());
            let ok = client.is_some();
            if ok {
                setsuna_core::diagnostics::log("fcitx5 engine started");
                self.engine_retry_at = None;
            } else {
                self.engine_retry_at = Some(now + ENGINE_RETRY_EVERY);
            }
            self.engine_started = Some((engine_key, ok));
            let was_enabled = self.session.enabled();
            self.session = Session::new(&cfg, client);
            self.session.set_enabled(was_enabled);
            self.reranker_started = None;
            self.rewriter_started = None;
        } else if cfg_changed {
            self.session.configure(&cfg);
        }

        if needs_reranker {
            if let Some(key) = rerank_key {
                let reranker = rerank_client::spawn_with_config(&cfg, module_dir, self.wake());
                self.reranker_started = Some((key, reranker.is_some()));
                self.session.set_reranker(reranker, &cfg);
            } else {
                self.reranker_started = None;
                self.session.set_reranker(None, &cfg);
            }
        } else if cfg_changed {
            self.session.set_rerank_config(&cfg);
        }

        if needs_rewriter {
            if cfg.rewriter.enabled {
                let rewriter = rewrite_client::spawn_with_config(&cfg, module_dir, self.wake());
                self.rewriter_started = Some((cfg.rewriter.clone(), rewriter.is_some()));
                self.session
                    .set_rewriter(rewriter.map(|c| Box::new(c) as Box<dyn RewriteClient>));
            } else {
                self.rewriter_started = None;
                self.session.set_rewriter(None);
            }
        }
        self.session.set_rewrite_mode(true);
        self.applied_key = Some(effective_key);
        setsuna_core::diagnostics::log(format!(
            "fcitx5 configure program={program:?} app={} engine={} reranker={} rewriter={}",
            app.is_some(),
            self.session.has_client(),
            self.session.has_reranker(),
            self.session.rewrite_mode()
        ));
        start
    }
}

/// The addon's `user` pointer, carried to the reader threads. The addon
/// keeps it valid until `setsuna_session_free` returns, which joins them.
#[derive(Clone, Copy)]
struct UserPtr(*mut c_void);
unsafe impl Send for UserPtr {}

fn guard<R>(fallback: R, f: impl FnOnce() -> R) -> R {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        setsuna_core::diagnostics::log("panic in the fcitx5 frontend");
        fallback
    })
}

/// Start a session: load `config.toml` and spawn the daemons it asks for.
/// `module_dir` (may be null) is where the engine and models are looked for
/// last, and the mixed input model (`mixed`, installed with the addon) when
/// `[frontend] mixed_model` is not set. Never returns null.
///
/// # Safety
/// `module_dir` is null or a NUL-terminated string. `notify` may be called
/// with `user` from another thread until `setsuna_session_free` returns.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_new(
    module_dir: *const c_char,
    notify: SetsunaNotify,
    user: *mut c_void,
) -> *mut SetsunaSession {
    guard(std::ptr::null_mut(), || {
        setsuna_core::diagnostics::init("fcitx5.log");
        let module_dir = (!module_dir.is_null())
            .then(|| PathBuf::from(CStr::from_ptr(module_dir).to_string_lossy().into_owned()));
        let config_path = Config::appdata_config_path();
        let mut s = Box::new(SetsunaSession::empty(module_dir, config_path, notify, user));
        s.sync_config("", true);
        s.refresh();
        Box::into_raw(s)
    })
}

impl SetsunaSession {
    fn empty(
        module_dir: Option<PathBuf>,
        config_path: Option<PathBuf>,
        notify: SetsunaNotify,
        user: *mut c_void,
    ) -> Self {
        let base_config = with_module_defaults(Config::default(), module_dir.as_deref());
        Self {
            session: Session::new(&base_config, None),
            module_dir,
            notify,
            user: UserPtr(user),
            config_path,
            config_stamp: None,
            config_checked: None,
            base_config,
            last_program: String::new(),
            applied_key: None,
            engine_started: None,
            engine_retry_at: None,
            reranker_started: None,
            rewriter_started: None,
            commits: CList::default(),
            preedit: CList::default(),
            candidates: CList::default(),
            comments: CList::default(),
            texts: CList::default(),
            suggestions: CList::default(),
            suggestion_comments: CList::default(),
            view: SetsunaView {
                commits: std::ptr::null(),
                n_commits: 0,
                preedit: std::ptr::null(),
                n_preedit: 0,
                focus: -1,
                cursor: 0,
                candidates: std::ptr::null(),
                comments: std::ptr::null(),
                n_candidates: 0,
                selected: 0,
                reading: std::ptr::null(),
                form: std::ptr::null(),
                form_key: std::ptr::null(),
                suggestions: std::ptr::null(),
                suggestion_comments: std::ptr::null(),
                n_suggestions: 0,
                status: std::ptr::null(),
            },
        }
    }
}

/// Stop the daemons and free the session. No notify follows.
///
/// # Safety
/// `s` came from `setsuna_session_new` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_free(s: *mut SetsunaSession) {
    if !s.is_null() {
        guard((), || drop(Box::from_raw(s)));
    }
}

/// Run `f` on the session and refresh its view.
unsafe fn with<R>(s: *mut SetsunaSession, fallback: R, f: impl FnOnce(&mut Session) -> R) -> R {
    let Some(s) = s.as_mut() else {
        return fallback;
    };
    guard(fallback, || {
        let r = f(&mut s.session);
        s.refresh();
        r
    })
}

/// `with` for what needs the whole session (its config and daemons).
unsafe fn with_outer<R>(
    s: *mut SetsunaSession,
    fallback: R,
    f: impl FnOnce(&mut SetsunaSession) -> R,
) -> R {
    let Some(s) = s.as_mut() else {
        return fallback;
    };
    guard(fallback, || {
        let r = f(s);
        s.refresh();
        r
    })
}

/// Reload `config.toml` if it changed (and restart daemons whose settings
/// changed or that exited).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_configure(s: *mut SetsunaSession) {
    with_outer(s, (), |s| {
        s.sync_config("", true);
    });
}

/// `config.toml` over what the addon's install supplies, each where
/// config.toml sets nothing (the Windows installer writes them into it): the
/// mixed input model (`<module_dir>/mixed`), the Latin names dictionary
/// (`<module_dir>/names/latin-names.tsv`), the reading lexicon
/// (`<module_dir>/lexicon/words.lexicon`), the engine's analyzer (the one
/// `*manifest.json` of `<module_dir>/analyzer`) and the reranker's model (the
/// one `*manifest.json` of `<module_dir>/models/rerank`). Every (re)load goes
/// through here, so a reload on activation or focus keeps them.
fn with_module_defaults(mut cfg: Config, module_dir: Option<&Path>) -> Config {
    let Some(dir) = module_dir else {
        return cfg;
    };
    let path = |p: PathBuf| p.to_string_lossy().into_owned();
    if cfg.frontend.mixed_model.is_none() {
        let mixed = dir.join("mixed");
        cfg.frontend.mixed_model = mixed.join("kana.tsv").is_file().then(|| path(mixed));
    }
    if cfg.frontend.name_dictionary_file.is_none() {
        let names = dir.join("names").join("latin-names.tsv");
        cfg.frontend.name_dictionary_file = names.is_file().then(|| path(names));
    }
    if cfg.lexicon.file.is_none() {
        let lexicon = dir.join("lexicon").join("words.lexicon");
        cfg.lexicon.file = lexicon.is_file().then(|| path(lexicon));
    }
    if cfg.engine.analyzer_manifest.is_none() {
        cfg.engine.analyzer_manifest = the_manifest_in(&dir.join("analyzer"));
    }
    if cfg.rerank.manifest.is_none() {
        cfg.rerank.manifest = the_manifest_in(&dir.join("models").join("rerank"));
    }
    cfg
}

/// The one `*manifest.json` of `dir`: none when there is none or several.
fn the_manifest_in(dir: &Path) -> Option<PathBuf> {
    let mut found = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("manifest.json"))
        });
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

/// Apply `config.toml` and `[[apps]]` settings for `program` (empty when
/// unknown), reloading only when the config file, effective app settings or
/// daemon liveness changed. Returns `SETSUNA_APP_START_*` without changing
/// `session.enabled()` (the addon owns per-context on/off state).
///
/// # Safety
/// `s` is a live session; `program` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_configure_app(
    s: *mut SetsunaSession,
    program: *const c_char,
) -> c_int {
    with_outer(s, SETSUNA_APP_START_KEEP, |s| {
        let program = (!program.is_null())
            .then(|| CStr::from_ptr(program).to_string_lossy().into_owned())
            .unwrap_or_default();
        app_start_code(s.sync_config(&program, true))
    })
}

/// `setsuna_session_key`'s `field`: what the field typed into is.
pub const SETSUNA_FIELD_NORMAL: u32 = 0;
/// No context, no rewrite, nothing learned.
pub const SETSUNA_FIELD_PRIVATE: u32 = 1;
/// Direct input: keys that would start a composition go to the app.
pub const SETSUNA_FIELD_DIRECT: u32 = 2;

/// A key press typed into `field` (`SETSUNA_FIELD_*`), with `before_caret`
/// the app's text before the caret (null when it does not tell); true when
/// the IME took it.
///
/// # Safety
/// `s` is a live session; `before_caret` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_key(
    s: *mut SetsunaSession,
    sym: u32,
    code: u32,
    mods: u32,
    field: u32,
    before_caret: *const c_char,
) -> bool {
    with_outer(s, false, |s| {
        if !s.session.is_active() {
            let program = s.last_program.clone();
            s.sync_config(&program, false);
        }
        let field = match field {
            SETSUNA_FIELD_PRIVATE => Field::Private,
            SETSUNA_FIELD_DIRECT => Field::Direct,
            _ => Field::Normal,
        };
        let before = (!before_caret.is_null())
            .then(|| CStr::from_ptr(before_caret).to_string_lossy().into_owned());
        s.session
            .key_in(&Key { sym, code, mods }, field, before.as_deref())
    })
}

/// Rewrite mode is on (a rewriter is running): the mode shows 直.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_rewrite_mode(s: *const SetsunaSession) -> bool {
    guard(false, || {
        s.as_ref().is_some_and(|s| s.session.rewrite_mode())
    })
}

/// How long (ms) until a deferred preedit is shown anyway (0: none waits);
/// the shim then calls `setsuna_session_tick`.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_deferred_ms(s: *const SetsunaSession) -> u32 {
    guard(0, || s.as_ref().map_or(0, |s| s.session.deferred_ms()))
}

/// The deferred preedit's deadline passed; true when the view changed.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_tick(s: *mut SetsunaSession) -> bool {
    with(s, false, |sess| sess.tick())
}

/// Apply the engine's latest result (after a notify); true when the view
/// changed.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_apply_async(s: *mut SetsunaSession) -> bool {
    with(s, false, |sess| sess.apply_async_result())
}

/// Pick candidate `index` of the open list (moves selection there).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_select(s: *mut SetsunaSession, index: usize) -> bool {
    with(s, false, |sess| sess.select(index))
}

/// Pick candidate `index` of the open list or suggestions and accept it
/// (commits it, or fixes the focused segment and advances to the next).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_pick(s: *mut SetsunaSession, index: usize) -> bool {
    with(s, false, |sess| sess.pick(index))
}

/// Commit what is currently shown without waiting for a pending conversion
/// (focus out / input method switch).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_flush(s: *mut SetsunaSession) {
    with(s, (), |sess| {
        setsuna_core::diagnostics::log("fcitx5 flush");
        sess.flush()
    })
}

/// Drop the composition (the app reset the input context).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_reset(s: *mut SetsunaSession) {
    with(s, (), |sess| {
        setsuna_core::diagnostics::log("fcitx5 reset");
        sess.reset()
    })
}

/// The current view; valid until the next call on `s`.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_view(s: *const SetsunaSession) -> *const SetsunaView {
    guard(std::ptr::null(), || {
        s.as_ref().map_or(std::ptr::null(), |s| &s.view)
    })
}

/// Candidates per page of the panel (`[ui.candidate_window] page_size`,
/// 1..=10).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_page_size(s: *const SetsunaSession) -> usize {
    guard(10, || s.as_ref().map_or(10, |s| s.session.page_size()))
}

/// Whether the IME is on (keys go to the app while off).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_enabled(s: *const SetsunaSession) -> bool {
    guard(false, || s.as_ref().is_some_and(|s| s.session.enabled()))
}

/// Turn the IME on or off (the status area's mode action), committing
/// what is composed.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_toggle(s: *mut SetsunaSession) {
    with(s, (), |sess| sess.toggle())
}

/// Set the IME on or off without committing (per-context state restore).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_set_enabled(s: *mut SetsunaSession, on: bool) {
    with(s, (), |sess| sess.set_enabled(on))
}

/// Whether the engine daemon was started (else raw hiragana only).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_has_engine(s: *const SetsunaSession) -> bool {
    guard(false, || s.as_ref().is_some_and(|s| s.session.has_client()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use setsuna_core::rerank_client::{RerankClient, RerankInfo, RerankResult};

    unsafe fn strings(ptrs: *const *const c_char, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| CStr::from_ptr(*ptrs.add(i)).to_string_lossy().into_owned())
            .collect()
    }

    struct DummyReranker;
    impl RerankClient for DummyReranker {
        fn dispatch(&self, _: u64, _: &str, _: &[String], _: Instant) {}
        fn cancel(&self) {}
        fn take_latest(&self) -> Option<RerankResult> {
            None
        }
        fn info(&self) -> Option<RerankInfo> {
            None
        }
    }

    #[test]
    fn the_installed_mixed_model_is_the_default_of_every_load() {
        // A reload on activation or focus once dropped it: mixed input then
        // worked only until the first app switch (commit read as cおっみっと).
        let dir = std::env::temp_dir().join(format!("setsuna-ffi-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("mixed")).unwrap();
        std::fs::write(dir.join("mixed/kana.tsv"), "").unwrap();
        let mixed = dir.join("mixed").to_string_lossy().into_owned();
        let cfg = with_module_defaults(Config::default(), Some(&dir));
        assert_eq!(cfg.frontend.mixed_model.as_deref(), Some(mixed.as_str()));
        let (cfg, _) = with_module_defaults(Config::default(), Some(&dir)).for_app("code");
        assert_eq!(cfg.frontend.mixed_model.as_deref(), Some(mixed.as_str()));
        // config.toml's own value (mixed_model = "" turns it off) wins
        let mut off = Config::default();
        off.frontend.mixed_model = Some(String::new());
        assert_eq!(
            with_module_defaults(off, Some(&dir))
                .frontend
                .mixed_model
                .as_deref(),
            Some("")
        );
        // no model installed: none
        std::fs::remove_file(dir.join("mixed/kana.tsv")).unwrap();
        assert!(with_module_defaults(Config::default(), Some(&dir))
            .frontend
            .mixed_model
            .is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_installed_names_dictionary_lexicon_analyzer_and_rerank_model_are_defaults() {
        let dir = std::env::temp_dir().join(format!("setsuna-ffi-rr-{}", std::process::id()));
        let rerank = dir.join("models/rerank");
        let analyzer = dir.join("analyzer");
        std::fs::create_dir_all(&rerank).unwrap();
        std::fs::create_dir_all(&analyzer).unwrap();
        std::fs::write(analyzer.join("analyzer.manifest.json"), "{}").unwrap();
        std::fs::write(analyzer.join("system.dic.zst"), "").unwrap();
        std::fs::create_dir_all(dir.join("names")).unwrap();
        std::fs::create_dir_all(dir.join("lexicon")).unwrap();
        std::fs::write(dir.join("names/latin-names.tsv"), "GitHub\t1\n").unwrap();
        std::fs::write(dir.join("lexicon/words.lexicon"), "LEXICON1").unwrap();
        std::fs::write(rerank.join("lm.manifest.json"), "{}").unwrap();
        let cfg = with_module_defaults(Config::default(), Some(&dir));
        let names = dir
            .join("names")
            .join("latin-names.tsv")
            .to_string_lossy()
            .into_owned();
        assert_eq!(cfg.frontend.name_dictionary_file, Some(names));
        let lexicon = dir.join("lexicon").join("words.lexicon");
        assert_eq!(
            cfg.lexicon.file,
            Some(lexicon.to_string_lossy().into_owned())
        );
        assert_eq!(cfg.rerank.manifest, Some(rerank.join("lm.manifest.json")));
        assert_eq!(
            cfg.engine.analyzer_manifest,
            Some(analyzer.join("analyzer.manifest.json"))
        );
        // config.toml's own wins (analyzer_manifest = "" turns it off)
        let mut own = Config::default();
        own.rerank.manifest = Some(PathBuf::from("/m.json"));
        own.engine.analyzer_manifest = Some(PathBuf::new());
        let cfg = with_module_defaults(own, Some(&dir));
        assert_eq!(cfg.rerank.manifest, Some(PathBuf::from("/m.json")));
        assert_eq!(cfg.engine.analyzer_manifest, Some(PathBuf::new()));
        // two models: which one is not guessed
        std::fs::write(rerank.join("other.manifest.json"), "{}").unwrap();
        let cfg = with_module_defaults(Config::default(), Some(&dir));
        assert!(cfg.rerank.manifest.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    extern "C" fn no_wake(_: *mut c_void) {}

    #[test]
    fn the_c_view_mirrors_the_session() {
        let mut s = SetsunaSession::empty(None, None, no_wake, std::ptr::null_mut());
        s.sync_config("", true);
        s.refresh();
        let p = &mut s as *mut SetsunaSession;
        unsafe {
            for c in "kana".chars() {
                assert!(setsuna_session_key(p, c as u32, 0, 0, 0, std::ptr::null()));
            }
            let v = &*setsuna_session_view(p);
            assert_eq!(strings(v.preedit, v.n_preedit), vec!["かな"]);
            assert_eq!(v.focus, -1);
            // the caret at the end of what is typed
            assert_eq!(v.cursor, "かな".len());
            assert_eq!(v.n_candidates, 0);
            assert!(setsuna_session_key(p, 0xff0d, 0, 0, 0, std::ptr::null()));
            let v = &*setsuna_session_view(p);
            assert_eq!(strings(v.commits, v.n_commits), vec!["かな"]);
            assert_eq!(v.n_preedit, 0);
            assert!(!setsuna_session_has_engine(p));
        }
    }

    #[test]
    fn configure_app_skips_unchanged_reloads_and_returns_app_start_without_clobbering_enabled() {
        let dir = std::env::temp_dir().join(format!("setsuna-ffi-cfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("config.toml");
        std::fs::write(
            &cfg_path,
            "[ui.candidate_window]\npage_size = 7\n\n[[apps]]\nprocess = \"konsole\"\nstart = \"off\"\n",
        )
        .unwrap();

        let mut s =
            SetsunaSession::empty(None, Some(cfg_path.clone()), no_wake, std::ptr::null_mut());
        s.sync_config("", true);
        s.refresh();
        let p = &mut s as *mut SetsunaSession;
        let konsole = CString::new("konsole").unwrap();
        let firefox = CString::new("firefox").unwrap();
        let dolphin = CString::new("dolphin").unwrap();
        unsafe {
            assert_eq!(setsuna_session_page_size(p), 7);
            assert!(setsuna_session_enabled(p));
            // konsole returns OFF without clobbering enabled itself
            assert_eq!(
                setsuna_session_configure_app(p, konsole.as_ptr()),
                SETSUNA_APP_START_OFF
            );
            assert!(setsuna_session_enabled(p));
            setsuna_session_set_enabled(p, false);
            assert!(!setsuna_session_enabled(p));

            // Switching between unrelated apps hits the fast path (keeps preedit/state)
            assert_eq!(
                setsuna_session_configure_app(p, firefox.as_ptr()),
                SETSUNA_APP_START_KEEP
            );
            setsuna_session_set_enabled(p, true);
            assert!(setsuna_session_key(
                p,
                'a' as u32,
                0,
                0,
                0,
                std::ptr::null()
            ));
            assert_eq!(
                setsuna_session_configure_app(p, dolphin.as_ptr()),
                SETSUNA_APP_START_KEEP
            );
            // Fast path did not tear down the session
            let v = &*setsuna_session_view(p);
            assert_eq!(strings(v.preedit, v.n_preedit), vec!["あ"]);

            // Editing config.toml (different length -> new stamp) reloads settings
            std::fs::write(
                &cfg_path,
                "[ui.candidate_window]\npage_size = 5\n\n[[apps]]\nprocess = \"konsole\"\nstart = \"on\"\n",
            )
            .unwrap();
            assert_eq!(
                setsuna_session_configure_app(p, konsole.as_ptr()),
                SETSUNA_APP_START_ON
            );
            assert_eq!(setsuna_session_page_size(p), 5);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clearing_rerank_manifest_stops_running_reranker_and_engine_change_tracks_key() {
        let dir = std::env::temp_dir().join(format!("setsuna-ffi-rrstop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("config.toml");
        std::fs::write(&cfg_path, "[engine]\nbeam_width = 10\n").unwrap();

        let mut s =
            SetsunaSession::empty(None, Some(cfg_path.clone()), no_wake, std::ptr::null_mut());
        s.sync_config("", true);
        let first_engine_key = s.engine_started.clone().unwrap().0;
        // Simulate a running reranker attached under a previous config
        s.session
            .set_reranker(Some(Box::new(DummyReranker)), &Config::default());
        s.reranker_started = Some(("old-manifest".into(), true));
        assert!(s.session.has_reranker());

        // Reloading with no [rerank] manifest stops the reranker
        s.sync_config("", true);
        assert!(!s.session.has_reranker());
        assert!(s.reranker_started.is_none());

        // Changing [engine] beam_width updates engine_started key (triggers restart path)
        std::fs::write(&cfg_path, "[engine]\nbeam_width = 25\n").unwrap();
        s.sync_config("", true);
        let second_engine_key = s.engine_started.clone().unwrap().0;
        assert_ne!(first_engine_key, second_engine_key);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
