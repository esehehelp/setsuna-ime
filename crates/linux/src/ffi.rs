//! The C ABI the fcitx5 addon calls (`include/setsuna.h`).
//!
//! Every function runs on the addon's event-loop thread except the notify
//! callback, which the engine client calls from its reader thread. Panics
//! never cross the boundary: each entry point catches them and returns its
//! "nothing happened" value.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use setsuna_core::child_daemon;
use setsuna_core::config::{Config, RewriterCfg};
use setsuna_core::engine_client::Notify;
use setsuna_core::rerank_client;
use setsuna_core::rewrite_client::{self, RewriteClient};

use crate::keys::Key;
use crate::session::{Field, Session};

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
    /// Where the engine and models are looked for last (see `load_config`).
    module_dir: Option<PathBuf>,
    /// What the daemons wake the addon with.
    notify: SetsunaNotify,
    user: UserPtr,
    /// The settings the running rewriter was started with.
    rewriter_started: Option<RewriterCfg>,
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

    /// Take up `cfg` and start the daemons it asks for, as the TSF DLL does
    /// on ActivateEx: an engine when there is none or its daemon exited
    /// (with a new bridge, so a new reranker and rewriter too), a reranker
    /// when there is none, the rewriter when it is on and was not started
    /// with these settings; then rewrite mode on (only with a rewriter).
    fn configure(&mut self, cfg: &Config) {
        let module_dir = self.module_dir.clone();
        let module_dir = module_dir.as_deref();
        let client = (!self.session.has_client())
            .then(|| child_daemon::spawn_with_config(cfg, module_dir, self.wake()))
            .flatten();
        let new_bridge = client.is_some();
        if new_bridge {
            setsuna_core::diagnostics::log("fcitx5 engine started");
        }
        self.session.configure_with(cfg, client);
        if new_bridge || !self.session.has_reranker() {
            let reranker = rerank_client::spawn_with_config(cfg, module_dir, self.wake());
            self.session.set_reranker(reranker, cfg);
        } else {
            self.session.set_rerank_config(cfg);
        }
        let needs_rewriter = cfg.rewriter.enabled
            && (!self.session.has_rewriter()
                || self.rewriter_started.as_ref() != Some(&cfg.rewriter));
        if needs_rewriter || !cfg.rewriter.enabled {
            let rewriter = needs_rewriter
                .then(|| rewrite_client::spawn_with_config(cfg, module_dir, self.wake()))
                .flatten();
            self.rewriter_started = rewriter.as_ref().map(|_| cfg.rewriter.clone());
            self.session
                .set_rewriter(rewriter.map(|c| Box::new(c) as Box<dyn RewriteClient>));
        }
        self.session.set_rewrite_mode(true);
        setsuna_core::diagnostics::log(format!(
            "fcitx5 daemons engine={} reranker={} rewriter={}",
            self.session.has_client(),
            self.session.has_reranker(),
            self.session.rewrite_mode()
        ));
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
    setsuna_core::diagnostics::init("fcitx5.log");
    let module_dir = (!module_dir.is_null())
        .then(|| PathBuf::from(CStr::from_ptr(module_dir).to_string_lossy().into_owned()));
    let mut s = Box::new(SetsunaSession {
        session: Session::new(&Config::default(), None),
        module_dir,
        notify,
        user: UserPtr(user),
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
    });
    guard((), || {
        let cfg = load_config(s.module_dir.as_deref());
        s.configure(&cfg);
    });
    s.refresh();
    Box::into_raw(s)
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
    let r = guard(fallback, || f(&mut s.session));
    s.refresh();
    r
}

/// Reload `config.toml` (the input method was activated again), dropping
/// any composition. The engine daemon is kept.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_configure(s: *mut SetsunaSession) {
    setsuna_core::diagnostics::log("fcitx5 configure");
    let cfg = load_config(module_dir_of(s));
    with_outer(s, |s| s.configure(&cfg));
}

/// `with` for what needs the whole session (its daemons).
unsafe fn with_outer(s: *mut SetsunaSession, f: impl FnOnce(&mut SetsunaSession)) {
    let Some(s) = s.as_mut() else {
        return;
    };
    guard((), || f(s));
    s.refresh();
}

/// `config.toml` over what the addon's install supplies, each where
/// config.toml sets nothing (the Windows installer writes them into it): the
/// mixed input model (`<module_dir>/mixed`), the Latin names dictionary
/// (`<module_dir>/names/latin-names.tsv`), the reading lexicon
/// (`<module_dir>/lexicon/words.lexicon`) and the reranker's model (the one
/// `*manifest.json` of `<module_dir>/models/rerank`). Every (re)load goes
/// through here, so a reload on activation or focus keeps them.
fn load_config(module_dir: Option<&Path>) -> Config {
    with_module_defaults(Config::load_default(), module_dir)
}

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

/// The session's module dir.
///
/// # Safety
/// `s` is null or a live session.
unsafe fn module_dir_of<'a>(s: *mut SetsunaSession) -> Option<&'a Path> {
    s.as_ref().and_then(|s| s.module_dir.as_deref())
}

/// `setsuna_session_configure` for the application `program` (the input
/// context's program, empty when unknown): its `[[apps]]` settings over the
/// rest, and its start state.
///
/// # Safety
/// `s` is a live session; `program` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_configure_app(
    s: *mut SetsunaSession,
    program: *const c_char,
) {
    let program = (!program.is_null())
        .then(|| CStr::from_ptr(program).to_string_lossy().into_owned())
        .unwrap_or_default();
    let (cfg, app) = load_config(module_dir_of(s)).for_app(&program);
    setsuna_core::diagnostics::log(format!(
        "fcitx5 configure for {program:?} app settings={}",
        app.is_some()
    ));
    with_outer(s, |s| {
        s.configure(&cfg);
        match app.map(|a| a.start) {
            Some(setsuna_core::config::AppStart::On) => s.session.set_enabled(true),
            Some(setsuna_core::config::AppStart::Off) => s.session.set_enabled(false),
            _ => {}
        }
    });
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
    let field = match field {
        SETSUNA_FIELD_PRIVATE => Field::Private,
        SETSUNA_FIELD_DIRECT => Field::Direct,
        _ => Field::Normal,
    };
    let before = (!before_caret.is_null())
        .then(|| CStr::from_ptr(before_caret).to_string_lossy().into_owned());
    with(s, false, |sess| {
        sess.key_in(&Key { sym, code, mods }, field, before.as_deref())
    })
}

/// Rewrite mode is on (a rewriter is running): the mode shows 直.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_rewrite_mode(s: *const SetsunaSession) -> bool {
    s.as_ref().is_some_and(|s| s.session.rewrite_mode())
}

/// How long (ms) until a deferred preedit is shown anyway (0: none waits);
/// the shim then calls `setsuna_session_tick`.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_deferred_ms(s: *mut SetsunaSession) -> u32 {
    with(s, 0, |sess| sess.deferred_ms())
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

/// Pick candidate `index` of the open list (a click in the panel).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_select(s: *mut SetsunaSession, index: usize) -> bool {
    with(s, false, |sess| sess.select(index))
}

/// Commit the composition (focus out / input method switch).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_flush(s: *mut SetsunaSession) {
    setsuna_core::diagnostics::log("fcitx5 flush");
    with(s, (), |sess| sess.flush())
}

/// Drop the composition (the app reset the input context).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_reset(s: *mut SetsunaSession) {
    setsuna_core::diagnostics::log("fcitx5 reset");
    with(s, (), |sess| sess.reset())
}

/// The current view; valid until the next call on `s`.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_view(s: *const SetsunaSession) -> *const SetsunaView {
    s.as_ref().map_or(std::ptr::null(), |s| &s.view)
}

/// Candidates per page of the panel (`[ui.candidate_window] page_size`,
/// 1..=10).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_page_size(s: *const SetsunaSession) -> usize {
    s.as_ref().map_or(10, |s| s.session.page_size())
}

/// Whether the IME is on (\` / 半角/全角 turn it off: keys go to the app).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_enabled(s: *const SetsunaSession) -> bool {
    s.as_ref().is_some_and(|s| s.session.enabled())
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

/// Whether the engine daemon was started (else raw hiragana only).
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_has_engine(s: *const SetsunaSession) -> bool {
    s.as_ref().is_some_and(|s| s.session.has_client())
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn strings(ptrs: *const *const c_char, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| CStr::from_ptr(*ptrs.add(i)).to_string_lossy().into_owned())
            .collect()
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
    fn an_installed_names_dictionary_lexicon_and_rerank_model_are_defaults() {
        let dir = std::env::temp_dir().join(format!("setsuna-ffi-rr-{}", std::process::id()));
        let rerank = dir.join("models/rerank");
        std::fs::create_dir_all(&rerank).unwrap();
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
        // config.toml's own wins
        let mut own = Config::default();
        own.rerank.manifest = Some(PathBuf::from("/m.json"));
        let cfg = with_module_defaults(own, Some(&dir));
        assert_eq!(cfg.rerank.manifest, Some(PathBuf::from("/m.json")));
        // two models: which one is not guessed
        std::fs::write(rerank.join("other.manifest.json"), "{}").unwrap();
        let cfg = with_module_defaults(Config::default(), Some(&dir));
        assert!(cfg.rerank.manifest.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    extern "C" fn no_wake(_: *mut c_void) {}

    #[test]
    fn the_c_view_mirrors_the_session() {
        let mut s = SetsunaSession {
            session: Session::new(&Config::default(), None),
            module_dir: None,
            notify: no_wake,
            user: UserPtr(std::ptr::null_mut()),
            rewriter_started: None,
            commits: CList::default(),
            preedit: CList::default(),
            candidates: CList::default(),
            comments: CList::default(),
            texts: CList::default(),
            suggestions: CList::default(),
            suggestion_comments: CList::default(),
            view: unsafe { std::mem::zeroed() },
        };
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
}
