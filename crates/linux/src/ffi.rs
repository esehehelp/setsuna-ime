//! The C ABI the fcitx5 addon calls (`include/setsuna.h`).
//!
//! Every function runs on the addon's event-loop thread except the notify
//! callback, which the engine client calls from its reader thread. Panics
//! never cross the boundary: each entry point catches them and returns its
//! "nothing happened" value.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use setsuna_core::child_daemon;
use setsuna_core::config::Config;
use setsuna_core::engine_client::EngineClient;

use crate::keys::Key;
use crate::session::Session;

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
    commits: CList,
    preedit: CList,
    candidates: CList,
    comments: CList,
    /// reading, form, form key
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
        self.texts
            .set(&[v.reading.clone(), form.to_string(), key.to_string()]);
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
        };
    }
}

/// The addon's `user` pointer, carried to the reader thread. The addon keeps
/// it valid until `setsuna_session_free` returns, which joins that thread.
struct UserPtr(*mut c_void);
unsafe impl Send for UserPtr {}

fn guard<R>(fallback: R, f: impl FnOnce() -> R) -> R {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        setsuna_core::diagnostics::log("panic in the fcitx5 frontend");
        fallback
    })
}

/// Start a session: load `config.toml` and spawn the engine daemon.
/// `module_dir` (may be null) is where the engine and models are looked for
/// last. Never returns null.
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
    let session = guard(None, move || {
        let cfg = Config::load_default();
        let user = UserPtr(user);
        let wake = Box::new(move |_req_id: u64| {
            let user = &user;
            notify(user.0)
        });
        let client = child_daemon::spawn_with_config(&cfg, module_dir.as_deref(), wake)
            .map(|c| Box::new(c) as Box<dyn EngineClient>);
        setsuna_core::diagnostics::log(format!("fcitx5 session: engine={}", client.is_some()));
        Some(Session::new(&cfg, client))
    });
    let session = session.unwrap_or_else(|| Session::new(&Config::default(), None));
    let mut s = Box::new(SetsunaSession {
        session,
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
        },
    });
    s.refresh();
    Box::into_raw(s)
}

/// Stop the engine daemon and free the session. No notify follows.
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
    let cfg = Config::load_default();
    with(s, (), |sess| sess.configure(&cfg));
}

/// A key press; true when the IME took it.
///
/// # Safety
/// `s` is a live session.
#[no_mangle]
pub unsafe extern "C" fn setsuna_session_key(
    s: *mut SetsunaSession,
    sym: u32,
    code: u32,
    mods: u32,
) -> bool {
    with(s, false, |sess| sess.key(&Key { sym, code, mods }))
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
    fn the_c_view_mirrors_the_session() {
        let mut s = SetsunaSession {
            session: Session::new(&Config::default(), None),
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
                assert!(setsuna_session_key(p, c as u32, 0, 0));
            }
            let v = &*setsuna_session_view(p);
            assert_eq!(strings(v.preedit, v.n_preedit), vec!["かな"]);
            assert_eq!(v.focus, -1);
            // preedit_cursor_at_start: the caret stays at the start
            assert_eq!(v.cursor, 0);
            assert_eq!(v.n_candidates, 0);
            assert!(setsuna_session_key(p, 0xff0d, 0, 0));
            let v = &*setsuna_session_view(p);
            assert_eq!(strings(v.commits, v.n_commits), vec!["かな"]);
            assert_eq!(v.n_preedit, 0);
            assert!(!setsuna_session_has_engine(p));
        }
    }
}
