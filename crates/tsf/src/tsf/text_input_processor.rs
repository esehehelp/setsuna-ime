//! ITfTextInputProcessor(Ex) — main TSF entry point.
//!
//! Implements the interfaces that together give us romaji → hiragana input:
//!   * `ITfTextInputProcessor(Ex)` — activation lifecycle
//!   * `ITfKeyEventSink` — key event dispatch (see `key_event_sink.rs`)
//!   * `ITfCompositionSink` — reentrant-safe composition termination
//!     (see `composition_sink.rs`)
//!
//! Every entry point runs its body through `guarded` (`crate::guard`): a
//! panic stops there and the composition is dropped, never the host app.
//!
//! `config.toml` is read at activation and again when an input starts (or
//! the focus moves) after it changed (`maybe_reload`), so a change saved in
//! the settings GUI applies from the next input.
//!
//! Shape copied from karukan-tsf (MIT/Apache-2.0).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime};

use windows::core::*;
use windows::Win32::Foundation::{E_FAIL, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::globals::{
    dll_add_ref, dll_dir, dll_release, GUID_PRESERVED_KEY_OFF, GUID_PRESERVED_KEY_ON,
    GUID_PRESERVED_KEY_ONOFF, WM_APP_CANDIDATE_CLICK, WM_APP_CONVERT_DONE, WM_APP_FIELD,
    WM_APP_TOGGLE,
};
use crate::keymap::{OnOff, OnOffKeys};
use crate::tsf::mode::{self, ModeItem, ModeState};
use crate::tsf::mode_popup::ModePopup;
use crate::ui::{self, CandidateUi};
use setsuna_core::child_daemon;
use setsuna_core::config::{AppCfg, Config};
use setsuna_core::engine_bridge::EngineBridge;
use setsuna_core::engine_client::{EngineClient, Notify};
use setsuna_core::rerank_client::{self, RerankClient};
use setsuna_core::rewrite_client::{self, RewriteClient};

#[implement(
    ITfTextInputProcessorEx,
    ITfTextInputProcessor,
    ITfKeyEventSink,
    ITfCompositionSink,
    ITfDisplayAttributeProvider,
    ITfThreadMgrEventSink,
    ITfCompartmentEventSink
)]
pub struct SetsunaTextService {
    pub(crate) inner: RefCell<Inner>,
}

pub(crate) struct Inner {
    pub(crate) thread_mgr: Option<ITfThreadMgr>,
    pub(crate) client_id: u32,
    pub(crate) activate_flags: u32,
    pub(crate) keystroke_advised: bool,
    /// `ITfSource` cookie of our `ITfThreadMgrEventSink` (focus changes).
    pub(crate) thread_mgr_sink_cookie: Option<u32>,
    pub(crate) engine: EngineBridge,
    pub(crate) composition: Option<ITfComposition>,
    /// IME ON = romaji→hiragana conversion active; OFF = pass-through so the
    /// user can type English (including capital letters). Toggled by the
    /// on / off keys (`[frontend] toggle_keys`: Hankaku/Zenkaku, Alt+`) or
    /// the taskbar's mode item, through the on / off compartment
    /// (`open_close`).
    pub(crate) enabled: bool,
    /// The thread's on / off compartment and our sink's cookie on it.
    pub(crate) open_close: Option<(ITfCompartment, u32)>,
    /// The taskbar's mode item (あ / A) and what it shows.
    pub(crate) mode: Option<(Rc<ModeState>, ITfLangBarItemButton)>,
    /// The notice shown by the caret when the mode changes.
    pub(crate) mode_popup: Option<ModePopup>,
    /// The IME was turned off because the field that took the focus starts
    /// so (a URL, a number: `key_event_sink::Field`), and the user has not
    /// turned it on or off since: a text field turns it on again.
    pub(crate) auto_off: bool,
    /// The context (its pointer) whose field was checked last (the one with
    /// the focus), and the one the user last turned the IME on or off in:
    /// a field the user switched in keeps that state when it takes the
    /// focus again.
    pub(crate) focused: usize,
    pub(crate) switched_in: usize,
    /// The next on / off change is the IME's own (a field's start state):
    /// no notice by the caret for it.
    pub(crate) quiet_mode_change: bool,
    /// Atom registered via ITfCategoryMgr for the input (dotted-underline)
    /// display attribute. 0 until ActivateEx succeeds.
    pub(crate) atom_input: u32,
    /// Atom for the converted (solid-underline) display attribute.
    pub(crate) atom_converted: u32,
    /// The last preedit string we actually wrote to the composition range.
    /// Consulted before every `UpdatePreedit` so a SetText that would be
    /// a no-op (new text == last text) can be skipped — some TSF hosts
    /// repaint the composition on every SetText even when the content is
    /// identical, which shows up as visible flicker.
    pub(crate) last_preedit: String,
    /// Floating candidate window. None until ActivateEx creates one.
    /// Held behind the `CandidateUi` trait so the GDI popup
    /// (`ui::gdi_window::GdiCandidateUi`) can be swapped for a future
    /// Direct2D / WinUI / Tauri impl without touching the TIP plumbing.
    pub(crate) candidate_window: Option<Rc<RefCell<Box<dyn CandidateUi>>>>,
    /// Configurable symbol outputs keyed by names such as `slash` or
    /// `shift_8`.
    pub(crate) symbols: BTreeMap<String, String>,
    /// The keys that turn the IME on and off (`[frontend] toggle_keys`,
    /// `on_keys`, `off_keys`), and those preserved with TSF (unpreserved in
    /// Deactivate).
    pub(crate) on_off_keys: OnOffKeys,
    pub(crate) preserved: Vec<(GUID, TF_PRESERVEDKEY)>,
    /// Msg-only HWND the engine daemon's reader thread `PostMessage`s to
    /// when a conversion result is ready. Destroyed FIRST in Deactivate so
    /// stray async callbacks can't reach a half-torn service.
    pub(crate) notify_hwnd: HWND,
    /// What the running rewriter was started with.
    pub(crate) rewriter_started: Option<setsuna_core::config::RewriterCfg>,
    /// The config the service runs with (`[[apps]]` applied), the file it
    /// was read from and that file's modification time and size then.
    pub(crate) cfg: Config,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) config_stamp: Option<(SystemTime, u64)>,
    /// When the file was last looked at from the key path.
    pub(crate) config_checked: Option<Instant>,
    /// What the engine client, the reranker and the candidate window were
    /// made with (`[engine]`, `[rerank]`, `[ui]` as `Debug` prints them):
    /// made again only when that changed.
    pub(crate) engine_started: Option<String>,
    pub(crate) reranker_started: Option<String>,
    pub(crate) window_made: Option<String>,
    /// No engine client: when to try starting one again.
    pub(crate) engine_retry_at: Option<Instant>,
}

/// How often the key path looks at `config.toml`'s modification time.
const CONFIG_CHECK_EVERY: Duration = Duration::from_secs(1);
/// How long to wait before starting an engine client again after one
/// could not be had (no model, no shared engine to reach).
const ENGINE_RETRY: Duration = Duration::from_secs(30);

#[allow(clippy::new_without_default)]
impl SetsunaTextService {
    pub fn new() -> Self {
        setsuna_core::diagnostics::log("SetsunaTextService::new");
        dll_add_ref();
        Self {
            inner: RefCell::new(Inner {
                thread_mgr: None,
                client_id: 0,
                activate_flags: 0,
                keystroke_advised: false,
                thread_mgr_sink_cookie: None,
                engine: EngineBridge::new(None),
                composition: None,
                enabled: true,
                open_close: None,
                mode: None,
                mode_popup: None,
                auto_off: false,
                focused: 0,
                switched_in: 0,
                quiet_mode_change: false,
                atom_input: 0,
                atom_converted: 0,
                last_preedit: String::new(),
                candidate_window: None,
                symbols: setsuna_core::config::default_symbol_map(),
                on_off_keys: OnOffKeys::default(),
                preserved: Vec::new(),
                notify_hwnd: HWND::default(),
                rewriter_started: None,
                cfg: Config::default(),
                config_path: None,
                config_stamp: None,
                config_checked: None,
                engine_started: None,
                reranker_started: None,
                window_made: None,
                engine_retry_at: None,
            }),
        }
    }
}

// ---- notify HWND for async convert results ----

fn create_notify_window(service_ptr: *const core::ffi::c_void) -> HWND {
    use crate::globals::DLL_INSTANCE;
    use windows::Win32::Foundation::{GetLastError, HINSTANCE};

    let hinstance = DLL_INSTANCE
        .get()
        .map(|s| HINSTANCE(s.0 .0))
        .unwrap_or_default();
    let class_name = format!(
        "SetsunaAsyncNotify_{:x}_{:x}",
        hinstance.0 as usize, service_ptr as usize
    );
    let class_w: Vec<u16> = class_name
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let class_pcw = PCWSTR(class_w.as_ptr());

    unsafe {
        let wc = WNDCLASSW {
            lpfnWndProc: Some(notify_wnd_proc),
            hInstance: hinstance,
            lpszClassName: class_pcw,
            ..Default::default()
        };
        let atom = RegisterClassW(&wc);
        if atom == 0 {
            setsuna_core::diagnostics::log(format!(
                "RegisterClassW failed class={class_name} err={:?}",
                GetLastError()
            ));
        }
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class_pcw,
            w!(""),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            HWND(-3isize as *mut _), // HWND_MESSAGE
            None,
            None,
            None,
        )
        .unwrap_or_default();
        if hwnd.0 as usize != 0 {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, service_ptr as isize);
            setsuna_core::diagnostics::log(format!(
                "notify_hwnd={:#x} class={class_name}",
                hwnd.0 as isize
            ));
        } else {
            setsuna_core::diagnostics::log(format!(
                "CreateWindowExW notify failed class={class_name} err={:?}",
                GetLastError()
            ));
        }
        hwnd
    }
}

unsafe extern "system" fn notify_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let ours = matches!(
        msg,
        WM_APP_CONVERT_DONE | WM_APP_TOGGLE | WM_APP_CANDIDATE_CLICK | WM_APP_FIELD
    ) || (msg == WM_TIMER && wparam.0 == TIMER_DEFERRED);
    if !ours {
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    if msg == WM_TIMER {
        let _ = KillTimer(hwnd, TIMER_DEFERRED);
    }
    let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const SetsunaTextService_Impl;
    if ptr.is_null() {
        return LRESULT(0);
    }
    // Safety: the service outlives its notify HWND — Deactivate / Drop tear
    // the HWND down (and clear USERDATA) BEFORE letting COM release the
    // service.
    let svc = unsafe { &*ptr };
    svc.guarded("notify window", None, (), || match msg {
        WM_APP_CONVERT_DONE => {
            setsuna_core::diagnostics::log(format!(
                "notify_wnd_proc WM_APP_CONVERT_DONE req_id={}",
                wparam.0
            ));
            on_async_done(svc);
        }
        WM_TIMER => on_deferred_deadline(svc),
        WM_APP_TOGGLE => {
            let _ = svc.toggle_enabled(None);
        }
        WM_APP_CANDIDATE_CLICK => on_candidate_click(svc, wparam.0),
        WM_APP_FIELD => svc.on_field(wparam.0 & 1 != 0, wparam.0 & 2 != 0),
        _ => {}
    });
    LRESULT(0)
}

/// The timer of a deferred preedit (`[rerank] defer_ms`).
const TIMER_DEFERRED: usize = 0x5E7;

/// Set the deferred preedit's timer when one waits.
fn arm_deferred(svc: &SetsunaTextService_Impl) {
    let (wait, hwnd) = {
        let Ok(inner) = svc.inner.try_borrow() else {
            return;
        };
        (inner.engine.deferred_for(), inner.notify_hwnd)
    };
    if let Some(wait) = wait {
        if hwnd.0 as usize != 0 {
            let ms = wait.as_millis().min(10_000) as u32 + 1;
            unsafe {
                SetTimer(hwnd, TIMER_DEFERRED, ms, None);
            }
        }
    }
}

/// The deferred preedit's deadline: the engine's top without the reranked one.
fn on_deferred_deadline(svc: &SetsunaTextService_Impl) {
    let actions = {
        let Ok(mut inner) = svc.inner.try_borrow_mut() else {
            return;
        };
        inner.engine.tick()
    };
    if actions.is_empty() {
        return;
    }
    if let Some(context) = focused_context(svc) {
        if let Err(e) = crate::tsf::key_event_sink::apply_actions(svc, &context, actions) {
            setsuna_core::diagnostics::log(format!("deferred apply failed: {e:?}"));
        }
    }
}

/// A candidate was clicked: pick it and commit, as its digit key would.
fn on_candidate_click(svc: &SetsunaTextService_Impl, index: usize) {
    setsuna_core::diagnostics::log(format!("candidate click index={index}"));
    let actions = {
        let Ok(mut inner) = svc.inner.try_borrow_mut() else {
            return;
        };
        let picked = inner.engine.handle_select(index);
        if !picked.consumed {
            return;
        }
        let mut actions = picked.actions;
        actions.extend(inner.engine.handle_enter().actions);
        actions
    };
    if let Some(context) = focused_context(svc) {
        if let Err(e) = crate::tsf::key_event_sink::apply_actions(svc, &context, actions) {
            setsuna_core::diagnostics::log(format!("candidate click apply failed: {e:?}"));
        }
    }
}

/// The context of the document that has the focus.
pub(crate) fn focused_context(svc: &SetsunaTextService_Impl) -> Option<ITfContext> {
    let tm = svc.inner.try_borrow().ok()?.thread_mgr.clone()?;
    unsafe { tm.GetFocus().ok()?.GetTop().ok() }
}

/// Drain the latest daemon result and apply the resulting actions (preedit
/// swap and/or candidate-window refresh) through the shared `apply_actions`
/// path, which also maintains the `last_preedit` flicker-dedup state.
fn on_async_done(svc: &SetsunaTextService_Impl) {
    setsuna_core::diagnostics::log("on_async_done");
    // Pull the actions out under a short borrow, then drop it before the
    // edit session so nothing re-enters while the TSF callback runs.
    let actions = {
        let mut inner = match svc.inner.try_borrow_mut() {
            Ok(i) => i,
            Err(_) => {
                setsuna_core::diagnostics::log("on_async_done skipped: inner borrowed");
                return;
            }
        };
        inner.engine.apply_async_result()
    };
    if actions.is_empty() {
        setsuna_core::diagnostics::log("on_async_done no actions");
        return;
    }
    setsuna_core::diagnostics::log(format!("on_async_done actions={actions:?}"));

    let Some(context) = focused_context(svc) else {
        setsuna_core::diagnostics::log("on_async_done no focused context");
        return;
    };
    match crate::tsf::key_event_sink::apply_actions(svc, &context, actions) {
        Ok(()) => setsuna_core::diagnostics::log("on_async_done applied"),
        Err(e) => setsuna_core::diagnostics::log(format!("on_async_done apply failed: {e:?}")),
    }
    arm_deferred(svc);
}

fn resolve_display_atoms() -> (u32, u32) {
    use crate::globals::{GUID_DISPLAY_ATTRIBUTE_CONVERTED, GUID_DISPLAY_ATTRIBUTE_INPUT};
    use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
    unsafe {
        let cat_mgr: Result<ITfCategoryMgr> =
            CoCreateInstance(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER);
        match cat_mgr {
            Ok(m) => {
                let input = m.RegisterGUID(&GUID_DISPLAY_ATTRIBUTE_INPUT).unwrap_or(0);
                let converted = m
                    .RegisterGUID(&GUID_DISPLAY_ATTRIBUTE_CONVERTED)
                    .unwrap_or(0);
                (input, converted)
            }
            Err(e) => {
                tracing::warn!("CategoryMgr unavailable, no display attributes: {}", e);
                (0, 0)
            }
        }
    }
}

// ---- config.toml ----

/// Whether this process is a Store app's sandbox (an AppContainer: the
/// Start menu's search, a UWP app).
pub(crate) fn in_app_container() -> bool {
    static IN: once_cell::sync::OnceCell<bool> = once_cell::sync::OnceCell::new();
    *IN.get_or_init(|| setsuna_pipe::win::is_app_container().unwrap_or(false))
}

/// `config.toml`: `%APPDATA%\setsuna-ime\config.toml`. In an AppContainer
/// %APPDATA% may name the package's own folder; the user's file is then
/// found from where the DLL is installed (`%LOCALAPPDATA%\setsuna-ime`,
/// beside Roaming). The installer lets AppContainers read that one file,
/// not the directory (learning, credentials).
fn config_path() -> Option<PathBuf> {
    let appdata = Config::appdata_config_path();
    if !in_app_container() {
        return appdata;
    }
    let beside_dll = dll_dir().and_then(|d| {
        Some(
            d.parent()?
                .parent()?
                .join("Roaming")
                .join("setsuna-ime")
                .join("config.toml"),
        )
    });
    [appdata.clone(), beside_dll]
        .into_iter()
        .flatten()
        .find(|p| p.is_file())
        .or(appdata)
}

/// The file's modification time and size: a save changes them.
fn config_stamp(path: Option<&Path>) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path?).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// Read `path`; a missing file is the defaults.
fn read_config(path: Option<&Path>) -> std::result::Result<Config, String> {
    match path {
        Some(p) if p.exists() => Config::load_from_path(p).map_err(|e| format!("{e:#}")),
        _ => Ok(Config::default()),
    }
}

/// This process's settings: the file's, with its `[[apps]]` entry applied.
fn config_for_process(cfg: &Config) -> (Config, Option<AppCfg>) {
    let process = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (cfg, app) = cfg.for_app(&process);
    setsuna_core::diagnostics::log(format!("process {process} app settings={}", app.is_some()));
    (cfg, app)
}

/// A callback posting `msg` with a request id to the notify HWND: how the
/// clients' threads wake the UI thread. HWND is !Send; the raw handle is
/// not, and a stale one just makes PostMessageW fail harmlessly (the
/// service tears the window down before dropping the clients).
fn poster(hwnd: HWND) -> Notify {
    let hwnd_raw = hwnd.0 as isize;
    Box::new(move |req_id: u64| unsafe {
        let _ = PostMessageW(
            HWND(hwnd_raw as *mut core::ffi::c_void),
            WM_APP_CONVERT_DONE,
            WPARAM(req_id as usize),
            LPARAM(0),
        );
    })
}

impl SetsunaTextService_Impl {
    /// Expose `cast` so sibling modules can reach the other interfaces
    /// implemented on this object (e.g. key_event_sink needs the
    /// `ITfCompositionSink` interface to hand to `StartComposition`).
    pub(crate) fn cast_interface<T: windows::core::Interface>(&self) -> Result<T> {
        unsafe { self.cast() }
    }

    /// Run `f`, the body of the entry point `what`: after a panic in it,
    /// drop what is composed (`recover`) and return `fallback`.
    pub(crate) fn guarded<T>(
        &self,
        what: &str,
        context: Option<&ITfContext>,
        fallback: T,
        f: impl FnOnce() -> T,
    ) -> T {
        match crate::guard::run(what, f) {
            Some(v) => v,
            None => {
                self.recover(context);
                fallback
            }
        }
    }

    /// After a panic: forget what was composed, hide the window and end the
    /// composition where it stands (its text stays in the document), so the
    /// next key starts clean. A bridge that panics again doing so is
    /// replaced, its client started again at the next input.
    pub(crate) fn recover(&self, context: Option<&ITfContext>) {
        let _ = crate::guard::run("recover", || {
            let (composition, window) = {
                let Ok(mut inner) = self.inner.try_borrow_mut() else {
                    return;
                };
                let inner = &mut *inner;
                if crate::guard::run("discard", || inner.engine.discard()).is_none() {
                    inner.engine = EngineBridge::new(None);
                    inner.engine_started = None;
                    inner.engine_retry_at = None;
                    // the next input sets the new bridge up as the config says
                    inner.config_stamp = None;
                }
                inner.last_preedit.clear();
                (inner.composition.take(), inner.candidate_window.clone())
            };
            if let Some(cw) = window {
                if let Ok(mut cw) = cw.try_borrow_mut() {
                    cw.hide();
                }
            }
            let context = context.cloned().or_else(|| focused_context(self));
            if let (Some(composition), Some(context)) = (composition, context) {
                crate::tsf::edit_session::end_composition(self, &context, composition);
            }
        });
    }

    /// Hide the candidate window if it is up (the borrow of the service is
    /// released before the window's is taken).
    pub(crate) fn hide_window(&self) {
        let window = match self.inner.try_borrow() {
            Ok(inner) => inner.candidate_window.clone(),
            Err(_) => return,
        };
        if let Some(cw) = window {
            if let Ok(mut cw) = cw.try_borrow_mut() {
                if cw.is_visible() {
                    cw.hide();
                }
            }
        }
    }

    /// Read `config.toml` again when it changed since it was read (and, with
    /// no engine client, start one again now and then), unless something
    /// is composed: called as an input starts (`force` false: at most once a
    /// second) and when the focus moves (`force`).
    pub(crate) fn maybe_reload(&self, force: bool) {
        let (path, stamp, retry_engine, state) = {
            let Ok(mut inner) = self.inner.try_borrow_mut() else {
                return;
            };
            if inner.engine.is_active() || inner.thread_mgr.is_none() {
                return;
            }
            let now = Instant::now();
            if !force
                && inner
                    .config_checked
                    .is_some_and(|t| now.duration_since(t) < CONFIG_CHECK_EVERY)
            {
                return;
            }
            inner.config_checked = Some(now);
            let alive = inner.engine.has_client();
            let retry_engine = !alive && inner.engine_retry_at.is_none_or(|t| now >= t);
            // the taskbar item says when the engine went (or came back)
            let state = inner
                .mode
                .as_ref()
                .map(|(s, _)| s.clone())
                .filter(|s| s.converts.get() != alive)
                .map(|s| (s, alive));
            (
                inner.config_path.clone(),
                inner.config_stamp,
                retry_engine,
                state,
            )
        };
        if let Some((state, alive)) = state {
            if !alive {
                setsuna_core::diagnostics::log(
                    "the conversion engine went: the IME types kana only",
                );
            }
            state.converts.set(alive);
            state.set(state.on.get());
        }
        let new_stamp = config_stamp(path.as_deref());
        let changed = new_stamp != stamp;
        if !changed && !retry_engine {
            return;
        }
        let cfg = if changed {
            match read_config(path.as_deref()) {
                Ok(file) => {
                    setsuna_core::diagnostics::log("config.toml changed: applying it");
                    Some(config_for_process(&file).0)
                }
                Err(e) => {
                    // a file being written, or one with a mistake: the
                    // settings in use stay
                    setsuna_core::diagnostics::log(format!("config.toml not applied: {e}"));
                    None
                }
            }
        } else {
            None
        };
        self.inner.borrow_mut().config_stamp = new_stamp;
        let cfg = cfg.unwrap_or_else(|| self.inner.borrow().cfg.clone());
        self.apply_config(cfg);
    }

    /// Set the service up as `cfg` says: the candidate window (made again
    /// when `[ui]` changed), the engine client (started again when
    /// `[engine]` changed or there is none), the reranker, the bridge's
    /// frontend settings, learning, lexicon and predictor, the rewriter,
    /// the mode notice and the on / off keys. Never while composing.
    fn apply_config(&self, cfg: Config) {
        let notify_hwnd = self.inner.borrow().notify_hwnd;

        // The candidate window, made (hidden) through `ui::default_ui` so a
        // swap is a one-line change; a click in it picks through the notify
        // HWND, on the UI thread like a key.
        let window_key = format!("{:?}", cfg.ui);
        let window = {
            let inner = self.inner.borrow();
            (inner.candidate_window.is_none() || inner.window_made.as_ref() != Some(&window_key))
                .then(|| {
                    let w: Rc<RefCell<Box<dyn CandidateUi>>> =
                        Rc::new(RefCell::new(ui::default_ui(&cfg)));
                    let hwnd_raw = notify_hwnd.0 as isize;
                    w.borrow_mut()
                        .set_on_select(Box::new(move |index: usize| unsafe {
                            let _ = PostMessageW(
                                HWND(hwnd_raw as *mut core::ffi::c_void),
                                WM_APP_CANDIDATE_CLICK,
                                WPARAM(index),
                                LPARAM(0),
                            );
                        }));
                    w
                })
        };

        // The engine client: started again when its settings changed or it
        // is gone (the daemon exited, or none could be had).
        let engine_key = format!("{:?}", cfg.engine);
        let needs_client = {
            let inner = self.inner.borrow();
            !inner.engine.has_client() || inner.engine_started.as_ref() != Some(&engine_key)
        };
        let client: Option<Box<dyn EngineClient>> = if needs_client {
            child_daemon::spawn_with_config(&cfg, dll_dir().as_deref(), poster(notify_hwnd))
        } else {
            None
        };
        if needs_client && client.is_none() {
            setsuna_core::diagnostics::log(
                "no conversion engine: the IME types kana only (see the lines above)",
            );
        }
        setsuna_core::diagnostics::log(format!(
            "engine_client={} reused={}",
            client.is_some() || !needs_client,
            !needs_client
        ));
        // the reranker wakes the UI thread the same way; a new bridge needs a new one
        let rerank_key = format!("{:?}", cfg.rerank);
        let needs_reranker = needs_client || {
            let inner = self.inner.borrow();
            !inner.engine.has_reranker() || inner.reranker_started.as_ref() != Some(&rerank_key)
        };
        let reranker: Option<Box<dyn RerankClient>> = if needs_reranker {
            rerank_client::spawn_with_config(&cfg, dll_dir().as_deref(), poster(notify_hwnd))
        } else {
            None
        };
        setsuna_core::diagnostics::log(format!(
            "rerank_client={} reused={}",
            reranker.is_some() || !needs_reranker,
            !needs_reranker
        ));

        let (on_off_keys, bad_keys) = OnOffKeys::from_config(&cfg.frontend);
        if !bad_keys.is_empty() {
            setsuna_core::diagnostics::log(format!("on / off keys not understood: {bad_keys:?}"));
        }
        {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            if needs_client {
                inner.engine_started = client.as_ref().map(|_| engine_key);
                // a client that goes (no shared engine to reach) is
                // started again no sooner than this
                inner.engine_retry_at = Some(Instant::now() + ENGINE_RETRY);
                inner.engine = EngineBridge::with_frontend(client, &cfg.frontend);
            } else {
                inner.engine.reset_preserving_client(&cfg.frontend);
            }
            if needs_reranker {
                inner.reranker_started = reranker.as_ref().map(|_| rerank_key);
                inner.engine.set_reranker(reranker, &cfg.rerank);
            } else {
                inner.engine.set_rerank_config(&cfg.rerank);
            }
            inner
                .engine
                .set_learning(setsuna_core::learning::Learning::open(&cfg.learning));
            inner.engine.set_ngram(&cfg.ngram);
            inner.engine.set_lexicon(&cfg.lexicon);
            inner
                .engine
                .set_hidden(setsuna_core::hidden::Hidden::open());
            inner.engine.use_user_dictionary_file();
            inner.engine.open_predictor(&cfg.frontend);
            // the GDI window shows 10 per page, labelled 1–9, 0
            inner
                .engine
                .set_candidate_paging(crate::tsf::candidate_window::page_size(&cfg), true);
            if let Some(w) = window {
                inner.candidate_window = Some(w);
                inner.window_made = Some(window_key);
            }
            inner.symbols = cfg.frontend.symbols.clone();
            inner.on_off_keys = on_off_keys;
            inner.mode_popup = Some(ModePopup::new(&cfg.ui.candidate_window));
            inner.last_preedit.clear();
        }
        // The rewriter: started when on, again when its settings changed;
        // with it, the conversion key also asks for a rewrite.
        let needs_rewriter = cfg.rewriter.enabled && {
            let inner = self.inner.borrow();
            !inner.engine.has_rewriter() || inner.rewriter_started.as_ref() != Some(&cfg.rewriter)
        };
        if needs_rewriter || !cfg.rewriter.enabled {
            let rewriter = needs_rewriter
                .then(|| {
                    rewrite_client::spawn_with_config(
                        &cfg,
                        dll_dir().as_deref(),
                        poster(notify_hwnd),
                    )
                })
                .flatten();
            let mut inner = self.inner.borrow_mut();
            inner.rewriter_started = rewriter.as_ref().map(|_| cfg.rewriter.clone());
            inner
                .engine
                .set_rewriter(rewriter.map(|c| Box::new(c) as Box<dyn RewriteClient>));
        }
        let (rewriting, converts, state) = {
            let mut inner = self.inner.borrow_mut();
            inner.engine.set_rewrite_mode(true);
            (
                inner.engine.rewrite_mode(),
                inner.engine.has_client(),
                inner.mode.as_ref().map(|(s, _)| s.clone()),
            )
        };
        setsuna_core::diagnostics::log(format!("rewriter on={rewriting}"));
        if let Some(state) = state {
            state.rewrite.set(rewriting);
            state.converts.set(converts);
            state.set(state.on.get());
        }
        self.inner.borrow_mut().cfg = cfg;
        self.preserve_keys();
    }

    /// Preserve the on / off keys with TSF so `OnPreservedKey` fires on a
    /// single press whatever the host's own dispatch does: one GUID per
    /// role. Those preserved before and no longer configured are let go.
    fn preserve_keys(&self) {
        let (thread_mgr, tid, advised, wanted, old) = {
            let inner = self.inner.borrow();
            let wanted: Vec<(GUID, TF_PRESERVEDKEY)> = inner
                .on_off_keys
                .preservable()
                .into_iter()
                .map(|(role, vk, mods)| {
                    let guid = match role {
                        OnOff::Toggle => GUID_PRESERVED_KEY_ONOFF,
                        OnOff::On => GUID_PRESERVED_KEY_ON,
                        OnOff::Off => GUID_PRESERVED_KEY_OFF,
                    };
                    (
                        guid,
                        TF_PRESERVEDKEY {
                            uVKey: vk,
                            uModifiers: mods,
                        },
                    )
                })
                .collect();
            (
                inner.thread_mgr.clone(),
                inner.client_id,
                inner.keystroke_advised,
                wanted,
                inner.preserved.clone(),
            )
        };
        let same = |a: &(GUID, TF_PRESERVEDKEY), b: &(GUID, TF_PRESERVEDKEY)| {
            a.0 == b.0 && a.1.uVKey == b.1.uVKey && a.1.uModifiers == b.1.uModifiers
        };
        let Some(tm) = thread_mgr.filter(|_| advised) else {
            return;
        };
        if wanted.len() == old.len() && wanted.iter().zip(&old).all(|(a, b)| same(a, b)) {
            return;
        }
        let Ok(km) = tm.cast::<ITfKeystrokeMgr>() else {
            return;
        };
        unpreserve(&km, &old);
        let desc: Vec<u16> = "IME On/Off".encode_utf16().collect();
        let mut preserved = Vec::new();
        for (guid, key) in wanted {
            match unsafe { km.PreserveKey(tid, &guid, &key, &desc) } {
                Ok(()) => preserved.push((guid, key)),
                Err(e) => setsuna_core::diagnostics::log(format!(
                    "PreserveKey vk={:#x} mods={:#x} failed: {e:?}",
                    key.uVKey, key.uModifiers
                )),
            }
        }
        setsuna_core::diagnostics::log(format!("preserved {} on / off keys", preserved.len()));
        self.inner.borrow_mut().preserved = preserved;
    }

    /// Activate on `thread_mgr` (`ActivateEx`'s body).
    fn activate(&self, thread_mgr: &ITfThreadMgr, tid: u32, dwflags: u32) -> Result<()> {
        // Defensive: if this service instance is already active (TSF
        // reused it without a matching Deactivate — rare but possible in
        // quick switching scenarios), tear the previous activation down
        // first so we never double-create the worker / notify HWND.
        if self.inner.borrow().keystroke_advised {
            let _ = self.deactivate();
        }

        // Resolve display attribute atoms once per activation.
        let (atom_input, atom_converted) = resolve_display_atoms();

        // Load TOML config so the same instance can flow to both the
        // daemon spawn (engine flags) and the UI factory (font/colors/page).
        // A file that does not parse is the defaults. As this application
        // uses it (`[[apps]]`): the DLL runs in its process.
        let path = config_path();
        let stamp = config_stamp(path.as_deref());
        let file = read_config(path.as_deref()).unwrap_or_else(|e| {
            setsuna_core::diagnostics::log(format!("config.toml not read, defaults: {e}"));
            Config::default()
        });
        let (cfg, app) = config_for_process(&file);
        setsuna_core::diagnostics::log(format!(
            "config {:?} app container={} manifest={:?} model_dir={:?} engine={:?}",
            path,
            in_app_container(),
            cfg.engine.model_manifest,
            cfg.engine.model_dir,
            cfg.engine.engine_exe
        ));

        // Create the notify HWND once per COM object and keep it across
        // transient Deactivate/ActivateEx cycles so the cold-starting daemon
        // has a stable callback target.
        let notify_hwnd = {
            let existing = self.inner.borrow().notify_hwnd;
            if existing.0 as usize != 0 {
                existing
            } else {
                create_notify_window(self as *const _ as *const core::ffi::c_void)
            }
        };
        {
            let mut inner = self.inner.borrow_mut();
            inner.thread_mgr = Some(thread_mgr.clone());
            inner.client_id = tid;
            inner.activate_flags = dwflags;
            inner.atom_input = atom_input;
            inner.atom_converted = atom_converted;
            inner.notify_hwnd = notify_hwnd;
            inner.config_path = path;
            inner.config_stamp = stamp;
            inner.config_checked = Some(Instant::now());
        }
        self.apply_config(cfg);

        unsafe {
            let key_sink: ITfKeyEventSink = self.cast_interface()?;
            let keystroke_mgr: ITfKeystrokeMgr = thread_mgr.cast()?;
            keystroke_mgr.AdviseKeyEventSink(tid, &key_sink, true)?;
            setsuna_core::diagnostics::log("AdviseKeyEventSink OK");
        }
        self.inner.borrow_mut().keystroke_advised = true;
        // the on / off keys (`[frontend] toggle_keys`, ...)
        self.preserve_keys();
        // Focus changes between documents (another field or window on this
        // thread) make the left context stale.
        let sink_cookie = unsafe {
            let sink: ITfThreadMgrEventSink = self.cast_interface()?;
            thread_mgr
                .cast::<ITfSource>()
                .and_then(|source| source.AdviseSink(&ITfThreadMgrEventSink::IID, &sink))
                .ok()
        };
        setsuna_core::diagnostics::log(format!("AdviseSink thread mgr event sink={sink_cookie:?}"));
        self.inner.borrow_mut().thread_mgr_sink_cookie = sink_cookie;

        // The mode: the thread's on / off compartment (unset: on, as
        // Microsoft asks of IMEs), the taskbar item showing it, the notice.
        let compartment = mode::open_close(thread_mgr);
        let start = app
            .as_ref()
            .map_or(setsuna_core::config::AppStart::Keep, |a| a.start);
        let on = match (compartment.as_ref(), start) {
            // the application's start state, `[[apps]] start`
            (Some(c), setsuna_core::config::AppStart::On | setsuna_core::config::AppStart::Off) => {
                let on = start == setsuna_core::config::AppStart::On;
                let _ = mode::write_open(c, tid, on);
                on
            }
            (Some(c), _) => mode::read_open(c).unwrap_or_else(|| {
                let _ = mode::write_open(c, tid, true);
                true
            }),
            (None, s) => s != setsuna_core::config::AppStart::Off,
        };
        let open_close = compartment.and_then(|c| {
            let sink: ITfCompartmentEventSink = self.cast_interface().ok()?;
            let cookie = unsafe {
                c.cast::<ITfSource>()
                    .ok()?
                    .AdviseSink(&ITfCompartmentEventSink::IID, &sink)
                    .ok()?
            };
            Some((c, cookie))
        });
        let hwnd_raw = notify_hwnd.0 as isize;
        let state = ModeState::new(
            on,
            Box::new(move || unsafe {
                let _ = PostMessageW(
                    HWND(hwnd_raw as *mut core::ffi::c_void),
                    WM_APP_TOGGLE,
                    WPARAM(0),
                    LPARAM(0),
                );
            }),
        );
        {
            let inner = self.inner.borrow();
            state.rewrite.set(inner.engine.rewrite_mode());
            state.converts.set(inner.engine.has_client());
        }
        let item: ITfLangBarItemButton = ModeItem::new(state.clone()).into();
        let added = unsafe {
            thread_mgr
                .cast::<ITfLangBarItemMgr>()
                .and_then(|mgr| mgr.AddItem(&item))
        };
        setsuna_core::diagnostics::log(format!(
            "mode on={on} compartment={} langbar item={added:?}",
            open_close.is_some()
        ));
        {
            let mut inner = self.inner.borrow_mut();
            inner.enabled = on;
            inner.auto_off = false;
            inner.quiet_mode_change = false;
            inner.open_close = open_close;
            inner.mode = added.is_ok().then_some((state, item));
        }
        // a URL or a number field focused already starts off
        if let Some(context) = focused_context(self) {
            crate::tsf::key_event_sink::check_field(self, &context);
        }

        tracing::info!("Setsuna TSF ActivateEx (tid={}, flags={:#x})", tid, dwflags);
        setsuna_core::diagnostics::log("ActivateEx OK");
        Ok(())
    }

    /// `Deactivate`'s body.
    fn deactivate(&self) -> Result<()> {
        setsuna_core::diagnostics::log("Deactivate");
        // Do not tear down the notify HWND or daemon here. TSF can
        // deactivate/reactivate a service repeatedly during focus/profile
        // churn, and the int4b16 daemon has a noticeable cold-start. Keeping
        // it alive for the COM object's lifetime lets the reader reach Ready
        // instead of losing every in-flight startup to a transient Deactivate.
        let (thread_mgr, client_id, advised, preserved) = {
            let mut inner = self.inner.borrow_mut();
            inner.composition = None;
            if let Some(ref cw) = inner.candidate_window {
                if let Ok(mut cw) = cw.try_borrow_mut() {
                    cw.hide();
                }
            }
            inner.candidate_window = None;
            inner.window_made = None;
            let advised = inner.keystroke_advised;
            inner.keystroke_advised = false;
            let tm = inner.thread_mgr.take();
            let cid = inner.client_id;
            inner.client_id = 0;
            inner.activate_flags = 0;
            (tm, cid, advised, std::mem::take(&mut inner.preserved))
        };
        let sink_cookie = self.inner.borrow_mut().thread_mgr_sink_cookie.take();
        if let (Some(tm), Some(cookie)) = (&thread_mgr, sink_cookie) {
            unsafe {
                if let Ok(source) = tm.cast::<ITfSource>() {
                    let _ = source.UnadviseSink(cookie);
                }
            }
        }
        // the mode: off the taskbar, no more compartment changes
        let (mode, open_close) = {
            let mut inner = self.inner.borrow_mut();
            inner.mode_popup = None;
            (inner.mode.take(), inner.open_close.take())
        };
        if let (Some(tm), Some((_, item))) = (&thread_mgr, mode) {
            unsafe {
                if let Ok(mgr) = tm.cast::<ITfLangBarItemMgr>() {
                    let _ = mgr.RemoveItem(&item);
                }
            }
        }
        if let Some((compartment, cookie)) = open_close {
            unsafe {
                if let Ok(source) = compartment.cast::<ITfSource>() {
                    let _ = source.UnadviseSink(cookie);
                }
            }
        }

        if advised {
            if let Some(tm) = &thread_mgr {
                if let Ok(km) = tm.cast::<ITfKeystrokeMgr>() {
                    unpreserve(&km, &preserved);
                    unsafe {
                        let _ = km.UnadviseKeyEventSink(client_id);
                    }
                }
            }
        }
        tracing::info!("Setsuna TSF Deactivate");
        Ok(())
    }
}

/// Let the preserved keys go.
fn unpreserve(km: &ITfKeystrokeMgr, keys: &[(GUID, TF_PRESERVEDKEY)]) {
    for (guid, key) in keys {
        unsafe {
            let _ = km.UnpreserveKey(guid, key);
        }
    }
}

impl Drop for SetsunaTextService {
    fn drop(&mut self) {
        setsuna_core::diagnostics::log("SetsunaTextService::drop");
        // Defensive teardown: if the service drops without a prior
        // `Deactivate` (rare but seen during rapid IME switches when TSF
        // releases the object as part of its own unwind) we still need
        // to kill the notify HWND and worker thread, otherwise:
        //   * WndProc receives WM_APP_CONVERT_DONE on a freed service →
        //     use-after-free,
        //   * daemon I/O threads stay alive past DLL unload → access
        //     violation when the loader zero-fills our code section.
        let _ = crate::guard::run("drop", || {
            if let Ok(mut inner) = self.inner.try_borrow_mut() {
                if inner.notify_hwnd.0 as usize != 0 {
                    unsafe {
                        SetWindowLongPtrW(inner.notify_hwnd, GWLP_USERDATA, 0);
                        let _ = DestroyWindow(inner.notify_hwnd);
                    }
                    inner.notify_hwnd = HWND::default();
                }
                // Replacing the bridge drops the EngineClient → the daemon
                // child is killed and its I/O threads joined.
                inner.engine = EngineBridge::new(None);
                inner.candidate_window = None;
            }
        });
        dll_release();
    }
}

impl ITfThreadMgrEventSink_Impl for SetsunaTextService_Impl {
    fn OnInitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    fn OnUninitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    /// Another document took the focus: its left context is not ours, no
    /// window stays up from the last one, a saved config applies, and its
    /// field may start with the IME off (a URL, a number).
    fn OnSetFocus(
        &self,
        pdimfocus: Option<&ITfDocumentMgr>,
        _pdimprevfocus: Option<&ITfDocumentMgr>,
    ) -> Result<()> {
        self.guarded("thread mgr OnSetFocus", None, Ok(()), || {
            setsuna_core::diagnostics::log("thread mgr OnSetFocus: forget context");
            if let Ok(mut inner) = self.inner.try_borrow_mut() {
                inner.engine.forget_context();
            }
            self.hide_window();
            self.maybe_reload(true);
            if let Some(context) = pdimfocus.and_then(|d| unsafe { d.GetTop().ok() }) {
                crate::tsf::key_event_sink::check_field(self, &context);
            }
            Ok(())
        })
    }

    fn OnPushContext(&self, _pic: Option<&ITfContext>) -> Result<()> {
        Ok(())
    }

    fn OnPopContext(&self, _pic: Option<&ITfContext>) -> Result<()> {
        Ok(())
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)] // OnChange takes *const GUID per the TSF interface
impl ITfCompartmentEventSink_Impl for SetsunaTextService_Impl {
    /// The on / off compartment changed (the key, the taskbar item, another
    /// program): turn the IME on or off to match.
    fn OnChange(&self, rguid: *const GUID) -> Result<()> {
        if rguid.is_null() || unsafe { *rguid } != GUID_COMPARTMENT_KEYBOARD_OPENCLOSE {
            return Ok(());
        }
        self.guarded("compartment OnChange", None, Ok(()), || {
            let compartment = self
                .inner
                .borrow()
                .open_close
                .as_ref()
                .map(|(c, _)| c.clone());
            if let Some(on) = compartment.as_ref().and_then(mode::read_open) {
                setsuna_core::diagnostics::log(format!("open/close compartment: {on}"));
                self.set_mode(on, None);
            }
            Ok(())
        })
    }
}

impl ITfTextInputProcessor_Impl for SetsunaTextService_Impl {
    fn Activate(&self, ptim: Option<&ITfThreadMgr>, tid: u32) -> Result<()> {
        setsuna_core::diagnostics::log(format!("Activate tid={tid}"));
        self.ActivateEx(ptim, tid, 0)
    }

    fn Deactivate(&self) -> Result<()> {
        self.guarded("Deactivate", None, Ok(()), || self.deactivate())
    }
}

impl ITfTextInputProcessorEx_Impl for SetsunaTextService_Impl {
    fn ActivateEx(&self, ptim: Option<&ITfThreadMgr>, tid: u32, dwflags: u32) -> Result<()> {
        setsuna_core::diagnostics::log(format!("ActivateEx tid={tid} flags={dwflags:#x}"));
        let Some(thread_mgr) = ptim else {
            setsuna_core::diagnostics::log("ActivateEx missing thread manager");
            return Err(Error::from(windows::Win32::Foundation::E_INVALIDARG));
        };
        self.guarded("ActivateEx", None, Err(E_FAIL.into()), || {
            self.activate(thread_mgr, tid, dwflags)
        })
    }
}
