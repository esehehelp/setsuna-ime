//! ITfTextInputProcessor(Ex) — main TSF entry point.
//!
//! Implements the interfaces that together give us romaji → hiragana input:
//!   * `ITfTextInputProcessor(Ex)` — activation lifecycle
//!   * `ITfKeyEventSink` — key event dispatch (see `key_event_sink.rs`)
//!   * `ITfCompositionSink` — reentrant-safe composition termination
//!     (see `composition_sink.rs`)
//!
//! Shape copied from karukan-tsf (MIT/Apache-2.0).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use windows::core::*;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::globals::{
    dll_add_ref, dll_dir, dll_release, WM_APP_CANDIDATE_CLICK, WM_APP_CONVERT_DONE, WM_APP_TOGGLE,
};
use crate::tsf::mode::{self, ModeItem, ModeState};
use crate::tsf::mode_popup::ModePopup;
use crate::ui::{self, CandidateUi};
use setsuna_core::child_daemon;
use setsuna_core::config::Config;
use setsuna_core::engine_bridge::EngineBridge;
use setsuna_core::engine_client::EngineClient;
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
    /// user can type English (including capital letters). Toggled by
    /// Hankaku/Zenkaku (VK_KANJI / VK_OEM_ATTN), Alt+` or the taskbar's mode
    /// item, through the on / off compartment (`open_close`).
    pub(crate) enabled: bool,
    /// The thread's on / off compartment and our sink's cookie on it.
    pub(crate) open_close: Option<(ITfCompartment, u32)>,
    /// The taskbar's mode item (あ / A) and what it shows.
    pub(crate) mode: Option<(Rc<ModeState>, ITfLangBarItemButton)>,
    /// The notice shown by the caret when the mode changes.
    pub(crate) mode_popup: Option<ModePopup>,
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
    /// `shift_8`. Loaded on ActivateEx so config edits apply on reactivation.
    pub(crate) symbols: BTreeMap<String, String>,
    /// Msg-only HWND the engine daemon's reader thread `PostMessage`s to
    /// when a conversion result is ready. Destroyed FIRST in Deactivate so
    /// stray async callbacks can't reach a half-torn service.
    pub(crate) notify_hwnd: HWND,
    /// What the running rewriter was started with.
    pub(crate) rewriter_started: Option<setsuna_core::config::RewriterCfg>,
}

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
                atom_input: 0,
                atom_converted: 0,
                last_preedit: String::new(),
                candidate_window: None,
                symbols: setsuna_core::config::default_symbol_map(),
                notify_hwnd: HWND::default(),
                rewriter_started: None,
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
    if msg == WM_APP_CONVERT_DONE {
        setsuna_core::diagnostics::log(format!(
            "notify_wnd_proc WM_APP_CONVERT_DONE req_id={}",
            wparam.0
        ));
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const SetsunaTextService_Impl;
        if !ptr.is_null() {
            // Safety: the service outlives its notify HWND — Deactivate
            // tears the HWND down (and clears USERDATA) BEFORE letting COM
            // release the service.
            let svc = unsafe { &*ptr };
            on_async_done(svc);
        }
        return LRESULT(0);
    }
    if msg == WM_APP_TOGGLE {
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const SetsunaTextService_Impl;
        if !ptr.is_null() {
            // Safety: as for WM_APP_CONVERT_DONE above.
            let svc = unsafe { &*ptr };
            let _ = svc.toggle_enabled(None);
        }
        return LRESULT(0);
    }
    if msg == WM_APP_CANDIDATE_CLICK {
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const SetsunaTextService_Impl;
        if !ptr.is_null() {
            // Safety: as for WM_APP_CONVERT_DONE above.
            let svc = unsafe { &*ptr };
            on_candidate_click(svc, wparam.0);
        }
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
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
    let tm = svc.inner.borrow().thread_mgr.clone()?;
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

    // Grab the current focused context via thread_mgr.
    let thread_mgr = svc.inner.borrow().thread_mgr.clone();
    let Some(tm) = thread_mgr else {
        setsuna_core::diagnostics::log("on_async_done no thread_mgr");
        return;
    };
    let context: ITfContext = unsafe {
        let doc = match tm.GetFocus() {
            Ok(d) => d,
            Err(_) => return,
        };
        match doc.GetTop() {
            Ok(c) => c,
            Err(e) => {
                setsuna_core::diagnostics::log(format!("on_async_done GetTop failed: {e:?}"));
                return;
            }
        }
    };

    match crate::tsf::key_event_sink::apply_actions(svc, &context, actions) {
        Ok(()) => setsuna_core::diagnostics::log("on_async_done applied"),
        Err(e) => setsuna_core::diagnostics::log(format!("on_async_done apply failed: {e:?}")),
    }
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

impl SetsunaTextService_Impl {
    /// Expose `cast` so sibling modules can reach the other interfaces
    /// implemented on this object (e.g. key_event_sink needs the
    /// `ITfCompositionSink` interface to hand to `StartComposition`).
    pub(crate) fn cast_interface<T: windows::core::Interface>(&self) -> Result<T> {
        unsafe { self.cast() }
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

    fn OnSetFocus(
        &self,
        _pdimfocus: Option<&ITfDocumentMgr>,
        _pdimprevfocus: Option<&ITfDocumentMgr>,
    ) -> Result<()> {
        setsuna_core::diagnostics::log("thread mgr OnSetFocus: forget context");
        if let Ok(mut inner) = self.inner.try_borrow_mut() {
            inner.engine.forget_context();
        }
        Ok(())
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
    }
}

impl ITfTextInputProcessor_Impl for SetsunaTextService_Impl {
    fn Activate(&self, ptim: Option<&ITfThreadMgr>, tid: u32) -> Result<()> {
        setsuna_core::diagnostics::log(format!("Activate tid={tid}"));
        self.ActivateEx(ptim, tid, 0)
    }

    fn Deactivate(&self) -> Result<()> {
        setsuna_core::diagnostics::log("Deactivate");
        // Do not tear down the notify HWND or daemon here. TSF can
        // deactivate/reactivate a service repeatedly during focus/profile
        // churn, and the int4b16 daemon has a noticeable cold-start. Keeping
        // it alive for the COM object's lifetime lets the reader reach Ready
        // instead of losing every in-flight startup to a transient Deactivate.
        let (thread_mgr, client_id, advised) = {
            let mut inner = self.inner.borrow_mut();
            inner.composition = None;
            if let Some(ref cw) = inner.candidate_window {
                cw.borrow_mut().hide();
            }
            inner.candidate_window = None;
            let advised = inner.keystroke_advised;
            inner.keystroke_advised = false;
            let tm = inner.thread_mgr.take();
            let cid = inner.client_id;
            inner.client_id = 0;
            inner.activate_flags = 0;
            (tm, cid, advised)
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
                unsafe {
                    if let Ok(km) = tm.cast::<ITfKeystrokeMgr>() {
                        use crate::globals::GUID_PRESERVED_KEY_ONOFF;
                        let key = TF_PRESERVEDKEY {
                            uVKey: 0x19,
                            uModifiers: 0,
                        };
                        let _ = km.UnpreserveKey(&GUID_PRESERVED_KEY_ONOFF, &key);
                        let _ = km.UnadviseKeyEventSink(client_id);
                    }
                }
            }
        }
        tracing::info!("Setsuna TSF Deactivate");
        Ok(())
    }
}

impl ITfTextInputProcessorEx_Impl for SetsunaTextService_Impl {
    fn ActivateEx(&self, ptim: Option<&ITfThreadMgr>, tid: u32, dwflags: u32) -> Result<()> {
        setsuna_core::diagnostics::log(format!("ActivateEx tid={tid} flags={dwflags:#x}"));
        let Some(thread_mgr) = ptim else {
            setsuna_core::diagnostics::log("ActivateEx missing thread manager");
            return Err(Error::from(windows::Win32::Foundation::E_INVALIDARG));
        };

        // Defensive: if this service instance is already active (TSF
        // reused it without a matching Deactivate — rare but possible in
        // quick switching scenarios), tear the previous activation down
        // first so we never double-create the worker / notify HWND.
        if self.inner.borrow().keystroke_advised {
            let _ = self.Deactivate();
        }

        // Resolve display attribute atoms once per activation.
        let (atom_input, atom_converted) = resolve_display_atoms();

        // Load TOML config so the same instance can flow to both the
        // daemon spawn (engine flags) and the UI factory (font/colors/page).
        // `load_default` already falls back to defaults on parse error.
        // as this application uses it (`[[apps]]`): the DLL runs in its process
        let process = std::env::current_exe()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (cfg, app) = Config::load_default().for_app(&process);
        setsuna_core::diagnostics::log(format!("process {process} app settings={}", app.is_some()));
        setsuna_core::diagnostics::log(format!(
            "config manifest={:?} model_dir={:?} engine={:?}",
            cfg.engine.model_manifest, cfg.engine.model_dir, cfg.engine.engine_exe
        ));

        // Create candidate window (hidden). Drop happens on Deactivate via
        // `Option::take` so the HWND doesn't leak across activations. Built
        // through `ui::default_ui` so a swap is a one-line change.
        let candidate_window: Rc<RefCell<Box<dyn CandidateUi>>> =
            Rc::new(RefCell::new(ui::default_ui(&cfg)));

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

        // a click in the candidate window picks through the notify HWND, on
        // the UI thread like a key
        {
            let hwnd_raw = notify_hwnd.0 as isize;
            candidate_window
                .borrow_mut()
                .set_on_select(Box::new(move |index: usize| unsafe {
                    let _ = PostMessageW(
                        HWND(hwnd_raw as *mut core::ffi::c_void),
                        WM_APP_CANDIDATE_CLICK,
                        WPARAM(index),
                        LPARAM(0),
                    );
                }));
        }

        let needs_client = !self.inner.borrow().engine.has_client();
        let client: Option<Box<dyn EngineClient>> = if needs_client {
            // Wake the UI thread through the notify HWND. HWND is !Send;
            // the raw handle is not, and a stale one just makes
            // PostMessageW fail harmlessly (the service tears the window
            // down before dropping the client).
            let hwnd_raw = notify_hwnd.0 as isize;
            let notify = Box::new(move |req_id: u64| unsafe {
                let _ = PostMessageW(
                    HWND(hwnd_raw as *mut core::ffi::c_void),
                    WM_APP_CONVERT_DONE,
                    WPARAM(req_id as usize),
                    LPARAM(0),
                );
            });
            child_daemon::spawn_with_config(&cfg, dll_dir().as_deref(), notify)
                .map(|c| Box::new(c) as Box<dyn EngineClient>)
        } else {
            None
        };
        setsuna_core::diagnostics::log(format!(
            "engine_client={} reused={}",
            client.is_some() || !needs_client,
            !needs_client
        ));
        // the reranker wakes the UI thread the same way; a new bridge needs a new one
        let needs_reranker = needs_client || !self.inner.borrow().engine.has_reranker();
        let reranker: Option<Box<dyn RerankClient>> = if needs_reranker {
            let hwnd_raw = notify_hwnd.0 as isize;
            let notify = Box::new(move |req_id: u64| unsafe {
                let _ = PostMessageW(
                    HWND(hwnd_raw as *mut core::ffi::c_void),
                    WM_APP_CONVERT_DONE,
                    WPARAM(req_id as usize),
                    LPARAM(0),
                );
            });
            rerank_client::spawn_with_config(&cfg, dll_dir().as_deref(), notify)
                .map(|c| Box::new(c) as Box<dyn RerankClient>)
        } else {
            None
        };
        setsuna_core::diagnostics::log(format!(
            "rerank_client={} reused={}",
            reranker.is_some() || !needs_reranker,
            !needs_reranker
        ));

        {
            let mut inner = self.inner.borrow_mut();
            inner.thread_mgr = Some(thread_mgr.clone());
            inner.client_id = tid;
            inner.activate_flags = dwflags;
            if needs_client {
                inner.engine = EngineBridge::with_frontend(client, &cfg.frontend);
            } else {
                inner.engine.reset_preserving_client(&cfg.frontend);
            }
            if needs_reranker {
                inner.engine.set_reranker(reranker, &cfg.rerank);
            } else {
                inner.engine.set_rerank_config(&cfg.rerank);
            }
            inner
                .engine
                .set_learning(setsuna_core::learning::Learning::open(&cfg.learning));
            inner
                .engine
                .set_hidden(setsuna_core::hidden::Hidden::open());
            inner.engine.use_user_dictionary_file();
            inner.engine.open_predictor(&cfg.frontend);
            // the GDI window shows 10 per page, labelled 1–9, 0
            inner
                .engine
                .set_candidate_paging(crate::tsf::candidate_window::page_size(&cfg), true);
            inner.atom_input = atom_input;
            inner.atom_converted = atom_converted;
            inner.candidate_window = Some(candidate_window);
            inner.symbols = cfg.frontend.symbols.clone();
            inner.notify_hwnd = notify_hwnd;
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
                    let hwnd_raw = notify_hwnd.0 as isize;
                    let notify = Box::new(move |req_id: u64| unsafe {
                        let _ = PostMessageW(
                            HWND(hwnd_raw as *mut core::ffi::c_void),
                            WM_APP_CONVERT_DONE,
                            WPARAM(req_id as usize),
                            LPARAM(0),
                        );
                    });
                    rewrite_client::spawn_with_config(&cfg, dll_dir().as_deref(), notify)
                })
                .flatten();
            let mut inner = self.inner.borrow_mut();
            inner.rewriter_started = rewriter.as_ref().map(|_| cfg.rewriter.clone());
            inner
                .engine
                .set_rewriter(rewriter.map(|c| Box::new(c) as Box<dyn RewriteClient>));
        }
        let rewriting = {
            let mut inner = self.inner.borrow_mut();
            inner.engine.set_rewrite_mode(true);
            inner.engine.rewrite_mode()
        };
        setsuna_core::diagnostics::log(format!("rewriter on={rewriting}"));

        unsafe {
            let key_sink: ITfKeyEventSink = self.cast_interface()?;
            let keystroke_mgr: ITfKeystrokeMgr = thread_mgr.cast()?;
            keystroke_mgr.AdviseKeyEventSink(tid, &key_sink, true)?;
            setsuna_core::diagnostics::log("AdviseKeyEventSink OK");

            // Register Hankaku/Zenkaku as a preserved key so our
            // `OnPreservedKey` fires on a single press regardless of the
            // host's own dispatch routing. Description uses UTF-16 "IME
            // On/Off".
            use crate::globals::GUID_PRESERVED_KEY_ONOFF;
            let key = TF_PRESERVEDKEY {
                uVKey: 0x19, // VK_KANJI (Hankaku/Zenkaku on JP keyboards)
                uModifiers: 0,
            };
            let desc: Vec<u16> = "IME On/Off".encode_utf16().collect();
            let _ = keystroke_mgr.PreserveKey(tid, &GUID_PRESERVED_KEY_ONOFF, &key, &desc);
        }
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
        {
            let mut inner = self.inner.borrow_mut();
            inner.keystroke_advised = true;
            inner.thread_mgr_sink_cookie = sink_cookie;
        }

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
        state.rewrite.set(rewriting);
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
            inner.open_close = open_close;
            inner.mode = added.is_ok().then_some((state, item));
            inner.mode_popup = Some(ModePopup::new(&cfg.ui.candidate_window));
        }

        tracing::info!("Setsuna TSF ActivateEx (tid={}, flags={:#x})", tid, dwflags);
        setsuna_core::diagnostics::log("ActivateEx OK");
        Ok(())
    }
}
