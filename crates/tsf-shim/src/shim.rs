//! The DLL's exports, the class factory and the forwarding text service.

use std::cell::RefCell;
use std::ffi::c_void;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use setsuna_tsf_reg::{TextService, IME, SHIM_TEST};
use sha2::{Digest, Sha256};
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::Storage::FileSystem::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::LibraryLoader::*;
use windows::Win32::System::SystemInformation::GetLocalTime;
use windows::Win32::System::Threading::*;
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

/// The version of `SetsunaCoreCreate` this shim calls. The core's other
/// calls are found by their names, each optional: a core without one is
/// used without it.
const CORE_ABI: u32 = 1;
/// Posted to the shim's window to swap the core outside TSF's call.
const WM_APP_SWAP: u32 = WM_APP + 1;
/// Looks at core.txt now and then where the directory cannot be watched.
const TIMER_POLL: usize = 2;
const POLL_MS: u32 = 10_000;
/// Retries a swap a composition put off.
const TIMER_RETRY: usize = 1;
const RETRY_MS: u32 = 1000;
/// The size past which shim.log moves to shim.log.1.
const LOG_LIMIT: u64 = 1 << 20;
/// Names the core in place of core.txt, in this process only
/// (setsuna-tsf-check's, which tries a core before core.txt names it).
const CORE_ENV: &str = "SETSUNA_SHIM_CORE";

struct Module(HMODULE);
// SAFETY: a module handle is a plain address, valid on any thread
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
static MODULE: OnceLock<Module> = OnceLock::new();

fn module() -> HMODULE {
    MODULE.get().map(|m| m.0).unwrap_or_default()
}

/// `%APPDATA%\setsuna-ime\shim.log`: a line with the local time and the
/// process id, whole in one write (every app appends to it); past
/// [`LOG_LIMIT`] it moves to shim.log.1 and starts again.
fn log(message: impl AsRef<str>) {
    let Some(dir) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) else {
        return;
    };
    let dir = PathBuf::from(dir).join("setsuna-ime");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("shim.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > LOG_LIMIT) {
        let _ = std::fs::rename(&path, dir.join("shim.log.1"));
    }
    let t = unsafe { GetLocalTime() };
    let line = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03} [{}] {}\n",
        t.wYear,
        t.wMonth,
        t.wDay,
        t.wHour,
        t.wMinute,
        t.wSecond,
        t.wMilliseconds,
        std::process::id(),
        message.as_ref()
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// `f`, an entry point's body: a panic in it is logged and `fallback`
/// returned, never unwound into the host.
fn guard<T>(what: &str, fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| {
        log(format!("panic in {what}"));
        fallback
    })
}

fn dll_path() -> String {
    let mut buf = [0u16; 1024];
    let len = unsafe { GetModuleFileNameW(module(), &mut buf) } as usize;
    String::from_utf16_lossy(&buf[..len])
}

fn shim_dir() -> Result<PathBuf> {
    PathBuf::from(dll_path())
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::new(E_FAIL, "the shim's directory"))
}

/// When `core.txt` was written last (its size and time), which a core
/// loaded from it carries. None for a core [`CORE_ENV`] names.
type Stamp = Option<(u64, SystemTime)>;

fn file_stamp(path: &Path) -> Stamp {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| Some((m.len(), m.modified().ok()?)))
}

fn core_override() -> Option<String> {
    std::env::var(CORE_ENV).ok().filter(|n| !n.is_empty())
}

/// core.txt's stamp now.
fn core_stamp() -> Stamp {
    if core_override().is_some() {
        return None;
    }
    file_stamp(&shim_dir().ok()?.join("core.txt"))
}

/// The hash a core's file name carries: `<name>_<the first 12 hex digits
/// of its SHA-256>.dll`, the name alone (the core is in the shim's
/// directory). None for any other name.
pub(crate) fn core_hash(name: &str) -> Option<&str> {
    let stem = name.strip_suffix(".dll")?;
    let (head, hash) = stem.rsplit_once('_')?;
    let named = !head.is_empty()
        && head
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    let hashed = hash.len() == 12 && hash.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'));
    (named && hashed).then_some(hash)
}

/// The files found to be what their names say, as they were then (their
/// size and time): read once in a process.
static VERIFIED: Mutex<Vec<(PathBuf, u64, SystemTime)>> = Mutex::new(Vec::new());

/// Whether `path`'s SHA-256 starts with `hash`: a file half written, or one
/// that is not the core its name says, is never loaded.
fn verify(path: &Path, hash: &str) -> Result<()> {
    let fail = |what: String| Error::new(E_FAIL, what);
    let meta = std::fs::metadata(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    let key = (
        path.to_path_buf(),
        meta.len(),
        meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
    );
    let verified = || VERIFIED.lock().unwrap_or_else(|e| e.into_inner());
    if verified().contains(&key) {
        return Ok(());
    }
    let bytes = std::fs::read(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    let digest = Sha256::digest(&bytes);
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    if hex != hash {
        return Err(fail(format!(
            "{} is not the file its name says (SHA-256 {hex}...)",
            path.display()
        )));
    }
    verified().push(key);
    Ok(())
}

/// The core core.txt beside the shim names (or [`CORE_ENV`]), checked, and
/// core.txt's stamp.
fn named_core() -> Result<(PathBuf, Stamp)> {
    let dir = shim_dir()?;
    let (name, stamp) = match core_override() {
        Some(name) => (name, None),
        None => {
            let file = dir.join("core.txt");
            let stamp = file_stamp(&file);
            let text = std::fs::read_to_string(&file)
                .map_err(|e| Error::new(E_FAIL, format!("{}: {e}", file.display())))?;
            (text.trim().to_string(), stamp)
        }
    };
    let hash = core_hash(&name).ok_or_else(|| {
        Error::new(
            E_FAIL,
            format!("core.txt names no core of the shim's directory: {name:?}"),
        )
    })?;
    let path = dir.join(&name);
    verify(&path, hash)?;
    Ok((path, stamp))
}

type CreateFn = unsafe extern "system" fn(u32, *const GUID, *mut *mut c_void) -> HRESULT;
type ComposingFn = unsafe extern "system" fn(*mut c_void) -> BOOL;
type StateFn = unsafe extern "system" fn(*mut c_void, *mut u8, usize) -> usize;
type SwapInFn = unsafe extern "system" fn(*mut c_void, *const u8, usize) -> HRESULT;
type Proc = unsafe extern "system" fn() -> isize;

/// A text service of a core and the core's optional calls.
#[derive(Clone)]
struct Core {
    path: PathBuf,
    stamp: Stamp,
    tip: ITfTextInputProcessorEx,
    composing: Option<ComposingFn>,
    state: Option<StateFn>,
    swap_in: Option<SwapInFn>,
}

impl Core {
    /// A text service of the core `core.txt` names (loaded once a path:
    /// LoadLibrary hands the loaded module back).
    fn load() -> Result<Core> {
        let (path, stamp) = named_core()?;
        unsafe {
            let module = LoadLibraryW(&HSTRING::from(path.as_os_str()))?;
            let create = GetProcAddress(module, s!("SetsunaCoreCreate"))
                .ok_or_else(|| Error::new(E_NOINTERFACE, "no SetsunaCoreCreate"))?;
            let create: CreateFn = std::mem::transmute::<Proc, CreateFn>(create);
            let composing = GetProcAddress(module, s!("SetsunaCoreComposing"))
                .map(|f| std::mem::transmute::<Proc, ComposingFn>(f));
            let state = GetProcAddress(module, s!("SetsunaCoreState"))
                .map(|f| std::mem::transmute::<Proc, StateFn>(f));
            let swap_in = GetProcAddress(module, s!("SetsunaCoreSwapIn"))
                .map(|f| std::mem::transmute::<Proc, SwapInFn>(f));
            let mut raw = std::ptr::null_mut();
            create(CORE_ABI, &ITfTextInputProcessorEx::IID, &mut raw).ok()?;
            if raw.is_null() {
                return Err(Error::new(E_NOINTERFACE, "SetsunaCoreCreate made nothing"));
            }
            let tip = ITfTextInputProcessorEx::from_raw(raw);
            log(format!("core loaded: {}", path.display()));
            Ok(Core {
                path,
                stamp,
                tip,
                composing,
                state,
                swap_in,
            })
        }
    }

    /// Whether the text service holds a composition (a core that cannot
    /// say: never).
    fn composing(&self) -> bool {
        self.composing
            .is_some_and(|f| unsafe { f(self.tip.as_raw()) }.as_bool())
    }

    /// What the text service hands the next one in a swap (none from a
    /// core that cannot say).
    fn state(&self) -> Vec<u8> {
        let Some(f) = self.state else {
            return Vec::new();
        };
        let mut buf = vec![0u8; 4096];
        let n = unsafe { f(self.tip.as_raw(), buf.as_mut_ptr(), buf.len()) };
        if n > buf.len() {
            buf.resize(n, 0);
            if unsafe { f(self.tip.as_raw(), buf.as_mut_ptr(), buf.len()) } != n {
                return Vec::new();
            }
        }
        buf.truncate(n);
        buf
    }

    /// The text service's next activation is a swap's, with `state`; false
    /// for a core that cannot take one (it starts as a new text service).
    fn swap_in(&self, state: &[u8]) -> bool {
        self.swap_in
            .is_some_and(|f| unsafe { f(self.tip.as_raw(), state.as_ptr(), state.len()) }.is_ok())
    }
}

/// The shim's directory watched: a change in it (core.txt written) posts a
/// swap to the shim's window from the thread pool.
struct Watch {
    change: HANDLE,
    wait: HANDLE,
    target: *mut WatchTarget,
}

struct WatchTarget {
    change: HANDLE,
    hwnd: isize,
}

impl Watch {
    fn start(dir: &Path, hwnd: HWND) -> Option<Watch> {
        unsafe {
            let change = FindFirstChangeNotificationW(
                &HSTRING::from(dir.as_os_str()),
                false,
                FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_LAST_WRITE,
            )
            .ok()?;
            let target = Box::into_raw(Box::new(WatchTarget {
                change,
                hwnd: hwnd.0 as isize,
            }));
            let mut wait = HANDLE::default();
            if RegisterWaitForSingleObject(
                &mut wait,
                change,
                Some(on_change),
                Some(target as *const c_void),
                INFINITE,
                WT_EXECUTEDEFAULT,
            )
            .is_err()
            {
                drop(Box::from_raw(target));
                let _ = FindCloseChangeNotification(change);
                return None;
            }
            Some(Watch {
                change,
                wait,
                target,
            })
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        unsafe {
            // returns once a callback running has
            let _ = UnregisterWaitEx(self.wait, INVALID_HANDLE_VALUE);
            let _ = FindCloseChangeNotification(self.change);
            drop(Box::from_raw(self.target));
        }
    }
}

unsafe extern "system" fn on_change(target: *mut c_void, _timed_out: BOOLEAN) {
    let target = &*(target as *const WatchTarget);
    let _ = PostMessageW(
        HWND(target.hwnd as *mut c_void),
        WM_APP_SWAP,
        WPARAM(0),
        LPARAM(0),
    );
    let _ = FindNextChangeNotification(target.change);
}

/// Where the text service is active: its thread manager, client id and
/// flags, and the cookie of the shim's focus sink on it.
struct Active {
    thread_mgr: ITfThreadMgr,
    tid: u32,
    flags: u32,
    sink_cookie: Option<u32>,
}

#[derive(Default)]
struct State {
    core: Option<Core>,
    active: Option<Active>,
    /// The message-only window a swap is posted to.
    hwnd: HWND,
    /// The directory watched while active.
    watch: Option<Watch>,
    /// core.txt's stamp a swap to failed: not tried again until it changes.
    failed: Stamp,
    /// In a swap (between the old core's deactivation and the new one's
    /// activation).
    swapping: bool,
}

#[implement(
    ITfTextInputProcessorEx,
    ITfTextInputProcessor,
    ITfDisplayAttributeProvider,
    ITfFunctionProvider,
    ITfThreadMgrEventSink
)]
struct ShimTip {
    /// The registration it was made for (the IME's, or the test's).
    service: &'static TextService,
    state: RefCell<State>,
}

impl ShimTip_Impl {
    /// The core's text service, loaded when there is none, or loaded again
    /// when core.txt changed while not active.
    fn core(&self) -> Result<Core> {
        let held = self.state.borrow().core.clone();
        if let Some(core) = held {
            if self.state.borrow().active.is_some() || core.stamp == core_stamp() {
                return Ok(core);
            }
        }
        let core = Core::load()?;
        self.state.borrow_mut().core = Some(core.clone());
        Ok(core)
    }

    fn activate(&self, thread_mgr: &ITfThreadMgr, tid: u32, flags: u32) -> Result<()> {
        let core = self.core()?;
        unsafe { core.tip.ActivateEx(thread_mgr, tid, flags)? };
        let sink: ITfThreadMgrEventSink = unsafe { self.cast()? };
        let sink_cookie = thread_mgr
            .cast::<ITfSource>()
            .and_then(|s| unsafe { s.AdviseSink(&ITfThreadMgrEventSink::IID, &sink) })
            .ok();
        let hwnd = {
            let held = self.state.borrow().hwnd;
            if held.0.is_null() {
                create_window(self as *const ShimTip_Impl as isize)
            } else {
                held
            }
        };
        let watch = if hwnd.0.is_null() {
            None
        } else {
            shim_dir().ok().and_then(|dir| Watch::start(&dir, hwnd))
        };
        if watch.is_none() && !hwnd.0.is_null() {
            // core.txt looked at now and then instead
            unsafe { SetTimer(hwnd, TIMER_POLL, POLL_MS, None) };
        }
        log(format!(
            "activated tid={tid} core={} watching={}",
            core.path.display(),
            watch.is_some()
        ));
        let mut st = self.state.borrow_mut();
        st.hwnd = hwnd;
        st.watch = watch;
        st.active = Some(Active {
            thread_mgr: thread_mgr.clone(),
            tid,
            flags,
            sink_cookie,
        });
        Ok(())
    }

    fn deactivate(&self) -> Result<()> {
        // TSF may deactivate the shim from inside the old core's
        // deactivation in a swap (a window it destroys, the thread's last):
        // that core is going already, and the swap activates no other
        let swapping = self.state.borrow().swapping;
        let (core, active, watch) = {
            let mut st = self.state.borrow_mut();
            if !st.hwnd.0.is_null() {
                unsafe {
                    let _ = KillTimer(st.hwnd, TIMER_POLL);
                    let _ = KillTimer(st.hwnd, TIMER_RETRY);
                }
            }
            (st.core.clone(), st.active.take(), st.watch.take())
        };
        drop(watch);
        if let Some(Active {
            thread_mgr,
            sink_cookie: Some(cookie),
            ..
        }) = &active
        {
            if let Ok(source) = thread_mgr.cast::<ITfSource>() {
                let _ = unsafe { source.UnadviseSink(*cookie) };
            }
        }
        log(if swapping {
            "deactivated (in a swap)"
        } else {
            "deactivated"
        });
        match core {
            Some(core) if !swapping => unsafe { core.tip.Deactivate() },
            _ => Ok(()),
        }
    }

    /// Swap the active text service for one of the core core.txt names now,
    /// when it names another and no composition is open (else in a while).
    /// The new one takes the old one's state (the mode as the user left it)
    /// and is activated on the same thread manager, client id and flags.
    fn swap(&self) {
        let (old, thread_mgr, tid, flags, hwnd, failed) = {
            let st = self.state.borrow();
            let (Some(old), Some(active)) = (st.core.clone(), st.active.as_ref()) else {
                return;
            };
            (
                old,
                active.thread_mgr.clone(),
                active.tid,
                active.flags,
                st.hwnd,
                st.failed,
            )
        };
        let now = core_stamp();
        if now == old.stamp || now == failed {
            return;
        }
        if old.composing() {
            log("swap put off: composing");
            unsafe { SetTimer(hwnd, TIMER_RETRY, RETRY_MS, None) };
            return;
        }
        let new = match Core::load() {
            Ok(c) => c,
            Err(e) => {
                log(format!("swap: no new core: {}", e.message()));
                self.state.borrow_mut().failed = now;
                return;
            }
        };
        if new.path == old.path {
            // the same core named again: nothing to swap
            self.state.borrow_mut().core = Some(Core { stamp: now, ..old });
            return;
        }
        let state = old.state();
        self.state.borrow_mut().swapping = true;
        let _ = unsafe { old.tip.Deactivate() };
        self.state.borrow_mut().swapping = false;
        if self.state.borrow().active.is_none() {
            // TSF deactivated the shim meanwhile: the new core is the one
            // its next activation gets
            log(format!(
                "swap: deactivated meanwhile; {} next",
                new.path.display()
            ));
            self.state.borrow_mut().core = Some(new);
            return;
        }
        let handed = new.swap_in(&state);
        match unsafe { new.tip.ActivateEx(&thread_mgr, tid, flags) } {
            Ok(()) => {
                log(format!(
                    "swapped {} -> {} (state handed over: {handed})",
                    old.path.display(),
                    new.path.display()
                ));
                self.state.borrow_mut().core = Some(new);
            }
            Err(e) => {
                log(format!(
                    "swap: the new core did not activate: {e:?}; the old one again"
                ));
                old.swap_in(&state);
                let _ = unsafe { old.tip.ActivateEx(&thread_mgr, tid, flags) };
                self.state.borrow_mut().failed = now;
            }
        }
    }

    /// The core's function provider (reconversion, the touch keyboard's
    /// layout, ...), when it has one.
    fn core_functions(&self) -> Option<ITfFunctionProvider> {
        self.core().ok()?.tip.cast().ok()
    }
}

impl Drop for ShimTip {
    fn drop(&mut self) {
        let st = self.state.get_mut();
        st.watch = None;
        if !st.hwnd.0.is_null() {
            unsafe {
                SetWindowLongPtrW(st.hwnd, GWLP_USERDATA, 0);
                let _ = DestroyWindow(st.hwnd);
            }
        }
    }
}

impl ITfTextInputProcessor_Impl for ShimTip_Impl {
    fn Activate(&self, ptim: Option<&ITfThreadMgr>, tid: u32) -> Result<()> {
        self.ActivateEx(ptim, tid, 0)
    }

    fn Deactivate(&self) -> Result<()> {
        guard("Deactivate", Err(E_FAIL.into()), || self.deactivate())
    }
}

impl ITfTextInputProcessorEx_Impl for ShimTip_Impl {
    fn ActivateEx(&self, ptim: Option<&ITfThreadMgr>, tid: u32, dwflags: u32) -> Result<()> {
        let Some(thread_mgr) = ptim else {
            return Err(E_INVALIDARG.into());
        };
        guard("ActivateEx", Err(E_FAIL.into()), || {
            self.activate(thread_mgr, tid, dwflags).inspect_err(|e| {
                log(format!("ActivateEx failed: {}", e.message()));
            })
        })
    }
}

impl ITfDisplayAttributeProvider_Impl for ShimTip_Impl {
    fn EnumDisplayAttributeInfo(&self) -> Result<IEnumTfDisplayAttributeInfo> {
        guard("EnumDisplayAttributeInfo", Err(E_FAIL.into()), || unsafe {
            self.core()?
                .tip
                .cast::<ITfDisplayAttributeProvider>()?
                .EnumDisplayAttributeInfo()
        })
    }

    fn GetDisplayAttributeInfo(&self, guid: *const GUID) -> Result<ITfDisplayAttributeInfo> {
        guard("GetDisplayAttributeInfo", Err(E_FAIL.into()), || unsafe {
            self.core()?
                .tip
                .cast::<ITfDisplayAttributeProvider>()?
                .GetDisplayAttributeInfo(guid)
        })
    }
}

/// What a core may offer through TSF's function provider, which TSF asks
/// the shim for: the shim cannot be swapped while an app runs, so it hands
/// on whatever a core has, a core of a later version too.
impl ITfFunctionProvider_Impl for ShimTip_Impl {
    fn GetType(&self) -> Result<GUID> {
        Ok(self.service.clsid)
    }

    fn GetDescription(&self) -> Result<BSTR> {
        guard("GetDescription", Err(E_FAIL.into()), || {
            match self.core_functions() {
                Some(f) => unsafe { f.GetDescription() },
                None => Ok(BSTR::from(self.service.name)),
            }
        })
    }

    fn GetFunction(&self, rguid: *const GUID, riid: *const GUID) -> Result<IUnknown> {
        guard("GetFunction", Err(E_FAIL.into()), || {
            match self.core_functions() {
                Some(f) => unsafe { f.GetFunction(rguid, riid) },
                None => Err(E_NOINTERFACE.into()),
            }
        })
    }
}

impl ITfThreadMgrEventSink_Impl for ShimTip_Impl {
    fn OnInitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    fn OnUninitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    /// The focus moved: a newer core in core.txt is swapped in, after TSF's
    /// call (posted to the shim's window). The watch posts it as well; this
    /// is for a directory that could not be watched.
    fn OnSetFocus(
        &self,
        _pdimfocus: Option<&ITfDocumentMgr>,
        _pdimprevfocus: Option<&ITfDocumentMgr>,
    ) -> Result<()> {
        guard("OnSetFocus", Ok(()), || {
            let st = self.state.borrow();
            let now = core_stamp();
            let changed = st
                .core
                .as_ref()
                .is_some_and(|c| c.stamp != now && now != st.failed);
            if changed && st.active.is_some() && !st.hwnd.0.is_null() {
                let _ = unsafe { PostMessageW(st.hwnd, WM_APP_SWAP, WPARAM(0), LPARAM(0)) };
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

/// A message-only window of the shim's class, `tip` (a `*const
/// ShimTip_Impl`) in its user data.
fn create_window(tip: isize) -> HWND {
    unsafe {
        let hinstance = HINSTANCE(module().0);
        let class = w!("SetsunaShimSwap");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance,
            lpszClassName: class,
            ..Default::default()
        };
        // a second registration fails, the first stands
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!(""),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            None,
            hinstance,
            None,
        )
        .unwrap_or_default();
        if !hwnd.0.is_null() {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, tip);
        } else {
            log("no swap window: swaps wait for the next activation");
        }
        hwnd
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let timer = |id| msg == WM_TIMER && wparam.0 == id;
    if msg == WM_APP_SWAP || timer(TIMER_RETRY) || timer(TIMER_POLL) {
        if timer(TIMER_RETRY) {
            let _ = KillTimer(hwnd, TIMER_RETRY);
        }
        let tip = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const ShimTip_Impl;
        if let Some(tip) = tip.as_ref() {
            guard("swap", (), || tip.swap());
        }
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

#[implement(IClassFactory)]
struct ShimFactory {
    service: &'static TextService,
}

impl IClassFactory_Impl for ShimFactory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Option<&IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut c_void,
    ) -> Result<()> {
        if ppvobject.is_null() || riid.is_null() {
            return Err(E_POINTER.into());
        }
        unsafe { *ppvobject = std::ptr::null_mut() };
        if punkouter.is_some() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let tip: IUnknown = ShimTip {
            service: self.service,
            state: RefCell::default(),
        }
        .into();
        unsafe { tip.query(&*riid, ppvobject).ok() }
    }

    fn LockServer(&self, _flock: BOOL) -> Result<()> {
        Ok(())
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(hinstance: HMODULE, reason: u32, _reserved: *mut c_void) -> BOOL {
    const DLL_PROCESS_ATTACH: u32 = 1;
    if reason == DLL_PROCESS_ATTACH {
        let _ = MODULE.set(Module(hinstance));
    }
    TRUE
}

/// The IME's class (the shim registered in the IME DLL's place), or the
/// test's.
#[unsafe(no_mangle)]
extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    if ppv.is_null() || rclsid.is_null() || riid.is_null() {
        return E_POINTER;
    }
    unsafe {
        *ppv = std::ptr::null_mut();
        let service = match *rclsid {
            c if c == IME.clsid => &IME,
            c if c == SHIM_TEST.clsid => &SHIM_TEST,
            _ => return CLASS_E_CLASSNOTAVAILABLE,
        };
        let factory: IUnknown = ShimFactory { service }.into();
        factory.query(&*riid, ppv)
    }
}

/// Never: the cores it loaded stay with it.
#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

/// The shim as the IME: its CLSID with this DLL the server, its profile and
/// categories (`regsvr32`, an install's).
#[unsafe(no_mangle)]
extern "system" fn DllRegisterServer() -> HRESULT {
    match setsuna_tsf_reg::register(&IME, &dll_path()) {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllUnregisterServer() -> HRESULT {
    match setsuna_tsf_reg::unregister(&IME) {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

/// `regsvr32 /n /i:test` (and `/u`): the shim registered beside the IME,
/// under the test's CLSID and profile, "Setsuna (shim test)".
#[unsafe(no_mangle)]
extern "system" fn DllInstall(install: BOOL, cmdline: PCWSTR) -> HRESULT {
    let arg = if cmdline.is_null() {
        String::new()
    } else {
        unsafe { cmdline.to_string() }.unwrap_or_default()
    };
    let service = match arg.trim() {
        "test" => &SHIM_TEST,
        "" | "ime" => &IME,
        _ => return E_INVALIDARG,
    };
    let done = if install.as_bool() {
        setsuna_tsf_reg::register(service, &dll_path())
    } else {
        setsuna_tsf_reg::unregister(service)
    };
    match done {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

#[cfg(test)]
mod tests {
    use super::core_hash;

    #[test]
    fn a_core_is_named_by_its_hash_and_nothing_else() {
        assert_eq!(
            core_hash("setsuna_tsf_0123456789ab.dll"),
            Some("0123456789ab")
        );
        for bad in [
            "setsuna_tsf.dll",
            "setsuna_tsf_0123456789AB.dll",
            "setsuna_tsf_0123456789a.dll",
            "..\\setsuna_tsf_0123456789ab.dll",
            "C:\\x\\setsuna_tsf_0123456789ab.dll",
            "sub/setsuna_tsf_0123456789ab.dll",
            "_0123456789ab.dll",
            "setsuna_tsf_0123456789ab.exe",
        ] {
            assert_eq!(core_hash(bad), None, "{bad}");
        }
    }
}
