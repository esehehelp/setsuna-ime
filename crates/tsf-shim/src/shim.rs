//! The DLL's exports, the class factory and the forwarding text service.

use std::cell::RefCell;
use std::ffi::c_void;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::LibraryLoader::*;
use windows::Win32::System::Registry::*;
use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;

/// The shim's CLSID and profile, beside the IME's (a spike).
/// `{E069C3FF-7026-4E4C-9153-89678BD4B47F}`
pub const CLSID_SHIM: GUID = GUID::from_u128(0xe069_c3ff_7026_4e4c_9153_8967_8bd4_b47f);
/// `{6C6D8475-1FF4-4D9F-80BB-750F50519548}`
pub const GUID_SHIM_PROFILE: GUID = GUID::from_u128(0x6c6d_8475_1ff4_4d9f_80bb_750f_5051_9548);

/// The core's calls this shim makes (`SetsunaCoreCreate`'s `abi`).
const CORE_ABI: u32 = 1;
const LANGID_JAPANESE: u16 = 0x0411;
/// Posted to the shim's window to swap the core outside TSF's call.
const WM_APP_SWAP: u32 = WM_APP + 1;
/// Looks at core.txt while active, a focus change or not (a stat of one
/// file every few seconds).
const TIMER_POLL: usize = 2;
const POLL_MS: u32 = 2000;
/// Retries a swap a composition put off.
const TIMER_RETRY: usize = 1;
const RETRY_MS: u32 = 1000;

// OS-defined capability categories (as the IME's registration.rs)
const GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT: GUID =
    GUID::from_u128(0x13a0_16df_560b_46cd_947a_4c3a_f1e0_e35d);
const GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT: GUID =
    GUID::from_u128(0x2550_4fb4_7bab_4bc1_9c69_cf81_890f_0ef5);
const GUID_TFCAT_TIPCAP_SECUREMODE: GUID =
    GUID::from_u128(0x49d2_f9ce_1f5e_11d7_a6d3_0006_5b84_435c);
const GUID_TFCAT_TIPCAP_UIELEMENTENABLED: GUID =
    GUID::from_u128(0x49d2_f9cf_1f5e_11d7_a6d3_0006_5b84_435c);

const CATEGORIES: [GUID; 8] = [
    GUID_TFCAT_TIP_KEYBOARD,
    GUID_TFCAT_DISPLAYATTRIBUTEPROVIDER,
    GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT,
    GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT,
    GUID_TFCAT_TIPCAP_SECUREMODE,
    GUID_TFCAT_TIPCAP_UIELEMENTENABLED,
    GUID_TFCAT_TIPCAP_COMLESS,
    GUID_TFCAT_TIPCAP_INPUTMODECOMPARTMENT,
];

struct Module(HMODULE);
// SAFETY: a module handle is a plain address, valid on any thread
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
static MODULE: OnceLock<Module> = OnceLock::new();

fn module() -> HMODULE {
    MODULE.get().map(|m| m.0).unwrap_or_default()
}

/// `%APPDATA%\setsuna-ime\shim.log`, with the process id.
fn log(message: impl AsRef<str>) {
    let Some(dir) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) else {
        return;
    };
    let dir = PathBuf::from(dir).join("setsuna-ime");
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("shim.log"))
    {
        let _ = writeln!(f, "[{}] {}", std::process::id(), message.as_ref());
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

/// When `core.txt` was written last (its size and time), which a core
/// loaded from it carries.
type Stamp = Option<(u64, SystemTime)>;

/// `core.txt` beside the shim: the core it names (a path, relative to the
/// shim's directory or absolute) and its stamp.
fn named_core() -> Option<(PathBuf, Stamp)> {
    let dir = PathBuf::from(dll_path()).parent()?.to_path_buf();
    let file = dir.join("core.txt");
    let stamp = std::fs::metadata(&file)
        .ok()
        .and_then(|m| Some((m.len(), m.modified().ok()?)));
    let text = std::fs::read_to_string(&file).ok()?;
    let name = text.trim();
    if name.is_empty() {
        return None;
    }
    let path = if Path::new(name).is_absolute() {
        PathBuf::from(name)
    } else {
        dir.join(name)
    };
    Some((path, stamp))
}

fn core_stamp() -> Stamp {
    named_core().and_then(|(_, s)| s)
}

type CreateFn = unsafe extern "system" fn(u32, *const GUID, *mut *mut c_void) -> HRESULT;
type ComposingFn = unsafe extern "system" fn(*mut c_void) -> BOOL;

/// A text service of a core and the core's composing check.
#[derive(Clone)]
struct Core {
    path: PathBuf,
    stamp: Stamp,
    tip: ITfTextInputProcessorEx,
    composing: ComposingFn,
}

impl Core {
    /// A text service of the core `core.txt` names (loaded once a path:
    /// LoadLibrary hands the loaded module back).
    fn load() -> Result<Core> {
        let (path, stamp) =
            named_core().ok_or_else(|| Error::new(E_FAIL, "no core named in core.txt"))?;
        unsafe {
            let module = LoadLibraryW(&HSTRING::from(path.as_os_str()))?;
            let create = GetProcAddress(module, s!("SetsunaCoreCreate"))
                .ok_or_else(|| Error::new(E_NOINTERFACE, "no SetsunaCoreCreate"))?;
            let composing = GetProcAddress(module, s!("SetsunaCoreComposing"))
                .ok_or_else(|| Error::new(E_NOINTERFACE, "no SetsunaCoreComposing"))?;
            let create: CreateFn = std::mem::transmute(create);
            let composing: ComposingFn = std::mem::transmute(composing);
            let mut raw = std::ptr::null_mut();
            create(CORE_ABI, &ITfTextInputProcessorEx::IID, &mut raw).ok()?;
            let tip = ITfTextInputProcessorEx::from_raw(raw);
            log(format!("core loaded: {}", path.display()));
            Ok(Core {
                path,
                stamp,
                tip,
                composing,
            })
        }
    }

    fn composing(&self) -> bool {
        unsafe { (self.composing)(self.tip.as_raw()).as_bool() }
    }
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
    ITfThreadMgrEventSink
)]
struct ShimTip {
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
        log(format!(
            "activated tid={tid} core={} focus sink={sink_cookie:?} window={:?}",
            core.path.display(),
            hwnd.0
        ));
        if !hwnd.0.is_null() {
            // core.txt looked at now and then, besides the focus changes
            unsafe { SetTimer(hwnd, TIMER_POLL, POLL_MS, None) };
        }
        let mut st = self.state.borrow_mut();
        st.hwnd = hwnd;
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
        let (core, active) = {
            let mut st = self.state.borrow_mut();
            if !st.hwnd.0.is_null() {
                unsafe {
                    let _ = KillTimer(st.hwnd, TIMER_POLL);
                    let _ = KillTimer(st.hwnd, TIMER_RETRY);
                }
            }
            (st.core.clone(), st.active.take())
        };
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
        log(format!(
            "swap: core.txt {now:?}, the core's {:?}",
            old.stamp
        ));
        if old.composing() {
            log("swap put off: composing");
            unsafe { SetTimer(hwnd, TIMER_RETRY, RETRY_MS, None) };
            return;
        }
        let new = match Core::load() {
            Ok(c) => c,
            Err(e) => {
                log(format!("swap: no new core: {e:?}"));
                self.state.borrow_mut().failed = now;
                return;
            }
        };
        if new.path == old.path {
            // the same core named again: nothing to swap
            self.state.borrow_mut().core = Some(Core { stamp: now, ..old });
            return;
        }
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
        match unsafe { new.tip.ActivateEx(&thread_mgr, tid, flags) } {
            Ok(()) => {
                log(format!(
                    "swapped {} -> {}",
                    old.path.display(),
                    new.path.display()
                ));
                self.state.borrow_mut().core = Some(new);
            }
            Err(e) => {
                log(format!(
                    "swap: the new core did not activate: {e:?}; the old one again"
                ));
                let _ = unsafe { old.tip.ActivateEx(&thread_mgr, tid, flags) };
                self.state.borrow_mut().failed = now;
            }
        }
    }
}

impl Drop for ShimTip {
    fn drop(&mut self) {
        let hwnd = self.state.get_mut().hwnd;
        if !hwnd.0.is_null() {
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                let _ = DestroyWindow(hwnd);
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
                log(format!("ActivateEx failed: {e:?}"));
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

impl ITfThreadMgrEventSink_Impl for ShimTip_Impl {
    fn OnInitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    fn OnUninitDocumentMgr(&self, _pdim: Option<&ITfDocumentMgr>) -> Result<()> {
        Ok(())
    }

    /// The focus moved: a newer core in core.txt is swapped in, after TSF's
    /// call (posted to the shim's window).
    fn OnSetFocus(
        &self,
        _pdimfocus: Option<&ITfDocumentMgr>,
        _pdimprevfocus: Option<&ITfDocumentMgr>,
    ) -> Result<()> {
        guard("OnSetFocus", Ok(()), || {
            let st = self.state.borrow();
            let changed = st
                .core
                .as_ref()
                .is_some_and(|c| c.stamp != core_stamp() && core_stamp() != st.failed);
            log(format!(
                "OnSetFocus: core.txt changed={changed} active={} window={}",
                st.active.is_some(),
                !st.hwnd.0.is_null()
            ));
            if changed && st.active.is_some() && !st.hwnd.0.is_null() {
                if let Err(e) = unsafe { PostMessageW(st.hwnd, WM_APP_SWAP, WPARAM(0), LPARAM(0)) }
                {
                    log(format!("OnSetFocus: no swap posted: {e:?}"));
                }
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
struct ShimFactory;

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
        if *rclsid != CLSID_SHIM {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        let factory: IUnknown = ShimFactory.into();
        factory.query(&*riid, ppv)
    }
}

/// Never: the cores it loaded stay with it.
#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

#[unsafe(no_mangle)]
extern "system" fn DllRegisterServer() -> HRESULT {
    match register() {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

#[unsafe(no_mangle)]
extern "system" fn DllUnregisterServer() -> HRESULT {
    match unregister() {
        Ok(()) => S_OK,
        Err(e) => e.code(),
    }
}

fn set_string(key: HKEY, name: PCWSTR, value: &str) -> Result<()> {
    let wide: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        RegSetValueExW(
            key,
            name,
            0,
            REG_SZ,
            Some(std::slice::from_raw_parts(
                wide.as_ptr() as *const u8,
                wide.len() * 2,
            )),
        )
        .ok()
    }
}

/// The CLSID with this DLL as its server, the profile (enabled, not made
/// the active one) and the IME's categories.
fn register() -> Result<()> {
    let clsid = format!("CLSID\\{{{CLSID_SHIM:?}}}");
    unsafe {
        let mut key = HKEY::default();
        RegCreateKeyW(HKEY_CLASSES_ROOT, &HSTRING::from(&clsid), &mut key).ok()?;
        set_string(key, PCWSTR::null(), "Setsuna shim test")?;
        RegCloseKey(key).ok()?;
        let mut inproc = HKEY::default();
        RegCreateKeyW(
            HKEY_CLASSES_ROOT,
            &HSTRING::from(format!("{clsid}\\InProcServer32")),
            &mut inproc,
        )
        .ok()?;
        set_string(inproc, PCWSTR::null(), &dll_path())?;
        set_string(inproc, w!("ThreadingModel"), "Apartment")?;
        RegCloseKey(inproc).ok()?;

        let profiles: ITfInputProcessorProfileMgr =
            CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
        let desc: Vec<u16> = "Setsuna (shim test)".encode_utf16().collect();
        profiles.RegisterProfile(
            &CLSID_SHIM,
            LANGID_JAPANESE,
            &GUID_SHIM_PROFILE,
            &desc,
            &[],
            0,
            HKL::default(),
            0,
            TRUE,
            0,
        )?;
        let categories: ITfCategoryMgr =
            CoCreateInstance(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER)?;
        for cat in CATEGORIES {
            categories.RegisterCategory(&CLSID_SHIM, &cat, &CLSID_SHIM)?;
        }
    }
    Ok(())
}

fn unregister() -> Result<()> {
    let clsid = format!("CLSID\\{{{CLSID_SHIM:?}}}");
    unsafe {
        if let Ok(profiles) = CoCreateInstance::<_, ITfInputProcessorProfileMgr>(
            &CLSID_TF_InputProcessorProfiles,
            None,
            CLSCTX_INPROC_SERVER,
        ) {
            let _ = profiles.UnregisterProfile(&CLSID_SHIM, LANGID_JAPANESE, &GUID_SHIM_PROFILE, 0);
        }
        if let Ok(categories) =
            CoCreateInstance::<_, ITfCategoryMgr>(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER)
        {
            for cat in CATEGORIES {
                let _ = categories.UnregisterCategory(&CLSID_SHIM, &cat, &CLSID_SHIM);
            }
        }
        let _ = RegDeleteKeyW(
            HKEY_CLASSES_ROOT,
            &HSTRING::from(format!("{clsid}\\InProcServer32")),
        );
        let _ = RegDeleteKeyW(HKEY_CLASSES_ROOT, &HSTRING::from(&clsid));
    }
    Ok(())
}
