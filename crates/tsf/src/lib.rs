//! Setsuna TSF — Windows Text Services Framework IME DLL.
//!
//! The IME logic (composition, conversion state machine, learning, config)
//! is `setsuna-core`, shared with the other frontends; this crate turns TSF
//! key events into it and renders its actions. The conversion engine is NOT
//! linked in-process — the DLL spawns a `setsuna-engine daemon` child and
//! talks a stdio JSONL protocol (`setsuna_core::child_daemon`), so
//! activation never blocks on model load.
//!
//! Structure mirrors karukan-tsf (MIT, see THIRD_PARTY_NOTICES.md):
//! DLL exports, CLSID/profile GUIDs, a ClassFactory, and a
//! TextInputProcessor + KeyEventSink stack.

#[cfg(target_os = "windows")]
pub mod globals;
pub mod guard;
#[cfg(target_os = "windows")]
pub mod keymap;
pub mod placement;
pub mod registration;
pub mod ui;

#[cfg(target_os = "windows")]
pub mod tsf;

#[cfg(target_os = "windows")]
mod dll_exports {
    use windows::core::*;
    use windows::Win32::Foundation::{
        BOOL, CLASS_E_CLASSNOTAVAILABLE, E_FAIL, E_POINTER, HMODULE, S_FALSE, S_OK, TRUE,
    };

    use crate::globals::*;
    use crate::tsf::class_factory::SetsunaClassFactory;

    const DLL_PROCESS_ATTACH: u32 = 1;

    #[unsafe(no_mangle)]
    extern "system" fn DllMain(
        hinstance: HMODULE,
        reason: u32,
        _reserved: *mut core::ffi::c_void,
    ) -> BOOL {
        if reason == DLL_PROCESS_ATTACH {
            let _ = DLL_INSTANCE.set(SyncHmodule(hinstance));
            setsuna_core::diagnostics::init("tsf.log");
            crate::guard::install_hook();
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
                )
                .with_writer(std::io::stderr)
                .try_init();
        }
        TRUE
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllGetClassObject(
        rclsid: *const GUID,
        riid: *const GUID,
        ppv: *mut *mut core::ffi::c_void,
    ) -> HRESULT {
        setsuna_core::diagnostics::log("DllGetClassObject");
        unsafe {
            if ppv.is_null() || rclsid.is_null() || riid.is_null() {
                setsuna_core::diagnostics::log("DllGetClassObject E_POINTER");
                return E_POINTER;
            }
            *ppv = core::ptr::null_mut();
            if *rclsid != CLSID_SETSUNA_TEXT_SERVICE {
                setsuna_core::diagnostics::log(format!(
                    "DllGetClassObject unknown clsid={:?}",
                    *rclsid
                ));
                return CLASS_E_CLASSNOTAVAILABLE;
            }
            let factory: IUnknown = SetsunaClassFactory::new().into();
            let hr = factory.query(&*riid, ppv);
            setsuna_core::diagnostics::log(format!("DllGetClassObject hr={hr:?}"));
            hr
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllCanUnloadNow() -> HRESULT {
        if dll_can_unload() {
            S_OK
        } else {
            S_FALSE
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllRegisterServer() -> HRESULT {
        setsuna_core::diagnostics::log("DllRegisterServer");
        match crate::registration::register_server() {
            Ok(()) => S_OK,
            Err(e) => e.code(),
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllUnregisterServer() -> HRESULT {
        setsuna_core::diagnostics::log("DllUnregisterServer");
        match crate::registration::unregister_server() {
            Ok(()) => S_OK,
            Err(e) => e.code(),
        }
    }

    /// The version of the calls below, the shim's way in (`crates/tsf-shim`,
    /// registered in this DLL's place, forwards TSF's calls to a text
    /// service of this DLL and swaps it for a newer DLL's while the app
    /// runs). A shim of another version is refused and keeps the core it has.
    const CORE_ABI: u32 = 1;

    /// A new text service, `riid` of it in `ppv`, as the class factory makes
    /// one.
    #[unsafe(no_mangle)]
    extern "system" fn SetsunaCoreCreate(
        abi: u32,
        riid: *const GUID,
        ppv: *mut *mut core::ffi::c_void,
    ) -> HRESULT {
        setsuna_core::diagnostics::log(format!("SetsunaCoreCreate abi={abi}"));
        if ppv.is_null() || riid.is_null() {
            return E_POINTER;
        }
        unsafe { *ppv = core::ptr::null_mut() };
        if abi != CORE_ABI {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        match unsafe { crate::tsf::class_factory::create_service(riid, ppv) } {
            Ok(()) => S_OK,
            Err(e) => e.code(),
        }
    }

    /// Whether `tip`, an `ITfTextInputProcessorEx` `SetsunaCoreCreate` made,
    /// holds a composition (or is busy): the shim swaps it only when not.
    #[unsafe(no_mangle)]
    extern "system" fn SetsunaCoreComposing(tip: *mut core::ffi::c_void) -> BOOL {
        use windows::Win32::UI::TextServices::ITfTextInputProcessorEx;
        let Some(tip) = (unsafe { ITfTextInputProcessorEx::from_raw_borrowed(&tip) }) else {
            return TRUE;
        };
        let service: &crate::tsf::text_input_processor::SetsunaTextService =
            unsafe { tip.as_impl() };
        BOOL::from(
            service
                .inner
                .try_borrow()
                .map_or(true, |inner| inner.composition.is_some()),
        )
    }

    /// The text service `tip` (`SetsunaCoreCreate`'s, active) as the next
    /// one in a swap takes it (`SwapState`, JSON) in `buf`, when it fits in
    /// `cap` bytes: its length (0: `tip` is busy, nothing to hand over).
    #[unsafe(no_mangle)]
    extern "system" fn SetsunaCoreState(
        tip: *mut core::ffi::c_void,
        buf: *mut u8,
        cap: usize,
    ) -> usize {
        use crate::tsf::text_input_processor::{SetsunaTextService, SwapState};
        use windows::Win32::UI::TextServices::ITfTextInputProcessorEx;
        let Some(tip) = (unsafe { ITfTextInputProcessorEx::from_raw_borrowed(&tip) }) else {
            return 0;
        };
        let service: &SetsunaTextService = unsafe { tip.as_impl() };
        let Ok(inner) = service.inner.try_borrow() else {
            return 0;
        };
        let json = SwapState {
            enabled: inner.enabled,
            auto_off: inner.auto_off,
            focused: inner.focused,
            switched_in: inner.switched_in,
        }
        .to_json();
        if !buf.is_null() && json.len() <= cap {
            unsafe { core::ptr::copy_nonoverlapping(json.as_ptr(), buf, json.len()) };
        }
        json.len()
    }

    /// `tip`'s next activation is a swap: it takes `state` (`len` bytes,
    /// `SetsunaCoreState`'s of the text service it replaces; none: as it
    /// was) instead of starting as a new text service does (no `[[apps]]
    /// start`, the fields' start states as they were).
    #[unsafe(no_mangle)]
    extern "system" fn SetsunaCoreSwapIn(
        tip: *mut core::ffi::c_void,
        state: *const u8,
        len: usize,
    ) -> HRESULT {
        use crate::tsf::text_input_processor::{SetsunaTextService, SwapState};
        use windows::Win32::UI::TextServices::ITfTextInputProcessorEx;
        let Some(tip) = (unsafe { ITfTextInputProcessorEx::from_raw_borrowed(&tip) }) else {
            return E_POINTER;
        };
        let service: &SetsunaTextService = unsafe { tip.as_impl() };
        let json = if state.is_null() {
            &[][..]
        } else {
            unsafe { core::slice::from_raw_parts(state, len) }
        };
        let swap = SwapState::from_json(json);
        setsuna_core::diagnostics::log(format!(
            "SetsunaCoreSwapIn enabled={} auto_off={}",
            swap.enabled, swap.auto_off
        ));
        match service.inner.try_borrow_mut() {
            Ok(mut inner) => {
                inner.swap_in = Some(swap);
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}
