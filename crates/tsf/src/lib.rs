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
#[cfg(target_os = "windows")]
pub mod keymap;
pub mod registration;
pub mod ui;

#[cfg(target_os = "windows")]
pub mod tsf;

#[cfg(target_os = "windows")]
mod dll_exports {
    use windows::core::*;
    use windows::Win32::Foundation::{
        BOOL, CLASS_E_CLASSNOTAVAILABLE, E_POINTER, HMODULE, S_FALSE, S_OK, TRUE,
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
            std::panic::set_hook(Box::new(|info| {
                eprintln!("setsuna-tsf panic: {}", info);
            }));
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
}
