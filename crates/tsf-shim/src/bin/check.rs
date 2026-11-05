//! `setsuna-tsf-check <core file name> [--test]`: the registered shim
//! activated in this process, as an app would have it, with that core (in
//! the shim's directory) in place of the one core.txt names. Exit code 0
//! when the text service came up (the core loaded, its on / off keys
//! preserved, its mode on the taskbar), 1 when not, with what the shim and
//! the core logged. install.ps1 runs it before core.txt names a new core,
//! so that a core that does not come up never reaches the apps that are
//! running. `--test`: the shim registered as "Setsuna (shim test)".
//!
//! APPDATA and LOCALAPPDATA point at a directory of its own, removed after:
//! no config.toml (no engine is started, no history written), and the logs
//! land there.

#[cfg(windows)]
fn main() {
    std::process::exit(run::main())
}

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
mod run {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use setsuna_tsf_reg::{TextService, IME, LANGID_JAPANESE, SHIM_TEST};
    use windows::core::*;
    use windows::Win32::System::Com::*;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::System::Registry::*;
    use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
    use windows::Win32::UI::TextServices::*;
    use windows::Win32::UI::WindowsAndMessaging::*;

    pub fn main() -> i32 {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let test = args.iter().any(|a| a == "--test");
        let names: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
        let [name] = names[..] else {
            eprintln!("usage: setsuna-tsf-check <core file name> [--test]");
            return 2;
        };
        let service = if test { &SHIM_TEST } else { &IME };
        let Some(shim) = registered(service) else {
            println!("failed: no shim is registered for {}", service.name);
            return 1;
        };
        let Some(core) = shim.parent().map(|d| d.join(name)) else {
            return 1;
        };
        if !core.is_file() {
            println!("failed: {} not found", core.display());
            return 1;
        }

        let work = std::env::temp_dir().join(format!("setsuna-tsf-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let appdata = work.join("appdata");
        let _ = std::fs::create_dir_all(&appdata);
        let _ = std::fs::create_dir_all(work.join("localappdata"));
        std::env::set_var("APPDATA", &appdata);
        std::env::set_var("LOCALAPPDATA", work.join("localappdata"));
        std::env::set_var("SETSUNA_SHIM_CORE", name);

        let came_up = unsafe { activate(service, &shim, &core) };
        let code = match &came_up {
            Ok(()) => {
                println!("ok: {} came up through {}", core.display(), shim.display());
                0
            }
            Err(e) => {
                println!("failed: {} did not come up: {e}", core.display());
                for log in ["shim.log", "tsf.log"] {
                    let text = std::fs::read_to_string(appdata.join("setsuna-ime").join(log))
                        .unwrap_or_default();
                    println!("---- {log}");
                    let lines: Vec<&str> = text.lines().collect();
                    for line in &lines[lines.len().saturating_sub(40)..] {
                        println!("{line}");
                    }
                }
                1
            }
        };
        let _ = std::fs::remove_dir_all(&work);
        code
    }

    /// The DLL registered as `service`'s server.
    fn registered(service: &TextService) -> Option<PathBuf> {
        let key = HSTRING::from(format!("CLSID\\{{{:?}}}\\InProcServer32", service.clsid));
        let mut buf = [0u16; 1024];
        let mut len = (buf.len() * 2) as u32;
        unsafe {
            RegGetValueW(
                HKEY_CLASSES_ROOT,
                &key,
                PCWSTR::null(),
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr() as *mut _),
                Some(&mut len),
            )
            .ok()
            .ok()?;
        }
        let chars = (len as usize / 2).saturating_sub(1);
        Some(PathBuf::from(String::from_utf16_lossy(&buf[..chars])))
    }

    /// A thread manager of this process with a window and a document of its
    /// own (TSF deactivates text services when a thread's last window
    /// goes), `service`'s profile activated on it for this process, and
    /// what came up looked at.
    unsafe fn activate(
        service: &TextService,
        shim: &Path,
        core: &Path,
    ) -> std::result::Result<(), String> {
        let err = |what: &str, e: Error| format!("{what}: {}", e.message());
        CoInitializeEx(None, COINIT_APARTMENTTHREADED)
            .ok()
            .map_err(|e| err("CoInitializeEx", e))?;
        let tm: ITfThreadMgr = CoCreateInstance(&CLSID_TF_ThreadMgr, None, CLSCTX_INPROC_SERVER)
            .map_err(|e| err("thread manager", e))?;
        let tid = tm.Activate().map_err(|e| err("thread manager", e))?;
        let doc = tm.CreateDocumentMgr().map_err(|e| err("document", e))?;
        let mut context = None;
        let mut ec = 0;
        doc.CreateContext(tid, 0, None, &mut context, &mut ec)
            .map_err(|e| err("context", e))?;
        if let Some(context) = &context {
            doc.Push(context).map_err(|e| err("context", e))?;
        }
        let window = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("STATIC"),
            w!("setsuna-tsf-check"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            100,
            100,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| err("window", e))?;
        // no document was associated before: null, which windows-rs reads
        // as an error
        let _ = tm.AssociateFocus(window, &doc);
        tm.SetFocus(&doc).map_err(|e| err("focus", e))?;

        let profiles: ITfInputProcessorProfileMgr =
            CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)
                .map_err(|e| err("profiles", e))?;
        profiles
            .ActivateProfile(
                TF_PROFILETYPE_INPUTPROCESSOR,
                LANGID_JAPANESE,
                &service.clsid,
                &service.profile,
                HKL::default(),
                TF_IPPMF_FORPROCESS | TF_IPPMF_DONTCARECURRENTINPUTLANGUAGE,
            )
            .map_err(|e| err("ActivateProfile", e))?;
        pump(Duration::from_millis(1500));

        let loaded = |p: &Path| GetModuleHandleW(&HSTRING::from(p.as_os_str())).is_ok();
        let mut missing = Vec::new();
        if !loaded(shim) {
            missing.push("the shim not loaded");
        }
        if !loaded(core) {
            missing.push("the core not loaded");
        }
        let km: Option<ITfKeystrokeMgr> = tm.cast().ok();
        let on_key = TF_PRESERVEDKEY {
            uVKey: 0x16, // VK_IME_ON
            uModifiers: 0,
        };
        let preserved = km
            .and_then(|km| km.GetPreservedKey(context.as_ref(), &on_key).ok())
            .is_some_and(|g| g != GUID::zeroed());
        if !preserved {
            missing.push("no on key preserved");
        }
        if !mode_item(&tm) {
            missing.push("no mode on the taskbar");
        }

        let _ = tm.Deactivate();
        pump(Duration::from_millis(100));
        let _ = DestroyWindow(window);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing.join(", "))
        }
    }

    /// Whether the IME's mode item is on the thread's taskbar items (the
    /// core adds it under the IME's CLSID, under the shim of either).
    unsafe fn mode_item(tm: &ITfThreadMgr) -> bool {
        let Ok(items) = tm.cast::<ITfLangBarItemMgr>().and_then(|m| m.EnumItems()) else {
            return false;
        };
        loop {
            let mut item = [None];
            let mut n = 0;
            if items.Next(&mut item, &mut n).is_err() || n == 0 {
                return false;
            }
            let mut info = TF_LANGBARITEMINFO::default();
            if let Some(i) = &item[0] {
                if i.GetInfo(&mut info).is_ok() && info.clsidService == IME.clsid {
                    return true;
                }
            }
        }
    }

    fn pump(for_: Duration) {
        let end = Instant::now() + for_;
        unsafe {
            let mut msg = MSG::default();
            while Instant::now() < end {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
