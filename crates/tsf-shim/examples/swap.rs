//! Spike harness: the shim, registered (regsvr32 of `<work dir>\shim.dll`,
//! elevated), activated by TSF in this process only (`ActivateProfile`,
//! for the process) on a thread manager of its own with core A, then
//! `core.txt` pointed at core B and the focus moved, and back: after each,
//! the cores loaded, the taskbar items, the preserved on / off keys and
//! whether a letter key goes to the IME. APPDATA and LOCALAPPDATA are
//! redirected into the work directory (no config.toml: no engine, kana
//! only; the shim's and the core's logs land there).
//!
//!   cargo run -p setsuna-tsf-shim --example swap -- <core.dll> <work dir>
//!
//! Core A and core B are copies of `<core.dll>` under two names, which the
//! loader keeps apart as two modules.

#[cfg(windows)]
fn main() -> windows::core::Result<()> {
    run::main()
}

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
mod run {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use windows::core::*;
    use windows::Win32::Foundation::*;
    use windows::Win32::System::Com::*;
    use windows::Win32::System::LibraryLoader::*;
    use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
    use windows::Win32::UI::TextServices::*;
    use windows::Win32::UI::WindowsAndMessaging::*;

    const CLSID_SHIM: GUID = GUID::from_u128(0xe069_c3ff_7026_4e4c_9153_8967_8bd4_b47f);
    const GUID_SHIM_PROFILE: GUID = GUID::from_u128(0x6c6d_8475_1ff4_4d9f_80bb_750f_5051_9548);
    const LANGID_JAPANESE: u16 = 0x0411;

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [core_src, work] = &args[..] else {
            panic!("usage: swap <core.dll> <work dir>");
        };
        let work = PathBuf::from(work);
        let appdata = work.join("appdata");
        let _ = std::fs::remove_dir_all(&appdata);
        std::fs::create_dir_all(&appdata).unwrap();
        std::fs::create_dir_all(work.join("localappdata")).unwrap();
        std::env::set_var("APPDATA", &appdata);
        std::env::set_var("LOCALAPPDATA", work.join("localappdata"));
        for name in ["core_a.dll", "core_b.dll"] {
            // a core an earlier run left loaded nowhere is replaced
            let _ = std::fs::copy(core_src, work.join(name));
        }
        std::fs::write(work.join("core.txt"), "core_a.dll").unwrap();

        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
            let tm: ITfThreadMgr =
                CoCreateInstance(&CLSID_TF_ThreadMgr, None, CLSCTX_INPROC_SERVER)?;
            // the user's IME comes up first; the shim's profile replaces it
            // in this process below
            let app_tid = tm.Activate()?;
            let doc1 = document(&tm, app_tid)?;
            let doc2 = document(&tm, app_tid)?;
            // a window of the app's own, its focus the first document, as
            // in an app (TSF deactivates text services when a thread's last
            // window goes)
            let window = CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!("shim swap harness"),
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                200,
                100,
                None,
                None,
                None,
                None,
            )?;

            // no document was associated before: null, which windows-rs reads as an error
            let _ = tm.AssociateFocus(window, &doc1);
            tm.SetFocus(&doc1)?;

            let profiles: ITfInputProcessorProfileMgr =
                CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
            profiles
                .ActivateProfile(
                    TF_PROFILETYPE_INPUTPROCESSOR,
                    LANGID_JAPANESE,
                    &CLSID_SHIM,
                    &GUID_SHIM_PROFILE,
                    HKL::default(),
                    TF_IPPMF_FORPROCESS | TF_IPPMF_DONTCARECURRENTINPUTLANGUAGE,
                )
                .inspect_err(|e| println!("ActivateProfile: {e:?} (is the shim registered?)"))?;
            pump(Duration::from_millis(300));
            report("the shim's profile activated (core A)", &tm, &work);

            // a newer core is named; the next focus change swaps it in
            std::thread::sleep(Duration::from_millis(50));
            std::fs::write(work.join("core.txt"), "core_b.dll").unwrap();
            tm.SetFocus(&doc2)?;
            pump(Duration::from_millis(2600));
            report("after a focus change (core B named)", &tm, &work);

            // and back, to see a swap repeat
            std::thread::sleep(Duration::from_millis(50));
            std::fs::write(work.join("core.txt"), "core_a.dll").unwrap();
            tm.SetFocus(&doc1)?;
            pump(Duration::from_millis(2600));
            report("after another (core A named again)", &tm, &work);

            tm.Deactivate()?;
            pump(Duration::from_millis(100));
        }
        for log in ["shim.log", "tsf.log"] {
            let path = appdata.join("setsuna-ime").join(log);
            println!("---- {log}");
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            for line in text.lines().filter(|l| {
                log == "shim.log"
                    || [
                        "SetsunaCoreCreate",
                        "ActivateEx",
                        "Deactivate",
                        "drop",
                        "panic",
                        "preserved",
                        "mode on",
                        "OnTestKeyDown",
                        "OnSetFocus",
                    ]
                    .iter()
                    .any(|k| l.contains(k))
            }) {
                println!("{line}");
            }
        }
        Ok(())
    }

    /// A document manager with an empty context (no text store) on it.
    unsafe fn document(tm: &ITfThreadMgr, app_tid: u32) -> Result<ITfDocumentMgr> {
        let dim = tm.CreateDocumentMgr()?;
        let mut context = None;
        let mut ec = 0;
        dim.CreateContext(app_tid, 0, None, &mut context, &mut ec)?;
        dim.Push(context.as_ref().unwrap())?;
        Ok(dim)
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

    fn report(when: &str, tm: &ITfThreadMgr, work: &Path) {
        println!("== {when}");
        let loaded: Vec<&str> = ["shim.dll", "core_a.dll", "core_b.dll"]
            .into_iter()
            .filter(|n| unsafe {
                GetModuleHandleW(&HSTRING::from(work.join(n).as_os_str())).is_ok()
            })
            .collect();
        println!("  loaded: {loaded:?}");
        unsafe {
            let mut items = Vec::new();
            if let Ok(e) = tm.cast::<ITfLangBarItemMgr>().and_then(|m| m.EnumItems()) {
                loop {
                    let mut item = [None];
                    let mut n = 0;
                    if e.Next(&mut item, &mut n).is_err() || n == 0 {
                        break;
                    }
                    let mut info = TF_LANGBARITEMINFO::default();
                    if let Some(i) = &item[0] {
                        if i.GetInfo(&mut info).is_ok() {
                            let d = &info.szDescription;
                            let len = d.iter().position(|&c| c == 0).unwrap_or(d.len());
                            items.push(String::from_utf16_lossy(&d[..len]));
                        }
                    }
                }
            }
            println!("  taskbar items: {items:?}");
            let km: ITfKeystrokeMgr = tm.cast().unwrap();
            let keys: Vec<String> = [
                (0x16u32, "ImeOn"),
                (0x1a, "ImeOff"),
                (0x19, "Kanji"),
                (0xf3, "OemAuto"),
                (0xf4, "OemEnlw"),
            ]
            .into_iter()
            .filter(|(vk, _)| {
                let key = TF_PRESERVEDKEY {
                    uVKey: *vk,
                    uModifiers: 0,
                };
                km.GetPreservedKey(None, &key)
                    .is_ok_and(|g| g != GUID::zeroed())
            })
            .map(|(_, n)| n.to_string())
            .collect();
            println!("  preserved: {keys:?}");
            let eaten = km.TestKeyDown(WPARAM(0x4b), LPARAM(0x0025_0001));
            println!("  'K' to the IME: {eaten:?}");
        }
    }
}
