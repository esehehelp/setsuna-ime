//! Harness: the shim registered as "Setsuna (shim test)" (`regsvr32 /n
//! /i:test <work dir>\shim.dll`, elevated), activated by TSF in this
//! process only on a thread manager of its own with core A; then core.txt
//! pointed at core B with nothing else done (the shim watches its
//! directory), and back. Core A and core B are `<core.dll>` under their
//! content-addressed names, B with a byte added at its end (which the
//! loader ignores) so that the two are two files and two modules.
//!
//! APPDATA and LOCALAPPDATA are redirected into the work directory, where
//! config.toml has the IME start on in this process (`[[apps]] start =
//! "on"`) and nothing else (no engine). The harness turns it off before the
//! swaps: a new core that takes the old one's state keeps it off, one that
//! started as a new text service would turn it on again. After each step:
//! the cores loaded, the taskbar items, the preserved on / off keys and the
//! on / off compartment.
//!
//!   cargo run -p setsuna-tsf-shim --example swap -- <core.dll> <work dir>

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

    use setsuna_tsf_reg::{LANGID_JAPANESE, SHIM_TEST};
    use sha2::{Digest, Sha256};
    use windows::core::*;
    use windows::Win32::System::Com::*;
    use windows::Win32::System::LibraryLoader::*;
    use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
    use windows::Win32::UI::TextServices::*;
    use windows::Win32::UI::WindowsAndMessaging::*;

    /// `bytes` in `dir` under their content-addressed name; the name.
    fn place(dir: &Path, bytes: &[u8]) -> String {
        let hash: String = Sha256::digest(bytes)
            .iter()
            .take(6)
            .map(|b| format!("{b:02x}"))
            .collect();
        let name = format!("setsuna_tsf_{hash}.dll");
        // one of an earlier run, loaded nowhere now, is the same file
        let _ = std::fs::write(dir.join(&name), bytes);
        name
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [core_src, work] = &args[..] else {
            panic!("usage: swap <core.dll> <work dir>");
        };
        let work = PathBuf::from(work);
        let appdata = work.join("appdata");
        let _ = std::fs::remove_dir_all(&appdata);
        std::fs::create_dir_all(appdata.join("setsuna-ime")).unwrap();
        std::fs::create_dir_all(work.join("localappdata")).unwrap();
        std::env::set_var("APPDATA", &appdata);
        std::env::set_var("LOCALAPPDATA", work.join("localappdata"));
        std::fs::write(
            appdata.join("setsuna-ime").join("config.toml"),
            "[[apps]]\nprocess = \"swap\"\nstart = \"on\"\n",
        )
        .unwrap();
        let bytes = std::fs::read(core_src).unwrap();
        let a = place(&work, &bytes);
        let b = place(&work, &[&bytes[..], &[0]].concat());
        std::fs::write(work.join("core.txt"), &a).unwrap();
        let names = [
            ("shim", "shim.dll".to_string()),
            ("A", a.clone()),
            ("B", b.clone()),
        ];

        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
            let tm: ITfThreadMgr =
                CoCreateInstance(&CLSID_TF_ThreadMgr, None, CLSCTX_INPROC_SERVER)?;
            let app_tid = tm.Activate()?;
            let doc = tm.CreateDocumentMgr()?;
            let mut context = None;
            let mut ec = 0;
            doc.CreateContext(app_tid, 0, None, &mut context, &mut ec)?;
            doc.Push(context.as_ref().unwrap())?;
            // a window of the app's own, as in an app (TSF deactivates text
            // services when a thread's last window goes)
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
            // no document was associated before: null, which windows-rs
            // reads as an error
            let _ = tm.AssociateFocus(window, &doc);
            tm.SetFocus(&doc)?;

            let profiles: ITfInputProcessorProfileMgr =
                CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
            profiles
                .ActivateProfile(
                    TF_PROFILETYPE_INPUTPROCESSOR,
                    LANGID_JAPANESE,
                    &SHIM_TEST.clsid,
                    &SHIM_TEST.profile,
                    HKL::default(),
                    TF_IPPMF_FORPROCESS | TF_IPPMF_DONTCARECURRENTINPUTLANGUAGE,
                )
                .inspect_err(|e| {
                    println!("ActivateProfile: {e:?} (is the test shim registered?)")
                })?;
            pump(Duration::from_millis(500));
            report(
                "activated with core A (starts on)",
                &tm,
                context.as_ref(),
                &work,
                &names,
            );

            let open_close = tm
                .cast::<ITfCompartmentMgr>()?
                .GetCompartment(&GUID_COMPARTMENT_KEYBOARD_OPENCLOSE)?;
            open_close.SetValue(app_tid, &VARIANT::from(0i32))?;
            pump(Duration::from_millis(200));
            report("turned off", &tm, context.as_ref(), &work, &names);

            // a newer core named, nothing else: the watch swaps it in
            std::fs::write(work.join("core.txt"), &b).unwrap();
            pump(Duration::from_millis(1500));
            report("core B named", &tm, context.as_ref(), &work, &names);

            std::fs::write(work.join("core.txt"), &a).unwrap();
            pump(Duration::from_millis(1500));
            report("core A named again", &tm, context.as_ref(), &work, &names);

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
                        "SetsunaCore",
                        "ActivateEx",
                        "Deactivate",
                        "drop",
                        "panic",
                        "preserved",
                        "mode on",
                    ]
                    .iter()
                    .any(|k| l.contains(k))
            }) {
                println!("{line}");
            }
        }
        Ok(())
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

    fn report(
        when: &str,
        tm: &ITfThreadMgr,
        context: Option<&ITfContext>,
        work: &Path,
        names: &[(&str, String)],
    ) {
        println!("== {when}");
        let loaded: Vec<&str> = names
            .iter()
            .filter(|(_, n)| unsafe {
                GetModuleHandleW(&HSTRING::from(work.join(n).as_os_str())).is_ok()
            })
            .map(|(label, _)| *label)
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
            let keys: Vec<&str> = [(0x16u32, "ImeOn"), (0x1a, "ImeOff"), (0x19, "Kanji")]
                .into_iter()
                .filter(|(vk, _)| {
                    let key = TF_PRESERVEDKEY {
                        uVKey: *vk,
                        uModifiers: 0,
                    };
                    km.GetPreservedKey(context, &key)
                        .is_ok_and(|g| g != GUID::zeroed())
                })
                .map(|(_, n)| n)
                .collect();
            println!("  preserved: {keys:?}");
            let open = tm
                .cast::<ITfCompartmentMgr>()
                .and_then(|m| m.GetCompartment(&GUID_COMPARTMENT_KEYBOARD_OPENCLOSE))
                .and_then(|c| c.GetValue())
                .ok()
                .and_then(|v| i32::try_from(&v).ok());
            println!("  on / off compartment: {open:?}");
        }
    }
}
