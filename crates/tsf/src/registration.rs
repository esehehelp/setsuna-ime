//! COM / TSF registration (adapted from karukan-tsf MIT/Apache-2.0).
//!
//! Sets up the registry entries TSF needs to discover the IME and grants the
//! AppContainer ACL on the DLL so UWP hosts can load it. The category list
//! matches karukan's (IMMERSIVESUPPORT / SYSTRAYSUPPORT / SECUREMODE /
//! UIELEMENTENABLED) — omitting any of these caused key events to never
//! reach us in modern apps.

#[cfg(target_os = "windows")]
use windows::core::w;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::BOOL;
#[cfg(target_os = "windows")]
use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
#[cfg(target_os = "windows")]
use windows::Win32::System::Registry::*;

#[cfg(target_os = "windows")]
use crate::globals::*;

#[cfg(target_os = "windows")]
pub fn register_server() -> windows::core::Result<()> {
    use windows::Win32::UI::TextServices::*;

    let dll_path = get_dll_path()?;
    let clsid_str = format!("{{{:?}}}", CLSID_SETSUNA_TEXT_SERVICE);

    let key_path = format!("CLSID\\{}", clsid_str);
    unsafe {
        let mut hkey = HKEY::default();
        RegCreateKeyW(
            HKEY_CLASSES_ROOT,
            &windows::core::HSTRING::from(&key_path),
            &mut hkey,
        )
        .ok()?;
        let desc = w!("Setsuna Japanese Input Method");
        RegSetValueExW(
            hkey,
            None,
            0,
            REG_SZ,
            Some(std::slice::from_raw_parts(
                desc.as_ptr() as *const u8,
                (desc.len() + 1) * 2,
            )),
        )
        .ok()?;
        RegCloseKey(hkey).ok()?;

        let inproc_path = format!("{}\\InProcServer32", key_path);
        let mut hkey_inproc = HKEY::default();
        RegCreateKeyW(
            HKEY_CLASSES_ROOT,
            &windows::core::HSTRING::from(&inproc_path),
            &mut hkey_inproc,
        )
        .ok()?;

        let dll_path_wide: Vec<u16> = dll_path.encode_utf16().chain(std::iter::once(0)).collect();
        RegSetValueExW(
            hkey_inproc,
            None,
            0,
            REG_SZ,
            Some(std::slice::from_raw_parts(
                dll_path_wide.as_ptr() as *const u8,
                dll_path_wide.len() * 2,
            )),
        )
        .ok()?;

        let threading = w!("Apartment");
        RegSetValueExW(
            hkey_inproc,
            w!("ThreadingModel"),
            0,
            REG_SZ,
            Some(std::slice::from_raw_parts(
                threading.as_ptr() as *const u8,
                (threading.len() + 1) * 2,
            )),
        )
        .ok()?;
        RegCloseKey(hkey_inproc).ok()?;
    }

    unsafe {
        use windows::Win32::UI::Input::KeyboardAndMouse::HKL;

        let profile_mgr: ITfInputProcessorProfileMgr =
            windows::Win32::System::Com::CoCreateInstance(
                &CLSID_TF_InputProcessorProfiles,
                None,
                windows::Win32::System::Com::CLSCTX_INPROC_SERVER,
            )?;

        let desc_wide: Vec<u16> = "Setsuna".encode_utf16().collect();
        profile_mgr.RegisterProfile(
            &CLSID_SETSUNA_TEXT_SERVICE,
            LANGID_JAPANESE,
            &GUID_SETSUNA_PROFILE,
            &desc_wide,
            &[],
            0,
            HKL::default(),
            0,
            BOOL(1),
            0,
        )?;

        if let Ok(profiles) =
            windows::Win32::System::Com::CoCreateInstance::<_, ITfInputProcessorProfiles>(
                &CLSID_TF_InputProcessorProfiles,
                None,
                windows::Win32::System::Com::CLSCTX_INPROC_SERVER,
            )
        {
            let _ = profiles.EnableLanguageProfile(
                &CLSID_SETSUNA_TEXT_SERVICE,
                LANGID_JAPANESE,
                &GUID_SETSUNA_PROFILE,
                BOOL(1),
            );
            let _ = profiles.EnableLanguageProfileByDefault(
                &CLSID_SETSUNA_TEXT_SERVICE,
                LANGID_JAPANESE,
                &GUID_SETSUNA_PROFILE,
                BOOL(1),
            );
        }

        let _ = profile_mgr.ActivateProfile(
            TF_PROFILETYPE_INPUTPROCESSOR,
            LANGID_JAPANESE,
            &CLSID_SETSUNA_TEXT_SERVICE,
            &GUID_SETSUNA_PROFILE,
            HKL::default(),
            TF_IPPMF_FORSESSION | TF_IPPMF_ENABLEPROFILE | TF_IPPMF_DONTCARECURRENTINPUTLANGUAGE,
        );

        let cat_mgr: ITfCategoryMgr = windows::Win32::System::Com::CoCreateInstance(
            &CLSID_TF_CategoryMgr,
            None,
            windows::Win32::System::Com::CLSCTX_INPROC_SERVER,
        )?;

        for cat in [
            GUID_TFCAT_TIP_KEYBOARD,
            GUID_TFCAT_DISPLAYATTRIBUTEPROVIDER,
            GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT,
            GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT,
            GUID_TFCAT_TIPCAP_SECUREMODE,
            GUID_TFCAT_TIPCAP_UIELEMENTENABLED,
        ] {
            cat_mgr.RegisterCategory(
                &CLSID_SETSUNA_TEXT_SERVICE,
                &cat,
                &CLSID_SETSUNA_TEXT_SERVICE,
            )?;
        }
    }

    set_appcontainer_acl(&dll_path);
    Ok(())
}

#[cfg(target_os = "windows")]
pub fn unregister_server() -> windows::core::Result<()> {
    use windows::Win32::UI::TextServices::*;

    let clsid_str = format!("{{{:?}}}", CLSID_SETSUNA_TEXT_SERVICE);

    unsafe {
        if let Ok(profile_mgr) =
            windows::Win32::System::Com::CoCreateInstance::<_, ITfInputProcessorProfileMgr>(
                &CLSID_TF_InputProcessorProfiles,
                None,
                windows::Win32::System::Com::CLSCTX_INPROC_SERVER,
            )
        {
            let _ = profile_mgr.UnregisterProfile(
                &CLSID_SETSUNA_TEXT_SERVICE,
                LANGID_JAPANESE,
                &GUID_SETSUNA_PROFILE,
                0,
            );
        }

        if let Ok(cat_mgr) = windows::Win32::System::Com::CoCreateInstance::<_, ITfCategoryMgr>(
            &CLSID_TF_CategoryMgr,
            None,
            windows::Win32::System::Com::CLSCTX_INPROC_SERVER,
        ) {
            for cat in [
                GUID_TFCAT_TIP_KEYBOARD,
                GUID_TFCAT_DISPLAYATTRIBUTEPROVIDER,
                GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT,
                GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT,
                GUID_TFCAT_TIPCAP_SECUREMODE,
                GUID_TFCAT_TIPCAP_UIELEMENTENABLED,
            ] {
                let _ = cat_mgr.UnregisterCategory(
                    &CLSID_SETSUNA_TEXT_SERVICE,
                    &cat,
                    &CLSID_SETSUNA_TEXT_SERVICE,
                );
            }
        }
    }

    unsafe {
        let inproc_path = format!("CLSID\\{}\\InProcServer32", clsid_str);
        let _ = RegDeleteKeyW(
            HKEY_CLASSES_ROOT,
            &windows::core::HSTRING::from(&inproc_path),
        );
        let key_path = format!("CLSID\\{}", clsid_str);
        let _ = RegDeleteKeyW(HKEY_CLASSES_ROOT, &windows::core::HSTRING::from(&key_path));
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn get_dll_path() -> windows::core::Result<String> {
    let hmodule = DLL_INSTANCE.get().map(|s| s.0).unwrap_or_default();
    let mut buf = [0u16; 260];
    let len = unsafe { GetModuleFileNameW(hmodule, &mut buf) } as usize;
    Ok(String::from_utf16_lossy(&buf[..len]))
}

#[cfg(target_os = "windows")]
fn set_appcontainer_acl(dll_path: &str) {
    match std::process::Command::new("icacls")
        .arg(dll_path)
        .arg("/grant")
        .arg("*S-1-15-2-1:(RX)")
        .output()
    {
        Ok(output) if output.status.success() => {
            tracing::debug!("AppContainer ACL set on {}", dll_path);
        }
        Ok(output) => {
            tracing::warn!("icacls failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        Err(e) => {
            tracing::warn!("icacls spawn failed: {}", e);
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn register_server() -> Result<(), String> {
    Err("TSF registration is Windows-only".into())
}

#[cfg(not(target_os = "windows"))]
pub fn unregister_server() -> Result<(), String> {
    Err("TSF unregistration is Windows-only".into())
}
