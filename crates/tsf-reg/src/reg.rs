use windows::core::{w, Result, GUID, HSTRING, PCWSTR};
use windows::Win32::Foundation::BOOL;
use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
use windows::Win32::System::Registry::*;
use windows::Win32::UI::Input::KeyboardAndMouse::HKL;
use windows::Win32::UI::TextServices::*;

/// A text service as TSF knows it: its COM class and its one profile.
pub struct TextService {
    pub clsid: GUID,
    pub profile: GUID,
    /// The profile's name, in the list of input methods.
    pub name: &'static str,
    /// The COM class's description.
    pub description: &'static str,
    /// Enabled by default and made the session's input method when
    /// registered (the IME), or only listed (a test's).
    pub activate: bool,
}

/// The IME. `{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}`, profile
/// `{ACEB29C0-964E-4FF1-8BDC-B507AAB78873}`.
pub const IME: TextService = TextService {
    clsid: GUID::from_u128(0xc3f4_dfa2_6e28_4aa0_9daa_407b_575e_f52d),
    profile: GUID::from_u128(0xaceb_29c0_964e_4ff1_8bdc_b507_aab7_8873),
    name: "Setsuna",
    description: "Setsuna Japanese Input Method",
    activate: true,
};

/// The shim beside the IME, for trying a shim without touching the IME's
/// registration (`regsvr32 /n /i:test`). `{E069C3FF-7026-4E4C-9153-89678BD4B47F}`,
/// profile `{6C6D8475-1FF4-4D9F-80BB-750F50519548}`.
pub const SHIM_TEST: TextService = TextService {
    clsid: GUID::from_u128(0xe069_c3ff_7026_4e4c_9153_8967_8bd4_b47f),
    profile: GUID::from_u128(0x6c6d_8475_1ff4_4d9f_80bb_750f_5051_9548),
    name: "Setsuna (shim test)",
    description: "Setsuna shim test",
    activate: false,
};

pub const LANGID_JAPANESE: u16 = 0x0411;

// OS-defined capability categories (not in windows-rs as constants)
pub const GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT: GUID =
    GUID::from_u128(0x13a0_16df_560b_46cd_947a_4c3a_f1e0_e35d);
pub const GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT: GUID =
    GUID::from_u128(0x2550_4fb4_7bab_4bc1_9c69_cf81_890f_0ef5);
pub const GUID_TFCAT_TIPCAP_SECUREMODE: GUID =
    GUID::from_u128(0x49d2_f9ce_1f5e_11d7_a6d3_0006_5b84_435c);
pub const GUID_TFCAT_TIPCAP_UIELEMENTENABLED: GUID =
    GUID::from_u128(0x49d2_f9cf_1f5e_11d7_a6d3_0006_5b84_435c);

/// The categories the text service is registered in.
pub const CATEGORIES: [GUID; 8] = [
    GUID_TFCAT_TIP_KEYBOARD,
    GUID_TFCAT_DISPLAYATTRIBUTEPROVIDER,
    GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT,
    GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT,
    GUID_TFCAT_TIPCAP_SECUREMODE,
    GUID_TFCAT_TIPCAP_UIELEMENTENABLED,
    GUID_TFCAT_TIPCAP_COMLESS,
    GUID_TFCAT_TIPCAP_INPUTMODECOMPARTMENT,
];

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

fn clsid_key(service: &TextService) -> String {
    format!("CLSID\\{{{:?}}}", service.clsid)
}

/// `service`'s COM class with `dll_path` its server, its profile and its
/// categories (the HKCR / HKLM entries: an administrator's).
pub fn register(service: &TextService, dll_path: &str) -> Result<()> {
    let key_path = clsid_key(service);
    unsafe {
        let mut key = HKEY::default();
        RegCreateKeyW(HKEY_CLASSES_ROOT, &HSTRING::from(&key_path), &mut key).ok()?;
        let set = set_string(key, PCWSTR::null(), service.description);
        let _ = RegCloseKey(key);
        set?;

        let mut inproc = HKEY::default();
        RegCreateKeyW(
            HKEY_CLASSES_ROOT,
            &HSTRING::from(format!("{key_path}\\InProcServer32")),
            &mut inproc,
        )
        .ok()?;
        let set = set_string(inproc, PCWSTR::null(), dll_path)
            .and_then(|()| set_string(inproc, w!("ThreadingModel"), "Apartment"));
        let _ = RegCloseKey(inproc);
        set?;

        let profile_mgr: ITfInputProcessorProfileMgr =
            CoCreateInstance(&CLSID_TF_InputProcessorProfiles, None, CLSCTX_INPROC_SERVER)?;
        let name: Vec<u16> = service.name.encode_utf16().collect();
        profile_mgr.RegisterProfile(
            &service.clsid,
            LANGID_JAPANESE,
            &service.profile,
            &name,
            &[],
            0,
            HKL::default(),
            0,
            BOOL(1),
            0,
        )?;

        if service.activate {
            if let Ok(profiles) = CoCreateInstance::<_, ITfInputProcessorProfiles>(
                &CLSID_TF_InputProcessorProfiles,
                None,
                CLSCTX_INPROC_SERVER,
            ) {
                let _ = profiles.EnableLanguageProfile(
                    &service.clsid,
                    LANGID_JAPANESE,
                    &service.profile,
                    BOOL(1),
                );
                let _ = profiles.EnableLanguageProfileByDefault(
                    &service.clsid,
                    LANGID_JAPANESE,
                    &service.profile,
                    BOOL(1),
                );
            }
            let _ = profile_mgr.ActivateProfile(
                TF_PROFILETYPE_INPUTPROCESSOR,
                LANGID_JAPANESE,
                &service.clsid,
                &service.profile,
                HKL::default(),
                TF_IPPMF_FORSESSION
                    | TF_IPPMF_ENABLEPROFILE
                    | TF_IPPMF_DONTCARECURRENTINPUTLANGUAGE,
            );
        }

        let categories: ITfCategoryMgr =
            CoCreateInstance(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER)?;
        for cat in CATEGORIES {
            categories.RegisterCategory(&service.clsid, &cat, &service.clsid)?;
        }
    }
    grant_appcontainers(dll_path);
    Ok(())
}

/// `service`'s profile, categories and COM class gone.
pub fn unregister(service: &TextService) -> Result<()> {
    let key_path = clsid_key(service);
    unsafe {
        if let Ok(profile_mgr) = CoCreateInstance::<_, ITfInputProcessorProfileMgr>(
            &CLSID_TF_InputProcessorProfiles,
            None,
            CLSCTX_INPROC_SERVER,
        ) {
            let _ =
                profile_mgr.UnregisterProfile(&service.clsid, LANGID_JAPANESE, &service.profile, 0);
        }
        if let Ok(categories) =
            CoCreateInstance::<_, ITfCategoryMgr>(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER)
        {
            for cat in CATEGORIES {
                let _ = categories.UnregisterCategory(&service.clsid, &cat, &service.clsid);
            }
        }
        let _ = RegDeleteKeyW(
            HKEY_CLASSES_ROOT,
            &HSTRING::from(format!("{key_path}\\InProcServer32")),
        );
        let _ = RegDeleteKeyW(HKEY_CLASSES_ROOT, &HSTRING::from(&key_path));
    }
    Ok(())
}

/// Let AppContainers (ALL APPLICATION PACKAGES, S-1-15-2-1) and less
/// privileged ones (ALL RESTRICTED APPLICATION PACKAGES, S-1-15-2-2) load
/// the DLL. Best effort: install.ps1 grants the install directory as well.
fn grant_appcontainers(dll_path: &str) {
    let _ = std::process::Command::new("icacls")
        .arg(dll_path)
        .arg("/grant")
        .arg("*S-1-15-2-1:(RX)")
        .arg("*S-1-15-2-2:(RX)")
        .output();
}
