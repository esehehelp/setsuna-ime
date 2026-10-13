//! Global state and GUID definitions for the Setsuna TSF IME.
//!
//! Structure and helpers adapted from karukan-tsf (MIT/Apache-2.0).
//! The CLSID / profile / display-attribute / preserved-key GUIDs are minted
//! for Setsuna. The `GUID_TFCAT_TIPCAP_*` values are OS-defined capability
//! categories and must stay verbatim.

#[cfg(target_os = "windows")]
use once_cell::sync::OnceCell;
#[cfg(target_os = "windows")]
use windows::core::GUID;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HMODULE;

/// Stable CLSID for the Setsuna TSF service.
/// `{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}`
#[cfg(target_os = "windows")]
pub const CLSID_SETSUNA_TEXT_SERVICE: GUID =
    GUID::from_u128(0xc3f4_dfa2_6e28_4aa0_9daa_407b_575e_f52d);

/// Stable language profile GUID for the Setsuna TSF service.
/// `{ACEB29C0-964E-4FF1-8BDC-B507AAB78873}`
#[cfg(target_os = "windows")]
pub const GUID_SETSUNA_PROFILE: GUID = GUID::from_u128(0xaceb_29c0_964e_4ff1_8bdc_b507_aab7_8873);

/// Preedit input display attribute (dotted underline).
/// `{849A1339-948F-4836-96EB-7E87FFBE2051}`
#[cfg(target_os = "windows")]
pub const GUID_DISPLAY_ATTRIBUTE_INPUT: GUID =
    GUID::from_u128(0x849a_1339_948f_4836_96eb_7e87_ffbe_2051);

/// Converted text display attribute (solid bold underline).
/// `{21CCC3C2-35DE-4F7F-984D-E1CA1B292D46}`
#[cfg(target_os = "windows")]
pub const GUID_DISPLAY_ATTRIBUTE_CONVERTED: GUID =
    GUID::from_u128(0x21cc_c3c2_35de_4f7f_984d_e1ca_1b29_2d46);

/// Immersive (modern/UWP) support — required so UWP hosts load us.
/// `{13A016DF-560B-46CD-947A-4C3AF1E0E35D}` (OS-defined).
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT: GUID =
    GUID::from_u128(0x13A0_16DF_560B_46CD_947A_4C3A_F1E0_E35D);

/// System tray support — required for proper system tray integration.
/// `{25504FB4-7BAB-4BC1-9C69-CF81890F0EF5}` (OS-defined).
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT: GUID =
    GUID::from_u128(0x2550_4FB4_7BAB_4BC1_9C69_CF81_890F_0EF5);

/// Secure-mode capability — required for key delivery in UWP apps.
/// `{49D2F9CE-1F5E-11D7-A6D3-00065B84435C}` (OS-defined).
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_SECUREMODE: GUID =
    GUID::from_u128(0x49D2_F9CE_1F5E_11D7_A6D3_0006_5B84_435C);

/// UI element support.
/// `{49D2F9CF-1F5E-11D7-A6D3-00065B84435C}` (OS-defined).
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_UIELEMENTENABLED: GUID =
    GUID::from_u128(0x49D2_F9CF_1F5E_11D7_A6D3_0006_5B84_435C);

/// PreservedKey GUID for the Hankaku/Zenkaku (IME on/off) toggle.
/// `{6B41F5FA-4378-4730-BC12-F0B5C9F885EB}`
#[cfg(target_os = "windows")]
pub const GUID_PRESERVED_KEY_ONOFF: GUID =
    GUID::from_u128(0x6b41_f5fa_4378_4730_bc12_f0b5_c9f8_85eb);

/// Japanese LANGID (0x0411).
pub const LANGID_JAPANESE: u16 = 0x0411;

/// Newtype so HMODULE can live in a OnceCell (HMODULE is `*mut void` and
/// therefore `!Send`/`!Sync` by default).
#[cfg(target_os = "windows")]
#[derive(Clone, Copy)]
pub struct SyncHmodule(pub HMODULE);

#[cfg(target_os = "windows")]
unsafe impl Send for SyncHmodule {}
#[cfg(target_os = "windows")]
unsafe impl Sync for SyncHmodule {}

#[cfg(target_os = "windows")]
pub static DLL_INSTANCE: OnceCell<SyncHmodule> = OnceCell::new();

#[cfg(target_os = "windows")]
pub static DLL_REF_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(target_os = "windows")]
pub fn dll_add_ref() {
    DLL_REF_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(target_os = "windows")]
pub fn dll_release() {
    let prev = DLL_REF_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    if prev == 0 {
        tracing::error!("dll_release underflow");
        return;
    }
    DLL_REF_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(target_os = "windows")]
pub fn dll_can_unload() -> bool {
    DLL_REF_COUNT.load(std::sync::atomic::Ordering::SeqCst) == 0
}

/// Message posted to the service's notify HWND when a conversion result is
/// ready. `wparam` carries the request id; `lparam` is unused.
#[cfg(target_os = "windows")]
pub const WM_APP_CONVERT_DONE: u32 = 0x8000 + 1;

/// Posted to the same window when a candidate in the candidate window is
/// clicked. `wparam` carries its index in the list.
#[cfg(target_os = "windows")]
pub const WM_APP_CANDIDATE_CLICK: u32 = 0x8000 + 2;

/// Posted to the same window when the taskbar's mode item is clicked: turn
/// the IME on or off.
#[cfg(target_os = "windows")]
pub const WM_APP_TOGGLE: u32 = 0x8000 + 3;

/// Directory the loaded TSF DLL lives in: where the engine and models are
/// looked for last.
#[cfg(target_os = "windows")]
pub fn dll_dir() -> Option<std::path::PathBuf> {
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
    let hmodule = DLL_INSTANCE.get().map(|s| s.0).unwrap_or_default();
    let mut buf = [0u16; 260];
    let len = unsafe { GetModuleFileNameW(hmodule, &mut buf) } as usize;
    if len == 0 {
        return None;
    }
    let full = String::from_utf16_lossy(&buf[..len]);
    std::path::PathBuf::from(full)
        .parent()
        .map(|p| p.to_path_buf())
}
