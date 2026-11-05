//! Global state and GUID definitions for the Setsuna TSF IME.
//!
//! Structure and helpers adapted from karukan-tsf (MIT/Apache-2.0).
//! The CLSID / profile / display-attribute / preserved-key GUIDs are minted
//! for Setsuna; the categories it is registered in are `setsuna_tsf_reg`'s.

#[cfg(target_os = "windows")]
use once_cell::sync::OnceCell;
#[cfg(target_os = "windows")]
use windows::core::GUID;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HMODULE;

/// The CLSID and profile of the Setsuna TSF service (shared with the shim
/// registered in this DLL's place: `setsuna_tsf_reg::IME`).
#[cfg(target_os = "windows")]
pub const CLSID_SETSUNA_TEXT_SERVICE: GUID = setsuna_tsf_reg::IME.clsid;
#[cfg(target_os = "windows")]
pub const GUID_SETSUNA_PROFILE: GUID = setsuna_tsf_reg::IME.profile;

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

/// PreservedKey GUID of the keys that turn the IME on or off
/// (`[frontend] toggle_keys`: Hankaku/Zenkaku by default).
/// `{6B41F5FA-4378-4730-BC12-F0B5C9F885EB}`
#[cfg(target_os = "windows")]
pub const GUID_PRESERVED_KEY_ONOFF: GUID =
    GUID::from_u128(0x6b41_f5fa_4378_4730_bc12_f0b5_c9f8_85eb);

/// PreservedKey GUID of the keys that turn it on (`on_keys`).
/// `{FF3722C7-22B6-403D-B433-BAA71BEDD731}`
#[cfg(target_os = "windows")]
pub const GUID_PRESERVED_KEY_ON: GUID = GUID::from_u128(0xff37_22c7_22b6_403d_b433_baa7_1bed_d731);

/// PreservedKey GUID of the keys that turn it off (`off_keys`).
/// `{AED35535-08E0-4408-A5A6-6AD68539A015}`
#[cfg(target_os = "windows")]
pub const GUID_PRESERVED_KEY_OFF: GUID = GUID::from_u128(0xaed3_5535_08e0_4408_a5a6_6ad6_8539_a015);

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

/// Posted to the same window when the input scopes of the field that took
/// the focus were read: `wparam` bit 0 is set for a field that starts with
/// the IME off (a URL, a number, a password), bit 1 for one the user turned
/// the IME on or off in before.
#[cfg(target_os = "windows")]
pub const WM_APP_FIELD: u32 = 0x8000 + 4;

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
