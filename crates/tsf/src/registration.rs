//! COM / TSF registration of this DLL as the IME (`regsvr32` of the DLL
//! itself; an install registers the shim in its place, `crates/tsf-shim`).
//! The entries are `setsuna_tsf_reg`'s, which the shim registers too.

#[cfg(target_os = "windows")]
use windows::Win32::System::LibraryLoader::GetModuleFileNameW;

#[cfg(target_os = "windows")]
pub fn register_server() -> windows::core::Result<()> {
    setsuna_tsf_reg::register(&setsuna_tsf_reg::IME, &dll_path())
}

#[cfg(target_os = "windows")]
pub fn unregister_server() -> windows::core::Result<()> {
    setsuna_tsf_reg::unregister(&setsuna_tsf_reg::IME)
}

#[cfg(target_os = "windows")]
fn dll_path() -> String {
    let hmodule = crate::globals::DLL_INSTANCE
        .get()
        .map(|s| s.0)
        .unwrap_or_default();
    let mut buf = [0u16; 1024];
    let len = unsafe { GetModuleFileNameW(hmodule, &mut buf) } as usize;
    String::from_utf16_lossy(&buf[..len])
}

#[cfg(not(target_os = "windows"))]
pub fn register_server() -> Result<(), String> {
    Err("TSF registration is Windows-only".into())
}

#[cfg(not(target_os = "windows"))]
pub fn unregister_server() -> Result<(), String> {
    Err("TSF unregistration is Windows-only".into())
}
