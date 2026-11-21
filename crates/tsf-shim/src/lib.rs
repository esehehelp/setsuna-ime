//! The text service TSF loads in the IME DLL's place, so an update reaches
//! the apps that are running.
//!
//! An app keeps the DLL TSF loaded into it until it exits: an update
//! installs a DLL of another name, and only apps started after it load that
//! one. The shim is that DLL instead, registered under the IME's CLSID and
//! profile, and rarely changes. It loads the IME DLL (the core) named in
//! `core.txt` beside it, makes a text service of it (`SetsunaCoreCreate`)
//! and forwards TSF's calls to it. A core is named by the hash of its
//! content (`setsuna_tsf_<12 hex digits of its SHA-256>.dll`, in the shim's
//! directory), which the shim checks before loading it.
//!
//! When `core.txt` names another core (the shim watches its directory, and
//! looks on a focus change), the running text service is swapped for one
//! of the new core: the old one hands over its state (`SetsunaCoreState`:
//! the mode as the user left it) and is deactivated, the new one takes the
//! state (`SetsunaCoreSwapIn`) and is activated on the same thread manager,
//! client id and flags, with no composition open (the swap waits for it:
//! `SetsunaCoreComposing`). TSF keeps talking to the shim and sees nothing
//! change. An old core stays loaded, unused (its threads end with its text
//! service; freeing a DLL whose code a thread may still run is not worth a
//! few MB).
//!
//! The core's calls besides `SetsunaCoreCreate` are found by their names,
//! each optional, and what TSF asks of a text service is handed to the core
//! as far as the shim knows it (`ITfFunctionProvider` included), so that a
//! later core can do more without a later shim, which only apps started
//! after it would load.
//!
//! `regsvr32 /n /i:test` registers it beside the IME instead, as "Setsuna
//! (shim test)", for trying a shim; `setsuna-tsf-check` (src/bin/check.rs)
//! tries a core through the registered shim before an install names it.
//!
//! An install keeps the registered shim while it is of the version the
//! install brings ([`SetsunaShimVersion`]): the bytes of a build differ
//! with what it is built with (cargo unifies the features of the packages
//! built together), and a new shim reaches only the apps started after it.

#[cfg(windows)]
mod shim;

/// The shim's version, where install.ps1 reads it from the DLL without
/// loading it (the digits after `setsuna-shim-version:`). Raise it with any
/// change to what the shim does: `src/shim.rs`, and `crates/tsf-reg`, which
/// its registration runs (`the_version_follows_the_source`). Exported, so
/// that the linker keeps it.
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static SetsunaShimVersion: [u8; 23] = *b"setsuna-shim-version:4\0";

#[cfg(test)]
mod tests {
    /// The version and a digest of the sources it stands for: a change to
    /// them fails this until the version is raised (or, for one that changes
    /// nothing the shim does, the digest alone is recorded again).
    #[test]
    fn the_version_follows_the_source() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut reg: Vec<_> = std::fs::read_dir(root.join("../tsf-reg/src"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        reg.sort();
        let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
        for file in std::iter::once(root.join("src/shim.rs")).chain(reg) {
            let name = file.file_name().unwrap().to_string_lossy().into_owned();
            // LF line ends, as a checkout may have CRLF
            let text = std::fs::read(&file)
                .unwrap()
                .into_iter()
                .filter(|&b| b != b'\r');
            // FNV-1a
            for b in name.bytes().chain([0]).chain(text).chain([0]) {
                digest = (digest ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        let version: u32 = std::str::from_utf8(&super::SetsunaShimVersion)
            .unwrap()
            .strip_prefix("setsuna-shim-version:")
            .and_then(|v| v.strip_suffix('\0'))
            .and_then(|v| v.parse().ok())
            .expect("setsuna-shim-version:<digits>\\0");
        assert_eq!(
            (version, format!("{digest:016x}")),
            (4, "0bb480a717419e79".to_string()),
            "the shim's sources changed: raise SetsunaShimVersion if what the shim does changed, then record both"
        );
    }
}
