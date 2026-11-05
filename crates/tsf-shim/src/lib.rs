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

#[cfg(windows)]
mod shim;
