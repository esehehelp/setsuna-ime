//! The text service TSF loads in the IME DLL's place, so an update reaches
//! the apps that are running.
//!
//! An app keeps the DLL TSF loaded into it until it exits: an update
//! installs a DLL of another name, and only apps started after it load that
//! one. The shim is that DLL instead, and rarely changes. It loads the IME
//! DLL (the core) named in `core.txt` beside it, makes a text service of it
//! (`SetsunaCoreCreate`) and forwards TSF's calls to it. When `core.txt`
//! names another core, the next focus change swaps the running text service
//! for one of the new core: the old one is deactivated, the new one activated
//! on the same thread manager and client id, with no composition open (the
//! swap waits for it: `SetsunaCoreComposing`). TSF keeps talking to the shim
//! and sees nothing change. An old core stays loaded, unused (its threads end
//! with its text service; freeing a DLL whose code a thread may still run is
//! not worth a few MB).
//!
//! A spike: registered under a CLSID and profile of its own ("Setsuna (shim
//! test)"), beside the IME's, with `core.txt` written by hand.

#[cfg(windows)]
mod shim;
