//! Setsuna on Android: the Rust half of the input method service.
//!
//! The service (`app/`, Kotlin) draws the keyboard and the candidate bar
//! and owns the app's `InputConnection`. It hands each key to this library
//! as the X keysym it types (`setsuna_core::keysym`), through the JNI calls
//! of `jni_api`, and renders the `setsuna_core::session::View` that comes
//! back: text to commit, the preedit, the candidates.
//!
//! The IME logic is `setsuna-core`, shared with the TSF DLL and the fcitx5
//! addon; conversion runs in a `setsuna-engine daemon` child process, the
//! engine binary the APK carries as `libsetsuna_engine.so` (the system
//! extracts a library executable, a file of another name it does not).
//!
//! The app's settings screen is the desktop settings page in a WebView,
//! whose messages `settings` answers with crates/settings' library.

pub mod host;
mod jni_api;
pub mod settings;
