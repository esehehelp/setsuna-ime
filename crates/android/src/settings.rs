//! The settings page: crates/settings' page (`web/index.html`, an asset of
//! the APK) in the app's WebView, its config and dictionary messages
//! answered by `setsuna_settings::Page` as the desktop app answers them, on
//! the app's own config.toml, learning and user dictionary
//! (`files/config/setsuna-ime`). The service reads config.toml again when a
//! field is focused, so a saved change applies from the next field.

use std::path::Path;
use std::sync::Mutex;

use serde_json::Value;
use setsuna_settings::Page;

static PAGE: Mutex<Option<Page>> = Mutex::new(None);

/// The reply to a message of the page, as JSON, with the config under the
/// files directory `files`; None for a message it does not answer (the
/// rewriter's: the app has no rewriter) or one that is no JSON.
pub fn handle(files: &Path, message: &str) -> Option<String> {
    crate::host::export_files(files);
    let message: Value = serde_json::from_str(message).ok()?;
    let mut page = PAGE.lock().unwrap_or_else(|e| e.into_inner());
    let page = page.get_or_insert_with(Page::load);
    page.handle(&message).map(|reply| reply.to_string())
}

/// Where XDG_CONFIG_HOME names the config directory, as on Android:
/// elsewhere it would be this user's own.
#[cfg(all(test, unix, not(target_os = "macos")))]
mod tests {
    use super::*;

    /// The page edits the app's config.toml; the rewriter's messages get no
    /// reply.
    #[test]
    fn the_page_edits_the_app_s_config() {
        let files =
            std::env::temp_dir().join(format!("setsuna-android-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&files);
        std::fs::create_dir_all(&files).unwrap();
        let state: Value =
            serde_json::from_str(&handle(&files, r#"{"type":"init"}"#).unwrap()).unwrap();
        let path = files.join("config").join("setsuna-ime").join("config.toml");
        assert_eq!(state["type"], "state");
        assert_eq!(state["path"], path.display().to_string());
        assert!(handle(&files, r#"{"type":"rewriter_state"}"#).is_none());
        assert!(handle(&files, "not json").is_none());

        let mut config = state["config"].clone();
        config["frontend"]["typo_correction"] = Value::Bool(false);
        let message = serde_json::json!({"type": "save", "config": config}).to_string();
        let saved: Value = serde_json::from_str(&handle(&files, &message).unwrap()).unwrap();
        assert_eq!(saved["ok"], true, "{saved}");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("typo_correction = false"), "{text}");
        let _ = std::fs::remove_dir_all(&files);
    }
}
