//! The settings page's messages that are not the rewriter's, apart from the
//! window that shows the page: the desktop app (`src/main.rs`, a webview)
//! and the Android app (crates/android, a WebView) answer them alike.
//!
//! The page (`web/index.html`) posts `{"type": "init" | "save" |
//! "clear_learning" | "show_hidden", ...}` for `config.toml`, the learned
//! choices and the hidden candidates, and the user dictionary's
//! `dict_load` / `dict_save` / `dict_import`; [`Page::handle`] answers each
//! with the JSON the page's `setsuna.receive` takes. Saving keeps the file's
//! comments and every key the page does not show.

use std::path::PathBuf;

use serde_json::{json, Value};
use setsuna_core::config::Config;
use setsuna_core::hidden::Hidden;
use setsuna_core::learning::Learning;
use setsuna_core::user_dict::{self, Entry};

/// The config being edited, as the page last loaded or saved it.
pub struct Page {
    settings: Settings,
}

impl Page {
    pub fn load() -> Self {
        Self {
            settings: Settings::load(),
        }
    }

    /// The reply to `message`, or None when it is none of this page's (the
    /// rewriter's). `init` reads the config again: a page opened anew shows
    /// the file as it is.
    pub fn handle(&mut self, message: &Value) -> Option<Value> {
        let action = message["type"].as_str().unwrap_or_default();
        match action {
            "init" => {
                self.settings = Settings::load();
                Some(self.settings.handle(action, message))
            }
            "save" | "clear_learning" | "show_hidden" => {
                Some(self.settings.handle(action, message))
            }
            _ if action.starts_with("dict_") => Some(dictionary(action, message)),
            _ => None,
        }
    }
}

/// The reply to a user dictionary message: its entries, and a status.
fn dictionary(action: &str, message: &Value) -> Value {
    let Some(path) = user_dict::path() else {
        return json!({"type": "dict", "entries": [], "status": "設定ディレクトリが見つかりません"});
    };
    let as_json = |entries: &[Entry]| -> Vec<Value> {
        entries
            .iter()
            .map(|e| json!({"reading": e.reading, "word": e.word, "pos": e.pos, "comment": e.comment}))
            .collect()
    };
    match action {
        "dict_save" => {
            let mut entries = Vec::new();
            let mut bad = 0;
            for v in message["entries"].as_array().into_iter().flatten() {
                let s = |k: &str| v[k].as_str().unwrap_or_default().trim().to_string();
                let (word, pos) = (s("word"), s("pos"));
                let reading = match user_dict::normalize_reading(&s("reading")) {
                    Some(r) => r,
                    None if s("reading").is_empty()
                        && user_dict::Kind::of(&pos) == user_dict::Kind::Suppress =>
                    {
                        String::new()
                    }
                    None => {
                        bad += 1;
                        continue;
                    }
                };
                if word.is_empty() {
                    bad += 1;
                    continue;
                }
                let pos = if pos.is_empty() {
                    user_dict::DEFAULT_POS.to_string()
                } else {
                    pos
                };
                entries.push(Entry {
                    reading,
                    word,
                    pos,
                    comment: s("comment"),
                });
            }
            let status = match user_dict::save(&path, &entries) {
                Ok(()) if bad > 0 => format!(
                    "{} 件を保存しました（読みか語のない {bad} 件は除きました）",
                    entries.len()
                ),
                Ok(()) => format!(
                    "{} 件を保存しました。変換にすぐ反映されます。",
                    entries.len()
                ),
                Err(e) => format!("保存できませんでした: {e}"),
            };
            json!({"type": "dict", "entries": as_json(&user_dict::load(&path)), "status": status})
        }
        "dict_import" => {
            let (entries, skipped) = user_dict::parse(message["text"].as_str().unwrap_or_default());
            let status = if skipped.is_empty() {
                format!(
                    "{} 件を読み込みました。保存すると反映されます。",
                    entries.len()
                )
            } else {
                format!(
                    "{} 件を読み込みました。{} 行は読めませんでした（{}）",
                    entries.len(),
                    skipped.len(),
                    skipped
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                )
            };
            json!({"type": "dict", "imported": as_json(&entries), "status": status})
        }
        _ => {
            json!({"type": "dict", "entries": as_json(&user_dict::load(&path)), "path": path.display().to_string()})
        }
    }
}

/// Each `[frontend.symbols]` key the page lists: its name, the ASCII
/// symbol of its key and what it may type (`config::symbol_forms`); the
/// page's 全角 / 半角 do as `config::symbols_in_width`.
fn symbol_keys() -> Value {
    setsuna_core::config::SYMBOL_CHARS
        .iter()
        .filter_map(|&(name, c)| {
            let f = setsuna_core::config::symbol_forms(name)?;
            Some(json!({
                "name": name,
                "key": c.to_string(),
                "japanese": f.japanese,
                "full": f.full,
                "half": f.half,
            }))
        })
        .collect()
}

struct Settings {
    path: Option<PathBuf>,
    cfg: Config,
    /// Why the config could not be read. Saving is refused then, so a file
    /// the GUI did not understand is never overwritten with defaults.
    load_error: Option<String>,
}

impl Settings {
    fn load() -> Self {
        let path = Config::appdata_config_path();
        let (cfg, load_error) = match &path {
            Some(p) if p.exists() => match Config::load_from_path(p) {
                Ok(c) => (c, None),
                Err(e) => (Config::default(), Some(format!("{e:#}"))),
            },
            Some(_) => (Config::default(), None),
            None => (
                Config::default(),
                Some("設定ディレクトリが見つかりません".into()),
            ),
        };
        Self {
            path,
            cfg,
            load_error,
        }
    }

    fn learned(&self) -> Option<usize> {
        Config::learning_path().map(|p| Learning::at(p, &self.cfg.learning).len())
    }

    /// The candidates hidden with Ctrl+Delete, as `[reading, surface]`.
    fn hidden(&self) -> Vec<(String, String)> {
        Hidden::open().map(|mut h| h.list()).unwrap_or_default()
    }

    /// The reply to one of the config's messages from the page.
    fn handle(&mut self, action: &str, message: &Value) -> Value {
        let (ok, status) = match action {
            "init" => (true, None),
            "save" => match self.save(&message["config"]) {
                Ok(s) => (true, Some(s)),
                Err(s) => (false, Some(s)),
            },
            "clear_learning" => match self.clear_learning() {
                Ok(s) => (true, Some(s)),
                Err(s) => (false, Some(s)),
            },
            "show_hidden" => {
                let (reading, surface) = (
                    message["reading"].as_str().unwrap_or_default(),
                    message["surface"].as_str().unwrap_or_default(),
                );
                match Hidden::open().map(|mut h| h.show(reading, surface)) {
                    Some(Ok(())) => (true, Some(format!("「{surface}」をまた表示します。"))),
                    _ => (false, Some("戻せませんでした".to_string())),
                }
            }
            _ => (false, Some(format!("不明なメッセージです: {action}"))),
        };
        json!({
            "type": "state",
            "action": action,
            "ok": ok,
            "status": status,
            "path": self.path.as_ref().map(|p| p.display().to_string()),
            "load_error": self.load_error,
            "config": self.cfg,
            "learned": self.learned(),
            "hidden": self.hidden(),
            "symbol_keys": symbol_keys(),
        })
    }

    fn save(&mut self, config: &Value) -> Result<String, String> {
        if self.load_error.is_some() {
            return Err("設定を読み込めなかったため保存できません".into());
        }
        let Some(path) = self.path.clone() else {
            return Err("設定ファイルの場所が見つかりません".into());
        };
        let cfg: Config = serde_json::from_value(config.clone())
            .map_err(|e| format!("保存できませんでした: {e}"))?;
        cfg.save_to_path(&path)
            .map_err(|e| format!("保存できませんでした: {e:#}"))?;
        #[cfg(windows)]
        grant_app_containers(&path);
        self.cfg = cfg;
        Ok("保存しました。次の入力から反映されます。".into())
    }

    fn clear_learning(&mut self) -> Result<String, String> {
        let Some(path) = Config::learning_path() else {
            return Err("学習履歴の場所が見つかりません".into());
        };
        match std::fs::remove_file(&path) {
            Ok(()) => Ok("学習履歴を消去しました。".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok("学習履歴はありません。".into())
            }
            Err(e) => Err(format!("消去できませんでした: {e}")),
        }
    }
}

/// Let AppContainers read `path` again, as install.ps1 does: the IME in
/// the Start menu's search and in Store apps reads `config.toml` there, and
/// a save replaces the file (written beside it, then renamed), which drops
/// the grant. The file only: its directory holds the learning and the
/// rewriter's credentials.
#[cfg(windows)]
fn grant_app_containers(path: &std::path::Path) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("icacls")
        .arg(path)
        .args(["/grant", "*S-1-15-2-1:(R)", "*S-1-15-2-2:(R)"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

#[cfg(test)]
mod tests {
    use super::*;
    use setsuna_core::config::{
        PunctuationStyle, Replacement, RewriteApi, RewriteProvider, SymbolWidth,
    };

    /// The page edits the config as JSON: the round trip must not change it.
    #[test]
    fn config_survives_the_json_round_trip() {
        let mut cfg = Config::default();
        cfg.frontend.symbol_width = SymbolWidth::Auto;
        cfg.frontend.punctuation = PunctuationStyle::CommaKuten;
        cfg.frontend.replacements.push(Replacement {
            from: "～".into(),
            to: "〜".into(),
            enabled: false,
        });
        cfg.learning.strength = 0.7;
        cfg.learning.similarity_threshold = 0.65;
        cfg.rewriter.enabled = true;
        cfg.rewriter.provider = RewriteProvider::Compatible;
        cfg.rewriter.base_url = Some("http://127.0.0.1:8080/v1".into());
        cfg.rewriter.api = Some(RewriteApi::ChatCompletions);
        cfg.rewriter.model = "local".into();
        let json = serde_json::to_value(&cfg).unwrap();
        let back: Config = serde_json::from_value(json).unwrap();
        assert_eq!(
            toml::to_string(&back).unwrap(),
            toml::to_string(&cfg).unwrap()
        );
    }

    /// The page sends `null` for an unset optional key.
    #[test]
    fn nulls_from_the_page_are_unset_keys() {
        let mut json = serde_json::to_value(Config::default()).unwrap();
        json["rewriter"]["base_url"] = Value::Null;
        json["rewriter"]["instructions"] = Value::Null;
        let back: Config = serde_json::from_value(json).unwrap();
        assert_eq!(back.rewriter.base_url, None);
    }

    /// The rewriter's messages are not the page's: the window hands them on.
    #[test]
    fn only_the_page_s_messages_are_answered() {
        let mut page = Page {
            settings: Settings {
                path: None,
                cfg: Config::default(),
                load_error: None,
            },
        };
        assert!(page.handle(&json!({"type": "sign_in"})).is_none());
        assert!(page.handle(&json!({"type": "list_models"})).is_none());
        let reply = page.handle(&json!({"type": "save", "config": {}})).unwrap();
        assert_eq!(reply["type"], "state");
        assert_eq!(reply["ok"], false);
    }
}
