//! `setsuna-settings`: settings GUI for the Setsuna IME.
//!
//! A web page (`web/index.html`, compiled in) in the platform's webview
//! (WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux), so the text
//! is drawn by the system's own font rendering. The page edits the candidate
//! post-processing (symbol width, touten / kuten, middle dot, literal
//! replacements), the date / time candidates, the learning and the
//! rewriter of `config.toml`, clears the learned choices, and signs the
//! rewriter in (`rewriter`). Saving keeps the file's comments and every key
//! the page does not show. The IME reads the file when it is activated, so
//! a change applies from the next activation (Win+Space twice on Windows,
//! switching the input method away and back on fcitx5).
//!
//! The page and this process talk in JSON messages: the page posts
//! `{"type": "init" | "save" | "clear_learning" | "show_hidden", ...}` (and the rewriter's)
//! through `window.ipc`, and each reply goes back as
//! `setsuna.receive(<message>)`.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod rewriter;

use std::path::PathBuf;

use serde_json::{json, Value};
use setsuna_core::config::Config;
use setsuna_core::hidden::Hidden;
use setsuna_core::learning::Learning;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

const INDEX: &str = include_str!("../web/index.html");

/// What reaches the event loop: a message from the page, or a reply a
/// thread finished.
pub enum UserEvent {
    Ipc(String),
    Reply(Value),
}

fn main() -> wry::Result<()> {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title("Setsuna 設定")
        .with_inner_size(tao::dpi::LogicalSize::new(720.0, 820.0))
        .with_min_inner_size(tao::dpi::LogicalSize::new(480.0, 420.0))
        .build(&event_loop)
        .expect("window");

    // The page's messages go through the event loop, which owns the webview
    // the replies are evaluated in.
    let proxy = event_loop.create_proxy();
    let builder = WebViewBuilder::new()
        .with_html(INDEX)
        .with_ipc_handler(move |req| {
            let _ = proxy.send_event(UserEvent::Ipc(req.into_body()));
        });
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    let webview = builder.build(&window)?;
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let webview = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        builder.build_gtk(window.default_vbox().expect("gtk vbox"))?
    };

    let mut app = Settings::load();
    let mut rewriter = rewriter::Rewriter::new(event_loop.create_proxy());
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let reply = match event {
            Event::UserEvent(UserEvent::Ipc(message)) => {
                let message: Value = serde_json::from_str(&message).unwrap_or(Value::Null);
                let action = message["type"].as_str().unwrap_or_default();
                if matches!(action, "init" | "save" | "clear_learning" | "show_hidden") {
                    Some(app.handle(action, &message))
                } else {
                    rewriter.handle(action, &message)
                }
            }
            Event::UserEvent(UserEvent::Reply(reply)) if reply["type"] == "rewriter_refresh" => {
                Some(rewriter.refreshed(&reply))
            }
            Event::UserEvent(UserEvent::Reply(reply)) => Some(reply),
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
                None
            }
            _ => None,
        };
        if let Some(reply) = reply {
            let _ = webview.evaluate_script(&format!("setsuna.receive({reply})"));
        }
    });
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
            "default_instructions": setsuna_rewrite::DEFAULT_INSTRUCTIONS,
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
        self.cfg = cfg;
        Ok("保存しました。IME を再度有効にすると反映されます。".into())
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
}
