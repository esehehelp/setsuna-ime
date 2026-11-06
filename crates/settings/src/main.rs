//! `setsuna-settings`: settings GUI for the Setsuna IME.
//!
//! A web page (`web/index.html`, compiled in) in the platform's webview
//! (WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux), so the text
//! is drawn by the system's own font rendering. The page edits the candidate
//! post-processing (symbol width, touten / kuten, middle dot, literal
//! replacements), what the symbol keys type (`[frontend.symbols]`, full- or
//! half-width), the date / time candidates, the learning and the
//! rewriter of `config.toml`, clears the learned choices, and signs the
//! rewriter in (`rewriter`). Saving keeps the file's comments and every key
//! the page does not show. The Windows IME reads the file again when an
//! input starts after it was saved, so a change applies from the next input
//! (on fcitx5, from the next activation: switching the input method away and
//! back). On Windows the saved file stays readable by AppContainers, as the
//! installer made it (`grant_app_containers`).
//!
//! The page and this process talk in JSON messages: the page posts
//! `{"type": "init" | "save" | "clear_learning" | "show_hidden", ...}`, the
//! user dictionary's `dict_load` / `dict_save` / `dict_import` (answered by
//! the library, [`setsuna_settings::Page`], which the Android app's page
//! shares), and the rewriter's (`rewriter`, this window's only)
//! through `window.ipc`, and each reply goes back as
//! `setsuna.receive(<message>)`.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod rewriter;

use serde_json::{json, Value};
use setsuna_settings::Page;
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

    let mut page = Page::load();
    let mut rewriter = rewriter::Rewriter::new(event_loop.create_proxy());
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let reply = match event {
            Event::UserEvent(UserEvent::Ipc(message)) => {
                let message: Value = serde_json::from_str(&message).unwrap_or(Value::Null);
                match page.handle(&message) {
                    Some(mut reply) => {
                        if reply["type"] == "state" {
                            reply["default_instructions"] =
                                json!(setsuna_rewrite::DEFAULT_INSTRUCTIONS);
                        }
                        Some(reply)
                    }
                    None => {
                        let action = message["type"].as_str().unwrap_or_default();
                        rewriter.handle(action, &message)
                    }
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
