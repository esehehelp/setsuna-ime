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

#[cfg(target_os = "linux")]
mod raise;
mod rewriter;

use serde_json::{json, Value};
use setsuna_settings::Page;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

const INDEX: &str = include_str!("../web/index.html");

/// What reaches the event loop: a message from the page, a reply a thread
/// finished, or (Linux) a later start asking for the window, with the
/// activation token its launcher gave it.
pub enum UserEvent {
    Ipc(String),
    Reply(Value),
    #[cfg(target_os = "linux")]
    Present(String),
}

/// The webview's profile: in the user's local app data, under the name
/// WebView2 gave it next to the exe of an install in the user's directory
/// (`%LOCALAPPDATA%\setsuna-ime`), which keeps it.
#[cfg(windows)]
fn webview_data_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .filter(|d| !d.is_empty())
        .map(|d| {
            std::path::PathBuf::from(d)
                .join("setsuna-ime")
                .join("setsuna-settings.exe.WebView2")
        })
}

const TITLE: &str = "Setsuna 設定";

/// One settings window at a time: two each held the config as it was when
/// they opened, and the one saved last undid what the other had saved. The
/// first takes a lock in the config directory and holds it while it runs
/// (`Some`); a second finds it held, brings the first forward and ends
/// (`None`). Without a config directory or a lock to take, it runs anyway.
fn first_instance() -> Option<Option<std::fs::File>> {
    let Some(dir) = setsuna_core::config::config_dir() else {
        return Some(None);
    };
    let _ = std::fs::create_dir_all(&dir);
    let Ok(lock) = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("settings.lock"))
    else {
        return Some(None);
    };
    match lock.try_lock() {
        Ok(()) => Some(Some(lock)),
        Err(std::fs::TryLockError::WouldBlock) => None,
        Err(std::fs::TryLockError::Error(_)) => Some(None),
    }
}

/// The settings window another process has open, restored and brought to
/// the front.
#[cfg(windows)]
fn show_open_window() {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{
        FindWindowW, IsIconic, SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    unsafe {
        if let Ok(hwnd) = FindWindowW(None, &HSTRING::from(TITLE)) {
            if IsIconic(hwnd).as_bool() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            }
            let _ = SetForegroundWindow(hwnd);
        }
    }
}

/// On Linux no process may raise another's window: the one that has it open
/// is asked to, with this start's activation token (`raise`).
#[cfg(target_os = "linux")]
fn show_open_window() {
    raise::ask(&raise::launch_token());
}

#[cfg(not(any(windows, target_os = "linux")))]
fn show_open_window() {
    eprintln!("the settings window is open already");
}

fn main() -> wry::Result<()> {
    // held until the process ends: the event loop below never returns
    let Some(instance) = first_instance() else {
        show_open_window();
        return Ok(());
    };
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_inner_size(tao::dpi::LogicalSize::new(720.0, 820.0))
        .with_min_inner_size(tao::dpi::LogicalSize::new(480.0, 420.0))
        .build(&event_loop)
        .expect("window");

    // The page's messages go through the event loop, which owns the webview
    // the replies are evaluated in.
    let proxy = event_loop.create_proxy();
    // WebView2 keeps its profile next to the exe unless given a folder, and
    // an install in Program Files may not write there (E_ACCESSDENIED: the
    // window never came up). The context lives as long as `main`, which the
    // event loop never returns to.
    #[cfg(windows)]
    let mut context = wry::WebContext::new(webview_data_dir());
    #[cfg(windows)]
    let builder = WebViewBuilder::new_with_web_context(&mut context);
    #[cfg(not(windows))]
    let builder = WebViewBuilder::new();
    // The page is the one built in, and stays: it loads as inline HTML
    // (about:blank or a data: URL), and no navigation goes to a page from
    // anywhere else (one taken there would talk to the settings' IPC as the
    // page does); nor does any window open (links go to the browser
    // through `open_url`).
    let builder = builder
        .with_html(INDEX)
        .with_navigation_handler(|url| url.starts_with("about:") || url.starts_with("data:"))
        .with_new_window_req_handler(|_url, _features| wry::NewWindowResponse::Deny)
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

    // The holder of the lock is the one a later start asks for its window;
    // without a lock there may be several, and none listens.
    #[cfg(target_os = "linux")]
    if instance.is_some() {
        raise::listen(event_loop.create_proxy());
    }
    #[cfg(not(target_os = "linux"))]
    let _ = &instance;

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
            #[cfg(target_os = "linux")]
            Event::UserEvent(UserEvent::Present(token)) => {
                raise::present(&window, &token);
                None
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                #[cfg(target_os = "linux")]
                if instance.is_some() {
                    raise::forget();
                }
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
