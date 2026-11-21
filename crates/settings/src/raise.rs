//! Linux: a second start of the settings brings the window that is open to
//! the front.
//!
//! The start that holds the settings' lock listens on a socket in the
//! user's runtime directory; a start that finds the lock taken sends it the
//! activation token its launcher gave it and exits. A Wayland compositor
//! lets a window come to the front only for such a token (without one the
//! window asks for attention in the task bar), and no process may raise
//! another's window itself.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

use tao::event_loop::EventLoopProxy;

use crate::UserEvent;

/// The longest token taken: the compositors' are a few dozen characters.
const MAX_TOKEN: u64 = 512;

fn socket_path() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty())?);
    dir.is_absolute().then(|| dir.join("setsuna-settings.sock"))
}

/// The token the launcher gave this start (xdg-activation, or the startup
/// notification's id it is also passed as), "" for none. Read before the
/// toolkit starts, which takes it out of the environment.
pub fn launch_token() -> String {
    ["XDG_ACTIVATION_TOKEN", "DESKTOP_STARTUP_ID"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|token| is_token(token))
        .unwrap_or_default()
}

/// A token is passed on only as printable ASCII without spaces.
fn is_token(token: &str) -> bool {
    !token.is_empty()
        && token.len() as u64 <= MAX_TOKEN
        && token.bytes().all(|b| b.is_ascii_graphic())
}

/// The start that holds the lock: every later start's token reaches the
/// event loop as [`UserEvent::Present`]. A socket a start before left is
/// taken over; without a runtime directory nothing listens, and a second
/// start only exits.
pub fn listen(proxy: EventLoopProxy<UserEvent>) {
    let Some(path) = socket_path() else {
        return;
    };
    let _ = std::fs::remove_file(&path);
    let Ok(listener) = UnixListener::bind(&path) else {
        return;
    };
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
            let mut token = String::new();
            let _ = stream.take(MAX_TOKEN + 1).read_to_string(&mut token);
            let token = if is_token(&token) {
                token
            } else {
                String::new()
            };
            if proxy.send_event(UserEvent::Present(token)).is_err() {
                break;
            }
        }
    });
}

/// A start that found the lock taken: ask the one that holds it to show its
/// window, with this start's `token`.
pub fn ask(token: &str) {
    let Some(path) = socket_path() else {
        return;
    };
    if let Ok(mut stream) = UnixStream::connect(path) {
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
        let _ = stream.write_all(token.as_bytes());
    }
}

/// Bring `window` to the front for a later start's `token` ("" when it had
/// none: the compositor then decides, and may only mark the window).
pub fn present(window: &tao::window::Window, token: &str) {
    use gtk::prelude::GtkWindowExt;
    use tao::platform::unix::WindowExtUnix;

    window.set_minimized(false);
    let gtk_window = window.gtk_window();
    if !token.is_empty() {
        // on Wayland this activates the window's surface with the token
        gtk_window.set_startup_id(token);
    }
    gtk_window.present();
}

/// The window closed: the socket goes with it.
pub fn forget() {
    if let Some(path) = socket_path() {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_printable_ascii_of_a_bounded_length() {
        assert!(is_token("kwin-1795-42_TIME123"));
        assert!(!is_token(""));
        assert!(!is_token("two words"));
        assert!(!is_token("改行\n"));
        assert!(!is_token(&"a".repeat(MAX_TOKEN as usize + 1)));
    }
}
