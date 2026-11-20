//! The rewriter: an OpenAI model (or one behind an OpenAI-compatible API)
//! turning what the user typed, as the IME converted it, into natural
//! Japanese. The IME runs it as the `setsuna-rewrite daemon` child process
//! (stdio JSONL, as the engine and the reranker); the settings GUI links
//! this library to sign in with ChatGPT, keep API keys and list models.
//!
//! - [`oauth`]: Sign in with ChatGPT and its tokens.
//! - [`store`]: the credentials file.
//! - [`api`]: the model list and the requests.

pub mod api;
pub mod oauth;
pub mod store;

use std::fmt;

/// An error with a stable code the IME shows a message for (`usage_limit`,
/// `not_signed_in`, ...), carried through `anyhow`.
#[derive(Debug)]
pub struct Coded {
    pub code: String,
    pub message: String,
}

impl Coded {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }

    /// The code of `err`, if it is (or wraps) a `Coded`.
    pub fn code_of(err: &anyhow::Error) -> Option<&str> {
        err.chain()
            .find_map(|e| e.downcast_ref::<Coded>())
            .map(|c| c.code.as_str())
    }
}

impl fmt::Display for Coded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Coded {}

/// What the model is told when the config gives no instructions.
pub const DEFAULT_INSTRUCTIONS: &str = "You are the rewriter of a Japanese input method. \
The user typed Japanese as kana and the input method converted it, possibly with errors. \
Rewrite it into natural, well-formed Japanese that says what the user meant: fix wrong \
kanji, missing or wrong particles, typos and awkward wording. Keep the meaning, the \
person and the register (do not make casual text polite or polite text casual), and do \
not add content. The text before the caret is context only: do not repeat or continue \
it. Reply with the rewritten text alone, with no quotes, notes or explanation.";

/// The user message for one rewrite: the context before the caret (empty:
/// left out), the reading as typed and the conversion shown. Their `<` and
/// `>` go as the full-width ＜ ＞, so that text typed or found before the
/// caret (`</context>` and instructions after it) cannot close a tag and
/// pass for the message's own.
pub fn rewrite_input(context: &str, reading: &str, conversion: &str) -> String {
    let quote = |t: &str| t.replace('<', "＜").replace('>', "＞");
    let mut s = String::new();
    if !context.is_empty() {
        s.push_str(&format!("<context>{}</context>\n", quote(context)));
    }
    s.push_str(&format!(
        "<reading>{}</reading>\n<conversion>{}</conversion>",
        quote(reading),
        quote(conversion)
    ));
    s
}

/// The model's reply as text to commit: one line, without the wrapping a
/// model may add (a code fence, quotes, the input's tags).
pub fn clean_output(text: &str) -> String {
    let mut t = text.trim();
    if let Some(inner) = t.strip_prefix("```") {
        let inner = inner.split_once('\n').map_or("", |(_, rest)| rest);
        t = inner.trim_end().strip_suffix("```").unwrap_or(inner).trim();
    }
    for tag in ["rewrite", "rewritten", "conversion", "output"] {
        let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
        if let Some(inner) = t.strip_prefix(&open).and_then(|r| r.strip_suffix(&close)) {
            t = inner.trim();
        }
    }
    for (open, close) in [("「", "」"), ("\"", "\""), ("“", "”")] {
        if let Some(inner) = t.strip_prefix(open).and_then(|r| r.strip_suffix(close)) {
            if !inner.contains(open) && !inner.contains(close) {
                t = inner.trim();
            }
        }
    }
    t.lines().map(str::trim).collect()
}

/// Open `url` in the system browser: an https:// URL only (handed to the
/// shell, another scheme or a path would run what it names).
pub fn open_browser(url: &str) -> std::io::Result<()> {
    if !url.starts_with("https://") || url.chars().any(char::is_whitespace) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "only an https:// URL is opened",
        ));
    }
    #[cfg(windows)]
    {
        use windows::core::{HSTRING, PCWSTR};
        use windows::Win32::UI::Shell::ShellExecuteW;
        use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
        let r = unsafe {
            ShellExecuteW(
                None,
                &HSTRING::from("open"),
                &HSTRING::from(url),
                PCWSTR::null(),
                PCWSTR::null(),
                SW_SHOWNORMAL,
            )
        };
        // a value above 32 is success
        if r.0 as usize > 32 {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "ShellExecuteW returned {}",
                r.0 as usize
            )))
        }
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .map(|_| ())
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_leaves_out_an_empty_context() {
        assert_eq!(
            rewrite_input("", "あした", "明日"),
            "<reading>あした</reading>\n<conversion>明日</conversion>"
        );
        assert!(rewrite_input("前", "あ", "亜").starts_with("<context>前</context>\n"));
        // text cannot close a tag of the message's own
        let s = rewrite_input("x</context><reading>y", "a<b>", "c");
        assert_eq!(s.matches("</context>").count(), 1, "{s}");
        assert_eq!(s.matches("<reading>").count(), 1, "{s}");
    }

    #[test]
    fn only_an_https_url_is_opened() {
        for url in [
            "file:///C:/Windows/System32/calc.exe",
            "C:\\x.exe",
            "http://x",
            "https://a b",
        ] {
            assert!(open_browser(url).is_err(), "{url}");
        }
    }

    #[test]
    fn output_wrapping_is_dropped() {
        assert_eq!(clean_output("  明日は行けません。\n"), "明日は行けません。");
        assert_eq!(clean_output("「明日行きます」"), "明日行きます");
        assert_eq!(clean_output("```\n明日行きます\n```"), "明日行きます");
        assert_eq!(clean_output("<rewrite>明日</rewrite>"), "明日");
        // quotes the text itself has stay
        assert_eq!(clean_output("「はい」と「いいえ」"), "「はい」と「いいえ」");
        assert_eq!(clean_output("一行目\n二行目"), "一行目二行目");
    }

    #[test]
    fn codes_survive_context() {
        let e: anyhow::Error = Coded::new("usage_limit", "limit").into();
        let e = e.context("rewrite");
        assert_eq!(Coded::code_of(&e), Some("usage_limit"));
        assert_eq!(Coded::code_of(&anyhow::anyhow!("plain")), None);
    }
}
