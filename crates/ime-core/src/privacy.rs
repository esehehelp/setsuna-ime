//! Text that should not leave the machine, nor steer a conversion: what
//! the text before the caret may hold besides prose. The rules follow
//! Mozkey's context privacy policy (a Mozc fork, BSD-3-Clause:
//! `src/session/zenz_text_privacy_analyzer.cc`), matched on token
//! boundaries instead of substrings (`tokenizer` is no token) and with more
//! secret prefixes: URLs, e-mail addresses, domains, paths, IPv4
//! addresses, API keys and tokens (`sk-`, `ghp_`, `AKIA`, JWTs, long hex or
//! base64), long digit runs (card and phone numbers), and the words that
//! announce a credential (password, パスワード, 暗証番号).
//!
//! The rewriter, which sends the context to a server, drops a context with
//! any hit; the local model reads the text after the last one.

/// A hit: the byte range of something sensitive in the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub start: usize,
    pub end: usize,
}

/// Prefixes of API keys and tokens, as their issuers write them.
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "rk_test_",
    "pk_live_",
    "pk_test_",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "xoxa-",
    "xoxb-",
    "xoxp-",
    "xoxs-",
    "xapp-",
    "ya29.",
    "AIza",
    "npm_",
    "hf_",
    "-----BEGIN",
];

/// Words that announce a credential, matched as whole ASCII words in any
/// case.
const CREDENTIAL_WORDS: &[&str] = &[
    "password",
    "passwd",
    "passphrase",
    "pwd",
    "passcode",
    "pin",
    "secret",
    "token",
    "apikey",
    "api_key",
    "api-key",
    "credential",
    "credentials",
    "privatekey",
    "private_key",
    "authorization",
    "bearer",
    "cookie",
    "sessionid",
    "session_id",
    "client_secret",
    "access_key",
    "otp",
];

/// The same in Japanese, matched anywhere.
const CREDENTIAL_WORDS_JA: &[&str] = &[
    "パスワード",
    "暗証番号",
    "認証コード",
    "認証番号",
    "確認コード",
    "秘密鍵",
    "秘密キー",
    "トークン",
    "APIキー",
    "ＡＰＩキー",
    "口座番号",
    "カード番号",
    "セキュリティコード",
];

const TLDS: &[&str] = &[
    "com",
    "net",
    "org",
    "jp",
    "io",
    "dev",
    "app",
    "ai",
    "co",
    "info",
    "biz",
    "me",
    "local",
    "localhost",
    "edu",
    "gov",
    "us",
    "uk",
];

/// Everything sensitive in `text`, in order.
pub fn hits(text: &str) -> Vec<Hit> {
    let mut out = Vec::new();
    // ASCII tokens: runs of printable ASCII (full-width forms folded)
    let mut start = None;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let ascii = to_ascii(c).is_some_and(|a| a.is_ascii_graphic());
        if ascii && start.is_none() {
            start = Some(i);
        }
        let end_here = !ascii || chars.peek().is_none();
        if let (Some(s), true) = (start, end_here) {
            let e = if ascii { i + c.len_utf8() } else { i };
            if let Some(hit) = token_hit(text, s, e) {
                out.push(hit);
            }
            start = None;
        }
    }
    for word in CREDENTIAL_WORDS_JA {
        for (i, _) in text.match_indices(word) {
            out.push(Hit {
                start: i,
                end: i + word.len(),
            });
        }
    }
    out.extend(digit_runs(text));
    out.sort_by_key(|h| (h.start, h.end));
    out
}

/// Whether `text` holds anything sensitive.
pub fn is_sensitive(text: &str) -> bool {
    !hits(text).is_empty()
}

/// The text after the last sensitive thing in it (all of it when there is
/// none): what a local model may read as context.
pub fn after_last_hit(text: &str) -> &str {
    match hits(text).iter().map(|h| h.end).max() {
        Some(end) => &text[end..],
        None => text,
    }
}

fn to_ascii(c: char) -> Option<char> {
    match c {
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0),
        c if c.is_ascii() => Some(c),
        _ => None,
    }
}

/// Whether the ASCII token `text[s..e]` (and so where it is) is sensitive.
fn token_hit(text: &str, s: usize, e: usize) -> Option<Hit> {
    let token: String = text[s..e].chars().filter_map(to_ascii).collect();
    // trailing sentence punctuation is no part of it (`example.com.`)
    let token = token.trim_end_matches(['.', ',', ')', '!', '?', ';', ':', '"', '\'']);
    let token = token.trim_start_matches(['(', '"', '\'']);
    let hit = Some(Hit { start: s, end: e });
    if token.is_empty() {
        return None;
    }
    let lower = token.to_ascii_lowercase();
    if lower.contains("://") || lower.starts_with("www.") || lower.starts_with("mailto:") {
        return hit;
    }
    if is_email(token) || is_domain(&lower) || is_path(token) || is_ipv4(token) {
        return hit;
    }
    if SECRET_PREFIXES
        .iter()
        .any(|p| token.starts_with(p) && token.len() > p.len() + 3)
    {
        return hit;
    }
    if aws_key(token) || jwt(token) {
        return hit;
    }
    // credential words and identifier-like pieces of the token
    for word in token.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-') {
        let w = word.to_ascii_lowercase();
        if CREDENTIAL_WORDS.contains(&w.as_str()) {
            return hit;
        }
    }
    for piece in token.split(|c: char| {
        !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+' | '/' | '='))
    }) {
        if secret_like(piece) {
            return hit;
        }
    }
    None
}

fn is_email(t: &str) -> bool {
    let Some((local, domain)) = t.split_once('@') else {
        return false;
    };
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

/// `label.label.tld` with a known TLD (`example.com`, `foo.co.jp`), not
/// `e.g.` or `node.js`.
fn is_domain(lower: &str) -> bool {
    let labels: Vec<&str> = lower.split('.').collect();
    labels.len() >= 2
        && labels
            .iter()
            .all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        && TLDS.contains(labels.last().unwrap())
        && labels[0].len() >= 2
}

fn is_path(t: &str) -> bool {
    let b = t.as_bytes();
    let drive =
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && matches!(b[2], b'\\' | b'/');
    drive
        || t.starts_with("\\\\")
        || t.contains('\\')
        || t.starts_with("~/")
        || t.starts_with("./")
        || t.starts_with("../")
        // `/usr/bin`, not `/` or `1/2`
        || t.starts_with('/') && t[1..].contains('/')
}

fn is_ipv4(t: &str) -> bool {
    let parts: Vec<&str> = t.split('.').collect();
    parts.len() == 4
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.len() <= 3 && p.chars().all(|c| c.is_ascii_digit()))
}

fn aws_key(t: &str) -> bool {
    (t.starts_with("AKIA") || t.starts_with("ASIA"))
        && t.len() >= 20
        && t[4..20]
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn jwt(t: &str) -> bool {
    t.starts_with("eyJ") && t.split('.').filter(|p| !p.is_empty()).count() >= 3
}

/// A version number (`v1.2.3`, `3.10`, `2.0-beta4`) is no secret.
fn is_version(t: &str) -> bool {
    let t = t.strip_prefix(['v', 'V']).unwrap_or(t);
    let (num, suffix) = t.split_once('-').unwrap_or((t, ""));
    let groups: Vec<&str> = num.split('.').collect();
    (2..=4).contains(&groups.len())
        && groups
            .iter()
            .all(|g| !g.is_empty() && g.len() <= 6 && g.chars().all(|c| c.is_ascii_digit()))
        && suffix.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Mozkey's token shapes: what an identifier, key or hash looks like and a
/// word does not.
fn secret_like(p: &str) -> bool {
    if p.is_empty() || is_version(p) {
        return false;
    }
    let len = p.len();
    let letter = p.chars().any(|c| c.is_ascii_alphabetic());
    let digit = p.chars().any(|c| c.is_ascii_digit());
    let joiner = p.chars().any(|c| matches!(c, '_' | '-' | '.'));
    let hex = p.chars().all(|c| c.is_ascii_hexdigit());
    let longest_digits = p
        .split(|c: char| !c.is_ascii_digit())
        .map(str::len)
        .max()
        .unwrap_or(0);
    longest_digits >= 8
        || len >= 12 && letter && digit && joiner
        || len >= 16 && letter && digit
        || len >= 16 && hex && letter
        || len >= 24 && letter && digit
        || len >= 32
}

/// Digits that may be a card, phone or account number: ten or more, a
/// space or `-` between groups allowed (090-1234-5678), and eight or more
/// in a row.
fn digit_runs(text: &str) -> Vec<Hit> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let digit = |c: char| c.is_ascii_digit() || ('０'..='９').contains(&c);
    let mut k = 0;
    while k < chars.len() {
        if !digit(chars[k].1) {
            k += 1;
            continue;
        }
        let start = k;
        let (mut digits, mut run, mut longest) = (0, 0, 0);
        let mut end = k;
        while end < chars.len() {
            let c = chars[end].1;
            if digit(c) {
                digits += 1;
                run += 1;
                longest = longest.max(run);
            } else if matches!(c, '-' | ' ' | 'ー' | '－')
                && end + 1 < chars.len()
                && digit(chars[end + 1].1)
                && end > start
                && digit(chars[end - 1].1)
            {
                run = 0;
            } else {
                break;
            }
            end += 1;
        }
        if digits >= 10 || longest >= 8 {
            let to = chars.get(end).map_or(text.len(), |c| c.0);
            out.push(Hit {
                start: chars[start].0,
                end: to,
            });
        }
        k = end.max(k + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensitive(t: &str) -> bool {
        is_sensitive(t)
    }

    #[test]
    fn secrets_and_addresses_are_sensitive() {
        for t in [
            "詳細はhttps://example.com/aを参照",
            "www.example.jpにある",
            "連絡先はfoo.bar@example.co.jpです",
            "example.comで",
            "C:\\Users\\me\\secret.txtを開く",
            "~/projects/setsuna",
            "/usr/local/bin",
            "192.168.0.1に接続",
            "キーはsk-proj-abcdEFGH1234でした",
            "ghp_0123456789abcdefABCDEF",
            "AKIAIOSFODNN7EXAMPLE",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
            "d41d8cd98f00b204e9800998ecf8427e",
            "パスワードは",
            "my password is",
            "暗証番号",
            "090-1234-5678に電話",
            "4111 1111 1111 1111",
            "12345678",
        ] {
            assert!(sensitive(t), "{t}");
        }
    }

    #[test]
    fn prose_and_names_are_not() {
        for t in [
            "今日はいい天気ですね。",
            "GitHubでissueを立てた",
            "Python 3.12とv1.2.3とGPT-4",
            "tokenizerの設定",
            "e.g. this, node.js and Wi-Fi",
            "1/2の確率で",
            "2026年10月15日",
            "〒123-4567",
            "Hello, world!",
            "C++とC#",
        ] {
            assert!(!sensitive(t), "{t}: {:?}", hits(t));
        }
    }

    #[test]
    fn the_local_context_starts_after_the_last_hit() {
        assert_eq!(after_last_hit("前の文。パスワードは"), "は");
        assert_eq!(
            after_last_hit("https://example.com/を見て、それから"),
            "を見て、それから"
        );
        assert_eq!(after_last_hit("普通の文"), "普通の文");
    }
}
