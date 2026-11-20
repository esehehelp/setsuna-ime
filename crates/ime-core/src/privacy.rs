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
//! A token is looked at whole and in its pieces between `=`, `:`, quotes
//! and brackets (`KEY=sk-...`, `host=10.0.0.1`, `` `ghp_...` ``), and an
//! identifier by its parts (`OPENAI_API_KEY`, `X-API-Key`). A word that
//! announces a credential takes the rest of its sentence with it: what
//! follows it is the credential (`パスワードは hunter2 です`).
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

/// The same in Japanese, matched anywhere (Latin letters in any case).
const CREDENTIAL_WORDS_JA: &[&str] = &[
    "パスワード",
    "暗証番号",
    "認証コード",
    "認証番号",
    "確認コード",
    "秘密鍵",
    "秘密キー",
    "トークン",
    "apiキー",
    "ａｐｉキー",
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

/// Everything sensitive in `text`, in order; a word announcing a
/// credential with the rest of its sentence.
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
            match token_hit(text, s, e) {
                Some(Found::Secret) => out.push(Hit { start: s, end: e }),
                Some(Found::Announcer) => out.push(Hit {
                    start: s,
                    end: sentence_end(text, e),
                }),
                None => {}
            }
            start = None;
        }
    }
    let folded = fold_case(text);
    for word in CREDENTIAL_WORDS_JA {
        for (i, _) in folded.match_indices(word) {
            out.push(Hit {
                start: i,
                end: sentence_end(text, i + word.len()),
            });
        }
    }
    out.extend(digit_runs(text));
    out.sort_by_key(|h| (h.start, h.end));
    out
}

/// Where the sentence that goes on at byte `from` ends: its full stop or
/// mark (a `.` with a space or nothing after it: `abc.def` goes on), the
/// line's end, or the text's.
fn sentence_end(text: &str, from: usize) -> usize {
    let rest = &text[from..];
    rest.char_indices()
        .find(|&(i, c)| {
            matches!(c, '。' | '．' | '！' | '？' | '!' | '?' | '\n')
                || c == '.' && rest[i + 1..].chars().next().is_none_or(char::is_whitespace)
        })
        .map_or(text.len(), |(i, _)| from + i)
}

/// `text` with its Latin capitals (half and full width) in lowercase, every
/// char as long in bytes as it was.
fn fold_case(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'Ａ'..='Ｚ' => char::from_u32(c as u32 + 0x20).unwrap_or(c),
            c => c.to_ascii_lowercase(),
        })
        .collect()
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

/// What an ASCII token is when sensitive.
enum Found {
    /// A secret, an address, a path: the token itself.
    Secret,
    /// A word announcing a credential, which follows it.
    Announcer,
}

/// Whether the ASCII token `text[s..e]` (and so where it is) is sensitive.
fn token_hit(text: &str, s: usize, e: usize) -> Option<Found> {
    let token: String = text[s..e].chars().filter_map(to_ascii).collect();
    let token = trim_punctuation(&token);
    if token.is_empty() {
        return None;
    }
    // the token whole, and its pieces between `=`, `:`, quotes, brackets
    if is_secret(token)
        || token
            .split([
                '=', ':', '`', '"', '\'', '(', ')', '[', ']', '{', '}', '<', '>', ',', ';',
            ])
            .any(|p| is_secret(trim_punctuation(p)))
    {
        return Some(Found::Secret);
    }
    if announces(token) {
        return Some(Found::Announcer);
    }
    // identifier-like pieces of the token
    for piece in token.split(|c: char| {
        !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+' | '/' | '='))
    }) {
        if secret_like(piece) {
            return Some(Found::Secret);
        }
    }
    None
}

/// `t` without the sentence punctuation around it (`example.com.`,
/// `(foo)`).
fn trim_punctuation(t: &str) -> &str {
    t.trim_end_matches(['.', ',', ')', '!', '?', ';', ':', '"', '\''])
        .trim_start_matches(['(', '"', '\''])
}

/// Whether `t` is an address, a path or a key by its shape.
fn is_secret(t: &str) -> bool {
    if t.is_empty() {
        return false;
    }
    let lower = t.to_ascii_lowercase();
    lower.contains("://")
        || lower.starts_with("www.")
        || lower.starts_with("mailto:")
        || is_email(t)
        || is_domain(&lower)
        // a domain with a path or a port (`github.com/org/repo`,
        // `example.com:8443`)
        || lower
            .split(['/', ':'])
            .next()
            .is_some_and(|host| host.len() < lower.len() && is_domain(host))
        || is_path(t)
        || is_ipv4(t)
        || is_ipv6(t.trim_start_matches('[').split(']').next().unwrap_or(t))
        // a PEM block's header, whatever follows it
        || t.starts_with("-----BEGIN")
        || SECRET_PREFIXES
            .iter()
            .any(|p| t.starts_with(p) && t.len() > p.len() + 3)
        || aws_key(t)
        || jwt(t)
}

/// Whether `token` has a word announcing a credential: whole
/// (`password`), or a part or two of an identifier (`DB_PASSWORD`,
/// `OPENAI_API_KEY`, `X-API-Key`).
fn announces(token: &str) -> bool {
    let known = |w: &str| CREDENTIAL_WORDS.contains(&w);
    token
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
        .any(|word| {
            let w = word.to_ascii_lowercase();
            let parts: Vec<&str> = w.split(['_', '-']).filter(|p| !p.is_empty()).collect();
            known(&w)
                || parts.len() > 1
                    && (parts.iter().any(|p| known(p))
                        || parts.windows(2).any(|p| {
                            known(&format!("{}{}", p[0], p[1]))
                                || known(&format!("{}_{}", p[0], p[1]))
                        }))
        })
}

fn is_email(t: &str) -> bool {
    let Some((local, domain)) = t.split_once('@') else {
        return false;
    };
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

/// `label.label.tld` with a known TLD (`example.com`, `foo.co.jp`,
/// `x.com`), not `e.g.` or `node.js`.
fn is_domain(lower: &str) -> bool {
    let labels: Vec<&str> = lower.split('.').collect();
    labels.len() >= 2
        && labels
            .iter()
            .all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        && TLDS.contains(labels.last().unwrap())
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

/// An IPv6 address (`fe80::1`, `2001:db8::8a2e:370:7334`, a zone after
/// `%`): hex groups, with `::` or eight of them (a time, 12:34:56, is
/// neither).
fn is_ipv6(t: &str) -> bool {
    let t = t.split('%').next().unwrap_or(t);
    let groups: Vec<&str> = t.split(':').collect();
    t.chars().all(|c| c.is_ascii_hexdigit() || c == ':')
        && t.chars().any(|c| c.is_ascii_hexdigit())
        && (3..=8).contains(&groups.len())
        && groups.iter().all(|g| g.len() <= 4)
        && (t.contains("::") || groups.len() == 8)
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
            } else if matches!(c, '-' | ' ' | 'ー' | '－' | '(' | ')' | '（' | '）')
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
    fn keys_in_assignments_identifiers_and_quotes_are_found() {
        for t in [
            "-----BEGIN RSA PRIVATE KEY-----",
            "OPENAI_API_KEY=abc",
            "DB_PASSWORD=hunter2",
            "AUTH_TOKEN: x",
            "X-API-Key: abc",
            "KEY=sk-projAbCdEfGh",
            "`sk-1234567`を使う",
            "192.168.0.1:8080に",
            "host=10.0.0.1",
            "fe80::1%eth0",
            "[2001:db8::1]:443",
            "github.com/org/private-repo",
            "example.com:8443",
            "x.comで見た",
            "apiキーを",
            "ａｐｉキー",
            "03(1234)5678まで",
        ] {
            assert!(sensitive(t), "{t}");
        }
        for t in ["12:34:56に", "比率は16:9", "a:b"] {
            assert!(!sensitive(t), "{t}: {:?}", hits(t));
        }
    }

    #[test]
    fn the_local_context_starts_after_the_last_hit() {
        // what follows a word announcing a credential, to the sentence's
        // end, is the credential
        assert_eq!(after_last_hit("前の文。パスワードは"), "");
        assert_eq!(
            after_last_hit("パスワードは mypass123 です。次の文"),
            "。次の文"
        );
        assert_eq!(after_last_hit("my password is hunter2. Then"), ". Then");
        assert_eq!(
            after_last_hit("https://example.com/を見て、それから"),
            "を見て、それから"
        );
        assert_eq!(after_last_hit("普通の文"), "普通の文");
    }
}
