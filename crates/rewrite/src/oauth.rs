//! Sign in with ChatGPT, with ChatGPT plan usage, as OpenAI documents it
//! for open-source apps that run locally
//! (<https://developers.openai.com/siwc/token-sharing-open-source>):
//!
//! 1. a loopback listener on `127.0.0.1`, fresh `state`, `nonce` and PKCE;
//! 2. the authorization page in the system browser, registering a client
//!    (`dynamic_agent_client`) the first time and with the issued client ID
//!    after that;
//! 3. the code and the issued client ID on the callback, `state` checked;
//! 4. the code exchanged with PKCE (no client secret);
//! 5. the ID token's signature, issuer, audience, expiry and nonce checked,
//!    and the plan-usage scope looked for;
//! 6. the tokens saved (`store`), refreshed near expiry, revoked on sign-out.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::store::{ChatGptAccount, Store};
use crate::Coded;

pub const ISSUER: &str = "https://auth.openai.com";
const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
const REVOKE_URL: &str = "https://auth.openai.com/api/accounts/oauth/revoke";
const JWKS_URL: &str = "https://auth.openai.com/.well-known/jwks.json";
/// The API the tokens are for (`resource`).
pub const RESOURCE: &str = "https://api.openai.com/v1";
const DYNAMIC_CLIENT: &str = "dynamic_agent_client";
const SCOPES: &str =
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
/// The scope that lets requests use the ChatGPT plan.
pub const PLAN_SCOPE: &str = "chatgpt.tokens.use.direct";
const CALLBACK_PATH: &str = "/auth/callback";
/// The callback port tried first; another one when it is taken.
const PREFERRED_PORT: u16 = 1455;
/// The name the registration shows in ChatGPT's settings.
pub const APP_NAME: &str = "Setsuna IME";
/// A token this close to its expiry is refreshed first.
const REFRESH_MARGIN_S: i64 = 120;

pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn random_bytes<const N: usize>() -> Result<[u8; N]> {
    let mut b = [0u8; N];
    getrandom::getrandom(&mut b).map_err(|e| anyhow!("random: {e}"))?;
    Ok(b)
}

fn random_token() -> Result<String> {
    Ok(B64.encode(random_bytes::<32>()?))
}

/// A UUIDv4 host ID (`urn:uuid:...`).
fn new_host_id() -> Result<String> {
    let mut b = random_bytes::<16>()?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    Ok(format!(
        "urn:uuid:{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    ))
}

/// A sign-in waiting for the browser to come back to the loopback listener.
pub struct PendingSignIn {
    /// The authorization page to open in the system browser.
    pub url: String,
    listener: TcpListener,
    redirect_uri: String,
    state: String,
    nonce: String,
    verifier: String,
    /// Signing in again to this registration (`None`: a new one).
    client_id: Option<String>,
    expected_sub: Option<String>,
    store: Store,
    cancel: Arc<AtomicBool>,
}

/// The account a sign-in ended with.
#[derive(Debug, Clone)]
pub struct SignedIn {
    pub label: String,
    pub plan_usage: bool,
    /// The first sign-in with plan usage: show the plan notice once.
    pub show_plan_notice: bool,
}

/// Start signing in: to the registration `client_id` again, or (`None`) a
/// new one. The host ID is saved before the browser opens.
pub fn begin(store: &Store, client_id: Option<&str>) -> Result<PendingSignIn> {
    let (host_id, hints) = store.update(|c| {
        if c.host_id.is_none() {
            c.host_id = Some(new_host_id()?);
        }
        let hints = client_id.and_then(|id| {
            c.chatgpt
                .iter()
                .find(|a| a.client_id == id)
                .map(|a| (a.id_token.clone(), a.email.clone(), a.sub.clone()))
        });
        Ok((c.host_id.clone().unwrap_or_default(), hints))
    })?;
    if client_id.is_some() && hints.is_none() {
        bail!("no saved registration {}", client_id.unwrap_or_default());
    }
    let listener = TcpListener::bind(("127.0.0.1", PREFERRED_PORT))
        .or_else(|_| TcpListener::bind(("127.0.0.1", 0)))
        .context("open the sign-in callback listener")?;
    let port = listener.local_addr()?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}{CALLBACK_PATH}");
    let state = random_token()?;
    let nonce = random_token()?;
    let verifier = random_token()?;
    let challenge = B64.encode(Sha256::digest(verifier.as_bytes()));

    let mut q = url::form_urlencoded::Serializer::new(String::new());
    q.append_pair("response_type", "code")
        .append_pair("client_id", client_id.unwrap_or(DYNAMIC_CLIENT))
        .append_pair("redirect_uri", &redirect_uri)
        .append_pair("scope", SCOPES)
        .append_pair("state", &state)
        .append_pair("nonce", &nonce)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("resource", RESOURCE)
        .append_pair("ext_agent_host_id", &host_id);
    let mut expected_sub = None;
    match &hints {
        None => {
            q.append_pair("agent_name_hint", APP_NAME);
        }
        Some((id_token, email, sub)) => {
            if let Some(t) = id_token {
                q.append_pair("id_token_hint", t);
            }
            if let Some(e) = email {
                q.append_pair("login_hint", e);
            }
            expected_sub = Some(sub.clone());
        }
    }
    Ok(PendingSignIn {
        url: format!("{AUTHORIZE_URL}?{}", q.finish()),
        listener,
        redirect_uri,
        state,
        nonce,
        verifier,
        client_id: client_id.map(str::to_string),
        expected_sub,
        store: store.clone(),
        cancel: Arc::new(AtomicBool::new(false)),
    })
}

/// What came back on the callback.
struct Callback {
    code: String,
    client_id: Option<String>,
}

impl PendingSignIn {
    /// Set to give up waiting.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// Wait for the browser (up to `timeout`), then exchange the code,
    /// check the tokens and save the account as the active one.
    pub fn finish(self, agent: &ureq::Agent, timeout: Duration) -> Result<SignedIn> {
        let cb = self.wait_callback(timeout)?;
        let client_id = match (&self.client_id, cb.client_id) {
            (Some(saved), Some(got)) if *saved != got => {
                bail!("the callback named another client ({got}, not {saved})")
            }
            (Some(saved), _) => saved.clone(),
            (None, Some(got)) if !got.is_empty() && got != DYNAMIC_CLIENT => got,
            (None, _) => bail!("the registration did not finish (no issued client ID)"),
        };
        let tokens = post_token(
            agent,
            &[
                ("grant_type", "authorization_code"),
                ("client_id", &client_id),
                ("code", &cb.code),
                ("code_verifier", &self.verifier),
                ("redirect_uri", &self.redirect_uri),
                ("resource", RESOURCE),
            ],
        )?;
        let id_token = tokens
            .id_token
            .clone()
            .ok_or_else(|| anyhow!("no ID token in the token response"))?;
        let claims = verify_id_token(agent, &id_token, &client_id, Some(&self.nonce))?;
        if let Some(sub) = &self.expected_sub {
            if *sub != claims.sub {
                bail!("signed in to another account than the one chosen");
            }
        }
        let plan_usage = tokens.scope.split_whitespace().any(|s| s == PLAN_SCOPE);
        let label = claims
            .email
            .clone()
            .or_else(|| claims.name.clone())
            .unwrap_or_else(|| claims.sub.clone());
        let show_plan_notice = self.store.update(|c| {
            let account = ChatGptAccount {
                client_id: client_id.clone(),
                sub: claims.sub.clone(),
                email: claims.email.clone(),
                name: claims.name.clone(),
                id_token: Some(id_token.clone()),
                access_token: Some(tokens.access_token.clone()),
                refresh_token: tokens.refresh_token.clone(),
                expires_at: now() + tokens.expires_in.unwrap_or(3600),
                scope: tokens.scope.clone(),
                saved_at: now(),
            };
            match c.chatgpt.iter_mut().find(|a| a.client_id == client_id) {
                Some(a) => *a = account,
                None => c.chatgpt.push(account),
            }
            c.active = Some(client_id.clone());
            Ok(plan_usage && !c.plan_notice_shown)
        })?;
        Ok(SignedIn {
            label,
            plan_usage,
            show_plan_notice,
        })
    }

    fn wait_callback(&self, timeout: Duration) -> Result<Callback> {
        self.listener.set_nonblocking(true)?;
        let deadline = Instant::now() + timeout;
        loop {
            if self.cancel.load(Ordering::SeqCst) {
                bail!("sign-in cancelled");
            }
            if Instant::now() >= deadline {
                bail!("sign-in timed out");
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(cb) = self.handle_connection(stream)? {
                        return Ok(cb);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(e).context("sign-in callback"),
            }
        }
    }

    /// One request to the listener: `Some` for the callback, `None` for
    /// anything else (the browser asking for a favicon).
    fn handle_connection(&self, mut stream: TcpStream) -> Result<Option<Callback>> {
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 2048];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") && buf.len() < 16384 {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
        let head = String::from_utf8_lossy(&buf);
        let target = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("");
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        if path != CALLBACK_PATH {
            let _ = respond(&mut stream, 404, "Not found");
            return Ok(None);
        }
        let params: std::collections::HashMap<String, String> =
            url::form_urlencoded::parse(query.as_bytes())
                .into_owned()
                .collect();
        if params.get("state").map(String::as_str) != Some(self.state.as_str()) {
            let _ = respond(&mut stream, 400, PAGE_FAILED);
            bail!("the sign-in callback's state did not match");
        }
        if let Some(err) = params.get("error") {
            let _ = respond(&mut stream, 200, PAGE_FAILED);
            if err == "access_denied" {
                return Err(Coded::new("declined", "sign-in was declined").into());
            }
            let desc = params.get("error_description").cloned().unwrap_or_default();
            bail!("sign-in failed: {err} {desc}");
        }
        let Some(code) = params.get("code").cloned() else {
            let _ = respond(&mut stream, 400, PAGE_FAILED);
            bail!("the sign-in callback had no code");
        };
        let _ = respond(&mut stream, 200, PAGE_DONE);
        Ok(Some(Callback {
            code,
            client_id: params.get("client_id").cloned(),
        }))
    }
}

const PAGE_DONE: &str = "<!doctype html><meta charset=utf-8><title>Setsuna IME</title>\
<body style=\"font-family:system-ui,sans-serif;margin:3em\"><h1>サインインしました</h1>\
<p>このウィンドウを閉じて Setsuna IME の設定画面に戻ってください。</p>";
const PAGE_FAILED: &str = "<!doctype html><meta charset=utf-8><title>Setsuna IME</title>\
<body style=\"font-family:system-ui,sans-serif;margin:3em\"><h1>サインインできませんでした</h1>\
<p>このウィンドウを閉じて、設定画面からもう一度お試しください。</p>";

fn respond(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    scope: String,
}

#[derive(Deserialize, Default)]
struct TokenError {
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

/// Refresh errors after which the refresh token is no use.
const TERMINAL_REFRESH_ERRORS: &[&str] = &[
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
];

fn post_token(agent: &ureq::Agent, form: &[(&str, &str)]) -> Result<TokenResponse> {
    let resp = agent
        .post(TOKEN_URL)
        .send_form(form.iter().copied())
        .map_err(|e| Coded::new("network", format!("token request: {e}")))?;
    let status = resp.status().as_u16();
    let body = resp
        .into_body()
        .read_to_string()
        .context("read the token response")?;
    if status != 200 {
        let err: TokenError = serde_json::from_str(&body).unwrap_or_default();
        let code = if TERMINAL_REFRESH_ERRORS.contains(&err.error.as_str()) {
            "signed_out"
        } else {
            "auth"
        };
        return Err(Coded::new(
            code,
            format!(
                "token endpoint {status}: {} {}",
                err.error, err.error_description
            ),
        )
        .into());
    }
    serde_json::from_str(&body).context("parse the token response")
}

/// A valid access token of the active account, refreshed first when it is
/// about to expire (under the store's lock, so processes do not race the
/// rotating refresh token). A refresh token that no longer works signs the
/// account out.
pub fn access_token(store: &Store, agent: &ureq::Agent) -> Result<String> {
    store.update(|c| {
        let account = c
            .active_account_mut()
            .ok_or_else(|| Coded::new("not_signed_in", "not signed in to ChatGPT"))?;
        if let Some(token) = &account.access_token {
            if account.expires_at - REFRESH_MARGIN_S > now() {
                return Ok(token.clone());
            }
        }
        let Some(refresh) = account.refresh_token.clone() else {
            return Err(Coded::new("not_signed_in", "not signed in to ChatGPT").into());
        };
        let client_id = account.client_id.clone();
        match post_token(
            agent,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", &client_id),
                ("refresh_token", &refresh),
                ("resource", RESOURCE),
            ],
        ) {
            Ok(t) => {
                account.access_token = Some(t.access_token.clone());
                if t.refresh_token.is_some() {
                    account.refresh_token = t.refresh_token;
                }
                if t.id_token.is_some() {
                    account.id_token = t.id_token;
                }
                account.expires_at = now() + t.expires_in.unwrap_or(3600);
                if !t.scope.is_empty() {
                    account.scope = t.scope;
                }
                account.saved_at = now();
                Ok(t.access_token)
            }
            Err(e) => {
                if Coded::code_of(&e) == Some("signed_out") {
                    account.access_token = None;
                    account.refresh_token = None;
                }
                Err(e)
            }
        }
    })
}

/// Sign the active account out: revoke the refresh token, then forget the
/// tokens (keeping the registration for the next sign-in). `Ok(false)`:
/// the revocation was not confirmed (the user can disconnect the app in
/// ChatGPT's settings).
pub fn sign_out(store: &Store, agent: &ureq::Agent) -> Result<bool> {
    let (client_id, refresh) = {
        let c = store.load()?;
        match c.active_account() {
            Some(a) => (a.client_id.clone(), a.refresh_token.clone()),
            None => return Ok(true),
        }
    };
    let mut confirmed = refresh.is_none();
    if let Some(token) = refresh {
        for attempt in 0..3 {
            let r = agent.post(REVOKE_URL).send_form([
                ("token", token.as_str()),
                ("token_type_hint", "refresh_token"),
                ("client_id", client_id.as_str()),
            ]);
            match r {
                Ok(resp) if resp.status().as_u16() == 200 => {
                    confirmed = true;
                    break;
                }
                Ok(resp) if resp.status().as_u16() < 500 => break,
                _ => std::thread::sleep(Duration::from_millis(500 << attempt)),
            }
        }
    }
    store.update(|c| {
        if let Some(a) = c.chatgpt.iter_mut().find(|a| a.client_id == client_id) {
            a.access_token = None;
            a.refresh_token = None;
            a.id_token = None;
        }
        Ok(())
    })?;
    Ok(confirmed)
}

/// The claims of a checked ID token this module uses.
#[derive(Debug, Deserialize)]
pub struct IdClaims {
    pub sub: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Deserialize)]
struct Header {
    alg: String,
    #[serde(default)]
    kid: Option<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct Jwk {
    #[serde(default)]
    kid: Option<String>,
    kty: String,
    #[serde(default)]
    n: String,
    #[serde(default)]
    e: String,
}

/// Check an ID token: RS256 signature by a key of OpenAI's JWKS, issuer,
/// audience (the issued client ID), expiry and nonce.
fn verify_id_token(
    agent: &ureq::Agent,
    token: &str,
    client_id: &str,
    nonce: Option<&str>,
) -> Result<IdClaims> {
    let jwks: Jwks = {
        let resp = agent
            .get(JWKS_URL)
            .call()
            .map_err(|e| Coded::new("network", format!("JWKS: {e}")))?;
        serde_json::from_str(&resp.into_body().read_to_string()?).context("parse the JWKS")?
    };
    verify_with(&jwks, token, client_id, nonce, now())
}

fn verify_with(
    jwks: &Jwks,
    token: &str,
    client_id: &str,
    nonce: Option<&str>,
    now: i64,
) -> Result<IdClaims> {
    use rsa::pkcs1v15::{Signature, VerifyingKey};
    use rsa::signature::Verifier;
    use rsa::{BigUint, RsaPublicKey};

    let mut parts = token.split('.');
    let (Some(h), Some(p), Some(s), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        bail!("the ID token is not a JWS");
    };
    let header: Header = serde_json::from_slice(&B64.decode(h)?).context("ID token header")?;
    if header.alg != "RS256" {
        bail!("ID token algorithm {} is not RS256", header.alg);
    }
    let jwk = jwks
        .keys
        .iter()
        .find(|k| k.kty == "RSA" && (header.kid.is_none() || k.kid == header.kid))
        .ok_or_else(|| anyhow!("no JWKS key for the ID token"))?;
    let key = RsaPublicKey::new(
        BigUint::from_bytes_be(&B64.decode(&jwk.n)?),
        BigUint::from_bytes_be(&B64.decode(&jwk.e)?),
    )?;
    let signature = Signature::try_from(B64.decode(s)?.as_slice())?;
    VerifyingKey::<Sha256>::new(key)
        .verify(format!("{h}.{p}").as_bytes(), &signature)
        .map_err(|_| anyhow!("the ID token's signature does not verify"))?;

    let claims: serde_json::Value =
        serde_json::from_slice(&B64.decode(p)?).context("ID token claims")?;
    if claims["iss"].as_str() != Some(ISSUER) {
        bail!("ID token issuer {:?}", claims["iss"]);
    }
    let aud_ok = match &claims["aud"] {
        serde_json::Value::String(a) => a == client_id,
        serde_json::Value::Array(a) => a.iter().any(|x| x.as_str() == Some(client_id)),
        _ => false,
    };
    if !aud_ok {
        bail!("ID token audience {:?} is not {client_id}", claims["aud"]);
    }
    if claims["exp"].as_i64().unwrap_or(0) + 60 < now {
        bail!("the ID token expired");
    }
    if let Some(n) = nonce {
        if claims["nonce"].as_str() != Some(n) {
            bail!("the ID token's nonce does not match");
        }
    }
    Ok(serde_json::from_value(claims)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_ids_are_uuid_v4_urns() {
        let id = new_host_id().unwrap();
        assert!(id.starts_with("urn:uuid:"), "{id}");
        let uuid = &id["urn:uuid:".len()..];
        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[14..15], "4");
        assert!(matches!(&uuid[19..20], "8" | "9" | "a" | "b"));
        assert_ne!(id, new_host_id().unwrap());
    }

    #[test]
    fn the_authorization_url_registers_a_new_client_with_pkce() {
        let dir = std::env::temp_dir().join(format!("setsuna-oauth-begin-{}", std::process::id()));
        let store = Store::at(&dir);
        let pending = begin(&store, None).unwrap();
        let (base, query) = pending.url.split_once('?').unwrap();
        assert_eq!(base, AUTHORIZE_URL);
        let q: std::collections::HashMap<String, String> =
            url::form_urlencoded::parse(query.as_bytes())
                .into_owned()
                .collect();
        assert_eq!(q["client_id"], DYNAMIC_CLIENT);
        assert_eq!(q["agent_name_hint"], APP_NAME);
        assert_eq!(q["code_challenge_method"], "S256");
        assert_eq!(
            q["code_challenge"],
            B64.encode(Sha256::digest(pending.verifier.as_bytes()))
        );
        assert_eq!(q["resource"], RESOURCE);
        assert!(q["scope"].split(' ').any(|s| s == PLAN_SCOPE));
        assert!(q["redirect_uri"].starts_with("http://127.0.0.1:"));
        assert!(q["redirect_uri"].ends_with(CALLBACK_PATH));
        // the host ID is saved before the browser opens
        assert_eq!(
            Some(q["ext_agent_host_id"].clone()),
            store.load().unwrap().host_id
        );
        drop(pending);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_token_that_is_not_signed_by_the_jwks_is_refused() {
        let header = B64.encode(br#"{"alg":"RS256","kid":"k"}"#);
        let claims = B64.encode(
            br#"{"iss":"https://auth.openai.com","aud":"oaiapp_1","exp":9999999999,"sub":"u","nonce":"n"}"#,
        );
        let token = format!("{header}.{claims}.{}", B64.encode([0u8; 256]));
        let jwks = Jwks {
            keys: vec![Jwk {
                kid: Some("k".into()),
                kty: "RSA".into(),
                // 2048-bit modulus of all 0xff bytes is not a real key, but
                // it parses: the signature check is what refuses the token
                n: B64.encode([0xffu8; 256]),
                e: B64.encode([1u8, 0, 1]),
            }],
        };
        let err = verify_with(&jwks, &token, "oaiapp_1", Some("n"), 0).unwrap_err();
        assert!(err.to_string().contains("signature"), "{err}");
        let none = B64.encode(br#"{"alg":"none"}"#);
        let err =
            verify_with(&jwks, &format!("{none}.{claims}."), "oaiapp_1", None, 0).unwrap_err();
        assert!(err.to_string().contains("RS256"), "{err}");
    }
}
