//! The rewriter's half of the settings page: signing in with ChatGPT,
//! keeping API keys and listing models, through `setsuna_rewrite`. What
//! needs the network runs on a thread and answers through the event loop.
//!
//! Page -> here: `rewriter_state`, `sign_in` (`new_account`: a new
//! registration), `cancel_sign_in`, `sign_out`, `set_api_key`,
//! `list_models`, `ack_plan_notice`, `open_url`; the endpoint ones carry
//! the page's `provider`, `base_url` and `api`.
//! Here -> page: `{"type": "rewriter", ...}` and `{"type": "models", ...}`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use setsuna_rewrite::api::{self, Api, Endpoint, Provider};
use setsuna_rewrite::oauth;
use setsuna_rewrite::store::{api_key_slot, Store};
use tao::event_loop::EventLoopProxy;

use crate::UserEvent;

/// How long a sign-in waits for the browser.
const SIGN_IN_TIMEOUT: Duration = Duration::from_secs(300);
/// Pages the page may open in the system browser.
const OPEN_URLS: &[&str] = &[
    "https://chatgpt.com/settings",
    "https://help.openai.com/",
    "https://platform.openai.com/",
];

pub struct Rewriter {
    store: Option<Store>,
    proxy: EventLoopProxy<UserEvent>,
    /// Set to give up the sign-in waiting for the browser.
    signing_in: Option<Arc<AtomicBool>>,
    /// The endpoint the page last named.
    last: Endpoint,
}

impl Rewriter {
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        Self {
            store: setsuna_core::config::config_dir().map(Store::at),
            proxy,
            signing_in: None,
            last: endpoint_of(&Value::Null),
        }
    }

    /// The reply to `action` from the page, if it has one now.
    pub fn handle(&mut self, action: &str, msg: &Value) -> Option<Value> {
        let Some(store) = self.store.clone() else {
            return Some(json!({"type": "rewriter", "status": "設定ディレクトリが見つかりません"}));
        };
        let endpoint = endpoint_of(msg);
        self.last = endpoint.clone();
        let status = match action {
            "rewriter_state" => None,
            "sign_in" => Some(self.sign_in(&store, msg["new_account"].as_bool() == Some(true))),
            "cancel_sign_in" => {
                if let Some(flag) = self.signing_in.take() {
                    flag.store(true, Ordering::SeqCst);
                }
                Some("サインインを中止しました".to_string())
            }
            "sign_out" => {
                let proxy = self.proxy.clone();
                std::thread::spawn(move || {
                    let agent = api::agent(Duration::from_secs(15));
                    let status = match oauth::sign_out(&store, &agent) {
                        Ok(true) => "サインアウトしました".to_string(),
                        Ok(false) => "サインアウトしました（取り消しを確認できなかったため、ChatGPT の設定からもアプリの接続を解除してください）".to_string(),
                        Err(e) => format!("サインアウトできませんでした: {e:#}"),
                    };
                    let _ = proxy.send_event(UserEvent::Reply(json!({
                        "type": "rewriter_refresh", "status": status,
                    })));
                });
                return None;
            }
            "set_api_key" => {
                let key = msg["key"].as_str().unwrap_or_default().trim().to_string();
                let slot = api_key_slot(&endpoint.base_url);
                let r = store.update(|c| {
                    if key.is_empty() {
                        c.api_keys.remove(&slot);
                    } else {
                        c.api_keys.insert(slot.clone(), key.clone());
                    }
                    Ok(())
                });
                Some(match r {
                    Ok(()) if key.is_empty() => "API キーを削除しました".to_string(),
                    Ok(()) => "API キーを保存しました".to_string(),
                    Err(e) => format!("API キーを保存できませんでした: {e:#}"),
                })
            }
            "ack_plan_notice" => {
                let _ = store.update(|c| {
                    c.plan_notice_shown = true;
                    Ok(())
                });
                None
            }
            "list_models" => {
                let proxy = self.proxy.clone();
                std::thread::spawn(move || {
                    let agent = api::agent(Duration::from_secs(15));
                    let reply = match api::list_models(&endpoint, &store, &agent) {
                        Ok(models) => json!({"type": "models", "models": models}),
                        Err(e) => json!({
                            "type": "models",
                            "error": format!("モデルの一覧を取得できませんでした: {e:#}"),
                        }),
                    };
                    let _ = proxy.send_event(UserEvent::Reply(reply));
                });
                return None;
            }
            "open_url" => {
                let url = msg["url"].as_str().unwrap_or_default();
                if OPEN_URLS.iter().any(|p| url.starts_with(p)) {
                    let _ = setsuna_rewrite::open_browser(url);
                }
                return None;
            }
            _ => return None,
        };
        Some(self.state(&store, &endpoint, status))
    }

    /// The account and key state for the page's endpoint.
    pub fn state(&self, store: &Store, endpoint: &Endpoint, status: Option<String>) -> Value {
        let creds = store.load().unwrap_or_default();
        let account = creds.active_account().map(|a| {
            json!({
                "label": a.label(),
                "signed_in": a.signed_in(),
                "plan_usage": a.plan_usage(),
            })
        });
        json!({
            "type": "rewriter",
            "account": account,
            "api_key_set": creds.api_keys.contains_key(&api_key_slot(&endpoint.base_url)),
            "signing_in": self.signing_in.is_some(),
            "status": status,
        })
    }

    /// A thread's `rewriter_refresh`: the state as it is now, with the
    /// thread's status (and, after a sign-in, whether to show the notice).
    pub fn refreshed(&mut self, done: &Value) -> Value {
        if done.get("signed_in").is_some() {
            self.signing_in = None;
        }
        let status = done["status"].as_str().map(str::to_string);
        let Some(store) = self.store.clone() else {
            return json!({"type": "rewriter", "status": status});
        };
        let mut state = self.state(&store, &self.last.clone(), status);
        state["show_plan_notice"] = done["show_plan_notice"].clone();
        state
    }

    /// Open the authorization page and wait for the browser on a thread.
    fn sign_in(&mut self, store: &Store, new_account: bool) -> String {
        if let Some(flag) = self.signing_in.take() {
            flag.store(true, Ordering::SeqCst);
        }
        let client_id = if new_account {
            None
        } else {
            store
                .load()
                .ok()
                .and_then(|c| c.active_account().map(|a| a.client_id.clone()))
        };
        let pending = match oauth::begin(store, client_id.as_deref()) {
            Ok(p) => p,
            Err(e) => return format!("サインインを開始できませんでした: {e:#}"),
        };
        if let Err(e) = setsuna_rewrite::open_browser(&pending.url) {
            return format!("ブラウザを開けませんでした: {e}");
        }
        self.signing_in = Some(pending.cancel_flag());
        let proxy = self.proxy.clone();
        std::thread::spawn(move || {
            let agent = api::agent(Duration::from_secs(30));
            let reply = match pending.finish(&agent, SIGN_IN_TIMEOUT) {
                Ok(s) => json!({
                    "type": "rewriter_refresh",
                    "signed_in": true,
                    "show_plan_notice": s.show_plan_notice,
                    "status": if s.plan_usage {
                        format!("{} でサインインしました", s.label)
                    } else {
                        format!("{} でサインインしましたが、ChatGPT プランの利用は許可されていません", s.label)
                    },
                }),
                Err(e) => json!({
                    "type": "rewriter_refresh",
                    "signed_in": false,
                    "status": match setsuna_rewrite::Coded::code_of(&e) {
                        Some("declined") => "サインインが拒否されました".to_string(),
                        _ => format!("サインインできませんでした: {e:#}"),
                    },
                }),
            };
            let _ = proxy.send_event(UserEvent::Reply(reply));
        });
        "ブラウザでサインインしてください".to_string()
    }
}

/// The endpoint the page has chosen (saved or not).
pub fn endpoint_of(msg: &Value) -> Endpoint {
    let provider = msg["provider"]
        .as_str()
        .and_then(Provider::parse)
        .unwrap_or(Provider::ChatGpt);
    let base_url = msg["base_url"].as_str().filter(|s| !s.trim().is_empty());
    let api = msg["api"].as_str().and_then(Api::parse);
    Endpoint::new(provider, base_url, api)
}
