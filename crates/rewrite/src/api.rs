//! The model server: OpenAI with a ChatGPT sign-in or an API key, or an
//! OpenAI-compatible server. Models come from `GET {base}/models`; a
//! request goes to the Responses API (`POST {base}/responses`, streamed,
//! nothing stored, as plan usage requires) or to Chat Completions
//! (`POST {base}/chat/completions`), which most compatible servers speak.

use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::store::{api_key_slot, Store};
use crate::Coded;

pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Who pays for the requests and how they authenticate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// The ChatGPT plan, through Sign in with ChatGPT.
    ChatGpt,
    /// OpenAI with an API key.
    OpenAi,
    /// An OpenAI-compatible server; the API key is optional.
    Compatible,
}

impl Provider {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "chatgpt" => Some(Self::ChatGpt),
            "openai" => Some(Self::OpenAi),
            "compatible" => Some(Self::Compatible),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Responses,
    ChatCompletions,
}

impl Api {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "responses" => Some(Self::Responses),
            "chat_completions" => Some(Self::ChatCompletions),
            _ => None,
        }
    }
}

/// Where requests go and how.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub provider: Provider,
    pub base_url: String,
    pub api: Api,
}

impl Endpoint {
    /// `base_url` and `api` default by provider: OpenAI's Responses API,
    /// Chat Completions for a compatible server. Plan usage is only for
    /// OpenAI's Responses API.
    pub fn new(provider: Provider, base_url: Option<&str>, api: Option<Api>) -> Self {
        let base_url = match (provider, base_url) {
            (Provider::ChatGpt, _) | (_, None) => OPENAI_BASE_URL.to_string(),
            (_, Some(u)) => u.trim().trim_end_matches('/').to_string(),
        };
        let api = match (provider, api) {
            (Provider::ChatGpt, _) => Api::Responses,
            (_, Some(a)) => a,
            (Provider::OpenAi, None) => Api::Responses,
            (Provider::Compatible, None) => Api::ChatCompletions,
        };
        Self {
            provider,
            base_url,
            api,
        }
    }

    /// The bearer token for a request: a fresh ChatGPT access token, the
    /// saved API key, or none (a compatible server without one).
    pub fn bearer(&self, store: &Store, agent: &ureq::Agent) -> Result<Option<String>> {
        match self.provider {
            Provider::ChatGpt => {
                let creds = store.load()?;
                if let Some(a) = creds.active_account() {
                    if a.signed_in() && !a.plan_usage() {
                        return Err(Coded::new(
                            "no_plan_usage",
                            "the sign-in did not allow ChatGPT plan usage",
                        )
                        .into());
                    }
                }
                crate::oauth::access_token(store, agent).map(Some)
            }
            Provider::OpenAi | Provider::Compatible => {
                let key = store
                    .load()?
                    .api_keys
                    .get(&api_key_slot(&self.base_url))
                    .cloned()
                    .filter(|k| !k.is_empty());
                if key.is_none() && self.provider == Provider::OpenAi {
                    return Err(Coded::new("no_api_key", "no OpenAI API key is saved").into());
                }
                Ok(key)
            }
        }
    }
}

/// An HTTP agent: no error for an error status (its body says why), the
/// whole request bounded by `timeout`.
pub fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .user_agent(concat!("setsuna-ime/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

/// A model to choose from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Model {
    /// What requests send as `model`.
    pub id: String,
    /// What the picker shows.
    pub label: String,
}

/// The models the endpoint offers this account or key.
pub fn list_models(endpoint: &Endpoint, store: &Store, agent: &ureq::Agent) -> Result<Vec<Model>> {
    let bearer = endpoint.bearer(store, agent)?;
    let mut req = agent.get(format!("{}/models", endpoint.base_url));
    if let Some(b) = &bearer {
        req = req.header("Authorization", format!("Bearer {b}"));
    }
    let resp = req
        .call()
        .map_err(|e| Coded::new("network", format!("models: {e}")))?;
    let status = resp.status().as_u16();
    let body = resp.into_body().read_to_string()?;
    if status != 200 {
        return Err(http_error(status, &body));
    }
    let v: Value = serde_json::from_str(&body).context("parse the model list")?;
    Ok(parse_models(&v))
}

/// The model list: OpenAI's `data` of `{id}`, or the account's catalog of
/// `{slug, display_name, visibility}` of which only `"list"` ones are shown.
fn parse_models(v: &Value) -> Vec<Model> {
    let items = v["data"]
        .as_array()
        .or_else(|| v["models"].as_array())
        .cloned()
        .unwrap_or_default();
    let mut models: Vec<Model> = items
        .iter()
        .filter(|m| m["visibility"].as_str().is_none_or(|vis| vis == "list"))
        .filter_map(|m| {
            let id = m["slug"].as_str().or_else(|| m["id"].as_str())?.to_string();
            let label = m["display_name"].as_str().unwrap_or(&id).to_string();
            Some(Model { id, label })
        })
        .collect();
    if v["models"].is_null() {
        // an API's list is in no order
        models.sort_by(|a, b| a.id.cmp(&b.id));
    }
    models.dedup_by(|a, b| a.id == b.id);
    models
}

/// One request: `instructions` as the system / developer text, `input` as
/// the user's message. Returns the model's text.
pub struct Request<'a> {
    pub model: &'a str,
    pub instructions: &'a str,
    pub input: &'a str,
    /// `reasoning.effort` for the Responses API, when set.
    pub reasoning_effort: Option<&'a str>,
}

pub fn complete(
    endpoint: &Endpoint,
    store: &Store,
    agent: &ureq::Agent,
    req: &Request,
) -> Result<String> {
    if req.model.is_empty() {
        return Err(Coded::new("no_model", "no model is chosen").into());
    }
    let bearer = endpoint.bearer(store, agent)?;
    match endpoint.api {
        Api::Responses => responses(endpoint, bearer.as_deref(), agent, req),
        Api::ChatCompletions => chat_completions(endpoint, bearer.as_deref(), agent, req),
    }
}

fn post_json(
    agent: &ureq::Agent,
    url: &str,
    bearer: Option<&str>,
    body: &Value,
) -> Result<ureq::http::Response<ureq::Body>> {
    let mut r = agent.post(url).header("Content-Type", "application/json");
    if let Some(b) = bearer {
        r = r.header("Authorization", format!("Bearer {b}"));
    }
    r.send(body.to_string())
        .map_err(|e| Coded::new("network", format!("{url}: {e}")).into())
}

fn responses(
    endpoint: &Endpoint,
    bearer: Option<&str>,
    agent: &ureq::Agent,
    req: &Request,
) -> Result<String> {
    let mut body = json!({
        "model": req.model,
        "instructions": req.instructions,
        "input": [{"role": "user", "content": req.input}],
        "store": false,
        "stream": true,
    });
    if let Some(effort) = req.reasoning_effort {
        body["reasoning"] = json!({ "effort": effort });
    }
    let resp = post_json(
        agent,
        &format!("{}/responses", endpoint.base_url),
        bearer,
        &body,
    )?;
    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp.into_body().read_to_string().unwrap_or_default();
        return Err(http_error(status, &body));
    }
    read_response_stream(BufReader::new(resp.into_body().into_reader()))
}

/// The text of a Responses stream, which counts only once it ends in
/// `response.completed`.
fn read_response_stream(reader: impl BufRead) -> Result<String> {
    let mut text = String::new();
    for line in reader.lines() {
        let line = line.map_err(|e| Coded::new("network", format!("stream: {e}")))?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match ev["type"].as_str().unwrap_or_default() {
            "response.output_text.delta" => text.push_str(ev["delta"].as_str().unwrap_or_default()),
            "response.completed" => {
                if text.is_empty() {
                    text = output_text(&ev["response"]);
                }
                return Ok(text);
            }
            "response.failed" => return Err(stream_error(&ev["response"]["error"])),
            "response.incomplete" => {
                let why = ev["response"]["incomplete_details"]["reason"]
                    .as_str()
                    .unwrap_or("unknown");
                return Err(
                    Coded::new("incomplete", format!("the response was cut off ({why})")).into(),
                );
            }
            // the code is under `error` or, with some servers, on the event
            "error" => {
                return Err(coded_error(&ev["error"])
                    .or_else(|| coded_error(&ev))
                    .unwrap_or_else(|| stream_error(&ev)))
            }
            _ => {}
        }
    }
    Err(Coded::new(
        "incomplete",
        "the stream ended before the response completed",
    )
    .into())
}

/// The output text of a whole response object.
fn output_text(response: &Value) -> String {
    response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter(|c| c["type"] == "output_text")
        .filter_map(|c| c["text"].as_str())
        .collect()
}

fn chat_completions(
    endpoint: &Endpoint,
    bearer: Option<&str>,
    agent: &ureq::Agent,
    req: &Request,
) -> Result<String> {
    let body = json!({
        "model": req.model,
        "messages": [
            {"role": "system", "content": req.instructions},
            {"role": "user", "content": req.input},
        ],
        "stream": false,
    });
    let resp = post_json(
        agent,
        &format!("{}/chat/completions", endpoint.base_url),
        bearer,
        &body,
    )?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().unwrap_or_default();
    if status != 200 {
        return Err(http_error(status, &text));
    }
    let v: Value = serde_json::from_str(&text).context("parse the completion")?;
    v["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Coded::new("bad_response", "the completion has no text").into())
}

/// An error status before any stream: the body's `error.code` when it has
/// one, else the status.
fn http_error(status: u16, body: &str) -> anyhow::Error {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let err = &v["error"];
    if err.is_object() {
        if let Some(e) = coded_error(err) {
            return e;
        }
    }
    let detail = err["message"]
        .as_str()
        .or_else(|| v["detail"].as_str())
        .unwrap_or_default();
    let code = match status {
        401 => "auth",
        403 => "forbidden",
        404 => "not_found",
        429 => "usage_limit",
        500..=599 => "unavailable",
        _ => "http",
    };
    Coded::new(code, format!("HTTP {status} {detail}")).into()
}

/// The plan-usage error codes OpenAI documents, as this crate's codes.
fn coded_error(err: &Value) -> Option<anyhow::Error> {
    let code = err["code"].as_str()?;
    let message = err["message"].as_str().unwrap_or(code);
    let ours = match code {
        "subscription_sharing_usage_limit_exceeded"
        | "rate_limit_exceeded"
        | "insufficient_quota" => "usage_limit",
        "subscription_sharing_user_not_eligible" => "not_eligible",
        "subscription_sharing_usage_unavailable" | "subscription_sharing_user_unavailable" => {
            "unavailable"
        }
        "subscription_sharing_unsupported_capability" => "unsupported",
        "subscription_sharing_invalid_user" | "invalid_api_key" => "auth",
        "model_not_found" => "not_found",
        _ => "http",
    };
    let param = err["param"]
        .as_str()
        .map(|p| format!(" ({p})"))
        .unwrap_or_default();
    Some(Coded::new(ours, format!("{code}: {message}{param}")).into())
}

fn stream_error(err: &Value) -> anyhow::Error {
    coded_error(err)
        .unwrap_or_else(|| Coded::new("http", format!("the response failed: {err}")).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_default_by_provider() {
        let e = Endpoint::new(
            Provider::ChatGpt,
            Some("http://x"),
            Some(Api::ChatCompletions),
        );
        assert_eq!(
            (e.base_url.as_str(), e.api),
            (OPENAI_BASE_URL, Api::Responses)
        );
        let e = Endpoint::new(Provider::OpenAi, None, None);
        assert_eq!(
            (e.base_url.as_str(), e.api),
            (OPENAI_BASE_URL, Api::Responses)
        );
        let e = Endpoint::new(
            Provider::Compatible,
            Some(" http://127.0.0.1:8080/v1/ "),
            None,
        );
        assert_eq!(
            (e.base_url.as_str(), e.api),
            ("http://127.0.0.1:8080/v1", Api::ChatCompletions)
        );
    }

    #[test]
    fn the_account_catalog_shows_only_listed_models() {
        let v = json!({"models": [
            {"slug": "gpt-a", "display_name": "GPT A", "visibility": "list"},
            {"slug": "gpt-hidden", "display_name": "Hidden", "visibility": "hide"},
            {"slug": "gpt-b", "display_name": "GPT B", "visibility": "list"},
        ]});
        assert_eq!(
            parse_models(&v),
            [
                Model {
                    id: "gpt-a".into(),
                    label: "GPT A".into()
                },
                Model {
                    id: "gpt-b".into(),
                    label: "GPT B".into()
                },
            ]
        );
        let v = json!({"object": "list", "data": [{"id": "z"}, {"id": "a"}]});
        let ids: Vec<_> = parse_models(&v).into_iter().map(|m| m.id).collect();
        assert_eq!(ids, ["a", "z"]);
    }

    #[test]
    fn a_stream_counts_once_it_completes() {
        let ok = "event: response.output_text.delta\n\
                  data: {\"type\":\"response.output_text.delta\",\"delta\":\"明日\"}\n\n\
                  data: {\"type\":\"response.output_text.delta\",\"delta\":\"行きます\"}\n\n\
                  data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert_eq!(read_response_stream(ok.as_bytes()).unwrap(), "明日行きます");
        let whole = "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"全部\"}]}]}}\n";
        assert_eq!(read_response_stream(whole.as_bytes()).unwrap(), "全部");
        let cut = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"途中\"}\n";
        let err = read_response_stream(cut.as_bytes()).unwrap_err();
        assert_eq!(Coded::code_of(&err), Some("incomplete"));
        let limit = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"subscription_sharing_usage_limit_exceeded\",\"message\":\"limit\"}}}\n";
        let err = read_response_stream(limit.as_bytes()).unwrap_err();
        assert_eq!(Coded::code_of(&err), Some("usage_limit"));
    }

    #[test]
    fn error_statuses_keep_the_documented_codes() {
        let err = http_error(
            400,
            r#"{"error":{"code":"subscription_sharing_unsupported_capability","message":"no","param":"reasoning"}}"#,
        );
        assert_eq!(Coded::code_of(&err), Some("unsupported"));
        assert!(err.to_string().contains("reasoning"));
        assert_eq!(
            Coded::code_of(&http_error(401, r#"{"detail":"x"}"#)),
            Some("auth")
        );
        assert_eq!(Coded::code_of(&http_error(503, "")), Some("unavailable"));
    }
}
