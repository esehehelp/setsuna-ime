//! The JSONL protocol of `setsuna-rerank daemon` (stdio) and `serve` (a
//! named pipe), one JSON object per line:
//!   ready:    {"ready":true,"version":"...","threads":16,"max_context_chars":40,
//!              "max_candidates":20,"lambda":0.3}
//!   request:  {"id":1,"context":"...","candidates":["...","..."]}
//!   response: {"id":1,"logp":[-12.3,-15.1],"score_ms":42.0,"cached":23,"decoded":40}
//!             or {"id":1,"error":"..."}

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{RerankConfig, Reranker};

#[derive(Deserialize)]
pub struct Request {
    pub id: u64,
    #[serde(default)]
    pub context: String,
    pub candidates: Vec<String>,
}

#[derive(Serialize)]
pub struct Response {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logp: Option<Vec<f64>>,
    pub score_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoded: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    pub fn error(id: u64, error: String) -> Self {
        Self {
            id,
            logp: None,
            score_ms: 0.0,
            cached: None,
            decoded: None,
            error: Some(error),
        }
    }
}

/// The line a reranker sends first, once its model is up.
pub fn ready(config: &RerankConfig) -> serde_json::Value {
    let mut ready = serde_json::to_value(config).expect("plain fields");
    ready["ready"] = true.into();
    ready["version"] = env!("CARGO_PKG_VERSION").into();
    ready
}

/// The response to one request line.
pub fn handle(reranker: &mut Reranker, line: &str) -> Response {
    match serde_json::from_str::<Request>(line) {
        Ok(req) => respond(reranker, req.id, &req.context, &req.candidates),
        Err(e) => Response::error(0, format!("invalid request json: {e}")),
    }
}

pub fn respond(reranker: &mut Reranker, id: u64, context: &str, candidates: &[String]) -> Response {
    let t0 = Instant::now();
    let result = reranker.score(context, candidates);
    let score_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok((logp, stats)) => Response {
            id,
            logp: Some(logp),
            score_ms,
            cached: Some(stats.cached),
            decoded: Some(stats.decoded),
            error: None,
        },
        Err(e) => Response {
            score_ms,
            ..Response::error(id, format!("{e:#}"))
        },
    }
}
