//! Drives the rewriter from JSONL stdio (daemon mode), or one request or
//! the model list on the command line.
//!
//! Daemon protocol (one JSON object per line):
//!   ready:    {"ready":true,"version":"...","model":"..."}
//!   request:  {"id":1,"context":"...","reading":"...","text":"..."}
//!   response: {"id":1,"text":"...","ms":812.0}
//!             or {"id":1,"error":"...","code":"usage_limit","ms":120.0}
//! Requests are answered one at a time, in order.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use setsuna_rewrite::api::{self, Api, Endpoint, Provider};
use setsuna_rewrite::store::Store;
use setsuna_rewrite::Coded;

#[derive(Parser, Debug)]
#[command(
    name = "setsuna-rewrite",
    version,
    about = "Setsuna rewriter: an OpenAI(-compatible) model rewriting the conversion into natural Japanese."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Rewrite one input and print the result as JSON, then exit.
    Rewrite {
        #[command(flatten)]
        args: Args,
        #[arg(long, default_value = "")]
        context: String,
        #[arg(long)]
        reading: String,
        /// The conversion shown for the reading.
        #[arg(long)]
        text: String,
    },
    /// Print the models the endpoint offers, one JSON object per line.
    Models {
        #[command(flatten)]
        args: Args,
    },
    /// Read JSONL requests from stdin, write JSONL responses to stdout.
    Daemon {
        #[command(flatten)]
        args: Args,
    },
}

#[derive(Parser, Debug)]
struct Args {
    /// chatgpt (the ChatGPT plan, signed in), openai (an API key) or
    /// compatible (an OpenAI-compatible server).
    #[arg(long, default_value = "chatgpt")]
    provider: String,
    /// The API's base URL (openai / compatible). Default: OpenAI's.
    #[arg(long)]
    base_url: Option<String>,
    /// responses or chat_completions. Default: responses for OpenAI,
    /// chat_completions for a compatible server.
    #[arg(long)]
    api: Option<String>,
    #[arg(long, default_value = "")]
    model: String,
    /// The model's instructions. Default: the built-in ones.
    #[arg(long)]
    instructions: Option<String>,
    /// `reasoning.effort` (Responses API), e.g. low.
    #[arg(long)]
    reasoning_effort: Option<String>,
    /// Longest a request may take, in ms.
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,
    /// The directory holding the credentials (the config directory).
    #[arg(long)]
    credentials_dir: PathBuf,
}

struct Rewriter {
    endpoint: Endpoint,
    store: Store,
    agent: ureq::Agent,
    model: String,
    instructions: String,
    reasoning_effort: Option<String>,
}

impl Args {
    fn rewriter(self) -> Result<Rewriter> {
        let provider = Provider::parse(&self.provider)
            .ok_or_else(|| anyhow!("unknown provider {}", self.provider))?;
        let api = match &self.api {
            Some(a) => Some(Api::parse(a).ok_or_else(|| anyhow!("unknown api {a}"))?),
            None => None,
        };
        Ok(Rewriter {
            endpoint: Endpoint::new(provider, self.base_url.as_deref(), api),
            store: Store::at(self.credentials_dir),
            agent: api::agent(Duration::from_millis(self.timeout_ms)),
            model: self.model,
            instructions: self
                .instructions
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| setsuna_rewrite::DEFAULT_INSTRUCTIONS.to_string()),
            reasoning_effort: self.reasoning_effort.filter(|s| !s.is_empty()),
        })
    }
}

#[derive(Deserialize)]
struct Request {
    id: u64,
    #[serde(default)]
    context: String,
    #[serde(default)]
    reading: String,
    text: String,
}

#[derive(Serialize)]
struct Response {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    ms: f64,
}

impl Rewriter {
    fn respond(&self, id: u64, context: &str, reading: &str, text: &str) -> Response {
        let t0 = Instant::now();
        let input = setsuna_rewrite::rewrite_input(context, reading, text);
        let result = api::complete(
            &self.endpoint,
            &self.store,
            &self.agent,
            &api::Request {
                model: &self.model,
                instructions: &self.instructions,
                input: &input,
                reasoning_effort: self.reasoning_effort.as_deref(),
            },
        )
        .map(|t| setsuna_rewrite::clean_output(&t))
        .and_then(|t| {
            if t.is_empty() {
                Err(Coded::new("bad_response", "the model returned no text").into())
            } else {
                Ok(t)
            }
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(text) => Response {
                id,
                text: Some(text),
                error: None,
                code: None,
                ms,
            },
            Err(e) => Response {
                id,
                text: None,
                code: Some(Coded::code_of(&e).unwrap_or("error").to_string()),
                error: Some(format!("{e:#}")),
                ms,
            },
        }
    }
}

fn write_line<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", serde_json::to_string(value)?)?;
    out.flush()?;
    Ok(())
}

fn run_daemon(args: Args) -> Result<()> {
    let rewriter = args.rewriter()?;
    write_line(&serde_json::json!({
        "ready": true,
        "version": env!("CARGO_PKG_VERSION"),
        "model": rewriter.model,
    }))?;
    for line in BufReader::new(std::io::stdin().lock()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(r) => rewriter.respond(r.id, &r.context, &r.reading, &r.text),
            Err(e) => Response {
                id: 0,
                text: None,
                error: Some(format!("invalid request json: {e}")),
                code: Some("error".into()),
                ms: 0.0,
            },
        };
        // the request's text is the user's: only its outcome goes to the log
        eprintln!(
            "rewrite id={} ms={:.0} code={}",
            response.id,
            response.ms,
            response.code.as_deref().unwrap_or("ok")
        );
        write_line(&response)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Rewrite {
            args,
            context,
            reading,
            text,
        } => write_line(&args.rewriter()?.respond(0, &context, &reading, &text)),
        Command::Models { args } => {
            let r = args.rewriter()?;
            for m in api::list_models(&r.endpoint, &r.store, &r.agent)? {
                write_line(&m)?;
            }
            Ok(())
        }
        Command::Daemon { args } => run_daemon(args),
    }
}
