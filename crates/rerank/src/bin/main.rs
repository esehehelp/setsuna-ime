//! Drives a `setsuna_rerank::Reranker` from JSONL stdio (daemon mode) or
//! one request on the command line.
//!
//! Daemon protocol (one JSON object per line):
//!   ready:    {"ready":true,"version":"...","threads":16,"max_context_chars":40,
//!              "max_candidates":20,"lambda":0.3}
//!   request:  {"id":1,"context":"...","candidates":["...","..."]}
//!   response: {"id":1,"logp":[-12.3,-15.1],"score_ms":42.0,"cached":23,"decoded":40}
//!             or {"id":1,"error":"..."}
//! A request with the candidates of the one before plus more, after the same
//! context, decodes only the new ones.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use setsuna_rerank::{RerankOptions, Reranker};

#[derive(Parser, Debug)]
#[command(
    name = "setsuna-rerank",
    version,
    about = "Setsuna candidate reranker: a causal LM's log-probability of each candidate."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Score one request and print the result as JSON, then exit.
    Score {
        #[command(flatten)]
        args: Args,
        #[arg(long, default_value = "")]
        context: String,
        /// The candidates, best first.
        #[arg(long, num_args = 1.., required = true)]
        candidates: Vec<String>,
    },
    /// Read JSONL requests from stdin, write JSONL responses to stdout.
    Daemon {
        #[command(flatten)]
        args: Args,
    },
}

#[derive(Parser, Debug)]
struct Args {
    /// LM scorer manifest (`neo-ime-lm-scorer-export-v1`).
    #[arg(long)]
    manifest: PathBuf,
    /// Threads llama.cpp computes with. Default: half the logical cores.
    #[arg(long)]
    threads: Option<usize>,
    /// Left context chars read before a candidate. Default: the manifest's, else 40.
    #[arg(long)]
    max_context_chars: Option<usize>,
    /// Candidates a request may hold. Default: the manifest's, else 20.
    #[arg(long)]
    max_candidates: Option<usize>,
    /// The LM score's weight, reported to the caller. Default: the manifest's, else 0.3.
    #[arg(long)]
    lambda: Option<f32>,
}

impl Args {
    fn load(&self) -> Result<Reranker> {
        Reranker::load_path(
            &self.manifest,
            &RerankOptions {
                threads: self.threads,
                max_context_chars: self.max_context_chars,
                max_candidates: self.max_candidates,
                lambda: self.lambda,
            },
        )
        .with_context(|| format!("load reranker from {}", self.manifest.display()))
    }
}

#[derive(Deserialize)]
struct Request {
    id: u64,
    #[serde(default)]
    context: String,
    candidates: Vec<String>,
}

#[derive(Serialize)]
struct Response {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    logp: Option<Vec<f64>>,
    score_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoded: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn respond(reranker: &mut Reranker, id: u64, context: &str, candidates: &[String]) -> Response {
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
            id,
            logp: None,
            score_ms,
            cached: None,
            decoded: None,
            error: Some(format!("{e:#}")),
        },
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
    let mut reranker = args.load()?;
    let mut ready = serde_json::to_value(reranker.config)?;
    ready["ready"] = true.into();
    ready["version"] = env!("CARGO_PKG_VERSION").into();
    write_line(&ready)?;
    for line in BufReader::new(std::io::stdin().lock()).lines() {
        let line = line.context("read stdin line")?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => respond(&mut reranker, req.id, &req.context, &req.candidates),
            Err(e) => Response {
                id: 0,
                logp: None,
                score_ms: 0.0,
                cached: None,
                decoded: None,
                error: Some(format!("invalid request json: {e}")),
            },
        };
        write_line(&response)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Score {
            args,
            context,
            candidates,
        } => {
            let mut reranker = args.load()?;
            write_line(&respond(&mut reranker, 0, &context, &candidates))
        }
        Command::Daemon { args } => run_daemon(args),
    }
}
