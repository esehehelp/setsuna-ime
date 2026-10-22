//! Drives a `setsuna_rerank::Reranker` from JSONL stdio (daemon mode), a
//! named pipe shared by every app of a Windows login (serve mode) or one
//! request on the command line. The protocol: `setsuna_rerank::protocol`.
//! A request with the candidates of the one before plus more, after the
//! same context, decodes only the new ones.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use setsuna_rerank::{protocol, RerankOptions, Reranker};

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
    /// Serve every client of this user on a named pipe, named after the
    /// user, the manifest, this binary and the options; exit at once when
    /// another server has it.
    #[cfg(windows)]
    Serve {
        #[command(flatten)]
        args: Args,
        /// Seconds without a request before the llama.cpp context (KV cache,
        /// compute buffers) is freed; the model stays loaded.
        #[arg(long, default_value_t = 60)]
        idle_release_secs: u64,
        /// Seconds without a connected client before the server exits.
        #[arg(long, default_value_t = 600)]
        exit_after_secs: u64,
        /// File to append the server's messages to. Default: stderr.
        #[arg(long)]
        log: Option<PathBuf>,
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
    /// Tokens llama.cpp computes in one pass (memory, not results). Default: 512.
    #[arg(long, hide = true)]
    ubatch: Option<u32>,
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
                n_ubatch: self.ubatch,
            },
        )
        .with_context(|| format!("load reranker from {}", self.manifest.display()))
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
    write_line(&protocol::ready(&reranker.config))?;
    for line in BufReader::new(std::io::stdin().lock()).lines() {
        let line = line.context("read stdin line")?;
        if line.trim().is_empty() {
            continue;
        }
        write_line(&protocol::handle(&mut reranker, &line))?;
    }
    Ok(())
}

#[cfg(windows)]
fn run_serve(
    args: Args,
    idle_release_secs: u64,
    exit_after_secs: u64,
    log: Option<PathBuf>,
) -> Result<()> {
    use setsuna_pipe::{exe_identity, rerank_pipe_name, win, RerankKey};
    let user = win::user_sid().context("this process's user")?;
    let manifest = std::fs::canonicalize(&args.manifest)
        .with_context(|| format!("{}", args.manifest.display()))?;
    let exe = exe_identity(&std::env::current_exe()?).context("this binary")?;
    let name = rerank_pipe_name(&RerankKey {
        user: &user,
        manifest: &manifest,
        exe: &exe,
        threads: args.threads,
        max_candidates: args.max_candidates,
        max_context_chars: args.max_context_chars,
        lambda: args.lambda,
    });
    setsuna_rerank::server::serve(
        setsuna_rerank::server::ServeOptions {
            pipe_name: name,
            user_sid: user,
            idle_after: std::time::Duration::from_secs(idle_release_secs),
            exit_after: std::time::Duration::from_secs(exit_after_secs),
            log,
            label: "setsuna-rerank serve",
        },
        move || args.load(),
    )
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Score {
            args,
            context,
            candidates,
        } => {
            let mut reranker = args.load()?;
            write_line(&protocol::respond(&mut reranker, 0, &context, &candidates))
        }
        Command::Daemon { args } => run_daemon(args),
        #[cfg(windows)]
        Command::Serve {
            args,
            idle_release_secs,
            exit_after_secs,
            log,
        } => run_serve(args, idle_release_secs, exit_after_secs, log),
    }
}
