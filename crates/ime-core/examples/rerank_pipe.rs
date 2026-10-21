//! Score candidates through the login's shared reranker server, as the IME
//! of an app does (Windows), starting the server when there is none.
//!
//!   cargo run -p setsuna-core --example rerank_pipe -- \
//!       <setsuna-rerank.exe> <manifest> [--threads N] [--repeat N] [--gap SECS] \
//!       [--hold SECS] <context> <candidate>...
//!
//! Prints the server's pid and each candidate's log-probability, `--repeat`
//! times `--gap` seconds apart, then keeps the connection for `--hold`
//! seconds (a second process run meanwhile shares the server).

#[cfg(windows)]
fn main() {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use setsuna_core::rerank_client::{PipeRerankClient, RerankClient};

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<u64> {
        let i = args.iter().position(|a| a == flag)?;
        let value = args.get(i + 1).map(|v| v.parse().expect("a number"));
        args.drain(i..(i + 2).min(args.len()));
        value
    };
    let threads = take("--threads").map(|t| t as usize);
    let repeat = take("--repeat").unwrap_or(1);
    let gap = take("--gap").unwrap_or(0);
    let hold = take("--hold").unwrap_or(0);
    let [exe, manifest, context, candidates @ ..] = &args[..] else {
        eprintln!(
            "usage: rerank_pipe <exe> <manifest> [--threads N] [--repeat N] [--gap SECS] \
             [--hold SECS] <context> <candidate>..."
        );
        std::process::exit(2);
    };
    let cfg = setsuna_core::config::RerankCfg {
        threads,
        ..Default::default()
    };
    let (tx, rx) = mpsc::channel();
    let client = PipeRerankClient::start(
        manifest.as_ref(),
        exe.as_ref(),
        Box::new(move |id| {
            let _ = tx.send(id);
        }),
        &cfg,
    )
    .expect("start the client");
    for id in 1..=repeat {
        if id > 1 {
            std::thread::sleep(Duration::from_secs(gap));
        }
        let t0 = Instant::now();
        client.dispatch(id, context, candidates, Instant::now());
        if rx.recv_timeout(Duration::from_secs(60)).is_err() {
            println!("request {id}: no result in 60 s");
            continue;
        }
        let result = client.take_latest().expect("a result");
        println!(
            "request {id}: server pid={:?} total_ms={:.0} score_ms={:.1}",
            client.server_pid(),
            t0.elapsed().as_secs_f64() * 1000.0,
            result.score_ms
        );
        for (c, lp) in candidates.iter().zip(&result.logp) {
            println!("{lp:.6}\t{c}");
        }
    }
    std::thread::sleep(Duration::from_secs(hold));
}

#[cfg(not(windows))]
fn main() {
    eprintln!("the shared reranker server is Windows only");
}
