//! `setsuna-rerank serve`: one reranker for every app of a Windows login,
//! on a named pipe (`setsuna_pipe`), instead of a child per app each
//! holding its own llama.cpp context.
//!
//! The server's loop is `setsuna_pipe::server`'s: one worker thread owns the
//! reranker and scores the requests of every client in turn. Two clients
//! interleaving make its cache miss, never mix them up (what it keeps is
//! matched token by token). Idle, the llama.cpp context is freed and the
//! next request makes it again.

use anyhow::Result;
pub use setsuna_pipe::server::ServeOptions;
use setsuna_pipe::server::{self, Service};

use crate::{protocol, Reranker};

struct Scoring(Reranker);

impl Service for Scoring {
    fn handle(&mut self, line: &str) -> String {
        let response = protocol::handle(&mut self.0, line);
        serde_json::to_string(&response).unwrap_or_else(|e| {
            serde_json::json!({ "id": response.id, "error": e.to_string() }).to_string()
        })
    }

    fn idle(&mut self) -> Option<String> {
        self.0
            .release()
            .then(|| "idle, the llama.cpp context released".to_string())
    }
}

/// Claim the pipe, load the reranker with `load` and serve its clients until
/// none has been connected for `opts.exit_after`. Returns at once when
/// another server holds the pipe.
pub fn serve(
    opts: ServeOptions,
    load: impl FnOnce() -> Result<Reranker> + Send + 'static,
) -> Result<()> {
    server::serve(opts, move || {
        let reranker = load()?;
        let ready = serde_json::to_string(&protocol::ready(&reranker.config))?;
        Ok((Scoring(reranker), ready))
    })
}
