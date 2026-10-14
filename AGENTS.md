# Repository Rules

## Safety

- Do not delete, overwrite or `git restore`/`checkout --`/`clean` anything the user did not ask you to.
- Stage explicit file paths only: no `git add -A`, `git add .` or `git add <directory>`.
- One-off scripts and logs go under `_scratch/`, large or generated files under `local/` (both gitignored).
- Codenames (model names) belong in configs, paths and manifests, never in code.

## Layers

Dependencies point one way: artifact -> runtime -> platform.

| layer    | where                                  | may use                                                 |
|----------|----------------------------------------|---------------------------------------------------------|
| artifact | `crates/artifact`                      | manifest schemas only                                   |
| runtime  | `crates/engine`, `crates/rerank`       | `crates/artifact`; models load only through a manifest  |
| platform | `crates/ime-core`, `crates/tsf`, `crates/linux`, `crates/rewrite`, `crates/settings`, `crates/demo` | the engine, reranker and rewriter daemons' stdio protocol; `crates/settings` also links `crates/rewrite` (sign-in, keys, models) |

The engine does inference only (proposal -> CTC beam -> candidates), and so
does the reranker (an LM's log-probability of each candidate). IME-level
behaviour (romaji composition, which kana candidates to show and where, when
to rerank and how to combine the scores) lives in `crates/ime-core` and
`crates/tsf`. The rewriter (`crates/rewrite`) only talks to a model server
(OpenAI with Sign in with ChatGPT or an API key, or an OpenAI-compatible
API); when to ask it and where its rewrite goes is `crates/ime-core`'s.
Credentials never go into logs. The manifest schema names (`*-export-v1`) are
the identifiers the existing model artifacts carry: keep them.

## Checks

`cargo fmt --all --check`, `cargo test`, `cargo build --release`.
