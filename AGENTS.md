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
| platform | `crates/ime-core`, `crates/tsf`, `crates/linux`, `crates/settings`, `crates/demo` | the engine and reranker daemons' stdio protocol |

The engine does inference only (proposal -> CTC beam -> candidates), and so
does the reranker (an LM's log-probability of each candidate). IME-level
behaviour (romaji composition, which kana candidates to show and where, when
to rerank and how to combine the scores) lives in `crates/ime-core` and
`crates/tsf`. The manifest schema is shared with the research repository
`neo-ime` (`src/neo_ime/artifact`); change both together.
`tests/fixtures/ctc_nat_manifest` is the same fixture as `neo-ime`'s (the
Python writer test reproduces it byte for byte); keep the two copies identical.

## Checks

`cargo fmt --all --check`, `cargo test`, `cargo build --release`.
