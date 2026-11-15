# Repository Rules

## Safety

- Do not delete, overwrite or `git restore`/`checkout --`/`clean` anything the user did not ask you to.
- Stage explicit file paths only: no `git add -A`, `git add .` or `git add <directory>`.
- `_scratch/` and `local/` (gitignored) are scratch space and are wiped without notice: nothing that must last
  stays only there (Data).
- Codenames (model names) belong in configs, paths and manifests, never in code.

## Data

A run that makes data (a build, a download, a bench, an evaluation) is a task in `kairo.toml`, run through Kairo, so
what it makes lands where Kairo keeps it:

| class     | what                                                                 | how |
|-----------|----------------------------------------------------------------------|-----|
| durable   | results: bench runs and scores, evaluation outputs, anything not made again | the task's output directory (`KAIRO_OUTPUT_DIR`), packed and copied to the durable store when the run ends |
| temporal  | builds, public models and dictionaries at a pinned revision, data derived from inputs kept elsewhere | the output of a task declared `regenerable = true`: not copied, made again when a reader needs it after eviction |
| volatile  | scratch, caches (cargo's target and home, downloads)                 | `KAIRO_SCRATCH` (TMP) and `KAIRO_CACHE_DIR` |

- Tasks run in the code snapshot made at apply (`source = "setsuna-ime"`), confined: they write only those
  directories. A task reads another's output as an input (`task = "..."`, `KAIRO_INPUT_<NAME>`); data that exists
  only outside Kairo and cannot be made again is brought in once with `kairo data import` and read by `data = "..."`.
- Tools take their paths as arguments or from a config naming `${KAIRO_*}` (`setsuna-bench harness --config`,
  `kairo/bench/harness.toml`). The IME itself never expands variables in its config.
- The glue a task runs is in the repo (`kairo/`), never in `_scratch/`. A task is immutable: a new run is a new task
  with a new name.

## Layers

Dependencies point one way: artifact -> runtime -> platform.

| layer    | where                                  | may use                                                 |
|----------|----------------------------------------|---------------------------------------------------------|
| artifact | `crates/artifact`                      | manifest schemas only                                   |
| runtime  | `crates/engine`, `crates/rerank`, `crates/analyzer`, `crates/pipe` | `crates/artifact`, `crates/pipe` for `serve`, and `crates/analyzer` for the engine's `--analyzer`; models load only through a manifest  |
| platform | `crates/ime-core`, `crates/tsf`, `crates/tsf-shim`, `crates/tsf-reg`, `crates/linux`, `crates/android`, `crates/rewrite`, `crates/settings` | the engine, reranker and rewriter daemons' stdio protocol, and the shared engine's and reranker's named pipes (`crates/pipe`, Windows); `crates/settings` also links `crates/rewrite` (sign-in, keys, models) |
| tool     | `crates/bench`                         | `crates/ime-core` (the session and its clients), and any installed input method through TSF; its sets (`crates/bench/sets`) are evaluation data, never training data |

The engine does inference only (proposal -> CTC beam -> candidates), and so
does the reranker (an LM's log-probability of each candidate), and so does
the analyzer (text -> words with their readings), which the engine's daemon
hosts so that a login's one process holds its dictionary. IME-level
behaviour (romaji composition, which kana candidates to show and where, when
to rerank and how to combine the scores) lives in `crates/ime-core` and
`crates/tsf`. The rewriter (`crates/rewrite`) only talks to a model server
(OpenAI with Sign in with ChatGPT or an API key, or an OpenAI-compatible
API); when to ask it and where its rewrite goes is `crates/ime-core`'s.
Credentials never go into logs. The manifest schema names (`*-export-v1`) are
the identifiers the existing model artifacts carry: keep them.

## Checks

`cargo fmt --all --check`, `cargo test`, `cargo build --release`.
