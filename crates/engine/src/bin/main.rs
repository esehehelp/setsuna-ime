//! setsuna-engine: ONNX CTC-NAT inference daemon / one-shot CLI.
//!
//! Drives a `setsuna_engine::EngineSession` from JSONL stdio (daemon mode) or
//! a single CLI invocation (convert mode). Hosts (the IME frontend, the
//! research bench) talk to the daemon only through this protocol; see
//! `backend-manifest`.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use setsuna_artifact::{CtcNatManifest, DecodeDefaults};
use setsuna_engine::candidate::identity_pos;
use setsuna_engine::session::{ConvertPerf, FrameView};
use setsuna_engine::{Candidate, CandidateSource, ConversionRequest, EngineSession};

#[derive(Parser, Debug)]
#[command(
    name = "setsuna-engine",
    version,
    about = "Setsuna ONNX CTC-NAT inference engine: daemon or one-shot convert."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert a single reading and print candidates as JSON, then exit.
    Convert {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long)]
        reading: String,
        #[arg(long, default_value = "")]
        context: String,
        /// Ask for more candidates of the same reading: the beam and the
        /// candidate limit widened by 10 per step.
        #[arg(long, default_value_t = 0)]
        continue_beam: usize,
    },
    /// Read JSONL requests from stdin, write JSONL responses to stdout.
    /// Stays alive until stdin closes.
    Daemon {
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Print the stable backend protocol description as JSON.
    BackendManifest {
        /// Pretty-print the JSON for humans.
        #[arg(long)]
        pretty: bool,
    },
}

/// Decode knobs. Every knob is optional: a value given here wins, then the
/// manifest's `decode_defaults`, then the engine default.
#[derive(Parser, Debug, Clone)]
struct EngineArgs {
    /// CTC-NAT artifact manifest. The proposal graph and vocab are read from
    /// it; nothing is inferred from file names.
    #[arg(long)]
    manifest: PathBuf,

    /// ONNX Runtime library to load (Microsoft's official build). Default:
    /// SETSUNA_ORT_DYLIB, else onnxruntime.dll next to this executable.
    #[arg(long)]
    ort_dylib: Option<PathBuf>,

    /// Beam width (1 = greedy). Default 1.
    #[arg(long)]
    beam_width: Option<usize>,

    /// Max candidates returned per request. Default 5.
    #[arg(long)]
    top_k: Option<usize>,

    /// Per-step expansion breadth in the prefix-beam loop. Default 16.
    #[arg(long)]
    top_k_per_step: Option<usize>,

    /// Maximum left-context tokens fed to the model (a char the vocabulary lacks takes one a byte). Default 64.
    #[arg(long)]
    max_context: Option<usize>,

    /// Comma-separated token IDs whose log-prob is forced to -inf before
    /// decoding. The bare flag sets an empty list. Default none.
    #[arg(long, value_delimiter = ',', num_args = 0..=1)]
    vocab_blacklist: Option<Vec<u32>>,

    /// Split the reading at bracket / annotation symbols, convert each kana
    /// span separately, and splice the symbols back. `--symbol-split` alone
    /// means true. Default false.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    symbol_split: Option<bool>,

    /// Read each run of symbols in the reading as one `◇` token (models
    /// trained on rows with symbol tokens) and put the runs back into the
    /// candidates; wins over symbol split. `--symbol-token` alone means
    /// true. Default false.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    symbol_token: Option<bool>,

    /// Cut the runs of ASCII digits (with a `.` / `,` between two digits) out
    /// of the reading, splice them back verbatim, and convert the rest after
    /// them; before symbol token / symbol split. `--digit-split` alone means
    /// true. Default false.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    digit_split: Option<bool>,

    /// Map the candidates' punctuation and full-width ASCII the IME way
    /// (`,` `.` -> `、` `。` but after a Latin letter or a digit, full-width
    /// ASCII -> ASCII, `?` `!` -> full-width). Default false: a candidate is
    /// the model's output as decoded. `--normalize-candidates` alone means
    /// true.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    normalize_candidates: Option<bool>,

    /// Include each candidate's score and source in each response.
    #[arg(long, default_value_t = false)]
    emit_meta: bool,

    /// Include each candidate's CTC alignment (`candidate_reading_pos`: for
    /// each char, the index of the reading char it was emitted at).
    #[arg(long, default_value_t = false)]
    emit_alignment: bool,

    /// Include the model's output frame by frame (`frames`: the input token,
    /// the blank's probability and the N most probable other tokens), for
    /// inspection. Default 0: left out.
    #[arg(long, default_value_t = 0)]
    emit_frames: usize,
}

impl EngineArgs {
    fn explicit_knobs(&self) -> DecodeDefaults {
        DecodeDefaults {
            beam_width: self.beam_width,
            top_k: self.top_k,
            top_k_per_step: self.top_k_per_step,
            max_context: self.max_context,
            vocab_blacklist: self.vocab_blacklist.clone(),
            symbol_split: self.symbol_split,
            symbol_token: self.symbol_token,
            digit_split: self.digit_split,
            normalize_candidates: self.normalize_candidates,
        }
    }
}

fn build_session(args: &EngineArgs) -> Result<(EngineSession, &'static str)> {
    let ort_path = setsuna_engine::runtime::init(args.ort_dylib.as_deref())?;
    let manifest = CtcNatManifest::load(&args.manifest)
        .with_context(|| format!("load manifest {}", args.manifest.display()))?;
    let mut session = EngineSession::from_manifest(&manifest, &args.explicit_knobs())
        .with_context(|| format!("load engine from {}", args.manifest.display()))?;
    session.inspect = args.emit_frames;
    // One line with every resolved decode setting, so a run can be compared
    // against another by what the engine actually used, not by its argv.
    let cfg = &session.config;
    let effective = serde_json::json!({
        "onnxruntime": ort_path,
        "onnxruntime_build": setsuna_engine::runtime::build_info(),
        "manifest": manifest.manifest_path,
        "onnx": manifest.proposal_path,
        "precision": manifest.precision,
        "vocab": manifest.vocab_path,
        "beam_width": cfg.beam_width,
        "max_candidates": cfg.max_candidates,
        "top_k_per_step": cfg.top_k_per_step,
        "max_context": cfg.max_context,
        "vocab_blacklist": cfg.vocab_blacklist,
        "symbol_split": cfg.symbol_split,
        "symbol_token": cfg.symbol_token,
        "digit_split": cfg.digit_split,
        "normalize_candidates": cfg.normalize_candidates,
    });
    eprintln!("[engine] effective {effective}");
    let artifact_format = match manifest.precision.as_str() {
        "fp16" => "onnx-fp16",
        "int8" => "onnx-int8",
        "int4" => "onnx-int4",
        _ => "onnx-fp32",
    };
    Ok((session, artifact_format))
}

/// Candidates of one request and the conversion timings.
fn decode_one(
    session: &mut EngineSession,
    context: &str,
    reading: &str,
    continue_beam: usize,
) -> Result<(Vec<Candidate>, Option<ConvertPerf>, Option<Vec<FrameView>>)> {
    if reading.is_empty() {
        let empty = Candidate {
            text: String::new(),
            score: 0.0,
            source: CandidateSource::Fallback,
            reading_pos: Vec::new(),
        };
        return Ok((vec![empty], None, None));
    }
    let set = session.convert(ConversionRequest {
        context,
        reading,
        continue_beam,
    })?;
    let mut cands = set.candidates;
    if cands.is_empty() {
        cands.push(Candidate {
            text: reading.to_string(),
            score: 0.0,
            source: CandidateSource::Fallback,
            reading_pos: identity_pos(reading),
        });
    }
    // Greedy decoding has exactly one answer.
    if session.config.beam_width <= 1 && continue_beam == 0 {
        cands.truncate(1);
    }
    Ok((cands, Some(set.perf), set.frames))
}

#[derive(Deserialize, Debug)]
struct Request {
    id: u64,
    #[serde(default)]
    context: String,
    reading: String,
    #[serde(default)]
    continue_beam: usize,
}

#[derive(Serialize, Debug)]
struct Response<'a> {
    id: u64,
    candidates: Vec<String>,
    /// Per-candidate CTC score, aligned with `candidates`; greedy and
    /// fallback candidates score 0.0.
    candidate_scores: Vec<f32>,
    engine_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    perf: Option<PerfResponse>,
    /// Per-candidate score and source, aligned with `candidates`. Present
    /// only when `--emit-meta` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_meta: Option<Vec<CandidateMetaJson>>,
    /// Per-candidate reading positions, aligned with `candidates` (one per
    /// char; empty when unknown). Present only when `--emit-alignment` is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_reading_pos: Option<Vec<Vec<u32>>>,
    /// The model's output frame by frame. Present only with `--emit-frames`
    /// (and not for a reading split at symbols).
    #[serde(skip_serializing_if = "Option::is_none")]
    frames: Option<Vec<FrameView>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

impl<'a> Response<'a> {
    fn ok(
        id: u64,
        cands: Vec<Candidate>,
        engine_ms: f64,
        perf: Option<PerfResponse>,
        meta: bool,
        alignment: bool,
        frames: Option<Vec<FrameView>>,
    ) -> Self {
        let candidate_meta = meta.then(|| cands.iter().map(CandidateMetaJson::from).collect());
        let candidate_reading_pos =
            alignment.then(|| cands.iter().map(|c| c.reading_pos.clone()).collect());
        let (candidates, candidate_scores) = cands.into_iter().map(|c| (c.text, c.score)).unzip();
        Self {
            id,
            candidates,
            candidate_scores,
            engine_ms,
            perf,
            candidate_meta,
            candidate_reading_pos,
            frames,
            error: None,
        }
    }

    fn err(id: u64, engine_ms: f64, error: &'a str) -> Self {
        Self {
            id,
            candidates: Vec::new(),
            candidate_scores: Vec::new(),
            engine_ms,
            perf: None,
            candidate_meta: None,
            candidate_reading_pos: None,
            frames: None,
            error: Some(error),
        }
    }
}

#[derive(Serialize, Debug)]
struct CandidateMetaJson {
    score: f32,
    source: &'static str,
}

impl From<&Candidate> for CandidateMetaJson {
    fn from(c: &Candidate) -> Self {
        Self {
            score: c.score,
            source: c.source.as_str(),
        }
    }
}

#[derive(Serialize, Debug)]
struct PerfResponse {
    input_len: usize,
    proposal_ms: f64,
    beam_ms: f64,
    post_ms: f64,
    total_ms: f64,
    hyp_count: usize,
    candidate_count: usize,
}

impl From<ConvertPerf> for PerfResponse {
    fn from(p: ConvertPerf) -> Self {
        Self {
            input_len: p.input_len,
            proposal_ms: p.proposal_ms,
            beam_ms: p.beam_ms,
            post_ms: p.post_ms,
            total_ms: p.total_ms,
            hyp_count: p.hyp_count,
            candidate_count: p.candidate_count,
        }
    }
}

#[derive(Serialize, Debug)]
struct Ready {
    ready: bool,
    version: &'static str,
    artifact_format: &'static str,
    beam_width: usize,
    top_k: usize,
}

#[derive(Serialize, Debug)]
struct BackendManifest {
    name: &'static str,
    version: &'static str,
    protocol_version: u32,
    transports: Vec<TransportManifest>,
    artifact_formats: Vec<&'static str>,
    optional_features: Vec<&'static str>,
    request_fields: Vec<FieldManifest>,
    response_fields: Vec<FieldManifest>,
}

#[derive(Serialize, Debug)]
struct TransportManifest {
    name: &'static str,
    command: &'static str,
    ready_line: &'static str,
    request: &'static str,
    response: &'static str,
}

#[derive(Serialize, Debug)]
struct FieldManifest {
    name: &'static str,
    ty: &'static str,
    required: bool,
    description: &'static str,
}

fn field(
    name: &'static str,
    ty: &'static str,
    required: bool,
    description: &'static str,
) -> FieldManifest {
    FieldManifest {
        name,
        ty,
        required,
        description,
    }
}

fn backend_manifest() -> BackendManifest {
    BackendManifest {
        name: "setsuna-engine",
        version: env!("CARGO_PKG_VERSION"),
        protocol_version: 1,
        transports: vec![TransportManifest {
            name: "stdio-jsonl",
            command: "setsuna-engine daemon --manifest <model.manifest.json>",
            ready_line: r#"{"ready":true,...}"#,
            request: r#"{"id":1,"context":"...","reading":"..."}"#,
            response: r#"{"id":1,"candidates":["..."],"candidate_scores":[0.0],"engine_ms":1.0}"#,
        }],
        artifact_formats: vec!["onnx-fp32", "onnx-fp16", "onnx-int8", "onnx-int4"],
        optional_features: vec![
            "ctc-prefix-beam",
            "symbol-split",
            "symbol-token",
            "digit-split",
            "candidate-scores",
            "candidate-meta",
            "perf-metrics",
            "candidate-alignment",
        ],
        request_fields: vec![
            field("id", "u64", true, "Caller-owned request id echoed in the response."),
            field(
                "reading",
                "string",
                true,
                "Hiragana reading to convert. Empty string returns one empty candidate.",
            ),
            field(
                "context",
                "string",
                false,
                "Left surrounding text. Omitted or empty means no context.",
            ),
            field(
                "continue_beam",
                "u32",
                false,
                "n > 0: more candidates of the same request, the beam and the candidate limit \
                 widened by 10 per step (the model is not run again for the last request). \
                 Omitted or 0: a conversion.",
            ),
        ],
        response_fields: vec![
            field(
                "id",
                "u64",
                true,
                "Echo of the request id, or 0 for invalid JSON requests.",
            ),
            field(
                "candidates",
                "string[]",
                true,
                "Ranked UTF-8 conversion candidates, best first.",
            ),
            field(
                "candidate_scores",
                "number[]",
                true,
                "CTC scores aligned with candidates. Greedy and fallback candidates are 0.",
            ),
            field(
                "engine_ms",
                "number",
                true,
                "Total backend latency for this request in milliseconds.",
            ),
            field(
                "perf",
                "object",
                false,
                "Detailed timing object when SETSUNA_PERF=1.",
            ),
            field(
                "candidate_meta",
                "object[]",
                false,
                "Per-candidate {score, source} aligned with candidates. Present only with --emit-meta.",
            ),
            field(
                "candidate_reading_pos",
                "number[][]",
                false,
                "Per candidate, for each char the index of the reading char it was emitted at \
                 (CTC alignment); empty when unknown. Present only with --emit-alignment.",
            ),
            field(
                "error",
                "string",
                false,
                "Present when conversion failed; candidates is empty in that case.",
            ),
        ],
    }
}

fn write_line<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", serde_json::to_string(value)?)?;
    out.flush()?;
    Ok(())
}

fn run_daemon(args: EngineArgs) -> Result<()> {
    let (mut session, af) = build_session(&args)?;
    let perf_enabled = env_flag("SETSUNA_PERF");
    write_line(&Ready {
        ready: true,
        version: env!("CARGO_PKG_VERSION"),
        artifact_format: af,
        beam_width: session.config.beam_width,
        top_k: session.config.max_candidates,
    })?;

    let stdin = std::io::stdin();
    for line in BufReader::new(stdin.lock()).lines() {
        let line = line.context("read stdin line")?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let err = format!("invalid request json: {e}");
                write_line(&Response::err(0, 0.0, &err))?;
                continue;
            }
        };
        let t0 = Instant::now();
        let result = decode_one(&mut session, &req.context, &req.reading, req.continue_beam);
        let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok((cands, perf, frames)) => {
                let perf = perf.filter(|_| perf_enabled).map(PerfResponse::from);
                write_line(&Response::ok(
                    req.id,
                    cands,
                    engine_ms,
                    perf,
                    args.emit_meta,
                    args.emit_alignment,
                    frames,
                ))?;
            }
            Err(e) => write_line(&Response::err(req.id, engine_ms, &e.to_string()))?,
        }
    }
    Ok(())
}

fn run_convert(
    args: EngineArgs,
    reading: String,
    context: String,
    continue_beam: usize,
) -> Result<()> {
    let (mut session, _) = build_session(&args)?;
    let perf_enabled = env_flag("SETSUNA_PERF");
    let t0 = Instant::now();
    let (cands, perf, frames) = decode_one(&mut session, &context, &reading, continue_beam)?;
    let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let perf = perf.filter(|_| perf_enabled).map(PerfResponse::from);
    println!(
        "{}",
        serde_json::to_string(&Response::ok(
            0,
            cands,
            engine_ms,
            perf,
            args.emit_meta,
            args.emit_alignment,
            frames,
        ))?
    );
    Ok(())
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn run_backend_manifest(pretty: bool) -> Result<()> {
    let manifest = backend_manifest();
    if pretty {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
    } else {
        println!("{}", serde_json::to_string(&manifest)?);
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Convert {
            engine,
            reading,
            context,
            continue_beam,
        } => run_convert(engine, reading, context, continue_beam),
        Command::Daemon { engine } => run_daemon(engine),
        Command::BackendManifest { pretty } => run_backend_manifest(pretty),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_manifest_declares_protocol_v1() {
        let manifest = backend_manifest();
        assert_eq!(manifest.name, "setsuna-engine");
        assert_eq!(manifest.protocol_version, 1);
        assert!(manifest.transports.iter().any(|t| t.name == "stdio-jsonl"));
        assert!(manifest
            .request_fields
            .iter()
            .any(|f| f.name == "reading" && f.required));
        assert!(manifest
            .response_fields
            .iter()
            .any(|f| f.name == "candidates" && f.required));
    }

    #[test]
    fn meta_carries_score_and_source() {
        let cands = vec![Candidate {
            text: "天気".into(),
            score: -0.5,
            source: CandidateSource::Beam,
            reading_pos: vec![0, 2],
        }];
        let resp = Response::ok(7, cands.clone(), 1.0, None, true, false, None);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["candidates"][0], "天気");
        assert_eq!(json["candidate_meta"][0]["source"], "beam");
        assert_eq!(json["candidate_meta"][0]["score"], -0.5);
        assert!(json.get("candidate_reading_pos").is_none());
        let json =
            serde_json::to_value(Response::ok(7, cands, 1.0, None, false, true, None)).unwrap();
        assert_eq!(json["candidate_reading_pos"][0], serde_json::json!([0, 2]));
    }
}
