//! setsuna-engine: CTC-NAT inference daemon / one-shot CLI (the engine's own implementation, no ONNX Runtime).
//!
//! Drives a `setsuna_engine::EngineSession` from JSONL stdio (daemon mode),
//! a named pipe shared by every app of a Windows login (serve mode, the same
//! lines) or a single CLI invocation (convert mode). Hosts (the IME frontend,
//! the research bench) talk to the daemon only through this protocol; see
//! `backend-manifest`. With `--analyzer`, the daemon also answers `analyze`
//! requests (text -> words with their readings, `setsuna_analyzer`), so the
//! one process a login shares holds the analyzer's dictionary once.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use setsuna_analyzer::{Analyzer, Token};
use setsuna_artifact::{CtcNatManifest, DecodeDefaults};
use setsuna_engine::candidate::identity_pos;
use setsuna_engine::pipeline::correct::Correction;
use setsuna_engine::session::{ConvertPerf, FrameView};
use setsuna_engine::{Candidate, CandidateSource, ConversionRequest, EngineSession};

#[derive(Parser, Debug)]
#[command(
    name = "setsuna-engine",
    version,
    about = "Setsuna CTC-NAT inference engine: daemon or one-shot convert."
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
    /// Serve the daemon's protocol to every client of this user on the named
    /// pipe `--pipe`; exit at once when another server has it.
    #[cfg(windows)]
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        /// The pipe (`\\.\pipe\setsuna-engine-...`), named by the client
        /// from the user, the manifest, this binary and the flags.
        #[arg(long)]
        pipe: String,
        /// Seconds without a connected client before the server exits.
        #[arg(long, default_value_t = 600)]
        exit_after_secs: u64,
        /// File to append the server's messages to. Default: stderr.
        #[arg(long)]
        log: Option<PathBuf>,
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
    /// CTC-NAT artifact manifest. The weights and vocab are read from it;
    /// nothing is inferred from file names.
    #[arg(long)]
    manifest: PathBuf,

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

    /// Weight of the rescoring's two-character term (the manifest's
    /// `rescore.pairs`); 0 turns it off. Default the manifest's, else 0.
    #[arg(long)]
    pair_weight: Option<f32>,

    /// Weight of the rescoring's word term (the manifest's `rescore.words`);
    /// 0 turns it off. Default the manifest's, else 0.
    #[arg(long)]
    word_weight: Option<f32>,

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

    /// Analyzer manifest: answer `analyze` requests with its analysis.
    /// Loaded after the ready line; a request before it is ready is answered
    /// with an error. Default none.
    #[arg(long)]
    analyzer: Option<PathBuf>,

    /// A reading corrector's manifest, run before each conversion (as
    /// SETSUNA_CORRECTOR_MANIFEST): a reading with a slip is converted as it
    /// reads it, the reading as typed offered too. Default: the model's own
    /// corrector, if its manifest carries one.
    #[arg(long)]
    corrector: Option<PathBuf>,

    /// The margin log P(corrected) - log P(as typed) above which the reading
    /// corrector's reading is converted first (as
    /// SETSUNA_CORRECTOR_THRESHOLD): lower, and more slips are converted as
    /// corrected, and more readings typed right are too. Default: the
    /// corrector's own (its manifest's).
    #[arg(long, allow_negative_numbers = true)]
    corrector_threshold: Option<f32>,
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
            pair_weight: self.pair_weight,
            word_weight: self.word_weight,
            // the corrector's own manifest's (or --corrector-threshold, SETSUNA_CORRECTOR_THRESHOLD), not a
            // conversion knob
            corrector_threshold: None,
        }
    }
}

fn build_session(args: &EngineArgs) -> Result<(EngineSession, &'static str)> {
    let manifest = CtcNatManifest::load(&args.manifest)
        .with_context(|| format!("load manifest {}", args.manifest.display()))?;
    let mut session = EngineSession::from_manifest_with_corrector(
        &manifest,
        &args.explicit_knobs(),
        args.corrector.as_deref(),
        args.corrector_threshold,
    )
    .with_context(|| format!("load engine from {}", args.manifest.display()))?;
    session.inspect = args.emit_frames;
    // One line with every resolved decode setting, so a run can be compared
    // against another by what the engine actually used, not by its argv.
    let cfg = &session.config;
    let effective = serde_json::json!({
        "manifest": manifest.manifest_path,
        "encoder": manifest.proposal_native_path,
        "ar_step": manifest.ar_decoder.as_ref().and_then(|a| a.native_path.as_ref()),
        "corrector": session.corrector().map(|c| c.0),
        // +inf (a corrector with no threshold: never converted first) has no JSON number
        "corrector_threshold": session.corrector().map(|c| c.1).filter(|t| t.is_finite()),
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
        "pair_weight": cfg.pair_weight,
        "word_weight": cfg.word_weight,
        "rescore_pairs": manifest.rescore_pairs,
        "rescore_words": manifest.rescore_words,
    });
    eprintln!("[engine] effective {effective}");
    // the manifest's load refuses a package without the native weights, and those are int8
    Ok((session, "native-int8"))
}

/// Candidates of one request and the conversion timings.
/// One request's answer (`decode_one`).
struct Decoded {
    cands: Vec<Candidate>,
    perf: Option<ConvertPerf>,
    frames: Option<Vec<FrameView>>,
    /// The reading corrector's reading, when it changed the one requested
    /// (both were converted, their candidates merged).
    corrected_reading: Option<String>,
    /// What the reading corrector made of the reading, when there is one.
    correction: Option<Correction>,
}

fn decode_one(
    session: &mut EngineSession,
    context: &str,
    reading: &str,
    continue_beam: usize,
) -> Result<Decoded> {
    if reading.is_empty() {
        let empty = Candidate {
            text: String::new(),
            score: 0.0,
            source: CandidateSource::Fallback,
            reading_pos: Vec::new(),
            parts: None,
        };
        return Ok(Decoded {
            cands: vec![empty],
            perf: None,
            frames: None,
            corrected_reading: None,
            correction: None,
        });
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
            parts: None,
        });
    }
    if session.config.beam_width <= 1 && continue_beam == 0 {
        cands.truncate(greedy_answers(&cands));
    }
    Ok(Decoded {
        cands,
        perf: Some(set.perf),
        frames: set.frames,
        corrected_reading: set
            .correction
            .as_ref()
            .filter(|c| c.changed)
            .map(|c| c.reading.clone()),
        correction: set.correction,
    })
}

/// Greedy decoding has one answer for each reading converted: the first
/// candidate, and with a reading corrector's two readings the other's too,
/// which `correct::merge` puts second.
fn greedy_answers(cands: &[Candidate]) -> usize {
    let other = |c: &Candidate| c.source == CandidateSource::OtherReading;
    let other_second = cands.first().is_some_and(|c| !other(c)) && cands.get(1).is_some_and(other);
    1 + usize::from(other_second)
}

#[derive(Deserialize, Debug)]
struct Request {
    id: u64,
    #[serde(default)]
    context: String,
    /// Required but for an `analyze` request.
    #[serde(default)]
    reading: Option<String>,
    #[serde(default)]
    continue_beam: usize,
    /// Texts to analyze instead of a conversion.
    #[serde(default)]
    analyze: Option<Vec<String>>,
}

/// The response to an `analyze` request: each text's words.
#[derive(Serialize, Debug)]
struct AnalyzeResponse<'a> {
    id: u64,
    analyses: Vec<Vec<Token>>,
    engine_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
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
    /// The reading corrector's reading, when it changed the one requested:
    /// `candidates` then holds both readings' (`candidate_reading_pos` still
    /// on the reading requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    corrected_reading: Option<String>,
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
        corrected_reading: Option<String>,
    ) -> Self {
        let candidate_meta = meta.then(|| cands.iter().map(CandidateMetaJson::from).collect());
        let candidate_reading_pos =
            alignment.then(|| cands.iter().map(|c| c.reading_pos.clone()).collect());
        let (candidates, candidate_scores) = cands
            .into_iter()
            .map(|c| (c.text, finite_score(c.score)))
            .unzip();
        Self {
            id,
            candidates,
            candidate_scores,
            engine_ms,
            perf,
            candidate_meta,
            candidate_reading_pos,
            frames,
            corrected_reading,
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
            corrected_reading: None,
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
            score: finite_score(c.score),
            source: c.source.as_str(),
        }
    }
}

/// A score as JSON has numbers: JSON has no infinity or NaN, and serde
/// writes them as `null`, which no client takes for an `f32` (the whole
/// response was lost). Minus infinity and NaN (a hypothesis nothing could
/// score) as the lowest finite score, plus infinity as the highest.
fn finite_score(score: f32) -> f32 {
    if score.is_nan() || score == f32::NEG_INFINITY {
        f32::MIN
    } else if score == f32::INFINITY {
        f32::MAX
    } else {
        score
    }
}

#[derive(Serialize, Debug)]
struct PerfResponse {
    input_len: usize,
    proposal_ms: f64,
    beam_ms: f64,
    post_ms: f64,
    rescore_ms: f64,
    /// The reading corrector's pass, and the other reading's conversion when it changed the reading (0 without it).
    corrector_ms: f64,
    other_reading_ms: f64,
    total_ms: f64,
    hyp_count: usize,
    candidate_count: usize,
    /// With a reading corrector: what it made of the reading.
    #[serde(skip_serializing_if = "Option::is_none")]
    correction: Option<CorrectionJson>,
}

/// `pipeline::correct::Correction` as the response gives it: the log probabilities null when -inf (no alignment
/// fits) or when the corrector did not run (a reading too long for one pass).
#[derive(Serialize, Debug)]
struct CorrectionJson {
    reading: String,
    changed: bool,
    applied: bool,
    score_fix: Option<f32>,
    score_typed: Option<f32>,
    margin: Option<f32>,
}

impl PerfResponse {
    fn new(p: ConvertPerf, correction: Option<&Correction>) -> Self {
        let finite = |x: f32| x.is_finite().then_some(x);
        Self {
            input_len: p.input_len,
            proposal_ms: p.proposal_ms,
            beam_ms: p.beam_ms,
            post_ms: p.post_ms,
            rescore_ms: p.rescore_ms,
            corrector_ms: p.corrector_ms,
            other_reading_ms: p.other_reading_ms,
            total_ms: p.total_ms,
            hyp_count: p.hyp_count,
            candidate_count: p.candidate_count,
            correction: correction.map(|c| CorrectionJson {
                reading: c.reading.clone(),
                changed: c.changed,
                applied: c.applied,
                score_fix: finite(c.score_fix),
                score_typed: finite(c.score_typed),
                margin: c.margin(),
            }),
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
    /// `analyze` requests are answered (once the analyzer has loaded).
    analyzer: bool,
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
        artifact_formats: vec!["native-int8"],
        optional_features: vec![
            "ctc-prefix-beam",
            "symbol-split",
            "symbol-token",
            "digit-split",
            "candidate-scores",
            "candidate-meta",
            "perf-metrics",
            "candidate-alignment",
            "analyze",
        ],
        request_fields: vec![
            field("id", "u64", true, "Caller-owned request id echoed in the response."),
            field(
                "reading",
                "string",
                true,
                "Hiragana reading to convert. Empty string returns one empty candidate. \
                 Not read in an analyze request.",
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
            field(
                "analyze",
                "string[]",
                false,
                "Texts to analyze instead of a conversion (with --analyzer): the response has \
                 `analyses`, each text's words {surface, start, end, reading?, pos, pos2, alts?, cost}.",
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

/// How long the analyzer stays loaded with nothing asked of the engine
/// (`AnalyzerSlot`), and how often that is looked at.
const ANALYZER_IDLE: Duration = Duration::from_secs(120);
const ANALYZER_CHECK: Duration = Duration::from_secs(15);

/// The analyzer: loaded on a thread of its own once anything is asked (a
/// conversion: the user types, and segment editing, which asks for
/// analyses, comes after typing; the dictionary loads in a fraction of a
/// second, which no conversion waits for), and let go by a thread of its own
/// once nothing has been asked for `ANALYZER_IDLE`. Its dictionary is 27 MB
/// of the engine's 59, and stays loaded only while the IME is in use.
#[derive(Clone)]
struct AnalyzerSlot {
    manifest: PathBuf,
    slot: Arc<Mutex<Slot>>,
}

struct Slot {
    state: SlotState,
    last_asked: Instant,
}

enum SlotState {
    Unloaded,
    Loading,
    Ready(Arc<Analyzer>),
    Failed(String),
}

impl AnalyzerSlot {
    fn new(manifest: PathBuf) -> Self {
        let slot = Arc::new(Mutex::new(Slot {
            state: SlotState::Unloaded,
            last_asked: Instant::now(),
        }));
        // until the slot goes
        let watched = Arc::downgrade(&slot);
        std::thread::spawn(move || loop {
            std::thread::sleep(ANALYZER_CHECK);
            let Some(slot) = watched.upgrade() else {
                return;
            };
            if release_if_idle(&slot) {
                eprintln!("[engine] analyzer let go: nothing asked for {ANALYZER_IDLE:?}");
            }
        });
        Self { manifest, slot }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Slot> {
        lock_slot(&self.slot)
    }

    /// Something was asked: the analyzer loads unless it is (or is being)
    /// loaded already.
    fn warm(&self) {
        let mut slot = self.lock();
        slot.last_asked = Instant::now();
        if !matches!(slot.state, SlotState::Unloaded) {
            return;
        }
        slot.state = SlotState::Loading;
        let (shared, manifest) = (self.slot.clone(), self.manifest.clone());
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let next = match Analyzer::from_manifest_path(&manifest) {
                Ok(a) => {
                    eprintln!(
                        "[engine] analyzer {} loaded in {:.2} s",
                        manifest.display(),
                        t0.elapsed().as_secs_f64()
                    );
                    SlotState::Ready(Arc::new(a))
                }
                Err(e) => {
                    eprintln!("[engine] analyzer {}: {e:#}", manifest.display());
                    SlotState::Failed(format!("{e:#}"))
                }
            };
            lock_slot(&shared).state = next;
        });
    }

    /// The analyzer, an analysis asked: loaded now, or not yet (its load
    /// started).
    fn get(&self) -> Result<Arc<Analyzer>, String> {
        self.warm();
        match &self.lock().state {
            SlotState::Ready(a) => Ok(a.clone()),
            SlotState::Failed(e) => Err(e.clone()),
            SlotState::Unloaded | SlotState::Loading => Err("analyzer loading".to_string()),
        }
    }
}

fn lock_slot(slot: &Mutex<Slot>) -> std::sync::MutexGuard<'_, Slot> {
    slot.lock().unwrap_or_else(|p| p.into_inner())
}

/// Let the analyzer go when nothing has been asked for `ANALYZER_IDLE`
/// (one being used goes once its analysis is done); true when it went.
fn release_if_idle(slot: &Mutex<Slot>) -> bool {
    let mut slot = lock_slot(slot);
    if slot.last_asked.elapsed() < ANALYZER_IDLE || !matches!(slot.state, SlotState::Ready(_)) {
        return false;
    }
    slot.state = SlotState::Unloaded;
    true
}

/// What one request may ask for (any process of the user can reach the
/// shared server): a longer reading, more widening or a larger analysis is
/// answered with an error; a longer context is cut to its end, the part the
/// model reads (its 64 tokens). The IME converts readings of 80 chars at
/// most in one go, and analyzes a conversion at a time.
const MAX_READING_CHARS: usize = 1024;

/// The longest request line the daemon reads from stdin, as the shared
/// engine's pipe takes (`setsuna_pipe::server::MAX_LINE`).
const MAX_REQUEST_LINE: usize = 256 << 10;
const MAX_CONTEXT_CHARS: usize = 4096;
const MAX_CONTINUE_BEAM: usize = 32;
const MAX_ANALYZE_TEXTS: usize = 16;
const MAX_ANALYZE_CHARS: usize = 2048;

/// The last `max` chars of `s`.
fn tail(s: &str, max: usize) -> &str {
    if max == 0 {
        return "";
    }
    match s.char_indices().rev().nth(max - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// A loaded session answering request lines, the daemon's and the server's.
struct Daemon {
    session: EngineSession,
    args: EngineArgs,
    perf_enabled: bool,
    analyzer: Option<AnalyzerSlot>,
}

impl Daemon {
    /// Load the session; the ready line hosts wait for before the first request.
    fn load(args: EngineArgs) -> Result<(Self, String)> {
        let (session, af) = build_session(&args)?;
        let ready = serde_json::to_string(&Ready {
            ready: true,
            version: env!("CARGO_PKG_VERSION"),
            artifact_format: af,
            beam_width: session.config.beam_width,
            top_k: session.config.max_candidates,
            analyzer: args.analyzer.is_some(),
        })?;
        let analyzer = args.analyzer.clone().map(AnalyzerSlot::new);
        let daemon = Self {
            session,
            args,
            perf_enabled: env_flag("SETSUNA_PERF"),
            analyzer,
        };
        Ok((daemon, ready))
    }

    /// The response line to one request line.
    fn respond(&mut self, line: &str) -> String {
        let req: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let err = format!("invalid request json: {e}");
                return to_line(&Response::err(0, 0.0, &err));
            }
        };
        let t0 = Instant::now();
        // the user types: the analyzer segment editing will want comes in
        if let Some(slot) = &self.analyzer {
            slot.warm();
        }
        if let Some(texts) = &req.analyze {
            if texts.len() > MAX_ANALYZE_TEXTS
                || texts.iter().any(|t| t.chars().count() > MAX_ANALYZE_CHARS)
            {
                return to_line(&Response::err(req.id, 0.0, "analysis too large"));
            }
            return self.analyze(req.id, texts, t0);
        }
        let Some(reading) = &req.reading else {
            return to_line(&Response::err(req.id, 0.0, "missing field `reading`"));
        };
        if reading.chars().count() > MAX_READING_CHARS || req.continue_beam > MAX_CONTINUE_BEAM {
            return to_line(&Response::err(
                req.id,
                0.0,
                "reading or continue_beam too large",
            ));
        }
        let context = tail(&req.context, MAX_CONTEXT_CHARS);
        let result = decode_one(&mut self.session, context, reading, req.continue_beam);
        let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(d) => {
                let perf = d
                    .perf
                    .filter(|_| self.perf_enabled)
                    .map(|p| PerfResponse::new(p, d.correction.as_ref()));
                to_line(&Response::ok(
                    req.id,
                    d.cands,
                    engine_ms,
                    perf,
                    self.args.emit_meta,
                    self.args.emit_alignment,
                    d.frames,
                    d.corrected_reading,
                ))
            }
            Err(e) => to_line(&Response::err(req.id, engine_ms, &e.to_string())),
        }
    }
}

impl Daemon {
    /// The response line to an `analyze` request.
    fn analyze(&self, id: u64, texts: &[String], t0: Instant) -> String {
        let analyzer = match &self.analyzer {
            Some(slot) => slot.get(),
            None => Err("no analyzer (--analyzer)".to_string()),
        };
        let (analyses, error) = match &analyzer {
            Ok(a) => (a.analyze_many(texts), None),
            Err(e) => (Vec::new(), Some(e.as_str())),
        };
        let response = AnalyzeResponse {
            id,
            analyses,
            engine_ms: t0.elapsed().as_secs_f64() * 1000.0,
            error,
        };
        serde_json::to_string(&response)
            .unwrap_or_else(|e| serde_json::json!({ "id": id, "error": e.to_string() }).to_string())
    }
}

fn to_line(response: &Response) -> String {
    serde_json::to_string(response).unwrap_or_else(|e| {
        serde_json::json!({ "id": response.id, "error": e.to_string() }).to_string()
    })
}

fn run_daemon(args: EngineArgs) -> Result<()> {
    pin_cpus();
    let (mut daemon, ready) = Daemon::load(args)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{ready}")?;
    out.flush()?;
    let stdin = std::io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let mut buf = Vec::new();
    loop {
        buf.clear();
        // a line no longer than a request may be (the shared engine's limit):
        // one that goes on is read no further and ends the daemon, whose
        // client starts another
        let n = (&mut input)
            .take(MAX_REQUEST_LINE as u64 + 1)
            .read_until(b'\n', &mut buf)
            .context("read stdin line")?;
        if n == 0 {
            break;
        }
        if buf.len() > MAX_REQUEST_LINE {
            anyhow::bail!("a request line over {MAX_REQUEST_LINE} bytes");
        }
        let line = std::str::from_utf8(&buf).context("read stdin line")?;
        if line.trim().is_empty() {
            continue;
        }
        writeln!(
            out,
            "{}",
            daemon.respond(line.trim_end_matches(['\r', '\n']))
        )?;
        out.flush()?;
    }
    // stdin closed: the client is gone. Leave without running exit's
    // handlers: nothing needs them (they once raced an inference runtime's
    // threads, and bionic aborted on the mutex they destroyed).
    #[cfg(unix)]
    unsafe {
        libc::_exit(0)
    }
    #[cfg(not(unix))]
    Ok(())
}

#[cfg(windows)]
impl setsuna_pipe::server::Service for Daemon {
    fn handle(&mut self, line: &str) -> String {
        self.respond(line)
    }
}

/// Serve the clients of `pipe` (the name the IME's client computed for this
/// user, manifest, binary and these flags) until none has been connected
/// for `exit_after_secs`.
#[cfg(windows)]
fn run_serve(
    args: EngineArgs,
    pipe: String,
    exit_after_secs: u64,
    log: Option<PathBuf>,
) -> Result<()> {
    pin_cpus();
    let user = setsuna_pipe::win::user_sid().context("this process's user")?;
    let exit_after = std::time::Duration::from_secs(exit_after_secs);
    setsuna_pipe::server::serve(
        setsuna_pipe::server::ServeOptions {
            pipe_name: pipe,
            user_sid: user,
            // the session has nothing to free and make again (the analyzer
            // goes by itself: `AnalyzerSlot`)
            idle_after: exit_after,
            exit_after,
            log,
            label: "setsuna-engine serve",
        },
        move || Daemon::load(args),
    )
}

/// The daemon's threads on one L3 (`setsuna_engine::cpu`, with SETSUNA_CPU_AFFINITY=1), before the session starts
/// any.
fn pin_cpus() {
    if let Some(cpus) = setsuna_engine::affinity::pin_to_one_l3() {
        eprintln!("[engine] pinned to CPUs {cpus:?} (one L3)");
    }
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
    let d = decode_one(&mut session, &context, &reading, continue_beam)?;
    let engine_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let perf = d
        .perf
        .filter(|_| perf_enabled)
        .map(|p| PerfResponse::new(p, d.correction.as_ref()));
    println!(
        "{}",
        serde_json::to_string(&Response::ok(
            0,
            d.cands,
            engine_ms,
            perf,
            args.emit_meta,
            args.emit_alignment,
            d.frames,
            d.corrected_reading,
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
        #[cfg(windows)]
        Command::Serve {
            engine,
            pipe,
            exit_after_secs,
            log,
        } => run_serve(engine, pipe, exit_after_secs, log),
        Command::BackendManifest { pretty } => run_backend_manifest(pretty),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_long_context_is_cut_to_its_end() {
        assert_eq!(super::tail("あいうえお", 3), "うえお");
        assert_eq!(super::tail("あい", 3), "あい");
        assert_eq!(super::tail("あい", 0), "");
    }

    use super::*;

    /// A score JSON cannot hold goes out as a number all the same (serde
    /// wrote `null`, and the client lost the whole response).
    #[test]
    fn every_score_is_a_json_number() {
        let scores = [f32::NEG_INFINITY, f32::NAN, f32::INFINITY, -1.5];
        let json = serde_json::to_string(&scores.map(finite_score)).unwrap();
        let back: Vec<f32> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, vec![f32::MIN, f32::MIN, f32::MAX, -1.5]);
    }

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
            parts: None,
        }];
        let resp = Response::ok(7, cands.clone(), 1.0, None, true, false, None, None);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["candidates"][0], "天気");
        assert_eq!(json["candidate_meta"][0]["source"], "beam");
        assert_eq!(json["candidate_meta"][0]["score"], -0.5);
        assert!(json.get("candidate_reading_pos").is_none());
        let json = serde_json::to_value(Response::ok(7, cands, 1.0, None, false, true, None, None))
            .unwrap();
        assert_eq!(json["candidate_reading_pos"][0], serde_json::json!([0, 2]));
    }

    #[test]
    fn analyze_requests_need_no_reading() {
        let r: Request = serde_json::from_str(r#"{"id":3,"analyze":["今日は"]}"#).unwrap();
        assert_eq!(r.analyze.as_deref(), Some(&["今日は".to_string()][..]));
        assert_eq!(r.reading, None);
        let r: Request = serde_json::from_str(r#"{"id":4,"reading":"きょう"}"#).unwrap();
        assert_eq!((r.reading.as_deref(), r.analyze), (Some("きょう"), None));
        // a manifest that is not there: loading, then its error
        let slot = AnalyzerSlot::new(PathBuf::from("no such analyzer.json"));
        assert_eq!(slot.get().err().as_deref(), Some("analyzer loading"));
        let t0 = Instant::now();
        while slot.get().err().as_deref() == Some("analyzer loading") {
            assert!(t0.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(slot.get().is_err());
        // what failed is not let go (nor loaded again at each ask)
        slot.lock().last_asked = Instant::now() - ANALYZER_IDLE;
        assert!(!release_if_idle(&slot.slot));
        let json = serde_json::to_value(AnalyzeResponse {
            id: 3,
            analyses: vec![vec![Token {
                surface: "今日".into(),
                start: 0,
                end: 2,
                reading: Some("きょう".into()),
                pos: "名詞".into(),
                pos2: "普通名詞".into(),
                alts: vec![("こんにち".into(), 1348)],
                cost: 100,
            }]],
            engine_ms: 0.1,
            error: None,
        })
        .unwrap();
        assert_eq!(json["analyses"][0][0]["reading"], "きょう");
        assert_eq!(
            json["analyses"][0][0]["alts"][0],
            serde_json::json!(["こんにち", 1348])
        );
        assert!(json.get("error").is_none());
    }
}
