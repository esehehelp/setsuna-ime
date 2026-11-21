//! Engine session: a CTC-NAT artifact loaded through its manifest, its
//! resolved decode settings, and the entry point that runs the conversion
//! pipeline (`crate::pipeline`) for one request.
//!
//! The process environment is read in exactly one place, `RuntimeEnv::read`,
//! when a session is created.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use setsuna_artifact::{CtcNatManifest, DecodeDefaults};

use crate::candidate::{Candidate, CandidateSource, ScoreParts};
use crate::pipeline::artifact_load::Model;
use crate::pipeline::proposal::FrameReduce;
use crate::pipeline::{
    ar, artifact_load, chunk, correct, ctc_score, decode, digitsplit, pool, proposal, rescore,
    symsplit, symtoken,
};
use crate::tokenizer::SharedCharTokenizer;

#[derive(Debug, Clone, Copy, Default)]
pub struct ConvertPerf {
    pub input_len: usize,
    pub proposal_ms: f64,
    /// The proposal's parts: the encoder (its logits; `ort_ms`, the name the
    /// daemon's perf has always given it) and the logits reduced to the
    /// frames' top-k (0 when the proposal came from the cache of the last
    /// conversion).
    pub ort_ms: f64,
    pub reduce_ms: f64,
    pub beam_ms: f64,
    pub post_ms: f64,
    /// The rescoring of the candidates (`pipeline::rescore`).
    pub rescore_ms: f64,
    /// The fusion's scoring of the AR candidates by the CTC model: all of it,
    /// and its encoder alone; 0 without the fusion.
    pub fusion_ms: f64,
    pub fusion_ort_ms: f64,
    /// The reading corrector (`pipeline::correct`): its pass, and its encoder
    /// alone; then the conversion of the other reading when it changed the
    /// reading (the rest of the perf is the first reading's). 0 without it.
    pub corrector_ms: f64,
    pub corrector_ort_ms: f64,
    pub other_reading_ms: f64,
    pub total_ms: f64,
    pub hyp_count: usize,
    pub candidate_count: usize,
}

/// One conversion: a hiragana reading and its left context.
#[derive(Debug, Clone, Copy)]
pub struct ConversionRequest<'a> {
    pub context: &'a str,
    pub reading: &'a str,
    /// 0 for a conversion; n > 0 asks for more candidates of the same
    /// request: the beam and the candidate limit widened by
    /// `CONTINUE_STEP * n`, over the proposal already computed for it.
    pub continue_beam: usize,
}

/// How much each `continue_beam` step widens the beam and the candidates.
pub const CONTINUE_STEP: usize = 10;

/// Candidates after the display limit, best first, with the timings of the
/// conversion that produced them.
#[derive(Debug, Clone)]
pub struct CandidateSet {
    pub candidates: Vec<Candidate>,
    pub perf: ConvertPerf,
    /// The model's output frame by frame, when [`EngineSession::inspect`]
    /// asks for it (and the reading was converted in one pass, not split
    /// at symbols).
    pub frames: Option<Vec<FrameView>>,
    /// What the reading corrector made of the reading, when there is one.
    pub correction: Option<correct::Correction>,
}

/// One output frame of the model, for inspection: the input token it sits
/// on, the blank's probability and the most probable other tokens.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FrameView {
    /// The input token (`[CLS]`, a context char, `[SEP]`, a reading char;
    /// a byte-fallback token as `<0xE3>`).
    pub input: String,
    /// The reading char the frame holds; `None` for `[CLS]`, the context
    /// and `[SEP]`.
    pub reading_index: Option<u32>,
    /// Probability of emitting nothing here.
    pub blank: f32,
    /// The best non-blank tokens and their probabilities, best first.
    pub top: Vec<(String, f32)>,
}

/// The AR search's default beam threshold in nats: on the standard sets of 2026-11-12 the AR model's top 1 and
/// top 5 were those of the search without one, in half the time (one thread, AJIMEE, p50 2.7 ms against 5.6); what
/// it drops lies e^8 below the best.
const AR_MARGIN: f32 = 8.0;

/// The environment variables the engine honours. Everything else comes from
/// the caller and the manifest.
#[derive(Debug, Clone)]
struct RuntimeEnv {
    /// `SETSUNA_TOP_K_PER_STEP` (default 16): beam expansion breadth when
    /// neither the caller nor the manifest sets one.
    top_k_per_step: usize,
    /// `SETSUNA_AR_MARGIN` (default `AR_MARGIN`; `off`: none): the AR search's beam threshold in nats
    /// (`ar::ArSearch::margin`).
    ar_margin: Option<f32>,
    /// `SETSUNA_SCRIPT_BIAS` (unset: none): four numbers, `other,kanji,hiragana,katakana`, added to the prior's
    /// logits of those classes by a model with the script latent (`ar::ArSearch::script_bias`; e.g. `0,0,4,0`
    /// leans to hiragana where the model may choose); a research knob, the IME's setting in setsuna-ime.
    script_bias: [f32; 4],
    /// `SETSUNA_FUSION_MANIFEST` (unset: no fusion): a CTC model's manifest whose model scores an AR model's
    /// candidates ([`Fusion`]), a research knob until the fusion is measured (and has a place in the manifest).
    fusion_manifest: Option<PathBuf>,
    /// `SETSUNA_FUSION_WEIGHT` (default 1): the CTC log-likelihood's weight in a fused candidate's score.
    fusion_weight: f32,
    /// `SETSUNA_CORRECTOR_MANIFEST` (unset: the manifest's own `corrector`, if any): a reading corrector's manifest
    /// ([`Corrector`], `pipeline::correct`) run in place of the manifest's, a research knob.
    corrector_manifest: Option<PathBuf>,
    /// `SETSUNA_CORRECTOR_THRESHOLD` (unset: the manifest's corrector's threshold, or for a corrector of its own
    /// manifest its decode_defaults.corrector_threshold, else +inf: the corrected reading never converted first, only
    /// offered): the margin
    /// log P(corrected) - log P(as typed) above which the corrected reading is converted first.
    corrector_threshold: Option<f32>,
    /// `SETSUNA_CORRECTOR=0` (or `off`): no reading corrector, not even the manifest's (the conversion alone, a
    /// research comparison).
    corrector_off: bool,
}

/// The reading corrector's threshold: the one `given` (the daemon's `--corrector-threshold`), else the environment's
/// (`SETSUNA_CORRECTOR_THRESHOLD`, as its text), else none. Only the one taken is checked, so an environment value
/// that is not a number does not stop an engine given its own.
fn corrector_threshold_of(given: Option<f32>, env: Option<&str>) -> Result<Option<f32>> {
    match (given, env) {
        (Some(t), _) if t.is_nan() => {
            bail!("the reading corrector's threshold is not a number")
        }
        (Some(t), _) => Ok(Some(t)),
        (None, Some(s)) => Ok(Some(
            s.parse::<f32>()
                .ok()
                .filter(|t| !t.is_nan())
                .with_context(|| format!("SETSUNA_CORRECTOR_THRESHOLD={s} is not a number"))?,
        )),
        (None, None) => Ok(None),
    }
}

impl RuntimeEnv {
    /// The environment, with the corrector's threshold `given` over its own ([`corrector_threshold_of`]).
    fn read(given_corrector_threshold: Option<f32>) -> Result<Self> {
        let fusion_weight = match std::env::var("SETSUNA_FUSION_WEIGHT") {
            Ok(s) => s
                .parse::<f32>()
                .ok()
                .filter(|w| w.is_finite())
                .with_context(|| format!("SETSUNA_FUSION_WEIGHT={s} is not a number"))?,
            Err(_) => 1.0,
        };
        let corrector_threshold = corrector_threshold_of(
            given_corrector_threshold,
            std::env::var("SETSUNA_CORRECTOR_THRESHOLD").ok().as_deref(),
        )?;
        let script_bias = match std::env::var("SETSUNA_SCRIPT_BIAS") {
            Ok(s) if !s.trim().is_empty() => parse_script_bias(&s).with_context(|| {
                format!("SETSUNA_SCRIPT_BIAS={s}: four numbers, other,kanji,hiragana,katakana")
            })?,
            _ => [0.0; 4],
        };
        let corrector_off = match std::env::var("SETSUNA_CORRECTOR") {
            Ok(s) if s == "0" || s.eq_ignore_ascii_case("off") => true,
            Ok(s) if s.is_empty() || s == "1" || s.eq_ignore_ascii_case("on") => false,
            Ok(s) => bail!(
                "SETSUNA_CORRECTOR={s}: 0 / off turns the reading corrector off, 1 / on leaves it"
            ),
            Err(_) => false,
        };
        Ok(Self {
            top_k_per_step: env_usize("SETSUNA_TOP_K_PER_STEP", 16),
            ar_margin: match std::env::var("SETSUNA_AR_MARGIN") {
                Err(_) => Some(AR_MARGIN),
                Ok(s) if matches!(s.trim(), "off" | "none") => None,
                Ok(s) => Some(
                    s.trim()
                        .parse::<f32>()
                        .ok()
                        // a margin is nats behind the best: none below 0, no NaN
                        .filter(|m| m.is_finite() && *m >= 0.0)
                        .with_context(|| {
                            format!("SETSUNA_AR_MARGIN={s}: nats (0 or more), or off")
                        })?,
                ),
            },
            script_bias,
            fusion_manifest: std::env::var_os("SETSUNA_FUSION_MANIFEST")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
            fusion_weight,
            corrector_manifest: std::env::var_os("SETSUNA_CORRECTOR_MANIFEST")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
            corrector_threshold,
            corrector_off,
        })
    }
}

/// A reading corrector (`pipeline::correct`): a CTC model over the conversion model's vocabulary, read greedily
/// over the reading and its context before the conversion, with the last reading's correction kept (a request for
/// more candidates of it, or the same request again, does not run it again).
struct Corrector {
    /// The corrector's own session (its manifest's decode settings: context length, blacklist, symbol token).
    session: Box<EngineSession>,
    manifest: PathBuf,
    threshold: f32,
    last: Option<(String, String, correct::Correction)>,
}

/// An AR model's candidates scored by a CTC model too: each hypothesis' tokens get the CTC model's log-likelihood
/// over every alignment to its frames for the same context and reading (`pipeline::ctc_score`), and the candidates
/// are ordered by `ar + weight * ctc` (both in [`Candidate::parts`]). The two models share one vocabulary.
struct Fusion {
    /// The CTC model's own session (its manifest's decode settings: its context length and blacklist).
    scorer: Box<EngineSession>,
    manifest: PathBuf,
    weight: f32,
}

/// Resolved decode settings of a session.
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Prefix-beam width; 1 means true CTC greedy collapse.
    pub beam_width: usize,
    pub top_k_per_step: usize,
    /// Maximum returned candidates.
    pub max_candidates: usize,
    /// Maximum left-context characters fed to the model.
    pub max_context: usize,
    /// Token ids whose log-prob is forced to -inf before decoding.
    pub vocab_blacklist: Vec<u32>,
    /// Convert bracket / annotation symbols' kana spans separately and splice
    /// the symbols back (see `pipeline::symsplit`).
    pub symbol_split: bool,
    /// Read each run of symbols in the reading as one `◇` token, in one pass,
    /// and put the runs back into the candidates (see `pipeline::symtoken`).
    /// Wins over `symbol_split`: the reading is not split.
    pub symbol_token: bool,
    /// Cut the runs of ASCII digits out of the reading and splice them back
    /// verbatim; the other segments are converted after them (see
    /// `pipeline::digitsplit`). Applies before symbol token / symbol split,
    /// which then handle each other segment.
    pub digit_split: bool,
    /// Map the candidates' punctuation and full-width ASCII the IME way
    /// (`normalize_candidate`); off, a candidate is the model's output as
    /// decoded (the research default: what training and the bench compare).
    pub normalize_candidates: bool,
    /// Weights of the rescoring's two-character and word terms
    /// (`pipeline::rescore`); 0 leaves a term out (and its table unloaded).
    pub pair_weight: f32,
    pub word_weight: f32,
}

impl DecodeConfig {
    /// Every knob: `explicit` value > the manifest's `decode_defaults` > the
    /// engine default.
    fn resolve(explicit: &DecodeDefaults, d: &DecodeDefaults, env: &RuntimeEnv) -> Self {
        Self {
            beam_width: explicit.beam_width.or(d.beam_width).unwrap_or(1).max(1),
            // 0 would underflow the per-step top-K selection in prefix beam.
            top_k_per_step: explicit
                .top_k_per_step
                .or(d.top_k_per_step)
                .unwrap_or(env.top_k_per_step)
                .max(1),
            max_candidates: explicit.top_k.or(d.top_k).unwrap_or(5).max(1),
            max_context: explicit.max_context.or(d.max_context).unwrap_or(64),
            vocab_blacklist: explicit
                .vocab_blacklist
                .clone()
                .or_else(|| d.vocab_blacklist.clone())
                .unwrap_or_default(),
            symbol_split: explicit.symbol_split.or(d.symbol_split).unwrap_or(false),
            symbol_token: explicit.symbol_token.or(d.symbol_token).unwrap_or(false),
            digit_split: explicit.digit_split.or(d.digit_split).unwrap_or(false),
            normalize_candidates: explicit
                .normalize_candidates
                .or(d.normalize_candidates)
                .unwrap_or(false),
            pair_weight: explicit.pair_weight.or(d.pair_weight).unwrap_or(0.0),
            word_weight: explicit.word_weight.or(d.word_weight).unwrap_or(0.0),
        }
    }

    /// Candidates kept per span before the display cut: the whole beam, so a
    /// symbol-split re-splice sees every hypothesis.
    fn pool_limit(&self) -> usize {
        self.beam_width.max(self.max_candidates)
    }
}

pub struct EngineSession {
    pub tokenizer: SharedCharTokenizer,
    /// The proposal graph (CTC) or the AR decoder's graphs.
    model: Model,
    /// Positional-table size of the model (manifest `max_seq_len`).
    pub seq_len: usize,
    pub config: DecodeConfig,
    /// How the proposal reduces a frame.
    reduce: FrameReduce,
    /// Tokens per frame [`CandidateSet::frames`] lists; 0 (the default)
    /// leaves the frames out.
    pub inspect: usize,
    /// Candidates of the chunks of long readings, by (context, reading,
    /// beam width).
    chunk_cache: HashMap<(String, String, usize), Vec<Candidate>>,
    /// Two-slot LRU cache of recent span proposals: a `continue_beam` request
    /// (or both readings of a reading-corrected conversion) decodes again
    /// without re-running the encoder.
    last_proposals: ProposalCache,
    /// The model's CTC-CRF transitions (manifest `crf`), added by the beam.
    crf: Option<crate::pipeline::crf::CrfTables>,
    /// The candidates' rescoring (manifest `rescore`), when a weight is set.
    rescorer: Option<rescore::Rescorer>,
    /// The AR search's beam threshold (`RuntimeEnv::ar_margin`).
    ar_margin: Option<f32>,
    /// The AR search's preference among the scripts of a model with the script latent
    /// (`RuntimeEnv::script_bias`, `ar::ArSearch::script_bias`).
    script_bias: [f32; 4],
    /// A CTC model scoring the AR candidates (`RuntimeEnv::fusion_manifest`).
    fusion: Option<Fusion>,
    /// A reading corrector run before the conversion (`RuntimeEnv::corrector_manifest`).
    corrector: Option<Corrector>,
}

/// A proposal with what it was computed for.
struct LastProposal {
    context: String,
    reading: String,
    greedy: bool,
    inspect: usize,
    proposal: proposal::ProposalOutput,
    frame_reading: Vec<u32>,
    inputs: Vec<i64>,
}

/// Two-slot LRU cache of recent span proposals so a reading corrector's
/// `first` / `other` pair does not evict each other before `continue_beam`.
#[derive(Default)]
struct ProposalCache {
    slots: [Option<LastProposal>; 2],
    /// The settings its proposals were made with.
    settings: ProposalSettings,
}

/// What a proposal depends on besides its key (context, reading, greedy, inspect): the session's settings, which a
/// caller of the library may change between conversions (the daemon never does).
#[derive(Default, PartialEq)]
struct ProposalSettings {
    max_context: usize,
    top_k: usize,
    vocab_blacklist: Vec<u32>,
}

impl ProposalCache {
    /// Empty the cache when its proposals were made with other settings.
    fn settings(&mut self, max_context: usize, top_k: usize, vocab_blacklist: &[u32]) {
        let s = &self.settings;
        if s.max_context != max_context || s.top_k != top_k || s.vocab_blacklist != vocab_blacklist
        {
            self.slots = [None, None];
            self.settings = ProposalSettings {
                max_context,
                top_k,
                vocab_blacklist: vocab_blacklist.to_vec(),
            };
        }
    }

    fn get(
        &mut self,
        context: &str,
        reading: &str,
        greedy: bool,
        inspect: usize,
    ) -> Option<&LastProposal> {
        let idx = self.slots.iter().position(|slot| {
            slot.as_ref().is_some_and(|p| {
                p.greedy == greedy
                    && p.inspect == inspect
                    && p.context == context
                    && p.reading == reading
            })
        })?;
        if idx == 1 {
            self.slots.swap(0, 1);
        }
        self.slots[0].as_ref()
    }

    fn insert(&mut self, entry: LastProposal) {
        self.slots[1] = self.slots[0].take();
        self.slots[0] = Some(entry);
    }

    fn mru(&self) -> Option<&LastProposal> {
        self.slots[0].as_ref()
    }
}

/// Chunk results [`EngineSession`] keeps before starting over.
const CHUNK_CACHE: usize = 64;

impl EngineSession {
    /// Load the artifact a CTC-NAT manifest names and resolve every decode
    /// knob as: `explicit` value > the manifest's `decode_defaults` > engine
    /// default.
    pub fn from_manifest(manifest: &CtcNatManifest, explicit: &DecodeDefaults) -> Result<Self> {
        Self::from_manifest_with_corrector(manifest, explicit, None, None)
    }

    /// `from_manifest` with a reading corrector's manifest of its own (the daemon's `--corrector`), taken as
    /// SETSUNA_CORRECTOR_MANIFEST is, and a threshold for the corrector there is (`--corrector-threshold`), taken
    /// as SETSUNA_CORRECTOR_THRESHOLD is.
    pub fn from_manifest_with_corrector(
        manifest: &CtcNatManifest,
        explicit: &DecodeDefaults,
        corrector: Option<&Path>,
        corrector_threshold: Option<f32>,
    ) -> Result<Self> {
        let mut env = RuntimeEnv::read(corrector_threshold)?;
        if let Some(p) = corrector {
            env.corrector_manifest = Some(p.to_path_buf());
        }
        Self::build(manifest, explicit, &env)
    }

    /// The fusion's CTC manifest and weight, when there is one.
    pub fn fusion(&self) -> Option<(&Path, f32)> {
        self.fusion
            .as_ref()
            .map(|f| (f.manifest.as_path(), f.weight))
    }

    /// The reading corrector's source (its manifest, or the bundled corrector's graph) and threshold, when there is one.
    pub fn corrector(&self) -> Option<(&Path, f32)> {
        self.corrector
            .as_ref()
            .map(|c| (c.manifest.as_path(), c.threshold))
    }

    fn build(
        manifest: &CtcNatManifest,
        explicit: &DecodeDefaults,
        env: &RuntimeEnv,
    ) -> Result<Self> {
        let (tokenizer, model) = artifact_load::load(manifest)?;
        let config = DecodeConfig::resolve(explicit, &manifest.decode_defaults, &env);
        if config.symbol_token {
            let mut buf = [0u8; 4];
            let token: &str = symtoken::SYMBOL_TOKEN.encode_utf8(&mut buf);
            // Without it the token reaches the model as [UNK] (or as bytes it
            // never saw as one); blacklisted, no run of symbols comes back.
            let Some(&id) = tokenizer.token_to_id.get(token) else {
                bail!(
                    "symbol_token is on but the vocabulary of {} has no {token} token",
                    manifest.manifest_path.display()
                );
            };
            if config.vocab_blacklist.contains(&id) {
                bail!(
                    "symbol_token is on but vocab_blacklist forbids the {token} token (id {id}); \
                     drop it from the blacklist"
                );
            }
        }
        let crf = manifest
            .crf
            .as_ref()
            .map(|(path, rank)| {
                crate::pipeline::crf::CrfTables::load(path, *rank, tokenizer.vocab_size())
            })
            .transpose()?;
        let rescorer = rescore::Rescorer::load(
            manifest.rescore_pairs.as_deref(),
            config.pair_weight,
            manifest.rescore_words.as_deref(),
            config.word_weight,
        )?;
        let fusion = match &env.fusion_manifest {
            None => None,
            Some(path) => Some(Self::load_fusion(path, env, &tokenizer, &model, manifest)?),
        };
        // SETSUNA_CORRECTOR=0: none; SETSUNA_CORRECTOR_MANIFEST (or the daemon's --corrector): that one (threshold
        // from the environment, else its manifest's decode_defaults.corrector_threshold, else +inf); else the
        // manifest's own (its threshold unless the environment gives one)
        let corrector = if env.corrector_off {
            None
        } else if let Some(path) = &env.corrector_manifest {
            let m = CtcNatManifest::load(path).with_context(|| {
                format!("load the reading corrector's manifest {}", path.display())
            })?;
            let threshold = env
                .corrector_threshold
                .or(m.decode_defaults.corrector_threshold)
                .unwrap_or(f32::INFINITY);
            Some(Self::load_corrector(&m, path, threshold, env)?)
        } else if let Some(b) = &manifest.corrector {
            let threshold = env.corrector_threshold.unwrap_or(b.threshold);
            // named by its graph, or by its native weights when the manifest has no graph
            let source = b
                .manifest
                .proposal_path
                .as_ref()
                .or(b.manifest.proposal_native_path.as_ref());
            Some(Self::load_corrector(
                &b.manifest,
                source.context("the bundled corrector has neither a graph nor native weights")?,
                threshold,
                env,
            )?)
        } else {
            None
        };
        Ok(Self {
            tokenizer,
            model,
            seq_len: manifest.max_seq_len,
            config,
            reduce: FrameReduce::default(),
            inspect: 0,
            chunk_cache: HashMap::new(),
            last_proposals: ProposalCache::default(),
            crf,
            rescorer,
            ar_margin: env.ar_margin,
            script_bias: env.script_bias,
            fusion,
            corrector,
        })
    }

    /// The reading corrector's session from its manifest (its own decode defaults; no fusion or corrector of its
    /// own): a CTC model. Its vocabulary need not be the conversion model's: what it gives is a reading. `source`
    /// names it (its manifest, or a bundled corrector's graph).
    fn load_corrector(
        corrector_manifest: &CtcNatManifest,
        source: &Path,
        threshold: f32,
        env: &RuntimeEnv,
    ) -> Result<Corrector> {
        let path = source;
        let corrector_env = RuntimeEnv {
            fusion_manifest: None,
            corrector_manifest: None,
            corrector_off: true,
            ..env.clone()
        };
        let session = Self::build(
            corrector_manifest,
            &DecodeDefaults::default(),
            &corrector_env,
        )
        .with_context(|| format!("load the reading corrector from {}", path.display()))?;
        if !matches!(session.model, Model::Ctc(_)) {
            bail!(
                "the reading corrector {} is not a CTC model (it has an AR decoder)",
                path.display()
            );
        }
        Ok(Corrector {
            session: Box::new(session),
            manifest: path.to_path_buf(),
            threshold,
            last: None,
        })
    }

    /// As a reading corrector: the greedy output for `reading` after `context` (each run of symbols one token and
    /// put back, with symbol token on), log P(output) and log P(`reading`) under this model over the same frames
    /// less one constant (`proposal::greedy_scores`, `ctc_score`; -inf where no alignment fits), and the graph's time
    /// in ms. `None` for a reading the model does not take in one pass, or with a token its logits leave out (a head
    /// cut to the tokens of readings, manifest `proposal.logit_ids`: it could not give that reading back).
    fn correct_reading(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<Option<(String, f32, f32, f64)>> {
        let cfg = &self.config;
        let tokenized = cfg.symbol_token.then(|| symtoken::tokenize(reading));
        let text = tokenized.as_ref().map_or(reading, |t| t.text.as_str());
        let Ok(encoded) = proposal::encode(
            &self.tokenizer,
            self.seq_len,
            cfg.max_context,
            context,
            text,
        ) else {
            return Ok(None);
        };
        // the reading's tokens: what follows [SEP]
        let (ids, actual) = (&encoded.0, encoded.2);
        let sep = ids[1..actual]
            .iter()
            .position(|&id| id == self.tokenizer.sep_id as i64)
            .map_or(actual, |i| i + 2);
        let typed: Vec<u32> = ids[sep..actual].iter().map(|&id| id as u32).collect();
        let Model::Ctc(runner) = &mut self.model else {
            bail!("a reading corrector is a CTC model");
        };
        if let Some(columns) = runner.logit_ids() {
            if typed.iter().any(|id| columns.binary_search(id).is_err()) {
                return Ok(None);
            }
        }
        let (path, scores) =
            runner.greedy_log_probs(&self.tokenizer, encoded, &cfg.vocab_blacklist, &typed)?;
        let forward = ctc_score::Forward::new(&scores);
        let (score_fix, score_typed) = (
            forward.log_likelihood(&path),
            forward.log_likelihood(&typed),
        );
        let raw = self.tokenizer.decode(&path);
        let corrected = tokenized.map_or(raw.clone(), |t| t.restore(&raw));
        Ok(Some((
            corrected,
            score_fix,
            score_typed,
            runner.last_ort_ms,
        )))
    }

    /// The fusion's CTC session from its manifest (its own decode defaults, no fusion of its own), checked against
    /// the AR model it scores: a CTC model, over the same vocabulary.
    fn load_fusion(
        path: &Path,
        env: &RuntimeEnv,
        tokenizer: &SharedCharTokenizer,
        model: &Model,
        manifest: &CtcNatManifest,
    ) -> Result<Fusion> {
        if !matches!(model, Model::Ar(_)) {
            bail!(
                "SETSUNA_FUSION_MANIFEST scores an AR model's candidates, but {} has no AR decoder",
                manifest.manifest_path.display()
            );
        }
        let scorer_manifest = CtcNatManifest::load(path)
            .with_context(|| format!("load the fusion's manifest {}", path.display()))?;
        let scorer_env = RuntimeEnv {
            fusion_manifest: None,
            corrector_manifest: None,
            corrector_off: true,
            ..env.clone()
        };
        let scorer = Self::build(&scorer_manifest, &DecodeDefaults::default(), &scorer_env)
            .with_context(|| format!("load the fusion's model from {}", path.display()))?;
        if !matches!(scorer.model, Model::Ctc(_)) {
            bail!(
                "the fusion's model {} is not a CTC model (it has an AR decoder)",
                path.display()
            );
        }
        let (a, b) = (tokenizer, &scorer.tokenizer);
        anyhow::ensure!(
            a.id_to_token == b.id_to_token
                && (a.pad_id, a.unk_id, a.cls_id, a.sep_id, a.blank_id, a.mask_id)
                    == (b.pad_id, b.unk_id, b.cls_id, b.sep_id, b.blank_id, b.mask_id),
            "the fusion's model {} has another vocabulary than {} (a candidate's token ids must mean the same to both)",
            path.display(),
            manifest.manifest_path.display()
        );
        Ok(Fusion {
            scorer: Box::new(scorer),
            manifest: path.to_path_buf(),
            weight: env.fusion_weight,
        })
    }

    /// The CTC log-likelihood of each token sequence for `reading` after `context`, over the frames a conversion of
    /// them reads (`ctc_score`): -inf for a sequence no alignment fits. The fusion's scorer side; a CTC model only.
    /// Returns the scores and the graph's time in ms.
    fn ctc_log_likelihoods(
        &mut self,
        context: &str,
        reading: &str,
        sequences: &[&[u32]],
    ) -> Result<(Vec<f32>, f64)> {
        let cfg = &self.config;
        let encoded = proposal::encode(
            &self.tokenizer,
            self.seq_len,
            cfg.max_context,
            context,
            reading,
        )?;
        let Model::Ctc(runner) = &mut self.model else {
            bail!("only a CTC model scores token sequences by their alignments");
        };
        let mut ids: Vec<u32> = sequences.iter().flat_map(|s| s.iter().copied()).collect();
        ids.sort_unstable();
        ids.dedup();
        let lp = runner.token_log_probs(
            &self.tokenizer,
            encoded,
            &cfg.vocab_blacklist,
            &ids,
            self.reduce,
        )?;
        let forward = ctc_score::Forward::new(&lp);
        Ok((
            sequences
                .iter()
                .map(|s| forward.log_likelihood(s))
                .collect(),
            runner.last_ort_ms,
        ))
    }

    /// Convert one request: the pipeline's candidates, cut to the display
    /// limit. A reading longer than a chunk (`chunk::MAX_CHUNK`, or what the
    /// model's sequence holds less [CLS] and [SEP]) is converted chunk by
    /// chunk, a sentence after another. A reading
    /// carrying bracket / annotation symbols, or chars the vocabulary has no
    /// token for, goes through symbol split when it is enabled; with symbol
    /// token, every run of symbols is read as one token instead. With digit
    /// split, the runs of digits are cut out first and spliced back. A
    /// `continue_beam` request decodes with a wider beam and returns more
    /// candidates; which of them are new is the caller's to tell.
    ///
    /// With a reading corrector, the reading goes through it first: when it
    /// changes the reading, both readings are converted, the one it picks
    /// first (`pipeline::correct`), and their candidates merged, every
    /// candidate's reading positions on the reading requested; the perf is
    /// the first reading's with the corrector's and the other reading's time
    /// added.
    pub fn convert(&mut self, request: ConversionRequest<'_>) -> Result<CandidateSet> {
        if self.corrector.is_none() {
            return self.convert_reading(request);
        }
        let t0 = Instant::now();
        let (correction, ort_ms) = self.correction(request.context, request.reading)?;
        let corrector_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut set = if correction.changed {
            let (first, other) = if correction.applied {
                (correction.reading.as_str(), request.reading)
            } else {
                (request.reading, correction.reading.as_str())
            };
            let limit = self.config.max_candidates + CONTINUE_STEP * request.continue_beam;
            let mut set = self.convert_reading(ConversionRequest {
                reading: first,
                ..request
            })?;
            let mut other_set = self.convert_reading(ConversionRequest {
                reading: other,
                ..request
            })?;
            // positions on the reading as requested, the frontend's
            let corrected = if correction.applied {
                &mut set.candidates
            } else {
                &mut other_set.candidates
            };
            correct::to_typed_positions(
                corrected,
                &correct::typed_positions(request.reading, &correction.reading),
            );
            set.candidates = correct::merge(set.candidates, other_set.candidates, limit);
            set.perf.candidate_count = set.candidates.len();
            set.perf.other_reading_ms = other_set.perf.total_ms;
            set.perf.total_ms += other_set.perf.total_ms;
            set
        } else {
            self.convert_reading(request)?
        };
        set.perf.corrector_ms = corrector_ms;
        set.perf.corrector_ort_ms = ort_ms;
        set.perf.total_ms += corrector_ms;
        set.correction = Some(correction);
        Ok(set)
    }

    /// The reading corrector's verdict on `reading` after `context` (the last one again when asked for once more)
    /// and its graph's time in ms (0 when kept). A session with a corrector only.
    fn correction(&mut self, context: &str, reading: &str) -> Result<(correct::Correction, f64)> {
        let c = self
            .corrector
            .as_mut()
            .expect("a session with a reading corrector");
        if let Some((ctx, rd, kept)) = &c.last {
            if ctx == context && rd == reading {
                return Ok((kept.clone(), 0.0));
            }
        }
        let (correction, ort_ms) = match c.session.correct_reading(context, reading)? {
            None => (correct::Correction::unchanged(reading), 0.0),
            Some((corrected, score_fix, score_typed, ort_ms)) => (
                correct::Correction::judged(
                    reading,
                    corrected,
                    score_fix,
                    score_typed,
                    c.threshold,
                ),
                ort_ms,
            ),
        };
        c.last = Some((context.to_string(), reading.to_string(), correction.clone()));
        Ok((correction, ort_ms))
    }

    /// [`Self::convert`] of the reading as given.
    fn convert_reading(&mut self, request: ConversionRequest<'_>) -> Result<CandidateSet> {
        if request.continue_beam == 0 {
            return self.convert_at(request);
        }
        let saved = self.config.clone();
        let extra = CONTINUE_STEP * request.continue_beam;
        self.config.beam_width += extra;
        self.config.max_candidates += extra;
        let out = self.convert_at(request);
        self.config = saved;
        out
    }

    fn convert_at(&mut self, request: ConversionRequest<'_>) -> Result<CandidateSet> {
        let max_chunk = chunk::MAX_CHUNK.min(self.seq_len.saturating_sub(2));
        let chunks = chunk::split_with_tokens(
            request.reading,
            max_chunk,
            self.seq_len.saturating_sub(2),
            self.config.symbol_token,
            self.config.digit_split,
            |c| {
                let in_vocab = self.tokenizer.has_char(c);
                if self.config.symbol_split
                    && !self.config.symbol_token
                    && (symsplit::is_strip_symbol(c) || !in_vocab)
                {
                    0
                } else if in_vocab {
                    1
                } else {
                    // as the input encodes it: one [UNK] or one token a byte
                    let mut buf = [0u8; 4];
                    self.tokenizer
                        .encode_with_byte_fallback(c.encode_utf8(&mut buf))
                        .len()
                }
            },
        )?;
        let (mut candidates, mut perf, frames) = if chunks.len() > 1 {
            let total_t0 = Instant::now();
            let pool_limit = self.config.pool_limit();
            let candidates =
                chunk::convert_chunks(request.context, &chunks, pool_limit, |c, r| {
                    self.convert_chunk(c, r)
                })?;
            let perf = ConvertPerf {
                total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
                candidate_count: candidates.len(),
                hyp_count: candidates.len(),
                ..ConvertPerf::default()
            };
            (candidates, perf, None)
        } else {
            self.convert_piece(request.context, request.reading)?
        };
        if let Some(r) = &self.rescorer {
            let t0 = Instant::now();
            r.rescore(request.context, request.reading, &mut candidates);
            perf.rescore_ms = t0.elapsed().as_secs_f64() * 1000.0;
            perf.total_ms += perf.rescore_ms;
        }
        candidates.truncate(self.config.max_candidates);
        perf.candidate_count = candidates.len();
        Ok(CandidateSet {
            candidates,
            perf,
            frames,
            correction: None,
        })
    }

    /// A chunk's candidates. Remembered: while a long reading is typed, the
    /// chunks before the last keep their reading and context.
    fn convert_chunk(&mut self, context: &str, reading: &str) -> Result<Vec<Candidate>> {
        let key = (
            context.to_string(),
            reading.to_string(),
            self.config.beam_width,
        );
        if let Some(hit) = self.chunk_cache.get(&key) {
            return Ok(hit.clone());
        }
        let (candidates, _, _) = self.convert_piece(context, reading)?;
        if self.chunk_cache.len() >= CHUNK_CACHE {
            self.chunk_cache.clear();
        }
        self.chunk_cache.insert(key, candidates.clone());
        Ok(candidates)
    }

    /// A reading the model takes in one sequence. With digit split and a
    /// digit in it: its runs of digits spliced back verbatim and every other
    /// segment through [`Self::convert_segment`], after the context and the
    /// output so far (so the digits reach the model only as context; training
    /// readings never hold them). Else the reading through
    /// [`Self::convert_segment`]. Candidates up to the pool limit.
    fn convert_piece(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        if !(self.config.digit_split && digitsplit::has_digit(reading)) {
            return self.convert_segment(context, reading);
        }
        let total_t0 = Instant::now();
        let pool_limit = self.config.pool_limit();
        let candidates = digitsplit::convert(context, reading, pool_limit, |c, r| {
            self.convert_segment(c, r)
                .map(|(candidates, _, _)| candidates)
        })?;
        let perf = ConvertPerf {
            total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            candidate_count: candidates.len(),
            hyp_count: candidates.len(),
            ..ConvertPerf::default()
        };
        Ok((candidates, perf, None))
    }

    /// A digit-free segment (or a reading, without digit split): with symbol
    /// token, one pass over it with its runs of symbols as tokens; else
    /// through symbol split when it carries split chars, else one pass of
    /// the pipeline. Candidates up to the pool limit.
    fn convert_segment(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        if self.config.symbol_token {
            return self.convert_tokenized(context, reading);
        }
        let out_of_vocab: HashSet<char> = reading
            .chars()
            .filter(|&c| !self.tokenizer.has_char(c))
            .collect();
        // Only [UNK] chars leave the context: one the vocabulary spells with
        // byte-fallback tokens is still read (a model with the byte tokens spells ~ that
        // way, and dropping it turned 東京~大阪間 into 東京~大阪館).
        let dropped_from_context: HashSet<char> = out_of_vocab
            .iter()
            .copied()
            .filter(|&c| self.tokenizer.is_unknown(c))
            .collect();
        let is_split = move |c: char| symsplit::is_strip_symbol(c) || out_of_vocab.contains(&c);
        if self.config.symbol_split && reading.chars().any(&is_split) {
            let total_t0 = Instant::now();
            let pool_limit = self.config.pool_limit();
            let candidates =
                symsplit::convert_split(context, reading, pool_limit, is_split, |c, r| {
                    // A spliced char the vocabulary lacks would reach the
                    // model as [UNK] through the running context: with
                    // a model with the byte tokens minus `.`, 2ぱーせんと after `6.`
                    // gave 2ーセン%, after `6` 2パーセント.
                    let c: String = c
                        .chars()
                        .filter(|ch| !dropped_from_context.contains(ch))
                        .collect();
                    self.convert_span(&c, r)
                        .map(|(candidates, _, _)| candidates)
                })?;
            let perf = ConvertPerf {
                total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
                candidate_count: candidates.len(),
                hyp_count: candidates.len(),
                ..ConvertPerf::default()
            };
            Ok((candidates, perf, None))
        } else {
            self.convert_span(context, reading)
        }
    }

    /// Symbol token: one pass of the pipeline over the reading with each run
    /// of symbols as one token (the context as typed), and the runs put back
    /// into every candidate (`symtoken::Tokenized::restore_candidates`: a
    /// token past the runs dropped, the reading positions moved to the
    /// reading, duplicates after restoring dropped). The frames' reading
    /// indices are moved to the reading too; their inputs stay the tokens
    /// the model read. Run even on a reading without symbols, so a stray
    /// token in an output is dropped.
    fn convert_tokenized(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        let tokenized = symtoken::tokenize(reading);
        let (candidates, mut perf, frames) = self.convert_span(context, &tokenized.text)?;
        let candidates = tokenized.restore_candidates(&candidates);
        perf.candidate_count = candidates.len();
        let frames = frames.map(|frames| {
            frames
                .into_iter()
                .map(|mut f| {
                    f.reading_index = f.reading_index.and_then(|i| tokenized.origin(i));
                    f
                })
                .collect()
        });
        Ok((candidates, perf, frames))
    }

    /// One pass of the pipeline over a symbol-free reading: proposal, decode,
    /// candidates.
    fn convert_span(
        &mut self,
        context: &str,
        reading: &str,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        let total_t0 = Instant::now();
        if reading.is_empty() {
            return Ok((Vec::new(), ConvertPerf::default(), None));
        }
        if matches!(self.model, Model::Ar(_)) {
            return self.convert_span_ar(context, reading, total_t0);
        }
        let cfg = &self.config;
        let greedy = cfg.beam_width <= 1;
        // Inspection needs probabilities even when decoding is greedy.
        let greedy_only = greedy && self.inspect == 0;

        let proposal_t0 = Instant::now();
        let top_k = cfg.top_k_per_step.max(self.inspect.saturating_add(1));
        self.last_proposals
            .settings(cfg.max_context, top_k, &cfg.vocab_blacklist);
        let (ort_ms, reduce_ms) = if self
            .last_proposals
            .get(context, reading, greedy_only, self.inspect)
            .is_some()
        {
            (0.0, 0.0)
        } else {
            let encoded = proposal::encode(
                &self.tokenizer,
                self.seq_len,
                cfg.max_context,
                context,
                reading,
            )?;
            let frame_reading =
                proposal::frame_reading(&self.tokenizer, &encoded.0, encoded.2, reading);
            let inputs: Vec<i64> = encoded.0[..encoded.2].to_vec();
            let Model::Ctc(runner) = &mut self.model else {
                unreachable!("an AR model returned above")
            };
            let proposal = runner.run(
                &self.tokenizer,
                encoded,
                &cfg.vocab_blacklist,
                top_k,
                self.reduce,
                greedy_only,
            )?;
            let (ort_ms, reduce_ms) = (runner.last_ort_ms, runner.last_reduce_ms);
            self.last_proposals.insert(LastProposal {
                context: context.to_string(),
                reading: reading.to_string(),
                greedy: greedy_only,
                inspect: self.inspect,
                proposal,
                frame_reading,
                inputs,
            });
            (ort_ms, reduce_ms)
        };
        let last = self.last_proposals.mru().expect("stored or promoted above");
        let (proposal, frame_reading, inputs) = (&last.proposal, &last.frame_reading, &last.inputs);
        let proposal_ms = proposal_t0.elapsed().as_secs_f64() * 1000.0;

        let beam_t0 = Instant::now();
        let hyps = decode::hypotheses(
            proposal,
            self.tokenizer.blank_id,
            cfg.beam_width,
            cfg.top_k_per_step,
            self.crf.as_ref(),
        );
        let beam_ms = beam_t0.elapsed().as_secs_f64() * 1000.0;
        let hyp_count = hyps.len();

        let post_t0 = Instant::now();
        let mut candidates = pool::candidates(
            hyps,
            greedy,
            proposal,
            reading,
            &self.tokenizer,
            frame_reading,
            cfg.normalize_candidates,
        );
        candidates.truncate(cfg.pool_limit());
        let perf = ConvertPerf {
            input_len: proposal.input_len,
            proposal_ms,
            ort_ms,
            reduce_ms,
            beam_ms,
            post_ms: post_t0.elapsed().as_secs_f64() * 1000.0,
            rescore_ms: 0.0,
            fusion_ms: 0.0,
            fusion_ort_ms: 0.0,
            corrector_ms: 0.0,
            corrector_ort_ms: 0.0,
            other_reading_ms: 0.0,
            total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            hyp_count,
            candidate_count: candidates.len(),
        };
        let frames =
            (self.inspect > 0).then(|| self.frame_views(&proposal.steps, inputs, frame_reading));
        Ok((candidates, perf, frames))
    }

    /// [`Self::convert_span`] for a model with an AR decoder (`pipeline::ar`): the encoder graph once, then the
    /// beam over the step graph (`beam_width` live hypotheses, `top_k_per_step` next tokens each, until the pool's
    /// worth closed), the hypotheses decoded best first (empty and duplicate texts dropped; the reading and its
    /// katakana when nothing is left). The special ids but the end and the blacklist are never written. No frames to
    /// inspect, no reading positions; the perf's proposal is the encoder, its beam the search.
    fn convert_span_ar(
        &mut self,
        context: &str,
        reading: &str,
        total_t0: Instant,
    ) -> Result<(Vec<Candidate>, ConvertPerf, Option<Vec<FrameView>>)> {
        let cfg = &self.config;
        let tok = &self.tokenizer;
        let encoded = proposal::encode(tok, self.seq_len, cfg.max_context, context, reading)?;
        let input_len = encoded.2;
        let Model::Ar(runner) = &mut self.model else {
            unreachable!("called for AR models")
        };
        let mut forbidden: Vec<u32> = vec![
            tok.pad_id,
            tok.unk_id,
            tok.cls_id,
            tok.blank_id,
            tok.mask_id,
        ];
        forbidden.extend(cfg.vocab_blacklist.iter().copied());
        let search = ar::ArSearch {
            bos: runner.spec.bos,
            eos: runner.spec.eos,
            beam: cfg.beam_width,
            top_k: cfg.top_k_per_step,
            // as the model's own greedy decode (ar_generate): at most the input's length
            max_len: input_len,
            want: cfg.pool_limit(),
            forbidden,
            margin: self.ar_margin,
            script_bias: self.script_bias,
        };
        let mut hyps = runner.run(encoded, &search)?;
        let (encoder_ms, search_ms) = (runner.last_encoder_ms, runner.last_search_ms);
        // the fusion: each hypothesis scored by the CTC model too, the hypotheses ordered by the sum (a stable sort:
        // equal sums keep the AR order)
        let (mut fusion_ms, mut fusion_ort_ms) = (0.0, 0.0);
        let mut parts: Vec<Option<ScoreParts>> = vec![None; hyps.len()];
        if let Some(f) = self.fusion.as_mut() {
            let t0 = Instant::now();
            let sequences: Vec<&[u32]> = hyps.iter().map(|h| h.ids.as_slice()).collect();
            let (ctc, ort_ms) = f.scorer.ctc_log_likelihoods(context, reading, &sequences)?;
            let mut fused: Vec<(ar::ArHypothesis, ScoreParts)> = hyps
                .into_iter()
                .zip(ctc)
                .map(|(mut h, ctc)| {
                    let p = ScoreParts { ar: h.score, ctc };
                    // weight 0: the AR order (0 * -inf would be NaN); a hypothesis
                    // the CTC model cannot align (-inf) goes last whatever the
                    // weight's sign (a negative one made it +inf, first)
                    if f.weight != 0.0 {
                        h.score = if ctc.is_finite() {
                            h.score + f.weight * ctc
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    if h.score.is_nan() {
                        h.score = f32::NEG_INFINITY;
                    }
                    (h, p)
                })
                .collect();
            fused.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
            parts = fused.iter().map(|x| Some(x.1)).collect();
            hyps = fused.into_iter().map(|x| x.0).collect();
            fusion_ms = t0.elapsed().as_secs_f64() * 1000.0;
            fusion_ort_ms = ort_ms;
        }
        let post_t0 = Instant::now();
        let source = if cfg.beam_width <= 1 {
            CandidateSource::Greedy
        } else {
            CandidateSource::Beam
        };
        let hyp_count = hyps.len();
        let mut seen: HashSet<String> = HashSet::new();
        let mut candidates: Vec<Candidate> = hyps
            .into_iter()
            .zip(parts)
            .filter_map(|(h, parts)| {
                let text = pool::ar_text(tok, &h.ids, cfg.normalize_candidates);
                (!text.is_empty() && seen.insert(text.clone())).then(|| Candidate {
                    text,
                    score: h.score,
                    source,
                    reading_pos: Vec::new(),
                    parts,
                })
            })
            .collect();
        if candidates.is_empty() {
            candidates = pool::fallback(reading);
        }
        candidates.truncate(cfg.pool_limit());
        let perf = ConvertPerf {
            input_len,
            proposal_ms: encoder_ms,
            ort_ms: encoder_ms,
            reduce_ms: 0.0,
            beam_ms: search_ms,
            post_ms: post_t0.elapsed().as_secs_f64() * 1000.0,
            rescore_ms: 0.0,
            fusion_ms,
            fusion_ort_ms,
            corrector_ms: 0.0,
            corrector_ort_ms: 0.0,
            other_reading_ms: 0.0,
            total_ms: total_t0.elapsed().as_secs_f64() * 1000.0,
            hyp_count,
            candidate_count: candidates.len(),
        };
        Ok((candidates, perf, None))
    }

    /// The proposal's frames as [`FrameView`]s; `inputs` are the encoded
    /// ids, one per frame.
    fn frame_views(
        &self,
        steps: &[proposal::FrameLogProbs],
        inputs: &[i64],
        frame_reading: &[u32],
    ) -> Vec<FrameView> {
        let tok = &self.tokenizer;
        // reading frames start after the first [SEP]
        let reading_from = inputs
            .iter()
            .skip(1)
            .position(|&id| id == tok.sep_id as i64)
            .map_or(inputs.len(), |i| i + 2);
        let name = |id: u32| {
            tok.id_to_token
                .get(id as usize)
                .cloned()
                .unwrap_or_default()
        };
        steps
            .iter()
            .zip(inputs)
            .enumerate()
            .map(|(f, (step, &input))| FrameView {
                input: name(input as u32),
                reading_index: (f >= reading_from)
                    .then(|| frame_reading.get(f).copied())
                    .flatten(),
                blank: step.blank.exp(),
                top: step
                    .top
                    .iter()
                    .filter(|(id, _)| *id != tok.blank_id)
                    .take(self.inspect)
                    .map(|&(id, lp)| (name(id), lp.exp()))
                    .collect(),
            })
            .collect()
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Four finite numbers separated by commas (`RuntimeEnv::script_bias`).
fn parse_script_bias(s: &str) -> Result<[f32; 4]> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse::<f32>().ok().filter(|x| x.is_finite()))
        .collect::<Option<_>>()
        .context("not numbers")?;
    v.try_into()
        .map_err(|v: Vec<f32>| anyhow::anyhow!("{} numbers, not 4", v.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The daemon's `--corrector-threshold` over the environment's, and only the one taken checked.
    #[test]
    fn the_given_corrector_threshold_wins_over_the_environments() {
        assert_eq!(corrector_threshold_of(None, None).unwrap(), None);
        assert_eq!(
            corrector_threshold_of(None, Some("9.5")).unwrap(),
            Some(9.5)
        );
        assert_eq!(
            corrector_threshold_of(Some(4.0), Some("9.5")).unwrap(),
            Some(4.0)
        );
        // an environment value that is not a number stops the engine only when it is the one taken
        assert_eq!(
            corrector_threshold_of(Some(4.0), Some("bogus")).unwrap(),
            Some(4.0)
        );
        assert!(corrector_threshold_of(None, Some("bogus")).is_err());
        assert!(corrector_threshold_of(None, Some("nan")).is_err());
        assert!(corrector_threshold_of(Some(f32::NAN), None).is_err());
        assert_eq!(
            corrector_threshold_of(Some(f32::NEG_INFINITY), None).unwrap(),
            Some(f32::NEG_INFINITY)
        );
    }

    fn env() -> RuntimeEnv {
        RuntimeEnv {
            top_k_per_step: 16,
            ar_margin: None,
            script_bias: [0.0; 4],
            fusion_manifest: None,
            fusion_weight: 1.0,
            corrector_manifest: None,
            corrector_threshold: None,
            corrector_off: false,
        }
    }

    /// SETSUNA_SCRIPT_BIAS: four numbers in class order, spaces allowed; other counts, words and non-finite
    /// numbers refused.
    #[test]
    fn script_bias_is_four_numbers() {
        assert_eq!(parse_script_bias("0,0,4,0").unwrap(), [0.0, 0.0, 4.0, 0.0]);
        assert_eq!(
            parse_script_bias(" -1, 2.5 ,0,0 ").unwrap(),
            [-1.0, 2.5, 0.0, 0.0]
        );
        for bad in ["0,0,4", "0,0,4,0,0", "hiragana=4", "0,0,inf,0", ""] {
            assert!(parse_script_bias(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn explicit_beats_manifest_beats_default() {
        let explicit = DecodeDefaults {
            beam_width: Some(8),
            ..DecodeDefaults::default()
        };
        let manifest = DecodeDefaults {
            beam_width: Some(20),
            top_k: Some(10),
            symbol_split: Some(true),
            ..DecodeDefaults::default()
        };
        let cfg = DecodeConfig::resolve(&explicit, &manifest, &env());
        assert_eq!(cfg.beam_width, 8);
        assert_eq!(cfg.max_candidates, 10);
        assert!(cfg.symbol_split);
        assert!(!cfg.symbol_token);
        assert_eq!(cfg.max_context, 64);
        assert_eq!(cfg.top_k_per_step, 16);
        assert_eq!(cfg.pool_limit(), 10);
        assert_eq!((cfg.pair_weight, cfg.word_weight), (0.0, 0.0));
    }

    #[test]
    fn symbol_token_resolves_like_symbol_split() {
        let on = DecodeDefaults {
            symbol_token: Some(true),
            ..DecodeDefaults::default()
        };
        let off = DecodeDefaults {
            symbol_token: Some(false),
            ..DecodeDefaults::default()
        };
        let none = DecodeDefaults::default();
        assert!(DecodeConfig::resolve(&none, &on, &env()).symbol_token);
        assert!(!DecodeConfig::resolve(&off, &on, &env()).symbol_token);
        assert!(DecodeConfig::resolve(&on, &none, &env()).symbol_token);
        assert!(!DecodeConfig::resolve(&none, &none, &env()).symbol_token);
    }

    #[test]
    fn digit_split_resolves_like_symbol_split() {
        let on = DecodeDefaults {
            digit_split: Some(true),
            ..DecodeDefaults::default()
        };
        let off = DecodeDefaults {
            digit_split: Some(false),
            ..DecodeDefaults::default()
        };
        let none = DecodeDefaults::default();
        assert!(DecodeConfig::resolve(&none, &on, &env()).digit_split);
        assert!(!DecodeConfig::resolve(&off, &on, &env()).digit_split);
        assert!(DecodeConfig::resolve(&on, &none, &env()).digit_split);
        assert!(!DecodeConfig::resolve(&none, &none, &env()).digit_split);
    }

    #[test]
    fn explicit_empty_blacklist_clears_the_manifest_one() {
        let explicit = DecodeDefaults {
            vocab_blacklist: Some(Vec::new()),
            ..DecodeDefaults::default()
        };
        let manifest = DecodeDefaults {
            vocab_blacklist: Some(vec![724]),
            ..DecodeDefaults::default()
        };
        assert!(DecodeConfig::resolve(&explicit, &manifest, &env())
            .vocab_blacklist
            .is_empty());
    }

    #[test]
    fn zero_widths_are_clamped() {
        let explicit = DecodeDefaults {
            beam_width: Some(0),
            top_k: Some(0),
            top_k_per_step: Some(0),
            ..DecodeDefaults::default()
        };
        let cfg = DecodeConfig::resolve(&explicit, &DecodeDefaults::default(), &env());
        assert_eq!(
            (cfg.beam_width, cfg.max_candidates, cfg.top_k_per_step),
            (1, 1, 1)
        );
        assert_eq!(cfg.pool_limit(), 1);
    }

    #[test]
    fn proposal_cache_keeps_two_readings_warm_lru() {
        let entry = |ctx: &str, rd: &str, greedy: bool, inspect: usize, len: usize| LastProposal {
            context: ctx.into(),
            reading: rd.into(),
            greedy,
            inspect,
            proposal: proposal::ProposalOutput {
                steps: Vec::new(),
                frames: Vec::new(),
                input_len: len,
            },
            frame_reading: Vec::new(),
            inputs: Vec::new(),
        };
        let mut cache = ProposalCache::default();
        cache.insert(entry("ctx", "first", false, 0, 1));
        cache.insert(entry("ctx", "other", false, 0, 2));
        // Alternating between the two readings hits both without eviction (reading corrector + continue_beam)
        assert_eq!(
            cache
                .get("ctx", "first", false, 0)
                .map(|p| p.proposal.input_len),
            Some(1)
        );
        assert_eq!(
            cache
                .get("ctx", "other", false, 0)
                .map(|p| p.proposal.input_len),
            Some(2)
        );
        // Greedy / inspect mismatches do not hit
        assert!(cache.get("ctx", "other", true, 0).is_none());
        assert!(cache.get("ctx", "other", false, 1).is_none());
        // Other settings empty it, the same ones keep it
        cache.settings(64, 16, &[]);
        cache.insert(entry("ctx", "other", false, 0, 2));
        cache.settings(64, 16, &[]);
        assert!(cache.get("ctx", "other", false, 0).is_some());
        cache.settings(0, 16, &[]);
        assert!(cache.get("ctx", "other", false, 0).is_none());
        cache.insert(entry("ctx", "other", false, 0, 2));
        cache.settings(0, 16, &[7]);
        assert!(cache.get("ctx", "other", false, 0).is_none());
        cache.insert(entry("ctx", "first", false, 0, 1));
        cache.insert(entry("ctx", "other", false, 0, 2));
        // Inserting a third reading evicts the LRU ("first"), keeping MRU ("other") and the new one ("third")
        cache.insert(entry("ctx", "third", false, 0, 3));
        assert!(cache.get("ctx", "first", false, 0).is_none());
        assert_eq!(
            cache
                .get("ctx", "other", false, 0)
                .map(|p| p.proposal.input_len),
            Some(2)
        );
        assert_eq!(
            cache
                .get("ctx", "third", false, 0)
                .map(|p| p.proposal.input_len),
            Some(3)
        );
    }
}
