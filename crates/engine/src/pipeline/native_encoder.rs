//! The encoder in Rust (the int8 manifest's `proposal.native`, the research export's native encoder): the int8head
//! graph's computation without ONNX Runtime, for one row of
//! ids (no padding), from a safetensors file of the graph's own weights.
//!
//! - The Linears are the graph's `neo_ime.LinearQ4G` and the experts its `neo_ime.TokenMoEQ4G`: the same kernels
//!   (`group_q`) on the same codes, scales and zero points. The LayerNorms are its `neo_ime.AddLayerNorm` (`norm_op`).
//! - The tables ONNX Runtime's dynamic quantization made 8-bit are dequantized as its DequantizeLinear does,
//!   (q - zero) * scale: the token embedding or the tied CTC head's rows, the position tables, the PLE table.
//! - The PLE rows are hashed as the graph hashes them (the research side's PLE hash, v2): u64 products
//!   xor-ed over the window ending at the frame, the slot's salt, SplitMix64, then modulo the slot's table. Positions
//!   before the row's start read PLE_BOS. The relative position biases are read by the frames' segments (after the
//!   first [SEP]: 1, else 0) and the bucket of their distance.
//! - Attention is com.microsoft.MultiHeadAttention's: softmax(q k^T / sqrt(head_dim) + bias) v. GELU is exact.
//! - The 8-bit products at the end are the graph's DynamicQuantizeLinear + MatMulInteger: the CTC head, or the AR
//!   layers' cross-attention keys and values. The activations become u8 codes over the whole tensor (one scale and
//!   zero point, round half to even), then integer sums, times the two scales.
//!
//! Outputs: "ctc", the NAT decoder's frames through the CTC head, as logits (T, V); "ar_cross", the AR layers'
//! cross-attention keys and values, each (layers, 1, heads, T, head_dim) as the encoder graph gives them.

// the x86 kernels and what only they read are unused elsewhere
#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::group_q::{group_moe, GroupMoe, GroupMoeScratch, GroupScratch, GroupWeights};
use crate::moe_op::{detect_isa, gelu, gemm_u8i8_f32, Epilogue, Isa, Packed};
use crate::norm_op::add_layer_norm_here;
use crate::pipeline::ar_native::{lane_sum, parse_safetensors, row_max, Weight, Weights};
use crate::pipeline::lse::exp;

/// The file's `__metadata__.format`.
pub const ENCODER_FORMAT: &str = "neo-ime-encoder-v1";

/// The id a PLE window reads before the row's start (the research side's PLE_BOS).
const PLE_BOS: u64 = 0xFFFF_FFFE;

/// The encoder's shapes and constants (the file's `__metadata__.config`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocab: usize,
    /// rows of the position tables and of the relative buckets' map: the longest input
    pub max_positions: usize,
    pub norm_eps: f32,
    pub sep_id: u32,
    /// "tied": the input rows are the CTC head's (`head.*`); "own": the file's `embedding.*`
    pub embedding: String,
    /// the position table's rows added to the input rows (`positions.*`), counted from the row's start
    pub absolute_positions: bool,
    /// the relative position biases' buckets (`relative.*`), or none
    #[serde(default)]
    pub relative_buckets: Option<usize>,
    #[serde(default)]
    pub ple: Option<PleConfig>,
    #[serde(default)]
    pub moe: Option<MoeConfig>,
    /// "ctc" (`decoder` and `head.*`) or "ar_cross" (`ar` and `ar.*`)
    pub output: String,
    /// The CTC head's rows when it holds some of the vocabulary's tokens alone (a reading corrector's head cut to the
    /// manifest's `proposal.logit_ids`, in their order; its input rows then the file's own `embedding.*`); none: one
    /// a token.
    #[serde(default)]
    pub head_rows: Option<usize>,
    #[serde(default)]
    pub decoder: Option<DecoderConfig>,
    #[serde(default)]
    pub ar: Option<ArCrossConfig>,
}

/// The per-layer PLE: the hash of every slot (an n-gram order and a head), the table, and per layer a gate and a
/// projection (`ple.gate.<i>`, `ple.proj.<i>`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PleConfig {
    /// the table's width
    pub dim: usize,
    /// the v2 finish (salt and SplitMix64) before the modulo
    pub finalize: bool,
    /// each slot's n-gram order
    pub orders: Vec<usize>,
    /// each slot's table rows and their first row in the table
    pub table_sizes: Vec<u64>,
    pub row_offsets: Vec<u64>,
    /// each slot's multipliers over the window (the longest order wide, the frame last; a slot of a shorter order
    /// reads the window's tail)
    pub multipliers: Vec<Vec<u64>>,
    pub salts: Vec<u64>,
    /// the table's rows
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeConfig {
    /// the encoder layers with routed experts (`layers.<i>.moe.*`)
    pub layers: Vec<usize>,
    pub experts: usize,
    pub top_k: usize,
}

/// The NAT decoder on the encoder's frames (pre-norm; `decoder.*`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderConfig {
    pub layers: usize,
    pub heads: usize,
    pub absolute_positions: bool,
    /// the self- and cross-attention's relative position biases (`decoder.relative.*`)
    pub relative: bool,
}

/// The AR decoder's cross-attention keys and values (`ar.layers.<l>.{k,v}.*`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArCrossConfig {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
}

/// An 8-bit table as ONNX Runtime's DequantizeLinear reads it: row r is (q[r] - zero) * scale.
struct Table8 {
    q: Vec<i16>,
    zero: i16,
    scale: f32,
    cols: usize,
}

impl Table8 {
    fn load(w: &Weights, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let scale = w.f32(&format!("{name}.scale"), &[1])?[0];
        let (q, zero): (Vec<i16>, i16) = match (
            w.0.get(&format!("{name}.q")),
            w.0.get(&format!("{name}.zero")),
        ) {
            (Some(Weight::U8(q, s)), Some(Weight::U8(z, zs)))
                if s[..] == [rows, cols] && zs[..] == [1] =>
            {
                (q.iter().map(|&v| v as i16).collect(), z[0] as i16)
            }
            (Some(Weight::I8(q, s)), Some(Weight::I8(z, zs)))
                if s[..] == [rows, cols] && zs[..] == [1] =>
            {
                (q.iter().map(|&v| v as i16).collect(), z[0] as i16)
            }
            _ => anyhow::bail!(
                "{name}: q ({rows}, {cols}) and zero (1), both uint8 or both int8, expected"
            ),
        };
        Ok(Table8 {
            q,
            zero,
            scale,
            cols,
        })
    }

    /// out = row r.
    fn row(&self, r: usize, out: &mut [f32]) {
        for (o, &v) in out
            .iter_mut()
            .zip(&self.q[r * self.cols..(r + 1) * self.cols])
        {
            *o = (v - self.zero) as f32 * self.scale;
        }
    }

    /// out += row r.
    fn add_row(&self, r: usize, out: &mut [f32]) {
        for (o, &v) in out
            .iter_mut()
            .zip(&self.q[r * self.cols..(r + 1) * self.cols])
        {
            *o += (v - self.zero) as f32 * self.scale;
        }
    }
}

/// A `neo_ime.LinearQ4G` (`<name>.w` (K / g, N, g / 2) codes, `<name>.scales` / `<name>.zeros` (N, K / g)) and the
/// Add of its bias after it (`<name>.bias`, when the graph has one).
struct Q4g {
    /// the codes `w` points into (a heap buffer that does not move with the struct)
    _codes: Vec<u8>,
    w: GroupWeights,
    bias: Option<Vec<f32>>,
    n: usize,
    k: usize,
}

impl Q4g {
    fn load(w: &Weights, name: &str, bias: bool) -> Result<Self> {
        let (codes, shape) = w.u8_any(&format!("{name}.w"))?;
        anyhow::ensure!(
            shape.len() == 3,
            "{name}.w is {shape:?}, expected (K / g, N, g / 2)"
        );
        let (groups, n, g) = (shape[0], shape[1], shape[2] * 2);
        let scales = w.f32(&format!("{name}.scales"), &[n, groups])?;
        let zeros = w.f32(&format!("{name}.zeros"), &[n, groups])?;
        let codes = codes.to_vec();
        // SAFETY: `codes` lives in the struct with `w`, unchanged; its heap buffer stays put when the struct moves
        let gw = unsafe { GroupWeights::in_place(&codes, scales, zeros, n, groups * g, g) }
            .map_err(|m| anyhow::anyhow!("{name}: {m}"))?;
        let bias = if bias {
            Some(w.f32(&format!("{name}.bias"), &[n])?.to_vec())
        } else {
            None
        };
        Ok(Q4g {
            _codes: codes,
            w: gw,
            bias,
            n,
            k: groups * g,
        })
    }

    /// y (rows, N) of x (rows, K), the bias added (by the kernel).
    fn forward(&self, x: &[f32], y: &mut Vec<f32>, s: &mut GroupScratch) {
        let rows = x.len() / self.k;
        y.resize(rows * self.n, 0.0);
        self.w.forward_bias(x, y, self.bias.as_deref(), s);
    }
}

/// A LayerNorm's weight and bias.
struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(w: &Weights, name: &str, d: usize) -> Result<Self> {
        Ok(Norm {
            g: w.f32(&format!("{name}.weight"), &[d])?.to_vec(),
            b: w.f32(&format!("{name}.bias"), &[d])?.to_vec(),
        })
    }
}

/// An int8 matrix with one scale (ONNX Runtime's dynamically quantized MatMul's weight), its rows the outputs, and
/// the Add of a bias after it or none.
struct Lin8 {
    w: Packed,
    scale: f32,
    bias: Option<Vec<f32>>,
    out: usize,
}

impl Lin8 {
    /// `name`.q (out, inp) int8, `.scale` (1), `.zero` (1) int8 and 0, `.bias` (out) when `bias`.
    fn load(isa: Isa, w: &Weights, name: &str, out: usize, inp: usize, bias: bool) -> Result<Self> {
        let q = w.i8(&format!("{name}.q"), &[out, inp])?;
        let zero = w.i8(&format!("{name}.zero"), &[1])?[0];
        anyhow::ensure!(
            zero == 0,
            "{name}: zero point {zero}, the engine takes symmetric int8 weights (0)"
        );
        let blocked = isa != Isa::Plain && inp % 4 == 0;
        let stride = if blocked { out.div_ceil(16) * 16 } else { out };
        let mut rows = q.to_vec();
        rows.resize(stride * inp, 0);
        Ok(Lin8 {
            w: Packed::new(&rows, stride, inp, blocked),
            scale: w.f32(&format!("{name}.scale"), &[1])?[0],
            bias: if bias {
                Some(w.f32(&format!("{name}.bias"), &[out])?.to_vec())
            } else {
                None
            },
            out,
        })
    }

    /// y (rows, out) of the activations' codes `q` (rows, inp) with their one scale and zero point: the graph's
    /// MatMulInteger, its Cast and its Mul by the two scales' product, then the bias.
    fn forward(
        &self,
        isa: Isa,
        q: &[u8],
        rows: usize,
        (sx, zx): (f32, i32),
        acc: &mut Vec<i32>,
        y: &mut Vec<f32>,
    ) {
        y.resize(rows * self.out, 0.0);
        // (acc - zx cs) * (sx scale), then the bias: in the GEMM's registers with AVX-512 VNNI (`gemm_u8i8_f32`)
        let e = Epilogue {
            xs: &[(sx, zx)],
            col: None,
            scale: self.scale,
            bias: self.bias.as_deref(),
            out: self.out,
        };
        gemm_u8i8_f32(isa, q, rows, &self.w, &e, acc, y);
    }
}

/// ONNX Runtime's DynamicQuantizeLinear: u8 codes of the whole of `x` with one scale and zero point, the range
/// widened to hold 0, (max - min) / 255, the zero point and the codes rounded half to even.
fn quantize_tensor(x: &[f32], q: &mut Vec<u8>) -> (f32, i32) {
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for &v in x {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let scale = if hi == lo { 1.0 } else { (hi - lo) / 255.0 };
    let zp = (0.0 - lo / scale).clamp(0.0, 255.0).round_ties_even();
    let (lo_c, hi_c) = (0.0 - zp, 255.0 - zp);
    q.clear();
    q.extend(
        x.iter()
            .map(|&v| ((v / scale).clamp(lo_c, hi_c).round_ties_even() + zp) as u8),
    );
    (scale, zp as i32)
}

/// The relative position biases of an attention stack: the table (SEGMENTS^2 x buckets, heads) with its heads (the
/// stack's attentions' heads side by side), the bucket of every frame pair (max_positions^2).
struct Relative {
    table: Vec<f32>,
    table_heads: usize,
    buckets: usize,
    bucket: Vec<u8>,
    positions: usize,
}

impl Relative {
    fn load(
        w: &Weights,
        name: &str,
        buckets: usize,
        heads: usize,
        positions: usize,
    ) -> Result<Self> {
        let (bucket, shape) = w.u8_any(&format!("{name}.bucket"))?;
        anyhow::ensure!(
            shape[..] == [positions, positions],
            "{name}.bucket is {shape:?}, expected ({positions}, {positions})"
        );
        anyhow::ensure!(
            bucket.iter().all(|&b| (b as usize) < buckets),
            "{name}.bucket: a bucket past {buckets}"
        );
        Ok(Relative {
            table: w
                .f32(&format!("{name}.table"), &[4 * buckets, heads])?
                .to_vec(),
            table_heads: heads,
            buckets,
            bucket: bucket.to_vec(),
            positions,
        })
    }

    /// The bias (heads, T, T) of the table's heads `first..first + heads` for frames of segments `seg`.
    fn bias(&self, seg: &[usize], first: usize, heads: usize, out: &mut Vec<f32>) {
        let t = seg.len();
        out.resize(heads * t * t, 0.0);
        for q in 0..t {
            for k in 0..t {
                let row = (seg[q] * 2 + seg[k]) * self.buckets
                    + self.bucket[q * self.positions + k] as usize;
                let src = &self.table
                    [row * self.table_heads + first..row * self.table_heads + first + heads];
                for (h, &v) in src.iter().enumerate() {
                    out[(h * t + q) * t + k] = v;
                }
            }
        }
    }
}

/// SplitMix64's output for state `x` (as the research side hashes).
fn splitmix64(x: u64) -> u64 {
    let mut x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Ple {
    c: PleConfig,
    max_order: usize,
    table: Table8,
    gates: Vec<Q4g>,
    projs: Vec<Q4g>,
}

impl Ple {
    /// The table rows of every frame and slot (T, slots).
    fn rows(&self, ids: &[u64], out: &mut Vec<usize>) {
        let (m, slots) = (self.max_order, self.c.orders.len());
        out.clear();
        for t in 0..ids.len() {
            for s in 0..slots {
                let mult = &self.c.multipliers[s];
                let mut mixed = 0u64;
                for j in m - self.c.orders[s]..m {
                    let tok = if t + j + 1 < m {
                        PLE_BOS
                    } else {
                        ids[t + j + 1 - m]
                    };
                    mixed ^= tok.wrapping_mul(mult[j]);
                }
                if self.c.finalize {
                    mixed = splitmix64(mixed ^ self.c.salts[s]);
                }
                out.push((mixed % self.c.table_sizes[s] + self.c.row_offsets[s]) as usize);
            }
        }
    }

    /// The lookup (T, dim): each frame's slot rows summed, slot after slot (the graph's ReduceSum).
    fn lookup(&self, rows: &[usize], t: usize, out: &mut Vec<f32>) {
        let (slots, dim) = (self.c.orders.len(), self.c.dim);
        out.clear();
        out.resize(t * dim, 0.0);
        for f in 0..t {
            let o = &mut out[f * dim..(f + 1) * dim];
            self.table.row(rows[f * slots], o);
            for s in 1..slots {
                self.table.add_row(rows[f * slots + s], o);
            }
        }
    }
}

struct EncoderLayer {
    qkv: Q4g,
    out: Q4g,
    norm1: Norm,
    ffn1: Q4g,
    ffn2: Q4g,
    norm2: Norm,
    moe: Option<(GroupMoe, Vec<Vec<u8>>)>,
}

struct DecoderLayer {
    self_norm: Norm,
    self_qkv: Q4g,
    self_out: Q4g,
    cross_norm: Norm,
    cross_q: Q4g,
    cross_out: Q4g,
    ffn_norm: Norm,
    ffn1: Q4g,
    ffn2: Q4g,
}

struct Decoder {
    c: DecoderConfig,
    /// every layer's cross-attention keys and values from the encoder's frames, columns [K_0 | V_0 | K_1 | ...]
    kv: Q4g,
    positions: Option<Table8>,
    relative: Option<Relative>,
    layers: Vec<DecoderLayer>,
    final_norm: Norm,
}

enum Output {
    Ctc {
        decoder: Decoder,
        head: Lin8,
    },
    ArCross {
        c: ArCrossConfig,
        kv: Vec<(Lin8, Lin8)>,
    },
}

/// Buffers reused across conversions.
#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    f: Vec<f32>,
    qkv: Vec<f32>,
    kv: Vec<f32>,
    bias: Vec<f32>,
    bias2: Vec<f32>,
    attn: AttentionScratch,
    rows: Vec<usize>,
    lookup: Vec<f32>,
    q8: Vec<u8>,
    acc: Vec<i32>,
    lin: GroupScratch,
    moe: GroupMoeScratch,
}

/// The parts of a conversion [`Profile`] times.
const PARTS: [&str; 16] = [
    "input",
    "ple_inject",
    "qkv",
    "attention",
    "out_norm1",
    "ffn",
    "moe",
    "norm2",
    "final_norm",
    "decoder_kv",
    "decoder_self",
    "decoder_cross",
    "decoder_ffn",
    "head_quantize",
    "head_gemm",
    "ar_cross",
];

/// SETSUNA_NATIVE_PROFILE=<path>: the time of each part of the computation ([`PARTS`]), summed over the conversions
/// and written to <path> as JSON every 256 conversions and when the encoder is dropped (a research measure).
struct Profile {
    path: PathBuf,
    ms: [f64; PARTS.len()],
    conversions: usize,
    frames: usize,
    last: Instant,
}

impl Profile {
    fn from_env() -> Option<Self> {
        let path = std::env::var_os("SETSUNA_NATIVE_PROFILE")?;
        Some(Profile {
            path: path.into(),
            ms: [0.0; PARTS.len()],
            conversions: 0,
            frames: 0,
            last: Instant::now(),
        })
    }

    /// The time since the last lap (or `start`) to `part`.
    fn lap(&mut self, part: usize) {
        let now = Instant::now();
        self.ms[part] += (now - self.last).as_secs_f64() * 1000.0;
        self.last = now;
    }

    fn write(&self) {
        let parts: serde_json::Map<String, serde_json::Value> = PARTS
            .iter()
            .zip(self.ms)
            .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect();
        let body = serde_json::json!({"conversions": self.conversions, "frames": self.frames, "ms": parts});
        // a research file: a failed write loses the measure, not the conversion
        let _ = std::fs::write(&self.path, body.to_string());
    }
}

/// `p.lap(part)` when profiling.
fn lap(p: &mut Option<Profile>, part: usize) {
    if let Some(p) = p.as_mut() {
        p.lap(part);
    }
}

/// The encoder's weights, prepacked.
pub struct NativeEncoder {
    pub config: EncoderConfig,
    profile: Option<Profile>,
    isa: Isa,
    /// the input rows: the token embedding, or the tied head's rows
    embedding: Table8,
    positions: Option<Table8>,
    relative: Option<Relative>,
    ple: Option<Ple>,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    output: Output,
    scratch: Scratch,
}

/// The file's config and tensors.
pub fn read(path: &Path) -> Result<(EncoderConfig, Weights)> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (config, weights) = parse_safetensors(&data, ENCODER_FORMAT)
        .with_context(|| format!("native encoder {}", path.display()))?;
    let config = serde_json::from_str(&config)
        .with_context(|| format!("native encoder {}: config", path.display()))?;
    Ok((config, weights))
}

impl NativeEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let (config, weights) = read(path)?;
        Self::new(config, &weights).with_context(|| format!("native encoder {}", path.display()))
    }

    pub fn new(config: EncoderConfig, w: &Weights) -> Result<Self> {
        let c = &config;
        let (d, p, v) = (c.hidden, c.max_positions, c.vocab);
        anyhow::ensure!(
            c.layers > 0 && c.heads > 0 && d % c.heads == 0 && v > 0 && p > 0,
            "config {c:?}: sizes must be positive and hidden a multiple of heads"
        );
        let rows = c.head_rows.unwrap_or(v);
        anyhow::ensure!(
            rows > 0 && rows <= v,
            "config {c:?}: head_rows must be 1 to the vocabulary's {v}"
        );
        let isa = detect_isa();
        let head = match (c.output.as_str(), &c.decoder, &c.ar) {
            ("ctc", Some(_), None) => Some(Lin8::load(isa, w, "head", rows, d, false)?),
            ("ar_cross", None, Some(_)) => None,
            (o, dec, ar) => anyhow::bail!(
                "output {o:?} with decoder {} and ar {}: \"ctc\" takes a decoder, \"ar_cross\" the AR layers",
                dec.is_some(),
                ar.is_some()
            ),
        };
        let embedding = match c.embedding.as_str() {
            "own" => Table8::load(w, "embedding", v, d)?,
            "tied" => {
                anyhow::ensure!(
                    c.output == "ctc" && rows == v,
                    "a tied embedding reads the CTC head's rows, every token's"
                );
                Table8::load(w, "head", v, d)?
            }
            other => anyhow::bail!("embedding {other:?}, expected \"own\" or \"tied\""),
        };
        let positions = if c.absolute_positions {
            Some(Table8::load(w, "positions", p, d)?)
        } else {
            None
        };
        let relative = match c.relative_buckets {
            Some(b) => Some(Relative::load(w, "relative", b, c.heads, p)?),
            None => None,
        };
        let ple = match &c.ple {
            None => None,
            Some(pc) => {
                let slots = pc.orders.len();
                let max_order = pc.orders.iter().copied().max().unwrap_or(0);
                anyhow::ensure!(
                    slots > 0
                        && [pc.table_sizes.len(), pc.row_offsets.len(), pc.multipliers.len(), pc.salts.len()]
                            .iter()
                            .all(|&n| n == slots)
                        && pc.multipliers.iter().all(|m| m.len() == max_order)
                        && pc.orders.iter().all(|&o| o >= 1)
                        && pc.table_sizes.iter().all(|&s| s > 0)
                        && pc.table_sizes.iter().zip(&pc.row_offsets).all(|(&s, &o)| (s + o) as usize <= pc.rows),
                    "PLE config: one order, size, offset, salt and max-order multipliers a slot, every table in the rows"
                );
                Some(Ple {
                    table: Table8::load(w, "ple.table", pc.rows, pc.dim)?,
                    gates: (0..c.layers)
                        .map(|i| Q4g::load(w, &format!("ple.gate.{i}"), true))
                        .collect::<Result<_>>()?,
                    projs: (0..c.layers)
                        .map(|i| Q4g::load(w, &format!("ple.proj.{i}"), false))
                        .collect::<Result<_>>()?,
                    max_order,
                    c: pc.clone(),
                })
            }
        };
        let moe_layers = c.moe.as_ref().map_or(&[][..], |m| &m.layers[..]);
        let layers = (0..c.layers)
            .map(|i| {
                let n = format!("layers.{i}");
                let moe = if moe_layers.contains(&i) {
                    let mc = c.moe.as_ref().expect("moe layers");
                    Some(load_moe(w, &format!("{n}.moe"), mc, d)?)
                } else {
                    None
                };
                Ok(EncoderLayer {
                    qkv: Q4g::load(w, &format!("{n}.qkv"), true)?,
                    out: Q4g::load(w, &format!("{n}.out"), true)?,
                    norm1: Norm::load(w, &format!("{n}.norm1"), d)?,
                    ffn1: Q4g::load(w, &format!("{n}.ffn1"), true)?,
                    ffn2: Q4g::load(w, &format!("{n}.ffn2"), true)?,
                    norm2: Norm::load(w, &format!("{n}.norm2"), d)?,
                    moe,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for l in &layers {
            anyhow::ensure!(
                (l.qkv.k, l.qkv.n, l.out.k, l.out.n, l.ffn1.k, l.ffn2.n, l.ffn2.k)
                    == (d, 3 * d, d, d, d, d, l.ffn1.n),
                "an encoder layer's Linears are not shaped for hidden {d}"
            );
        }
        let output = match (&c.decoder, &c.ar) {
            (Some(dc), None) => {
                let relative = if dc.relative {
                    Some(Relative::load(
                        w,
                        "decoder.relative",
                        c.relative_buckets
                            .context("decoder biases need the buckets")?,
                        2 * dc.heads,
                        p,
                    )?)
                } else {
                    None
                };
                let dl = (0..dc.layers)
                    .map(|i| {
                        let n = format!("decoder.layers.{i}");
                        Ok(DecoderLayer {
                            self_norm: Norm::load(w, &format!("{n}.self_norm"), d)?,
                            self_qkv: Q4g::load(w, &format!("{n}.self_qkv"), true)?,
                            self_out: Q4g::load(w, &format!("{n}.self_out"), true)?,
                            cross_norm: Norm::load(w, &format!("{n}.cross_norm"), d)?,
                            cross_q: Q4g::load(w, &format!("{n}.cross_q"), true)?,
                            cross_out: Q4g::load(w, &format!("{n}.cross_out"), true)?,
                            ffn_norm: Norm::load(w, &format!("{n}.ffn_norm"), d)?,
                            ffn1: Q4g::load(w, &format!("{n}.ffn1"), true)?,
                            ffn2: Q4g::load(w, &format!("{n}.ffn2"), true)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let kv = Q4g::load(w, "decoder.kv", true)?;
                anyhow::ensure!(
                    (kv.k, kv.n) == (d, 2 * d * dc.layers) && d % dc.heads == 0,
                    "decoder.kv is ({}, {}), expected ({d}, {})",
                    kv.k,
                    kv.n,
                    2 * d * dc.layers
                );
                Output::Ctc {
                    decoder: Decoder {
                        kv,
                        positions: if dc.absolute_positions {
                            Some(Table8::load(w, "decoder.positions", p, d)?)
                        } else {
                            None
                        },
                        relative,
                        layers: dl,
                        final_norm: Norm::load(w, "decoder.final_norm", d)?,
                        c: dc.clone(),
                    },
                    head: head.expect("ctc"),
                }
            }
            (None, Some(ac)) => {
                anyhow::ensure!(
                    ac.heads * ac.head_dim == d,
                    "ar: heads x head_dim must be hidden {d}"
                );
                let kv = (0..ac.layers)
                    .map(|l| {
                        Ok((
                            Lin8::load(isa, w, &format!("ar.layers.{l}.k"), d, d, true)?,
                            Lin8::load(isa, w, &format!("ar.layers.{l}.v"), d, d, true)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Output::ArCross { c: ac.clone(), kv }
            }
            _ => unreachable!("checked above"),
        };
        Ok(NativeEncoder {
            isa,
            embedding,
            positions,
            relative,
            ple,
            layers,
            final_norm: Norm::load(w, "final_norm", d)?,
            output,
            scratch: Scratch::default(),
            profile: Profile::from_env(),
            config,
        })
    }

    /// The profile back after a conversion of `t` frames, written every 256 conversions.
    fn profiled(&mut self, prof: Option<Profile>, t: usize) {
        self.profile = prof.map(|mut p| {
            p.conversions += 1;
            p.frames += t;
            if p.conversions % 256 == 0 {
                p.write();
            }
            p
        });
    }

    /// "ctc" or "ar_cross".
    pub fn output(&self) -> &str {
        &self.config.output
    }

    /// The AR layers' (layers, heads, head_dim), for an "ar_cross" encoder.
    pub fn ar_shape(&self) -> Option<(usize, usize, usize)> {
        match &self.output {
            Output::ArCross { c, .. } => Some((c.layers, c.heads, c.head_dim)),
            Output::Ctc { .. } => None,
        }
    }

    /// The width of the CTC logits: the vocabulary's, or the head's rows when it holds some of its tokens.
    pub fn logit_width(&self) -> usize {
        self.config.head_rows.unwrap_or(self.config.vocab)
    }

    /// The CTC logits (T, `logit_width`) of one row of ids.
    pub fn logits(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            matches!(self.output, Output::Ctc { .. }),
            "this encoder's output is {}, not CTC logits",
            self.config.output
        );
        let mut prof = self.profile.take();
        let (seg, mut s) = self.encode(ids, &mut prof)?;
        let t = ids.len();
        let Output::Ctc { decoder, head } = &self.output else {
            unreachable!()
        };
        decode(
            decoder,
            self.config.hidden,
            self.config.norm_eps,
            &seg,
            &mut s,
            &mut prof,
        );
        let sx = quantize_tensor(&s.x[..t * self.config.hidden], &mut s.q8);
        lap(&mut prof, 13);
        let mut logits = Vec::new();
        head.forward(self.isa, &s.q8, t, sx, &mut s.acc, &mut logits);
        lap(&mut prof, 14);
        self.scratch = s;
        self.profiled(prof, t);
        Ok(logits)
    }

    /// The AR layers' cross-attention keys and values of one row of ids, each (layers, 1, heads, T, head_dim).
    pub fn cross(&mut self, ids: &[i64]) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(
            matches!(self.output, Output::ArCross { .. }),
            "this encoder's output is {}, not the AR layers' keys and values",
            self.config.output
        );
        let mut prof = self.profile.take();
        let (_, mut s) = self.encode(ids, &mut prof)?;
        let t = ids.len();
        let Output::ArCross { c, kv } = &self.output else {
            unreachable!()
        };
        let (h, dh) = (c.heads, c.head_dim);
        let d = h * dh;
        let sx = quantize_tensor(&s.x[..t * d], &mut s.q8);
        let mut ks = vec![0f32; c.layers * h * t * dh];
        let mut vs = vec![0f32; c.layers * h * t * dh];
        for (l, (k, v)) in kv.iter().enumerate() {
            for (lin, dst) in [(k, &mut ks), (v, &mut vs)] {
                lin.forward(self.isa, &s.q8, t, sx, &mut s.acc, &mut s.a);
                let dst = &mut dst[l * h * t * dh..(l + 1) * h * t * dh];
                for f in 0..t {
                    for hh in 0..h {
                        dst[(hh * t + f) * dh..(hh * t + f + 1) * dh]
                            .copy_from_slice(&s.a[f * d + hh * dh..f * d + (hh + 1) * dh]);
                    }
                }
            }
        }
        lap(&mut prof, 15);
        self.scratch = s;
        self.profiled(prof, t);
        Ok((ks, vs))
    }

    /// The encoder's frames (T, hidden) in the scratch's `x`, and each frame's segment (1 after the row's first [SEP];
    /// a row without one counts from its first frame).
    fn encode(&mut self, ids: &[i64], prof: &mut Option<Profile>) -> Result<(Vec<usize>, Scratch)> {
        if let Some(p) = prof.as_mut() {
            p.last = Instant::now();
        }
        let c = &self.config;
        let (t, d) = (ids.len(), c.hidden);
        anyhow::ensure!(
            t > 0 && t <= c.max_positions,
            "{t} frames, the encoder takes 1 to {}",
            c.max_positions
        );
        anyhow::ensure!(
            ids.iter().all(|&id| id >= 0 && (id as usize) < c.vocab),
            "an id out of the vocabulary ({} tokens)",
            c.vocab
        );
        let mut s = std::mem::take(&mut self.scratch);
        let sep = ids
            .iter()
            .position(|&id| id == c.sep_id as i64)
            .unwrap_or(0);
        let seg: Vec<usize> = (0..t).map(|f| usize::from(f > sep)).collect();
        // the input rows
        s.x.clear();
        s.x.resize(t * d, 0.0);
        for (f, &id) in ids.iter().enumerate() {
            self.embedding
                .row(id as usize, &mut s.x[f * d..(f + 1) * d]);
        }
        if let Some(pt) = &self.positions {
            for f in 0..t {
                pt.add_row(f, &mut s.x[f * d..(f + 1) * d]);
            }
        }
        if let Some(ple) = &self.ple {
            let u: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
            ple.rows(&u, &mut s.rows);
            ple.lookup(&s.rows, t, &mut s.lookup);
        }
        let mut bias = std::mem::take(&mut s.bias);
        if let Some(r) = &self.relative {
            r.bias(&seg, 0, c.heads, &mut bias);
        }
        let rel = self.relative.as_ref().map(|_| &bias[..]);
        lap(prof, 0);
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple {
                // x + sigmoid(gate(x)) * proj(lookup)
                ple.gates[i].forward(&s.x, &mut s.a, &mut s.lin);
                ple.projs[i].forward(&s.lookup, &mut s.b, &mut s.lin);
                add_gated(&mut s.x, &s.a, &s.b);
                lap(prof, 1);
            }
            layer.qkv.forward(&s.x, &mut s.qkv, &mut s.lin);
            lap(prof, 2);
            s.a.resize(t * d, 0.0);
            attention(
                &s.qkv,
                3 * d,
                0,
                &s.qkv,
                3 * d,
                d,
                &s.qkv,
                3 * d,
                2 * d,
                t,
                t,
                c.heads,
                d / c.heads,
                rel,
                &mut s.a,
                &mut s.attn,
            );
            lap(prof, 3);
            layer.out.forward(&s.a, &mut s.b, &mut s.lin);
            s.h.resize(t * d, 0.0);
            add_layer_norm_here(
                &s.x,
                Some(&s.b),
                &layer.norm1.g,
                &layer.norm1.b,
                c.norm_eps,
                &mut s.h,
                None,
            );
            lap(prof, 4);
            // the FFN (and the routed experts) on the normed frames
            layer.ffn1.forward(&s.h, &mut s.f, &mut s.lin);
            gelu_in_place(&mut s.f);
            layer.ffn2.forward(&s.f, &mut s.b, &mut s.lin);
            lap(prof, 5);
            if let Some((moe, _)) = &layer.moe {
                s.a.resize(t * d, 0.0);
                moe.forward(&s.h, &mut s.a, &mut s.moe);
                for (bv, &mv) in s.b.iter_mut().zip(&s.a) {
                    *bv += mv;
                }
                lap(prof, 6);
            }
            add_layer_norm_here(
                &s.h,
                Some(&s.b),
                &layer.norm2.g,
                &layer.norm2.b,
                c.norm_eps,
                &mut s.x,
                None,
            );
            lap(prof, 7);
        }
        // the final norm (the graph's AddLayerNorm with a zero skip: the norm of x itself)
        s.h.resize(t * d, 0.0);
        add_layer_norm_here(
            &s.x,
            None,
            &self.final_norm.g,
            &self.final_norm.b,
            c.norm_eps,
            &mut s.h,
            None,
        );
        std::mem::swap(&mut s.x, &mut s.h);
        s.bias = bias;
        lap(prof, 8);
        Ok((seg, s))
    }
}

impl Drop for NativeEncoder {
    fn drop(&mut self) {
        if let Some(p) = &self.profile {
            p.write();
        }
    }
}

/// The routed experts of a layer: `<name>.router` (E, d), `.w1` (E, d / g, f, g / 2), `.s1` / `.z1` (E, f, d / g),
/// `.b1` (E, f), `.w2` (E, f / g, d, g / 2), `.s2` / `.z2` (E, d, f / g), `.b2` (E, d) as TokenMoEQ4G's inputs.
fn load_moe(w: &Weights, name: &str, mc: &MoeConfig, d: usize) -> Result<(GroupMoe, Vec<Vec<u8>>)> {
    let e = mc.experts;
    let (w1, s1shape) = w.u8_any(&format!("{name}.w1"))?;
    anyhow::ensure!(
        s1shape.len() == 4 && s1shape[0] == e,
        "{name}.w1 is {s1shape:?}, expected (E {e}, d / g, f, g / 2)"
    );
    let (f, g) = (s1shape[2], s1shape[3] * 2);
    let (w2, _) = w.u8_any(&format!("{name}.w2"))?;
    let (w1, w2) = (w1.to_vec(), w2.to_vec());
    let s1 = w.f32(&format!("{name}.s1"), &[e, f, d / g])?;
    let z1 = w.f32(&format!("{name}.z1"), &[e, f, d / g])?;
    let s2 = w.f32(&format!("{name}.s2"), &[e, d, f / g])?;
    let z2 = w.f32(&format!("{name}.z2"), &[e, d, f / g])?;
    let router = w.f32(&format!("{name}.router"), &[e, d])?.to_vec();
    let b1 = w.f32(&format!("{name}.b1"), &[e, f])?.to_vec();
    let b2 = w.f32(&format!("{name}.b2"), &[e, d])?.to_vec();
    // SAFETY: the codes are returned with the layer and kept beside it, unchanged; their heap buffers do not move
    let moe = unsafe {
        group_moe(
            e,
            d,
            f,
            mc.top_k,
            g,
            router,
            (&w1, s1, z1, b1),
            (&w2, s2, z2, b2),
        )
    }
    .map_err(|m| anyhow::anyhow!("{name}: {m}"))?;
    Ok((moe, vec![w1, w2]))
}

/// The NAT decoder over the encoder's frames `s.x` (T, d), its output in `s.x`.
fn decode(
    dec: &Decoder,
    d: usize,
    eps: f32,
    seg: &[usize],
    s: &mut Scratch,
    prof: &mut Option<Profile>,
) {
    let t = seg.len();
    let heads = dec.c.heads;
    let dh = d / heads;
    let layers = dec.layers.len();
    // every layer's keys and values from the encoder's frames
    dec.kv.forward(&s.x, &mut s.kv, &mut s.lin);
    let (mut self_bias, mut cross_bias) =
        (std::mem::take(&mut s.bias), std::mem::take(&mut s.bias2));
    if let Some(r) = &dec.relative {
        r.bias(seg, 0, heads, &mut self_bias);
        r.bias(seg, heads, heads, &mut cross_bias);
    }
    let (sb, cb) = match &dec.relative {
        Some(_) => (Some(&self_bias[..]), Some(&cross_bias[..])),
        None => (None, None),
    };
    // the residual stream as (x, skip), summed in the next norm: x the encoder's frames, skip the positions' rows
    let mut x = std::mem::take(&mut s.x);
    let mut skip = vec![0f32; t * d];
    if let Some(pt) = &dec.positions {
        for f in 0..t {
            pt.row(f, &mut skip[f * d..(f + 1) * d]);
        }
    }
    let mut sum = vec![0f32; t * d];
    s.h.resize(t * d, 0.0);
    lap(prof, 9);
    for (i, layer) in dec.layers.iter().enumerate() {
        add_layer_norm_here(
            &x,
            Some(&skip),
            &layer.self_norm.g,
            &layer.self_norm.b,
            eps,
            &mut s.h,
            Some(&mut sum),
        );
        std::mem::swap(&mut x, &mut sum);
        layer.self_qkv.forward(&s.h, &mut s.qkv, &mut s.lin);
        s.a.resize(t * d, 0.0);
        attention(
            &s.qkv,
            3 * d,
            0,
            &s.qkv,
            3 * d,
            d,
            &s.qkv,
            3 * d,
            2 * d,
            t,
            t,
            heads,
            dh,
            sb,
            &mut s.a,
            &mut s.attn,
        );
        layer.self_out.forward(&s.a, &mut skip, &mut s.lin);
        lap(prof, 10);
        add_layer_norm_here(
            &x,
            Some(&skip),
            &layer.cross_norm.g,
            &layer.cross_norm.b,
            eps,
            &mut s.h,
            Some(&mut sum),
        );
        std::mem::swap(&mut x, &mut sum);
        layer.cross_q.forward(&s.h, &mut s.b, &mut s.lin);
        let kvw = 2 * d * layers;
        attention(
            &s.b,
            d,
            0,
            &s.kv,
            kvw,
            2 * d * i,
            &s.kv,
            kvw,
            2 * d * i + d,
            t,
            t,
            heads,
            dh,
            cb,
            &mut s.a,
            &mut s.attn,
        );
        layer.cross_out.forward(&s.a, &mut skip, &mut s.lin);
        lap(prof, 11);
        add_layer_norm_here(
            &x,
            Some(&skip),
            &layer.ffn_norm.g,
            &layer.ffn_norm.b,
            eps,
            &mut s.h,
            Some(&mut sum),
        );
        std::mem::swap(&mut x, &mut sum);
        layer.ffn1.forward(&s.h, &mut s.f, &mut s.lin);
        gelu_in_place(&mut s.f);
        layer.ffn2.forward(&s.f, &mut skip, &mut s.lin);
        lap(prof, 12);
    }
    s.x.resize(t * d, 0.0);
    add_layer_norm_here(
        &x,
        Some(&skip),
        &dec.final_norm.g,
        &dec.final_norm.b,
        eps,
        &mut s.x,
        None,
    );
    s.bias = self_bias;
    s.bias2 = cross_bias;
    lap(prof, 12);
}

/// Whether the AVX2 and FMA paths run (`crate::cpu`): the attention, the GELU and the sigmoids then run compiled for
/// them (the same code; the engine's build otherwise compiles it for SSE2).
fn avx2() -> bool {
    crate::cpu::avx2()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// v = GELU(v), element by element.
fn gelu_in_place(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::pipeline::lse::avx512f() {
        // SAFETY: the CPU has AVX-512 F (3941823: 46 us a conversion of 30 frames in AVX2)
        return unsafe { gelu_avx512(v) };
    }
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: the CPU has the features (`avx2`)
        return unsafe { gelu_avx2(v) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx() {
        // SAFETY: the CPU has AVX
        return unsafe { gelu_avx1(v) };
    }
    gelu_body(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn gelu_avx512(v: &mut [f32]) {
    gelu_body(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_avx2(v: &mut [f32]) {
    gelu_body(v)
}

/// The AVX level's build (no AVX2 or FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn gelu_avx1(v: &mut [f32]) {
    gelu_body(v)
}

#[inline(always)]
fn gelu_body(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = gelu(*x);
    }
}

/// x += sigmoid(g) * p, element by element (the PLE's gated injection).
fn add_gated(x: &mut [f32], g: &[f32], p: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::pipeline::lse::avx512f() {
        // SAFETY: the CPU has AVX-512 F
        return unsafe { add_gated_avx512(x, g, p) };
    }
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: the CPU has the features (`avx2`)
        return unsafe { add_gated_avx2(x, g, p) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx() {
        // SAFETY: the CPU has AVX
        return unsafe { add_gated_avx1(x, g, p) };
    }
    add_gated_body(x, g, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn add_gated_avx512(x: &mut [f32], g: &[f32], p: &[f32]) {
    add_gated_body(x, g, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_gated_avx2(x: &mut [f32], g: &[f32], p: &[f32]) {
    add_gated_body(x, g, p)
}

/// The AVX level's build (no AVX2 or FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn add_gated_avx1(x: &mut [f32], g: &[f32], p: &[f32]) {
    add_gated_body(x, g, p)
}

#[inline(always)]
fn add_gated_body(x: &mut [f32], g: &[f32], p: &[f32]) {
    for ((xv, &gv), &pv) in x.iter_mut().zip(g).zip(p) {
        *xv += sigmoid(gv) * pv;
    }
}

/// One head's keys transposed (head_dim, T) and values (T, head_dim), its scaled queries (T, head_dim), and the
/// scores (a row, or every query's rows padded to 8s).
#[derive(Default)]
struct AttentionScratch {
    kt: Vec<f32>,
    vh: Vec<f32>,
    qh: Vec<f32>,
    scores: Vec<f32>,
}

/// The attention of 32-wide heads in AVX-512 (`attention`'s function; 3941823's AVX2 kernel took 109 us a
/// conversion of 30 frames over 6 layers, 3 us a head): per head the keys transposed (32, T in 16s) and the values
/// and scaled queries copied, the scores S as up to 8 queries x 16 keys of sums in registers along the head (the
/// same fused multiply-adds in the same order as the AVX2 kernel), the bias and the softmax 16 keys at a time (the
/// row's tail by masks), then P v as up to 8 queries x 32 values along the keys.
///
/// # Safety
/// The CPU has AVX-512 F, AVX2 and FMA; the slices are shaped as `attention` reads them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_avx512_32(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    use std::arch::x86_64::*;
    const DH: usize = 32;
    let scale = 1.0 / (DH as f32).sqrt();
    let width = heads * DH;
    let tp = tk.div_ceil(16) * 16;
    s.kt.clear();
    s.kt.resize(DH * tp, 0.0);
    s.vh.resize(tk * DH, 0.0);
    s.qh.resize(tq * DH, 0.0);
    s.scores.resize(tq * tp, 0.0);
    assert!(out.len() >= tq * width);
    if let Some(b) = bias {
        assert!(b.len() >= heads * tq * tk);
    }
    // the last 16 keys' lanes in the row
    let tail: __mmask16 = if tk % 16 == 0 {
        0xFFFF
    } else {
        (1u16 << (tk % 16)) - 1
    };
    for h in 0..heads {
        for j in 0..tk {
            let kr = &k[j * k_stride + k_off + h * DH..j * k_stride + k_off + (h + 1) * DH];
            for (c, &kv) in kr.iter().enumerate() {
                s.kt[c * tp + j] = kv;
            }
            s.vh[j * DH..(j + 1) * DH].copy_from_slice(
                &v[j * v_stride + v_off + h * DH..j * v_stride + v_off + (h + 1) * DH],
            );
        }
        for i in 0..tq {
            let qr = &q[i * q_stride + q_off + h * DH..i * q_stride + q_off + (h + 1) * DH];
            for (o, &qv) in s.qh[i * DH..(i + 1) * DH].iter_mut().zip(qr) {
                *o = qv * scale;
            }
        }
        let (kt, qh, sc) = (s.kt.as_ptr(), s.qh.as_ptr(), s.scores.as_mut_ptr());
        let mut i = 0;
        while i < tq {
            let rows = (tq - i).min(8);
            match rows {
                8 => scores_block::<8>(kt, qh, sc, i, tp),
                7 => scores_block::<7>(kt, qh, sc, i, tp),
                6 => scores_block::<6>(kt, qh, sc, i, tp),
                5 => scores_block::<5>(kt, qh, sc, i, tp),
                4 => scores_block::<4>(kt, qh, sc, i, tp),
                3 => scores_block::<3>(kt, qh, sc, i, tp),
                2 => scores_block::<2>(kt, qh, sc, i, tp),
                _ => scores_block::<1>(kt, qh, sc, i, tp),
            }
            i += rows;
        }
        // the bias and the softmax over each query's tk keys
        for i in 0..tq {
            let b = bias.map_or(std::ptr::null(), |b| b.as_ptr().add((h * tq + i) * tk));
            softmax16(sc.add(i * tp), tp, tk, tail, b);
        }
        // P v
        let (vh, sc, op) = (
            s.vh.as_ptr(),
            s.scores.as_ptr(),
            out.as_mut_ptr().add(h * DH),
        );
        let mut i = 0;
        while i < tq {
            let rows = (tq - i).min(8);
            match rows {
                8 => pv_block::<8>(vh, sc, op, i, tk, tp, width),
                7 => pv_block::<7>(vh, sc, op, i, tk, tp, width),
                6 => pv_block::<6>(vh, sc, op, i, tk, tp, width),
                5 => pv_block::<5>(vh, sc, op, i, tk, tp, width),
                4 => pv_block::<4>(vh, sc, op, i, tk, tp, width),
                3 => pv_block::<3>(vh, sc, op, i, tk, tp, width),
                2 => pv_block::<2>(vh, sc, op, i, tk, tp, width),
                _ => pv_block::<1>(vh, sc, op, i, tk, tp, width),
            }
            i += rows;
        }
    }
}

/// A row of scores (`tp` floats, the first `tk` the keys', `tail` the last 16's lanes among them) plus the bias row
/// `b` (tk floats, or null) turned to its softmax over the keys, the padding zero.
///
/// # Safety
/// As `attention_avx512_32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[inline]
unsafe fn softmax16(row: *mut f32, tp: usize, tk: usize, tail: u16, b: *const f32) {
    use std::arch::x86_64::*;
    let mut m = _mm512_set1_ps(f32::NEG_INFINITY);
    for jb in (0..tp).step_by(16) {
        let mask = if jb + 16 <= tk { 0xFFFF } else { tail };
        let mut x = _mm512_maskz_loadu_ps(mask, row.add(jb));
        if !b.is_null() {
            x = _mm512_add_ps(x, _mm512_maskz_loadu_ps(mask, b.add(jb)));
        }
        _mm512_storeu_ps(row.add(jb), x);
        m = _mm512_mask_max_ps(m, mask, x, m);
    }
    let vm = _mm512_set1_ps(_mm512_reduce_max_ps(m));
    let mut sum = _mm512_setzero_ps();
    for jb in (0..tp).step_by(16) {
        let mask = if jb + 16 <= tk { 0xFFFF } else { tail };
        let e = _mm512_maskz_mov_ps(
            mask,
            crate::pipeline::lse::exp16(_mm512_sub_ps(_mm512_loadu_ps(row.add(jb)), vm)),
        );
        _mm512_storeu_ps(row.add(jb), e);
        sum = _mm512_add_ps(sum, e);
    }
    let inv = _mm512_set1_ps(1.0 / _mm512_reduce_add_ps(sum));
    for jb in (0..tp).step_by(16) {
        _mm512_storeu_ps(
            row.add(jb),
            _mm512_mul_ps(_mm512_loadu_ps(row.add(jb)), inv),
        );
    }
}

/// The AR step's cross-attention of one 32-wide head (`ar_native`; 3941823: 2 us a hypothesis, the frames' loops 30
/// wide with scalar tails): `rows` queries (q's rows `q_stride` apart from column `q_off`) against the encoder's
/// frames, their keys `kt` transposed (32, `tp` = frames in 16s, zero past the frames) and values `v` (frames, 32), the
/// outputs into `out`'s rows `width` apart; `qh` and `scores` are scratch. The encoder's own blocks: the same sums in
/// the same order as its attention.
///
/// # Safety
/// The CPU has AVX-512 F, AVX2 and FMA; the slices are shaped as said; `out` holds the rows' 32 outputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cross_rows_avx512_32(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    rows: usize,
    kt: &[f32],
    tp: usize,
    v: &[f32],
    tk: usize,
    out: *mut f32,
    width: usize,
    qh: &mut Vec<f32>,
    scores: &mut Vec<f32>,
) {
    const DH: usize = 32;
    assert!(kt.len() >= DH * tp && v.len() >= tk * DH && tp >= tk && tp % 16 == 0 && tk > 0);
    let scale = 1.0 / (DH as f32).sqrt();
    qh.resize(rows * DH, 0.0);
    for i in 0..rows {
        for (o, &qv) in qh[i * DH..(i + 1) * DH]
            .iter_mut()
            .zip(&q[i * q_stride + q_off..i * q_stride + q_off + DH])
        {
            *o = qv * scale;
        }
    }
    scores.resize(rows * tp, 0.0);
    let (ktp, qp, sc) = (kt.as_ptr(), qh.as_ptr(), scores.as_mut_ptr());
    let tail: u16 = if tk % 16 == 0 {
        0xFFFF
    } else {
        (1u16 << (tk % 16)) - 1
    };
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(8);
        match r {
            8 => scores_block::<8>(ktp, qp, sc, i, tp),
            7 => scores_block::<7>(ktp, qp, sc, i, tp),
            6 => scores_block::<6>(ktp, qp, sc, i, tp),
            5 => scores_block::<5>(ktp, qp, sc, i, tp),
            4 => scores_block::<4>(ktp, qp, sc, i, tp),
            3 => scores_block::<3>(ktp, qp, sc, i, tp),
            2 => scores_block::<2>(ktp, qp, sc, i, tp),
            _ => scores_block::<1>(ktp, qp, sc, i, tp),
        }
        i += r;
    }
    for i in 0..rows {
        softmax16(sc.add(i * tp), tp, tk, tail, std::ptr::null());
    }
    let (vp, sc) = (v.as_ptr(), scores.as_ptr());
    let mut i = 0;
    while i < rows {
        let r = (rows - i).min(8);
        match r {
            8 => pv_block::<8>(vp, sc, out, i, tk, tp, width),
            7 => pv_block::<7>(vp, sc, out, i, tk, tp, width),
            6 => pv_block::<6>(vp, sc, out, i, tk, tp, width),
            5 => pv_block::<5>(vp, sc, out, i, tk, tp, width),
            4 => pv_block::<4>(vp, sc, out, i, tk, tp, width),
            3 => pv_block::<3>(vp, sc, out, i, tk, tp, width),
            2 => pv_block::<2>(vp, sc, out, i, tk, tp, width),
            _ => pv_block::<1>(vp, sc, out, i, tk, tp, width),
        }
        i += r;
    }
}

/// Scores of queries i..i + R against every 16 keys: sums of q[c] k[c] along the head in R x 1 registers.
///
/// # Safety
/// As `attention_avx512_32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[inline]
unsafe fn scores_block<const R: usize>(
    kt: *const f32,
    qh: *const f32,
    sc: *mut f32,
    i: usize,
    tp: usize,
) {
    use std::arch::x86_64::*;
    const DH: usize = 32;
    for jb in (0..tp).step_by(16) {
        let mut acc = [_mm512_setzero_ps(); R];
        for c in 0..DH {
            let kv = _mm512_loadu_ps(kt.add(c * tp + jb));
            for (r, a) in acc.iter_mut().enumerate() {
                *a = _mm512_fmadd_ps(_mm512_set1_ps(*qh.add((i + r) * DH + c)), kv, *a);
            }
        }
        for (r, a) in acc.iter().enumerate() {
            _mm512_storeu_ps(sc.add((i + r) * tp + jb), *a);
        }
    }
}

/// Outputs of queries i..i + R (32 values each, `op` at the head's column, rows `width` apart): their weights times
/// the values, summed along the keys in R x 2 registers.
///
/// # Safety
/// As `attention_avx512_32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
#[inline]
unsafe fn pv_block<const R: usize>(
    vh: *const f32,
    sc: *const f32,
    op: *mut f32,
    i: usize,
    tk: usize,
    tp: usize,
    width: usize,
) {
    use std::arch::x86_64::*;
    let mut acc = [[_mm512_setzero_ps(); 2]; R];
    for j in 0..tk {
        let (v0, v1) = (
            _mm512_loadu_ps(vh.add(j * 32)),
            _mm512_loadu_ps(vh.add(j * 32 + 16)),
        );
        for (r, a) in acc.iter_mut().enumerate() {
            let p = _mm512_set1_ps(*sc.add((i + r) * tp + j));
            a[0] = _mm512_fmadd_ps(p, v0, a[0]);
            a[1] = _mm512_fmadd_ps(p, v1, a[1]);
        }
    }
    for (r, a) in acc.iter().enumerate() {
        _mm512_storeu_ps(op.add((i + r) * width), a[0]);
        _mm512_storeu_ps(op.add((i + r) * width + 16), a[1]);
    }
}

/// The attention of 32-wide heads in AVX2 + FMA (`attention`'s function): per head, the scores S = (q / sqrt(32))
/// k^T as 4 queries x 8 keys of sums in registers along the head, the bias and the softmax per query row, then P v as
/// 2 queries x 32 values in registers along the keys.
///
/// # Safety
/// The CPU has AVX2 and FMA; the slices are shaped as `attention` reads them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_avx2_32(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    use std::arch::x86_64::*;
    const DH: usize = 32;
    let scale = 1.0 / (DH as f32).sqrt();
    let width = heads * DH;
    let tp = tk.div_ceil(8) * 8;
    s.kt.clear();
    s.kt.resize(DH * tp, 0.0);
    s.vh.resize(tk * DH, 0.0);
    s.qh.resize(tq * DH, 0.0);
    s.scores.resize(tq * tp, 0.0);
    assert!(out.len() >= tq * width);
    for h in 0..heads {
        for j in 0..tk {
            let kr = &k[j * k_stride + k_off + h * DH..j * k_stride + k_off + (h + 1) * DH];
            for (c, &kv) in kr.iter().enumerate() {
                s.kt[c * tp + j] = kv;
            }
            s.vh[j * DH..(j + 1) * DH].copy_from_slice(
                &v[j * v_stride + v_off + h * DH..j * v_stride + v_off + (h + 1) * DH],
            );
        }
        for i in 0..tq {
            let qr = &q[i * q_stride + q_off + h * DH..i * q_stride + q_off + (h + 1) * DH];
            for (o, &qv) in s.qh[i * DH..(i + 1) * DH].iter_mut().zip(qr) {
                *o = qv * scale;
            }
        }
        // S: 4 queries x 8 keys a block (the queries' tail one at a time)
        let (kt, qh, sc) = (s.kt.as_ptr(), s.qh.as_ptr(), s.scores.as_mut_ptr());
        let mut i = 0;
        while i < tq {
            let rows = if tq - i >= 4 { 4 } else { 1 };
            for jb in (0..tp).step_by(8) {
                let mut acc = [_mm256_setzero_ps(); 4];
                for c in 0..DH {
                    let kv = _mm256_loadu_ps(kt.add(c * tp + jb));
                    if rows == 4 {
                        acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(*qh.add(i * DH + c)), kv, acc[0]);
                        acc[1] =
                            _mm256_fmadd_ps(_mm256_set1_ps(*qh.add((i + 1) * DH + c)), kv, acc[1]);
                        acc[2] =
                            _mm256_fmadd_ps(_mm256_set1_ps(*qh.add((i + 2) * DH + c)), kv, acc[2]);
                        acc[3] =
                            _mm256_fmadd_ps(_mm256_set1_ps(*qh.add((i + 3) * DH + c)), kv, acc[3]);
                    } else {
                        acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(*qh.add(i * DH + c)), kv, acc[0]);
                    }
                }
                for (r, a) in acc.iter().take(rows).enumerate() {
                    _mm256_storeu_ps(sc.add((i + r) * tp + jb), *a);
                }
            }
            i += rows;
        }
        // the bias and the softmax over each query's tk keys
        for i in 0..tq {
            let row = &mut s.scores[i * tp..i * tp + tk];
            if let Some(b) = bias {
                for (r, &bv) in row
                    .iter_mut()
                    .zip(&b[(h * tq + i) * tk..(h * tq + i + 1) * tk])
                {
                    *r += bv;
                }
            }
            let m = row_max(row);
            for r in row.iter_mut() {
                *r = exp(*r - m);
            }
            let inv = 1.0 / lane_sum(row);
            for r in row.iter_mut() {
                *r *= inv;
            }
        }
        // P v: 2 queries x 32 values a block
        let (vh, sc, op) = (s.vh.as_ptr(), s.scores.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i < tq {
            let rows = if tq - i >= 2 { 2 } else { 1 };
            let mut a0 = [_mm256_setzero_ps(); 4];
            let mut a1 = [_mm256_setzero_ps(); 4];
            for j in 0..tk {
                let vp = vh.add(j * DH);
                let vr = [
                    _mm256_loadu_ps(vp),
                    _mm256_loadu_ps(vp.add(8)),
                    _mm256_loadu_ps(vp.add(16)),
                    _mm256_loadu_ps(vp.add(24)),
                ];
                let p0 = _mm256_set1_ps(*sc.add(i * tp + j));
                for x in 0..4 {
                    a0[x] = _mm256_fmadd_ps(p0, vr[x], a0[x]);
                }
                if rows == 2 {
                    let p1 = _mm256_set1_ps(*sc.add((i + 1) * tp + j));
                    for x in 0..4 {
                        a1[x] = _mm256_fmadd_ps(p1, vr[x], a1[x]);
                    }
                }
            }
            for x in 0..4 {
                _mm256_storeu_ps(op.add(i * width + h * DH + 8 * x), a0[x]);
                if rows == 2 {
                    _mm256_storeu_ps(op.add((i + 1) * width + h * DH + 8 * x), a1[x]);
                }
            }
            i += rows;
        }
    }
}

/// Multi-head attention (com.microsoft.MultiHeadAttention on one row, no padding): for each head h, query frame i
/// and key frame j, softmax_j(q_i . k_j / sqrt(dh) + bias[h][i][j]) and the weighted sum of the v_j, into out (tq, heads
/// x dh). q, k and v are read from rows of `*_stride` floats at column `*_off` (the packed projections in place). Per
/// head the keys are transposed once, so a query's scores are multiply-adds along the key frames, and the values
/// copied together; the softmax's exponentials are `lse::exp`.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    dh: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    #[cfg(target_arch = "x86_64")]
    if dh == 32 && crate::pipeline::lse::avx512f() && avx2() {
        // SAFETY: the CPU has AVX-512 F, AVX2 and FMA; the slices as the callers shape them
        return unsafe {
            attention_avx512_32(
                q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, bias,
                out, s,
            )
        };
    }
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: the CPU has the features (`avx2`); the slices as the callers shape them
        return unsafe {
            if dh == 32 {
                attention_avx2_32(
                    q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads,
                    bias, out, s,
                )
            } else {
                attention_avx2(
                    q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, dh,
                    bias, out, s,
                )
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx() {
        // SAFETY: the CPU has AVX; the slices as the callers shape them
        return unsafe {
            attention_avx1(
                q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, dh,
                bias, out, s,
            )
        };
    }
    attention_body(
        q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, dh, bias, out, s,
    )
}

/// The AVX level's build of the plain attention (no AVX2 or FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_avx1(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    dh: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    attention_body(
        q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, dh, bias, out, s,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn attention_avx2(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    dh: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    attention_body(
        q, q_stride, q_off, k, k_stride, k_off, v, v_stride, v_off, tq, tk, heads, dh, bias, out, s,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attention_body(
    q: &[f32],
    q_stride: usize,
    q_off: usize,
    k: &[f32],
    k_stride: usize,
    k_off: usize,
    v: &[f32],
    v_stride: usize,
    v_off: usize,
    tq: usize,
    tk: usize,
    heads: usize,
    dh: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    s: &mut AttentionScratch,
) {
    let scale = 1.0 / (dh as f32).sqrt();
    let width = heads * dh;
    s.kt.resize(dh * tk, 0.0);
    s.vh.resize(tk * dh, 0.0);
    s.scores.resize(tk, 0.0);
    for h in 0..heads {
        for j in 0..tk {
            let kr = &k[j * k_stride + k_off + h * dh..j * k_stride + k_off + (h + 1) * dh];
            for (c, &kv) in kr.iter().enumerate() {
                s.kt[c * tk + j] = kv;
            }
            s.vh[j * dh..(j + 1) * dh].copy_from_slice(
                &v[j * v_stride + v_off + h * dh..j * v_stride + v_off + (h + 1) * dh],
            );
        }
        for i in 0..tq {
            let row = &mut s.scores[..tk];
            match bias {
                Some(b) => row.copy_from_slice(&b[(h * tq + i) * tk..(h * tq + i + 1) * tk]),
                None => row.fill(0.0),
            }
            let qi = &q[i * q_stride + q_off + h * dh..i * q_stride + q_off + (h + 1) * dh];
            for (c, &qv) in qi.iter().enumerate() {
                let qc = qv * scale;
                for (r, &kv) in row.iter_mut().zip(&s.kt[c * tk..(c + 1) * tk]) {
                    *r += qc * kv;
                }
            }
            let m = row_max(row);
            for r in row.iter_mut() {
                *r = exp(*r - m);
            }
            let inv = 1.0 / lane_sum(row);
            let o = &mut out[i * width + h * dh..i * width + (h + 1) * dh];
            o.fill(0.0);
            for (j, &p) in row.iter().enumerate() {
                let p = p * inv;
                for (ov, &vv) in o.iter_mut().zip(&s.vh[j * dh..(j + 1) * dh]) {
                    *ov += p * vv;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_the_reference_generator() {
        // SplitMix64's first outputs from seed 0 (state += gamma before each mix)
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn quantize_tensor_is_dynamic_quantize_linear() {
        // range [-1, 3]: scale 4 / 255, zero point round(255 / 4) = 64 (63.75)
        let x = [-1.0f32, 0.0, 3.0, 1.0, 0.5];
        let mut q = Vec::new();
        let (s, z) = quantize_tensor(&x, &mut q);
        assert_eq!((s, z), (4.0 / 255.0, 64));
        assert_eq!(q[1], 64);
        assert_eq!(q[2], 255);
        assert_eq!(q[0], 0);
        for (&v, &c) in x.iter().zip(&q) {
            assert!(((c as i32 - z) as f32 * s - v).abs() <= s / 2.0 + 1e-6);
        }
        // all positive: the range widened to 0, zero point 0
        let (s, z) = quantize_tensor(&[2.0, 0.5], &mut q);
        assert_eq!((s, z), (2.0 / 255.0, 0));
        assert_eq!(q, [255, 64]); // 0.5 / (2 / 255) = 63.75
                                  // scale 1: 2.5 and 3.5 tie, to even
        quantize_tensor(&[2.5, 3.5, 255.0], &mut q);
        assert_eq!(q, [2, 4, 255]);
    }

    /// lse::exp (the softmax's and the sigmoid's) within 3e-7 of e^x (relative) over its range, and clamped outside
    /// it; the sigmoid from it.
    #[test]
    fn exp_is_exp() {
        let mut x = -87.0f64;
        while x <= 88.0 {
            let (got, want) = (exp(x as f32) as f64, (x as f32 as f64).exp());
            assert!(
                ((got - want) / want).abs() < 3e-7,
                "e^{x}: {got} against {want}"
            );
            x += 0.0137;
        }
        assert_eq!(exp(-200.0), exp(-87.3));
        assert_eq!(exp(0.0), 1.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7 && (sigmoid(3.0) - 0.952_574_1).abs() < 1e-6);
    }

    /// The attention against a plain f64 one with a bias, on the CPU's path and the plain one.
    #[test]
    fn attention_matches_f64() {
        let (t, heads, dh) = (5usize, 2usize, 4usize);
        let d = heads * dh;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = splitmix64(seed);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let qkv: Vec<f32> = (0..t * 3 * d).map(|_| rnd()).collect();
        let bias: Vec<f32> = (0..heads * t * t).map(|_| rnd()).collect();
        let mut out = vec![0f32; t * d];
        let mut plain = vec![0f32; t * d];
        let mut scratch = AttentionScratch::default();
        attention(
            &qkv,
            3 * d,
            0,
            &qkv,
            3 * d,
            d,
            &qkv,
            3 * d,
            2 * d,
            t,
            t,
            heads,
            dh,
            Some(&bias),
            &mut out,
            &mut scratch,
        );
        attention_body(
            &qkv,
            3 * d,
            0,
            &qkv,
            3 * d,
            d,
            &qkv,
            3 * d,
            2 * d,
            t,
            t,
            heads,
            dh,
            Some(&bias),
            &mut plain,
            &mut scratch,
        );
        for (a, b) in out.iter().zip(&plain) {
            assert!((a - b).abs() < 1e-6);
        }
        for h in 0..heads {
            for i in 0..t {
                let sc: Vec<f64> = (0..t)
                    .map(|j| {
                        (0..dh)
                            .map(|c| {
                                qkv[i * 3 * d + h * dh + c] as f64
                                    * qkv[j * 3 * d + d + h * dh + c] as f64
                            })
                            .sum::<f64>()
                            / (dh as f64).sqrt()
                            + bias[(h * t + i) * t + j] as f64
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = sc.iter().map(|s| (s - m).exp()).sum();
                for c in 0..dh {
                    let want: f64 = (0..t)
                        .map(|j| (sc[j] - m).exp() / z * qkv[j * 3 * d + 2 * d + h * dh + c] as f64)
                        .sum();
                    assert!((out[i * d + h * dh + c] as f64 - want).abs() < 1e-5);
                }
            }
        }
    }

    /// 32-wide heads (the AVX2 kernel when the CPU has it) against the plain path: self-attention on packed q / k / v
    /// with a bias, and cross-attention from other rows without, lengths not in 8s.
    #[test]
    fn attention_32_matches_the_plain_one() {
        let (heads, dh) = (3usize, 32usize);
        let d = heads * dh;
        let mut seed = 11u64;
        let mut rnd = || {
            seed = splitmix64(seed);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        for (tq, tk) in [
            (13usize, 13usize),
            (5, 5),
            (1, 1),
            (9, 9),
            (30, 30),
            (33, 33),
            (16, 16),
            (17, 17),
        ] {
            let qkv: Vec<f32> = (0..tk * 3 * d).map(|_| rnd()).collect();
            let bias: Vec<f32> = (0..heads * tq * tk).map(|_| rnd()).collect();
            let (mut a, mut b) = (vec![0f32; tq * d], vec![0f32; tq * d]);
            let mut s = AttentionScratch::default();
            attention(
                &qkv,
                3 * d,
                0,
                &qkv,
                3 * d,
                d,
                &qkv,
                3 * d,
                2 * d,
                tq,
                tk,
                heads,
                dh,
                Some(&bias),
                &mut a,
                &mut s,
            );
            attention_body(
                &qkv,
                3 * d,
                0,
                &qkv,
                3 * d,
                d,
                &qkv,
                3 * d,
                2 * d,
                tq,
                tk,
                heads,
                dh,
                Some(&bias),
                &mut b,
                &mut s,
            );
            assert!(
                a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5),
                "self {tq}"
            );
            // the AVX2 kernel too, where the CPU's path is the AVX-512 one
            #[cfg(target_arch = "x86_64")]
            if avx2() {
                let mut c = vec![0f32; tq * d];
                // SAFETY: the CPU has AVX2 and FMA
                unsafe {
                    attention_avx2_32(
                        &qkv,
                        3 * d,
                        0,
                        &qkv,
                        3 * d,
                        d,
                        &qkv,
                        3 * d,
                        2 * d,
                        tq,
                        tk,
                        heads,
                        Some(&bias),
                        &mut c,
                        &mut s,
                    )
                };
                assert!(
                    c.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5),
                    "self {tq}, AVX2"
                );
            }
            // cross: queries (tq, d), keys and values (tk, 4d) at columns 2d and 3d
            let q: Vec<f32> = (0..tq * d).map(|_| rnd()).collect();
            let kv: Vec<f32> = (0..tk * 4 * d).map(|_| rnd()).collect();
            attention(
                &q,
                d,
                0,
                &kv,
                4 * d,
                2 * d,
                &kv,
                4 * d,
                3 * d,
                tq,
                tk,
                heads,
                dh,
                None,
                &mut a,
                &mut s,
            );
            attention_body(
                &q,
                d,
                0,
                &kv,
                4 * d,
                2 * d,
                &kv,
                4 * d,
                3 * d,
                tq,
                tk,
                heads,
                dh,
                None,
                &mut b,
                &mut s,
            );
            assert!(
                a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5),
                "cross {tq}"
            );
        }
    }

    /// The PLE rows against the hash spelled out (u128 products), with a window reaching before the row's start.
    #[test]
    fn ple_rows_follow_the_hash() {
        let c = PleConfig {
            dim: 2,
            finalize: true,
            orders: vec![2, 3],
            table_sizes: vec![7, 11],
            row_offsets: vec![0, 7],
            multipliers: vec![
                vec![0, 0x9E37_79B9_7F4A_7C15 | 1, 0xD1B5_4A32_D192_ED03],
                vec![3, 0xFFFF_FFFF_FFFF_FFFF, 12345],
            ],
            salts: vec![0xABCD, u64::MAX],
            rows: 18,
        };
        let ple = Ple {
            table: Table8 {
                q: vec![0; 36],
                zero: 0,
                scale: 1.0,
                cols: 2,
            },
            gates: Vec::new(),
            projs: Vec::new(),
            max_order: 3,
            c: c.clone(),
        };
        let ids = [5u64, 9, 7, 3];
        let mut rows = Vec::new();
        ple.rows(&ids, &mut rows);
        for t in 0..ids.len() {
            for s in 0..2 {
                let order = c.orders[s];
                let mut mixed = 0u64;
                for back in 0..order {
                    // the token `back` frames before t multiplies the window's column 2 - back
                    let tok = if t >= back { ids[t - back] } else { PLE_BOS };
                    let m = c.multipliers[s][2 - back];
                    mixed ^= ((tok as u128 * m as u128) & u64::MAX as u128) as u64;
                }
                let mixed = splitmix64(mixed ^ c.salts[s]);
                assert_eq!(
                    rows[t * 2 + s],
                    (mixed % c.table_sizes[s] + c.row_offsets[s]) as usize,
                    "frame {t} slot {s}"
                );
            }
        }
    }

    #[test]
    fn tables_dequantize_as_dequantize_linear() {
        let mut w = Weights::default();
        w.0.insert("t.q".into(), Weight::U8(vec![0, 128, 255, 10], vec![2, 2]));
        w.0.insert("t.zero".into(), Weight::U8(vec![128], vec![1]));
        w.0.insert("t.scale".into(), Weight::F32(vec![0.5], vec![1]));
        let t = Table8::load(&w, "t", 2, 2).unwrap();
        let mut row = [0f32; 2];
        t.row(0, &mut row);
        assert_eq!(row, [-64.0, 0.0]);
        t.add_row(1, &mut row);
        assert_eq!(row, [-64.0 + 63.5, -59.0]);
        assert!(Table8::load(&w, "t", 4, 1).is_err());
    }

    /// Research timing on an export's own encoder (SETSUNA_NATIVE_ENCODER_FILE=<.safetensors>; skipped without it):
    /// whole conversions of 12, 30 and 60 frames, then each kernel on 30 frames of random activations (the layers'
    /// Linears, the routed experts, the attention, the norms, the head), us a call, summed over the layers. Run with
    /// --ignored --nocapture on one core (kairo/build.ps1 -Mode time).
    #[test]
    #[ignore]
    fn time_the_encoder() {
        let Some(path) = std::env::var_os("SETSUNA_NATIVE_ENCODER_FILE") else {
            eprintln!("SETSUNA_NATIVE_ENCODER_FILE not set");
            return;
        };
        let mut enc = NativeEncoder::load(Path::new(&path)).unwrap();
        eprintln!(
            "level {:?} (SETSUNA_CPU_LEVEL {:?})",
            crate::cpu::level(),
            std::env::var("SETSUNA_CPU_LEVEL").ok()
        );
        let c = enc.config.clone();
        let time = |reps: usize, f: &mut dyn FnMut()| -> f64 {
            for _ in 0..reps / 10 + 1 {
                f();
            }
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            t0.elapsed().as_secs_f64() * 1e6 / reps as f64
        };
        for t in [12usize, 30, 60] {
            let ids: Vec<i64> = (0..t)
                .map(|f| {
                    if f == t / 3 {
                        c.sep_id as i64
                    } else {
                        (splitmix64(f as u64) % (c.vocab as u64 - 10) + 10) as i64
                    }
                })
                .collect();
            let us = if c.output == "ctc" {
                time(400, &mut || {
                    std::hint::black_box(enc.logits(&ids).unwrap());
                })
            } else {
                time(400, &mut || {
                    std::hint::black_box(enc.cross(&ids).unwrap());
                })
            };
            eprintln!("{t:2} frames: {us:7.1} us a conversion");
        }
        // with the caches flushed before each conversion (a 128 MB buffer read through), as the latency runs see
        // them (1b20e28: the experts 0.33 ms a conversion there, 0.19 here warm), each tile mode at 30 frames
        {
            let flush = vec![1u8; 128 << 20];
            let ids: Vec<i64> = (0..30)
                .map(|f| {
                    if f == 10 {
                        c.sep_id as i64
                    } else {
                        (splitmix64(f as u64) % (c.vocab as u64 - 10) + 10) as i64
                    }
                })
                .collect();
            let ctc = c.output == "ctc";
            for round in 0..2 {
                for mode in [0u8, 4, 5, 6] {
                    crate::group_q::TILE_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
                    let mut total = 0f64;
                    let reps = 60;
                    for _ in 0..reps {
                        let mut sum = 0u64;
                        for line in flush.chunks(64) {
                            sum = sum.wrapping_add(line[0] as u64);
                        }
                        std::hint::black_box(sum);
                        let t0 = Instant::now();
                        if ctc {
                            std::hint::black_box(enc.logits(&ids).unwrap());
                        } else {
                            std::hint::black_box(enc.cross(&ids).unwrap());
                        }
                        total += t0.elapsed().as_secs_f64();
                    }
                    eprintln!(
                        "cold, 30 frames, tile mode {mode} (round {round}): {:7.1} us a conversion",
                        total * 1e6 / reps as f64
                    );
                }
            }
            crate::group_q::TILE_MODE.store(
                crate::group_q::TILE_MODE_DEFAULT,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let (t, d) = (30usize, c.hidden);
        let mut seed = 17u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = splitmix64(seed);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        };
        let x = rnd(t * 8 * d);
        let (mut y, mut s, mut ms) = (
            Vec::new(),
            GroupScratch::default(),
            GroupMoeScratch::default(),
        );
        let mut parts: Vec<(String, f64)> = Vec::new();
        let mut add = |name: &str, us: f64| match parts.iter_mut().find(|p| p.0 == name) {
            Some(p) => p.1 += us,
            None => parts.push((name.to_string(), us)),
        };
        for (l, layer) in enc.layers.iter().enumerate() {
            for (name, q) in [
                ("qkv", &layer.qkv),
                ("out", &layer.out),
                ("ffn1", &layer.ffn1),
                ("ffn2", &layer.ffn2),
            ] {
                add(
                    name,
                    time(2000, &mut || q.forward(&x[..t * q.k], &mut y, &mut s)),
                );
            }
            if let Some(ple) = &enc.ple {
                add(
                    "ple_gate",
                    time(2000, &mut || {
                        ple.gates[l].forward(&x[..t * d], &mut y, &mut s)
                    }),
                );
                add(
                    "ple_proj",
                    time(2000, &mut || {
                        ple.projs[l].forward(&x[..t * ple.c.dim], &mut y, &mut s)
                    }),
                );
            }
            if let Some((moe, _)) = &layer.moe {
                y.resize(t * d, 0.0);
                add(
                    "moe",
                    time(2000, &mut || {
                        moe.forward(&x[..t * d], &mut y[..t * d], &mut ms)
                    }),
                );
                let p = moe.time_parts(&x[..t * d], 2000);
                for (name, us) in ["moe_route", "moe_l1", "moe_gelu", "moe_l2", "moe_add"]
                    .iter()
                    .zip(p)
                {
                    add(name, us);
                }
            }
            let mut h = vec![0f32; t * d];
            add(
                "norm",
                time(4000, &mut || {
                    add_layer_norm_here(
                        &x[..t * d],
                        Some(&x[d..(t + 1) * d]),
                        &layer.norm1.g,
                        &layer.norm1.b,
                        c.norm_eps,
                        &mut h,
                        None,
                    )
                }),
            );
            let qkv = rnd(t * 3 * d);
            let bias = rnd(c.heads * t * t);
            let mut a = vec![0f32; t * d];
            let mut sa = AttentionScratch::default();
            add(
                "attention",
                time(2000, &mut || {
                    attention(
                        &qkv,
                        3 * d,
                        0,
                        &qkv,
                        3 * d,
                        d,
                        &qkv,
                        3 * d,
                        2 * d,
                        t,
                        t,
                        c.heads,
                        d / c.heads,
                        Some(&bias),
                        &mut a,
                        &mut sa,
                    )
                }),
            );
            let mut f = x[..t * layer.ffn1.n].to_vec();
            add("gelu", time(2000, &mut || gelu_in_place(&mut f)));
        }
        let mut q8 = Vec::new();
        let mut acc = Vec::new();
        match &enc.output {
            Output::Ctc { head, .. } => {
                add(
                    "head_quantize",
                    time(2000, &mut || {
                        quantize_tensor(&x[..t * d], &mut q8);
                    }),
                );
                let sx = quantize_tensor(&x[..t * d], &mut q8);
                add(
                    "head_gemm",
                    time(1000, &mut || {
                        head.forward(enc.isa, &q8, t, sx, &mut acc, &mut y)
                    }),
                );
            }
            Output::ArCross { kv, .. } => {
                let sx = quantize_tensor(&x[..t * d], &mut q8);
                for (k, v) in kv {
                    add(
                        "ar_kv",
                        time(2000, &mut || {
                            k.forward(enc.isa, &q8, t, sx, &mut acc, &mut y);
                            v.forward(enc.isa, &q8, t, sx, &mut acc, &mut y);
                        }),
                    );
                }
            }
        }
        eprintln!("kernels on {t} frames, us a conversion (all layers):");
        for (name, us) in &parts {
            eprintln!("  {name:14} {us:8.2}");
        }
        // the 4-bit kernel's tiles side by side (group_q::TILE_MODE), each over all layers, twice in turn
        use std::sync::atomic::Ordering::Relaxed;
        for round in 0..2 {
            for mode in [0u8, 4, 5, 6] {
                crate::group_q::TILE_MODE.store(mode, Relaxed);
                let (mut dense, mut experts) = (0f64, 0f64);
                for layer in &enc.layers {
                    for q in [&layer.qkv, &layer.out, &layer.ffn1, &layer.ffn2] {
                        dense += time(1000, &mut || q.forward(&x[..t * q.k], &mut y, &mut s));
                    }
                    if let Some((moe, _)) = &layer.moe {
                        y.resize(t * d, 0.0);
                        experts += time(1000, &mut || {
                            moe.forward(&x[..t * d], &mut y[..t * d], &mut ms)
                        });
                    }
                }
                eprintln!("tile mode {mode} (round {round}): dense Linears {dense:7.2} us, experts {experts:7.2} us");
            }
        }
        crate::group_q::TILE_MODE.store(crate::group_q::TILE_MODE_DEFAULT, Relaxed);
    }
}
