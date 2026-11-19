//! The AR decoder's step in Rust (the int8 manifest's `ar_decoder.native`, the research export's native step, up to
//! beam 20): one position of N hypotheses as the step graph computes
//! it, without ONNX Runtime. At beam 20 a position took 771 us in the int8 step graph, of which about 176 us was
//! ONNX Runtime's own run, 100 us its TopK and most of the rest some 60 small ops (the profile of 2026-11-12).
//!
//! Weights ([`NativeDecoder`]): every Linear's int8 rows (symmetric, a scale a row) prepacked for the MoE kernel's
//! u8 x i8 GEMM (`moe_op`), the activations quantized per row as there; the tied embedding int8 the same, read as the
//! input rows (dequantized) and as the output projection (its rows padded to 16s, the logit scale folded into the row
//! scales). LayerNorms, attention, softmax and the activation in f32.
//!
//! State ([`NativeSearch`]): the self-attention keys and values of every position the search computed, per layer,
//! in an arena of rows (heads x head_dim each) that never move; a hypothesis is the arena rows of its prefix, so
//! following the beam's parents copies a few indices instead of the cache. The encoder's cross-attention keys
//! (layers, 1, heads, T, head_dim) are transposed once per conversion to (layers, heads, head_dim, T), so a score
//! row is multiply-adds along the frames; the values are read as given.
//!
//! The next tokens of a row ([`top_k_log_probs`]): the log-softmax over the whole vocabulary (the step graph's), the
//! K best tokens `forbidden` allows, best first, by one pass that keeps the best K so far and skips a 16-wide chunk
//! with nothing above the K-th.

// the x86 kernels and what only they read are unused elsewhere
#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

#[cfg(test)]
use crate::moe_op::gemm_u8i8;
use crate::moe_op::{detect_isa, dot, gelu, gemm_u8i8_f32, quantize_row, Epilogue, Isa, Packed};
use crate::pipeline::ar::{StepModel, StepTop};
use crate::pipeline::lse;

/// The file's `__metadata__.format`.
pub const NATIVE_FORMAT: &str = "neo-ime-ar-step-v1";

/// The decoder's shapes and constants (the file's `__metadata__.config`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub max_positions: usize,
    /// "gelu" (exact) or "relu"
    pub activation: String,
    pub norm_eps: f32,
    pub logit_scale: f32,
    pub bos_id: u32,
    pub eos_id: u32,
    /// "choice": the script latent at choice points ([`ScriptHead`]); absent without it
    #[serde(default)]
    pub script: Option<String>,
}

/// A tensor of the file.
#[derive(Debug, Clone)]
pub enum Weight {
    I8(Vec<i8>, Vec<usize>),
    U8(Vec<u8>, Vec<usize>),
    F32(Vec<f32>, Vec<usize>),
}

impl Weight {
    fn kind(&self) -> &'static str {
        match self {
            Weight::I8(..) => "int8",
            Weight::U8(..) => "uint8",
            Weight::F32(..) => "f32",
        }
    }
}

/// The file's tensors by name.
#[derive(Debug, Clone, Default)]
pub struct Weights(pub HashMap<String, Weight>);

impl Weights {
    pub(crate) fn i8(&self, name: &str, shape: &[usize]) -> Result<&[i8]> {
        match self.0.get(name) {
            Some(Weight::I8(v, s)) if s == shape => Ok(v),
            Some(Weight::I8(_, s)) => anyhow::bail!("{name} is {s:?}, expected {shape:?}"),
            Some(w) => anyhow::bail!("{name} is {}, expected int8", w.kind()),
            None => anyhow::bail!("{name} missing"),
        }
    }

    pub(crate) fn f32(&self, name: &str, shape: &[usize]) -> Result<&[f32]> {
        match self.0.get(name) {
            Some(Weight::F32(v, s)) if s == shape => Ok(v),
            Some(Weight::F32(_, s)) => anyhow::bail!("{name} is {s:?}, expected {shape:?}"),
            Some(w) => anyhow::bail!("{name} is {}, expected f32", w.kind()),
            None => anyhow::bail!("{name} missing"),
        }
    }

    /// An int8 tensor of any shape, with its shape.
    pub(crate) fn i8_any(&self, name: &str) -> Result<(&[i8], &[usize])> {
        match self.0.get(name) {
            Some(Weight::I8(v, s)) => Ok((v, s)),
            Some(w) => anyhow::bail!("{name} is {}, expected int8", w.kind()),
            None => anyhow::bail!("{name} missing"),
        }
    }

    /// A uint8 tensor of any shape, with its shape.
    pub(crate) fn u8_any(&self, name: &str) -> Result<(&[u8], &[usize])> {
        match self.0.get(name) {
            Some(Weight::U8(v, s)) => Ok((v, s)),
            Some(w) => anyhow::bail!("{name} is {}, expected uint8", w.kind()),
            None => anyhow::bail!("{name} missing"),
        }
    }
}

/// A native step file: a safetensors file (a little-endian u64 header length, the JSON header, the data), its
/// `__metadata__` holding `format` ([`NATIVE_FORMAT`]) and `config` ([`NativeConfig`] as JSON); I8 and F32 tensors.
pub fn read(path: &Path) -> Result<(NativeConfig, Weights)> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse(&data).with_context(|| format!("native AR step {}", path.display()))
}

fn parse(data: &[u8]) -> Result<(NativeConfig, Weights)> {
    let (config, weights) = parse_safetensors(data, NATIVE_FORMAT)?;
    Ok((
        serde_json::from_str::<NativeConfig>(&config).context("config")?,
        weights,
    ))
}

/// A safetensors file of the engine's own implementations (this step, `pipeline::native_encoder`'s encoder): a
/// little-endian u64 header length, the JSON header, the data; its `__metadata__` holding `format` (which must be
/// `format`) and `config` (JSON, returned as it is); I8, U8 and F32 tensors.
pub(crate) fn parse_safetensors(data: &[u8], format: &str) -> Result<(String, Weights)> {
    anyhow::ensure!(data.len() >= 8, "shorter than its header length");
    let n = u64::from_le_bytes(data[..8].try_into().expect("8 bytes")) as usize;
    anyhow::ensure!(
        n <= data.len() - 8,
        "header of {n} bytes in a file of {}",
        data.len()
    );
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&data[8..8 + n]).context("header")?;
    let body = &data[8 + n..];
    let mut config = None;
    let mut weights = Weights::default();
    for (name, v) in header {
        if name == "__metadata__" {
            let found = v.get("format").and_then(|f| f.as_str()).unwrap_or("");
            anyhow::ensure!(found == format, "format {found:?}, expected {format}");
            let c = v
                .get("config")
                .and_then(|c| c.as_str())
                .context("config missing")?;
            config = Some(c.to_string());
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .with_context(|| format!("{name}: dtype"))?;
        let shape: Vec<usize> = serde_json::from_value(v.get("shape").cloned().unwrap_or_default())
            .with_context(|| format!("{name}: shape"))?;
        let [b, e]: [usize; 2] =
            serde_json::from_value(v.get("data_offsets").cloned().unwrap_or_default())
                .with_context(|| format!("{name}: data_offsets"))?;
        anyhow::ensure!(
            b <= e && e <= body.len(),
            "{name}: data [{b}, {e}) past {} bytes",
            body.len()
        );
        let bytes = &body[b..e];
        let count: usize = shape.iter().product();
        let w = match dtype {
            "I8" => {
                anyhow::ensure!(
                    bytes.len() == count,
                    "{name}: {} bytes for {shape:?}",
                    bytes.len()
                );
                Weight::I8(bytes.iter().map(|&x| x as i8).collect(), shape)
            }
            "U8" => {
                anyhow::ensure!(
                    bytes.len() == count,
                    "{name}: {} bytes for {shape:?}",
                    bytes.len()
                );
                Weight::U8(bytes.to_vec(), shape)
            }
            "F32" => {
                anyhow::ensure!(
                    bytes.len() == 4 * count,
                    "{name}: {} bytes for {shape:?}",
                    bytes.len()
                );
                let v = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect();
                Weight::F32(v, shape)
            }
            other => anyhow::bail!("{name}: dtype {other}, expected I8, U8 or F32"),
        };
        weights.0.insert(name, w);
    }
    Ok((config.context("no __metadata__")?, weights))
}

/// An int8 Linear: y = x W^T + bias, x's rows quantized to u8 first.
struct Linear8 {
    w: Packed,
    scale: Vec<f32>,
    bias: Vec<f32>,
    out: usize,
    inp: usize,
    /// the packed rows: `out` padded to 16s when blocked (zero rows; read by the step's timing)
    #[cfg_attr(not(test), allow(dead_code))]
    stride: usize,
}

#[derive(Default)]
struct LinScratch {
    q: Vec<u8>,
    qs: Vec<(f32, i32)>,
    acc: Vec<i32>,
}

impl Linear8 {
    /// `q` (out, inp) int8 rows with their `scale`s (times `scale_mul`), `bias` (out) or none.
    fn new(
        isa: Isa,
        q: &[i8],
        scale: &[f32],
        bias: Option<&[f32]>,
        out: usize,
        inp: usize,
        scale_mul: f32,
    ) -> Self {
        let blocked = isa != Isa::Plain && inp % 4 == 0;
        let stride = if blocked { out.div_ceil(16) * 16 } else { out };
        let mut w = q.to_vec();
        w.resize(stride * inp, 0);
        Linear8 {
            w: Packed::new(&w, stride, inp, blocked),
            scale: scale.iter().map(|&s| s * scale_mul).collect(),
            bias: bias.map_or_else(|| vec![0.0; out], <[f32]>::to_vec),
            out,
            inp,
            stride,
        }
    }

    /// y (n, out) of x (n, inp).
    #[inline(always)]
    fn forward(&self, isa: Isa, x: &[f32], n: usize, y: &mut Vec<f32>, s: &mut LinScratch) {
        let (inp, out) = (self.inp, self.out);
        s.q.resize(n * inp, 0);
        s.qs.clear();
        for r in 0..n {
            let qs = quantize_row(&x[r * inp..(r + 1) * inp], &mut s.q[r * inp..(r + 1) * inp]);
            s.qs.push(qs);
        }
        y.resize(n * out, 0.0);
        // sx * scale[o] * (acc - zx cs) + bias[o]: in the GEMM's registers with AVX-512 VNNI (`gemm_u8i8_f32`;
        // 3941823: the head's pass over the sums 23 of its 75 us at 20 hypotheses)
        let e = Epilogue {
            xs: &s.qs,
            col: Some(&self.scale[..out]),
            scale: 0.0,
            bias: Some(&self.bias[..out]),
            out,
        };
        gemm_u8i8_f32(isa, &s.q, n, &self.w, &e, &mut s.acc, y);
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    self_norm: Norm,
    /// the self-attention's q, k, v rows stacked (3 hidden)
    qkv: Linear8,
    self_out: Linear8,
    cross_norm: Norm,
    cross_q: Linear8,
    cross_out: Linear8,
    ffn_norm: Norm,
    linear1: Linear8,
    linear2: Linear8,
}

/// The script latent at choice points (the research export's script scores, the file's `script_*` tensors): the next
/// token's score is the better of its two readings, max over z in {0, class(y)} of log p(z) + log p(y | z), where
/// p(z) is the softmax of the prior's logits plus the search's bias, p(y | 0) the softmax over every token of the
/// logits plus class 0's offsets, and p(y | c) for c in 1..=3 the softmax over class c's tokens of the logits plus
/// class c's offsets. The class is not fed back, so a beam keeps the better reading of a token.
struct ScriptHead {
    /// (4, hidden) and (4): the prior's logits from the final state
    prior_w: Vec<f32>,
    prior_b: Vec<f32>,
    /// (4, vocab): each class's offset to the logits, the logit scale applied
    table: Vec<f32>,
    /// (vocab): each token's class, 0 other, 1 kanji, 2 hiragana, 3 katakana
    classes: Vec<u8>,
    /// (vocab): each token's own class's offset (`table[class][y]`; class 0's tokens: 0, unread), so a row's class
    /// terms are read in id order
    own: Vec<f32>,
}

impl ScriptHead {
    fn new(prior_w: Vec<f32>, prior_b: Vec<f32>, table: Vec<f32>, classes: Vec<u8>) -> Self {
        let v = classes.len();
        let own = classes
            .iter()
            .enumerate()
            .map(|(y, &c)| {
                if c == 0 {
                    0.0
                } else {
                    table[c as usize * v + y]
                }
            })
            .collect();
        ScriptHead {
            prior_w,
            prior_b,
            table,
            classes,
            own,
        }
    }
}

/// The decoder's step weights, prepacked.
pub struct NativeDecoder {
    pub config: NativeConfig,
    isa: Isa,
    gelu: bool,
    emb_q: Vec<i8>,
    emb_scale: Vec<f32>,
    positions: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Norm,
    /// the tied output projection (vocab rows), the logit scale in its row scales
    head: Linear8,
    script: Option<ScriptHead>,
}

impl NativeDecoder {
    pub fn load(path: &Path) -> Result<Self> {
        let (config, weights) = read(path)?;
        Self::new(config, &weights).with_context(|| format!("native AR step {}", path.display()))
    }

    pub fn new(config: NativeConfig, w: &Weights) -> Result<Self> {
        let c = &config;
        anyhow::ensure!(
            c.layers > 0
                && c.heads > 0
                && c.head_dim > 0
                && c.hidden == c.heads * c.head_dim
                && c.ffn > 0
                && c.vocab > 0
                && c.max_positions > 1,
            "config {c:?}: sizes must be positive and hidden = heads x head_dim"
        );
        anyhow::ensure!(
            matches!(c.activation.as_str(), "gelu" | "relu"),
            "activation {:?}, expected gelu or relu",
            c.activation
        );
        let isa = detect_isa();
        let (d, f, v) = (c.hidden, c.ffn, c.vocab);
        let lin = |name: &str, out: usize, inp: usize| -> Result<Linear8> {
            Ok(Linear8::new(
                isa,
                w.i8(&format!("{name}.q"), &[out, inp])?,
                w.f32(&format!("{name}.scale"), &[out])?,
                Some(w.f32(&format!("{name}.bias"), &[out])?),
                out,
                inp,
                1.0,
            ))
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                g: w.f32(&format!("{name}.weight"), &[d])?.to_vec(),
                b: w.f32(&format!("{name}.bias"), &[d])?.to_vec(),
            })
        };
        let emb_q = w.i8("embedding.q", &[v, d])?;
        let emb_scale = w.f32("embedding.scale", &[v])?;
        let layers = (0..c.layers)
            .map(|i| {
                let p = format!("layers.{i}.");
                Ok(Layer {
                    self_norm: norm(&format!("{p}self_norm"))?,
                    qkv: lin(&format!("{p}self_qkv"), 3 * d, d)?,
                    self_out: lin(&format!("{p}self_out"), d, d)?,
                    cross_norm: norm(&format!("{p}cross_norm"))?,
                    cross_q: lin(&format!("{p}cross_q"), d, d)?,
                    cross_out: lin(&format!("{p}cross_out"), d, d)?,
                    ffn_norm: norm(&format!("{p}ffn_norm"))?,
                    linear1: lin(&format!("{p}linear1"), f, d)?,
                    linear2: lin(&format!("{p}linear2"), d, f)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let script = match c.script.as_deref() {
            None => None,
            Some("choice") => {
                let classes: Vec<u8> = w
                    .i8("script_classes", &[v])?
                    .iter()
                    .map(|&x| u8::try_from(x).ok().filter(|&x| x < 4))
                    .collect::<Option<_>>()
                    .context("script_classes: a class outside 0..=3")?;
                Some(ScriptHead::new(
                    w.f32("script_prior.weight", &[4, d])?.to_vec(),
                    w.f32("script_prior.bias", &[4])?.to_vec(),
                    w.f32("script_table", &[4, v])?
                        .iter()
                        .map(|&x| x * c.logit_scale)
                        .collect(),
                    classes,
                ))
            }
            Some(other) => anyhow::bail!("script latent {other:?}, expected \"choice\""),
        };
        Ok(NativeDecoder {
            isa,
            gelu: c.activation == "gelu",
            emb_q: emb_q.to_vec(),
            emb_scale: emb_scale.to_vec(),
            positions: w.f32("positions", &[c.max_positions, d])?.to_vec(),
            layers,
            final_norm: norm("final_norm")?,
            head: Linear8::new(isa, emb_q, emb_scale, None, v, d, c.logit_scale),
            script,
            config,
        })
    }

    /// Whether the decoder has the script latent at choice points.
    pub fn script(&self) -> bool {
        self.script.is_some()
    }
}

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    o: Vec<f32>,
    t: Vec<f32>,
    ff: Vec<f32>,
    logits: Vec<f32>,
    /// one row's scores with the script latent
    row: Vec<f32>,
    scores: Vec<f32>,
    /// a head's scaled queries (the AVX-512 cross-attention)
    qh: Vec<f32>,
    offs: Vec<usize>,
    top: Vec<(f32, u32)>,
    lin: LinScratch,
}

/// One search's state over a [`NativeDecoder`] (see the module docs).
pub struct NativeSearch<'a> {
    dec: &'a NativeDecoder,
    frames: usize,
    /// (layers, heads, head_dim, T)
    cross_kt: Vec<f32>,
    /// with AVX-512 and 32-wide heads: the same keys with T padded to 16s (zeros), `cross_tp` frames a row; else
    /// empty
    cross_kt16: Vec<f32>,
    cross_tp: usize,
    /// (layers, heads, T, head_dim)
    cross_v: Vec<f32>,
    /// the best tokens a hypothesis gets
    k: usize,
    /// with the script latent: added to the prior's logits (other, kanji, hiragana, katakana)
    script_bias: [f32; 4],
    /// per layer, the keys and values of every position computed (rows of hidden)
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// per hypothesis of the last call: the arena rows of its prefix, and its own position's row
    prefix: Vec<Vec<u32>>,
    own: Vec<u32>,
    /// the positions run
    pub positions: usize,
    /// the hypotheses run, summed over the positions
    pub rows: usize,
    /// time in the stages so far, ns: self-attention (its norm and Linears included), cross-attention, FFN, the
    /// final norm and output projection, the log-softmax and best tokens (the scan), the script latent's terms
    /// (research timing)
    pub stage_ns: [u64; 6],
    s: Scratch,
}

impl<'a> NativeSearch<'a> {
    /// A search over the encoder's `cross_k` / `cross_v` (layers, 1, heads, `frames`, head_dim), giving `k` tokens
    /// a hypothesis a position.
    pub fn new(
        dec: &'a NativeDecoder,
        cross_k: &[f32],
        cross_v: Vec<f32>,
        frames: usize,
        k: usize,
    ) -> Result<Self> {
        let c = &dec.config;
        let (l, h, dh) = (c.layers, c.heads, c.head_dim);
        let n = l * h * frames * dh;
        anyhow::ensure!(
            cross_k.len() == n && cross_v.len() == n,
            "cross keys / values of {} / {} values, expected {n} ({l} layers, {h} heads, {frames} frames, {dh})",
            cross_k.len(),
            cross_v.len()
        );
        let mut cross_kt = vec![0f32; n];
        for lh in 0..l * h {
            let (src, dst) = (
                &cross_k[lh * frames * dh..(lh + 1) * frames * dh],
                &mut cross_kt[lh * dh * frames..],
            );
            for t in 0..frames {
                for e in 0..dh {
                    dst[e * frames + t] = src[t * dh + e];
                }
            }
        }
        // the AVX-512 cross-attention's keys: each head's rows padded to 16 frames
        #[cfg(target_arch = "x86_64")]
        let wide = dh == 32 && frames > 0 && crate::cpu::avx512();
        #[cfg(not(target_arch = "x86_64"))]
        let wide = false;
        let cross_tp = if wide { frames.div_ceil(16) * 16 } else { 0 };
        let mut cross_kt16 = vec![0f32; l * h * dh * cross_tp];
        if wide {
            for lh in 0..l * h {
                for e in 0..dh {
                    let (src, dst) = (lh * dh * frames + e * frames, (lh * dh + e) * cross_tp);
                    cross_kt16[dst..dst + frames].copy_from_slice(&cross_kt[src..src + frames]);
                }
            }
        }
        Ok(NativeSearch {
            dec,
            frames,
            cross_kt16,
            cross_tp,
            cross_kt,
            cross_v,
            k: k.max(1),
            script_bias: [0.0; 4],
            keys: vec![Vec::new(); l],
            values: vec![Vec::new(); l],
            prefix: Vec::new(),
            own: Vec::new(),
            positions: 0,
            rows: 0,
            stage_ns: [0; 6],
            s: Scratch::default(),
        })
    }

    /// The preference among the scripts of a decoder with the script latent: `bias` added to the prior's logits of
    /// other, kanji, hiragana, katakana at every position (zeros: none; a decoder without the latent ignores it).
    pub fn set_script_bias(&mut self, bias: [f32; 4]) {
        self.script_bias = bias;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    unsafe fn step_avx512(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop> {
        self.step_body(parents, tokens, position, forbidden)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn step_avx2(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop> {
        self.step_body(parents, tokens, position, forbidden)
    }

    /// The AVX level's build of the step (no AVX2 or FMA).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn step_avx1(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop> {
        self.step_body(parents, tokens, position, forbidden)
    }

    #[inline(always)]
    fn step_body(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop> {
        let NativeSearch {
            dec,
            frames,
            cross_kt,
            cross_kt16,
            cross_tp,
            cross_v,
            k,
            script_bias,
            keys,
            values,
            prefix: last_prefix,
            own: last_own,
            positions,
            rows,
            stage_ns,
            s,
        } = self;
        let dec: &NativeDecoder = *dec;
        let c = &dec.config;
        let (d, nh, dh, t, f, v, isa, eps) = (
            c.hidden, c.heads, c.head_dim, *frames, c.ffn, c.vocab, dec.isa, c.norm_eps,
        );
        let n = tokens.len();
        anyhow::ensure!(n > 0, "no hypotheses");
        anyhow::ensure!(
            position < c.max_positions,
            "position {position} past the decoder's {}",
            c.max_positions
        );
        anyhow::ensure!(
            forbidden.len() == v,
            "{} forbidden flags for {v} tokens",
            forbidden.len()
        );
        // each hypothesis's prefix: its parent's prefix and the parent's own position
        let prefix: Vec<Vec<u32>> = if position == 0 {
            vec![Vec::new(); n]
        } else {
            anyhow::ensure!(
                parents.len() == n,
                "{} parents for {n} hypotheses",
                parents.len()
            );
            parents
                .iter()
                .map(|&p| {
                    anyhow::ensure!(
                        p < last_prefix.len(),
                        "parent {p} of {} hypotheses",
                        last_prefix.len()
                    );
                    let mut rows = Vec::with_capacity(position);
                    rows.extend_from_slice(&last_prefix[p]);
                    rows.push(last_own[p]);
                    Ok(rows)
                })
                .collect::<Result<_>>()?
        };
        let base = keys[0].len() / d;
        let own: Vec<u32> = (base..base + n).map(|r| r as u32).collect();
        // the input rows: the token's embedding and the position's
        s.x.resize(n * d, 0.0);
        let pos = &dec.positions[position * d..(position + 1) * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            anyhow::ensure!(tok < v, "token {tok} of a vocabulary of {v}");
            let (q, sc) = (&dec.emb_q[tok * d..(tok + 1) * d], dec.emb_scale[tok]);
            for ((x, &qv), &p) in s.x[i * d..(i + 1) * d].iter_mut().zip(q).zip(pos) {
                *x = qv as f32 * sc + p;
            }
        }
        let scale = 1.0 / (dh as f32).sqrt();
        for (l, layer) in dec.layers.iter().enumerate() {
            let mut clock = std::time::Instant::now();
            let mut lap = |stage: usize, clock: &mut std::time::Instant| {
                stage_ns[stage] += clock.elapsed().as_nanos() as u64;
                *clock = std::time::Instant::now();
            };
            // causal self-attention: the prefix's keys and values, then the position's own
            layer_norm(&s.x, &layer.self_norm, eps, &mut s.h);
            layer.qkv.forward(isa, &s.h, n, &mut s.qkv, &mut s.lin);
            for i in 0..n {
                let row = &s.qkv[i * 3 * d..(i + 1) * 3 * d];
                keys[l].extend_from_slice(&row[d..2 * d]);
                values[l].extend_from_slice(&row[2 * d..]);
            }
            s.o.resize(n * d, 0.0);
            let (lk, lv) = (&keys[l], &values[l]);
            for i in 0..n {
                for h in 0..nh {
                    // the rows' offsets of this head: the prefix's, then the position's own
                    s.offs.clear();
                    for &r in &prefix[i] {
                        s.offs.push(r as usize * d + h * dh);
                    }
                    s.offs.push(own[i] as usize * d + h * dh);
                    let q = &s.qkv[i * 3 * d + h * dh..i * 3 * d + (h + 1) * dh];
                    s.scores.resize(s.offs.len(), 0.0);
                    for (sc, &off) in s.scores.iter_mut().zip(&s.offs) {
                        *sc = dot(q, &lk[off..off + dh]) * scale;
                    }
                    softmax(&mut s.scores);
                    mix(
                        &mut s.o[i * d + h * dh..i * d + (h + 1) * dh],
                        &s.scores,
                        lv,
                        &s.offs,
                    );
                }
            }
            layer.self_out.forward(isa, &s.o, n, &mut s.t, &mut s.lin);
            add(&mut s.x, &s.t);
            lap(0, &mut clock);
            // cross-attention over the encoder's frames (one unpadded row: no frame masked), a head at a time for
            // every hypothesis: the scores (n, T) a key dimension at a time, the outputs a frame at a time, so each
            // pass updates n rows in turn instead of one row back to back
            layer_norm(&s.x, &layer.cross_norm, eps, &mut s.h);
            layer.cross_q.forward(isa, &s.h, n, &mut s.q, &mut s.lin);
            for h in 0..nh {
                let lh = l * nh + h;
                #[cfg(target_arch = "x86_64")]
                if *cross_tp > 0 {
                    // the encoder's AVX-512 blocks (3941823: 2 us a hypothesis in the loops below)
                    let tp = *cross_tp;
                    // SAFETY: the CPU has the features (`new`); s.o holds n rows of d, this head's 32 columns written
                    unsafe {
                        crate::pipeline::native_encoder::cross_rows_avx512_32(
                            &s.q,
                            d,
                            h * dh,
                            n,
                            &cross_kt16[lh * dh * tp..(lh + 1) * dh * tp],
                            tp,
                            &cross_v[lh * t * dh..(lh + 1) * t * dh],
                            t,
                            s.o.as_mut_ptr().add(h * dh),
                            d,
                            &mut s.qh,
                            &mut s.scores,
                        );
                    }
                    continue;
                }
                let kt = &cross_kt[lh * dh * t..(lh + 1) * dh * t];
                s.scores.clear();
                s.scores.resize(n * t, 0.0);
                for e in 0..dh {
                    let krow = &kt[e * t..(e + 1) * t];
                    for (i, srow) in s.scores.chunks_exact_mut(t).enumerate() {
                        let qe = s.q[i * d + h * dh + e] * scale;
                        for (sv, &kv) in srow.iter_mut().zip(krow) {
                            *sv += qe * kv;
                        }
                    }
                }
                for srow in s.scores.chunks_exact_mut(t) {
                    softmax(srow);
                }
                let vv = &cross_v[lh * t * dh..(lh + 1) * t * dh];
                for i in 0..n {
                    s.o[i * d + h * dh..i * d + (h + 1) * dh].fill(0.0);
                }
                for tt in 0..t {
                    let vrow = &vv[tt * dh..(tt + 1) * dh];
                    for i in 0..n {
                        let p = s.scores[i * t + tt];
                        for (ov, &x) in s.o[i * d + h * dh..i * d + (h + 1) * dh]
                            .iter_mut()
                            .zip(vrow)
                        {
                            *ov += p * x;
                        }
                    }
                }
            }
            layer.cross_out.forward(isa, &s.o, n, &mut s.t, &mut s.lin);
            add(&mut s.x, &s.t);
            lap(1, &mut clock);
            // the FFN
            layer_norm(&s.x, &layer.ffn_norm, eps, &mut s.h);
            layer.linear1.forward(isa, &s.h, n, &mut s.ff, &mut s.lin);
            debug_assert_eq!(s.ff.len(), n * f);
            if dec.gelu {
                s.ff.iter_mut().for_each(|x| *x = gelu(*x));
            } else {
                s.ff.iter_mut()
                    .for_each(|x| *x = if *x > 0.0 { *x } else { 0.0 });
            }
            layer.linear2.forward(isa, &s.ff, n, &mut s.t, &mut s.lin);
            add(&mut s.x, &s.t);
            lap(2, &mut clock);
        }
        let clock = std::time::Instant::now();
        layer_norm(&s.x, &dec.final_norm, eps, &mut s.h);
        dec.head.forward(isa, &s.h, n, &mut s.logits, &mut s.lin);
        stage_ns[3] += clock.elapsed().as_nanos() as u64;
        let clock = std::time::Instant::now();
        let k = *k;
        let mut top = StepTop {
            k,
            ids: Vec::with_capacity(n * k),
            log_probs: Vec::with_capacity(n * k),
        };
        let mut script_ns = 0u64;
        for i in 0..n {
            let row = &s.logits[i * v..(i + 1) * v];
            match &dec.script {
                None => top_k_log_probs(
                    row,
                    forbidden,
                    k,
                    &mut s.top,
                    &mut top.ids,
                    &mut top.log_probs,
                ),
                Some(head) => {
                    // the script terms with the best tokens' scan (one pass since 2026-11-19), timed apart from the
                    // rest (research timing)
                    let t0 = std::time::Instant::now();
                    let h = &s.h[i * d..(i + 1) * d];
                    script_top_k(
                        row,
                        h,
                        head,
                        script_bias,
                        forbidden,
                        k,
                        &mut s.row,
                        &mut s.top,
                        &mut top.ids,
                        &mut top.log_probs,
                    );
                    script_ns += t0.elapsed().as_nanos() as u64;
                }
            }
        }
        let all = clock.elapsed().as_nanos() as u64;
        stage_ns[5] += script_ns;
        stage_ns[4] += all.saturating_sub(script_ns);
        *last_prefix = prefix;
        *last_own = own;
        *positions = position + 1;
        *rows += n;
        Ok(top)
    }
}

impl StepModel for NativeSearch<'_> {
    fn vocab(&self) -> usize {
        self.dec.config.vocab
    }

    fn step(
        &mut self,
        parents: &[usize],
        tokens: &[u32],
        position: usize,
        forbidden: &[bool],
    ) -> Result<StepTop> {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::cpu::avx512() {
                // SAFETY: the CPU supports the target features.
                return unsafe { self.step_avx512(parents, tokens, position, forbidden) };
            }
            if crate::cpu::avx2() {
                // SAFETY: as above.
                return unsafe { self.step_avx2(parents, tokens, position, forbidden) };
            }
            if crate::cpu::avx() {
                // SAFETY: as above.
                return unsafe { self.step_avx1(parents, tokens, position, forbidden) };
            }
        }
        self.step_body(parents, tokens, position, forbidden)
    }
}

/// sum of `v` in 16 partial sums (vectorizes).
#[inline(always)]
pub(crate) fn lane_sum(v: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let chunks = v.chunks_exact(16);
    let tail: f32 = chunks.remainder().iter().sum();
    for c in chunks {
        for i in 0..16 {
            acc[i] += c[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// the largest value of `v` (compare-selects in 16 lanes, which vectorize; NaN skipped).
#[inline(always)]
pub(crate) fn row_max(v: &[f32]) -> f32 {
    let mut m = [f32::NEG_INFINITY; 16];
    let chunks = v.chunks_exact(16);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..16 {
            m[i] = if c[i] > m[i] { c[i] } else { m[i] };
        }
    }
    m.iter()
        .chain(rest)
        .fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a })
}

/// LayerNorm of each row of `x` (rows of `n.g.len()`), the biased variance as PyTorch's.
#[inline(always)]
fn layer_norm(x: &[f32], n: &Norm, eps: f32, y: &mut Vec<f32>) {
    let d = n.g.len();
    y.resize(x.len(), 0.0);
    for (xr, yr) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let mean = lane_sum(xr) / d as f32;
        for (yv, &xv) in yr.iter_mut().zip(xr) {
            *yv = (xv - mean) * (xv - mean);
        }
        let inv = 1.0 / (lane_sum(yr) / d as f32 + eps).sqrt();
        for (((yv, &xv), &g), &b) in yr.iter_mut().zip(xr).zip(&n.g).zip(&n.b) {
            *yv = (xv - mean) * inv * g + b;
        }
    }
}

#[inline(always)]
fn add(x: &mut [f32], y: &[f32]) {
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// o = sum_j p[j] * vals[offs[j]..offs[j] + o.len()], four rows a pass over o.
#[inline(always)]
fn mix(o: &mut [f32], p: &[f32], vals: &[f32], offs: &[usize]) {
    let dh = o.len();
    o.fill(0.0);
    let mut j = 0;
    while j + 4 <= offs.len() {
        let (a, b) = (
            &vals[offs[j]..offs[j] + dh],
            &vals[offs[j + 1]..offs[j + 1] + dh],
        );
        let (c, e) = (
            &vals[offs[j + 2]..offs[j + 2] + dh],
            &vals[offs[j + 3]..offs[j + 3] + dh],
        );
        let (pa, pb, pc, pe) = (p[j], p[j + 1], p[j + 2], p[j + 3]);
        for ((((ov, &x), &y), &z), &w) in o.iter_mut().zip(a).zip(b).zip(c).zip(e) {
            *ov += pa * x + pb * y + pc * z + pe * w;
        }
        j += 4;
    }
    for (&off, &pj) in offs[j..].iter().zip(&p[j..]) {
        for (ov, &x) in o.iter_mut().zip(&vals[off..off + dh]) {
            *ov += pj * x;
        }
    }
}

/// softmax of `v` in place.
#[inline(always)]
fn softmax(v: &mut [f32]) {
    let m = row_max(v);
    for x in v.iter_mut() {
        *x = lse::exp(*x - m);
    }
    let inv = 1.0 / lane_sum(v);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// The `k` best tokens of a logits `row` that `forbidden` allows, best first (equal logits: the lower id first),
/// appended to `ids` with their log-probabilities over the whole row (`log_probs`); a row with fewer allowed tokens
/// is padded with id 0 at -inf. `buf` is scratch.
///
/// One pass in id order keeping the best `k` so far (`offer`); once `k` are kept, a 16-wide chunk with nothing above
/// the k-th is skipped by one vector compare (2026-11-12: the first version gathered every token within 8 nats of
/// the best and selected among them, 8 us a row of 8006 on a flat row).
#[inline(always)]
fn top_k_log_probs(
    row: &[f32],
    forbidden: &[bool],
    k: usize,
    buf: &mut Vec<(f32, u32)>,
    ids: &mut Vec<u32>,
    log_probs: &mut Vec<f32>,
) {
    top_k_shifted(row, forbidden, k, buf, ids, log_probs, row_lse(row));
}

/// A row's next-token scores with the script latent ([`ScriptHead`]): `logits` the row's (the logit scale
/// applied), `h` its final state, `bias` the search's on the prior; the scores (log-probabilities of each token's
/// better reading) into `out`.
#[inline(always)]
fn script_scores(
    logits: &[f32],
    h: &[f32],
    head: &ScriptHead,
    bias: &[f32; 4],
    out: &mut Vec<f32>,
) {
    let log_pz = script_prior(h, head, bias);
    // every value is written below: the buffer kept at the row's length, not zeroed each row (32 KB a hypothesis)
    out.resize(logits.len(), 0.0);
    #[cfg(target_arch = "x86_64")]
    if lse::avx512f() {
        // SAFETY: the CPU has AVX-512 F
        return unsafe { script_x86::class_terms_avx512(logits, head, log_pz, out) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx2() {
        // SAFETY: the CPU has AVX2 and FMA
        return unsafe { script_x86::class_terms_avx2(logits, head, log_pz, out) };
    }
    class_terms_lanes(logits, head, log_pz, out)
}

/// The best `k` tokens of a row's [`script_scores`] that `forbidden` allows, as `top_k_shifted` of the row with no
/// shift: with AVX-512 or AVX2 each 16 or 8 tokens' scores made in registers and only those above the k-th offered
/// (2026-11-19: the row of scores was written, read back and scanned, 32 KB a hypothesis), the same scores and the
/// same offers in id order, so the same tokens and values; elsewhere the row (`row`) and its scan.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn script_top_k(
    logits: &[f32],
    h: &[f32],
    head: &ScriptHead,
    bias: &[f32; 4],
    forbidden: &[bool],
    k: usize,
    row: &mut Vec<f32>,
    buf: &mut Vec<(f32, u32)>,
    ids: &mut Vec<u32>,
    log_probs: &mut Vec<f32>,
) {
    #[cfg(target_arch = "x86_64")]
    if lse::avx512f() || crate::cpu::avx2() {
        let log_pz = script_prior(h, head, bias);
        buf.clear();
        // SAFETY: the CPU has AVX-512 F, or AVX2 and FMA
        unsafe {
            if lse::avx512f() {
                script_x86::script_top_avx512(logits, head, log_pz, forbidden, k, buf)
            } else {
                script_x86::script_top_avx2(logits, head, log_pz, forbidden, k, buf)
            }
        }
        return push_top(buf, k, ids, log_probs, 0.0);
    }
    script_scores(logits, h, head, bias, row);
    top_k_shifted(row, forbidden, k, buf, ids, log_probs, 0.0);
}

/// The script classes' log-probabilities from a row's final state `h`, the search's `bias` added to the logits.
#[inline(always)]
fn script_prior(h: &[f32], head: &ScriptHead, bias: &[f32; 4]) -> [f32; 4] {
    let d = h.len();
    let mut p = [0f32; 4];
    for (z, pz) in p.iter_mut().enumerate() {
        *pz = dot(h, &head.prior_w[z * d..(z + 1) * d]) + head.prior_b[z] + bias[z];
    }
    let m = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse_p = m + p.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    p.map(|x| x - lse_p)
}

/// `script_scores`' terms given the prior's log-probabilities `pz`: class 0 reads every token (logits + table[0]),
/// classes 1-3 their own (logits + `own`), each normalized over its tokens; the scores into `out` (the row's length).
/// 16 tokens at a time without branches, a token's terms selected by its class (2026-11-19: the per-token branches
/// and table reads were most of the AR model's search).
#[inline(always)]
fn class_terms_lanes(logits: &[f32], head: &ScriptHead, pz: [f32; 4], out: &mut [f32]) {
    let v = logits.len();
    let (t0, own, cls) = (&head.table[..v], &head.own[..v], &head.classes[..v]);
    for ((o, &a), &b) in out.iter_mut().zip(logits).zip(t0) {
        *o = a + b;
    }
    let lse0 = row_lse(out);
    let whole = v - v % LANES;
    let mut mxl = [[f32::NEG_INFINITY; LANES]; 3];
    for j in (0..whole).step_by(LANES) {
        let (lg, ow, cl) = (
            &logits[j..j + LANES],
            &own[j..j + LANES],
            &cls[j..j + LANES],
        );
        for (k, mk) in mxl.iter_mut().enumerate() {
            for i in 0..LANES {
                let e = if cl[i] as usize == k + 1 {
                    lg[i] + ow[i]
                } else {
                    f32::NEG_INFINITY
                };
                mk[i] = if e > mk[i] { e } else { mk[i] };
            }
        }
    }
    let mut mx = [f32::NEG_INFINITY; 4];
    for (k, mk) in mxl.iter().enumerate() {
        mx[k + 1] = mk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
    for y in whole..v {
        let c = cls[y] as usize;
        if c != 0 {
            mx[c] = mx[c].max(logits[y] + own[y]);
        }
    }
    let mut suml = [[0f32; LANES]; 3];
    for j in (0..whole).step_by(LANES) {
        let (lg, ow, cl) = (
            &logits[j..j + LANES],
            &own[j..j + LANES],
            &cls[j..j + LANES],
        );
        for i in 0..LANES {
            let c = cl[i];
            let m = if c == 1 {
                mx[1]
            } else if c == 2 {
                mx[2]
            } else {
                mx[3]
            };
            // class 0's lanes: an exponential of a finite value (the clamp), not added
            let t = lse::exp(lg[i] + ow[i] - m);
            for (k, sk) in suml.iter_mut().enumerate() {
                sk[i] += if c as usize == k + 1 { t } else { 0.0 };
            }
        }
    }
    let mut sum = [0f32; 4];
    for (k, sk) in suml.iter().enumerate() {
        sum[k + 1] = sk.iter().sum();
    }
    for y in whole..v {
        let c = cls[y] as usize;
        if c != 0 {
            sum[c] += lse::exp(logits[y] + own[y] - mx[c]);
        }
    }
    let lc = [0usize, 1, 2, 3].map(|c| mx[c] + sum[c].ln());
    for j in (0..whole).step_by(LANES) {
        let (lg, ow, cl) = (
            &logits[j..j + LANES],
            &own[j..j + LANES],
            &cls[j..j + LANES],
        );
        let o = &mut out[j..j + LANES];
        for i in 0..LANES {
            let c = cl[i];
            let free = pz[0] + o[i] - lse0;
            let (p, l) = if c == 1 {
                (pz[1], lc[1])
            } else if c == 2 {
                (pz[2], lc[2])
            } else {
                (pz[3], lc[3])
            };
            let alt = p + lg[i] + ow[i] - l;
            o[i] = if c != 0 && alt > free { alt } else { free };
        }
    }
    for y in whole..v {
        let free = pz[0] + out[y] - lse0;
        let c = cls[y] as usize;
        out[y] = if c == 0 {
            free
        } else {
            free.max(pz[c] + logits[y] + own[y] - lc[c])
        };
    }
}

/// The script terms in AVX-512 and AVX2 intrinsics ([`script_scores`], [`script_top_k`]): one pass for the class 0
/// terms' maximum and every class's, one for the sums (an exponential of each term), one for the scores; a token's
/// class by a mask of the tokens' classes, its class 0 term (logits + table[0]) made again in each pass (the same
/// add, the same bits as storing it). (73b0865's lanes took 14 us a row, 2e6ed32's chunk skipping more: a row's
/// terms are rarely all 24 nats below the maximum; 69fec4c: the lanes 1.6 ms of the AVX2 level's AR search.)
#[cfg(target_arch = "x86_64")]
mod script_x86 {
    use super::{offer, ScriptHead};
    use crate::pipeline::lse;
    use std::arch::x86_64::*;

    /// A token's score past the last whole chunk: the free reading, or its class's when better.
    #[inline(always)]
    fn tail_score(
        logits: &[f32],
        head: &ScriptHead,
        pz: &[f32; 4],
        lse0: f32,
        lc: &[f32; 4],
        y: usize,
    ) -> f32 {
        let free = pz[0] + (logits[y] + head.table[y]) - lse0;
        let c = head.classes[y] as usize;
        if c == 0 {
            free
        } else {
            free.max(pz[c] + logits[y] + head.own[y] - lc[c])
        }
    }

    /// A row's sums' tail (the tokens past the last whole chunk) and its normalizers: class 0's log-sum-exp and each
    /// class's (`mx`, `sum` the whole chunks' maxima and sums).
    #[inline(always)]
    fn tails(
        logits: &[f32],
        head: &ScriptHead,
        whole: usize,
        mut max0: f32,
        mut mx: [f32; 4],
    ) -> (f32, [f32; 4]) {
        let v = logits.len();
        for y in whole..v {
            max0 = max0.max(logits[y] + head.table[y]);
            let c = head.classes[y] as usize;
            if c != 0 {
                mx[c] = mx[c].max(logits[y] + head.own[y]);
            }
        }
        (max0, mx)
    }

    #[inline(always)]
    fn sum_tails(
        logits: &[f32],
        head: &ScriptHead,
        whole: usize,
        max0: f32,
        mx: &[f32; 4],
        mut sum0: f32,
        mut sum: [f32; 4],
    ) -> (f32, [f32; 4]) {
        for y in whole..logits.len() {
            sum0 += lse::exp(logits[y] + head.table[y] - max0);
            let c = head.classes[y] as usize;
            if c != 0 {
                sum[c] += lse::exp(logits[y] + head.own[y] - mx[c]);
            }
        }
        (
            max0 + sum0.ln(),
            [0usize, 1, 2, 3].map(|c| mx[c] + sum[c].ln()),
        )
    }

    // ---- AVX-512 ----

    /// The masks of 16 tokens' classes 1, 2, 3 (`c` their u8 classes).
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn class_masks(c: *const u8) -> [__mmask16; 3] {
        let c = _mm512_cvtepu8_epi32(_mm_loadu_si128(c as *const __m128i));
        [
            _mm512_cmpeq_epi32_mask(c, _mm512_set1_epi32(1)),
            _mm512_cmpeq_epi32_mask(c, _mm512_set1_epi32(2)),
            _mm512_cmpeq_epi32_mask(c, _mm512_set1_epi32(3)),
        ]
    }

    /// A row's normalizers (the first two passes): class 0's log-sum-exp and each class's.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn norms16(logits: &[f32], head: &ScriptHead) -> (f32, [f32; 4]) {
        let v = logits.len();
        let (t0, own, cls) = (&head.table[..v], &head.own[..v], &head.classes[..v]);
        let whole = v - v % 16;
        let (lp, tp, op, cp) = (logits.as_ptr(), t0.as_ptr(), own.as_ptr(), cls.as_ptr());
        let ninf = _mm512_set1_ps(f32::NEG_INFINITY);
        let (mut mx0, mut mxv) = (ninf, [ninf; 3]);
        for j in (0..whole).step_by(16) {
            let lg = _mm512_loadu_ps(lp.add(j));
            mx0 = _mm512_max_ps(_mm512_add_ps(lg, _mm512_loadu_ps(tp.add(j))), mx0);
            let e = _mm512_add_ps(lg, _mm512_loadu_ps(op.add(j)));
            let m = class_masks(cp.add(j));
            for k in 0..3 {
                mxv[k] = _mm512_mask_max_ps(mxv[k], m[k], e, mxv[k]);
            }
        }
        let mut mx = [f32::NEG_INFINITY; 4];
        for k in 0..3 {
            mx[k + 1] = _mm512_reduce_max_ps(mxv[k]);
        }
        let (max0, mx) = tails(logits, head, whole, _mm512_reduce_max_ps(mx0), mx);
        // the sums: every token's class 0 term, and its own class's
        let vmax0 = _mm512_set1_ps(max0);
        let vmx = [
            _mm512_set1_ps(mx[1]),
            _mm512_set1_ps(mx[2]),
            _mm512_set1_ps(mx[3]),
        ];
        let (mut s0, mut sv) = (_mm512_setzero_ps(), [_mm512_setzero_ps(); 3]);
        for j in (0..whole).step_by(16) {
            let lg = _mm512_loadu_ps(lp.add(j));
            s0 = _mm512_add_ps(
                s0,
                lse::exp16_le0(_mm512_sub_ps(
                    _mm512_add_ps(lg, _mm512_loadu_ps(tp.add(j))),
                    vmax0,
                )),
            );
            let e = _mm512_add_ps(lg, _mm512_loadu_ps(op.add(j)));
            let m = class_masks(cp.add(j));
            // each lane's own class's maximum (class 0's lanes: class 3's, their term not added)
            let mc = _mm512_mask_mov_ps(_mm512_mask_mov_ps(vmx[2], m[1], vmx[1]), m[0], vmx[0]);
            let t = lse::exp16_le0(_mm512_sub_ps(e, mc));
            for k in 0..3 {
                sv[k] = _mm512_mask_add_ps(sv[k], m[k], sv[k], t);
            }
        }
        let mut sum = [0f32; 4];
        for k in 0..3 {
            sum[k + 1] = _mm512_reduce_add_ps(sv[k]);
        }
        sum_tails(
            logits,
            head,
            whole,
            max0,
            &mx,
            _mm512_reduce_add_ps(s0),
            sum,
        )
    }

    /// A row's constants of the scores, broadcast: the prior's log-probabilities and the normalizers.
    struct Terms16 {
        p0: __m512,
        l0: __m512,
        p: [__m512; 3],
        l: [__m512; 3],
    }

    /// 16 tokens' scores from `j`: the free reading, or the class's when better (`class_terms_lanes`' arithmetic).
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn scores16(
        lp: *const f32,
        tp: *const f32,
        op: *const f32,
        cp: *const u8,
        j: usize,
        t: &Terms16,
    ) -> __m512 {
        let lg = _mm512_loadu_ps(lp.add(j));
        let free = _mm512_sub_ps(
            _mm512_add_ps(t.p0, _mm512_add_ps(lg, _mm512_loadu_ps(tp.add(j)))),
            t.l0,
        );
        let m = class_masks(cp.add(j));
        let p = _mm512_mask_mov_ps(_mm512_mask_mov_ps(t.p[2], m[1], t.p[1]), m[0], t.p[0]);
        let l = _mm512_mask_mov_ps(_mm512_mask_mov_ps(t.l[2], m[1], t.l[1]), m[0], t.l[0]);
        let alt = _mm512_sub_ps(
            _mm512_add_ps(_mm512_add_ps(p, lg), _mm512_loadu_ps(op.add(j))),
            l,
        );
        let better = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(alt, free) & (m[0] | m[1] | m[2]);
        _mm512_mask_mov_ps(free, better, alt)
    }

    /// The row's scores into `out` (the row's length).
    ///
    /// # Safety
    /// The CPU has AVX-512 F.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn class_terms_avx512(
        logits: &[f32],
        head: &ScriptHead,
        pz: [f32; 4],
        out: &mut [f32],
    ) {
        let v = logits.len();
        assert!(
            out.len() == v
                && head.table.len() >= v
                && head.own.len() >= v
                && head.classes.len() >= v
        );
        let (lse0, lc) = norms16(logits, head);
        let s = |x: f32| _mm512_set1_ps(x);
        let t = Terms16 {
            p0: s(pz[0]),
            l0: s(lse0),
            p: [s(pz[1]), s(pz[2]), s(pz[3])],
            l: [s(lc[1]), s(lc[2]), s(lc[3])],
        };
        let whole = v - v % 16;
        let (lp, tp, op, cp, outp) = (
            logits.as_ptr(),
            head.table.as_ptr(),
            head.own.as_ptr(),
            head.classes.as_ptr(),
            out.as_mut_ptr(),
        );
        for j in (0..whole).step_by(16) {
            _mm512_storeu_ps(outp.add(j), scores16(lp, tp, op, cp, j, &t));
        }
        for y in whole..v {
            out[y] = tail_score(logits, head, &pz, lse0, &lc, y);
        }
    }

    /// The best `k` of the row's scores that `forbidden` allows into `buf` (`offer`'s order), the scores made 16 at
    /// a time in registers and those above the k-th offered in id order: `top_k_scan_avx512` of `class_terms_avx512`'s
    /// row, without the row.
    ///
    /// # Safety
    /// The CPU has AVX-512 F.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn script_top_avx512(
        logits: &[f32],
        head: &ScriptHead,
        pz: [f32; 4],
        forbidden: &[bool],
        k: usize,
        buf: &mut Vec<(f32, u32)>,
    ) {
        let v = logits.len();
        assert!(
            forbidden.len() >= v
                && head.table.len() >= v
                && head.own.len() >= v
                && head.classes.len() >= v
        );
        let (lse0, lc) = norms16(logits, head);
        let s = |x: f32| _mm512_set1_ps(x);
        let t = Terms16 {
            p0: s(pz[0]),
            l0: s(lse0),
            p: [s(pz[1]), s(pz[2]), s(pz[3])],
            l: [s(lc[1]), s(lc[2]), s(lc[3])],
        };
        let whole = v - v % 16;
        let (lp, tp, op, cp) = (
            logits.as_ptr(),
            head.table.as_ptr(),
            head.own.as_ptr(),
            head.classes.as_ptr(),
        );
        let mut thr = f32::NEG_INFINITY;
        let mut lanes = [0f32; 16];
        for j in (0..whole).step_by(16) {
            let sc = scores16(lp, tp, op, cp, j, &t);
            let mut hit = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(sc, _mm512_set1_ps(thr)) as u32;
            if hit == 0 {
                continue;
            }
            _mm512_storeu_ps(lanes.as_mut_ptr(), sc);
            while hit != 0 {
                let i = hit.trailing_zeros() as usize;
                hit &= hit - 1;
                let (x, id) = (lanes[i], j + i);
                if x > thr && !forbidden[id] {
                    offer(buf, k, x, id as u32, &mut thr);
                }
            }
        }
        for y in whole..v {
            let x = tail_score(logits, head, &pz, lse0, &lc, y);
            if x > thr && !forbidden[y] {
                offer(buf, k, x, y as u32, &mut thr);
            }
        }
    }

    // ---- AVX2 ----

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hmax8(v: __m256) -> f32 {
        let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let m = _mm_max_ss(m, _mm_shuffle_ps::<1>(m, m));
        _mm_cvtss_f32(m)
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum8(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    /// The masks of 8 tokens' classes 1, 2, 3 (`c` their u8 classes).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn class_masks8(c: *const u8) -> [__m256; 3] {
        let c = _mm256_cvtepu8_epi32(_mm_loadl_epi64(c as *const __m128i));
        [
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(c, _mm256_set1_epi32(1))),
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(c, _mm256_set1_epi32(2))),
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(c, _mm256_set1_epi32(3))),
        ]
    }

    /// `norms16` 8 tokens at a time, each class by a compare of the tokens' classes and blends.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn norms8(logits: &[f32], head: &ScriptHead) -> (f32, [f32; 4]) {
        let v = logits.len();
        let (t0, own, cls) = (&head.table[..v], &head.own[..v], &head.classes[..v]);
        let whole = v - v % 8;
        let (lp, tp, op, cp) = (logits.as_ptr(), t0.as_ptr(), own.as_ptr(), cls.as_ptr());
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let (mut mx0, mut mxv) = (ninf, [ninf; 3]);
        for j in (0..whole).step_by(8) {
            let lg = _mm256_loadu_ps(lp.add(j));
            mx0 = _mm256_max_ps(_mm256_add_ps(lg, _mm256_loadu_ps(tp.add(j))), mx0);
            let e = _mm256_add_ps(lg, _mm256_loadu_ps(op.add(j)));
            let m = class_masks8(cp.add(j));
            for k in 0..3 {
                mxv[k] = _mm256_blendv_ps(mxv[k], _mm256_max_ps(e, mxv[k]), m[k]);
            }
        }
        let mut mx = [f32::NEG_INFINITY; 4];
        for k in 0..3 {
            mx[k + 1] = hmax8(mxv[k]);
        }
        let (max0, mx) = tails(logits, head, whole, hmax8(mx0), mx);
        let vmax0 = _mm256_set1_ps(max0);
        let vmx = [
            _mm256_set1_ps(mx[1]),
            _mm256_set1_ps(mx[2]),
            _mm256_set1_ps(mx[3]),
        ];
        let (mut s0, mut sv) = (_mm256_setzero_ps(), [_mm256_setzero_ps(); 3]);
        for j in (0..whole).step_by(8) {
            let lg = _mm256_loadu_ps(lp.add(j));
            s0 = _mm256_add_ps(
                s0,
                lse::exp8(_mm256_sub_ps(
                    _mm256_add_ps(lg, _mm256_loadu_ps(tp.add(j))),
                    vmax0,
                )),
            );
            let e = _mm256_add_ps(lg, _mm256_loadu_ps(op.add(j)));
            let m = class_masks8(cp.add(j));
            // each lane's own class's maximum (class 0's lanes: class 3's, their term not added)
            let mc = _mm256_blendv_ps(_mm256_blendv_ps(vmx[2], vmx[1], m[1]), vmx[0], m[0]);
            let t = lse::exp8(_mm256_sub_ps(e, mc));
            for k in 0..3 {
                sv[k] = _mm256_add_ps(sv[k], _mm256_and_ps(t, m[k]));
            }
        }
        let mut sum = [0f32; 4];
        for k in 0..3 {
            sum[k + 1] = hsum8(sv[k]);
        }
        sum_tails(logits, head, whole, max0, &mx, hsum8(s0), sum)
    }

    struct Terms8 {
        p0: __m256,
        l0: __m256,
        p: [__m256; 3],
        l: [__m256; 3],
    }

    /// `scores16` for 8 tokens.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn scores8(
        lp: *const f32,
        tp: *const f32,
        op: *const f32,
        cp: *const u8,
        j: usize,
        t: &Terms8,
    ) -> __m256 {
        let lg = _mm256_loadu_ps(lp.add(j));
        let free = _mm256_sub_ps(
            _mm256_add_ps(t.p0, _mm256_add_ps(lg, _mm256_loadu_ps(tp.add(j)))),
            t.l0,
        );
        let m = class_masks8(cp.add(j));
        let p = _mm256_blendv_ps(_mm256_blendv_ps(t.p[2], t.p[1], m[1]), t.p[0], m[0]);
        let l = _mm256_blendv_ps(_mm256_blendv_ps(t.l[2], t.l[1], m[1]), t.l[0], m[0]);
        let alt = _mm256_sub_ps(
            _mm256_add_ps(_mm256_add_ps(p, lg), _mm256_loadu_ps(op.add(j))),
            l,
        );
        let any = _mm256_or_ps(_mm256_or_ps(m[0], m[1]), m[2]);
        let better = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(alt, free), any);
        _mm256_blendv_ps(free, alt, better)
    }

    /// `class_terms_avx512` with AVX2.
    ///
    /// # Safety
    /// The CPU has AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn class_terms_avx2(
        logits: &[f32],
        head: &ScriptHead,
        pz: [f32; 4],
        out: &mut [f32],
    ) {
        let v = logits.len();
        assert!(
            out.len() == v
                && head.table.len() >= v
                && head.own.len() >= v
                && head.classes.len() >= v
        );
        let (lse0, lc) = norms8(logits, head);
        let s = |x: f32| _mm256_set1_ps(x);
        let t = Terms8 {
            p0: s(pz[0]),
            l0: s(lse0),
            p: [s(pz[1]), s(pz[2]), s(pz[3])],
            l: [s(lc[1]), s(lc[2]), s(lc[3])],
        };
        let whole = v - v % 8;
        let (lp, tp, op, cp, outp) = (
            logits.as_ptr(),
            head.table.as_ptr(),
            head.own.as_ptr(),
            head.classes.as_ptr(),
            out.as_mut_ptr(),
        );
        for j in (0..whole).step_by(8) {
            _mm256_storeu_ps(outp.add(j), scores8(lp, tp, op, cp, j, &t));
        }
        for y in whole..v {
            out[y] = tail_score(logits, head, &pz, lse0, &lc, y);
        }
    }

    /// `script_top_avx512` with AVX2: the offers of `top_k_scan` on `class_terms_avx2`'s row (every value above the
    /// k-th then, in id order; a value under the k-th at its chunk's start stays under it), without the row.
    ///
    /// # Safety
    /// The CPU has AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn script_top_avx2(
        logits: &[f32],
        head: &ScriptHead,
        pz: [f32; 4],
        forbidden: &[bool],
        k: usize,
        buf: &mut Vec<(f32, u32)>,
    ) {
        let v = logits.len();
        assert!(
            forbidden.len() >= v
                && head.table.len() >= v
                && head.own.len() >= v
                && head.classes.len() >= v
        );
        let (lse0, lc) = norms8(logits, head);
        let s = |x: f32| _mm256_set1_ps(x);
        let t = Terms8 {
            p0: s(pz[0]),
            l0: s(lse0),
            p: [s(pz[1]), s(pz[2]), s(pz[3])],
            l: [s(lc[1]), s(lc[2]), s(lc[3])],
        };
        let whole = v - v % 8;
        let (lp, tp, op, cp) = (
            logits.as_ptr(),
            head.table.as_ptr(),
            head.own.as_ptr(),
            head.classes.as_ptr(),
        );
        let mut thr = f32::NEG_INFINITY;
        let mut lanes = [0f32; 8];
        for j in (0..whole).step_by(8) {
            let sc = scores8(lp, tp, op, cp, j, &t);
            let mut hit =
                _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(sc, _mm256_set1_ps(thr))) as u32;
            if hit == 0 {
                continue;
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), sc);
            while hit != 0 {
                let i = hit.trailing_zeros() as usize;
                hit &= hit - 1;
                let (x, id) = (lanes[i], j + i);
                if x > thr && !forbidden[id] {
                    offer(buf, k, x, id as u32, &mut thr);
                }
            }
        }
        for y in whole..v {
            let x = tail_score(logits, head, &pz, lse0, &lc, y);
            if x > thr && !forbidden[y] {
                offer(buf, k, x, y as u32, &mut thr);
            }
        }
    }
}

/// 16 lanes: the AVX-512 width of f32, the AVX2 width twice.
const LANES: usize = 16;

/// [`top_k_log_probs`] of a row less `shift` (its log-sum-exp for log-probabilities, 0 for scores already
/// normalized).
#[inline(always)]
fn top_k_shifted(
    row: &[f32],
    forbidden: &[bool],
    k: usize,
    buf: &mut Vec<(f32, u32)>,
    ids: &mut Vec<u32>,
    log_probs: &mut Vec<f32>,
    lse: f32,
) {
    buf.clear();
    #[cfg(target_arch = "x86_64")]
    let scanned = lse::avx512f() && {
        // SAFETY: the CPU has AVX-512 F
        unsafe { top_k_scan_avx512(row, forbidden, k, buf) };
        true
    };
    #[cfg(not(target_arch = "x86_64"))]
    let scanned = false;
    if !scanned {
        top_k_scan(row, forbidden, k, buf);
    }
    push_top(buf, k, ids, log_probs, lse);
}

/// `buf`'s tokens (best first) into `ids` and their values less `shift` into `log_probs`, padded to `k` (id 0 at
/// -inf).
#[inline(always)]
fn push_top(
    buf: &[(f32, u32)],
    k: usize,
    ids: &mut Vec<u32>,
    log_probs: &mut Vec<f32>,
    shift: f32,
) {
    for &(x, id) in buf {
        ids.push(id);
        log_probs.push(x - shift);
    }
    for _ in buf.len()..k {
        ids.push(0);
        log_probs.push(f32::NEG_INFINITY);
    }
}

/// The best `k` of `row` that `forbidden` allows into `buf` (`offer`'s order): one pass in id order keeping the best
/// `k` so far; once `k` are kept, a 16-wide chunk with nothing above the k-th is skipped by one test (2026-11-12: the
/// first version gathered every token within 8 nats of the best and selected among them, 8 us a row of 8006 on a
/// flat row).
#[inline(always)]
fn top_k_scan(row: &[f32], forbidden: &[bool], k: usize, buf: &mut Vec<(f32, u32)>) {
    let mut thr = f32::NEG_INFINITY;
    let chunks = row.chunks_exact(16);
    let rest = row.len() - chunks.remainder().len();
    for (ci, c) in chunks.enumerate() {
        let mut hit = false;
        for &x in c {
            hit |= x > thr;
        }
        if hit {
            for (j, &x) in c.iter().enumerate() {
                let id = ci * 16 + j;
                if x > thr && !forbidden[id] {
                    offer(buf, k, x, id as u32, &mut thr);
                }
            }
        }
    }
    for (id, &x) in row.iter().enumerate().skip(rest) {
        if x > thr && !forbidden[id] {
            offer(buf, k, x, id as u32, &mut thr);
        }
    }
}

/// `top_k_scan` with AVX-512 (3941823: 3.4 us a row of 8006 by the 16-wide test): a chunk's values above the k-th
/// as one mask, then those alone offered in id order (a value the mask leaves out is at most the k-th at the chunk's
/// start, which only rises: the plain scan would not keep it either).
///
/// # Safety
/// The CPU has AVX-512 F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn top_k_scan_avx512(row: &[f32], forbidden: &[bool], k: usize, buf: &mut Vec<(f32, u32)>) {
    use std::arch::x86_64::*;
    let mut thr = f32::NEG_INFINITY;
    let n = row.len();
    let whole = n - n % 16;
    let p = row.as_ptr();
    for j in (0..whole).step_by(16) {
        let mut m =
            _mm512_cmp_ps_mask::<_CMP_GT_OQ>(_mm512_loadu_ps(p.add(j)), _mm512_set1_ps(thr)) as u32;
        while m != 0 {
            let id = j + m.trailing_zeros() as usize;
            m &= m - 1;
            let x = row[id];
            if x > thr && !forbidden[id] {
                offer(buf, k, x, id as u32, &mut thr);
            }
        }
    }
    for (id, &x) in row.iter().enumerate().skip(whole) {
        if x > thr && !forbidden[id] {
            offer(buf, k, x, id as u32, &mut thr);
        }
    }
}

/// [`lse::peaked`]: the log-sum-exp leaving out 16-wide chunks with nothing within 24 nats of the row's best.
#[inline(always)]
fn row_lse(row: &[f32]) -> f32 {
    lse::peaked(row)
}

/// Keep (x, id) among the best `k` of `buf` (best first; ids offered in increasing order, so an equal value goes
/// after the ones kept, the lower id first); `thr` becomes the k-th value once `k` are kept.
#[inline(always)]
fn offer(buf: &mut Vec<(f32, u32)>, k: usize, x: f32, id: u32, thr: &mut f32) {
    let mut i = buf.len();
    while i > 0 && buf[i - 1].0 < x {
        i -= 1;
    }
    buf.insert(i, (x, id));
    if buf.len() > k {
        buf.pop();
    }
    if buf.len() == k {
        *thr = buf[k - 1].0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        /// uniform in [-1, 1)
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) * 2.0
                - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    fn config() -> NativeConfig {
        NativeConfig {
            layers: 2,
            hidden: 32,
            heads: 2,
            head_dim: 16,
            // not a multiple of 16: linear1's rows padded
            ffn: 40,
            // the output rows padded too
            vocab: 50,
            max_positions: 12,
            activation: "gelu".into(),
            norm_eps: 1e-5,
            logit_scale: 32f32.powf(-0.5),
            bos_id: 3,
            eos_id: 2,
            script: None,
        }
    }

    /// int8 rows with their scales, as the export writes them.
    fn q8(w: &[f32], out: usize) -> (Vec<i8>, Vec<f32>) {
        let inp = w.len() / out;
        let mut q = Vec::with_capacity(w.len());
        let mut s = Vec::with_capacity(out);
        for r in w.chunks_exact(inp) {
            let sc = r.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-12) / 127.0;
            q.extend(
                r.iter()
                    .map(|&x| (x / sc).round().clamp(-127.0, 127.0) as i8),
            );
            s.push(sc);
        }
        (q, s)
    }

    fn weights(c: &NativeConfig, seed: u64) -> Weights {
        let mut rng = Rng(seed);
        let (d, f, v) = (c.hidden, c.ffn, c.vocab);
        let mut w = Weights::default();
        let lin = |w: &mut Weights, rng: &mut Rng, name: &str, out: usize, inp: usize| {
            let (q, s) = q8(&rng.vec(out * inp, 0.4), out);
            w.0.insert(format!("{name}.q"), Weight::I8(q, vec![out, inp]));
            w.0.insert(format!("{name}.scale"), Weight::F32(s, vec![out]));
            w.0.insert(
                format!("{name}.bias"),
                Weight::F32(rng.vec(out, 0.1), vec![out]),
            );
        };
        let norm = |w: &mut Weights, rng: &mut Rng, name: &str| {
            w.0.insert(
                format!("{name}.weight"),
                Weight::F32(rng.vec(d, 0.2).iter().map(|x| 1.0 + x).collect(), vec![d]),
            );
            w.0.insert(
                format!("{name}.bias"),
                Weight::F32(rng.vec(d, 0.1), vec![d]),
            );
        };
        let (q, s) = q8(&rng.vec(v * d, 1.0), v);
        w.0.insert("embedding.q".into(), Weight::I8(q, vec![v, d]));
        w.0.insert("embedding.scale".into(), Weight::F32(s, vec![v]));
        w.0.insert(
            "positions".into(),
            Weight::F32(rng.vec(c.max_positions * d, 0.3), vec![c.max_positions, d]),
        );
        for i in 0..c.layers {
            let p = format!("layers.{i}.");
            norm(&mut w, &mut rng, &format!("{p}self_norm"));
            lin(&mut w, &mut rng, &format!("{p}self_qkv"), 3 * d, d);
            lin(&mut w, &mut rng, &format!("{p}self_out"), d, d);
            norm(&mut w, &mut rng, &format!("{p}cross_norm"));
            lin(&mut w, &mut rng, &format!("{p}cross_q"), d, d);
            lin(&mut w, &mut rng, &format!("{p}cross_out"), d, d);
            norm(&mut w, &mut rng, &format!("{p}ffn_norm"));
            lin(&mut w, &mut rng, &format!("{p}linear1"), f, d);
            lin(&mut w, &mut rng, &format!("{p}linear2"), d, f);
        }
        norm(&mut w, &mut rng, "final_norm");
        w
    }

    /// The decoder in f64 over a whole prefix (no cache, no activation quantization): the log-probabilities of the
    /// token after `prefix` (the beginning first).
    fn reference(
        c: &NativeConfig,
        w: &Weights,
        ck: &[f32],
        cv: &[f32],
        t: usize,
        prefix: &[u32],
    ) -> Vec<f64> {
        let (_, logits) = reference_parts(c, w, ck, cv, t, prefix);
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z = logits.iter().map(|x| (x - m).exp()).sum::<f64>().ln() + m;
        logits.iter().map(|x| x - z).collect()
    }

    /// [`reference`]'s final state after `prefix` and its logits (the logit scale applied).
    fn reference_parts(
        c: &NativeConfig,
        w: &Weights,
        ck: &[f32],
        cv: &[f32],
        t: usize,
        prefix: &[u32],
    ) -> (Vec<f64>, Vec<f64>) {
        let (d, nh, dh, f, v) = (c.hidden, c.heads, c.head_dim, c.ffn, c.vocab);
        let deq = |name: &str, out: usize, inp: usize| -> Vec<f64> {
            let q = w.i8(&format!("{name}.q"), &[out, inp]).unwrap();
            let s = w.f32(&format!("{name}.scale"), &[out]).unwrap();
            q.iter()
                .enumerate()
                .map(|(i, &x)| x as f64 * s[i / inp] as f64)
                .collect()
        };
        let f64s = |name: &str, n: usize| -> Vec<f64> {
            w.f32(name, &[n])
                .unwrap()
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        let linear = |x: &[f64], name: &str, out: usize, inp: usize| -> Vec<f64> {
            let (m, b) = (deq(name, out, inp), f64s(&format!("{name}.bias"), out));
            x.chunks(inp)
                .flat_map(|r| {
                    (0..out)
                        .map(|o| b[o] + (0..inp).map(|i| r[i] * m[o * inp + i]).sum::<f64>())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let norm = |x: &[f64], name: &str| -> Vec<f64> {
            let (g, b) = (
                f64s(&format!("{name}.weight"), d),
                f64s(&format!("{name}.bias"), d),
            );
            x.chunks(d)
                .flat_map(|r| {
                    let mean = r.iter().sum::<f64>() / d as f64;
                    let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d as f64;
                    (0..d)
                        .map(|i| (r[i] - mean) / (var + c.norm_eps as f64).sqrt() * g[i] + b[i])
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let softmax = |s: &mut Vec<f64>| {
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            s.iter_mut().for_each(|x| *x = (*x - m).exp() / z);
        };
        let emb = deq("embedding", v, d);
        let pos = w.f32("positions", &[c.max_positions, d]).unwrap();
        let len = prefix.len();
        let mut x: Vec<f64> = (0..len)
            .flat_map(|j| (0..d).map(move |e| (j, e)))
            .map(|(j, e)| emb[prefix[j] as usize * d + e] + pos[j * d + e] as f64)
            .collect();
        let scale = 1.0 / (dh as f64).sqrt();
        for l in 0..c.layers {
            let p = format!("layers.{l}.");
            let qkv = linear(
                &norm(&x, &format!("{p}self_norm")),
                &format!("{p}self_qkv"),
                3 * d,
                d,
            );
            let mut o = vec![0f64; len * d];
            for j in 0..len {
                for h in 0..nh {
                    let q = &qkv[j * 3 * d + h * dh..][..dh];
                    let mut s: Vec<f64> = (0..=j)
                        .map(|r| {
                            (0..dh)
                                .map(|e| q[e] * qkv[r * 3 * d + d + h * dh + e])
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    softmax(&mut s);
                    for (r, p) in s.iter().enumerate() {
                        for e in 0..dh {
                            o[j * d + h * dh + e] += p * qkv[r * 3 * d + 2 * d + h * dh + e];
                        }
                    }
                }
            }
            let y = linear(&o, &format!("{p}self_out"), d, d);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            let q2 = linear(
                &norm(&x, &format!("{p}cross_norm")),
                &format!("{p}cross_q"),
                d,
                d,
            );
            let mut o = vec![0f64; len * d];
            for j in 0..len {
                for h in 0..nh {
                    let base = (l * nh + h) * t * dh;
                    let mut s: Vec<f64> = (0..t)
                        .map(|tt| {
                            (0..dh)
                                .map(|e| q2[j * d + h * dh + e] * ck[base + tt * dh + e] as f64)
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    softmax(&mut s);
                    for (tt, p) in s.iter().enumerate() {
                        for e in 0..dh {
                            o[j * d + h * dh + e] += p * cv[base + tt * dh + e] as f64;
                        }
                    }
                }
            }
            let y = linear(&o, &format!("{p}cross_out"), d, d);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            let hdn: Vec<f64> = linear(
                &norm(&x, &format!("{p}ffn_norm")),
                &format!("{p}linear1"),
                f,
                d,
            )
            .iter()
            .map(|&z| 0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2)))
            .collect();
            let y = linear(&hdn, &format!("{p}linear2"), d, f);
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        }
        let last = norm(&x, "final_norm")[(len - 1) * d..].to_vec();
        let logits: Vec<f64> = (0..v)
            .map(|o| (0..d).map(|e| last[e] * emb[o * d + e]).sum::<f64>() * c.logit_scale as f64)
            .collect();
        (last, logits)
    }

    /// The script latent's tensors as the export writes them: the prior's head, each class's offsets (before the
    /// logit scale), each token's class (its id mod 4).
    fn with_script(c: &NativeConfig, mut w: Weights, seed: u64) -> Weights {
        let mut rng = Rng(seed);
        let (d, v) = (c.hidden, c.vocab);
        w.0.insert(
            "script_prior.weight".into(),
            Weight::F32(rng.vec(4 * d, 0.5), vec![4, d]),
        );
        w.0.insert(
            "script_prior.bias".into(),
            Weight::F32(rng.vec(4, 0.5), vec![4]),
        );
        w.0.insert(
            "script_table".into(),
            Weight::F32(rng.vec(4 * v, 2.0), vec![4, v]),
        );
        w.0.insert(
            "script_classes".into(),
            Weight::I8((0..v).map(|y| (y % 4) as i8).collect(), vec![v]),
        );
        w
    }

    /// The script latent's scores in f64 from the reference's state and logits (the export's `script_scores`).
    fn reference_script(
        c: &NativeConfig,
        w: &Weights,
        last: &[f64],
        logits: &[f64],
        bias: [f32; 4],
    ) -> Vec<f64> {
        let (d, v) = (c.hidden, c.vocab);
        let pw = w.f32("script_prior.weight", &[4, d]).unwrap();
        let pb = w.f32("script_prior.bias", &[4]).unwrap();
        let table = w.f32("script_table", &[4, v]).unwrap();
        let classes = w.i8("script_classes", &[v]).unwrap();
        let lse = |xs: &[f64]| {
            let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
        };
        let p: Vec<f64> = (0..4)
            .map(|z| {
                (0..d).map(|e| last[e] * pw[z * d + e] as f64).sum::<f64>()
                    + (pb[z] + bias[z]) as f64
            })
            .collect();
        let lp = lse(&p);
        let scaled =
            |z: usize, y: usize| logits[y] + table[z * v + y] as f64 * c.logit_scale as f64;
        let norm: Vec<f64> = (0..4)
            .map(|z| {
                let row: Vec<f64> = (0..v)
                    .filter(|&y| z == 0 || classes[y] as usize == z)
                    .map(|y| scaled(z, y))
                    .collect();
                lse(&row)
            })
            .collect();
        (0..v)
            .map(|y| {
                let free = p[0] - lp + scaled(0, y) - norm[0];
                let c = classes[y] as usize;
                if c == 0 {
                    free
                } else {
                    free.max(p[c] - lp + scaled(c, y) - norm[c])
                }
            })
            .collect()
    }

    /// The script latent's step: three positions of a small beam with a bias on the prior, each hypothesis's best
    /// tokens against the f64 scores (each token's better reading) within the int8 error and none of the
    /// reference's better tokens missed; a large hiragana bias puts a hiragana token first; a decoder without the
    /// latent refuses none of it (the bias ignored).
    #[test]
    fn script_steps_score_each_tokens_better_reading() {
        let c = NativeConfig {
            script: Some("choice".into()),
            ..config()
        };
        let w = with_script(&c, weights(&c, 7), 5);
        let dec = NativeDecoder::new(c.clone(), &w).unwrap();
        assert!(dec.script());
        let t = 9;
        let mut rng = Rng(11);
        let n = c.layers * c.heads * t * c.head_dim;
        let (ck, cv) = (rng.vec(n, 1.0), rng.vec(n, 1.0));
        let mut forbidden = vec![false; c.vocab];
        forbidden[0] = true;
        let (k, bias) = (5, [0.0, 0.5, 2.0, -1.0]);
        let mut search = NativeSearch::new(&dec, &ck, cv.clone(), t, k).unwrap();
        search.set_script_bias(bias);
        let steps: [(&[usize], &[u32]); 3] =
            [(&[], &[3]), (&[0, 0], &[7, 9]), (&[1, 0, 1], &[11, 12, 13])];
        let mut prefixes: Vec<Vec<u32>> = vec![vec![]];
        let mut worst = 0f64;
        for (position, (parents, tokens)) in steps.iter().enumerate() {
            prefixes = if position == 0 {
                vec![vec![3]]
            } else {
                parents
                    .iter()
                    .zip(*tokens)
                    .map(|(&p, &tok)| [prefixes[p].clone(), vec![tok]].concat())
                    .collect()
            };
            let top = search.step(parents, tokens, position, &forbidden).unwrap();
            for (i, pre) in prefixes.iter().enumerate() {
                let (last, logits) = reference_parts(&c, &w, &ck, &cv, t, pre);
                let want = reference_script(&c, &w, &last, &logits, bias);
                let ids = &top.ids[i * k..(i + 1) * k];
                let lps = &top.log_probs[i * k..(i + 1) * k];
                assert!(lps.windows(2).all(|p| p[0] >= p[1]));
                for (&id, &lp) in ids.iter().zip(lps) {
                    assert!(!forbidden[id as usize]);
                    worst = worst.max((want[id as usize] - lp as f64).abs());
                }
                let mut allowed: Vec<f64> = (0..c.vocab)
                    .filter(|&j| !forbidden[j])
                    .map(|j| want[j])
                    .collect();
                allowed.sort_by(|a, b| b.total_cmp(a));
                assert!(
                    allowed[k - 1] <= lps[k - 1] as f64 + 0.05,
                    "{allowed:?} {lps:?}"
                );
            }
        }
        assert!(
            worst < 0.05,
            "native script scores off the f64 reference by {worst}"
        );
        // a large hiragana bias: the best token is a hiragana one (class 2: id mod 4 == 2)
        let mut search = NativeSearch::new(&dec, &ck, cv.clone(), t, k).unwrap();
        search.set_script_bias([0.0, 0.0, 30.0, 0.0]);
        let top = search.step(&[], &[3], 0, &forbidden).unwrap();
        assert_eq!(top.ids[0] % 4, 2, "{:?}", top.ids);
        // the same weights without the latent: plain log-probabilities, the bias ignored
        let plain = NativeDecoder::new(config(), &w).unwrap();
        assert!(!plain.script());
        let mut search = NativeSearch::new(&plain, &ck, cv.clone(), t, k).unwrap();
        search.set_script_bias([0.0, 0.0, 30.0, 0.0]);
        let top = search.step(&[], &[3], 0, &forbidden).unwrap();
        let want = reference(&config(), &w, &ck, &cv, t, &[3]);
        for (&id, &lp) in top.ids.iter().zip(&top.log_probs) {
            assert!(
                (want[id as usize] - lp as f64).abs() < 0.05,
                "{id}: {lp} against {}",
                want[id as usize]
            );
        }
    }

    /// A config naming another latent, or classes outside 0..=3, is refused.
    #[test]
    fn refuses_a_script_latent_it_does_not_know() {
        let c = NativeConfig {
            script: Some("within".into()),
            ..config()
        };
        let w = with_script(&c, weights(&c, 7), 5);
        assert!(NativeDecoder::new(c, &w).is_err());
        let c = NativeConfig {
            script: Some("choice".into()),
            ..config()
        };
        let mut w = with_script(&c, weights(&c, 7), 5);
        w.0.insert(
            "script_classes".into(),
            Weight::I8(vec![4; c.vocab], vec![c.vocab]),
        );
        assert!(NativeDecoder::new(c, &w).is_err());
    }

    /// erf to ~1e-7 (Abramowitz & Stegun 7.1.26 is 1.5e-7).
    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        if x >= 0.0 {
            y
        } else {
            -y
        }
    }

    /// Three positions of a small beam through the native step (parents reordering the cache), each hypothesis's
    /// best tokens against the f64 decoder run over its whole prefix: the log-probabilities within the activations'
    /// int8 error, and none of the reference's better tokens missed by more than that.
    #[test]
    fn steps_follow_the_decoder_over_each_prefix() {
        // 16-wide heads, and 32-wide ones (the AVX-512 cross-attention where the CPU has it) over 9 and 21 frames
        steps_follow(config(), 9);
        let wide = NativeConfig {
            hidden: 64,
            heads: 2,
            head_dim: 32,
            ..config()
        };
        steps_follow(wide.clone(), 9);
        steps_follow(wide, 21);
    }

    fn steps_follow(c: NativeConfig, t: usize) {
        let w = weights(&c, 7);
        let dec = NativeDecoder::new(c.clone(), &w).unwrap();
        let mut rng = Rng(11);
        let n = c.layers * c.heads * t * c.head_dim;
        let (ck, cv) = (rng.vec(n, 1.0), rng.vec(n, 1.0));
        let mut forbidden = vec![false; c.vocab];
        forbidden[0] = true;
        let k = 5;
        let mut search = NativeSearch::new(&dec, &ck, cv.clone(), t, k).unwrap();
        // position 0: the beginning; 1: two children of it; 2: three, the first and third of the second child
        let steps: [(&[usize], &[u32]); 3] =
            [(&[], &[3]), (&[0, 0], &[7, 9]), (&[1, 0, 1], &[11, 12, 13])];
        let mut prefixes: Vec<Vec<u32>> = vec![vec![]];
        let mut worst = 0f64;
        for (position, (parents, tokens)) in steps.iter().enumerate() {
            prefixes = if position == 0 {
                vec![vec![3]]
            } else {
                parents
                    .iter()
                    .zip(*tokens)
                    .map(|(&p, &tok)| [prefixes[p].clone(), vec![tok]].concat())
                    .collect()
            };
            let top = search.step(parents, tokens, position, &forbidden).unwrap();
            assert_eq!((top.k, top.ids.len()), (k, tokens.len() * k));
            for (i, pre) in prefixes.iter().enumerate() {
                let want = reference(&c, &w, &ck, &cv, t, pre);
                let ids = &top.ids[i * k..(i + 1) * k];
                let lps = &top.log_probs[i * k..(i + 1) * k];
                assert!(ids.iter().all(|&id| !forbidden[id as usize]));
                assert!(lps.windows(2).all(|p| p[0] >= p[1]));
                for (&id, &lp) in ids.iter().zip(lps) {
                    worst = worst.max((want[id as usize] - lp as f64).abs());
                }
                // the reference's k-th best allowed token is no better than the native's k-th by more than the error
                let mut allowed: Vec<f64> = (0..c.vocab)
                    .filter(|&j| !forbidden[j])
                    .map(|j| want[j])
                    .collect();
                allowed.sort_by(|a, b| b.total_cmp(a));
                assert!(
                    allowed[k - 1] <= lps[k - 1] as f64 + 0.05,
                    "{allowed:?} {lps:?}"
                );
            }
        }
        assert!(
            worst < 0.05,
            "native log-probs off the f64 decoder by {worst}"
        );
        assert_eq!(search.positions, 3);
    }

    /// The best tokens of rows of the export's vocabulary size against a sort: peaked and flat rows, ties, forbidden
    /// tokens among the best, fewer allowed tokens than asked.
    #[test]
    fn top_k_matches_a_sort() {
        let mut rng = Rng(3);
        let v = 8006;
        let mut rows: Vec<Vec<f32>> = vec![rng.vec(v, 1.0), rng.vec(v, 20.0)];
        let mut peaked = rng.vec(v, 3.0);
        peaked[17] = 40.0;
        peaked[8000] = 39.0;
        peaked[4000] = 39.0;
        rows.push(peaked);
        rows.push(vec![0.5; v]);
        let mut forbidden = vec![false; v];
        for f in [17usize, 1, 2, 4] {
            forbidden[f] = true;
        }
        let mut buf = Vec::new();
        for row in &rows {
            for k in [1usize, 8, 32] {
                let (mut ids, mut lps) = (Vec::new(), Vec::new());
                top_k_log_probs(row, &forbidden, k, &mut buf, &mut ids, &mut lps);
                let mut want: Vec<(f32, u32)> = row
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| !forbidden[j])
                    .map(|(j, &x)| (x, j as u32))
                    .collect();
                want.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                let z = lse::exact(row);
                assert_eq!(ids, want[..k].iter().map(|w| w.1).collect::<Vec<_>>());
                // the plain scan (the path without AVX-512) keeps the same
                let mut plain = Vec::new();
                top_k_scan(row, &forbidden, k, &mut plain);
                assert_eq!(plain.iter().map(|p| p.1).collect::<Vec<_>>(), ids);
                for (lp, w) in lps.iter().zip(&want) {
                    assert!((lp - (w.0 - z)).abs() < 1e-4, "{lp} {}", w.0 - z);
                }
            }
        }
        let tiny = [1.0f32, 3.0, 2.0];
        let (mut ids, mut lps) = (Vec::new(), Vec::new());
        top_k_log_probs(
            &tiny,
            &[false, true, false],
            3,
            &mut buf,
            &mut ids,
            &mut lps,
        );
        assert_eq!(ids, vec![2, 0, 0]);
        assert!(lps[1].is_finite() && lps[2] == f32::NEG_INFINITY);
    }

    /// The script scores as written first (a branch and a table read a token, the class sums in id order).
    fn script_scores_by_token(
        logits: &[f32],
        h: &[f32],
        head: &ScriptHead,
        bias: &[f32; 4],
    ) -> Vec<f32> {
        let (v, d) = (logits.len(), h.len());
        let mut p = [0f32; 4];
        for (z, pz) in p.iter_mut().enumerate() {
            *pz = dot(h, &head.prior_w[z * d..(z + 1) * d]) + head.prior_b[z] + bias[z];
        }
        let m = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse_p = m + p.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
        let log_pz = p.map(|x| x - lse_p);
        let mut out: Vec<f32> = logits
            .iter()
            .zip(&head.table[..v])
            .map(|(&a, &b)| a + b)
            .collect();
        let lse0 = row_lse(&out);
        let mut mx = [f32::NEG_INFINITY; 4];
        for (y, &c) in head.classes.iter().enumerate() {
            let c = c as usize;
            if c != 0 {
                mx[c] = mx[c].max(logits[y] + head.table[c * v + y]);
            }
        }
        let mut sum = [0f32; 4];
        for (y, &c) in head.classes.iter().enumerate() {
            let c = c as usize;
            if c != 0 {
                sum[c] += lse::exp(logits[y] + head.table[c * v + y] - mx[c]);
            }
        }
        let lse_c = [0usize, 1, 2, 3].map(|c| mx[c] + sum[c].ln());
        for (y, &c) in head.classes.iter().enumerate() {
            let free = log_pz[0] + out[y] - lse0;
            let c = c as usize;
            out[y] = if c == 0 {
                free
            } else {
                free.max(log_pz[c] + logits[y] + head.table[c * v + y] - lse_c[c])
            };
        }
        out
    }

    /// `exp16` gives `lse::exp`'s bits, over its range, past its clamps, and on NaN and the infinities.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp16_is_lse_exp() {
        if !lse::avx512f() {
            return;
        }
        #[target_feature(enable = "avx512f")]
        unsafe fn of(a: [f32; 16]) -> [f32; 16] {
            use std::arch::x86_64::*;
            let mut out = [0f32; 16];
            _mm512_storeu_ps(out.as_mut_ptr(), lse::exp16(_mm512_loadu_ps(a.as_ptr())));
            out
        }
        let xs: Vec<f32> = (0..4800)
            .map(|i| -110.0 + i as f32 * 0.0437)
            .chain([
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                88.3,
                -87.3,
                1e-30,
                -1e-30,
            ])
            .collect();
        for c in xs.chunks(16) {
            let mut a = [0f32; 16];
            a[..c.len()].copy_from_slice(c);
            // SAFETY: the CPU has AVX-512 F
            let got = unsafe { of(a) };
            for i in 0..c.len() {
                assert_eq!(got[i].to_bits(), lse::exp(a[i]).to_bits(), "e^{}", a[i]);
            }
        }
    }

    /// `exp16_le0` gives `lse::exp`'s bits on values at most 0: a fine sweep to past the lower clamp, every float from
    /// -87 to past the clamp (the smallest results, 2^fx at -126), the floats just under 0 and -1, NaN and -inf.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp16_le0_is_lse_exp() {
        if !lse::avx512f() {
            return;
        }
        #[target_feature(enable = "avx512f")]
        unsafe fn of(a: [f32; 16]) -> [f32; 16] {
            use std::arch::x86_64::*;
            let mut out = [0f32; 16];
            _mm512_storeu_ps(
                out.as_mut_ptr(),
                lse::exp16_le0(_mm512_loadu_ps(a.as_ptr())),
            );
            out
        }
        let xs: Vec<f32> = (0..200_000)
            .map(|i| -95.0 + i as f32 * 0.000_475)
            .filter(|&x| x <= 0.0)
            .chain((0..4096u32).map(|i| -f32::from_bits(i)))
            .chain((0..4096u32).map(|i| -f32::from_bits(0x3f80_0000 - i)))
            .chain((0..45_000u32).map(|i| f32::from_bits(0xc2ae_0000 + i)))
            .chain([
                f32::NAN,
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                -87.3,
                -87.299,
                -87.301,
                -1e-30,
            ])
            .collect();
        for c in xs.chunks(16) {
            let mut a = [0f32; 16];
            a[..c.len()].copy_from_slice(c);
            // SAFETY: the CPU has AVX-512 F
            let got = unsafe { of(a) };
            for i in 0..c.len() {
                assert_eq!(got[i].to_bits(), lse::exp(a[i]).to_bits(), "e^{}", a[i]);
            }
        }
    }

    /// `exp8` (the AVX2 level's) gives `lse::exp`'s bits likewise.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp8_is_lse_exp() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        #[target_feature(enable = "avx2,fma")]
        unsafe fn of(a: [f32; 8]) -> [f32; 8] {
            use std::arch::x86_64::*;
            let mut out = [0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), lse::exp8(_mm256_loadu_ps(a.as_ptr())));
            out
        }
        let xs: Vec<f32> = (0..4800)
            .map(|i| -110.0 + i as f32 * 0.0437)
            .chain([
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                88.3,
                -87.3,
                1e-30,
                -1e-30,
            ])
            .collect();
        for c in xs.chunks(8) {
            let mut a = [0f32; 8];
            a[..c.len()].copy_from_slice(c);
            // SAFETY: the CPU has AVX2 and FMA
            let got = unsafe { of(a) };
            for i in 0..c.len() {
                assert_eq!(got[i].to_bits(), lse::exp(a[i]).to_bits(), "e^{}", a[i]);
            }
        }
    }

    /// The scores 16 tokens at a time against the token-by-token ones: an 8006-token row (not a multiple of 16) with
    /// the four classes mixed, flat, peaked and with a class of no tokens, within the class sums' order of addition.
    #[test]
    fn script_scores_by_lanes_match_by_token() {
        let (v, d) = (8006usize, 24usize);
        let mut rng = Rng(21);
        let mut classes: Vec<u8> = (0..v)
            .map(|_| ((rng.next() + 1.0) * 2.0) as u8 % 4)
            .collect();
        let table = rng.vec(4 * v, 2.0);
        let (prior_w, prior_b) = (rng.vec(4 * d, 0.5), rng.vec(4, 0.5));
        let h = rng.vec(d, 1.0);
        let mut rows = vec![rng.vec(v, 1.0), rng.vec(v, 20.0), vec![0.25; v]];
        let mut peaked = rng.vec(v, 3.0);
        peaked[5] = 40.0;
        peaked[8003] = 41.0;
        rows.push(peaked);
        for no_katakana in [false, true] {
            if no_katakana {
                classes.iter_mut().filter(|c| **c == 3).for_each(|c| *c = 1);
            }
            let head = ScriptHead::new(
                prior_w.clone(),
                prior_b.clone(),
                table.clone(),
                classes.clone(),
            );
            for row in &rows {
                for bias in [[0.0f32; 4], [0.0, 0.5, 2.0, -1.0]] {
                    let want = script_scores_by_token(row, &h, &head, &bias);
                    // the CPU's path, and the lanes (the path without AVX-512)
                    let mut got = Vec::new();
                    script_scores(row, &h, &head, &bias, &mut got);
                    let mut lanes = vec![0f32; v];
                    class_terms_lanes(row, &head, script_prior(&h, &head, &bias), &mut lanes);
                    // the AVX2 level's intrinsics where this CPU has AVX2 (else the lanes again)
                    let mut wide = lanes.clone();
                    #[cfg(target_arch = "x86_64")]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // SAFETY: the CPU has AVX2 and FMA
                        unsafe {
                            script_x86::class_terms_avx2(
                                row,
                                &head,
                                script_prior(&h, &head, &bias),
                                &mut wide,
                            )
                        };
                    }
                    for got in [&got[..], &lanes[..], &wide[..]] {
                        assert_eq!(got.len(), v);
                        for (y, (&a, &b)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                (a - b).abs() <= 1e-5 * (1.0 + b.abs()),
                                "token {y}: {a} against {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The best tokens of the scores made in registers (`script_top_k`) against the scan of the row of scores: the
    /// same tokens and the same bits, on the CPU's path and the AVX2 level's, with tokens forbidden and ties (the flat
    /// row), k of 1, 8 and 16.
    #[test]
    fn script_top_k_is_the_scan_of_the_scores() {
        let (v, d) = (8006usize, 24usize);
        let mut rng = Rng(33);
        let classes: Vec<u8> = (0..v)
            .map(|_| ((rng.next() + 1.0) * 2.0) as u8 % 4)
            .collect();
        let table = rng.vec(4 * v, 2.0);
        let (prior_w, prior_b) = (rng.vec(4 * d, 0.5), rng.vec(4, 0.5));
        let h = rng.vec(d, 1.0);
        let head = ScriptHead::new(prior_w, prior_b, table, classes);
        let mut rows = vec![rng.vec(v, 1.0), rng.vec(v, 20.0), vec![0.25; v]];
        let mut peaked = rng.vec(v, 3.0);
        peaked[5] = 40.0;
        peaked[8003] = 41.0;
        rows.push(peaked);
        let forbidden: Vec<bool> = (0..v).map(|y| y % 7 == 3 || y == 8003).collect();
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for row in &rows {
            for bias in [[0.0f32; 4], [0.0, 0.5, 2.0, -1.0]] {
                for k in [1usize, 8, 16] {
                    let mut scores = Vec::new();
                    script_scores(row, &h, &head, &bias, &mut scores);
                    let (mut buf, mut ids, mut lps) = (Vec::new(), Vec::new(), Vec::new());
                    top_k_shifted(&scores, &forbidden, k, &mut buf, &mut ids, &mut lps, 0.0);
                    let (mut scratch, mut got_ids, mut got_lps) =
                        (Vec::new(), Vec::new(), Vec::new());
                    script_top_k(
                        row,
                        &h,
                        &head,
                        &bias,
                        &forbidden,
                        k,
                        &mut scratch,
                        &mut buf,
                        &mut got_ids,
                        &mut got_lps,
                    );
                    assert_eq!(got_ids, ids, "k {k}");
                    assert_eq!(bits(&got_lps), bits(&lps), "k {k}");
                    // the AVX2 level's, where this CPU has AVX2
                    #[cfg(target_arch = "x86_64")]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        let pz = script_prior(&h, &head, &bias);
                        let mut wide = vec![0f32; v];
                        let (mut want, mut got) = (Vec::new(), Vec::new());
                        // SAFETY: the CPU has AVX2 and FMA
                        unsafe {
                            script_x86::class_terms_avx2(row, &head, pz, &mut wide);
                            script_x86::script_top_avx2(row, &head, pz, &forbidden, k, &mut got);
                        }
                        top_k_scan(&wide, &forbidden, k, &mut want);
                        let pairs = |b: &[(f32, u32)]| {
                            b.iter()
                                .map(|&(x, id)| (x.to_bits(), id))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(pairs(&got), pairs(&want), "AVX2, k {k}");
                    }
                }
            }
        }
    }

    /// Research timing on an export's native step (SETSUNA_AR_NATIVE_FILE=<.safetensors>; skipped without it): N
    /// hypotheses (1, 8, 20) for 24 positions over 60 random frames, 20 searches each after one to warm up; the us a
    /// position, and a hypothesis-position, in each stage. Run with --ignored --nocapture on one core.
    #[test]
    #[ignore]
    fn time_the_step() {
        let Some(path) = std::env::var_os("SETSUNA_AR_NATIVE_FILE") else {
            eprintln!("SETSUNA_AR_NATIVE_FILE not set");
            return;
        };
        let dec = NativeDecoder::load(Path::new(&path)).unwrap();
        let c = dec.config.clone();
        let (t, positions, searches) = (60, 24, 20);
        let mut rng = Rng(5);
        let len = c.layers * c.heads * t * c.head_dim;
        let (ck, cv) = (rng.vec(len, 1.0), rng.vec(len, 1.0));
        let forbidden = vec![false; c.vocab];
        eprintln!(
            "isa {:?}, vocab {}, hidden {}, ffn {}",
            dec.isa, c.vocab, c.hidden, c.ffn
        );
        for n in [1usize, 8, 20] {
            let mut stages = [0u64; 6];
            let mut total = 0u128;
            for run in 0..=searches {
                let mut search = NativeSearch::new(&dec, &ck, cv.clone(), t, 16).unwrap();
                let t0 = std::time::Instant::now();
                search.step(&[], &[c.bos_id], 0, &forbidden).unwrap();
                for p in 1..positions {
                    let parents: Vec<usize> = if p == 1 {
                        vec![0; n]
                    } else {
                        (0..n).rev().collect()
                    };
                    let tokens: Vec<u32> = (0..n).map(|i| 10 + ((i + p) % 50) as u32).collect();
                    search.step(&parents, &tokens, p, &forbidden).unwrap();
                }
                if run > 0 {
                    total += t0.elapsed().as_nanos();
                    for (a, b) in stages.iter_mut().zip(search.stage_ns) {
                        *a += b;
                    }
                }
            }
            let per = (searches * positions) as f64 * 1000.0;
            eprintln!(
                "N {n:2}: {:7.1} us a position ({:5.2} a hypothesis-position); self {:6.1} cross {:6.1} ffn {:6.1} \
                 head {:6.1} top {:6.1} script {:6.1}",
                total as f64 / per,
                total as f64 / per / n as f64,
                stages[0] as f64 / per,
                stages[1] as f64 / per,
                stages[2] as f64 / per,
                stages[3] as f64 / per,
                stages[4] as f64 / per,
                stages[5] as f64 / per,
            );
        }

        // the parts alone, compiled as the step is: us a call
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
        unsafe fn parts(dec: &NativeDecoder, n: usize, reps: usize) -> [f64; 5] {
            let c = &dec.config;
            let mut rng = Rng(9);
            let h = rng.vec(n * c.hidden, 1.0);
            let (mut y, mut logits, mut lin) = (Vec::new(), Vec::new(), LinScratch::default());
            let forbidden = vec![false; c.vocab];
            let time = |f: &mut dyn FnMut()| {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    f();
                }
                t0.elapsed().as_nanos() as f64 / reps as f64 / 1000.0
            };
            let head = time(&mut || dec.head.forward(dec.isa, &h, n, &mut logits, &mut lin));
            let qkv = time(&mut || dec.layers[0].qkv.forward(dec.isa, &h, n, &mut y, &mut lin));
            let row = logits[..c.vocab].to_vec();
            let lse_row = time(&mut || {
                std::hint::black_box(row_lse(&row));
            });
            let (mut buf, mut ids, mut lps) = (Vec::new(), Vec::new(), Vec::new());
            let top = time(&mut || {
                ids.clear();
                lps.clear();
                top_k_log_probs(&row, &forbidden, 16, &mut buf, &mut ids, &mut lps);
            });
            let mut gemm_acc = vec![0i32; n * dec.head.stride];
            let q = vec![128u8; n * c.hidden];
            let gemm = time(&mut || gemm_u8i8(dec.isa, &q, n, &dec.head.w, &mut gemm_acc));
            [head, gemm, qkv, lse_row, top]
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            for n in [1usize, 20] {
                // SAFETY: the CPU supports the target features.
                let [head, gemm, qkv, lse_row, top] = unsafe { parts(&dec, n, 2000) };
                eprintln!(
                    "parts, {n} rows: head {head:.1} us (its GEMM {gemm:.1}), qkv {qkv:.2}; a row: log-sum-exp \
                     {lse_row:.2}, best 16 {top:.2}"
                );
            }
        }
    }

    /// A file as the export writes it read back; another format refused.
    #[test]
    fn reads_the_safetensors_layout() {
        let file = |format: &str| -> Vec<u8> {
            let config = serde_json::to_string(&serde_json::json!({
                "layers": 1, "hidden": 4, "heads": 1, "head_dim": 4, "ffn": 8, "vocab": 6, "max_positions": 4,
                "activation": "relu", "norm_eps": 1e-5, "logit_scale": 0.5, "bos_id": 3, "eos_id": 2
            }))
            .unwrap();
            let header = serde_json::json!({
                "__metadata__": {"format": format, "config": config},
                "a.q": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]},
                "a.scale": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
            });
            let mut head = serde_json::to_vec(&header).unwrap();
            head.resize(head.len().div_ceil(8) * 8, b' ');
            let mut out = (head.len() as u64).to_le_bytes().to_vec();
            out.extend(head);
            out.extend([1u8, 255, 127, 129]);
            out.extend(0.25f32.to_le_bytes());
            out.extend((-2.0f32).to_le_bytes());
            out
        };
        let (c, w) = parse(&file(NATIVE_FORMAT)).unwrap();
        assert_eq!(
            (c.hidden, c.ffn, c.activation.as_str(), c.eos_id),
            (4, 8, "relu", 2)
        );
        assert_eq!(w.i8("a.q", &[2, 2]).unwrap(), &[1, -1, 127, -127]);
        assert_eq!(w.f32("a.scale", &[2]).unwrap(), &[0.25, -2.0]);
        assert!(w.f32("a.q", &[2, 2]).is_err() && w.i8("a.q", &[4]).is_err());
        assert!(parse(&file("other")).is_err());
    }
}
