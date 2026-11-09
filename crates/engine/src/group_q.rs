//! `neo_ime.LinearQ4G`: a Linear on low-bit codes with a scale and a zero point per group of inputs, as one CPU
//! kernel (the research export's quantization-aware training grids: k bits from each group's min to its max, 1 bit
//! as +-s, 1.58 bits as -s / 0 / +s, all of them `w = scale * (code - zero)` with codes 0..15).
//!
//! Inputs: `x` f32 (..., K); `w` u8 (K / g, N, g / 2), each group's (N, g) codes in the int4 blocked layout of
//! `neo_ime.TokenMoEBlocked4` (16 outputs x 4 inputs a block, the blocks along the inputs, then the next 16 outputs;
//! each 64-value block as 32 bytes, value b in byte b's low nibble and value b + 32 in its high one; the nibble is the
//! code); `scales` f32 (N, K / g); `zeros` f32 (N, K / g). Attribute `group` (g). Output f32 (..., N). Read in place
//! from the session's initializers. N a multiple of 16, g of 4.
//!
//! Each row of `x` is quantized to u8 (asymmetric, `moe_op::quantize_row`: x ~= sx (q - zx)), and
//!     y = sx * ( sum_g [ s_g acc_g - s_g z_g A_g ] - zx * T ),  T = sum_g s_g (C_g - g z_g)
//! with acc_g = sum_i q_i c_i, A_g = sum_i q_i, C_g = sum_i c_i over the group (sum_i sx (q_i - zx) s_g (c_i - z_g)
//! expanded). With AVX-512 VNNI one kernel keeps each group's integer sums of a tile of 6 rows x 64 outputs in
//! registers, by u8 x i8 dot products of four on the codes unpacked to a byte each when the session loads (the file
//! keeps the nibbles), and at the group's end turns them to float and scales them into the output's running sums (two
//! FMAs); elsewhere the same arithmetic in plain Rust (`mul_add` is the same fused multiply-add), so every path gives
//! the same bits. (The first version called the experts' int4 GEMM once a group and
//! scaled its sums from memory: the 4M's 43 Linears 0.47 ms slower than ONNX Runtime's dynamic int8 MatMuls on a
//! conversion; ONNX Runtime 1.28's MatMulNBits 1.9 ms slower.)

use ort::operator::{
    ComputeContext, Kernel, KernelContext, Operator, OperatorInput, OperatorOutput,
};
use ort::value::{DynValueTypeMarker, TensorElementType};

use crate::moe_op::{add_routed, gelu, quantize_row, route};

/// The op (see the module docs).
pub struct LinearQ4G;

/// `neo_ime.TokenMoEQ4G`: `neo_ime.TokenMoE` (a token-routed mixture of experts, moe_op.rs: per frame the router's
/// fp32 logits, the top-`k` experts with their softmax renormalised over the k, each expert's two Linears with exact
/// GELU between them on the frames routed to it, the weighted outputs summed) with every expert's Linears as
/// `LinearQ4G`'s grouped low-bit codes. Inputs: `x` f32 (..., d); `router` f32 (E, d); `w1` u8 (E, d / g, f, g / 2),
/// `s1` / `z1` f32 (E, f, d / g), `b1` f32 (E, f); `w2` u8 (E, f / g, d, g / 2), `s2` / `z2` f32 (E, d, f / g), `b2`
/// f32 (E, d). Attributes `k`, `group`. Output f32 shaped as `x`. The experts one after another, each on its frames.
pub struct TokenMoEQ4G;

impl Operator for TokenMoEQ4G {
    type Kernel<'attr> = GroupMoeKernel;

    fn name(&self) -> &str {
        "TokenMoEQ4G"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        use TensorElementType::{Float32, Uint8};
        [
            Float32, Float32, Uint8, Float32, Float32, Float32, Uint8, Float32, Float32, Float32,
        ]
        .map(OperatorInput::required)
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<GroupMoeKernel> {
        let attr = |name: &str| -> ort::Result<usize> {
            kctx.get::<i64>(name)
                .map(|v| v as usize)
                .ok_or_else(|| ort::Error::new(format!("TokenMoEQ4G: attribute {name} missing")))
        };
        let (k, g) = (attr("k")?, attr("group")?);
        let f32s = |idx: usize| -> ort::Result<(Vec<usize>, Vec<f32>)> {
            let v = kctx.constant_input::<DynValueTypeMarker>(idx)?;
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            Ok((shape.iter().map(|&s| s as usize).collect(), data.to_vec()))
        };
        let (rs, router) = f32s(1)?;
        let (s1s, s1) = f32s(3)?;
        let (_, z1) = f32s(4)?;
        let (_, b1) = f32s(5)?;
        let (_, s2) = f32s(7)?;
        let (_, z2) = f32s(8)?;
        let (_, b2) = f32s(9)?;
        let (e, d, f) = (rs[0], rs[1], s1s[1]);
        let v1 = kctx.constant_input::<DynValueTypeMarker>(2)?;
        let (_, w1) = v1.try_extract_tensor::<u8>()?;
        let v2 = kctx.constant_input::<DynValueTypeMarker>(6)?;
        let (_, w2) = v2.try_extract_tensor::<u8>()?;
        if w1.len() != e * f * d / 2
            || w2.len() != e * d * f / 2
            || b1.len() != e * f
            || b2.len() != e * d
        {
            return Err(ort::Error::new(format!(
                "TokenMoEQ4G: weights not shaped for E {e}, d {d}, f {f}"
            )));
        }
        let (g1, g2) = (d / g, f / g);
        let mut experts = Vec::with_capacity(e);
        for ex in 0..e {
            // SAFETY: the codes are constant initializers of the session, alive and unchanged while the kernel lives
            let l1 = unsafe {
                GroupWeights::in_place(
                    &w1[ex * f * d / 2..(ex + 1) * f * d / 2],
                    &s1[ex * f * g1..(ex + 1) * f * g1],
                    &z1[ex * f * g1..(ex + 1) * f * g1],
                    f,
                    d,
                    g,
                )
            }
            .map_err(|m| ort::Error::new(format!("TokenMoEQ4G: expert {ex} w1: {m}")))?;
            let l2 = unsafe {
                GroupWeights::in_place(
                    &w2[ex * d * f / 2..(ex + 1) * d * f / 2],
                    &s2[ex * d * g2..(ex + 1) * d * g2],
                    &z2[ex * d * g2..(ex + 1) * d * g2],
                    d,
                    f,
                    g,
                )
            }
            .map_err(|m| ort::Error::new(format!("TokenMoEQ4G: expert {ex} w2: {m}")))?;
            experts.push((l1, l2));
        }
        Ok(GroupMoeKernel {
            moe: GroupMoe {
                e,
                d,
                f,
                k,
                router,
                experts,
                b1,
                b2,
            },
            scratch: GroupMoeScratch::default(),
        })
    }
}

pub struct GroupMoeKernel {
    moe: GroupMoe,
    scratch: GroupMoeScratch,
}

impl Kernel for GroupMoeKernel {
    fn compute(&mut self, ctx: &ComputeContext) -> ort::Result<()> {
        let x = ctx
            .input(0)?
            .ok_or_else(|| ort::Error::new("TokenMoEQ4G: input x missing"))?;
        let (shape, x) = x.try_extract_tensor::<f32>()?;
        let shape = shape.to_vec();
        let mut y = ctx
            .output(0, shape)?
            .ok_or_else(|| ort::Error::new("TokenMoEQ4G: output missing"))?;
        let (_, y) = y.try_extract_tensor_mut::<f32>()?;
        self.moe.forward(x, y, &mut self.scratch);
        Ok(())
    }
}

/// A TokenMoE layer whose experts' Linears are `GroupWeights`.
pub(crate) struct GroupMoe {
    e: usize,
    d: usize,
    f: usize,
    k: usize,
    router: Vec<f32>,
    experts: Vec<(GroupWeights, GroupWeights)>,
    b1: Vec<f32>,
    b2: Vec<f32>,
}

#[derive(Default)]
pub(crate) struct GroupMoeScratch {
    rows: Vec<Vec<(usize, f32)>>,
    xg: Vec<f32>,
    h: Vec<f32>,
    out: Vec<f32>,
    lin: GroupScratch,
}

impl GroupMoe {
    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        #[cfg(target_arch = "x86_64")]
        if self
            .experts
            .first()
            .is_some_and(|(l1, _)| l1.path == Path::Vnni512)
        {
            // SAFETY: the CPU has the features (`detect`)
            return unsafe { self.forward_avx512(x, y, s) };
        }
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    unsafe fn forward_avx512(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        self.forward_body(x, y, s)
    }

    #[inline(always)]
    fn forward_body(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        let (d, f) = (self.d, self.f);
        y.fill(0.0);
        route(x, &self.router, self.e, d, self.k, &mut s.rows);
        for (ex, (l1, l2)) in self.experts.iter().enumerate() {
            let routed = &s.rows[ex];
            if routed.is_empty() {
                continue;
            }
            let m = routed.len();
            s.xg.clear();
            for &(r, _) in routed {
                s.xg.extend_from_slice(&x[r * d..(r + 1) * d]);
            }
            s.h.resize(m * f, 0.0);
            l1.forward(&s.xg, &mut s.h, &mut s.lin);
            let b1 = &self.b1[ex * f..(ex + 1) * f];
            for hr in s.h.chunks_mut(f) {
                for (hv, &bv) in hr.iter_mut().zip(b1) {
                    *hv = gelu(*hv + bv);
                }
            }
            s.out.resize(m * d, 0.0);
            l2.forward(&s.h, &mut s.out, &mut s.lin);
            let b2 = &self.b2[ex * d..(ex + 1) * d];
            for or in s.out.chunks_mut(d) {
                for (ov, &bv) in or.iter_mut().zip(b2) {
                    *ov += bv;
                }
            }
            add_routed(y, routed, &s.out, d);
        }
    }
}

impl Operator for LinearQ4G {
    type Kernel<'attr> = GroupKernel;

    fn name(&self) -> &str {
        "LinearQ4G"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        use TensorElementType::{Float32, Uint8};
        [Float32, Uint8, Float32, Float32].map(OperatorInput::required)
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<GroupKernel> {
        let g = kctx
            .get::<i64>("group")
            .ok_or_else(|| ort::Error::new("LinearQ4G: attribute group missing"))?
            as usize;
        let f32s = |idx: usize| -> ort::Result<(Vec<usize>, Vec<f32>)> {
            let v = kctx.constant_input::<DynValueTypeMarker>(idx)?;
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            Ok((shape.iter().map(|&s| s as usize).collect(), data.to_vec()))
        };
        let (ss, scales) = f32s(2)?;
        let (_, zeros) = f32s(3)?;
        let wv = kctx.constant_input::<DynValueTypeMarker>(1)?;
        let (ws, w) = wv.try_extract_tensor::<u8>()?;
        let (n, groups) = (ss[0], ss[1]);
        if ws.len() != 3
            || ws[0] as usize != groups
            || ws[1] as usize != n
            || ws[2] as usize * 2 != g
        {
            return Err(ort::Error::new(format!(
                "LinearQ4G: w {ws:?} is not ({groups}, {n}, {}) for scales {ss:?} and group {g}",
                g / 2
            )));
        }
        // SAFETY: the codes are a constant initializer of the session, alive and unchanged while the kernel lives
        let weights = unsafe { GroupWeights::in_place(w, &scales, &zeros, n, groups * g, g) }
            .map_err(|m| ort::Error::new(format!("LinearQ4G: {m}")))?;
        Ok(GroupKernel {
            weights,
            scratch: GroupScratch::default(),
        })
    }
}

pub struct GroupKernel {
    weights: GroupWeights,
    scratch: GroupScratch,
}

impl Kernel for GroupKernel {
    fn compute(&mut self, ctx: &ComputeContext) -> ort::Result<()> {
        let x = ctx
            .input(0)?
            .ok_or_else(|| ort::Error::new("LinearQ4G: input x missing"))?;
        let (shape, x) = x.try_extract_tensor::<f32>()?;
        let mut out_shape = shape.to_vec();
        let last = out_shape.len() - 1;
        if out_shape[last] as usize != self.weights.k {
            return Err(ort::Error::new(format!(
                "LinearQ4G: x's last axis {} is not K {}",
                out_shape[last], self.weights.k
            )));
        }
        out_shape[last] = self.weights.n as i64;
        let mut y = ctx
            .output(0, out_shape)?
            .ok_or_else(|| ort::Error::new("LinearQ4G: output missing"))?;
        let (_, y) = y.try_extract_tensor_mut::<f32>()?;
        self.weights.forward(x, y, &mut self.scratch);
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Vnni512,
    Plain,
}

/// The codes in place and the per-group terms of the output.
pub(crate) struct GroupWeights {
    n: usize,
    k: usize,
    g: usize,
    w: *const u8,
    w_len: usize,
    /// The codes unpacked to one byte each, blocked as the packed ones (64-byte blocks: (K / g, N / 16, g / 4, 64)),
    /// read by the VNNI kernel as they are (unpacking the nibbles in the kernel cost more than the dot products)
    unpacked: Vec<i8>,
    /// (K / g, N): s_g, and -s_g z_g
    scale: Vec<f32>,
    scale_zero: Vec<f32>,
    /// (N): sum_g s_g (C_g - g z_g)
    t: Vec<f32>,
    path: Path,
}

// SAFETY: `w` is an immutable initializer of the session, only read
unsafe impl Send for GroupWeights {}
unsafe impl Sync for GroupWeights {}

#[derive(Default)]
pub(crate) struct GroupScratch {
    q: Vec<u8>,
    xs: Vec<(f32, i32)>,
    a: Vec<f32>,
}

impl GroupWeights {
    /// The weights read where they lie (`w`: (K / g) groups of (N, g) blocked codes, g / 2 bytes a row); `scales` and
    /// `zeros` (N, K / g).
    ///
    /// # Safety
    /// `w` stays valid and unchanged as long as the returned weights.
    pub(crate) unsafe fn in_place(
        w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        k: usize,
        g: usize,
    ) -> Result<Self, String> {
        if n % 16 != 0 || g % 4 != 0 || g == 0 || k % g != 0 {
            return Err(format!(
                "N {n} must be in 16s, the group {g} in 4s and divide K {k}"
            ));
        }
        let groups = k / g;
        if w.len() != groups * n * g / 2 || scales.len() != n * groups || zeros.len() != n * groups
        {
            return Err(format!(
                "{} code bytes, {} scales, {} zeros for N {n}, K {k}, group {g}",
                w.len(),
                scales.len(),
                zeros.len()
            ));
        }
        let mut weights = GroupWeights {
            n,
            k,
            g,
            w: w.as_ptr(),
            w_len: w.len(),
            // each 32-byte block: the low nibbles are values 0-31, the high ones 32-63
            unpacked: w
                .chunks(32)
                .flat_map(|b| {
                    b.iter()
                        .map(|&v| (v & 0x0f) as i8)
                        .chain(b.iter().map(|&v| (v >> 4) as i8))
                })
                .collect(),
            scale: vec![0f32; groups * n],
            scale_zero: vec![0f32; groups * n],
            t: vec![0f32; n],
            path: detect(),
        };
        for gi in 0..groups {
            for o in 0..n {
                let (s, z) = (scales[o * groups + gi], zeros[o * groups + gi]);
                let c: u32 = (0..g).map(|i| weights.code(o, gi * g + i) as u32).sum();
                weights.scale[gi * n + o] = s;
                weights.scale_zero[gi * n + o] = -(s * z);
                weights.t[o] += s * (c as f32 - g as f32 * z);
            }
        }
        Ok(weights)
    }

    fn codes(&self) -> &[u8] {
        // SAFETY: `in_place`'s contract
        unsafe { std::slice::from_raw_parts(self.w, self.w_len) }
    }

    /// The code of output `o`, input `i`.
    fn code(&self, o: usize, i: usize) -> u8 {
        let (g, n) = (self.g, self.n);
        let (gi, ii) = (i / g, i % g);
        let block = (o / 16) * (g / 4) + ii / 4;
        let v = (o % 16) * 4 + ii % 4;
        let byte = self.codes()[gi * (n * g / 2) + block * 32 + v % 32];
        if v < 32 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    }

    /// y (rows, N) for x (rows, K). With AVX-512 the whole of it is compiled for AVX-512 (the activations'
    /// quantization, the group sums and the last scaling are plain Rust, which the engine's build otherwise compiles for
    /// SSE2: on the 4M's 768 -> 192 Linear the quantization of 30 rows took most of the 18 us): the same code, the same
    /// bits.
    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32], s: &mut GroupScratch) {
        match self.path {
            #[cfg(target_arch = "x86_64")]
            Path::Vnni512 => unsafe { x86::forward(self, x, y, s) },
            _ => self.forward_with(x, y, s),
        }
    }

    #[inline(always)]
    fn forward_with(&self, x: &[f32], y: &mut [f32], s: &mut GroupScratch) {
        let (n, k, g) = (self.n, self.k, self.g);
        let groups = k / g;
        let rows = x.len() / k;
        s.q.resize(rows * k, 0);
        s.xs.clear();
        for r in 0..rows {
            s.xs.push(quantize_row(
                &x[r * k..(r + 1) * k],
                &mut s.q[r * k..(r + 1) * k],
            ));
        }
        s.a.clear();
        for r in 0..rows {
            for gi in 0..groups {
                let sum: u32 = s.q[r * k + gi * g..r * k + (gi + 1) * g]
                    .iter()
                    .map(|&v| v as u32)
                    .sum();
                s.a.push(sum as f32);
            }
        }
        let y = &mut y[..rows * n];
        match self.path {
            #[cfg(target_arch = "x86_64")]
            Path::Vnni512 => unsafe { x86::inner(self, &s.q, &s.a, rows, y) },
            _ => self.inner_plain(&s.q, &s.a, rows, y),
        }
        for r in 0..rows {
            let (sx, zx) = s.xs[r];
            let zx = zx as f32;
            for (yv, &tv) in y[r * n..(r + 1) * n].iter_mut().zip(&self.t) {
                *yv = (*yv - zx * tv) * sx;
            }
        }
    }

    /// The running sums sum_g [ s_g acc_g - s_g z_g A_g ] in plain Rust, as the VNNI kernel orders them.
    fn inner_plain(&self, q: &[u8], a: &[f32], rows: usize, y: &mut [f32]) {
        let (n, k, g) = (self.n, self.k, self.g);
        let groups = k / g;
        for r in 0..rows {
            for o in 0..n {
                let mut f = 0f32;
                for gi in 0..groups {
                    let acc: i32 = (0..g)
                        .map(|i| q[r * k + gi * g + i] as i32 * self.code(o, gi * g + i) as i32)
                        .sum();
                    f = (acc as f32).mul_add(self.scale[gi * n + o], f);
                    f = self.scale_zero[gi * n + o].mul_add(a[r * groups + gi], f);
                }
                y[r * n + o] = f;
            }
        }
    }

    #[cfg(test)]
    fn with_path(mut self, path: Path) -> Self {
        self.path = path;
        self
    }
}

fn detect() -> Path {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return Path::Vnni512;
        }
    }
    Path::Plain
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::GroupWeights;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn bcast4(q: *const u8) -> i32 {
        (q as *const i32).read_unaligned()
    }

    /// RT rows x C output blocks (16 outputs each) from block `ob`: every group's integer sums in registers (24 of
    /// the 32 with 6 rows x 4 blocks, each weight vector serving the tile's rows and each broadcast its blocks), turned
    /// to float and scaled into the running sums in `y` (zero at the start) at the group's end.
    #[target_feature(enable = "avx512f,avx512vnni,avx2,fma")]
    unsafe fn tile<const RT: usize, const C: usize>(
        w: &GroupWeights,
        q: *const u8,
        a: *const f32,
        ob: usize,
        y: *mut f32,
    ) {
        let (n, k, g) = (w.n, w.k, w.g);
        let (groups, steps, gstride) = (k / g, g / 4, n * g);
        let codes = w.unpacked.as_ptr();
        let (sc, sz) = (w.scale.as_ptr(), w.scale_zero.as_ptr());
        for gi in 0..groups {
            let mut acc = [[_mm512_setzero_si512(); C]; RT];
            let wg = codes.add(gi * gstride);
            for s in 0..steps {
                let mut wv = [_mm512_setzero_si512(); C];
                for (c, wc) in wv.iter_mut().enumerate() {
                    *wc = _mm512_loadu_si512(wg.add(((ob + c) * steps + s) * 64) as *const _);
                }
                for (r, ar) in acc.iter_mut().enumerate() {
                    let x = _mm512_set1_epi32(bcast4(q.add(r * k + gi * g + s * 4)));
                    for (ac, wc) in ar.iter_mut().zip(&wv) {
                        *ac = _mm512_dpbusd_epi32(*ac, x, *wc);
                    }
                }
            }
            for c in 0..C {
                let s1 = _mm512_loadu_ps(sc.add(gi * n + (ob + c) * 16));
                let s2 = _mm512_loadu_ps(sz.add(gi * n + (ob + c) * 16));
                for (r, ar) in acc.iter().enumerate() {
                    let av = _mm512_set1_ps(*a.add(r * groups + gi));
                    let yp = y.add(r * n + (ob + c) * 16);
                    let f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ar[c]), s1, _mm512_loadu_ps(yp));
                    _mm512_storeu_ps(yp, _mm512_fmadd_ps(s2, av, f));
                }
            }
        }
    }

    macro_rules! by_rows {
        ($rt:expr, $c:literal, $($arg:expr),*) => {
            match $rt {
                1 => tile::<1, $c>($($arg),*),
                2 => tile::<2, $c>($($arg),*),
                3 => tile::<3, $c>($($arg),*),
                4 => tile::<4, $c>($($arg),*),
                5 => tile::<5, $c>($($arg),*),
                _ => tile::<6, $c>($($arg),*),
            }
        };
    }

    /// `GroupWeights::forward_with` compiled for AVX-512 (its plain Rust parts inlined here and vectorized 16 wide).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    pub(super) unsafe fn forward(
        w: &GroupWeights,
        x: &[f32],
        y: &mut [f32],
        s: &mut super::GroupScratch,
    ) {
        w.forward_with(x, y, s)
    }

    /// `GroupWeights::inner_plain` with AVX-512 VNNI: tiles of up to 6 rows by 4 output blocks while 4 are left, then
    /// 1 block. (Tiles of 2 blocks took 11.6 us for 30 rows 192 -> 576, the same as with the codes unpacked in the
    /// kernel: a broadcast a row served only 2 dot products.)
    #[target_feature(enable = "avx512f,avx512vnni,avx2,fma")]
    pub(super) unsafe fn inner(w: &GroupWeights, q: &[u8], a: &[f32], rows: usize, y: &mut [f32]) {
        let (n, k, groups) = (w.n, w.k, w.k / w.g);
        let blocks = n / 16;
        y[..rows * n].fill(0.0);
        let mut ob = 0;
        while ob < blocks {
            let c4 = ob + 4 <= blocks;
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(6);
                let (qp, ap, yp) = (
                    q.as_ptr().add(r0 * k),
                    a.as_ptr().add(r0 * groups),
                    y.as_mut_ptr().add(r0 * n),
                );
                if c4 {
                    by_rows!(rt, 4, w, qp, ap, ob, yp)
                } else {
                    by_rows!(rt, 1, w, qp, ap, ob, yp)
                }
                r0 += rt;
            }
            ob += if c4 { 4 } else { 1 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// (N, K) codes 0..15 as `w` bytes: per group the int4 blocked layout (16 rows x 4 inputs a block, the blocks
    /// along the inputs, then the next 16 rows; 64 values as 32 bytes, value b low nibble of byte b, b + 32 high), as
    /// the research export writes them.
    fn pack(codes: &[u8], n: usize, k: usize, g: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * k / 2);
        for gi in 0..k / g {
            let mut blocked = Vec::with_capacity(n * g);
            for ob in 0..n / 16 {
                for ig in 0..g / 4 {
                    for j in 0..16 {
                        for t in 0..4 {
                            blocked.push(codes[(ob * 16 + j) * k + gi * g + ig * 4 + t]);
                        }
                    }
                }
            }
            for b in blocked.chunks(64) {
                for i in 0..32 {
                    out.push((b[i] & 0x0f) | ((b[i + 32] & 0x0f) << 4));
                }
            }
        }
        out
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32
    }

    /// The kernel against the dequantized weights times each row's quantized activations in f64 (the same
    /// quantization of x the kernel makes), for 4-bit codes, 3-bit ones (0..7) and a binary grid (zero 0.5), groups
    /// of 64 and 16, odd output blocks, rows 1 to 31; and the VNNI path gives the plain path's bits.
    #[test]
    fn linear_q4g_matches_the_dequantized_weights() {
        let mut seed = 9u64;
        for &(n, k, g, levels, zero_half) in &[
            (48usize, 192usize, 64usize, 15u8, false),
            (32, 768, 64, 7, false),
            (192, 16, 16, 15, false),
            (64, 128, 64, 1, true),
            (16, 64, 64, 15, false),
        ] {
            let codes: Vec<u8> = (0..n * k)
                .map(|_| (rnd(&mut seed) * (levels as f32 + 1.0)) as u8 % (levels + 1))
                .collect();
            let groups = k / g;
            let scales: Vec<f32> = (0..n * groups)
                .map(|_| 0.01 + rnd(&mut seed) * 0.05)
                .collect();
            let zeros: Vec<f32> = (0..n * groups)
                .map(|_| {
                    if zero_half {
                        0.5
                    } else {
                        (rnd(&mut seed) * (levels as f32 + 1.0)).floor() - 1.0
                    }
                })
                .collect();
            let w = pack(&codes, n, k, g);
            let weights = unsafe { GroupWeights::in_place(&w, &scales, &zeros, n, k, g) }.unwrap();
            for rows in [1usize, 5, 7, 31] {
                let x: Vec<f32> = (0..rows * k).map(|_| rnd(&mut seed) * 4.0 - 2.0).collect();
                let mut y = vec![0f32; rows * n];
                weights.forward(&x, &mut y, &mut GroupScratch::default());
                for r in 0..rows {
                    let mut q = vec![0u8; k];
                    let (sx, zx) = quantize_row(&x[r * k..(r + 1) * k], &mut q);
                    for o in 0..n {
                        let mut want = 0f64;
                        for i in 0..k {
                            let (sv, zv) = (
                                scales[o * groups + i / g] as f64,
                                zeros[o * groups + i / g] as f64,
                            );
                            want += sx as f64
                                * (q[i] as f64 - zx as f64)
                                * sv
                                * (codes[o * k + i] as f64 - zv);
                        }
                        let got = y[r * n + o] as f64;
                        assert!(
                            (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                            "n {n} k {k} g {g} row {r} out {o}: {got} vs {want}"
                        );
                    }
                }
                let plain = unsafe { GroupWeights::in_place(&w, &scales, &zeros, n, k, g) }
                    .unwrap()
                    .with_path(Path::Plain);
                let mut yp = vec![0f32; rows * n];
                plain.forward(&x, &mut yp, &mut GroupScratch::default());
                assert_eq!(
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    yp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "the kernel and the plain path, n {n} k {k} g {g} rows {rows}"
                );
            }
        }
    }

    /// `GroupMoe` against its parts frame by frame: `route`, then for each chosen expert its first Linear on the
    /// frame alone, GELU of it plus b1, the second Linear, plus b2, weighted and summed in expert order; the same bits
    /// (each Linear quantizes and computes its rows independently).
    #[test]
    fn group_moe_matches_its_parts() {
        let mut seed = 21u64;
        let (e, d, f, k, g) = (4usize, 64usize, 128usize, 2usize, 64usize);
        let router: Vec<f32> = (0..e * d).map(|_| rnd(&mut seed) * 2.0 - 1.0).collect();
        let mut experts = Vec::new();
        let mut keep = Vec::new();
        for _ in 0..e {
            let mut layer = |n: usize, kk: usize, seed: &mut u64| {
                let codes: Vec<u8> = (0..n * kk).map(|_| (rnd(seed) * 8.0) as u8 % 8).collect();
                let scales: Vec<f32> = (0..n * kk / g).map(|_| 0.01 + rnd(seed) * 0.05).collect();
                let zeros: Vec<f32> = (0..n * kk / g).map(|_| (rnd(seed) * 8.0).floor()).collect();
                let w = pack(&codes, n, kk, g);
                (w, scales, zeros)
            };
            let p1 = layer(f, d, &mut seed);
            let p2 = layer(d, f, &mut seed);
            let l1 = unsafe { GroupWeights::in_place(&p1.0, &p1.1, &p1.2, f, d, g) }.unwrap();
            let l2 = unsafe { GroupWeights::in_place(&p2.0, &p2.1, &p2.2, d, f, g) }.unwrap();
            experts.push((l1, l2));
            keep.push((p1, p2));
        }
        let b1: Vec<f32> = (0..e * f).map(|_| rnd(&mut seed) * 0.2 - 0.1).collect();
        let b2: Vec<f32> = (0..e * d).map(|_| rnd(&mut seed) * 0.2 - 0.1).collect();
        let moe = GroupMoe {
            e,
            d,
            f,
            k,
            router: router.clone(),
            experts,
            b1: b1.clone(),
            b2: b2.clone(),
        };
        let n = 13;
        let x: Vec<f32> = (0..n * d).map(|_| rnd(&mut seed) * 2.0 - 1.0).collect();
        let mut y = vec![0f32; n * d];
        moe.forward(&x, &mut y, &mut GroupMoeScratch::default());
        let mut rows = Vec::new();
        route(&x, &router, e, d, k, &mut rows);
        let mut want = vec![0f32; n * d];
        let mut sc = GroupScratch::default();
        for (ex, (l1, l2)) in moe.experts.iter().enumerate() {
            for &(r, wt) in &rows[ex] {
                let mut h = vec![0f32; f];
                l1.forward(&x[r * d..(r + 1) * d], &mut h, &mut sc);
                for (hv, &bv) in h.iter_mut().zip(&b1[ex * f..(ex + 1) * f]) {
                    *hv = gelu(*hv + bv);
                }
                let mut o = vec![0f32; d];
                l2.forward(&h, &mut o, &mut sc);
                for ((wv, ov), &bv) in want[r * d..(r + 1) * d]
                    .iter_mut()
                    .zip(&o)
                    .zip(&b2[ex * d..(ex + 1) * d])
                {
                    *wv += wt * (*ov + bv);
                }
            }
        }
        assert_eq!(
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        drop(keep);
    }

    /// A small model's Linear shapes on 30 rows (QKV 192 -> 576, FFN 192 -> 768 and 768 -> 192), median of 500.
    /// cargo test -p setsuna-engine --release --lib group_q::tests::timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn timing() {
        let mut seed = 5u64;
        for &(n, k) in &[(576usize, 192usize), (768, 192), (192, 768)] {
            let g = 64;
            let codes: Vec<u8> = (0..n * k)
                .map(|_| (rnd(&mut seed) * 16.0) as u8 % 16)
                .collect();
            let groups = k / g;
            let scales: Vec<f32> = (0..n * groups).map(|_| 0.02).collect();
            let zeros: Vec<f32> = (0..n * groups).map(|_| 8.0).collect();
            let w = pack(&codes, n, k, g);
            let weights = unsafe { GroupWeights::in_place(&w, &scales, &zeros, n, k, g) }.unwrap();
            let rows = 30;
            let x: Vec<f32> = (0..rows * k).map(|_| rnd(&mut seed) * 2.0 - 1.0).collect();
            let mut y = vec![0f32; rows * n];
            let mut s = GroupScratch::default();
            let mut t = Vec::new();
            for _ in 0..500 {
                let a = Instant::now();
                weights.forward(&x, &mut y, &mut s);
                t.push(a.elapsed().as_secs_f64());
            }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "LinearQ4G {k} -> {n} on {rows} rows ({:?}): {:.4} ms",
                weights.path,
                t[t.len() / 2] * 1e3
            );
        }
    }
}
