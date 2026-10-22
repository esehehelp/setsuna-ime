//! `neo_ime.TokenMoE`: a token-routed mixture of experts as one CPU kernel, registered with every session.
//!
//! The research export's int8 form of a token-routed `RoutedExperts` layer : per frame, the
//! router's logits (fp32), the top-`k` experts and their softmax weights renormalised over the `k`; each expert's two
//! Linears run only on the frames routed to it, with exact GELU between them; the weighted outputs are summed. ONNX
//! Runtime's own `com.microsoft.MoE` / `QMoE` run every frame through the experts one by one (3-4 ms a layer at 30
//! frames on one thread); here the frames of an expert share one pass over its weights.
//!
//! Inputs: `x` f32 (..., d); `router` f32 (E, d); `w1` i8 (E, f, d), `s1` f32 (E, f), `b1` f32 (E, f); `w2` i8 (E, d, f),
//! `s2` f32 (E, d), `b2` f32 (E, d), the weights symmetric int8 per output row (`w ~= s * q`). Attribute `k`. Output
//! f32 shaped as `x`; padding frames are routed too (the graph masks them after). The activations are quantized to u8
//! per row (asymmetric, zero exact) and multiplied with the int8 weights as u8 x i8 dot products of four, with
//! AVX-512 VNNI, AVX-VNNI or plain Rust, the same integer sums on every path.

use ort::operator::{
    ComputeContext, Kernel, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput,
};
use ort::value::{DynValueTypeMarker, TensorElementType};

/// The custom op domain of the research export's fused ops, as the exported
/// graphs name it (like the manifest schema names, kept as they are).
pub const DOMAIN: &str = "neo_ime";

/// The `neo_ime` operator domain, for `SessionBuilder::with_operators`.
pub fn domain() -> ort::Result<OperatorDomain> {
    OperatorDomain::new(DOMAIN)?.add(TokenMoE)
}

struct TokenMoE;

impl Operator for TokenMoE {
    type Kernel<'attr> = MoeKernel;

    fn name(&self) -> &str {
        "TokenMoE"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        use TensorElementType::{Float32, Int8};
        [
            Float32, Float32, Int8, Float32, Float32, Int8, Float32, Float32,
        ]
        .map(OperatorInput::required)
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<MoeKernel> {
        let k = kctx
            .get::<i64>("k")
            .ok_or_else(|| ort::Error::new("TokenMoE: attribute k missing"))?;
        let f32s = |idx: usize| -> ort::Result<(Vec<usize>, Vec<f32>)> {
            let v = kctx.constant_input::<DynValueTypeMarker>(idx)?;
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            Ok((shape.iter().map(|&s| s as usize).collect(), data.to_vec()))
        };
        let (rs, router) = f32s(1)?;
        let (_, s1) = f32s(3)?;
        let (_, b1) = f32s(4)?;
        let (_, s2) = f32s(6)?;
        let (_, b2) = f32s(7)?;
        // the int8 weights packed straight from the session's initializers, no copy between
        let w1v = kctx.constant_input::<DynValueTypeMarker>(2)?;
        let (w1s, w1) = w1v.try_extract_tensor::<i8>()?;
        let w2v = kctx.constant_input::<DynValueTypeMarker>(5)?;
        let (_, w2) = w2v.try_extract_tensor::<i8>()?;
        let (e, d, f) = (rs[0], rs[1], w1s[1] as usize);
        let moe = MoeWeights::new(e, d, f, k as usize, router, w1, s1, b1, w2, s2, b2)
            .map_err(|m| ort::Error::new(format!("TokenMoE: {m}")))?;
        Ok(MoeKernel {
            moe,
            scratch: Scratch::default(),
        })
    }
}

pub struct MoeKernel {
    moe: MoeWeights,
    scratch: Scratch,
}

impl Kernel for MoeKernel {
    fn compute(&mut self, ctx: &ComputeContext) -> ort::Result<()> {
        let x = ctx
            .input(0)?
            .ok_or_else(|| ort::Error::new("TokenMoE: input x missing"))?;
        let (shape, x) = x.try_extract_tensor::<f32>()?;
        let shape = shape.to_vec();
        let mut y = ctx
            .output(0, shape)?
            .ok_or_else(|| ort::Error::new("TokenMoE: output missing"))?;
        let (_, y) = y.try_extract_tensor_mut::<f32>()?;
        self.moe.forward(x, y, &mut self.scratch);
        Ok(())
    }
}

/// An int8 matrix (out, in) prepacked for the u8 x i8 dot products: 16 output rows x 4 inputs a 64-byte block (lane j
/// = row 16 * b + j), blocks along the inputs; `colsum` the sum of each row (for the activations' zero point). Only
/// one layout is kept: `blocked` for the VNNI paths, else `raw` (row-major) for the plain one.
struct Packed {
    out: usize,
    inp: usize,
    blocked: Vec<i8>,
    raw: Vec<i8>,
    colsum: Vec<i32>,
}

impl Packed {
    fn new(w: &[i8], out: usize, inp: usize, block: bool) -> Self {
        assert_eq!(w.len(), out * inp);
        let colsum = (0..out)
            .map(|o| w[o * inp..(o + 1) * inp].iter().map(|&v| v as i32).sum())
            .collect();
        let mut blocked = Vec::new();
        let mut raw = Vec::new();
        if block && out % 16 == 0 && inp % 4 == 0 {
            blocked = vec![0i8; out * inp];
            let mut i = 0;
            for ob in 0..out / 16 {
                for kg in 0..inp / 4 {
                    for j in 0..16 {
                        let row = (ob * 16 + j) * inp + kg * 4;
                        blocked[i..i + 4].copy_from_slice(&w[row..row + 4]);
                        i += 4;
                    }
                }
            }
        } else {
            raw = w.to_vec();
        }
        Packed {
            out,
            inp,
            blocked,
            raw,
            colsum,
        }
    }

    /// Row `o`'s input `i`, from whichever layout is kept.
    fn at(&self, o: usize, i: usize) -> i8 {
        if self.blocked.is_empty() {
            self.raw[o * self.inp + i]
        } else {
            self.blocked[(o / 16) * 16 * self.inp + (i / 4) * 64 + (o % 16) * 4 + i % 4]
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Isa {
    Vnni512,
    Vnni256,
    Plain,
}

fn detect_isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vnni")
        {
            return Isa::Vnni512;
        }
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return Isa::Vnni256;
        }
    }
    Isa::Plain
}

/// The integer sums acc[r][o] = sum_i q[r][i] * w[o][i] of `rows` u8 rows (stride `w.inp`) with the int8 matrix.
fn gemm_u8i8(isa: Isa, q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
    debug_assert!(q.len() >= rows * w.inp && acc.len() >= rows * w.out);
    if w.blocked.is_empty() {
        return gemm_plain(q, rows, w, acc);
    }
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni512 => unsafe { x86::gemm_vnni512(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni256 => unsafe { x86::gemm_vnni256(q, rows, w, acc) },
        _ => gemm_plain(q, rows, w, acc),
    }
}

fn gemm_plain(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
    let (n, k) = (w.out, w.inp);
    for r in 0..rows {
        let qr = &q[r * k..(r + 1) * k];
        for o in 0..n {
            acc[r * n + o] = if w.blocked.is_empty() {
                let wr = &w.raw[o * k..(o + 1) * k];
                qr.iter().zip(wr).map(|(&a, &b)| a as i32 * b as i32).sum()
            } else {
                // a VNNI matrix on the plain path (tests)
                qr.iter()
                    .enumerate()
                    .map(|(i, &a)| a as i32 * w.at(o, i) as i32)
                    .sum()
            };
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Packed;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn bcast4(q: *const u8) -> i32 {
        (q as *const i32).read_unaligned()
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn tile512<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [_mm512_setzero_si512(); RT];
        for kg in 0..k / 4 {
            let w = _mm512_loadu_si512(wb.add(kg * 64) as *const _);
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm512_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                *ar = _mm512_dpbusd_epi32(*ar, x, w);
            }
        }
        for (r, ar) in a.iter().enumerate() {
            _mm512_storeu_si512(acc.add(r * n) as *mut _, *ar);
        }
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) unsafe fn gemm_vnni512(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        for ob in 0..n / 16 {
            let wb = w.blocked.as_ptr().add(ob * k * 16);
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(8);
                let (qp, ap) = (
                    q.as_ptr().add(r0 * k),
                    acc.as_mut_ptr().add(r0 * n + ob * 16),
                );
                match rt {
                    1 => tile512::<1>(qp, k, wb, ap, n),
                    2 => tile512::<2>(qp, k, wb, ap, n),
                    3 => tile512::<3>(qp, k, wb, ap, n),
                    4 => tile512::<4>(qp, k, wb, ap, n),
                    5 => tile512::<5>(qp, k, wb, ap, n),
                    6 => tile512::<6>(qp, k, wb, ap, n),
                    7 => tile512::<7>(qp, k, wb, ap, n),
                    _ => tile512::<8>(qp, k, wb, ap, n),
                }
                r0 += rt;
            }
        }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn tile256<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        acc: *mut i32,
        n: usize,
    ) {
        let mut lo = [_mm256_setzero_si256(); RT];
        let mut hi = [_mm256_setzero_si256(); RT];
        for kg in 0..k / 4 {
            let wl = _mm256_loadu_si256(wb.add(kg * 64) as *const _);
            let wh = _mm256_loadu_si256(wb.add(kg * 64 + 32) as *const _);
            for r in 0..RT {
                let x = _mm256_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                lo[r] = _mm256_dpbusd_avx_epi32(lo[r], x, wl);
                hi[r] = _mm256_dpbusd_avx_epi32(hi[r], x, wh);
            }
        }
        for r in 0..RT {
            _mm256_storeu_si256(acc.add(r * n) as *mut _, lo[r]);
            _mm256_storeu_si256(acc.add(r * n + 8) as *mut _, hi[r]);
        }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    pub(super) unsafe fn gemm_vnni256(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        for ob in 0..n / 16 {
            let wb = w.blocked.as_ptr().add(ob * k * 16);
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(6);
                let (qp, ap) = (
                    q.as_ptr().add(r0 * k),
                    acc.as_mut_ptr().add(r0 * n + ob * 16),
                );
                match rt {
                    1 => tile256::<1>(qp, k, wb, ap, n),
                    2 => tile256::<2>(qp, k, wb, ap, n),
                    3 => tile256::<3>(qp, k, wb, ap, n),
                    4 => tile256::<4>(qp, k, wb, ap, n),
                    5 => tile256::<5>(qp, k, wb, ap, n),
                    _ => tile256::<6>(qp, k, wb, ap, n),
                }
                r0 += rt;
            }
        }
    }
}

/// Per row: u8 codes of `v` (asymmetric, the range widened to hold 0 exactly), its scale and zero point.
#[inline(always)]
fn quantize_row(v: &[f32], q: &mut [u8]) -> (f32, i32) {
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let zp = ((-lo / scale).clamp(0.0, 255.0) + 0.5).floor();
    let inv = 1.0 / scale;
    // codes >= 0 after the clamp, so +0.5 and truncation round to nearest (vectorizes; f32::round does not)
    for (c, &x) in q.iter_mut().zip(v) {
        *c = ((x * inv + zp).clamp(0.0, 255.0) + 0.5) as u8;
    }
    (scale, zp as i32)
}

/// e^x in f32, branch-free (vectorizes): x = n ln2 + r, |r| <= ln2 / 2, a degree-6 Taylor polynomial of e^r (relative
/// error ~2e-7) times 2^n built in the exponent bits.
#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    const ROUND: f32 = 12582912.0; // 1.5 * 2^23: x + ROUND - ROUND rounds to an integer
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + ROUND) - ROUND;
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    p * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// erf to ~2e-7 absolute (Abramowitz & Stegun 7.1.26), branch-free f32, for exact GELU.
#[inline(always)]
fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    let v = 1.0 - poly * exp_f32(-z * z);
    v.copysign(x)
}

#[inline(always)]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// sum a[i] * b[i] in 16 partial sums (vectorizes).
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

#[derive(Default)]
struct Scratch {
    xq: Vec<u8>,
    xs: Vec<(f32, i32)>,
    rows: Vec<Vec<(usize, f32)>>,
    gq: Vec<u8>,
    acc: Vec<i32>,
    h: Vec<f32>,
    hq: Vec<u8>,
    hs: Vec<(f32, i32)>,
}

/// One TokenMoE layer's weights, prepacked.
pub struct MoeWeights {
    e: usize,
    d: usize,
    f: usize,
    k: usize,
    router: Vec<f32>,
    w1: Vec<Packed>,
    s1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<Packed>,
    s2: Vec<f32>,
    b2: Vec<f32>,
    isa: Isa,
}

impl MoeWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        e: usize,
        d: usize,
        f: usize,
        k: usize,
        router: Vec<f32>,
        w1: &[i8],
        s1: Vec<f32>,
        b1: Vec<f32>,
        w2: &[i8],
        s2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self, String> {
        let want = [
            ("router", router.len(), e * d),
            ("w1", w1.len(), e * f * d),
            ("s1", s1.len(), e * f),
            ("b1", b1.len(), e * f),
            ("w2", w2.len(), e * d * f),
            ("s2", s2.len(), e * d),
            ("b2", b2.len(), e * d),
        ];
        for (name, got, n) in want {
            if got != n {
                return Err(format!(
                    "{name} has {got} values, expected {n} (E {e}, d {d}, f {f})"
                ));
            }
        }
        if !(1..=e).contains(&k) {
            return Err(format!("k {k} with {e} experts"));
        }
        let isa = detect_isa();
        let block = isa != Isa::Plain;
        let w1 = (0..e)
            .map(|i| Packed::new(&w1[i * f * d..(i + 1) * f * d], f, d, block))
            .collect();
        let w2 = (0..e)
            .map(|i| Packed::new(&w2[i * d * f..(i + 1) * d * f], d, f, block))
            .collect();
        Ok(MoeWeights {
            e,
            d,
            f,
            k,
            router,
            w1,
            s1,
            b1,
            w2,
            s2,
            b2,
            isa,
        })
    }

    fn forward(&self, x: &[f32], y: &mut [f32], s: &mut Scratch) {
        #[cfg(target_arch = "x86_64")]
        match self.isa {
            // AVX-512 VNNI CPUs: the routing, quantization and GELU loops 16 wide
            Isa::Vnni512 => return unsafe { self.forward_avx512(x, y, s) },
            // every AVX-VNNI CPU has AVX2 and FMA: 8 wide
            Isa::Vnni256 => return unsafe { self.forward_avx2(x, y, s) },
            Isa::Plain => {}
        }
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    unsafe fn forward_avx512(&self, x: &[f32], y: &mut [f32], s: &mut Scratch) {
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, x: &[f32], y: &mut [f32], s: &mut Scratch) {
        self.forward_body(x, y, s)
    }

    #[inline(always)]
    fn forward_body(&self, x: &[f32], y: &mut [f32], s: &mut Scratch) {
        let (e, d, f, k) = (self.e, self.d, self.f, self.k);
        let n = x.len() / d;
        y.fill(0.0);
        // route: top-k of the router logits, softmax over the k
        s.rows.resize_with(e, Vec::new);
        s.rows.iter_mut().for_each(Vec::clear);
        let mut logits = vec![0.0f32; e];
        let mut top: Vec<usize> = Vec::with_capacity(k);
        for r in 0..n {
            let xr = &x[r * d..(r + 1) * d];
            for (j, l) in logits.iter_mut().enumerate() {
                *l = dot(xr, &self.router[j * d..(j + 1) * d]);
            }
            top.clear();
            for _ in 0..k {
                let best = (0..e)
                    .filter(|j| !top.contains(j))
                    .fold(None, |b: Option<usize>, j| match b {
                        Some(i) if logits[i] >= logits[j] => Some(i),
                        _ => Some(j),
                    })
                    .unwrap();
                top.push(best);
            }
            let m = logits[top[0]];
            let z: f32 = top.iter().map(|&j| (logits[j] - m).exp()).sum();
            for &j in &top {
                s.rows[j].push((r, (logits[j] - m).exp() / z));
            }
        }
        // the frames' u8 codes, once for every expert they go to
        s.xq.resize(n * d, 0);
        s.xs.clear();
        for r in 0..n {
            let qs = quantize_row(&x[r * d..(r + 1) * d], &mut s.xq[r * d..(r + 1) * d]);
            s.xs.push(qs);
        }
        for ex in 0..e {
            let routed = &s.rows[ex];
            let m = routed.len();
            if m == 0 {
                continue;
            }
            s.gq.resize(m * d, 0);
            for (i, &(r, _)) in routed.iter().enumerate() {
                s.gq[i * d..(i + 1) * d].copy_from_slice(&s.xq[r * d..(r + 1) * d]);
            }
            s.acc.resize(m * f.max(d), 0);
            let w1 = &self.w1[ex];
            gemm_u8i8(self.isa, &s.gq, m, w1, &mut s.acc);
            s.h.resize(m * f, 0.0);
            let (s1, b1) = (
                &self.s1[ex * f..(ex + 1) * f],
                &self.b1[ex * f..(ex + 1) * f],
            );
            for (i, &(r, _)) in routed.iter().enumerate() {
                let (sx, zx) = s.xs[r];
                let (acc, h) = (&s.acc[i * f..(i + 1) * f], &mut s.h[i * f..(i + 1) * f]);
                for ((((hv, &a), &c), &sc), &b) in
                    h.iter_mut().zip(acc).zip(&w1.colsum).zip(s1).zip(b1)
                {
                    *hv = gelu(sx * sc * (a - zx * c) as f32 + b);
                }
            }
            s.hq.resize(m * f, 0);
            s.hs.clear();
            for i in 0..m {
                let qs = quantize_row(&s.h[i * f..(i + 1) * f], &mut s.hq[i * f..(i + 1) * f]);
                s.hs.push(qs);
            }
            let w2 = &self.w2[ex];
            gemm_u8i8(self.isa, &s.hq, m, w2, &mut s.acc);
            let (s2, b2) = (
                &self.s2[ex * d..(ex + 1) * d],
                &self.b2[ex * d..(ex + 1) * d],
            );
            for (i, &(r, wt)) in routed.iter().enumerate() {
                let (sh, zh) = s.hs[i];
                let (acc, yr) = (&s.acc[i * d..(i + 1) * d], &mut y[r * d..(r + 1) * d]);
                for ((((yv, &a), &c), &sc), &b) in
                    yr.iter_mut().zip(acc).zip(&w2.colsum).zip(s2).zip(b2)
                {
                    *yv += wt * (sh * sc * (a - zh * c) as f32 + b);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn quantize_rows(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
        let mut q = vec![0i8; rows * cols];
        let mut s = vec![0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            s[r] = if m > 0.0 { m / 127.0 } else { 1.0 };
            for c in 0..cols {
                q[r * cols + c] = (row[c] / s[r]).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (q, s)
    }

    #[test]
    fn every_isa_gives_the_same_integer_sums() {
        let mut rng = Rng(7);
        let (out, inp) = (48, 40);
        let w: Vec<i8> = (0..out * inp).map(|_| (rng.next() * 127.0) as i8).collect();
        let (plain, p) = (
            Packed::new(&w, out, inp, false),
            Packed::new(&w, out, inp, true),
        );
        assert!(plain.blocked.is_empty() && p.raw.is_empty());
        for rows in [1, 3, 8, 9, 13] {
            let q: Vec<u8> = (0..rows * inp)
                .map(|_| ((rng.next() + 1.0) * 127.0) as u8)
                .collect();
            let mut want = vec![0i32; rows * out];
            gemm_plain(&q, rows, &plain, &mut want);
            let mut got = vec![0i32; rows * out];
            gemm_plain(&q, rows, &p, &mut got);
            assert_eq!(got, want, "plain over the blocked layout, rows {rows}");
            for isa in [Isa::Vnni512, Isa::Vnni256] {
                if detect_isa() == Isa::Plain
                    || (isa == Isa::Vnni512 && detect_isa() != Isa::Vnni512)
                {
                    continue;
                }
                #[cfg(target_arch = "x86_64")]
                if isa == Isa::Vnni256 && !std::arch::is_x86_feature_detected!("avxvnni") {
                    continue;
                }
                let mut got = vec![0i32; rows * out];
                gemm_u8i8(isa, &q, rows, &p, &mut got);
                assert_eq!(got, want, "{isa:?} rows {rows}");
            }
        }
    }

    #[test]
    fn erf_and_gelu_match_known_values() {
        for (x, want) in [
            (0.0f32, 0.0f32),
            (0.5, 0.5204999),
            (1.0, 0.8427008),
            (-2.0, -0.9953223),
            (3.5, 0.99999926),
        ] {
            assert!(
                (erf(x) - want).abs() < 2e-7,
                "erf({x}) = {} want {want}",
                erf(x)
            );
        }
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-6);
    }

    /// The kernel against an fp32 MoE with the dequantized weights: top-2 of the same logits, exact GELU.
    #[test]
    fn kernel_matches_the_fp32_moe_of_its_weights() {
        let mut rng = Rng(11);
        let (e, d, f, k, n) = (6, 32, 48, 2, 23);
        let router: Vec<f32> = (0..e * d).map(|_| rng.next()).collect();
        let w1f: Vec<f32> = (0..e * f * d).map(|_| rng.next() * 0.3).collect();
        let w2f: Vec<f32> = (0..e * d * f).map(|_| rng.next() * 0.3).collect();
        let b1: Vec<f32> = (0..e * f).map(|_| rng.next() * 0.1).collect();
        let b2: Vec<f32> = (0..e * d).map(|_| rng.next() * 0.1).collect();
        let (w1, s1) = quantize_rows(&w1f, e * f, d);
        let (w2, s2) = quantize_rows(&w2f, e * d, f);
        let moe = MoeWeights::new(
            e,
            d,
            f,
            k,
            router.clone(),
            &w1,
            s1.clone(),
            b1.clone(),
            &w2,
            s2.clone(),
            b2.clone(),
        )
        .unwrap();
        let x: Vec<f32> = (0..n * d).map(|_| rng.next()).collect();
        let mut y = vec![0f32; n * d];
        moe.forward(&x, &mut y, &mut Scratch::default());
        let deq = |q: &[i8], s: &[f32], cols: usize| -> Vec<f32> {
            q.iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * s[i / cols])
                .collect()
        };
        let (w1d, w2d) = (deq(&w1, &s1, d), deq(&w2, &s2, f));
        let mut worst = 0f32;
        for r in 0..n {
            let xr = &x[r * d..(r + 1) * d];
            let logits: Vec<f32> = (0..e)
                .map(|j| {
                    xr.iter()
                        .zip(&router[j * d..(j + 1) * d])
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect();
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let top = &idx[..k];
            let z: f32 = top
                .iter()
                .map(|&j| (logits[j] - logits[top[0]]).exp())
                .sum();
            let mut want = vec![0f32; d];
            for &j in top {
                let wt = (logits[j] - logits[top[0]]).exp() / z;
                let h: Vec<f32> = (0..f)
                    .map(|o| {
                        let w = &w1d[(j * f + o) * d..(j * f + o + 1) * d];
                        gelu(xr.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() + b1[j * f + o])
                    })
                    .collect();
                for o in 0..d {
                    let w = &w2d[(j * d + o) * f..(j * d + o + 1) * f];
                    want[o] +=
                        wt * (h.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() + b2[j * d + o]);
                }
            }
            let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
            for o in 0..d {
                worst = worst.max((y[r * d + o] - want[o]).abs() / scale);
            }
        }
        // u8 activations on both Linears: about 1% of the output's range
        assert!(worst < 0.03, "worst relative error {worst}");
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use std::time::Instant;

    /// M-tok's layer shape (E 16, d 512, f 1024, top-2) on 30 frames: the kernel and its parts, median of 200.
    /// cargo test -p setsuna-engine --release --lib moe_op::timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mtok_layer_time() {
        let (e, d, f, k, n) = (16, 512, 1024, 2, 30);
        let mut seed = 5u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let router: Vec<f32> = (0..e * d).map(|_| rnd()).collect();
        let w1: Vec<i8> = (0..e * f * d).map(|_| (rnd() * 127.0) as i8).collect();
        let w2: Vec<i8> = (0..e * d * f).map(|_| (rnd() * 127.0) as i8).collect();
        let s1 = vec![0.001f32; e * f];
        let s2 = vec![0.001f32; e * d];
        let (b1, b2) = (vec![0.0f32; e * f], vec![0.0f32; e * d]);
        let moe = MoeWeights::new(e, d, f, k, router, &w1, s1, b1, &w2, s2, b2).unwrap();
        let x: Vec<f32> = (0..n * d).map(|_| rnd()).collect();
        let mut y = vec![0f32; n * d];
        let mut s = Scratch::default();
        let med = |mut t: Vec<f64>| {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2] * 1e3
        };
        let mut t = Vec::new();
        for _ in 0..200 {
            let a = Instant::now();
            moe.forward(&x, &mut y, &mut s);
            t.push(a.elapsed().as_secs_f64());
        }
        println!("isa {:?}: forward {:.3} ms", moe.isa, med(t));
        let q: Vec<u8> = (0..8 * d).map(|i| (i % 251) as u8).collect();
        let mut acc = vec![0i32; 8 * f];
        for rows in [4usize, 8] {
            let mut t = Vec::new();
            for _ in 0..200 {
                let a = Instant::now();
                for ex in 0..e {
                    gemm_u8i8(moe.isa, &q, rows, &moe.w1[ex], &mut acc);
                }
                t.push(a.elapsed().as_secs_f64());
            }
            println!("  gemm1 x16 experts, {rows} rows: {:.3} ms", med(t));
        }
        let h: Vec<f32> = (0..60 * f).map(|_| rnd() * 3.0).collect();
        let mut t = Vec::new();
        let mut sink = 0f32;
        for _ in 0..200 {
            let a = Instant::now();
            for &v in &h {
                sink += gelu(v);
            }
            t.push(a.elapsed().as_secs_f64());
        }
        println!("  gelu on 60 x {f}: {:.3} ms ({sink})", med(t));
    }
}
