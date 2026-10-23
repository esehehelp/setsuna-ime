//! `neo_ime.TokenMoE`: a token-routed mixture of experts as one CPU kernel, registered with every session.
//!
//! The research export's int8 form of a token-routed `RoutedExperts` layer: per frame, the
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
//!
//! `neo_ime.TokenMoEBlocked` is the same op with `w1` / `w2` written already in the kernel's blocked layout (each
//! expert's matrix 16 output rows x 4 inputs a block, `Packed`): the kernel reads them where ONNX Runtime holds
//! them, so the experts (most of a Shiran-v2 model) are in memory once. A distinct op so an engine that predates
//! it refuses such a graph instead of reading blocked weights as rows.

use ort::operator::{
    ComputeContext, Kernel, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput,
};
use ort::value::{DynValueTypeMarker, TensorElementType};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// The custom op domain of the research export's fused ops, as the exported
/// graphs name it (like the manifest schema names, kept as they are).
pub const DOMAIN: &str = "neo_ime";

/// The `neo_ime` operator domain, for `SessionBuilder::with_operators`.
pub fn domain() -> ort::Result<OperatorDomain> {
    OperatorDomain::new(DOMAIN)?
        .add(TokenMoE)?
        .add(TokenMoEBlocked)?
        .add(TokenMoEBlocked4)?
        .add(crate::norm_op::AddLayerNorm)
}

const MOE_INPUTS: [TensorElementType; 8] = {
    use TensorElementType::{Float32, Int8};
    [
        Float32, Float32, Int8, Float32, Float32, Int8, Float32, Float32,
    ]
};

/// `TokenMoEBlocked4`: the weights uint8, two int4 values a byte.
const MOE4_INPUTS: [TensorElementType; 8] = {
    use TensorElementType::{Float32, Uint8};
    [
        Float32, Float32, Uint8, Float32, Float32, Uint8, Float32, Float32,
    ]
};

/// `neo_ime.TokenMoEBlocked4`: `TokenMoEBlocked` with int4 weights (one scale per output row as before), `w1` u8
/// (E, f, d / 2) and `w2` u8 (E, d, f / 2), each 64-value block of the blocked layout as 32 bytes (value b in byte
/// b's low nibble, value b + 32 in its high one, each stored as value + 8, 0..15); read in place, the nibbles taken
/// as they are for the dot products and 8 times the activations' sum taken off after.
struct TokenMoEBlocked4;

impl Operator for TokenMoEBlocked4 {
    type Kernel<'attr> = MoeKernel;

    fn name(&self) -> &str {
        "TokenMoEBlocked4"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        MOE4_INPUTS.map(OperatorInput::required)
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
        let v1 = kctx.constant_input::<DynValueTypeMarker>(2)?;
        let (w1s, w1) = v1.try_extract_tensor::<u8>()?;
        let v2 = kctx.constant_input::<DynValueTypeMarker>(5)?;
        let (_, w2) = v2.try_extract_tensor::<u8>()?;
        let (e, d, f) = (rs[0], rs[1], w1s[1] as usize);
        // SAFETY: as TokenMoEBlocked's
        let moe =
            unsafe { MoeWeights::in_place4(e, d, f, k as usize, router, w1, s1, b1, w2, s2, b2) }
                .map_err(|m| ort::Error::new(format!("TokenMoE: {m}")))?;
        Ok(MoeKernel {
            moe,
            scratch: Scratch::default(),
        })
    }
}

/// The kernel of either op: `blocked`, the weights are in the blocked layout already and read in place.
fn moe_kernel(kctx: &KernelContext<'_>, blocked: bool) -> ort::Result<MoeKernel> {
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
    // the int8 weights are read where ONNX Runtime holds them: packed from there, or (blocked) used there
    let v1 = kctx.constant_input::<DynValueTypeMarker>(2)?;
    let (w1s, w1) = v1.try_extract_tensor::<i8>()?;
    let v2 = kctx.constant_input::<DynValueTypeMarker>(5)?;
    let (_, w2) = v2.try_extract_tensor::<i8>()?;
    let (e, d, f) = (rs[0], rs[1], w1s[1] as usize);
    let moe = if blocked {
        // SAFETY: a constant initializer of the session, immutable and alive as long as the session and so the
        // kernel it built
        unsafe { MoeWeights::in_place(e, d, f, k as usize, router, w1, s1, b1, w2, s2, b2) }
    } else {
        MoeWeights::new(e, d, f, k as usize, router, w1, s1, b1, w2, s2, b2)
    }
    .map_err(|m| ort::Error::new(format!("TokenMoE: {m}")))?;
    Ok(MoeKernel {
        moe,
        scratch: Scratch::default(),
    })
}

struct TokenMoEBlocked;

impl Operator for TokenMoEBlocked {
    type Kernel<'attr> = MoeKernel;

    fn name(&self) -> &str {
        "TokenMoEBlocked"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        MOE_INPUTS.map(OperatorInput::required)
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<MoeKernel> {
        moe_kernel(kctx, true)
    }
}

struct TokenMoE;

impl Operator for TokenMoE {
    type Kernel<'attr> = MoeKernel;

    fn name(&self) -> &str {
        "TokenMoE"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        MOE_INPUTS.map(OperatorInput::required)
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        [OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<MoeKernel> {
        moe_kernel(kctx, false)
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
/// one layout is kept (2026-10-23: both were, the experts' weights three times over with ONNX Runtime's own copy):
/// `blocked` for the VNNI paths, else `raw` (row-major, the plain path).
struct Packed {
    out: usize,
    inp: usize,
    blocked: Blocked,
    raw: Vec<i8>,
    colsum: Vec<i32>,
    /// `blocked` holds int4 values (`TokenMoEBlocked4`): each 64-value block as 32 bytes, byte b the value b in its
    /// low nibble and the value b + 32 in its high one, each as value + 8 (0..15)
    nibbles: bool,
}

/// The blocked layout: packed here, or read in place from ONNX Runtime's initializer of a `TokenMoEBlocked` graph.
enum Blocked {
    Owned(Vec<i8>),
    InPlace(*const i8, usize),
}

impl std::ops::Deref for Blocked {
    type Target = [i8];

    fn deref(&self) -> &[i8] {
        match self {
            Blocked::Owned(v) => v,
            // SAFETY: `in_place`'s contract: memory alive and unchanged while the kernel lives
            Blocked::InPlace(p, n) => unsafe { std::slice::from_raw_parts(*p, *n) },
        }
    }
}

// SAFETY: in place, an immutable initializer of the session, only read
unsafe impl Send for Blocked {}
unsafe impl Sync for Blocked {}

impl Packed {
    /// A matrix already blocked (`TokenMoEBlocked`), read where it lies.
    ///
    /// # Safety
    /// `w` stays valid and unchanged as long as the returned `Packed`.
    unsafe fn in_place(w: &[i8], out: usize, inp: usize) -> Self {
        assert_eq!(w.len(), out * inp);
        assert!(
            out % 16 == 0 && inp % 4 == 0,
            "a blocked matrix is 16 x 4 blocks"
        );
        let k4 = inp / 4;
        let colsum = (0..out)
            .map(|o| {
                (0..k4)
                    .map(|g| {
                        let b = ((o / 16 * k4 + g) * 16 + o % 16) * 4;
                        w[b..b + 4].iter().map(|&v| v as i32).sum::<i32>()
                    })
                    .sum()
            })
            .collect();
        Packed {
            out,
            inp,
            blocked: Blocked::InPlace(w.as_ptr(), w.len()),
            raw: Vec::new(),
            colsum,
            nibbles: false,
        }
    }

    /// A matrix of int4 values already blocked (`TokenMoEBlocked4`, `nibbles`), read where it lies.
    ///
    /// # Safety
    /// `w` stays valid and unchanged as long as the returned `Packed`.
    unsafe fn in_place4(w: &[u8], out: usize, inp: usize) -> Self {
        assert_eq!(w.len() * 2, out * inp);
        assert!(
            out % 16 == 0 && inp % 4 == 0,
            "a blocked matrix is 16 x 4 blocks"
        );
        let mut p = Packed {
            out,
            inp,
            blocked: Blocked::InPlace(w.as_ptr() as *const i8, w.len()),
            raw: Vec::new(),
            colsum: Vec::new(),
            nibbles: true,
        };
        p.colsum = (0..out)
            .map(|o| (0..inp).map(|i| p.at(o, i) as i32).sum())
            .collect();
        p
    }

    /// The value of row `o`, input `i`, whatever the layout.
    #[inline(always)]
    fn at(&self, o: usize, i: usize) -> i8 {
        if !self.raw.is_empty() {
            return self.raw[o * self.inp + i];
        }
        let block = o / 16 * (self.inp / 4) + i / 4;
        let v = (o % 16) * 4 + i % 4;
        if self.nibbles {
            let byte = self.blocked[block * 32 + v % 32] as u8;
            let nib = if v < 32 { byte & 0x0f } else { byte >> 4 };
            nib as i8 - 8
        } else {
            self.blocked[block * 64 + v]
        }
    }

    /// `blocked`: pack for the VNNI paths (when the shape allows 16 x 4 blocks), else keep the matrix as it is.
    fn new(w: &[i8], out: usize, inp: usize, blocked: bool) -> Self {
        assert_eq!(w.len(), out * inp);
        let colsum = (0..out)
            .map(|o| w[o * inp..(o + 1) * inp].iter().map(|&v| v as i32).sum())
            .collect();
        if !(blocked && out % 16 == 0 && inp % 4 == 0) {
            return Packed {
                out,
                inp,
                blocked: Blocked::Owned(Vec::new()),
                raw: w.to_vec(),
                colsum,
                nibbles: false,
            };
        }
        let mut blocked = vec![0i8; out * inp];
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
        Packed {
            out,
            inp,
            blocked: Blocked::Owned(blocked),
            raw: Vec::new(),
            colsum,
            nibbles: false,
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
    match (isa, w.nibbles) {
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni512, false) => unsafe { x86::gemm_vnni512(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni256, false) => unsafe { x86::gemm_vnni256(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni512, true) => unsafe { x86::gemm4_vnni512(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni256, true) => unsafe { x86::gemm4_vnni256(q, rows, w, acc) },
        _ => gemm_plain(q, rows, w, acc),
    }
}

/// Tests only: the VNNI-512 GEMMs on single 16-output blocks, as before the 4-block tiles (timing and sums compared).
#[cfg(test)]
static ONE_BLOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
fn one_block() -> bool {
    ONE_BLOCK.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(test))]
#[inline(always)]
fn one_block() -> bool {
    false
}

fn gemm_plain(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
    let (n, k) = (w.out, w.inp);
    for r in 0..rows {
        let qr = &q[r * k..(r + 1) * k];
        for o in 0..n {
            acc[r * n + o] = if w.nibbles {
                (0..k).map(|i| qr[i] as i32 * w.at(o, i) as i32).sum()
            } else if w.raw.is_empty() {
                // the blocked layout: row o's inputs 4g..4g+4 at ((o / 16 * k / 4 + g) * 16 + o % 16) * 4
                (0..k / 4)
                    .map(|g| {
                        let b = ((o / 16 * (k / 4) + g) * 16 + o % 16) * 4;
                        (0..4)
                            .map(|t| qr[g * 4 + t] as i32 * w.blocked[b + t] as i32)
                            .sum::<i32>()
                    })
                    .sum()
            } else {
                let wr = &w.raw[o * k..(o + 1) * k];
                qr.iter().zip(wr).map(|(&a, &b)| a as i32 * b as i32).sum()
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

    /// `$tile::<rt, $c>($args)` for a row count `rt` known only at run time, up to 6 rows ($c 4) or 8 ($c 1).
    macro_rules! by_rows {
        ($rt:expr, $tile:ident, 4, $($arg:expr),*) => {
            match $rt {
                1 => $tile::<1, 4>($($arg),*),
                2 => $tile::<2, 4>($($arg),*),
                3 => $tile::<3, 4>($($arg),*),
                4 => $tile::<4, 4>($($arg),*),
                5 => $tile::<5, 4>($($arg),*),
                _ => $tile::<6, 4>($($arg),*),
            }
        };
        ($rt:expr, $tile:ident, 1, $($arg:expr),*) => {
            match $rt {
                1 => $tile::<1, 1>($($arg),*),
                2 => $tile::<2, 1>($($arg),*),
                3 => $tile::<3, 1>($($arg),*),
                4 => $tile::<4, 1>($($arg),*),
                5 => $tile::<5, 1>($($arg),*),
                6 => $tile::<6, 1>($($arg),*),
                7 => $tile::<7, 1>($($arg),*),
                _ => $tile::<8, 1>($($arg),*),
            }
        };
    }

    /// The tiles of a blocked matrix: 4 output blocks (64 outputs) by up to 6 rows while 4 blocks are left (24 sums,
    /// 4 weight vectors and the broadcast input in 29 of the 32 registers), then single blocks by up to 8 rows;
    /// `tile(rows in the tile, first row, first block, blocks)`. An expert's few rows on one block left each sum
    /// waiting on the dot product's latency: on 4 blocks the sums are 4 times as many and each weight vector serves
    /// all of the tile's rows.
    #[inline(always)]
    unsafe fn tiles(rows: usize, blocks: usize, mut tile: impl FnMut(usize, usize, usize, usize)) {
        let mut ob = 0;
        while ob < blocks {
            let (c, max) = if ob + 4 <= blocks && !super::one_block() {
                (4, 6)
            } else {
                (1, 8)
            };
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(max);
                tile(rt, r0, ob, c);
                r0 += rt;
            }
            ob += c;
        }
    }

    /// RT rows x C output blocks of int8 weights (block c `bstride` bytes after block 0).
    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn tile512<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [[_mm512_setzero_si512(); C]; RT];
        for kg in 0..k / 4 {
            let mut w = [_mm512_setzero_si512(); C];
            for (c, wc) in w.iter_mut().enumerate() {
                *wc = _mm512_loadu_si512(wb.add(c * bstride + kg * 64) as *const _);
            }
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm512_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    *ac = _mm512_dpbusd_epi32(*ac, x, *wc);
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                _mm512_storeu_si512(acc.add(r * n + c * 16) as *mut _, *ac);
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) unsafe fn gemm_vnni512(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.blocked.as_ptr(), k * 16);
        tiles(rows, n / 16, |rt, r0, ob, c| {
            let (q, a, wb) = (
                qp.add(r0 * k),
                ap.add(r0 * n + ob * 16),
                wp.add(ob * bstride),
            );
            if c == 4 {
                by_rows!(rt, tile512, 4, q, k, wb, bstride, a, n)
            } else {
                by_rows!(rt, tile512, 1, q, k, wb, bstride, a, n)
            }
        });
    }

    /// `$tile::<rt, c>($args)` of the 256-bit tiles: up to 2 rows on 2 blocks, up to 6 on one.
    macro_rules! by_rows256 {
        ($rt:expr, $c:expr, $tile:ident, $($arg:expr),*) => {
            match ($c, $rt) {
                (2, 1) => $tile::<1, 2>($($arg),*),
                (2, _) => $tile::<2, 2>($($arg),*),
                (_, 1) => $tile::<1, 1>($($arg),*),
                (_, 2) => $tile::<2, 1>($($arg),*),
                (_, 3) => $tile::<3, 1>($($arg),*),
                (_, 4) => $tile::<4, 1>($($arg),*),
                (_, 5) => $tile::<5, 1>($($arg),*),
                _ => $tile::<6, 1>($($arg),*),
            }
        };
    }

    /// The 256-bit tiles (16 registers: a block's 16 outputs are two): up to 6 rows on one block (12 sums), and a
    /// last 1 or 2 rows on 2 blocks at once while 2 are left (8 sums, 4 weight vectors) instead of one block's 2 or 4
    /// sums waiting on the dot product's latency; `tile(rows in the tile, first row, first block, blocks)`.
    #[inline(always)]
    unsafe fn tiles256(
        rows: usize,
        blocks: usize,
        mut tile: impl FnMut(usize, usize, usize, usize),
    ) {
        let mut ob = 0;
        while ob < blocks {
            let c = if ob + 2 <= blocks && !super::one_block() {
                2
            } else {
                1
            };
            let mut r0 = 0;
            while r0 < rows {
                let left = rows - r0;
                if c == 2 && left <= 2 {
                    tile(left, r0, ob, 2);
                    r0 += left;
                } else {
                    let rt = left.min(6);
                    for b in ob..ob + c {
                        tile(rt, r0, b, 1);
                    }
                    r0 += rt;
                }
            }
            ob += c;
        }
    }

    /// RT rows x C blocks of int8 weights (block c `bstride` bytes after block 0), each block's outputs 0-7 and 8-15.
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn tile256<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [[[_mm256_setzero_si256(); 2]; C]; RT];
        for kg in 0..k / 4 {
            let mut w = [[_mm256_setzero_si256(); 2]; C];
            for (c, wc) in w.iter_mut().enumerate() {
                let p = wb.add(c * bstride + kg * 64);
                *wc = [
                    _mm256_loadu_si256(p as *const _),
                    _mm256_loadu_si256(p.add(32) as *const _),
                ];
            }
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm256_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    ac[0] = _mm256_dpbusd_avx_epi32(ac[0], x, wc[0]);
                    ac[1] = _mm256_dpbusd_avx_epi32(ac[1], x, wc[1]);
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                _mm256_storeu_si256(acc.add(r * n + c * 16) as *mut _, ac[0]);
                _mm256_storeu_si256(acc.add(r * n + c * 16 + 8) as *mut _, ac[1]);
            }
        }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    pub(super) unsafe fn gemm_vnni256(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.blocked.as_ptr(), k * 16);
        tiles256(rows, n / 16, |rt, r0, ob, c| {
            let (q, a, wb) = (
                qp.add(r0 * k),
                ap.add(r0 * n + ob * 16),
                wp.add(ob * bstride),
            );
            by_rows256!(rt, c, tile256, q, k, wb, bstride, a, n)
        });
    }

    /// The 64 int4 values of a block (32 bytes: value b in byte b's low nibble, value b + 32 in its high one) as
    /// int8, values 0-31 then 32-63.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack4(p: *const i8) -> (__m256i, __m256i) {
        let v = _mm256_loadu_si256(p as *const _);
        let m = _mm256_set1_epi8(0x0f);
        // value + 8 as stored (0..15): the products' 8 * (row sum of the codes) comes off after (`sub_offset`)
        (
            _mm256_and_si256(v, m),
            _mm256_and_si256(_mm256_srli_epi16(v, 4), m),
        )
    }

    /// acc[r][o] -= 8 * sum_i q[r][i]: the int4 values were multiplied as stored, value + 8.
    #[inline]
    fn sub_offset(q: &[u8], rows: usize, k: usize, acc: &mut [i32], n: usize) {
        for r in 0..rows {
            let off = 8 * q[r * k..(r + 1) * k].iter().map(|&v| v as i32).sum::<i32>();
            acc[r * n..(r + 1) * n].iter_mut().for_each(|a| *a -= off);
        }
    }

    /// `tile512` on int4 values: each block unpacked to int8 once for the tile's rows.
    #[target_feature(enable = "avx512f,avx512vnni,avx2")]
    unsafe fn tile4_512<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [[_mm512_setzero_si512(); C]; RT];
        for kg in 0..k / 4 {
            let mut w = [_mm512_setzero_si512(); C];
            for (c, wc) in w.iter_mut().enumerate() {
                let (lo, hi) = unpack4(wb.add(c * bstride + kg * 32));
                *wc = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            }
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm512_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    *ac = _mm512_dpbusd_epi32(*ac, x, *wc);
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                _mm512_storeu_si512(acc.add(r * n + c * 16) as *mut _, *ac);
            }
        }
    }

    /// `gemm_vnni512` on int4 values (`Packed::nibbles`).
    #[target_feature(enable = "avx512f,avx512vnni,avx2")]
    pub(super) unsafe fn gemm4_vnni512(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.blocked.as_ptr(), k * 8);
        tiles(rows, n / 16, |rt, r0, ob, c| {
            let (q, a, wb) = (
                qp.add(r0 * k),
                ap.add(r0 * n + ob * 16),
                wp.add(ob * bstride),
            );
            if c == 4 {
                by_rows!(rt, tile4_512, 4, q, k, wb, bstride, a, n)
            } else {
                by_rows!(rt, tile4_512, 1, q, k, wb, bstride, a, n)
            }
        });
        sub_offset(q, rows, k, acc, n);
    }

    /// `tile256` on int4 values: each block unpacked to int8 once for the tile's rows.
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn tile4_256<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [[[_mm256_setzero_si256(); 2]; C]; RT];
        for kg in 0..k / 4 {
            let mut w = [[_mm256_setzero_si256(); 2]; C];
            for (c, wc) in w.iter_mut().enumerate() {
                let (lo, hi) = unpack4(wb.add(c * bstride + kg * 32));
                *wc = [lo, hi];
            }
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm256_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    ac[0] = _mm256_dpbusd_avx_epi32(ac[0], x, wc[0]);
                    ac[1] = _mm256_dpbusd_avx_epi32(ac[1], x, wc[1]);
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                _mm256_storeu_si256(acc.add(r * n + c * 16) as *mut _, ac[0]);
                _mm256_storeu_si256(acc.add(r * n + c * 16 + 8) as *mut _, ac[1]);
            }
        }
    }

    /// `gemm_vnni256` on int4 values (`Packed::nibbles`).
    #[target_feature(enable = "avx2,avxvnni")]
    pub(super) unsafe fn gemm4_vnni256(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.blocked.as_ptr(), k * 8);
        tiles256(rows, n / 16, |rt, r0, ob, c| {
            let (q, a, wb) = (
                qp.add(r0 * k),
                ap.add(r0 * n + ob * 16),
                wp.add(ob * bstride),
            );
            by_rows256!(rt, c, tile4_256, q, k, wb, bstride, a, n)
        });
        sub_offset(q, rows, k, acc, n);
    }
}

/// Per row: u8 codes of `v` (asymmetric, the range widened to hold 0 exactly), its scale and zero point.
#[inline(always)]
fn quantize_row(v: &[f32], q: &mut [u8]) -> (f32, i32) {
    // the range in 16 lanes by compares (one vector min / max each; f32::min / max and one running pair did not
    // vectorize: 1.4 ns a value), NaN skipped as f32::min skips it
    let (mut lo16, mut hi16) = ([0.0f32; 16], [0.0f32; 16]);
    let chunks = v.chunks_exact(16);
    let tail = chunks.remainder();
    for c in chunks {
        for i in 0..16 {
            lo16[i] = if c[i] < lo16[i] { c[i] } else { lo16[i] };
            hi16[i] = if c[i] > hi16[i] { c[i] } else { hi16[i] };
        }
    }
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for &x in lo16.iter().chain(tail) {
        lo = lo.min(x);
    }
    for &x in hi16.iter().chain(tail) {
        hi = hi.max(x);
    }
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let zp = ((-lo / scale).clamp(0.0, 255.0) + 0.5).floor();
    let inv = 1.0 / scale;
    // codes >= 0 after the clamp, so +0.5 and truncation round to nearest (f32::round does not vectorize); the clamp
    // by compares, NaN failing the first one to code 0 as before, so the truncation needs no saturating checks (`as
    // i32` kept the loop scalar: 0.58 ns a value)
    for (c, &x) in q.iter_mut().zip(v) {
        let y = x * inv + zp;
        let y = if y >= 0.0 { y } else { 0.0 };
        let y = if y < 255.0 { y } else { 255.0 };
        // SAFETY: y + 0.5 is in [0.5, 255.5], finite and within i32
        *c = unsafe { (y + 0.5).to_int_unchecked::<i32>() } as u8;
    }
    (scale, zp as i32)
}

/// erf for exact GELU, branch-free f32 (vectorizes): one rational function on [-4, 4], an odd polynomial of degree 13
/// over an even one of degree 8 (the f32 erf of Eigen and XLA), and +-1 to f32 precision outside; 12 multiply-adds
/// and a division, 3.5x faster in the experts' loop than the exponential form it replaced (Abramowitz & Stegun 7.1.26
/// with a polynomial e^x), within 5e-7 of it.
#[inline(always)]
fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = x2 * -2.726_142_3e-10 + 2.770_681_4e-8;
    p = x2 * p - 2.101_024e-6;
    p = x2 * p - 5.692_506_4e-5;
    p = x2 * p - 7.349_906_3e-4;
    p = x2 * p - 2.954_599_8e-3;
    p = x2 * p - 1.609_603_3e-2;
    let mut q = x2 * -1.456_607_2e-5 - 2.133_740_6e-4;
    q = x2 * q - 1.682_827e-3;
    q = x2 * q - 7.373_329_2e-3;
    q = x2 * q - 1.426_473_9e-2;
    x * p / q
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
    out: Vec<f32>,
    es: ExpertScratch,
}

/// One expert's buffers.
#[derive(Default)]
struct ExpertScratch {
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
        let blocked = detect_isa() != Isa::Plain;
        Self::build(
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
            1,
            |w, out, inp| Packed::new(w, out, inp, blocked),
        )
    }

    /// `new` for weights already in the blocked layout (`TokenMoEBlocked`), read in place.
    ///
    /// # Safety
    /// `w1` and `w2` stay valid and unchanged as long as the returned weights.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn in_place(
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
        if f % 16 != 0 || d % 16 != 0 {
            return Err(format!(
                "blocked weights need d and f in 16s (d {d}, f {f})"
            ));
        }
        Self::build(
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
            1,
            |w, out, inp| Packed::in_place(w, out, inp),
        )
    }

    /// `in_place` for int4 weights (`TokenMoEBlocked4`): `w1` / `w2` two values a byte.
    ///
    /// # Safety
    /// As `in_place`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn in_place4(
        e: usize,
        d: usize,
        f: usize,
        k: usize,
        router: Vec<f32>,
        w1: &[u8],
        s1: Vec<f32>,
        b1: Vec<f32>,
        w2: &[u8],
        s2: Vec<f32>,
        b2: Vec<f32>,
    ) -> Result<Self, String> {
        if f % 16 != 0 || d % 16 != 0 {
            return Err(format!(
                "blocked weights need d and f in 16s (d {d}, f {f})"
            ));
        }
        Self::build(
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
            2,
            |w, out, inp| Packed::in_place4(w, out, inp),
        )
    }

    /// The weights of `e` experts from `w1` / `w2` holding `per` values an element, each expert's matrix made by
    /// `pack`.
    #[allow(clippy::too_many_arguments)]
    fn build<T>(
        e: usize,
        d: usize,
        f: usize,
        k: usize,
        router: Vec<f32>,
        w1: &[T],
        s1: Vec<f32>,
        b1: Vec<f32>,
        w2: &[T],
        s2: Vec<f32>,
        b2: Vec<f32>,
        per: usize,
        pack: impl Fn(&[T], usize, usize) -> Packed,
    ) -> Result<Self, String> {
        let want = [
            ("router", router.len(), e * d),
            ("w1", w1.len() * per, e * f * d),
            ("s1", s1.len(), e * f),
            ("b1", b1.len(), e * f),
            ("w2", w2.len() * per, e * d * f),
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
        let m = f * d / per;
        let w1 = (0..e)
            .map(|i| pack(&w1[i * m..(i + 1) * m], f, d))
            .collect();
        let w2 = (0..e)
            .map(|i| pack(&w2[i * m..(i + 1) * m], d, f))
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
        self.forward_in(x, y, s, pool())
    }

    /// `forward` with the experts on `pool` (None: one after another).
    fn forward_in(
        &self,
        x: &[f32],
        y: &mut [f32],
        s: &mut Scratch,
        pool: Option<&rayon::ThreadPool>,
    ) {
        #[cfg(target_arch = "x86_64")]
        match self.isa {
            // AVX-512 VNNI CPUs: the routing, quantization and GELU loops 16 wide
            Isa::Vnni512 => return unsafe { self.forward_avx512(x, y, s, pool) },
            // every AVX-VNNI CPU has AVX2 and FMA: 8 wide
            Isa::Vnni256 => return unsafe { self.forward_avx2(x, y, s, pool) },
            Isa::Plain => {}
        }
        self.forward_body(x, y, s, pool)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    unsafe fn forward_avx512(
        &self,
        x: &[f32],
        y: &mut [f32],
        s: &mut Scratch,
        pool: Option<&rayon::ThreadPool>,
    ) {
        self.forward_body(x, y, s, pool)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(
        &self,
        x: &[f32],
        y: &mut [f32],
        s: &mut Scratch,
        pool: Option<&rayon::ThreadPool>,
    ) {
        self.forward_body(x, y, s, pool)
    }

    #[inline(always)]
    fn forward_body(
        &self,
        x: &[f32],
        y: &mut [f32],
        s: &mut Scratch,
        pool: Option<&rayon::ThreadPool>,
    ) {
        let (e, d, k) = (self.e, self.d, self.k);
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
        let active: Vec<usize> = (0..e).filter(|&ex| !s.rows[ex].is_empty()).collect();
        let Scratch {
            rows,
            xq,
            xs,
            out,
            es,
        } = s;
        match pool {
            // the experts on the pool, each into its own buffer, then added in expert order as below: the same sums
            Some(p) if active.len() > 1 => {
                let outs: Vec<Vec<f32>> = p.install(|| {
                    active
                        .par_iter()
                        .map(|&ex| {
                            let mut o = vec![0f32; rows[ex].len() * d];
                            let mut sc = ExpertScratch::default();
                            self.expert(ex, &rows[ex], xq, xs, &mut o, &mut sc);
                            o
                        })
                        .collect()
                });
                for (&ex, o) in active.iter().zip(&outs) {
                    add_routed(y, &rows[ex], o, d);
                }
            }
            _ => {
                for &ex in &active {
                    out.resize(rows[ex].len() * d, 0.0);
                    self.expert(ex, &rows[ex], xq, xs, out, es);
                    add_routed(y, &rows[ex], out, d);
                }
            }
        }
    }

    /// Expert `ex` on its routed frames (`routed`: frame, weight; `xq` / `xs` every frame's u8 codes, scale and zero
    /// point): its output for each, unweighted, into `out` (m x d).
    fn expert(
        &self,
        ex: usize,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        out: &mut [f32],
        s: &mut ExpertScratch,
    ) {
        #[cfg(target_arch = "x86_64")]
        match self.isa {
            Isa::Vnni512 => return unsafe { self.expert_avx512(ex, routed, xq, xs, out, s) },
            Isa::Vnni256 => return unsafe { self.expert_avx2(ex, routed, xq, xs, out, s) },
            Isa::Plain => {}
        }
        self.expert_body(ex, routed, xq, xs, out, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
    unsafe fn expert_avx512(
        &self,
        ex: usize,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        out: &mut [f32],
        s: &mut ExpertScratch,
    ) {
        self.expert_body(ex, routed, xq, xs, out, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn expert_avx2(
        &self,
        ex: usize,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        out: &mut [f32],
        s: &mut ExpertScratch,
    ) {
        self.expert_body(ex, routed, xq, xs, out, s)
    }

    #[inline(always)]
    fn expert_body(
        &self,
        ex: usize,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        out: &mut [f32],
        s: &mut ExpertScratch,
    ) {
        let (d, f) = (self.d, self.f);
        let m = routed.len();
        s.gq.resize(m * d, 0);
        for (i, &(r, _)) in routed.iter().enumerate() {
            s.gq[i * d..(i + 1) * d].copy_from_slice(&xq[r * d..(r + 1) * d]);
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
            let (sx, zx) = xs[r];
            let (acc, h) = (&s.acc[i * f..(i + 1) * f], &mut s.h[i * f..(i + 1) * f]);
            for ((((hv, &a), &c), &sc), &b) in h.iter_mut().zip(acc).zip(&w1.colsum).zip(s1).zip(b1)
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
        for i in 0..m {
            let (sh, zh) = s.hs[i];
            let (acc, o) = (&s.acc[i * d..(i + 1) * d], &mut out[i * d..(i + 1) * d]);
            for ((((ov, &a), &c), &sc), &b) in o.iter_mut().zip(acc).zip(&w2.colsum).zip(s2).zip(b2)
            {
                *ov = sh * sc * (a - zh * c) as f32 + b;
            }
        }
    }
}

/// y[r] += w * out[i] for the i-th routed frame (r, w) of an expert.
#[inline(always)]
fn add_routed(y: &mut [f32], routed: &[(usize, f32)], out: &[f32], d: usize) {
    for (i, &(r, wt)) in routed.iter().enumerate() {
        for (yv, &v) in y[r * d..(r + 1) * d]
            .iter_mut()
            .zip(&out[i * d..(i + 1) * d])
        {
            *yv += wt * v;
        }
    }
}

/// Threads a layer's experts are spread over (`set_threads`: the session's intra-op threads); 1, one after another.
static THREADS: AtomicUsize = AtomicUsize::new(1);
static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// The experts' threads, set before the first conversion: a pool of `n` (n > 1) is built at the first one.
pub fn set_threads(n: usize) {
    THREADS.store(n.max(1), Ordering::Relaxed);
}

fn pool() -> Option<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        let n = THREADS.load(Ordering::Relaxed);
        (n > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("setsuna-moe-{i}"))
                    .build()
                    .ok()
            })
            .flatten()
    })
    .as_ref()
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
        let (out, inp) = (80, 40);
        let w: Vec<i8> = (0..out * inp).map(|_| (rng.next() * 127.0) as i8).collect();
        let (raw, p) = (
            Packed::new(&w, out, inp, false),
            Packed::new(&w, out, inp, true),
        );
        assert!(
            raw.blocked.is_empty() && p.raw.is_empty(),
            "one layout each"
        );
        for rows in [1, 3, 8, 9, 13] {
            let q: Vec<u8> = (0..rows * inp)
                .map(|_| ((rng.next() + 1.0) * 127.0) as u8)
                .collect();
            let mut want = vec![0i32; rows * out];
            gemm_plain(&q, rows, &raw, &mut want);
            // the plain path on the blocked layout (a CPU without VNNI never packs, but it must agree)
            let mut got = vec![0i32; rows * out];
            gemm_plain(&q, rows, &p, &mut got);
            assert_eq!(got, want, "plain on blocked, rows {rows}");
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

    /// `TokenMoEBlocked`: weights written blocked and read in place give the outputs of `TokenMoE` on the same
    /// weights as rows.
    #[test]
    fn blocked_weights_read_in_place_match_the_row_weights() {
        let mut rng = Rng(5);
        let (e, d, f, k, n) = (4, 64, 80, 2, 17);
        let router: Vec<f32> = (0..e * d).map(|_| rng.next()).collect();
        let w1f: Vec<f32> = (0..e * f * d).map(|_| rng.next() * 0.3).collect();
        let w2f: Vec<f32> = (0..e * d * f).map(|_| rng.next() * 0.3).collect();
        let b1: Vec<f32> = (0..e * f).map(|_| rng.next() * 0.1).collect();
        let b2: Vec<f32> = (0..e * d).map(|_| rng.next() * 0.1).collect();
        let (w1, s1) = quantize_rows(&w1f, e * f, d);
        let (w2, s2) = quantize_rows(&w2f, e * d, f);
        // each expert's matrix blocked as an export writes it
        let block = |w: &[i8], out: usize, inp: usize| -> Vec<i8> {
            (0..e)
                .flat_map(|i| {
                    Packed::new(&w[i * out * inp..(i + 1) * out * inp], out, inp, true)
                        .blocked
                        .to_vec()
                })
                .collect()
        };
        let (w1b, w2b) = (block(&w1, f, d), block(&w2, d, f));
        let rows = MoeWeights::new(
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
        let blocked =
            unsafe { MoeWeights::in_place(e, d, f, k, router, &w1b, s1, b1, &w2b, s2, b2) }
                .unwrap();
        let x: Vec<f32> = (0..n * d).map(|_| rng.next()).collect();
        let (mut ya, mut yb) = (vec![0f32; n * d], vec![0f32; n * d]);
        rows.forward(&x, &mut ya, &mut Scratch::default());
        blocked.forward(&x, &mut yb, &mut Scratch::default());
        assert_eq!(ya, yb);
        for (a, b) in rows.w1.iter().zip(&blocked.w1) {
            assert_eq!(a.colsum, b.colsum);
        }
    }

    /// int8 values in -8..=7 written as `TokenMoEBlocked4` writes them: blocked, each 64-value block as 32 bytes,
    /// value b in byte b's low nibble and value b + 32 in its high one.
    pub(super) fn nibble_pack(w: &[i8], out: usize, inp: usize) -> Vec<u8> {
        let b = Packed::new(w, out, inp, true);
        b.blocked
            .chunks(64)
            .flat_map(|blk| {
                (0..32).map(move |i| ((blk[i] + 8) as u8 & 0x0f) | (((blk[i + 32] + 8) as u8) << 4))
            })
            .collect()
    }

    #[test]
    fn int4_weights_give_the_int8_paths_sums() {
        let mut rng = Rng(9);
        let (out, inp) = (80, 40);
        let w: Vec<i8> = (0..out * inp)
            .map(|_| ((rng.next() * 8.0) as i8).clamp(-8, 7))
            .collect();
        let raw = Packed::new(&w, out, inp, false);
        let bytes = nibble_pack(&w, out, inp);
        let p = unsafe { Packed::in_place4(&bytes, out, inp) };
        assert_eq!(p.colsum, raw.colsum);
        for o in 0..out {
            for i in 0..inp {
                assert_eq!(p.at(o, i), w[o * inp + i], "value {o} {i}");
            }
        }
        for rows in [1, 3, 8, 9, 13] {
            let q: Vec<u8> = (0..rows * inp)
                .map(|_| ((rng.next() + 1.0) * 127.0) as u8)
                .collect();
            let mut want = vec![0i32; rows * out];
            gemm_plain(&q, rows, &raw, &mut want);
            for isa in [Isa::Plain, Isa::Vnni512, Isa::Vnni256] {
                if isa == Isa::Vnni512 && detect_isa() != Isa::Vnni512 {
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

    /// `TokenMoEBlocked4` gives `TokenMoE`'s outputs on the same int4-valued weights.
    #[test]
    fn int4_experts_match_the_int8_kernel_on_the_same_values() {
        let mut rng = Rng(13);
        let (e, d, f, k, n) = (4, 64, 80, 2, 19);
        let router: Vec<f32> = (0..e * d).map(|_| rng.next()).collect();
        let int4 = |len: usize, rng: &mut Rng| -> Vec<i8> {
            (0..len)
                .map(|_| ((rng.next() * 8.0) as i8).clamp(-7, 7))
                .collect()
        };
        let (w1, w2) = (int4(e * f * d, &mut rng), int4(e * d * f, &mut rng));
        let s1: Vec<f32> = (0..e * f)
            .map(|_| (rng.next().abs() + 0.1) * 0.05)
            .collect();
        let s2: Vec<f32> = (0..e * d)
            .map(|_| (rng.next().abs() + 0.1) * 0.05)
            .collect();
        let b1: Vec<f32> = (0..e * f).map(|_| rng.next() * 0.1).collect();
        let b2: Vec<f32> = (0..e * d).map(|_| rng.next() * 0.1).collect();
        let pack_all = |w: &[i8], out: usize, inp: usize| -> Vec<u8> {
            (0..e)
                .flat_map(|i| nibble_pack(&w[i * out * inp..(i + 1) * out * inp], out, inp))
                .collect()
        };
        let (w1b, w2b) = (pack_all(&w1, f, d), pack_all(&w2, d, f));
        let rows = MoeWeights::new(
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
        let nib = unsafe { MoeWeights::in_place4(e, d, f, k, router, &w1b, s1, b1, &w2b, s2, b2) }
            .unwrap();
        let x: Vec<f32> = (0..n * d).map(|_| rng.next()).collect();
        let (mut ya, mut yb) = (vec![0f32; n * d], vec![0f32; n * d]);
        rows.forward(&x, &mut ya, &mut Scratch::default());
        nib.forward(&x, &mut yb, &mut Scratch::default());
        assert_eq!(ya, yb);
        // the experts on a pool add up as one after another, bit for bit
        let p = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for moe in [&rows, &nib] {
            let (mut serial, mut pooled) = (vec![0f32; n * d], vec![0f32; n * d]);
            moe.forward_in(&x, &mut serial, &mut Scratch::default(), None);
            moe.forward_in(&x, &mut pooled, &mut Scratch::default(), Some(&p));
            assert_eq!(serial, pooled);
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
            (5.0, 1.0),
        ] {
            assert!(
                (erf(x) - want).abs() < 3e-7,
                "erf({x}) = {} want {want}",
                erf(x)
            );
        }
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-6);
    }

    /// The rational erf against the exponential form it replaced (Abramowitz & Stegun 7.1.26, within 1.5e-7 of erf;
    /// here in f64) on [-6, 6].
    #[test]
    fn erf_matches_the_exponential_form() {
        let reference = |x: f64| -> f64 {
            let z = x.abs();
            let t = 1.0 / (1.0 + 0.3275911 * z);
            let poly = t
                * (0.254829592
                    + t * (-0.284496736
                        + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            (1.0 - poly * (-z * z).exp()).copysign(x)
        };
        let worst = (-60_000..=60_000)
            .map(|i| i as f32 * 1e-4)
            .map(|x| (erf(x) as f64 - reference(x as f64)).abs())
            .fold(0f64, f64::max);
        assert!(worst < 5e-7, "worst difference {worst}");
    }

    /// The vectorized range and codes of `quantize_row` against the plain form it replaced, NaN and infinities
    /// included.
    #[test]
    fn quantize_row_gives_the_plain_forms_codes() {
        let plain = |v: &[f32], q: &mut [u8]| -> (f32, i32) {
            let (mut lo, mut hi) = (0.0f32, 0.0f32);
            for &x in v {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
            let zp = ((-lo / scale).clamp(0.0, 255.0) + 0.5).floor();
            let inv = 1.0 / scale;
            for (c, &x) in q.iter_mut().zip(v) {
                *c = ((x * inv + zp).clamp(0.0, 255.0) + 0.5) as u8;
            }
            (scale, zp as i32)
        };
        let mut rng = Rng(17);
        for len in [1usize, 15, 16, 17, 40, 512, 1024, 1031] {
            for case in 0..4 {
                let mut v: Vec<f32> = (0..len)
                    .map(|_| rng.next() * 7.0 + if case == 1 { 3.0 } else { 0.0 })
                    .collect();
                if case == 2 {
                    v[len / 2] = f32::NAN;
                }
                if case == 3 {
                    v[len - 1] = f32::INFINITY;
                }
                let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                let (sa, za) = plain(&v, &mut a);
                let (sb, zb) = quantize_row(&v, &mut b);
                assert_eq!(
                    (sa.to_bits(), za),
                    (sb.to_bits(), zb),
                    "len {len} case {case}"
                );
                assert_eq!(a, b, "len {len} case {case}");
            }
        }
    }

    /// The kernel against an fp32 MoE with the dequantized weights: top-2 of the same logits, exact GELU.
    #[test]
    fn kernel_matches_the_fp32_moe_of_its_weights() {
        let mut rng = Rng(11);
        let (e, d, f, k, n) = (6, 64, 80, 2, 23);
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
        // the expert's dequantize + GELU loop as `expert_body` runs it (AVX-512 enabled), 120 rows of f
        #[cfg(target_arch = "x86_64")]
        if moe.isa == Isa::Vnni512 {
            #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
            unsafe fn dequant_gelu(
                acc: &[i32],
                colsum: &[i32],
                s: &[f32],
                b: &[f32],
                h: &mut [f32],
            ) {
                for ((((hv, &a), &c), &sc), &bv) in h.iter_mut().zip(acc).zip(colsum).zip(s).zip(b)
                {
                    *hv = gelu(0.01 * sc * (a - 3 * c) as f32 + bv);
                }
            }
            let acc: Vec<i32> = (0..f).map(|i| (i as i32 % 2001) - 1000).collect();
            let colsum: Vec<i32> = (0..f).map(|i| i as i32 % 7).collect();
            let (sc, bv) = (vec![0.01f32; f], vec![0.1f32; f]);
            let mut h = vec![0f32; f];
            let mut t = Vec::new();
            for _ in 0..200 {
                let a = Instant::now();
                for _ in 0..120 {
                    unsafe { dequant_gelu(&acc, &colsum, &sc, &bv, &mut h) };
                    sink += h[3];
                }
                t.push(a.elapsed().as_secs_f64());
            }
            println!(
                "  dequantize + gelu, 120 x {f} as in expert_body: {:.3} ms ({sink})",
                med(t)
            );
            #[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
            unsafe fn quantize(h: &[f32], q: &mut [u8]) -> (f32, i32) {
                quantize_row(h, q)
            }
            let hv: Vec<f32> = (0..f)
                .map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.01)
                .collect();
            let mut hq = vec![0u8; f];
            let mut t = Vec::new();
            let mut zs = 0i32;
            for _ in 0..200 {
                let a = Instant::now();
                for _ in 0..120 {
                    zs += unsafe { quantize(&hv, &mut hq) }.1;
                }
                t.push(a.elapsed().as_secs_f64());
            }
            println!("  quantize_row, 120 x {f}: {:.3} ms ({zs})", med(t));
        }

        // int4 values (TokenMoEBlocked4) on more frames: the whole forward, and its two GEMMs over every expert
        let w1: Vec<i8> = w1.iter().map(|&v| (v / 16).clamp(-8, 7)).collect();
        let w2: Vec<i8> = w2.iter().map(|&v| (v / 16).clamp(-8, 7)).collect();
        let pack = |w: &[i8], out: usize, inp: usize| -> Vec<u8> {
            (0..e)
                .flat_map(|i| {
                    super::tests::nibble_pack(&w[i * out * inp..(i + 1) * out * inp], out, inp)
                })
                .collect()
        };
        let (w1b, w2b) = (pack(&w1, f, d), pack(&w2, d, f));
        let router: Vec<f32> = (0..e * d).map(|_| rnd()).collect();
        let nib = unsafe {
            MoeWeights::in_place4(
                e,
                d,
                f,
                k,
                router,
                &w1b,
                vec![0.01f32; e * f],
                vec![0.0f32; e * f],
                &w2b,
                vec![0.01f32; e * d],
                vec![0.0f32; e * d],
            )
        }
        .unwrap();
        let xs: Vec<Vec<f32>> = [10usize, 30, 60, 100]
            .iter()
            .map(|&n| (0..n * d).map(|_| rnd()).collect())
            .collect();
        let q: Vec<u8> = (0..16 * f).map(|i| (i % 251) as u8).collect();
        let mut acc = vec![0i32; 16 * f];
        // single 16-output blocks (the kernel before 4-block tiles), then the tiles
        for one in [true, false] {
            ONE_BLOCK.store(one, std::sync::atomic::Ordering::Relaxed);
            println!("{} blocks a tile:", if one { 1 } else { 4 });
            let mut t = Vec::new();
            for _ in 0..200 {
                let a = Instant::now();
                moe.forward(&xs[1], &mut y, &mut s);
                t.push(a.elapsed().as_secs_f64());
            }
            println!("  int8 forward, 30 frames: {:.3} ms", med(t));
            for x in &xs {
                let n = x.len() / d;
                let mut y = vec![0f32; n * d];
                let mut t = Vec::new();
                for _ in 0..200 {
                    let a = Instant::now();
                    nib.forward(x, &mut y, &mut s);
                    t.push(a.elapsed().as_secs_f64());
                }
                println!("  int4 forward, {n} frames: {:.3} ms", med(t));
            }
            for rows in [2usize, 4, 6, 8, 16] {
                let (mut t1, mut t2) = (Vec::new(), Vec::new());
                for _ in 0..200 {
                    let a = Instant::now();
                    for ex in 0..e {
                        gemm_u8i8(nib.isa, &q, rows, &nib.w1[ex], &mut acc);
                    }
                    t1.push(a.elapsed().as_secs_f64());
                    let a = Instant::now();
                    for ex in 0..e {
                        gemm_u8i8(nib.isa, &q, rows, &nib.w2[ex], &mut acc);
                    }
                    t2.push(a.elapsed().as_secs_f64());
                }
                println!(
                    "  int4 gemm1 / gemm2 x16 experts, {rows} rows: {:.3} / {:.3} ms",
                    med(t1),
                    med(t2)
                );
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("avxvnni") {
                    let mut t = Vec::new();
                    for _ in 0..200 {
                        let a = Instant::now();
                        for ex in 0..e {
                            gemm_u8i8(Isa::Vnni256, &q, rows, &nib.w1[ex], &mut acc);
                        }
                        t.push(a.elapsed().as_secs_f64());
                    }
                    println!("    the same gemm1 on AVX-VNNI (256): {:.3} ms", med(t));
                }
            }
        }
        ONE_BLOCK.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
