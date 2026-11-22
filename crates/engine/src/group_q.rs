//! `neo_ime.LinearQ4G`: a Linear on low-bit codes with a scale and a zero point per group of inputs, as one CPU
//! kernel (the research side's quantization-aware training grids: k bits from each group's min to its max, 1 bit
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
//! expanded). AVX-512 VNNI tiles of up to six rows keep their integer sums in registers. Paired nibbles
//! expand directly to two full vectors with no lane shuffles. Each group updates the running float sums
//! with separate multiply and add operations, in the same order on every path (no fused multiply-add).

// the x86 kernels and what only they read are unused elsewhere
#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

#[cfg(test)]
use crate::moe_op::quantize_row;
use crate::moe_op::{add_routed, gelu, quantize_row_scalar, route, span};

/// A `GroupMoe` on the experts' nibble codes `w1` (E, d / g, f, g / 2) and `w2` (E, f / g, d, g / 2) with their
/// scales and zero points (E, out, in / g) and biases (each expert's codes copied: `GroupWeights::in_place`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn group_moe(
    e: usize,
    d: usize,
    f: usize,
    k: usize,
    g: usize,
    router: Vec<f32>,
    (w1, s1, z1, b1): (&[u8], &[f32], &[f32], Vec<f32>),
    (w2, s2, z2, b2): (&[u8], &[f32], &[f32], Vec<f32>),
) -> Result<GroupMoe, String> {
    if g == 0 || d == 0 || f == 0 || d % g != 0 || f % g != 0 {
        return Err(format!(
            "the group {g} must divide d {d} and f {f}, none of them 0"
        ));
    }
    // a row goes to its k best experts: at least one, and no more than there are
    if k == 0 || k > e {
        return Err(format!("top {k} of {e} experts"));
    }
    if router.len() != e * d
        || w1.len() != e * f * d / 2
        || w2.len() != e * d * f / 2
        || b1.len() != e * f
        || b2.len() != e * d
    {
        return Err(format!("weights not shaped for E {e}, d {d}, f {f}"));
    }
    let (g1, g2) = (d / g, f / g);
    let mut experts = Vec::with_capacity(e);
    for ex in 0..e {
        let l1 = GroupWeights::in_place(
            &w1[ex * f * d / 2..(ex + 1) * f * d / 2],
            &s1[ex * f * g1..(ex + 1) * f * g1],
            &z1[ex * f * g1..(ex + 1) * f * g1],
            f,
            d,
            g,
        )
        .map_err(|m| format!("expert {ex} w1: {m}"))?;
        let l2 = GroupWeights::in_place(
            &w2[ex * d * f / 2..(ex + 1) * d * f / 2],
            &s2[ex * d * g2..(ex + 1) * d * g2],
            &z2[ex * d * g2..(ex + 1) * d * g2],
            d,
            f,
            g,
        )
        .map_err(|m| format!("expert {ex} w2: {m}"))?;
        experts.push((l1, l2));
    }
    Ok(GroupMoe {
        e,
        d,
        f,
        k,
        router,
        experts,
        b1,
        b2,
    })
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
    /// every frame's row quantized once: codes, scale and zero point, group sums
    xq: Vec<u8>,
    xs: Vec<(f32, i32)>,
    xa: Vec<f32>,
    out: Vec<f32>,
    expert: ExpertRunScratch,
}

/// Private working buffers for one worker running experts.
#[derive(Default)]
pub(crate) struct ExpertRunScratch {
    /// The expert's gathered inputs.
    gq: Vec<u8>,
    gxs: Vec<(f32, i32)>,
    ga: Vec<f32>,
    h: Vec<f32>,
    lin: GroupScratch,
}

impl GroupMoe {
    pub(crate) fn num_experts(&self) -> usize {
        self.e
    }
    pub(crate) fn d(&self) -> usize {
        self.d
    }
    pub(crate) fn f(&self) -> usize {
        self.f
    }
    pub(crate) fn g(&self) -> usize {
        self.experts[0].0.g()
    }
    #[allow(dead_code)]
    pub(crate) fn top_k(&self) -> usize {
        self.k
    }

    pub(crate) fn route_rows(&self, x: &[f32], rows: &mut Vec<Vec<(usize, f32)>>) {
        route(x, &self.router, self.num_experts(), self.d(), self.k, rows);
    }

    #[allow(dead_code)]
    pub(crate) fn route_rows_slice(&self, x: &[f32], out: &mut [(usize, f32)]) {
        crate::moe_op::route_rows_slice(x, &self.router, self.num_experts(), self.d(), self.k, out);
    }

    /// Unweighted outputs, in routed-list order. Weights and quantized inputs are shared;
    /// each caller supplies its own output and scratch. Accumulate in expert order for identical bits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert(
        &self,
        ex: usize,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        out: &mut Vec<f32>,
        scratch: &mut ExpertRunScratch,
    ) {
        let (l1, l2) = &self.experts[ex];
        let (d, m) = (self.d(), routed.len());
        out.resize(span(m, d), 0.0);
        if m == 0 {
            return;
        }
        self.expert_l1(ex, l1, routed, xq, xs, xa, scratch);
        let ExpertRunScratch { h, lin, .. } = scratch;
        l2.gelu_and_quantize_rows(h, &mut lin.q, &mut lin.xs, &mut lin.a);
        l2.codes_with(
            &lin.q,
            &lin.a,
            &lin.xs,
            m,
            out,
            Some(&self.b2[ex * d..(ex + 1) * d]),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_l1(
        &self,
        ex: usize,
        l1: &GroupWeights,
        routed: &[(usize, f32)],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        s: &mut ExpertRunScratch,
    ) {
        let (d, f, m) = (self.d(), self.f(), routed.len());
        let groups = d / self.g();
        s.gq.clear();
        s.gxs.clear();
        s.ga.clear();
        for &(r, _) in routed {
            s.gq.extend_from_slice(&xq[r * d..(r + 1) * d]);
            s.gxs.push(xs[r]);
            s.ga.extend_from_slice(&xa[r * groups..(r + 1) * groups]);
        }
        s.h.resize(span(m, f), 0.0);
        l1.codes_with(
            &s.gq,
            &s.ga,
            &s.gxs,
            m,
            &mut s.h,
            Some(&self.b1[ex * f..(ex + 1) * f]),
        );
    }

    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        // the layer's own code (routing, quantization, GELU, sums) compiled for the experts' kernel's level
        #[cfg(target_arch = "x86_64")]
        match self.experts.first().map(|(l1, _)| l1.path) {
            // SAFETY (each): the CPU has the features (`detect`)
            Some(Path::Vnni512) => return unsafe { self.forward_avx512(x, y, s) },
            Some(Path::Vnni256 | Path::Avx2) => return unsafe { self.forward_avx2(x, y, s) },
            Some(Path::Avx) => return unsafe { self.forward_avx1(x, y, s) },
            _ => {}
        }
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    unsafe fn forward_avx512(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        self.forward_body(x, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    unsafe fn forward_avx1(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        self.forward_body(x, y, s)
    }

    /// Forward using pre-quantized inputs `(xq, xs, xa)` shared with the layer's dense FFN.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_quantized(
        &self,
        x: &[f32],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        y: &mut [f32],
        s: &mut GroupMoeScratch,
    ) {
        #[cfg(target_arch = "x86_64")]
        match self.experts.first().map(|(l1, _)| l1.path) {
            Some(Path::Vnni512) => {
                return unsafe { self.forward_quantized_avx512(x, xq, xs, xa, y, s) }
            }
            Some(Path::Vnni256 | Path::Avx2) => {
                return unsafe { self.forward_quantized_avx2(x, xq, xs, xa, y, s) }
            }
            Some(Path::Avx) => return unsafe { self.forward_quantized_avx1(x, xq, xs, xa, y, s) },
            _ => {}
        }
        self.forward_quantized_body(x, xq, xs, xa, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_quantized_avx512(
        &self,
        x: &[f32],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        y: &mut [f32],
        s: &mut GroupMoeScratch,
    ) {
        self.forward_quantized_body(x, xq, xs, xa, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_quantized_avx2(
        &self,
        x: &[f32],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        y: &mut [f32],
        s: &mut GroupMoeScratch,
    ) {
        self.forward_quantized_body(x, xq, xs, xa, y, s)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_quantized_avx1(
        &self,
        x: &[f32],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        y: &mut [f32],
        s: &mut GroupMoeScratch,
    ) {
        self.forward_quantized_body(x, xq, xs, xa, y, s)
    }

    #[inline(always)]
    fn forward_quantized_body(
        &self,
        x: &[f32],
        xq: &[u8],
        xs: &[(f32, i32)],
        xa: &[f32],
        y: &mut [f32],
        s: &mut GroupMoeScratch,
    ) {
        y.fill(0.0);
        self.route_rows(x, &mut s.rows);
        for (ex, routed) in s.rows.iter().enumerate() {
            if !routed.is_empty() {
                self.forward_expert(ex, routed, xq, xs, xa, &mut s.out, &mut s.expert);
                add_routed(y, routed, &s.out, self.d);
            }
        }
    }

    #[inline(always)]
    fn forward_body(&self, x: &[f32], y: &mut [f32], s: &mut GroupMoeScratch) {
        y.fill(0.0);
        self.route_rows(x, &mut s.rows);
        self.experts[0]
            .0
            .quantize_rows(x, &mut s.xq, &mut s.xs, &mut s.xa);
        for (ex, routed) in s.rows.iter().enumerate() {
            if !routed.is_empty() {
                self.forward_expert(ex, routed, &s.xq, &s.xs, &s.xa, &mut s.out, &mut s.expert);
                add_routed(y, routed, &s.out, self.d);
            }
        }
    }
}

#[cfg(test)]
impl GroupMoe {
    /// Research timing (`native_encoder`'s `time_the_encoder`): us a forward of `x` in the routing, the experts'
    /// gathers and first Linears, their GELU, their second Linears and the weighted sums, over `reps` forwards.
    pub(crate) fn time_parts(&self, x: &[f32], reps: usize) -> [f64; 5] {
        #[cfg(target_arch = "x86_64")]
        if self
            .experts
            .first()
            .is_some_and(|(l1, _)| l1.path == Path::Vnni512)
        {
            // SAFETY: the CPU has the features (`detect`)
            return unsafe { self.time_parts_avx512(x, reps) };
        }
        self.time_parts_body(x, reps)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    unsafe fn time_parts_avx512(&self, x: &[f32], reps: usize) -> [f64; 5] {
        self.time_parts_body(x, reps)
    }

    #[inline(always)]
    fn time_parts_body(&self, x: &[f32], reps: usize) -> [f64; 5] {
        use std::time::Instant;
        let d = self.d;
        let mut s = GroupMoeScratch::default();
        let mut y = vec![0f32; x.len()];
        let mut ns = [0u128; 5];
        for _ in 0..reps {
            let t0 = Instant::now();
            y.fill(0.0);
            self.route_rows(x, &mut s.rows);
            self.experts[0]
                .0
                .quantize_rows(x, &mut s.xq, &mut s.xs, &mut s.xa);
            ns[0] += t0.elapsed().as_nanos();
            for (ex, (l1, l2)) in self.experts.iter().enumerate() {
                let routed = &s.rows[ex];
                if routed.is_empty() {
                    continue;
                }
                let m = routed.len();
                let t0 = Instant::now();
                self.expert_l1(ex, l1, routed, &s.xq, &s.xs, &s.xa, &mut s.expert);
                let t1 = Instant::now();
                let ExpertRunScratch { h, lin, .. } = &mut s.expert;
                l2.gelu_and_quantize_rows(h, &mut lin.q, &mut lin.xs, &mut lin.a);
                let t2 = Instant::now();
                s.out.resize(m * d, 0.0);
                l2.codes_with(
                    &lin.q,
                    &lin.a,
                    &lin.xs,
                    m,
                    &mut s.out,
                    Some(&self.b2[ex * d..(ex + 1) * d]),
                );
                let t3 = Instant::now();
                add_routed(&mut y, routed, &s.out, d);
                let t4 = Instant::now();
                ns[1] += (t1 - t0).as_nanos();
                ns[2] += (t2 - t1).as_nanos();
                ns[3] += (t3 - t2).as_nanos();
                ns[4] += (t4 - t3).as_nanos();
            }
        }
        ns.map(|v| v as f64 / reps as f64 / 1000.0)
    }
}

/// The kernel a `GroupWeights` runs (`detect`): AVX-512 VNNI, AVX-VNNI, AVX2, or plain Rust; the same bits on each.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Vnni512,
    Vnni256,
    Avx2,
    Avx,
    /// ARMv8.2 UDOT (`arm`).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Dot,
    Plain,
}

/// The codes in the layout the active execution path reads (never both unless tile mode 5 needs both).
enum Codes {
    /// Two 4-bit codes a byte (N * K / 2 bytes): default x86_64 (`TILE_MODE_DEFAULT = 6`) and `Path::Plain`.
    Packed(Vec<u8>),
    /// One byte per code, 64-byte blocks (N * K bytes): ARMv8.2 UDOT (`Path::Dot`) and x86_64 tile modes 0..=4.
    Unpacked(Vec<i8>),
    /// Both layouts: x86_64 tile mode 5 (packed for fewer than 6 rows, unpacked for 6 or more) and multi-mode tests.
    Both { packed: Vec<u8>, unpacked: Vec<i8> },
}

impl Codes {
    fn pack(w: &[u8], paired: bool) -> Vec<u8> {
        if !paired {
            return w.to_vec();
        }
        // Each pair is read by one ZMM load; don't split it across cache lines.
        let mut packed = Vec::<u8>::with_capacity(w.len() + 63);
        packed.resize(packed.as_ptr().align_offset(64), 0);
        for p in w.chunks_exact(64) {
            for i in 0..64 {
                let shift = (i / 32) * 4;
                let a = (p[i % 32] >> shift) & 0xf;
                let b = (p[32 + i % 32] >> shift) & 0xf;
                packed.push(a | (b << 4));
            }
        }
        packed
    }

    fn unpack(w: &[u8]) -> Vec<i8> {
        // each 32-byte block: the low nibbles are values 0-31, the high ones 32-63
        w.chunks(32)
            .flat_map(|b| {
                b.iter()
                    .map(|&v| (v & 0x0f) as i8)
                    .chain(b.iter().map(|&v| (v >> 4) as i8))
            })
            .collect()
    }

    #[inline(always)]
    fn packed(&self) -> &[u8] {
        match self {
            Codes::Packed(p) | Codes::Both { packed: p, .. } => p,
            Codes::Unpacked(_) => panic!("packed codes requested on an unpacked-only weight"),
        }
    }

    #[inline(always)]
    fn unpacked(&self) -> &[i8] {
        match self {
            Codes::Unpacked(u) | Codes::Both { unpacked: u, .. } => u,
            Codes::Packed(_) => panic!("unpacked codes requested on a packed-only weight"),
        }
    }
}

/// The codes and the per-group terms of the output.
pub(crate) struct GroupWeights {
    n: usize,
    k: usize,
    g: usize,
    codes: Codes,
    /// Packed nibbles hold two consecutive steps at each lane, not two halves of one step.
    paired: bool,
    /// (K / g, N): s_g, and -s_g z_g
    scale: Vec<f32>,
    scale_zero: Vec<f32>,
    /// (N): sum_g s_g (C_g - g z_g)
    t: Vec<f32>,
    path: Path,
}

#[derive(Default)]
pub(crate) struct GroupScratch {
    q: Vec<u8>,
    xs: Vec<(f32, i32)>,
    a: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Cols {
    y: *mut f32,
    stride: usize,
    start: usize,
    end: usize,
}

impl GroupWeights {
    pub(crate) fn n(&self) -> usize {
        self.n
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
    pub(crate) fn g(&self) -> usize {
        self.g
    }
    pub(crate) fn blocks(&self) -> usize {
        self.n / 16
    }

    /// The weights of the codes `w` ((K / g) groups of (N, g) blocked codes, g / 2 bytes a row), copied into the
    /// layout the active path reads; `scales` and `zeros` (N, K / g).
    pub(crate) fn in_place(
        w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        k: usize,
        g: usize,
    ) -> Result<Self, String> {
        Self::in_place_for(w, scales, zeros, n, k, g, detect(), tile_mode())
    }

    fn in_place_for(
        w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        k: usize,
        g: usize,
        path: Path,
        mode: u8,
    ) -> Result<Self, String> {
        if n == 0 || k == 0 || n % 16 != 0 || g % 4 != 0 || g == 0 || k % g != 0 {
            return Err(format!(
                "N {n} must be in 16s, the group {g} in 4s and divide K {k}, neither N nor K 0"
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
        let paired = path == Path::Vnni512 && g % 8 == 0 && mode >= 5;
        let codes = match path {
            Path::Plain => Codes::Packed(w.to_vec()),
            Path::Dot => Codes::Unpacked(Codes::unpack(w)),
            Path::Vnni512 | Path::Vnni256 | Path::Avx2 | Path::Avx => match mode {
                6 => Codes::Packed(Codes::pack(w, paired)),
                5 => Codes::Both {
                    packed: Codes::pack(w, paired),
                    unpacked: Codes::unpack(w),
                },
                _ => Codes::Unpacked(Codes::unpack(w)),
            },
        };
        let mut weights = GroupWeights {
            n,
            k,
            g,
            codes,
            paired,
            scale: vec![0f32; groups * n],
            scale_zero: vec![0f32; groups * n],
            t: vec![0f32; n],
            path,
        };
        for gi in 0..groups {
            for o in 0..n {
                let (s, z) = (scales[o * groups + gi], zeros[o * groups + gi]);
                let c: u32 = (0..g)
                    .map(|i| Self::packed_code(w, n, g, o, gi * g + i) as u32)
                    .sum();
                weights.scale[gi * n + o] = s;
                weights.scale_zero[gi * n + o] = -(s * z);
                weights.t[o] += s * (c as f32 - g as f32 * z);
            }
        }
        Ok(weights)
    }

    #[inline(always)]
    fn packed(&self) -> &[u8] {
        let p = self.codes.packed();
        if self.paired {
            &p[p.as_ptr().align_offset(64)..]
        } else {
            p
        }
    }

    /// The code of output `o`, input `i` from whichever layout `self.codes` holds.
    fn code(&self, o: usize, i: usize) -> u8 {
        match &self.codes {
            Codes::Packed(_) | Codes::Both { .. } => {
                let p = self.packed();
                if self.paired {
                    let (g, n) = (self.g, self.n);
                    let (gi, ii) = (i / g, i % g);
                    let pair = (o / 16) * (g / 8) + ii / 8;
                    let v = (o % 16) * 4 + ii % 4;
                    (p[gi * (n * g / 2) + pair * 64 + v] >> ((ii % 8) / 4 * 4)) & 0xf
                } else {
                    Self::packed_code(p, self.n, self.g, o, i)
                }
            }
            Codes::Unpacked(u) => {
                let (g, n) = (self.g, self.n);
                let (gi, ii) = (i / g, i % g);
                let block = (o / 16) * (g / 4) + ii / 4;
                let v = (o % 16) * 4 + ii % 4;
                u[(gi * (n * g / 2) + block * 32) * 2 + v] as u8
            }
        }
    }

    /// The code as the packed bytes hold it (32-byte blocks: low nibbles 0-31, high nibbles 32-63).
    fn packed_code(w: &[u8], n: usize, g: usize, o: usize, i: usize) -> u8 {
        let (gi, ii) = (i / g, i % g);
        let block = (o / 16) * (g / 4) + ii / 4;
        let v = (o % 16) * 4 + ii % 4;
        let byte = w[gi * (n * g / 2) + block * 32 + v % 32];
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
    #[cfg(test)]
    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32], s: &mut GroupScratch) {
        self.forward_bias(x, y, None, s)
    }

    /// `forward` with `bias` (N) added to each row. The VNNI kernel writes the first group's sums, and scales the
    /// last group's, in its registers (2026-11-19: the zero fill, the last scaling and the bias were three more passes
    /// over y, about a fifth of the 30-frame FFN's time), the plain path in passes of its own: the same bits.
    pub(crate) fn forward_bias(
        &self,
        x: &[f32],
        y: &mut [f32],
        bias: Option<&[f32]>,
        s: &mut GroupScratch,
    ) {
        if let Some(b) = bias {
            assert_eq!(
                b.len(),
                self.n,
                "a bias of {} for {} outputs",
                b.len(),
                self.n
            );
        }
        match self.path {
            // SAFETY (each): the CPU has the path's features (`detect`)
            #[cfg(target_arch = "x86_64")]
            Path::Vnni512 => unsafe { x86::forward(self, x, y, bias, s) },
            #[cfg(target_arch = "x86_64")]
            Path::Vnni256 | Path::Avx2 => unsafe { x86::forward_avx2(self, x, y, bias, s) },
            #[cfg(target_arch = "x86_64")]
            Path::Avx => unsafe { x86::forward_avx(self, x, y, bias, s) },
            _ => self.forward_with(x, y, bias, s),
        }
    }

    #[inline(always)]
    fn forward_with(&self, x: &[f32], y: &mut [f32], bias: Option<&[f32]>, s: &mut GroupScratch) {
        let rows = x.len() / self.k;
        self.quantize_rows(x, &mut s.q, &mut s.xs, &mut s.a);
        self.codes_with(&s.q, &s.a, &s.xs, rows, y, bias)
    }

    /// Each row of `x` (rows of K) quantized to u8 (`quantize_row`: codes `q`, scale and zero point `xs`) and its
    /// codes' sums over each group of g (`a`, rows x K / g): what the kernel reads.
    #[inline(always)]
    pub(crate) fn quantize_rows(
        &self,
        x: &[f32],
        q: &mut Vec<u8>,
        xs: &mut Vec<(f32, i32)>,
        a: &mut Vec<f32>,
    ) {
        self.resize_quantized(x.len(), q, xs, a);
        self.quantize_rows_into(x, q, xs, a);
    }

    fn resize_quantized(
        &self,
        len: usize,
        q: &mut Vec<u8>,
        xs: &mut Vec<(f32, i32)>,
        a: &mut Vec<f32>,
    ) {
        let k = self.k();
        let rows = len / k;
        q.resize(rows * k, 0);
        xs.resize(rows, (0.0, 0));
        a.resize(rows * (k / self.g()), 0.0);
    }

    /// Quantize into caller-owned row slices. No allocation or shared mutable scratch.
    #[inline(always)]
    pub(crate) fn quantize_rows_into(
        &self,
        x: &[f32],
        q: &mut [u8],
        xs: &mut [(f32, i32)],
        a: &mut [f32],
    ) {
        let (k, g) = (self.k, self.g);
        let rows = x.len() / k;
        let (q, xs, a) = (
            &mut q[..rows * k],
            &mut xs[..rows],
            &mut a[..rows * (k / g)],
        );
        #[cfg(target_arch = "x86_64")]
        if self.path == Path::Vnni512 {
            return unsafe { x86::quantize_rows_avx512(x, q, xs, a, k, g) };
        }
        for (r, xr) in x.chunks_exact(k).enumerate() {
            let groups = k / g;
            xs[r] = Self::quantize_one(xr, &mut q[r * k..][..k], &mut a[r * groups..][..groups], g);
        }
    }

    #[inline(always)]
    fn quantize_one(x: &[f32], q: &mut [u8], a: &mut [f32], g: usize) -> (f32, i32) {
        let params = quantize_row_scalar(x, q);
        for (sum, group) in a.iter_mut().zip(q.chunks_exact(g)) {
            *sum = group.iter().map(|&v| v as u32).sum::<u32>() as f32;
        }
        params
    }

    /// Apply GELU in-place to `h` (rows, K), quantize, and run `codes_with` into `y` (rows, N).
    pub(crate) fn forward_after_gelu_bias(
        &self,
        h: &mut [f32],
        y: &mut [f32],
        bias: Option<&[f32]>,
        s: &mut GroupScratch,
    ) {
        let rows = h.len() / self.k;
        self.gelu_and_quantize_rows(h, &mut s.q, &mut s.xs, &mut s.a);
        self.codes_with(&s.q, &s.a, &s.xs, rows, y, bias);
    }

    /// Run GELU and immediately quantize that row while its values are still hot.
    pub(crate) fn gelu_and_quantize_rows(
        &self,
        h: &mut [f32],
        q: &mut Vec<u8>,
        xs: &mut Vec<(f32, i32)>,
        a: &mut Vec<f32>,
    ) {
        self.resize_quantized(h.len(), q, xs, a);
        let (k, g) = (self.k, self.g);
        #[cfg(target_arch = "x86_64")]
        if self.path == Path::Vnni512 {
            return unsafe { x86::gelu_and_quantize_rows_avx512(h, q, xs, a, k, g) };
        }
        for (r, hr) in h.chunks_exact_mut(k).enumerate() {
            for v in hr.iter_mut() {
                *v = gelu(*v);
            }
            let groups = k / g;
            xs[r] = Self::quantize_one(hr, &mut q[r * k..][..k], &mut a[r * groups..][..groups], g);
        }
    }

    /// `forward_bias` of `rows` rows quantized by `quantize_rows`.
    #[inline(always)]
    fn codes_with(
        &self,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        rows: usize,
        y: &mut [f32],
        bias: Option<&[f32]>,
    ) {
        let n = self.n();
        assert!(y.len() >= span(rows, n));
        // SAFETY: this slice exclusively owns the full output rectangle.
        unsafe {
            self.codes_with_cols_ptr(q, a, xs, rows, y.as_mut_ptr(), n, bias, 0, self.blocks())
        };
    }

    /// Write the given 16-column blocks, at their original column offsets in each output row.
    /// # Safety
    /// `y + r * y_stride + ob_start * 16` must be valid for writes of all the requested columns
    /// for each row. These cells must not alias inputs, references, or cells touched by other
    /// workers; disjoint column ranges can share the same raw base pointer.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) unsafe fn codes_with_cols_ptr(
        &self,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        rows: usize,
        y: *mut f32,
        y_stride: usize,
        bias: Option<&[f32]>,
        ob_start: usize,
        ob_end: usize,
    ) {
        let n = self.n;
        assert!(ob_start <= ob_end && ob_end <= self.blocks() && y_stride >= ob_end * 16);
        assert!(bias.is_none_or(|b| b.len() == n));
        if rows == 0 || ob_start == ob_end {
            return;
        }
        span(rows, y_stride);
        let cols = Cols {
            y,
            stride: y_stride,
            start: ob_start,
            end: ob_end,
        };
        // checked in release too: the kernels read q, a and xs by raw pointers
        assert!(
            q.len() >= span(rows, self.k)
                && a.len() >= span(rows, self.k / self.g)
                && xs.len() >= rows,
            "{} codes, {} group sums and {} scales for {rows} rows of {}",
            q.len(),
            a.len(),
            xs.len(),
            self.k
        );
        // SAFETY (each): the CPU has the path's features (`detect`).
        #[cfg(target_arch = "x86_64")]
        match self.path {
            Path::Vnni512 => return x86::inner(self, q, a, xs, bias, rows, cols),
            Path::Vnni256 => return x86::inner_vnni256(self, q, a, xs, bias, rows, cols),
            Path::Avx2 => return x86::inner_avx2(self, q, a, xs, bias, rows, cols),
            Path::Avx => return x86::inner_avx(self, q, a, xs, bias, rows, cols),
            Path::Dot | Path::Plain => {}
        }
        #[cfg(target_arch = "aarch64")]
        if self.path == Path::Dot {
            arm::inner(self, q, a, xs, bias, rows, cols);
            return;
        }
        self.inner_plain(q, a, &xs[..rows], bias, cols);
    }

    /// Accumulate and scale in plain Rust, in the same order as the VNNI kernel.
    unsafe fn inner_plain(
        &self,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        bias: Option<&[f32]>,
        cols: Cols,
    ) {
        let (n, k, g) = (self.n, self.k, self.g);
        let groups = k / g;
        for (r, &(sx, zx)) in xs.iter().enumerate() {
            for o in cols.start * 16..cols.end * 16 {
                let mut f = 0f32;
                for gi in 0..groups {
                    let acc: i32 = (0..g)
                        .map(|i| q[r * k + gi * g + i] as i32 * self.code(o, gi * g + i) as i32)
                        .sum();
                    f = acc as f32 * self.scale[gi * n + o] + f;
                    f = self.scale_zero[gi * n + o] * a[r * groups + gi] + f;
                }
                f = (f - zx as f32 * self.t[o]) * sx;
                if let Some(b) = bias {
                    f += b[o];
                }
                cols.y.add(r * cols.stride + o).write(f);
            }
        }
    }

    #[cfg(test)]
    fn with_path(mut self, path: Path) -> Self {
        self.path = path;
        self
    }

    #[cfg(test)]
    fn set_both_codes(&mut self, w: &[u8]) {
        self.paired = self.path == Path::Vnni512 && self.g % 8 == 0;
        self.codes = Codes::Both {
            packed: Codes::pack(w, self.paired),
            unpacked: Codes::unpack(w),
        };
    }

    #[cfg(test)]
    fn with_both_codes(mut self, w: &[u8]) -> Self {
        self.set_both_codes(w);
        self
    }
}

/// The VNNI kernel's tiles (`x86::inner`'s modes 0-6; `TILE_MODE_DEFAULT` unless SETSUNA_TILE_MODE names another, a
/// research comparison; the tests set it). The same bits whichever; the timing (`time_the_encoder`) compares them.
pub(crate) static TILE_MODE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(u8::MAX);

/// 6: keep only packed codes. Use row-dependent tiles with AVX-512 paired codes, or the
/// original register tiles when the group has an odd number of steps.
pub(crate) const TILE_MODE_DEFAULT: u8 = 6;

/// The tiles' mode (`TILE_MODE`, set from SETSUNA_TILE_MODE at the first use).
pub(crate) fn tile_mode() -> u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let m = TILE_MODE.load(Relaxed);
    if m != u8::MAX {
        return m;
    }
    let m = std::env::var("SETSUNA_TILE_MODE")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&m: &u8| m <= 6)
        .unwrap_or(TILE_MODE_DEFAULT);
    TILE_MODE.store(m, Relaxed);
    m
}

/// The kernel for the engine's level (`crate::cpu`; SETSUNA_CPU_LEVEL caps it).
fn detect() -> Path {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::cpu::Level;
        match crate::cpu::level() {
            Level::Avx512 => return Path::Vnni512,
            Level::AvxVnni => return Path::Vnni256,
            Level::Avx2 => return Path::Avx2,
            Level::Avx => return Path::Avx,
            Level::Plain => {}
        }
    }
    if crate::cpu::dotprod() {
        return Path::Dot;
    }
    Path::Plain
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{Cols, GroupWeights};
    use std::arch::aarch64::*;

    /// UDOT: each lane of `acc` plus the dot product of its four u8 of `w` and of `x` (the intrinsic is not stable).
    #[inline(always)]
    unsafe fn udot(acc: uint32x4_t, w: uint8x16_t, x: uint8x16_t) -> uint32x4_t {
        let mut r = acc;
        std::arch::asm!(
            "udot {r:v}.4s, {w:v}.16b, {x:v}.16b",
            r = inout(vreg) r,
            w = in(vreg) w,
            x = in(vreg) x,
            options(pure, nomem, nostack, preserves_flags)
        );
        r
    }

    /// RT rows x one output block, as the x86 tiles: every group's sums in registers (16 of the 32 with 4 rows), then
    /// per output the running sum plus s_g acc_g, plus -s_g z_g A_g: a multiply and an add each, as the plain path
    /// orders them (no fused multiply-add), so the bits are its.
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn tile<const RT: usize>(
        w: &GroupWeights,
        q: *const u8,
        a: *const f32,
        ob: usize,
        y: *mut f32,
        stride: usize,
    ) {
        let (n, k, g) = (w.n, w.k, w.g);
        let (groups, steps, gstride) = (k / g, g / 4, n * g);
        let codes = w.codes.unpacked().as_ptr() as *const u8;
        let (sc, sz) = (w.scale.as_ptr(), w.scale_zero.as_ptr());
        for gi in 0..groups {
            let mut acc = [[vdupq_n_u32(0); 4]; RT];
            let wg = codes.add(gi * gstride + ob * steps * 64);
            for s in 0..steps {
                let p = wg.add(s * 64);
                let wv = [
                    vld1q_u8(p),
                    vld1q_u8(p.add(16)),
                    vld1q_u8(p.add(32)),
                    vld1q_u8(p.add(48)),
                ];
                for (r, ar) in acc.iter_mut().enumerate() {
                    let at = q.add(r * k + gi * g + s * 4) as *const u32;
                    let x = vreinterpretq_u8_u32(vdupq_n_u32(at.read_unaligned()));
                    for (av, wc) in ar.iter_mut().zip(&wv) {
                        *av = udot(*av, *wc, x);
                    }
                }
            }
            for v in 0..4 {
                let s1 = vld1q_f32(sc.add(gi * n + ob * 16 + v * 4));
                let s2 = vld1q_f32(sz.add(gi * n + ob * 16 + v * 4));
                for (r, ar) in acc.iter().enumerate() {
                    let av = vdupq_n_f32(*a.add(r * groups + gi));
                    let yp = y.add(r * stride + ob * 16 + v * 4);
                    let prev = if gi == 0 {
                        vdupq_n_f32(0.0)
                    } else {
                        vld1q_f32(yp)
                    };
                    let f = vaddq_f32(vmulq_f32(vcvtq_f32_u32(ar[v]), s1), prev);
                    vst1q_f32(yp, vaddq_f32(vmulq_f32(s2, av), f));
                }
            }
        }
    }

    /// `GroupWeights::inner_plain` with UDOT: tiles of up to 4 rows, block by block.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn inner(
        w: &GroupWeights,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        bias: Option<&[f32]>,
        rows: usize,
        cols: Cols,
    ) {
        let (k, groups, stride) = (w.k, w.k / w.g, cols.stride);
        for ob in cols.start..cols.end {
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(4);
                let (qp, ap, yp) = (
                    q.as_ptr().add(r0 * k),
                    a.as_ptr().add(r0 * groups),
                    cols.y.add(r0 * stride),
                );
                match rt {
                    1 => tile::<1>(w, qp, ap, ob, yp, stride),
                    2 => tile::<2>(w, qp, ap, ob, yp, stride),
                    3 => tile::<3>(w, qp, ap, ob, yp, stride),
                    _ => tile::<4>(w, qp, ap, ob, yp, stride),
                }
                r0 += rt;
            }
        }
        let range = cols.start * 16..cols.end * 16;
        for (r, &(sx, zx)) in xs[..rows].iter().enumerate() {
            let row =
                std::slice::from_raw_parts_mut(cols.y.add(r * stride + range.start), range.len());
            for (v, &t) in row.iter_mut().zip(&w.t[range.clone()]) {
                *v = (*v - zx as f32 * t) * sx;
            }
            if let Some(b) = bias {
                for (v, &b) in row.iter_mut().zip(&b[range.clone()]) {
                    *v += b;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Cols, GroupWeights};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn quantize_rows_avx512(
        x: &[f32],
        q: &mut [u8],
        xs: &mut [(f32, i32)],
        a: &mut [f32],
        k: usize,
        g: usize,
    ) {
        for (((x, q), xs), sums) in x
            .chunks_exact(k)
            .zip(q.chunks_exact_mut(k))
            .zip(xs)
            .zip(a.chunks_exact_mut(k / g))
        {
            let params = crate::moe_op::x86::quantize_range(x);
            *xs = quantize_one(x, q, sums, g, params);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn gelu_and_quantize_rows_avx512(
        h: &mut [f32],
        q: &mut [u8],
        xs: &mut [(f32, i32)],
        a: &mut [f32],
        k: usize,
        g: usize,
    ) {
        for (((h, q), xs), sums) in h
            .chunks_exact_mut(k)
            .zip(q.chunks_exact_mut(k))
            .zip(xs)
            .zip(a.chunks_exact_mut(k / g))
        {
            let params = crate::moe_op::x86::gelu_quantize_range(h);
            *xs = quantize_one(h, q, sums, g, params);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn quantize_one(
        x: &[f32],
        q: &mut [u8],
        a: &mut [f32],
        g: usize,
        (scale, zp): (f32, f32),
    ) -> (f32, i32) {
        use crate::moe_op::x86::quantize_group;
        let inv = 1.0 / scale;
        for ((x, q), sum) in x.chunks_exact(g).zip(q.chunks_exact_mut(g)).zip(a) {
            *sum = quantize_group(x, q, inv, zp);
        }
        (scale, zp as i32)
    }

    /// A 256-bit tile: RT rows (up to 6) x one output block of 16 (its outputs 0-7 and 8-15 in two registers), every
    /// group's integer sums in registers, scaled into the running sums in y at the group's end as `tile` does (the
    /// first group's written; at the last, each row's last scaling and the bias): the same operations in the same
    /// order, so the same bits. `$dot` adds a register of 4 u8 activations (broadcast) times 8 outputs' 4 codes:
    /// AVX-VNNI's dot product, or AVX2's u8 x i8 pair products (exact: codes 0-15 times 255, two of them, fit 16 bits)
    /// summed by pairs into 32 bits.
    macro_rules! tile256_fn {
        ($name:ident, $features:literal, $dot:ident) => {
            #[target_feature(enable = $features)]
            unsafe fn $name<const RT: usize, const NIB: bool>(
                w: &GroupWeights,
                q: *const u8,
                a: *const f32,
                xs: *const (f32, i32),
                b: *const f32,
                ob: usize,
                y: *mut f32,
                stride: usize,
            ) {
                let (n, k, g) = (w.n, w.k, w.g);
                let (groups, steps, gstride) = (k / g, g / 4, n * g);
                let (sc, sz, tt) = (w.scale.as_ptr(), w.scale_zero.as_ptr(), w.t.as_ptr());
                let ones = _mm256_set1_epi16(1);
                let low = _mm256_set1_epi8(0x0f);
                for gi in 0..groups {
                    let mut acc = [[_mm256_setzero_si256(); 2]; RT];
                    let off = gi * gstride + ob * steps * 64;
                    // NIB: the packed codes, 32 bytes for the block's 64 (the low nibbles its codes 0-31, the high
                    // ones 32-63: the two registers below), half the bytes from memory
                    let (wg, pg) = if NIB {
                        (std::ptr::null(), w.codes.packed().as_ptr().add(off / 2))
                    } else {
                        (w.codes.unpacked().as_ptr().add(off), std::ptr::null())
                    };
                    for s in 0..steps {
                        let (w0, w1) = if NIB {
                            let v = _mm256_loadu_si256(pg.add(s * 32) as *const __m256i);
                            (
                                _mm256_and_si256(v, low),
                                _mm256_and_si256(_mm256_srli_epi16::<4>(v), low),
                            )
                        } else {
                            (
                                _mm256_loadu_si256(wg.add(s * 64) as *const __m256i),
                                _mm256_loadu_si256(wg.add(s * 64 + 32) as *const __m256i),
                            )
                        };
                        for (r, ar) in acc.iter_mut().enumerate() {
                            let x = _mm256_set1_epi32(bcast4(q.add(r * k + gi * g + s * 4)));
                            ar[0] = $dot!(ar[0], x, w0, ones);
                            ar[1] = $dot!(ar[1], x, w1, ones);
                        }
                    }
                    let (first, last) = (gi == 0, gi + 1 == groups);
                    for h in 0..2 {
                        let o = ob * 16 + h * 8;
                        let s1 = _mm256_loadu_ps(sc.add(gi * n + o));
                        let s2 = _mm256_loadu_ps(sz.add(gi * n + o));
                        for (r, ar) in acc.iter().enumerate() {
                            let av = _mm256_set1_ps(*a.add(r * groups + gi));
                            let yp = y.add(r * stride + o);
                            let prev = if first {
                                _mm256_setzero_ps()
                            } else {
                                _mm256_loadu_ps(yp)
                            };
                            let f =
                                _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(ar[h]), s1), prev);
                            let mut f = _mm256_add_ps(_mm256_mul_ps(s2, av), f);
                            if last {
                                let (sx, zx) = *xs.add(r);
                                let zt = _mm256_mul_ps(
                                    _mm256_set1_ps(zx as f32),
                                    _mm256_loadu_ps(tt.add(o)),
                                );
                                f = _mm256_mul_ps(_mm256_sub_ps(f, zt), _mm256_set1_ps(sx));
                                if !b.is_null() {
                                    f = _mm256_add_ps(f, _mm256_loadu_ps(b.add(o)));
                                }
                            }
                            _mm256_storeu_ps(yp, f);
                        }
                    }
                }
            }
        };
    }

    macro_rules! dot_avx2 {
        ($acc:expr, $x:expr, $w:expr, $ones:expr) => {
            _mm256_add_epi32($acc, _mm256_madd_epi16(_mm256_maddubs_epi16($x, $w), $ones))
        };
    }

    macro_rules! dot_vnni {
        ($acc:expr, $x:expr, $w:expr, $ones:expr) => {{
            let _ = $ones;
            _mm256_dpbusd_avx_epi32($acc, $x, $w)
        }};
    }

    tile256_fn!(tile256_avx2, "avx2,fma", dot_avx2);
    tile256_fn!(tile256_vnni, "avx2,fma,avxvnni", dot_vnni);

    macro_rules! by_rows256 {
        ($tile:ident, $rt:expr, $nib:literal, $($arg:expr),*) => {
            match $rt {
                1 => $tile::<1, $nib>($($arg),*),
                2 => $tile::<2, $nib>($($arg),*),
                3 => $tile::<3, $nib>($($arg),*),
                4 => $tile::<4, $nib>($($arg),*),
                5 => $tile::<5, $nib>($($arg),*),
                _ => $tile::<6, $nib>($($arg),*),
            }
        };
    }

    /// Resolve `tile_mode()` against what `w.codes` stores: packed-only weights always use mode 6
    /// (avoiding even the atomic load on the default hot path), unpacked-only stay within modes 0..=4,
    /// and `Both` follows `tile_mode()` freely.
    #[inline(always)]
    fn effective_mode(w: &GroupWeights) -> u8 {
        match &w.codes {
            super::Codes::Packed(_) => 6,
            super::Codes::Unpacked(_) => super::tile_mode().min(4),
            super::Codes::Both { .. } => super::tile_mode(),
        }
    }

    /// `inner` with 256-bit tiles: up to 6 rows by one block with AVX-VNNI (12 sums, 2 codes, a broadcast; 5 with
    /// the codes read packed, their mask and unpacking a register more), 4 with AVX2 (its pair products a register
    /// more each). The codes read packed as `super::tile_mode` says (5: for fewer than 6 rows; 6, the default: all).
    macro_rules! inner256_fn {
        ($name:ident, $features:literal, $tile:ident, $rows:expr, $rows_nib:expr) => {
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name(
                w: &GroupWeights,
                q: &[u8],
                a: &[f32],
                xs: &[(f32, i32)],
                bias: Option<&[f32]>,
                rows: usize,
                cols: Cols,
            ) {
                let (k, groups, stride) = (w.k, w.k / w.g, cols.stride);
                let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
                let mode = effective_mode(w);
                let nib = mode == 6 || (mode == 5 && rows < 6);
                for ob in cols.start..cols.end {
                    let mut r0 = 0;
                    while r0 < rows {
                        let rt = (rows - r0).min(if nib { $rows_nib } else { $rows });
                        let (qp, ap, xp, yp) = (
                            q.as_ptr().add(r0 * k),
                            a.as_ptr().add(r0 * groups),
                            xs.as_ptr().add(r0),
                            cols.y.add(r0 * stride),
                        );
                        if nib {
                            by_rows256!($tile, rt, true, w, qp, ap, xp, b, ob, yp, stride);
                        } else {
                            by_rows256!($tile, rt, false, w, qp, ap, xp, b, ob, yp, stride);
                        }
                        r0 += rt;
                    }
                }
            }
        };
    }

    inner256_fn!(inner_avx2, "avx2,fma", tile256_avx2, 4, 4);
    inner256_fn!(inner_vnni256, "avx2,fma,avxvnni", tile256_vnni, 6, 5);

    /// The AVX level's tile (Sandy and Ivy Bridge: 128-bit integers, no FMA): RT rows (up to 2) x one block, its 16
    /// outputs in four registers of 4, each group's sums by SSSE3's u8 x i8 pair products (exact for codes 0-15)
    /// summed by pairs into 32 bits, scaled into y as `tile` does: the same bits. NIB: the codes read packed (the
    /// low nibbles of the block's first 16 bytes its outputs 0-3, of the next 16 its outputs 4-7, the high ones 8-11
    /// and 12-15).
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    unsafe fn tile128<const RT: usize, const NIB: bool>(
        w: &GroupWeights,
        q: *const u8,
        a: *const f32,
        xs: *const (f32, i32),
        b: *const f32,
        ob: usize,
        y: *mut f32,
        stride: usize,
    ) {
        let (n, k, g) = (w.n, w.k, w.g);
        let (groups, steps, gstride) = (k / g, g / 4, n * g);
        let (sc, sz, tt) = (w.scale.as_ptr(), w.scale_zero.as_ptr(), w.t.as_ptr());
        let ones = _mm_set1_epi16(1);
        let low = _mm_set1_epi8(0x0f);
        for gi in 0..groups {
            let mut acc = [[_mm_setzero_si128(); 4]; RT];
            let off = gi * gstride + ob * steps * 64;
            let (wg, pg) = if NIB {
                (std::ptr::null(), w.codes.packed().as_ptr().add(off / 2))
            } else {
                (w.codes.unpacked().as_ptr().add(off), std::ptr::null())
            };
            for s in 0..steps {
                let wv = if NIB {
                    let p0 = _mm_loadu_si128(pg.add(s * 32) as *const __m128i);
                    let p1 = _mm_loadu_si128(pg.add(s * 32 + 16) as *const __m128i);
                    [
                        _mm_and_si128(p0, low),
                        _mm_and_si128(p1, low),
                        _mm_and_si128(_mm_srli_epi16::<4>(p0), low),
                        _mm_and_si128(_mm_srli_epi16::<4>(p1), low),
                    ]
                } else {
                    let p = wg.add(s * 64);
                    [
                        _mm_loadu_si128(p as *const __m128i),
                        _mm_loadu_si128(p.add(16) as *const __m128i),
                        _mm_loadu_si128(p.add(32) as *const __m128i),
                        _mm_loadu_si128(p.add(48) as *const __m128i),
                    ]
                };
                for (r, ar) in acc.iter_mut().enumerate() {
                    let x = _mm_set1_epi32(bcast4(q.add(r * k + gi * g + s * 4)));
                    for (ac, wc) in ar.iter_mut().zip(&wv) {
                        *ac = _mm_add_epi32(*ac, _mm_madd_epi16(_mm_maddubs_epi16(x, *wc), ones));
                    }
                }
            }
            let (first, last) = (gi == 0, gi + 1 == groups);
            for c in 0..4 {
                let o = ob * 16 + c * 4;
                let s1 = _mm_loadu_ps(sc.add(gi * n + o));
                let s2 = _mm_loadu_ps(sz.add(gi * n + o));
                for (r, ar) in acc.iter().enumerate() {
                    let av = _mm_set1_ps(*a.add(r * groups + gi));
                    let yp = y.add(r * stride + o);
                    let prev = if first {
                        _mm_setzero_ps()
                    } else {
                        _mm_loadu_ps(yp)
                    };
                    let f = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(ar[c]), s1), prev);
                    let mut f = _mm_add_ps(_mm_mul_ps(s2, av), f);
                    if last {
                        let (sx, zx) = *xs.add(r);
                        let zt = _mm_mul_ps(_mm_set1_ps(zx as f32), _mm_loadu_ps(tt.add(o)));
                        f = _mm_mul_ps(_mm_sub_ps(f, zt), _mm_set1_ps(sx));
                        if !b.is_null() {
                            f = _mm_add_ps(f, _mm_loadu_ps(b.add(o)));
                        }
                    }
                    _mm_storeu_ps(yp, f);
                }
            }
        }
    }

    /// `inner` at the AVX level: 2 rows a tile, the codes read packed as `super::tile_mode` says.
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    pub(super) unsafe fn inner_avx(
        w: &GroupWeights,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        bias: Option<&[f32]>,
        rows: usize,
        cols: Cols,
    ) {
        let (k, groups, stride) = (w.k, w.k / w.g, cols.stride);
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        let mode = effective_mode(w);
        let nib = mode == 6 || (mode == 5 && rows < 6);
        for ob in cols.start..cols.end {
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(2);
                let (qp, ap, xp, yp) = (
                    q.as_ptr().add(r0 * k),
                    a.as_ptr().add(r0 * groups),
                    xs.as_ptr().add(r0),
                    cols.y.add(r0 * stride),
                );
                match (rt, nib) {
                    (1, false) => tile128::<1, false>(w, qp, ap, xp, b, ob, yp, stride),
                    (_, false) => tile128::<2, false>(w, qp, ap, xp, b, ob, yp, stride),
                    (1, true) => tile128::<1, true>(w, qp, ap, xp, b, ob, yp, stride),
                    (_, true) => tile128::<2, true>(w, qp, ap, xp, b, ob, yp, stride),
                }
                r0 += rt;
            }
        }
    }

    /// `GroupWeights::forward_with` compiled for AVX (no FMA: the same bits).
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    pub(super) unsafe fn forward_avx(
        w: &GroupWeights,
        x: &[f32],
        y: &mut [f32],
        bias: Option<&[f32]>,
        s: &mut super::GroupScratch,
    ) {
        w.forward_with(x, y, bias, s)
    }

    /// `GroupWeights::forward_with` compiled for AVX2 (its quantization and group sums vectorized 8 wide).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn forward_avx2(
        w: &GroupWeights,
        x: &[f32],
        y: &mut [f32],
        bias: Option<&[f32]>,
        s: &mut super::GroupScratch,
    ) {
        w.forward_with(x, y, bias, s)
    }

    #[inline(always)]
    unsafe fn bcast4(q: *const u8) -> i32 {
        (q as *const i32).read_unaligned()
    }

    /// RT rows x C output blocks (16 outputs each) from block `ob`: every group's integer sums in registers (24 of
    /// the 32 with 6 rows x 4 blocks, each weight vector serving the tile's rows and each broadcast its blocks), turned
    /// to float and scaled into the running sums in `y` at the group's end (the first group's written, not added to a
    /// zero fill), and at the last group each row's last scaling, (y - zx t) sx, and the bias (`b`, or null) added:
    /// `forward_with`'s plain passes' operations in their order, in registers.
    #[target_feature(enable = "avx512f,avx512vnni,avx2,fma")]
    unsafe fn tile<const RT: usize, const C: usize>(
        w: &GroupWeights,
        q: *const u8,
        a: *const f32,
        xs: *const (f32, i32),
        b: *const f32,
        ob: usize,
        y: *mut f32,
        stride: usize,
    ) {
        let (n, k, g) = (w.n, w.k, w.g);
        let (groups, steps, gstride) = (k / g, g / 4, n * g);
        let codes = w.codes.unpacked().as_ptr();
        let (sc, sz, tt) = (w.scale.as_ptr(), w.scale_zero.as_ptr(), w.t.as_ptr());
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
            let (first, last) = (gi == 0, gi + 1 == groups);
            for c in 0..C {
                let s1 = _mm512_loadu_ps(sc.add(gi * n + (ob + c) * 16));
                let s2 = _mm512_loadu_ps(sz.add(gi * n + (ob + c) * 16));
                for (r, ar) in acc.iter().enumerate() {
                    let av = _mm512_set1_ps(*a.add(r * groups + gi));
                    let yp = y.add(r * stride + (ob + c) * 16);
                    let prev = if first {
                        _mm512_setzero_ps()
                    } else {
                        _mm512_loadu_ps(yp)
                    };
                    let f = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(ar[c]), s1), prev);
                    let mut f = _mm512_add_ps(_mm512_mul_ps(s2, av), f);
                    if last {
                        let (sx, zx) = *xs.add(r);
                        let zt = _mm512_mul_ps(
                            _mm512_set1_ps(zx as f32),
                            _mm512_loadu_ps(tt.add((ob + c) * 16)),
                        );
                        f = _mm512_mul_ps(_mm512_sub_ps(f, zt), _mm512_set1_ps(sx));
                        if !b.is_null() {
                            f = _mm512_add_ps(f, _mm512_loadu_ps(b.add((ob + c) * 16)));
                        }
                    }
                    _mm512_storeu_ps(yp, f);
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

    /// `tile` with the running float sums of every group in registers (RT x C of them beside the RT x C integer
    /// sums, so fewer rows or blocks a tile), y written once at the end: the same operations in the same order.
    /// With NIB the codes are read packed (32 bytes for each 64, half the bytes from memory) and unpacked in
    /// registers into the bytes `unpacked` holds.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    unsafe fn tile_reg<const RT: usize, const C: usize, const NIB: bool, const PAIRED: bool>(
        w: &GroupWeights,
        q: *const u8,
        a: *const f32,
        xs: *const (f32, i32),
        b: *const f32,
        ob: usize,
        y: *mut f32,
        stride: usize,
    ) {
        let (n, k, g) = (w.n, w.k, w.g);
        let (groups, steps, gstride) = (k / g, g / 4, n * g);
        let (sc, sz, tt) = (w.scale.as_ptr(), w.scale_zero.as_ptr(), w.t.as_ptr());
        let low = _mm256_set1_epi8(0x0f);
        let low512 = _mm512_set1_epi8(0x0f);
        let codes = if NIB {
            w.packed().as_ptr()
        } else {
            w.codes.unpacked().as_ptr().cast()
        };
        let mut fs = [[_mm512_setzero_ps(); C]; RT];
        for gi in 0..groups {
            let mut acc = [[_mm512_setzero_si512(); C]; RT];
            let (wg, pg) = if NIB {
                (std::ptr::null(), codes.add(gi * gstride / 2))
            } else {
                (codes.add(gi * gstride), std::ptr::null())
            };
            if PAIRED {
                for s in (0..steps).step_by(2) {
                    let mut packed = [_mm512_setzero_si512(); C];
                    for (c, p) in packed.iter_mut().enumerate() {
                        *p = _mm512_load_si512(pg.add(((ob + c) * steps + s) * 32).cast());
                    }
                    let lo = packed.map(|p| _mm512_and_si512(p, low512));
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let x = _mm512_set1_epi32(bcast4(q.add(r * k + gi * g + s * 4)));
                        for (ac, &wc) in ar.iter_mut().zip(&lo) {
                            *ac = _mm512_dpbusd_epi32(*ac, x, wc);
                        }
                    }
                    let hi = packed.map(|p| _mm512_and_si512(_mm512_srli_epi16::<4>(p), low512));
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let x = _mm512_set1_epi32(bcast4(q.add(r * k + gi * g + (s + 1) * 4)));
                        for (ac, &wc) in ar.iter_mut().zip(&hi) {
                            *ac = _mm512_dpbusd_epi32(*ac, x, wc);
                        }
                    }
                }
            } else {
                for s in 0..steps {
                    let mut wv = [_mm512_setzero_si512(); C];
                    for (c, wc) in wv.iter_mut().enumerate() {
                        *wc = if NIB {
                            // 32 bytes: the low nibbles are the block's values 0-31, the high ones 32-63
                            let v = _mm256_loadu_si256(
                                pg.add(((ob + c) * steps + s) * 32) as *const __m256i
                            );
                            let lo = _mm256_and_si256(v, low);
                            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low);
                            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
                        } else {
                            _mm512_loadu_si512(wg.add(((ob + c) * steps + s) * 64) as *const _)
                        };
                    }
                    for (r, ar) in acc.iter_mut().enumerate() {
                        let x = _mm512_set1_epi32(bcast4(q.add(r * k + gi * g + s * 4)));
                        for (ac, wc) in ar.iter_mut().zip(&wv) {
                            *ac = _mm512_dpbusd_epi32(*ac, x, *wc);
                        }
                    }
                }
            }
            for c in 0..C {
                let s1 = _mm512_loadu_ps(sc.add(gi * n + (ob + c) * 16));
                let s2 = _mm512_loadu_ps(sz.add(gi * n + (ob + c) * 16));
                for r in 0..RT {
                    let av = _mm512_set1_ps(*a.add(r * groups + gi));
                    let f =
                        _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[r][c]), s1), fs[r][c]);
                    fs[r][c] = _mm512_add_ps(_mm512_mul_ps(s2, av), f);
                }
            }
        }
        for c in 0..C {
            let tv = _mm512_loadu_ps(tt.add((ob + c) * 16));
            for (r, fr) in fs.iter().enumerate() {
                let (sx, zx) = *xs.add(r);
                let zt = _mm512_mul_ps(_mm512_set1_ps(zx as f32), tv);
                let mut f = _mm512_mul_ps(_mm512_sub_ps(fr[c], zt), _mm512_set1_ps(sx));
                if !b.is_null() {
                    f = _mm512_add_ps(f, _mm512_loadu_ps(b.add((ob + c) * 16)));
                }
                _mm512_storeu_ps(y.add(r * stride + (ob + c) * 16), f);
            }
        }
    }

    macro_rules! by_rows_reg {
        ($rt:expr, $c:literal, $nib:literal, $($arg:expr),*) => {
            match $rt {
                1 => tile_reg::<1, $c, $nib, false>($($arg),*),
                2 => tile_reg::<2, $c, $nib, false>($($arg),*),
                3 => tile_reg::<3, $c, $nib, false>($($arg),*),
                4 => tile_reg::<4, $c, $nib, false>($($arg),*),
                5 => tile_reg::<5, $c, $nib, false>($($arg),*),
                _ => tile_reg::<6, $c, $nib, false>($($arg),*),
            }
        };
    }

    /// `GroupWeights::forward_with` compiled for AVX-512 (its plain Rust parts inlined here and vectorized 16 wide).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    pub(super) unsafe fn forward(
        w: &GroupWeights,
        x: &[f32],
        y: &mut [f32],
        bias: Option<&[f32]>,
        s: &mut super::GroupScratch,
    ) {
        w.forward_with(x, y, bias, s)
    }

    /// Fewer rows need more columns to keep enough independent VNNI chains active.
    /// Balance the row tiles so a long input doesn't end in a tiny tile.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    pub(super) unsafe fn inner_paired<const WIDE: bool>(
        w: &GroupWeights,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        b: *const f32,
        rows: usize,
        cols: Cols,
    ) {
        let (k, groups, stride) = (w.k, w.k / w.g, cols.stride);
        let tiles = rows.div_ceil(6);
        if tiles > 1 {
            let (min_rows, extra) = (rows / tiles, rows % tiles);
            let mut ob = cols.start;
            while ob < cols.end {
                let left = cols.end - ob;
                let c = if WIDE && left >= 4 { 4 } else { left.min(2) };
                let mut r0 = 0;
                for t in 0..tiles {
                    let rt = min_rows + usize::from(t < extra);
                    let (qp, ap, xp, yp) = (
                        q.as_ptr().add(r0 * k),
                        a.as_ptr().add(r0 * groups),
                        xs.as_ptr().add(r0),
                        cols.y.add(r0 * stride),
                    );
                    macro_rules! run {
                        ($c:literal) => {
                            match rt {
                                3 => {
                                    tile_reg::<3, $c, true, true>(w, qp, ap, xp, b, ob, yp, stride)
                                }
                                4 => {
                                    tile_reg::<4, $c, true, true>(w, qp, ap, xp, b, ob, yp, stride)
                                }
                                5 => {
                                    tile_reg::<5, $c, true, true>(w, qp, ap, xp, b, ob, yp, stride)
                                }
                                _ => {
                                    tile_reg::<6, $c, true, true>(w, qp, ap, xp, b, ob, yp, stride)
                                }
                            }
                        };
                    }
                    match c {
                        4 => run!(4),
                        2 => run!(2),
                        _ => run!(1),
                    }
                    r0 += rt;
                }
                ob += c;
            }
            return;
        }
        let mut r0 = 0;
        for t in 0..tiles {
            let rt = rows / tiles + usize::from(t < rows % tiles);
            let (qp, ap, xp, yp) = (
                q.as_ptr().add(r0 * k),
                a.as_ptr().add(r0 * groups),
                xs.as_ptr().add(r0),
                cols.y.add(r0 * stride),
            );
            let mut ob = cols.start;
            while ob < cols.end {
                let left = cols.end - ob;
                macro_rules! run {
                    ($r:literal, $c:literal) => {{
                        tile_reg::<$r, $c, true, true>(w, qp, ap, xp, b, ob, yp, stride);
                        $c
                    }};
                }
                macro_rules! narrow {
                    ($r:literal) => {
                        if left >= 2 {
                            run!($r, 2)
                        } else {
                            run!($r, 1)
                        }
                    };
                }
                macro_rules! four {
                    ($r:literal) => {
                        if left >= 4 {
                            run!($r, 4)
                        } else {
                            narrow!($r)
                        }
                    };
                }
                ob += match rt {
                    1 if left >= 8 => run!(1, 8),
                    1 if left >= 6 => run!(1, 6),
                    1 => four!(1),
                    2 if left >= 6 => run!(2, 6),
                    2 => four!(2),
                    3 => four!(3),
                    4 if left >= 3 => run!(4, 3),
                    4 => narrow!(4),
                    5 if WIDE => four!(5),
                    5 => narrow!(5),
                    _ if WIDE => four!(6),
                    _ => narrow!(6),
                };
            }
            r0 += rt;
        }
    }

    /// Tiled AVX-512 VNNI GEMM, including the strided output epilogue.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2,fma")]
    pub(super) unsafe fn inner(
        w: &GroupWeights,
        q: &[u8],
        a: &[f32],
        xs: &[(f32, i32)],
        bias: Option<&[f32]>,
        rows: usize,
        cols: Cols,
    ) {
        let (k, groups, stride) = (w.k, w.k / w.g, cols.stride);
        let b = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);
        // the tiles (`super::tile_mode`; blocks x rows, the blocks left over one at a time): 0, 4 x 6 with the
        // running sums in y; with them in registers: 1, 4 x 3; 2, 3 x 4; 3, 2 x 6; 4, 2 x 6 for 6 rows or more,
        // else 3 x 4 (3f32ce1's timing: the dense Linears ~17% faster in 2 x 6 than in 0, the experts' few rows a
        // little in 3 x 4); 5, as 4 with the codes read packed for fewer than 6 rows (1b20e28: the experts 190 -> 175
        // us a 30-frame conversion); 6 (the default), as 4 with the codes read packed for all (half the bytes: in the
        // latency runs, the caches cold between conversions, the weights' reads are most of the encoder's time)
        let mode = effective_mode(w);
        let (cmax, rmax, reg, nib) = match mode {
            0 => (4, 6, false, false),
            1 => (4, 3, true, false),
            2 => (3, 4, true, false),
            3 => (2, 6, true, false),
            4 if rows >= 6 => (2, 6, true, false),
            5 if rows >= 6 => (2, 6, true, false),
            5 => (3, 4, true, true),
            4 => (3, 4, true, false),
            _ if rows >= 6 => (2, 6, true, true),
            _ => (3, 4, true, true),
        };
        if nib && w.paired {
            // For 5-6 rows, two blocks keep both the integer and float sums in
            // registers. Four blocks spilled more often and lost in pinned Zen 5 runs.
            return inner_paired::<false>(w, q, a, xs, b, rows, cols);
        }
        let mut ob = cols.start;
        while ob < cols.end {
            let c = if ob + cmax <= cols.end { cmax } else { 1 };
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(if c == 1 { 6 } else { rmax });
                let (qp, ap, xp, yp) = (
                    q.as_ptr().add(r0 * k),
                    a.as_ptr().add(r0 * groups),
                    xs.as_ptr().add(r0),
                    cols.y.add(r0 * stride),
                );
                match (reg, nib, c) {
                    (true, false, 4) => {
                        by_rows_reg!(rt, 4, false, w, qp, ap, xp, b, ob, yp, stride)
                    }
                    (true, false, 3) => {
                        by_rows_reg!(rt, 3, false, w, qp, ap, xp, b, ob, yp, stride)
                    }
                    (true, false, 2) => {
                        by_rows_reg!(rt, 2, false, w, qp, ap, xp, b, ob, yp, stride)
                    }
                    (true, false, _) => {
                        by_rows_reg!(rt, 1, false, w, qp, ap, xp, b, ob, yp, stride)
                    }
                    (true, true, 3) => by_rows_reg!(rt, 3, true, w, qp, ap, xp, b, ob, yp, stride),
                    (true, true, 2) => by_rows_reg!(rt, 2, true, w, qp, ap, xp, b, ob, yp, stride),
                    (true, true, _) => by_rows_reg!(rt, 1, true, w, qp, ap, xp, b, ob, yp, stride),
                    (false, _, 4) => by_rows!(rt, 4, w, qp, ap, xp, b, ob, yp, stride),
                    _ => by_rows!(rt, 1, w, qp, ap, xp, b, ob, yp, stride),
                }
                r0 += rt;
            }
            ob += c;
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
            let weights = GroupWeights::in_place(&w, &scales, &zeros, n, k, g).unwrap();
            // the weights' own copy holds every code the packed bytes do
            for o in 0..n {
                for i in 0..k {
                    assert_eq!(
                        weights.code(o, i),
                        GroupWeights::packed_code(&w, n, g, o, i),
                        "output {o}, input {i}"
                    );
                }
            }
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
                let plain = GroupWeights::in_place(&w, &scales, &zeros, n, k, g)
                    .unwrap()
                    .with_path(Path::Plain);
                let mut yp = vec![0f32; rows * n];
                plain.forward(&x, &mut yp, &mut GroupScratch::default());
                assert_eq!(
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    yp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "the kernel and the plain path, n {n} k {k} g {g} rows {rows}"
                );
                // every tile of the kernel: the same bits (each built for its own mode, verifying the mode's layout)
                let both = GroupWeights::in_place(&w, &scales, &zeros, n, k, g)
                    .unwrap()
                    .with_both_codes(&w);
                for mode in 0..=6u8 {
                    TILE_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
                    let mut ym = vec![f32::NAN; rows * n];
                    both.forward(&x, &mut ym, &mut GroupScratch::default());
                    TILE_MODE.store(TILE_MODE_DEFAULT, std::sync::atomic::Ordering::Relaxed);
                    assert_eq!(
                        ym.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        yp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "tile mode {mode}, n {n} k {k} g {g} rows {rows}"
                    );
                }
                // with a bias: the kernel's in its registers, the plain path's pass, the same bits; over a y of
                // other values (the kernel writes the first group's sums, it reads nothing there)
                let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed) - 0.5).collect();
                let (mut yb, mut ypb) = (vec![7.5f32; rows * n], vec![0f32; rows * n]);
                weights.forward_bias(&x, &mut yb, Some(&bias), &mut GroupScratch::default());
                plain.forward_bias(&x, &mut ypb, Some(&bias), &mut GroupScratch::default());
                assert_eq!(
                    yb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    ypb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "with a bias, n {n} k {k} g {g} rows {rows}"
                );
                for (r, (a, b)) in yb.chunks(n).zip(y.chunks(n)).enumerate() {
                    for o in 0..n {
                        assert_eq!(
                            a[o].to_bits(),
                            (b[o] + bias[o]).to_bits(),
                            "row {r} out {o}: the bias added last"
                        );
                    }
                }
                // the 256- and 128-bit kernels (CPUs without AVX-512) where this CPU has their instructions: the same bits
                for path in [Path::Avx, Path::Avx2, Path::Vnni256, Path::Dot] {
                    if !has_path(path) {
                        continue;
                    }
                    // the codes read unpacked (mode 4) and packed (6), each built for that path and mode
                    for mode in [4u8, 6] {
                        let other =
                            GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, path, mode)
                                .unwrap();
                        TILE_MODE.store(mode, std::sync::atomic::Ordering::Relaxed);
                        let (mut yo, mut yob) = (vec![f32::NAN; rows * n], vec![7.5f32; rows * n]);
                        other.forward(&x, &mut yo, &mut GroupScratch::default());
                        other.forward_bias(&x, &mut yob, Some(&bias), &mut GroupScratch::default());
                        TILE_MODE.store(TILE_MODE_DEFAULT, std::sync::atomic::Ordering::Relaxed);
                        for (got, want) in [(&yo, &yp), (&yob, &ypb)] {
                            assert_eq!(
                                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "{path:?} mode {mode}, n {n} k {k} g {g} rows {rows}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Whether this CPU can run `path` (the tests of the paths below its own level).
    fn has_path(path: Path) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            match path {
                Path::Avx2 => avx2,
                Path::Avx => {
                    is_x86_feature_detected!("avx")
                        && is_x86_feature_detected!("ssse3")
                        && is_x86_feature_detected!("sse4.1")
                }
                Path::Vnni256 => avx2 && is_x86_feature_detected!("avxvnni"),
                Path::Vnni512 => crate::cpu::avx512(),
                Path::Dot => false,
                Path::Plain => true,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            path == Path::Plain || (path == Path::Dot && crate::cpu::dotprod())
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
            let layer = |n: usize, kk: usize, seed: &mut u64| {
                let codes: Vec<u8> = (0..n * kk).map(|_| (rnd(seed) * 8.0) as u8 % 8).collect();
                let scales: Vec<f32> = (0..n * kk / g).map(|_| 0.01 + rnd(seed) * 0.05).collect();
                let zeros: Vec<f32> = (0..n * kk / g).map(|_| (rnd(seed) * 8.0).floor()).collect();
                let w = pack(&codes, n, kk, g);
                (w, scales, zeros)
            };
            let p1 = layer(f, d, &mut seed);
            let p2 = layer(d, f, &mut seed);
            let l1 = GroupWeights::in_place(&p1.0, &p1.1, &p1.2, f, d, g).unwrap();
            let l2 = GroupWeights::in_place(&p2.0, &p2.1, &p2.2, d, f, g).unwrap();
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
        // the layer on every kernel this CPU can run: the same bits
        let mut moe = moe;
        for path in [
            Path::Plain,
            Path::Avx,
            Path::Avx2,
            Path::Vnni256,
            Path::Vnni512,
            Path::Dot,
        ] {
            if !has_path(path) {
                continue;
            }
            for (ex, (l1, l2)) in moe.experts.iter_mut().enumerate() {
                l1.path = path;
                l2.path = path;
                l1.set_both_codes(&keep[ex].0 .0);
                l2.set_both_codes(&keep[ex].1 .0);
            }
            let mut yp = vec![f32::NAN; n * d];
            moe.forward(&x, &mut yp, &mut GroupMoeScratch::default());
            assert_eq!(
                yp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{path:?}"
            );
        }
        drop(keep);
    }

    #[test]
    fn a_matrix_of_no_width_does_not_load() {
        // a model's hidden -> 0 -> hidden feed-forward loaded, then divided by
        // zero at its first conversion
        assert!(GroupWeights::in_place(&[], &[], &[], 0, 64, 64).is_err());
        assert!(GroupWeights::in_place(&[], &[], &[], 16, 0, 64).is_err());
    }

    #[test]
    fn group_weights_keeps_only_the_active_layout() {
        let (n, k, g) = (16usize, 64usize, 64usize);
        let w = vec![0x35u8; n * k / 2];
        let scales = vec![0.02f32; n];
        let zeros = vec![4.0f32; n];
        for path in [Path::Vnni512, Path::Vnni256, Path::Avx2, Path::Avx] {
            let gw = GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, path, 6).unwrap();
            assert!(matches!(gw.codes, Codes::Packed(_)), "{path:?} mode 6");
            let gw = GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, path, 4).unwrap();
            assert!(matches!(gw.codes, Codes::Unpacked(_)), "{path:?} mode 4");
            let gw = GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, path, 5).unwrap();
            assert!(matches!(gw.codes, Codes::Both { .. }), "{path:?} mode 5");
        }
        let plain =
            GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, Path::Plain, 0).unwrap();
        assert!(matches!(plain.codes, Codes::Packed(_)));
        let dot = GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, Path::Dot, 6).unwrap();
        assert!(matches!(dot.codes, Codes::Unpacked(_)));
    }

    #[test]
    fn quantized_rows_and_strided_columns_match_plain() {
        let mut seed = 81;
        for g in [4, 12, 16, 64, 68, 128] {
            let (rows, n, k) = (7, 144, 2 * g);
            let code: Vec<_> = (0..n * k).map(|_| (rnd(&mut seed) * 16.0) as u8).collect();
            let w = pack(&code, n, k, g);
            let scales = vec![0.02; 2 * n];
            let zeros = vec![7.25; 2 * n];
            let bias: Vec<_> = (0..n).map(|_| rnd(&mut seed) - 0.5).collect();
            let plain =
                GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, Path::Plain, 6).unwrap();
            let mut x: Vec<_> = (0..rows * k).map(|_| rnd(&mut seed) * 4.0 - 2.0).collect();
            x[..k].fill(-0.0);
            x[k] = f32::NAN;
            x[2 * k - 1] = f32::NAN;
            x[2 * k] = f32::INFINITY;
            x[3 * k] = f32::NEG_INFINITY;
            x[4 * k..5 * k].fill(f32::from_bits(1));
            let mut q0 = vec![0; rows * k];
            let mut xs0 = vec![(0.0, 0); rows];
            let mut a0 = vec![0.0; rows * 2];
            plain.quantize_rows_into(&x, &mut q0, &mut xs0, &mut a0);
            // Keep the special rows for quantization; matmul needs finite scales.
            let mut x_gemm = x.clone();
            x_gemm[2 * k] = 1.0;
            x_gemm[3 * k] = -1.0;
            let mut want = vec![0.0; n * rows];
            plain.forward_bias(
                &x_gemm,
                &mut want,
                Some(&bias),
                &mut GroupScratch::default(),
            );
            for path in [
                Path::Plain,
                Path::Avx,
                Path::Avx2,
                Path::Vnni256,
                Path::Vnni512,
                Path::Dot,
            ] {
                if !has_path(path) {
                    continue;
                }
                let w = GroupWeights::in_place_for(&w, &scales, &zeros, n, k, g, path, 6).unwrap();
                assert_eq!((w.n(), w.k(), w.g()), (n, k, g));
                let mut s = GroupScratch::default();
                w.quantize_rows(&x, &mut s.q, &mut s.xs, &mut s.a);
                assert_eq!(s.q, q0, "{path:?} g={g}");
                assert_eq!(s.a, a0);
                for (&(s, z), &(s0, z0)) in s.xs.iter().zip(&xs0) {
                    assert_eq!((s.to_bits(), z), (s0.to_bits(), z0));
                }
                w.quantize_rows(&x_gemm, &mut s.q, &mut s.xs, &mut s.a);
                let stride = n + 7;
                let marker = f32::NAN.to_bits();
                let mut got = vec![f32::from_bits(marker); rows * stride];
                for (start, end) in [(2, 7), (0, 2), (7, w.blocks())] {
                    unsafe {
                        w.codes_with_cols_ptr(
                            &s.q,
                            &s.a,
                            &s.xs,
                            rows,
                            got.as_mut_ptr(),
                            stride,
                            Some(&bias),
                            start,
                            end,
                        );
                    }
                    if start == 2 {
                        for r in got.chunks_exact(stride) {
                            assert!(r[..32]
                                .iter()
                                .chain(&r[112..])
                                .all(|v| v.to_bits() == marker));
                        }
                    }
                }
                for (got, want) in got.chunks_exact(stride).zip(want.chunks_exact(n)) {
                    assert!(
                        got[..n]
                            .iter()
                            .zip(want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{path:?} g={g}"
                    );
                    assert!(got[n..].iter().all(|v| v.to_bits() == marker));
                }
            }
        }
    }

    /// Representative dense and expert Linear shapes, median of 500.
    /// cargo test -p setsuna-engine --release --lib group_q::tests::timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn timing() {
        let mut seed = 5u64;
        for &(n, k) in &[
            (576usize, 192usize),
            (768, 192),
            (192, 768),
            (192, 192),
            (384, 192),
            (192, 384),
        ] {
            let g = 64;
            let codes: Vec<u8> = (0..n * k)
                .map(|_| (rnd(&mut seed) * 16.0) as u8 % 16)
                .collect();
            let groups = k / g;
            let scales: Vec<f32> = (0..n * groups).map(|_| 0.02).collect();
            let zeros: Vec<f32> = (0..n * groups).map(|_| 8.0).collect();
            let w = pack(&codes, n, k, g);
            let weights = GroupWeights::in_place(&w, &scales, &zeros, n, k, g).unwrap();
            let x: Vec<f32> = (0..30 * k).map(|_| rnd(&mut seed) * 2.0 - 1.0).collect();
            for rows in [1, 2, 3, 4, 5, 6, 22, 30] {
                let x = &x[..rows * k];
                let mut y = vec![0f32; rows * n];
                let mut s = GroupScratch::default();
                let median = |run: &mut dyn FnMut()| {
                    let mut times = [0u128; 500];
                    const BATCH: u128 = 16;
                    for t in &mut times {
                        let start = Instant::now();
                        for _ in 0..BATCH {
                            run();
                        }
                        *t = start.elapsed().as_nanos();
                    }
                    times.sort_unstable();
                    times[times.len() / 2] as f64 / (BATCH as f64 * 1000.0)
                };
                let all = median(&mut || {
                    weights.forward(x, &mut y, &mut s);
                    std::hint::black_box(&y);
                });
                print!(
                    "LinearQ4G {k} -> {n} on {rows:2} rows ({:?}): {all:.3} us",
                    weights.path
                );
                #[cfg(target_arch = "x86_64")]
                if weights.paired && rows >= 5 {
                    for wide in [false, true] {
                        let cols = Cols {
                            y: y.as_mut_ptr(),
                            stride: n,
                            start: 0,
                            end: n / 16,
                        };
                        let gemm = median(&mut || unsafe {
                            if wide {
                                x86::inner_paired::<true>(
                                    &weights,
                                    &s.q,
                                    &s.a,
                                    &s.xs,
                                    std::ptr::null(),
                                    rows,
                                    cols,
                                );
                            } else {
                                x86::inner_paired::<false>(
                                    &weights,
                                    &s.q,
                                    &s.a,
                                    &s.xs,
                                    std::ptr::null(),
                                    rows,
                                    cols,
                                );
                            }
                            std::hint::black_box(&y);
                        });
                        print!(", C={}: {gemm:.3} us (GEMM only)", if wide { 4 } else { 2 });
                    }
                }
                println!();
            }
        }
        let (e, d, f, g) = (16, 192, 384, 64);
        let router: Vec<_> = (0..e * d).map(|_| rnd(&mut seed) - 0.5).collect();
        let x: Vec<_> = (0..30 * d).map(|_| 2.0 * rnd(&mut seed) - 1.0).collect();
        let w: Vec<_> = (0..e * d * f / 2)
            .map(|_| (rnd(&mut seed) * 256.0) as u8)
            .collect();
        let scales = vec![0.02; e * d * f / g];
        let zeros = vec![7.5; scales.len()];
        let moe = group_moe(
            e,
            d,
            f,
            2,
            g,
            router,
            (&w, &scales, &zeros, vec![0.0; e * f]),
            (&w, &scales, &zeros, vec![0.0; e * d]),
        )
        .unwrap();
        let [route, l1, gelu_quant, l2, add] = moe.time_parts(&x, 1500);
        let total = route + l1 + gelu_quant + l2 + add;
        println!("Q4G MoE (30 rows): {total:.3} us (route+quant {route:.3}, L1 {l1:.3}, GELU+quant {gelu_quant:.3}, L2 {l2:.3}, add {add:.3})");
    }
}
