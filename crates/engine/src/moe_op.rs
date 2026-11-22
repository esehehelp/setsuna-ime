//! The int8 matrix products the encoder's kernels share, and the helpers of its token-routed experts.
//!
//! A row of activations is quantized to u8 per row (asymmetric, zero exact: `quantize_row`) and multiplied with int8
//! weights prepacked in blocks (`Packed`) as u8 x i8 dot products of four (`gemm_u8i8`, and `gemm_u8i8_f32` with its
//! epilogue: the scales, the bias and GELU), with AVX-512 VNNI, AVX-VNNI, AVX2, AVX or plain Rust, the same integer
//! sums on every path. `route` and `add_routed` are a token-routed mixture of experts' router and sum (per frame the
//! router's fp32 logits, the top-`k` experts with their softmax renormalised over the `k`, the weighted outputs
//! summed), the experts' Linears being `group_q`'s.

// the x86 kernels and what only they read are unused elsewhere
#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

/// An int8 matrix (out, in) prepacked for the u8 x i8 dot products: 16 output rows x 4 inputs a 64-byte block (lane j
/// = row 16 * b + j), blocks along the inputs; `colsum` the sum of each row (for the activations' zero point). Only
/// one layout is kept, matching the active ISA.
pub(crate) struct Packed {
    out: usize,
    inp: usize,
    weights: WeightLayout,
    colsum: Vec<i32>,
}

/// Mutually exclusive weight buffer for the active execution path.
enum WeightLayout {
    /// Row-major int8 (plain path or shape not aligned to 16 x 4 blocks).
    Raw(Vec<i8>),
    /// 16 x 4 blocked int8 (VNNI-512, AVX-VNNI, ARM SDOT).
    Blocked(BlockedWeights),
    /// 16-bit widened copy of the 16 x 4 blocks (AVX2, AVX).
    Wide(Vec<i16>),
}

impl Packed {
    /// `blocked`: pack for the active SIMD path (when the shape allows 16 x 4 blocks), else keep row-major.
    pub(crate) fn new(w: &[i8], out: usize, inp: usize, blocked: bool) -> Self {
        Self::new_for(w, out, inp, blocked, detect_isa())
    }

    fn new_for(w: &[i8], out: usize, inp: usize, blocked: bool, isa: Isa) -> Self {
        assert_eq!(w.len(), out * inp);
        let colsum = (0..out)
            .map(|o| w[o * inp..(o + 1) * inp].iter().map(|&v| v as i32).sum())
            .collect();
        let weights = if !(blocked && out % 16 == 0 && inp % 4 == 0) {
            WeightLayout::Raw(w.to_vec())
        } else if matches!(isa, Isa::Avx2 | Isa::Avx) {
            WeightLayout::Wide(widen_from_raw(w, out, inp))
        } else {
            WeightLayout::Blocked(pack_blocked(w, out, inp))
        };
        Packed {
            out,
            inp,
            weights,
            colsum,
        }
    }

    /// Number of 16-column work units. Only a raw matrix can have a short last block.
    pub(crate) fn blocks(&self) -> usize {
        self.out.div_ceil(16)
    }

    /// Tests only: pack into 16 x 4 `Blocked` bytes regardless of the CPU ISA.
    #[cfg(test)]
    pub(crate) fn new_blocked(w: &[i8], out: usize, inp: usize) -> Self {
        Self::new_for(w, out, inp, true, Isa::Vnni512)
    }

    /// Tests only: pack into the AVX2/AVX 16-bit `Wide` layout regardless of the CPU ISA.
    #[cfg(test)]
    pub(crate) fn new_wide(w: &[i8], out: usize, inp: usize) -> Self {
        Self::new_for(w, out, inp, true, Isa::Avx2)
    }
}

struct BlockedWeights {
    buf: Vec<i8>,
    offset: usize,
}

impl std::ops::Deref for BlockedWeights {
    type Target = [i8];

    fn deref(&self) -> &[i8] {
        &self.buf[self.offset..]
    }
}

/// Pack row-major `(out, inp)` int8 weights into cache-line-aligned 16 x 4 blocks.
fn pack_blocked(w: &[i8], out: usize, inp: usize) -> BlockedWeights {
    let mut buf = Vec::<i8>::with_capacity(w.len() + 63);
    let offset = buf.as_ptr().align_offset(64);
    buf.resize(offset + w.len(), 0);
    let blocked = &mut buf[offset..];
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
    BlockedWeights { buf, offset }
}

/// The AVX2/AVX kernels' 16-bit widened layout built directly from row-major `(out, inp)` int8 weights: for each
/// block of 16 outputs and each 4 inputs, 4 registers of 16 values, outputs 0-7 with inputs 0 and 1 (output after
/// output, its two inputs side by side), then with inputs 2 and 3, then outputs 8-15 likewise; so one 16-bit
/// multiply-add of a pair of inputs gives 8 outputs' partial sums (`x86::tile_avx2w`).
fn widen_from_raw(w: &[i8], out: usize, inp: usize) -> Vec<i16> {
    let mut wide = Vec::with_capacity(out * inp);
    for ob in 0..out / 16 {
        for kg in 0..inp / 4 {
            for half in 0..2 {
                for pair in 0..2 {
                    for o in 0..8 {
                        let at = (ob * 16 + half * 8 + o) * inp + kg * 4 + pair * 2;
                        wide.push(w[at] as i16);
                        wide.push(w[at + 1] as i16);
                    }
                }
            }
        }
    }
    wide
}

/// The int8 GEMM's kernels: AVX-512 VNNI, AVX-VNNI, AVX2 (without a dot product instruction), plain Rust; the same
/// integer sums on each.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Isa {
    Vnni512,
    Vnni256,
    Avx2,
    Avx,
    /// ARMv8.2 SDOT (`arm`).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Dot,
    Plain,
}

/// The kernel for the engine's level (`crate::cpu`; SETSUNA_CPU_LEVEL caps it).
pub(crate) fn detect_isa() -> Isa {
    use crate::cpu::Level;
    if crate::cpu::dotprod() {
        return Isa::Dot;
    }
    match crate::cpu::level() {
        Level::Avx512 => Isa::Vnni512,
        Level::AvxVnni => Isa::Vnni256,
        Level::Avx2 => Isa::Avx2,
        Level::Avx => Isa::Avx,
        Level::Plain => Isa::Plain,
    }
}

/// Whether the CPU has what `isa`'s kernels run (whatever SETSUNA_CPU_LEVEL caps): the safe entry points take an
/// `Isa` from anywhere (a test's too), and one the CPU lacks would run instructions it does not have.
pub(crate) fn supported(isa: Isa) -> bool {
    use crate::cpu::{hardware, Level};
    match isa {
        Isa::Vnni512 => hardware() == Level::Avx512,
        Isa::Vnni256 => crate::cpu::has_avx_vnni(),
        Isa::Avx2 => hardware() >= Level::Avx2,
        Isa::Avx => hardware() >= Level::Avx,
        Isa::Dot => crate::cpu::has_dotprod(),
        Isa::Plain => true,
    }
}

/// `rows * width` as the length a kernel reads or writes, which a slice must have: the kernels go by raw pointers.
pub(crate) fn span(rows: usize, width: usize) -> usize {
    rows.checked_mul(width).expect("a matrix's size overflows")
}

/// The integer sums acc[r][o] = sum_i q[r][i] * w[o][i] of `rows` u8 rows (stride `w.inp`) with the int8 matrix.
pub(crate) fn gemm_u8i8(isa: Isa, q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
    // checked in release too: the kernels read and write by raw pointers
    assert!(
        q.len() >= span(rows, w.inp) && acc.len() >= span(rows, w.out),
        "{} inputs and {} sums for {rows} rows of {} x {}",
        q.len(),
        acc.len(),
        w.inp,
        w.out
    );
    assert!(supported(isa), "{isa:?} on a CPU without it");
    match (isa, &w.weights) {
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni512, WeightLayout::Blocked(blocked)) => unsafe {
            x86::gemm_vnni512(q, rows, w, blocked, acc)
        },
        #[cfg(target_arch = "x86_64")]
        (Isa::Vnni256, WeightLayout::Blocked(blocked)) => unsafe {
            x86::gemm_vnni256(q, rows, w, blocked, acc)
        },
        // AVX2: the 16-bit copy's kernel when the matrix has it (packed at the AVX2/AVX level)
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, WeightLayout::Wide(wide)) => unsafe { x86::gemm_avx2w(q, rows, w, wide, acc) },
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, WeightLayout::Blocked(blocked)) => unsafe {
            x86::gemm_avx2(q, rows, w, blocked, acc)
        },
        // AVX (128-bit integers): the 16-bit copy's kernel
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx, WeightLayout::Wide(wide)) => unsafe { x86::gemm_avxw(q, rows, w, wide, acc) },
        // SAFETY: the CPU has the dot products (`detect_isa`)
        #[cfg(target_arch = "aarch64")]
        (Isa::Dot, WeightLayout::Blocked(blocked)) => unsafe {
            arm::gemm_dot(q, rows, w, blocked, acc)
        },
        _ => gemm_plain(q, rows, w, acc),
    }
}

/// How `gemm_u8i8_f32` turns a row's integer sums into its outputs: with the row's activation scale and zero point
/// (`xs`: one pair a row, or one for every row) and the column sums `cs`, i = acc - zx cs; then `i * (sx * scale)`
/// (one weight scale, `col` None: the native encoder's `Lin8`) or `sx * col[o] * i` (a scale an output: the AR step's
/// `Linear8`), plus `bias[o]` when given; the first `out` columns.
pub(crate) struct Epilogue<'a> {
    pub xs: &'a [(f32, i32)],
    pub col: Option<&'a [f32]>,
    pub scale: f32,
    pub bias: Option<&'a [f32]>,
    pub out: usize,
}

/// y (rows, `e.out`) of the u8 rows `q` with the int8 matrix: `gemm_u8i8`'s sums turned to floats by `e`, in the
/// GEMM's registers with AVX-512 VNNI (2026-11-19: an 8006-token head's i32 sums of 30 rows were 960 KB written and
/// read again), else the sums into `acc` and a pass: the same operations in the same order, the same bits.
pub(crate) fn gemm_u8i8_f32(
    isa: Isa,
    q: &[u8],
    rows: usize,
    w: &Packed,
    e: &Epilogue,
    acc: &mut Vec<i32>,
    y: &mut [f32],
) {
    let per_row = e.xs.len() > 1;
    validate_f32(isa, q, rows, w, e);
    assert!(y.len() >= span(rows, e.out));
    #[cfg(target_arch = "x86_64")]
    {
        let raw = x86::Epi::new(w, e);
        match (isa, &w.weights) {
            (Isa::Vnni512, WeightLayout::Blocked(blocked)) => {
                // SAFETY: the CPU has AVX-512 VNNI (`isa`); the slices hold what the asserts say
                return unsafe {
                    x86::gemm_vnni512_f32(
                        q,
                        rows,
                        w,
                        blocked,
                        &raw,
                        y.as_mut_ptr(),
                        0,
                        e.out.div_ceil(16),
                    )
                };
            }
            (Isa::Avx2 | Isa::Avx, WeightLayout::Wide(wide)) => {
                // the 16-bit copy's kernels read each row's inputs as 16-bit pairs (x0 | x1 << 16), made here into `acc`
                let k = w.inp;
                acc.clear();
                acc.extend(
                    q[..rows * k]
                        .chunks_exact(2)
                        .map(|p| p[0] as i32 | (p[1] as i32) << 16),
                );
                // SAFETY: the CPU has AVX2 and FMA, or AVX, SSSE3 and SSE4.1 (`isa`); as above
                return unsafe {
                    if isa == Isa::Avx2 {
                        x86::gemm_avx2w_f32(acc, rows, w, wide, &raw, y)
                    } else {
                        x86::gemm_avxw_f32(acc, rows, w, wide, &raw, y)
                    }
                };
            }
            _ => {}
        }
    }
    let n = w.out;
    acc.resize(rows * n, 0);
    gemm_u8i8(isa, q, rows, w, acc);
    let cs = &w.colsum[..e.out];
    for r in 0..rows {
        let (sx, zx) = e.xs[if per_row { r } else { 0 }];
        let (a, yr) = (
            &acc[r * n..r * n + e.out],
            &mut y[r * e.out..(r + 1) * e.out],
        );
        for o in 0..e.out {
            let f = a[o].wrapping_sub(zx.wrapping_mul(cs[o])) as f32;
            let mut v = match e.col {
                None => f * (sx * e.scale),
                Some(c) => sx * c[o] * f,
            };
            if let Some(b) = e.bias {
                v += b[o];
            }
            yr[o] = v;
        }
    }
}

fn validate_f32(isa: Isa, q: &[u8], rows: usize, w: &Packed, e: &Epilogue) {
    assert!(
        e.xs.len() >= rows || e.xs.len() == 1,
        "{} scales for {rows} rows",
        e.xs.len()
    );
    assert!(e.out <= w.out);
    assert!(e.col.is_none_or(|c| c.len() >= e.out) && e.bias.is_none_or(|b| b.len() >= e.out));
    assert!(q.len() >= span(rows, w.inp));
    assert!(supported(isa), "{isa:?} on a CPU without it");
}

/// Writes only the requested columns into the (rows, `e.out`) output.
/// # Safety
/// Each cell of the selected column blocks (up to `e.out`) in each row must be writable at `y`.
/// No overlapping accesses or references are permitted; workers can share a raw base pointer
/// to write disjoint column ranges.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) unsafe fn gemm_u8i8_f32_cols_ptr(
    isa: Isa,
    q: &[u8],
    rows: usize,
    w: &Packed,
    e: &Epilogue,
    y: *mut f32,
    ob_start: usize,
    ob_end: usize,
) {
    validate_f32(isa, q, rows, w, e);
    span(rows, e.out);
    assert!(ob_start <= ob_end && ob_end <= w.blocks());
    let end = ob_end.min(e.out.div_ceil(16));
    if ob_start >= end || rows == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if let (Isa::Vnni512, WeightLayout::Blocked(blocked)) = (isa, &w.weights) {
        return x86::gemm_vnni512_f32(q, rows, w, blocked, &x86::Epi::new(w, e), y, ob_start, end);
    }
    visit_fallback_tiles(isa, q, rows, w, e, ob_start, end, |r, o, v| {
        std::ptr::copy_nonoverlapping(v.as_ptr(), y.add(r * e.out + o), v.len());
    });
}

/// A row's reduction over one column range. Blacklisted columns affect the normalization,
/// but not `top`. The blank is recorded raw, even when blacklisted.
#[derive(Debug)]
pub(crate) struct PartialFrameReduce {
    pub max: f32,
    pub sum_exp: f32,
    pub top: Vec<(u32, f32)>,
    pub blank_logit: f32,
    pub has_non_finite: bool,
}

impl Default for PartialFrameReduce {
    fn default() -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum_exp: 0.0,
            top: Vec::new(),
            blank_logit: f32::NEG_INFINITY,
            has_non_finite: false,
        }
    }
}

impl PartialFrameReduce {
    #[inline]
    pub(crate) fn reset(&mut self, seed_max: f32) {
        self.max = seed_max;
        self.sum_exp = 0.0;
        self.top.clear();
        self.blank_logit = f32::NEG_INFINITY;
        self.has_non_finite = false;
    }

    #[inline]
    fn insert(&mut self, keep: usize, id: u32, v: f32) {
        let i = self
            .top
            .partition_point(|&(j, x)| x > v || (x == v && j < id));
        if i < keep {
            if self.top.len() == keep {
                self.top.copy_within(i..keep - 1, i + 1);
                self.top[i] = (id, v);
            } else {
                self.top.insert(i, (id, v));
            }
        }
    }

    #[inline]
    fn raise_max(&mut self, new: f32, greedy_only: bool) {
        if new > self.max {
            if !greedy_only && self.sum_exp != 0.0 {
                self.sum_exp *= crate::pipeline::lse::exp(self.max - new);
            }
            self.max = new;
        }
    }

    fn reduce_scalar(
        &mut self,
        start: usize,
        v: &[f32],
        blocked: u16,
        blank: u32,
        keep: usize,
        greedy_only: bool,
    ) {
        let mut max = f32::NEG_INFINITY;
        for (j, &x) in v.iter().enumerate() {
            let id = (start + j) as u32;
            if id == blank {
                self.blank_logit = x;
            }
            self.has_non_finite |= !x.is_finite();
            max = max.max(x);
            if blocked & (1 << j) == 0
                && !x.is_nan()
                && (self.top.len() < keep || self.top.last().is_some_and(|&(_, y)| x > y))
            {
                self.insert(keep, id, x);
            }
        }
        self.raise_max(max, greedy_only);
        // The scalar lse::peaked always includes the short final block.
        if !greedy_only && self.max.is_finite() && (v.len() < 16 || max > self.max - 24.0) {
            self.sum_exp += v
                .iter()
                .map(|&x| crate::pipeline::lse::exp(x - self.max))
                .sum::<f32>();
        }
    }
}

/// Reduce a column range during the GEMM, retaining just its per-row summaries. Ranges are
/// in units of 16 columns relative to the original matrix, clamped to `e.out`.
/// `blacklist` and `blank_id` are column IDs. Each worker owns its own returned summaries.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn gemm_u8i8_reduce_cols(
    isa: Isa,
    q: &[u8],
    rows: usize,
    w: &Packed,
    e: &Epilogue,
    ob_start: usize,
    ob_end: usize,
    blacklist: &[u32],
    blank_id: u32,
    k: usize,
    greedy_only: bool,
) -> Vec<PartialFrameReduce> {
    let mut partial = Vec::new();
    let mut masks = Vec::new();
    gemm_u8i8_reduce_cols_into(
        isa,
        q,
        rows,
        w,
        e,
        ob_start,
        ob_end,
        blacklist,
        blank_id,
        k,
        greedy_only,
        None,
        &mut partial,
        &mut masks,
    );
    partial
}

/// Zero-allocation version of [`gemm_u8i8_reduce_cols`] reusing caller-owned `partial` and `masks`.
/// When `seed_max` is provided (`seed_max[r] <= global_max[r]` from another block, e.g. block 0),
/// each row starts with `max = seed_max[r]` so blocks below `seed_max[r] - 24` skip `exp16_le0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_u8i8_reduce_cols_into(
    isa: Isa,
    q: &[u8],
    rows: usize,
    w: &Packed,
    e: &Epilogue,
    ob_start: usize,
    ob_end: usize,
    blacklist: &[u32],
    blank_id: u32,
    k: usize,
    greedy_only: bool,
    seed_max: Option<&[f32]>,
    partial: &mut Vec<PartialFrameReduce>,
    masks: &mut Vec<u16>,
) {
    validate_f32(isa, q, rows, w, e);
    assert!(ob_start <= ob_end && ob_end <= w.blocks() && u32::try_from(e.out).is_ok());
    let end = ob_end.min(e.out.div_ceil(16));
    let keep = if greedy_only { 2 } else { k.max(2) }.min(e.out);
    if partial.len() < rows {
        partial.resize_with(rows, PartialFrameReduce::default);
    }
    partial.truncate(rows);
    let cap = keep.min(end.saturating_sub(ob_start) * 16);
    for (r, p) in partial.iter_mut().enumerate() {
        let seed = seed_max.map_or(f32::NEG_INFINITY, |s| s[r]);
        p.reset(seed);
        if p.top.capacity() < cap {
            p.top.reserve(cap - p.top.len());
        }
    }
    if ob_start >= end || rows == 0 {
        return;
    }
    let n_blocks = end - ob_start;
    masks.clear();
    masks.resize(n_blocks, 0u16);
    for &id in blacklist {
        let (b, j) = (id as usize / 16, id % 16);
        if (ob_start..end).contains(&b) {
            masks[b - ob_start] |= 1 << j;
        }
    }
    #[cfg(target_arch = "x86_64")]
    if let (Isa::Vnni512, WeightLayout::Blocked(blocked)) = (isa, &w.weights) {
        unsafe {
            x86::gemm_vnni512_reduce(
                q,
                rows,
                w,
                blocked,
                &x86::Epi::new(w, e),
                ob_start,
                masks,
                blank_id,
                keep,
                greedy_only,
                partial,
            )
        };
        return;
    }
    visit_fallback_tiles(isa, q, rows, w, e, ob_start, end, |r, o, v| {
        partial[r].reduce_scalar(o, v, masks[o / 16 - ob_start], blank_id, keep, greedy_only);
    });
}

pub(crate) type FrameReduction = (Vec<(u32, f32)>, f32, u32, f32, f32);

/// Merge workers' summaries in any range order. Returns `(top, blank, top1_id, top1, top2)`
/// per frame. `None` requests an unfused fallback (non-finite logits). Greedy scores are
/// shifted by the best eligible logit, not normalized; the returned `top` has at most two entries.
#[allow(dead_code)]
pub(crate) fn merge_partial_reduce(
    parts: &[Vec<PartialFrameReduce>],
    blacklist: &[u32],
    blank_id: u32,
    k: usize,
    greedy_only: bool,
) -> Option<Vec<FrameReduction>> {
    let slices: Vec<&[PartialFrameReduce]> = parts.iter().map(Vec::as_slice).collect();
    merge_partial_reduce_slices(&slices, blacklist, blank_id, k, greedy_only)
}

/// Slice-based version of [`merge_partial_reduce`] that borrows worker-owned slices without moving Vecs.
pub(crate) fn merge_partial_reduce_slices(
    parts: &[&[PartialFrameReduce]],
    blacklist: &[u32],
    blank_id: u32,
    k: usize,
    greedy_only: bool,
) -> Option<Vec<FrameReduction>> {
    let rows = parts.first().map_or(0, |p| p.len());
    assert!(parts.iter().all(|p| p.len() == rows));
    let keep = if greedy_only { 2 } else { k.max(2) };
    let blank_blocked = blacklist.contains(&blank_id);
    let mut frames = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut combined = PartialFrameReduce {
            top: Vec::with_capacity(keep),
            ..PartialFrameReduce::default()
        };
        for p in parts {
            let p = &p[r];
            if p.has_non_finite {
                return None;
            }
            combined.max = combined.max.max(p.max);
            combined.blank_logit = combined.blank_logit.max(p.blank_logit);
            for &(id, v) in &p.top {
                if combined.top.len() < keep
                    || v > combined.top[keep - 1].1
                    || (v == combined.top[keep - 1].1 && id < combined.top[keep - 1].0)
                {
                    combined.insert(keep, id, v);
                }
            }
        }
        let norm = if greedy_only {
            combined.top.first().map_or(0.0, |&(_, v)| v)
        } else if combined.max.is_finite() {
            let sum: f32 = parts
                .iter()
                .map(|p| {
                    let p = &p[r];
                    if p.sum_exp == 0.0 {
                        0.0
                    } else {
                        p.sum_exp * crate::pipeline::lse::exp(p.max - combined.max)
                    }
                })
                .sum();
            combined.max + sum.ln()
        } else {
            0.0
        };
        let lp = |v: f32| v - norm;
        let (top1_id, top1) = combined
            .top
            .first()
            .map_or((blank_id, f32::NEG_INFINITY), |&(id, v)| (id, lp(v)));
        let top2 = combined
            .top
            .get(1)
            .map_or(f32::NEG_INFINITY, |&(_, v)| lp(v));
        let blank = if blank_blocked {
            f32::NEG_INFINITY
        } else {
            lp(combined.blank_logit)
        };
        combined.top.truncate(if greedy_only { 2 } else { k });
        for (_, v) in &mut combined.top {
            *v = lp(*v);
        }
        frames.push((combined.top, blank, top1_id, top1, top2));
    }
    Some(frames)
}

#[allow(clippy::too_many_arguments)]
fn visit_fallback_tiles(
    isa: Isa,
    q: &[u8],
    rows: usize,
    w: &Packed,
    e: &Epilogue,
    start: usize,
    end: usize,
    mut visit: impl FnMut(usize, usize, &[f32]),
) {
    const RT: usize = 4;
    let per_row = e.xs.len() > 1;
    for ob in start..end {
        let o = ob * 16;
        let len = (e.out - o).min(16);
        for r0 in (0..rows).step_by(RT) {
            let m = (rows - r0).min(RT);
            let input = &q[r0 * w.inp..(r0 + m) * w.inp];
            let mut acc = [[0; 16]; RT];
            #[cfg(target_arch = "x86_64")]
            let simd = unsafe { x86::fallback_tile(isa, input, m, w, ob, acc.as_mut_ptr().cast()) };
            #[cfg(target_arch = "aarch64")]
            let simd = if let (Isa::Dot, WeightLayout::Blocked(blocked)) = (isa, &w.weights) {
                unsafe { arm::fallback_tile(input, m, w, blocked, ob, acc.as_mut_ptr().cast()) };
                true
            } else {
                false
            };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let simd = false;
            for (r, a) in acc[..m].iter().enumerate() {
                let (sx, zx) = e.xs[if per_row { r0 + r } else { 0 }];
                let qr = &input[r * w.inp..(r + 1) * w.inp];
                let mut v = [0.0; 16];
                for j in 0..len {
                    let c = o + j;
                    let sum = if simd { a[j] } else { dot_plain(qr, w, c) };
                    let f = sum.wrapping_sub(zx.wrapping_mul(w.colsum[c])) as f32;
                    let mut f = match e.col {
                        None => f * (sx * e.scale),
                        Some(scale) => sx * scale[c] * f,
                    };
                    if let Some(b) = e.bias {
                        f += b[c];
                    }
                    v[j] = f;
                }
                visit(r0 + r, o, &v[..len]);
            }
        }
    }
}

/// The blocked int8 GEMM with ARM's SDOT (signed i8 x i8 dot products of four into each i32 lane): a block's 64 bytes
/// are four vectors, outputs 0-3, 4-7, 8-11 and 12-15, each lane an output's four inputs, so the layout of the VNNI
/// paths serves as it is. The u8 codes go in as i8 with their top bit flipped (code - 128), and the 128 * row sum of
/// the weights this takes off is put back after, so the sums are the plain path's.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::Packed;
    use std::arch::aarch64::*;

    /// SDOT: each lane of `acc` plus the dot product of its four i8 of `w` and of `x` (the intrinsic is not stable).
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
        let mut r = acc;
        std::arch::asm!(
            "sdot {r:v}.4s, {w:v}.16b, {x:v}.16b",
            r = inout(vreg) r,
            w = in(vreg) w,
            x = in(vreg) x,
            options(pure, nomem, nostack, preserves_flags)
        );
        r
    }

    /// Row r's four codes at input 4 kg, top bits flipped, in every lane.
    #[inline(always)]
    unsafe fn codes(q: *const u8, at: usize) -> int8x16_t {
        let v = vreinterpretq_u8_u32(vdupq_n_u32((q.add(at) as *const u32).read_unaligned()));
        vreinterpretq_s8_u8(veorq_u8(v, vdupq_n_u8(0x80)))
    }

    /// RT rows x C output blocks (block c `bstride` bytes after block 0).
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn tile<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let mut a = [[[vdupq_n_s32(0); 4]; C]; RT];
        for kg in 0..k / 4 {
            let mut x = [vdupq_n_s8(0); RT];
            for (r, xr) in x.iter_mut().enumerate() {
                *xr = codes(q, r * k + kg * 4);
            }
            for c in 0..C {
                let p = wb.add(c * bstride + kg * 64);
                let w = [
                    vld1q_s8(p),
                    vld1q_s8(p.add(16)),
                    vld1q_s8(p.add(32)),
                    vld1q_s8(p.add(48)),
                ];
                for (ar, xr) in a.iter_mut().zip(&x) {
                    for (av, wv) in ar[c].iter_mut().zip(&w) {
                        *av = sdot(*av, *wv, *xr);
                    }
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                for (v, av) in ac.iter().enumerate() {
                    vst1q_s32(acc.add(r * n + c * 16 + v * 4), *av);
                }
            }
        }
    }

    /// `tile::<rt, c>` for a row count known at run time: 1 row on 4 blocks, 2 on 2 (16 sums each), else up to 4
    /// rows on one block.
    macro_rules! by_rows {
        ($rt:expr, $c:expr, $($arg:expr),*) => {
            match ($rt, $c) {
                (1, 4) => tile::<1, 4>($($arg),*),
                (2, 2) => tile::<2, 2>($($arg),*),
                (1, _) => tile::<1, 1>($($arg),*),
                (2, _) => tile::<2, 1>($($arg),*),
                (3, _) => tile::<3, 1>($($arg),*),
                _ => tile::<4, 1>($($arg),*),
            }
        };
    }

    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn fallback_tile(
        q: &[u8],
        m: usize,
        w: &Packed,
        blocked: &[i8],
        ob: usize,
        a: *mut i32,
    ) {
        let stride = w.inp * 16;
        by_rows!(
            m,
            1,
            q.as_ptr(),
            w.inp,
            blocked.as_ptr().add(ob * stride),
            stride,
            a,
            16
        );
        for r in 0..m {
            for j in 0..16 {
                *a.add(r * 16 + j) += 128 * w.colsum[ob * 16 + j];
            }
        }
    }

    /// The integer sums of `super::gemm_u8i8` on a blocked matrix.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn gemm_dot(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let bstride = k * 16;
        let (qp, ap, wp) = (q.as_ptr(), acc.as_mut_ptr(), blocked.as_ptr());
        // few rows: more blocks at once, so the sums do not wait on each other
        let c = match rows {
            1 => 4,
            2 => 2,
            _ => 1,
        };
        let blocks = n / 16;
        let mut ob = 0;
        while ob < blocks {
            let cb = if ob + c <= blocks { c } else { 1 };
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(4);
                by_rows!(
                    rt,
                    cb,
                    qp.add(r0 * k),
                    k,
                    wp.add(ob * bstride),
                    bstride,
                    ap.add(r0 * n + ob * 16),
                    n
                );
                r0 += rt;
            }
            ob += cb;
        }
        // the codes went in as code - 128: put back what that took off
        for r in 0..rows {
            for (a, &cs) in acc[r * n..(r + 1) * n].iter_mut().zip(&w.colsum) {
                *a += 128 * cs;
            }
        }
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
            acc[r * n + o] = dot_plain(qr, w, o);
        }
    }
}

#[inline]
fn dot_plain(qr: &[u8], w: &Packed, o: usize) -> i32 {
    let k = w.inp;
    match &w.weights {
        WeightLayout::Raw(raw) => {
            let wr = &raw[o * k..(o + 1) * k];
            qr.iter().zip(wr).map(|(&a, &b)| a as i32 * b as i32).sum()
        }
        WeightLayout::Blocked(blocked) => {
            // the blocked layout: row o's inputs 4g..4g+4 at ((o / 16 * k / 4 + g) * 16 + o % 16) * 4
            (0..k / 4)
                .map(|g| {
                    let b = ((o / 16 * (k / 4) + g) * 16 + o % 16) * 4;
                    (0..4)
                        .map(|t| qr[g * 4 + t] as i32 * blocked[b + t] as i32)
                        .sum::<i32>()
                })
                .sum()
        }
        WeightLayout::Wide(wide) => {
            let (ob, j) = (o / 16, o % 16);
            let (half, oo) = (j / 8, j % 8);
            (0..k / 4)
                .map(|g| {
                    let base = (ob * (k / 4) + g) * 64 + half * 32 + oo * 2;
                    qr[g * 4] as i32 * wide[base] as i32
                        + qr[g * 4 + 1] as i32 * wide[base + 1] as i32
                        + qr[g * 4 + 2] as i32 * wide[base + 16] as i32
                        + qr[g * 4 + 3] as i32 * wide[base + 17] as i32
                })
                .sum()
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    use super::Packed;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn quantize_range(v: &[f32]) -> (f32, f32) {
        let mut lo = _mm512_setzero_ps();
        let mut hi = lo;
        let chunks = v.chunks_exact(16);
        let tail = chunks.remainder();
        for c in chunks {
            let x = _mm512_loadu_ps(c.as_ptr());
            // A NaN (or equal value) selects the old lane, the second operand.
            lo = _mm512_min_ps(x, lo);
            hi = _mm512_max_ps(x, hi);
        }
        let lo = tail
            .iter()
            .copied()
            .fold(_mm512_reduce_min_ps(lo), f32::min);
        let hi = tail
            .iter()
            .copied()
            .fold(_mm512_reduce_max_ps(hi), f32::max);
        super::quantize_params(lo, hi)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn quantize16(v: *const f32, inv: __m512, zp: __m512) -> __m128i {
        let y = _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(v), inv), zp);
        let y = _mm512_max_ps(y, _mm512_setzero_ps());
        let y = _mm512_min_ps(y, _mm512_set1_ps(255.0));
        _mm512_cvtusepi32_epi8(_mm512_cvttps_epi32(_mm512_add_ps(y, _mm512_set1_ps(0.5))))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    pub(crate) unsafe fn quantize_group(v: &[f32], q: &mut [u8], inv: f32, zp: f32) -> f32 {
        debug_assert_eq!(v.len(), q.len());
        let vi = _mm512_set1_ps(inv);
        let vz = _mm512_set1_ps(zp);
        let zero = _mm512_setzero_si512();
        let mut sums = zero;
        let whole = v.len() / 64 * 64;
        for i in (0..whole).step_by(64) {
            let p = v.as_ptr().add(i);
            let bytes = _mm512_inserti32x4::<3>(
                _mm512_inserti32x4::<2>(
                    _mm512_inserti32x4::<1>(
                        _mm512_castsi128_si512(quantize16(p, vi, vz)),
                        quantize16(p.add(16), vi, vz),
                    ),
                    quantize16(p.add(32), vi, vz),
                ),
                quantize16(p.add(48), vi, vz),
            );
            _mm512_storeu_si512(q.as_mut_ptr().add(i).cast(), bytes);
            sums = _mm512_add_epi64(sums, _mm512_sad_epu8(bytes, zero));
        }
        let tail = v.len() / 16 * 16;
        let mut short_sums = _mm_setzero_si128();
        for i in (whole..tail).step_by(16) {
            let bytes = quantize16(v.as_ptr().add(i), vi, vz);
            _mm_storeu_si128(q.as_mut_ptr().add(i).cast(), bytes);
            short_sums = _mm_add_epi64(short_sums, _mm_sad_epu8(bytes, _mm_setzero_si128()));
        }
        super::quantize_tail(&v[tail..], &mut q[tail..], inv, zp);
        let short_sum = _mm_cvtsi128_si64(_mm_add_epi64(
            short_sums,
            _mm_unpackhi_epi64(short_sums, short_sums),
        )) as u32;
        let tail_sum: u32 = q[tail..].iter().map(|&x| x as u32).sum();
        (_mm512_reduce_add_epi64(sums) as u32 + short_sum + tail_sum) as f32
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn quantize_codes(v: &[f32], q: &mut [u8], inv: f32, zp: f32) {
        let vi = _mm512_set1_ps(inv);
        let vz = _mm512_set1_ps(zp);
        let whole = v.len() / 16 * 16;
        for i in (0..whole).step_by(16) {
            _mm_storeu_si128(
                q.as_mut_ptr().add(i).cast(),
                quantize16(v.as_ptr().add(i), vi, vz),
            );
        }
        super::quantize_tail(&v[whole..], &mut q[whole..], inv, zp);
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn quantize_row(v: &[f32], q: &mut [u8]) -> (f32, i32) {
        let (scale, zp) = quantize_range(v);
        let n = v.len().min(q.len());
        quantize_codes(&v[..n], &mut q[..n], 1.0 / scale, zp);
        (scale, zp as i32)
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn gelu_slice(x: &mut [f32]) {
        gelu_slice_impl::<false>(x);
    }

    /// The GELU pass can track the quantizer's row range without reloading the output.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub(crate) unsafe fn gelu_quantize_range(x: &mut [f32]) -> (f32, f32) {
        gelu_slice_impl::<true>(x)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gelu_slice_impl<const RANGE: bool>(x: &mut [f32]) -> (f32, f32) {
        let mut lo = _mm512_setzero_ps();
        let mut hi = lo;
        let mut chunks = x.chunks_exact_mut(16);
        for c in &mut chunks {
            let v = _mm512_loadu_ps(c.as_ptr());
            let z = _mm512_mul_ps(v, _mm512_set1_ps(std::f32::consts::FRAC_1_SQRT_2));
            // NaN is the second operand so both halves of the clamp propagate it.
            let z = _mm512_min_ps(_mm512_set1_ps(4.0), _mm512_max_ps(_mm512_set1_ps(-4.0), z));
            let z2 = _mm512_mul_ps(z, z);
            let mut p = _mm512_set1_ps(super::ERF_P[0]);
            let mut q = _mm512_set1_ps(super::ERF_Q[0]);
            for &coef in &super::ERF_P[1..] {
                p = _mm512_add_ps(_mm512_mul_ps(z2, p), _mm512_set1_ps(coef));
            }
            for &coef in &super::ERF_Q[1..] {
                q = _mm512_add_ps(_mm512_mul_ps(z2, q), _mm512_set1_ps(coef));
            }
            let erf = _mm512_div_ps(_mm512_mul_ps(z, p), q);
            let h = _mm512_mul_ps(
                _mm512_mul_ps(_mm512_set1_ps(0.5), v),
                _mm512_add_ps(_mm512_set1_ps(1.0), erf),
            );
            if RANGE {
                lo = _mm512_min_ps(h, lo);
                hi = _mm512_max_ps(h, hi);
            }
            _mm512_storeu_ps(c.as_mut_ptr(), h);
        }
        let tail = chunks.into_remainder();
        for v in tail.iter_mut() {
            *v = super::gelu(*v);
        }
        if RANGE {
            let lo = tail
                .iter()
                .copied()
                .fold(_mm512_reduce_min_ps(lo), f32::min);
            let hi = tail
                .iter()
                .copied()
                .fold(_mm512_reduce_max_ps(hi), f32::max);
            super::quantize_params(lo, hi)
        } else {
            (0.0, 0.0)
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn route_logits(x: &[f32], w: &[f32], logits: &mut [f32]) {
        let d = x.len();
        let mut experts = logits.chunks_exact_mut(4);
        for (j, l) in (&mut experts).enumerate() {
            route_dot::<4>(x, w.as_ptr().add(j * 4 * d), l);
        }
        let start = logits.len() / 4 * 4;
        for (j, l) in logits[start..].iter_mut().enumerate() {
            route_dot::<1>(x, w.as_ptr().add((start + j) * d), std::slice::from_mut(l));
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn route_dot<const E: usize>(x: &[f32], w: *const f32, logits: &mut [f32]) {
        let d = x.len();
        let mut acc = [_mm512_setzero_ps(); E];
        let whole = d / 16 * 16;
        for i in (0..whole).step_by(16) {
            let v = _mm512_loadu_ps(x.as_ptr().add(i));
            for (j, a) in acc.iter_mut().enumerate() {
                let wy = _mm512_loadu_ps(w.add(j * d + i));
                *a = _mm512_add_ps(*a, _mm512_mul_ps(v, wy));
            }
        }
        // Sum each expert's lanes in the original order; vectorize across experts, not lanes.
        if E == 4 {
            let mut sums = _mm_setzero_ps();
            macro_rules! four_lanes {
                ($i:literal) => {{
                    let [a, b, c, d] =
                        std::array::from_fn(|j| _mm512_extractf32x4_ps::<$i>(acc[j]));
                    let (ab, cd) = (_mm_unpacklo_ps(a, b), _mm_unpacklo_ps(c, d));
                    sums = _mm_add_ps(sums, _mm_movelh_ps(ab, cd));
                    sums = _mm_add_ps(sums, _mm_movehl_ps(cd, ab));
                    let (ab, cd) = (_mm_unpackhi_ps(a, b), _mm_unpackhi_ps(c, d));
                    sums = _mm_add_ps(sums, _mm_movelh_ps(ab, cd));
                    sums = _mm_add_ps(sums, _mm_movehl_ps(cd, ab));
                }};
            }
            four_lanes!(0);
            four_lanes!(1);
            four_lanes!(2);
            four_lanes!(3);
            _mm_storeu_ps(logits.as_mut_ptr(), sums);
        } else {
            let mut lanes = [0.0; 16];
            _mm512_storeu_ps(lanes.as_mut_ptr(), acc[0]);
            logits[0] = lanes.iter().sum();
        }
        for (j, logit) in logits.iter_mut().enumerate() {
            let tail = std::slice::from_raw_parts(w.add(j * d + whole), d - whole);
            *logit += x[whole..]
                .iter()
                .zip(tail)
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
        }
    }

    #[inline(always)]
    unsafe fn bcast4(q: *const u8) -> i32 {
        (q as *const i32).read_unaligned()
    }

    /// Integer sums for the non-AVX-512 range and reducer paths, one block at a time.
    pub(super) unsafe fn fallback_tile(
        isa: super::Isa,
        q: &[u8],
        m: usize,
        w: &Packed,
        ob: usize,
        a: *mut i32,
    ) -> bool {
        use super::{Isa, WeightLayout};
        let (q, k) = (q.as_ptr(), w.inp);
        let off = ob * k * 16;
        macro_rules! run {
            ($tile:ident, $p:expr) => {
                match m {
                    1 => $tile::<1>(q, k, $p, a, 16),
                    2 => $tile::<2>(q, k, $p, a, 16),
                    3 => $tile::<3>(q, k, $p, a, 16),
                    _ => $tile::<4>(q, k, $p, a, 16),
                }
            };
        }
        match (isa, &w.weights) {
            (Isa::Vnni256, WeightLayout::Blocked(b)) => {
                let wb = b.as_ptr().add(off);
                match m {
                    1 => tile256::<1, 1>(q, k, wb, k * 16, a, 16),
                    2 => tile256::<2, 1>(q, k, wb, k * 16, a, 16),
                    3 => tile256::<3, 1>(q, k, wb, k * 16, a, 16),
                    _ => tile256::<4, 1>(q, k, wb, k * 16, a, 16),
                }
            }
            (Isa::Avx2, WeightLayout::Blocked(b)) => run!(tile_avx2, b.as_ptr().add(off)),
            (Isa::Avx2, WeightLayout::Wide(b)) => run!(tile_avx2w, b.as_ptr().add(off)),
            (Isa::Avx, WeightLayout::Wide(b)) => {
                let wb = b.as_ptr().add(off);
                if m == 1 {
                    tile_avxw::<1>(q, k, wb, a, 16);
                } else {
                    tile_avxw::<2>(q, k, wb, a, 16);
                }
                if m == 3 {
                    tile_avxw::<1>(q.add(2 * k), k, wb, a.add(32), 16);
                }
                if m == 4 {
                    tile_avxw::<2>(q.add(2 * k), k, wb, a.add(32), 16);
                }
            }
            _ => return false,
        }
        true
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
    /// all of the tile's rows. The rows are split evenly over the fewest tiles (2026-11-19: the AR step's 8 rows were a
    /// tile of 6 and one of 2, whose 8 sums waited on the dot product's latency; now 4 and 4, the same passes over the
    /// weights; each sum is still made whole in one tile, the same integers).
    #[inline(always)]
    unsafe fn tiles(rows: usize, blocks: usize, mut tile: impl FnMut(usize, usize, usize, usize)) {
        let mut ob = 0;
        while ob < blocks {
            let (c, max) = if ob + 4 <= blocks && !super::one_block() {
                (4, 6)
            } else {
                (1, 8)
            };
            let count = rows.div_ceil(max);
            let mut r0 = 0;
            for t in 0..count {
                // the first rows % count tiles one row more
                let rt = rows / count + usize::from(t < rows % count);
                tile(rt, r0, ob, c);
                r0 += rt;
            }
            ob += c;
        }
    }

    /// RT rows x C cache-line-aligned output blocks (block c `bstride` bytes after block 0).
    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn tile512<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        acc: *mut i32,
        n: usize,
    ) {
        let a = sums512::<RT, C>(q, k, wb, bstride);
        for (r, ar) in a.iter().enumerate() {
            for (c, ac) in ar.iter().enumerate() {
                _mm512_storeu_si512(acc.add(r * n + c * 16) as *mut _, *ac);
            }
        }
    }

    /// `gemm_u8i8_f32`'s outputs: its epilogue as raw pointers (`xs` one pair a row, or one for all rows).
    #[derive(Clone, Copy)]
    pub(super) struct Epi {
        pub xs: *const (f32, i32),
        pub per_row: bool,
        pub cs: *const i32,
        pub col: *const f32,
        pub scale: f32,
        pub bias: *const f32,
        pub out: usize,
    }

    impl Epi {
        pub(super) fn new(w: &Packed, e: &super::Epilogue<'_>) -> Self {
            Self {
                xs: e.xs.as_ptr(),
                per_row: e.xs.len() > 1,
                cs: w.colsum.as_ptr(),
                col: e.col.map_or(std::ptr::null(), <[f32]>::as_ptr),
                scale: e.scale,
                bias: e.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                out: e.out,
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    #[inline]
    unsafe fn sums512<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
    ) -> [[__m512i; C]; RT] {
        let mut a = [[_mm512_setzero_si512(); C]; RT];
        for kg in 0..k / 4 {
            let mut w = [_mm512_setzero_si512(); C];
            for (c, wc) in w.iter_mut().enumerate() {
                *wc = _mm512_load_si512(wb.add(c * bstride + kg * 64).cast());
            }
            for (r, ar) in a.iter_mut().enumerate() {
                let x = _mm512_set1_epi32(bcast4(q.add(r * k + kg * 4)));
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    *ac = _mm512_dpbusd_epi32(*ac, x, *wc);
                }
            }
        }
        a
    }

    /// RT rows x C output blocks of int8 weights as `tile512`, with the float epilogue.
    /// `col0` is the global column offset for weights and scales; `y` already points to the first column of the tile.
    #[target_feature(enable = "avx512f,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile512_f32<const RT: usize, const C: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        bstride: usize,
        y: *mut f32,
        ystride: usize,
        col0: usize,
        r0: usize,
        e: &Epi,
    ) {
        let a = sums512::<RT, C>(q, k, wb, bstride);
        for c in 0..C {
            let o = col0 + c * 16;
            if o >= e.out {
                break;
            }
            let m: __mmask16 = if o + 16 <= e.out {
                0xFFFF
            } else {
                (1u16 << (e.out - o)) - 1
            };
            let cs = _mm512_loadu_si512(e.cs.add(o) as *const _);
            let col = if e.col.is_null() {
                _mm512_setzero_ps()
            } else {
                _mm512_maskz_loadu_ps(m, e.col.add(o))
            };
            let bias = if e.bias.is_null() {
                _mm512_setzero_ps()
            } else {
                _mm512_maskz_loadu_ps(m, e.bias.add(o))
            };
            for (r, ar) in a.iter().enumerate() {
                let (sx, zx) = *e.xs.add(if e.per_row { r0 + r } else { 0 });
                let iv = _mm512_sub_epi32(ar[c], _mm512_mullo_epi32(_mm512_set1_epi32(zx), cs));
                let f = _mm512_cvtepi32_ps(iv);
                let mut f = if e.col.is_null() {
                    _mm512_mul_ps(f, _mm512_set1_ps(sx * e.scale))
                } else {
                    _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(sx), col), f)
                };
                if !e.bias.is_null() {
                    f = _mm512_add_ps(f, bias);
                }
                _mm512_mask_storeu_ps(y.add(r * ystride + c * 16), m, f);
            }
        }
    }

    /// `gemm_u8i8_f32` with AVX-512 VNNI (`gemm_vnni512`'s tiles).
    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) unsafe fn gemm_vnni512_f32(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        e: &Epi,
        y: *mut f32,
        start: usize,
        end: usize,
    ) {
        let (k, out) = (w.inp, e.out);
        let (qp, wp, bstride) = (q.as_ptr(), blocked.as_ptr(), k * 16);
        tiles(rows, end - start, |rt, r0, ob, c| {
            let ob = ob + start;
            let (q, yr, wb) = (
                qp.add(r0 * k),
                y.add(r0 * out + ob * 16),
                wp.add(ob * bstride),
            );
            if c == 4 {
                by_rows!(
                    rt,
                    tile512_f32,
                    4,
                    q,
                    k,
                    wb,
                    bstride,
                    yr,
                    out,
                    ob * 16,
                    r0,
                    e
                )
            } else {
                by_rows!(
                    rt,
                    tile512_f32,
                    1,
                    q,
                    k,
                    wb,
                    bstride,
                    yr,
                    out,
                    ob * 16,
                    r0,
                    e
                )
            }
        });
    }

    /// Consume a small GEMM tile before moving on to the next columns. The output
    /// buffer is independent of the width of the head and the number of frames.
    #[target_feature(enable = "avx512f,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_vnni512_reduce(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        e: &Epi,
        start: usize,
        masks: &[u16],
        blank: u32,
        keep: usize,
        greedy_only: bool,
        partial: &mut [super::PartialFrameReduce],
    ) {
        let k = w.inp;
        let (qp, wp, bstride) = (q.as_ptr(), blocked.as_ptr(), k * 16);
        let mut tile = [0.0f32; 6 * 64];
        tiles(rows, masks.len(), |rt, r0, i, c| {
            let ob = start + i;
            let (q, wb, o, stride) = (qp.add(r0 * k), wp.add(ob * bstride), ob * 16, c * 16);
            let y = tile.as_mut_ptr();
            if c == 4 {
                by_rows!(rt, tile512_f32, 4, q, k, wb, bstride, y, stride, o, r0, e);
            } else {
                by_rows!(rt, tile512_f32, 1, q, k, wb, bstride, y, stride, o, r0, e);
            }
            let len = (e.out - o).min(stride);
            let blocked = &masks[i..i + c];
            for r in 0..rt {
                let p = &mut partial[r0 + r];
                let v = &tile[r * stride..][..len];
                match len {
                    64 => reduce_blocks::<4>(p, o, v, blocked, blank, keep, greedy_only),
                    16 => reduce_blocks::<1>(p, o, v, blocked, blank, keep, greedy_only),
                    _ => {
                        for (j, (v, &mask)) in v.chunks(16).zip(blocked).enumerate() {
                            p.reduce_scalar(o + j * 16, v, mask, blank, keep, greedy_only);
                        }
                    }
                }
            }
        });
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn reduce_blocks<const C: usize>(
        p: &mut super::PartialFrameReduce,
        o: usize,
        tile: &[f32],
        blocked: &[u16],
        blank: u32,
        keep: usize,
        greedy_only: bool,
    ) {
        let logits: [__m512; C] =
            std::array::from_fn(|j| _mm512_loadu_ps(tile.as_ptr().add(j * 16)));
        let mut highest = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut finite = u16::MAX;
        for &v in &logits {
            highest = _mm512_max_ps(v, highest);
            let bits = _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(i32::MAX));
            finite &=
                _mm512_cmplt_epi32_mask(bits, _mm512_set1_epi32(f32::INFINITY.to_bits() as i32));
        }
        if finite != u16::MAX {
            for (j, (v, &mask)) in tile.chunks_exact(16).zip(blocked).enumerate() {
                p.reduce_scalar(o + j * 16, v, mask, blank, keep, greedy_only);
            }
            return;
        }
        if let Some(&b) = tile.get((blank as usize).wrapping_sub(o)) {
            p.blank_logit = b;
        }
        if _mm512_cmp_ps_mask::<_CMP_GT_OQ>(highest, _mm512_set1_ps(p.max)) != 0 {
            p.raise_max(_mm512_reduce_max_ps(highest), greedy_only);
        }
        let lim = _mm512_set1_ps(p.max - 24.0);
        if !greedy_only && _mm512_cmp_ps_mask::<_CMP_GT_OQ>(highest, lim) != 0 {
            let mut sum = _mm512_setzero_ps();
            let max = _mm512_set1_ps(p.max);
            for &v in &logits {
                if C == 1 || _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, lim) != 0 {
                    sum =
                        _mm512_add_ps(sum, crate::pipeline::lse::exp16_le0(_mm512_sub_ps(v, max)));
                }
            }
            p.sum_exp += _mm512_reduce_add_ps(sum);
        }
        if p.top.len() == keep
            && _mm512_cmp_ps_mask::<_CMP_GT_OQ>(highest, _mm512_set1_ps(p.top[keep - 1].1)) == 0
        {
            return;
        }
        for (j, &v) in logits.iter().enumerate() {
            let mut hits = !blocked[j];
            if p.top.len() == keep {
                // Columns arrive in increasing order, so ties can't displace the last one.
                hits &= _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, _mm512_set1_ps(p.top[keep - 1].1));
            }
            while hits != 0 {
                let lane = j * 16 + hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let x = tile[lane];
                if p.top.len() < keep || x > p.top[keep - 1].1 {
                    p.insert(keep, (o + lane) as u32, x);
                }
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) unsafe fn gemm_vnni512(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), blocked.as_ptr(), k * 16);
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

    /// RT rows x one block of int8 weights with AVX2 alone (no dot product instruction): u8 x i8 pair products into
    /// 16 bits would overflow (255 x 127 x 2), so each activation byte is its low 7 bits plus 128 times its top bit,
    /// the two pair products (at most 127 x 128 x 2 and 1 x 128 x 2) summed by pairs into 32 bits and the second
    /// shifted by 7: the exact integer sums.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile_avx2<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i8,
        acc: *mut i32,
        n: usize,
    ) {
        let ones = _mm256_set1_epi16(1);
        let mut a = [[_mm256_setzero_si256(); 2]; RT];
        for kg in 0..k / 4 {
            let p = wb.add(kg * 64);
            let w = [
                _mm256_loadu_si256(p as *const _),
                _mm256_loadu_si256(p.add(32) as *const _),
            ];
            for (r, ar) in a.iter_mut().enumerate() {
                let x4 = bcast4(q.add(r * k + kg * 4));
                let lo = _mm256_set1_epi32(x4 & 0x7f7f_7f7f);
                let hi = _mm256_set1_epi32((x4 >> 7) & 0x0101_0101);
                for (ac, wc) in ar.iter_mut().zip(&w) {
                    let l = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, *wc), ones);
                    let h = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, *wc), ones);
                    *ac = _mm256_add_epi32(*ac, _mm256_add_epi32(l, _mm256_slli_epi32::<7>(h)));
                }
            }
        }
        for (r, ar) in a.iter().enumerate() {
            _mm256_storeu_si256(acc.add(r * n) as *mut _, ar[0]);
            _mm256_storeu_si256(acc.add(r * n + 8) as *mut _, ar[1]);
        }
    }

    /// The integer sums of RT rows with one block of the 16-bit copy (`widen`; `wb` the block's start): for each 4
    /// inputs, a row's pairs (0, 1) and (2, 3) as two 16-bit values broadcast, 16-bit multiply-adds with the block's 4
    /// registers into its two registers of 8 sums (exact: a pair's products, 255 x 128 at most each, fit 32 bits);
    /// 4 operations for each 32 products against `tile_avx2`'s 7.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn sums_avx2w<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i16,
    ) -> [[__m256i; 2]; RT] {
        let mut a = [[_mm256_setzero_si256(); 2]; RT];
        for kg in 0..k / 4 {
            let p = wb.add(kg * 64);
            let w = [
                _mm256_loadu_si256(p as *const _),
                _mm256_loadu_si256(p.add(16) as *const _),
                _mm256_loadu_si256(p.add(32) as *const _),
                _mm256_loadu_si256(p.add(48) as *const _),
            ];
            for (r, ar) in a.iter_mut().enumerate() {
                let x = q.add(r * k + kg * 4);
                let x01 = _mm256_set1_epi32(((*x.add(1) as i32) << 16) | *x as i32);
                let x23 = _mm256_set1_epi32(((*x.add(3) as i32) << 16) | *x.add(2) as i32);
                ar[0] = _mm256_add_epi32(
                    ar[0],
                    _mm256_add_epi32(_mm256_madd_epi16(x01, w[0]), _mm256_madd_epi16(x23, w[1])),
                );
                ar[1] = _mm256_add_epi32(
                    ar[1],
                    _mm256_add_epi32(_mm256_madd_epi16(x01, w[2]), _mm256_madd_epi16(x23, w[3])),
                );
            }
        }
        a
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile_avx2w<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i16,
        acc: *mut i32,
        n: usize,
    ) {
        for (r, ar) in sums_avx2w::<RT>(q, k, wb).iter().enumerate() {
            _mm256_storeu_si256(acc.add(r * n) as *mut _, ar[0]);
            _mm256_storeu_si256(acc.add(r * n + 8) as *mut _, ar[1]);
        }
    }

    /// `gemm_u8i8` with AVX2 from the 16-bit copy: one block at a time, up to 4 rows a tile.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn gemm_avx2w(
        q: &[u8],
        rows: usize,
        w: &Packed,
        wide: &[i16],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), wide.as_ptr(), k * 16);
        for ob in 0..n / 16 {
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(4);
                let (q, a, wb) = (
                    qp.add(r0 * k),
                    ap.add(r0 * n + ob * 16),
                    wp.add(ob * bstride),
                );
                match rt {
                    1 => tile_avx2w::<1>(q, k, wb, a, n),
                    2 => tile_avx2w::<2>(q, k, wb, a, n),
                    3 => tile_avx2w::<3>(q, k, wb, a, n),
                    _ => tile_avx2w::<4>(q, k, wb, a, n),
                }
                r0 += rt;
            }
        }
    }

    /// `tile512_f32`'s outputs from `sums_avx2w`: each half block's 8 sums turned to the outputs in registers, the
    /// columns past `out` masked.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile_avx2w_f32<const RT: usize>(
        x: *const i32,
        k: usize,
        wb: *const i16,
        y: *mut f32,
        ystride: usize,
        col0: usize,
        r0: usize,
        e: &Epi,
    ) {
        let s = sums_avx2p::<RT>(x, k, wb);
        let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for h in 0..2 {
            let o = col0 + h * 8;
            if o >= e.out {
                break;
            }
            let m = _mm256_cmpgt_epi32(_mm256_set1_epi32((e.out - o).min(8) as i32), lanes);
            let cs = _mm256_loadu_si256(e.cs.add(o) as *const _);
            let col = if e.col.is_null() {
                _mm256_setzero_ps()
            } else {
                _mm256_maskload_ps(e.col.add(o), m)
            };
            let bias = if e.bias.is_null() {
                _mm256_setzero_ps()
            } else {
                _mm256_maskload_ps(e.bias.add(o), m)
            };
            for (r, sr) in s.iter().enumerate() {
                let (sx, zx) = *e.xs.add(if e.per_row { r0 + r } else { 0 });
                let iv = _mm256_sub_epi32(sr[h], _mm256_mullo_epi32(_mm256_set1_epi32(zx), cs));
                let f = _mm256_cvtepi32_ps(iv);
                let mut f = if e.col.is_null() {
                    _mm256_mul_ps(f, _mm256_set1_ps(sx * e.scale))
                } else {
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(sx), col), f)
                };
                if !e.bias.is_null() {
                    f = _mm256_add_ps(f, bias);
                }
                _mm256_maskstore_ps(y.add(r * ystride + o), m, f);
            }
        }
    }

    /// `gemm_u8i8_f32` with AVX2 from the 16-bit copy.
    #[target_feature(enable = "avx2,fma")]
    /// `gemm_avx2w_f32` from the rows' 16-bit pairs (`pairs`: row r's inputs 2i, 2i + 1 as x | y << 16 at
    /// r * k / 2 + i; 2026-11-19: the bytes put together for each broadcast took scalar operations as many as the
    /// multiply-adds).
    pub(super) unsafe fn gemm_avx2w_f32(
        pairs: &[i32],
        rows: usize,
        w: &Packed,
        wide: &[i16],
        e: &Epi,
        y: &mut [f32],
    ) {
        let (n, k, out) = (w.out, w.inp, e.out);
        assert!(
            y.len() >= rows * out
                && w.colsum.len() >= n
                && wide.len() == n * k
                && pairs.len() >= rows * k / 2
        );
        let (xp, yp, wp, bstride) = (pairs.as_ptr(), y.as_mut_ptr(), wide.as_ptr(), k * 16);
        for ob in 0..n / 16 {
            if ob * 16 >= out {
                break;
            }
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(4);
                let (x, yr, wb) = (xp.add(r0 * k / 2), yp.add(r0 * out), wp.add(ob * bstride));
                match rt {
                    1 => tile_avx2w_f32::<1>(x, k, wb, yr, out, ob * 16, r0, e),
                    2 => tile_avx2w_f32::<2>(x, k, wb, yr, out, ob * 16, r0, e),
                    3 => tile_avx2w_f32::<3>(x, k, wb, yr, out, ob * 16, r0, e),
                    _ => tile_avx2w_f32::<4>(x, k, wb, yr, out, ob * 16, r0, e),
                }
                r0 += rt;
            }
        }
    }

    /// `sums_avx2w` from the rows' 16-bit pairs (`gemm_avx2w_f32`'s), each pair broadcast by one load.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn sums_avx2p<const RT: usize>(
        x: *const i32,
        k: usize,
        wb: *const i16,
    ) -> [[__m256i; 2]; RT] {
        let mut a = [[_mm256_setzero_si256(); 2]; RT];
        let kp = k / 2;
        for kg in 0..k / 4 {
            let p = wb.add(kg * 64);
            let w = [
                _mm256_loadu_si256(p as *const _),
                _mm256_loadu_si256(p.add(16) as *const _),
                _mm256_loadu_si256(p.add(32) as *const _),
                _mm256_loadu_si256(p.add(48) as *const _),
            ];
            for (r, ar) in a.iter_mut().enumerate() {
                let x01 = _mm256_set1_epi32(*x.add(r * kp + kg * 2));
                let x23 = _mm256_set1_epi32(*x.add(r * kp + kg * 2 + 1));
                ar[0] = _mm256_add_epi32(
                    ar[0],
                    _mm256_add_epi32(_mm256_madd_epi16(x01, w[0]), _mm256_madd_epi16(x23, w[1])),
                );
                ar[1] = _mm256_add_epi32(
                    ar[1],
                    _mm256_add_epi32(_mm256_madd_epi16(x01, w[2]), _mm256_madd_epi16(x23, w[3])),
                );
            }
        }
        a
    }

    /// `sums_avxw` from the rows' 16-bit pairs (`gemm_avx2w_f32`'s).
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    #[inline]
    unsafe fn sums_avxp<const RT: usize>(
        x: *const i32,
        k: usize,
        wb: *const i16,
    ) -> [[__m128i; 4]; RT] {
        let mut a = [[_mm_setzero_si128(); 4]; RT];
        let kp = k / 2;
        for kg in 0..k / 4 {
            let p = wb.add(kg * 64);
            let ld = |i: usize| _mm_loadu_si128(p.add(i * 8) as *const _);
            for (r, ar) in a.iter_mut().enumerate() {
                let x01 = _mm_set1_epi32(*x.add(r * kp + kg * 2));
                let x23 = _mm_set1_epi32(*x.add(r * kp + kg * 2 + 1));
                for (h, ah) in ar.iter_mut().enumerate() {
                    let (p01, p23) = ((h / 2) * 4 + h % 2, (h / 2) * 4 + 2 + h % 2);
                    let s =
                        _mm_add_epi32(_mm_madd_epi16(x01, ld(p01)), _mm_madd_epi16(x23, ld(p23)));
                    *ah = _mm_add_epi32(*ah, s);
                }
            }
        }
        a
    }

    /// The AVX level's integer sums (128-bit integers): RT rows with one block of the 16-bit copy as `sums_avx2w`, its
    /// 16 outputs in four registers of 4 (each 16-value run of the copy is two registers: its outputs 0-3, then 4-7).
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    #[inline]
    unsafe fn sums_avxw<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i16,
    ) -> [[__m128i; 4]; RT] {
        let mut a = [[_mm_setzero_si128(); 4]; RT];
        for kg in 0..k / 4 {
            let p = wb.add(kg * 64);
            let ld = |i: usize| _mm_loadu_si128(p.add(i * 8) as *const _);
            for (r, ar) in a.iter_mut().enumerate() {
                let x = q.add(r * k + kg * 4);
                let x01 = _mm_set1_epi32(((*x.add(1) as i32) << 16) | *x as i32);
                let x23 = _mm_set1_epi32(((*x.add(3) as i32) << 16) | *x.add(2) as i32);
                // the copy's runs: outputs 0-7 with inputs 0, 1 (registers 0, 1), with 2, 3 (2, 3); 8-15 likewise
                for (h, ah) in ar.iter_mut().enumerate() {
                    let (p01, p23) = ((h / 2) * 4 + h % 2, (h / 2) * 4 + 2 + h % 2);
                    let s =
                        _mm_add_epi32(_mm_madd_epi16(x01, ld(p01)), _mm_madd_epi16(x23, ld(p23)));
                    *ah = _mm_add_epi32(*ah, s);
                }
            }
        }
        a
    }

    #[target_feature(enable = "avx,ssse3,sse4.1")]
    unsafe fn tile_avxw<const RT: usize>(
        q: *const u8,
        k: usize,
        wb: *const i16,
        acc: *mut i32,
        n: usize,
    ) {
        for (r, ar) in sums_avxw::<RT>(q, k, wb).iter().enumerate() {
            for (h, ah) in ar.iter().enumerate() {
                _mm_storeu_si128(acc.add(r * n + h * 4) as *mut _, *ah);
            }
        }
    }

    /// `gemm_u8i8` at the AVX level from the 16-bit copy: one block at a time, up to 2 rows a tile.
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    pub(super) unsafe fn gemm_avxw(
        q: &[u8],
        rows: usize,
        w: &Packed,
        wide: &[i16],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), wide.as_ptr(), k * 16);
        for ob in 0..n / 16 {
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(2);
                let (q, a, wb) = (
                    qp.add(r0 * k),
                    ap.add(r0 * n + ob * 16),
                    wp.add(ob * bstride),
                );
                if rt == 1 {
                    tile_avxw::<1>(q, k, wb, a, n);
                } else {
                    tile_avxw::<2>(q, k, wb, a, n);
                }
                r0 += rt;
            }
        }
    }

    /// `tile512_f32`'s outputs at the AVX level: each 4 outputs' sums turned to the outputs in registers, the columns
    /// past `out` masked.
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile_avxw_f32<const RT: usize>(
        x: *const i32,
        k: usize,
        wb: *const i16,
        y: *mut f32,
        ystride: usize,
        col0: usize,
        r0: usize,
        e: &Epi,
    ) {
        let s = sums_avxp::<RT>(x, k, wb);
        let lanes = _mm_setr_epi32(0, 1, 2, 3);
        for h in 0..4 {
            let o = col0 + h * 4;
            if o >= e.out {
                break;
            }
            let m = _mm_cmpgt_epi32(_mm_set1_epi32((e.out - o).min(4) as i32), lanes);
            let cs = _mm_loadu_si128(e.cs.add(o) as *const _);
            let col = if e.col.is_null() {
                _mm_setzero_ps()
            } else {
                _mm_maskload_ps(e.col.add(o), m)
            };
            let bias = if e.bias.is_null() {
                _mm_setzero_ps()
            } else {
                _mm_maskload_ps(e.bias.add(o), m)
            };
            for (r, sr) in s.iter().enumerate() {
                let (sx, zx) = *e.xs.add(if e.per_row { r0 + r } else { 0 });
                let iv = _mm_sub_epi32(sr[h], _mm_mullo_epi32(_mm_set1_epi32(zx), cs));
                let f = _mm_cvtepi32_ps(iv);
                let mut f = if e.col.is_null() {
                    _mm_mul_ps(f, _mm_set1_ps(sx * e.scale))
                } else {
                    _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(sx), col), f)
                };
                if !e.bias.is_null() {
                    f = _mm_add_ps(f, bias);
                }
                _mm_maskstore_ps(y.add(r * ystride + o), m, f);
            }
        }
    }

    /// `gemm_u8i8_f32` at the AVX level from the 16-bit copy and the rows' 16-bit pairs (`gemm_avx2w_f32`'s).
    #[target_feature(enable = "avx,ssse3,sse4.1")]
    pub(super) unsafe fn gemm_avxw_f32(
        pairs: &[i32],
        rows: usize,
        w: &Packed,
        wide: &[i16],
        e: &Epi,
        y: &mut [f32],
    ) {
        let (n, k, out) = (w.out, w.inp, e.out);
        assert!(
            y.len() >= rows * out
                && w.colsum.len() >= n
                && wide.len() == n * k
                && pairs.len() >= rows * k / 2
        );
        let (xp, yp, wp, bstride) = (pairs.as_ptr(), y.as_mut_ptr(), wide.as_ptr(), k * 16);
        for ob in 0..n / 16 {
            if ob * 16 >= out {
                break;
            }
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(2);
                let (x, yr, wb) = (xp.add(r0 * k / 2), yp.add(r0 * out), wp.add(ob * bstride));
                if rt == 1 {
                    tile_avxw_f32::<1>(x, k, wb, yr, out, ob * 16, r0, e);
                } else {
                    tile_avxw_f32::<2>(x, k, wb, yr, out, ob * 16, r0, e);
                }
                r0 += rt;
            }
        }
    }

    /// `gemm_u8i8` with AVX2 alone: one block at a time, up to 4 rows a tile.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn gemm_avx2(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), blocked.as_ptr(), k * 16);
        for ob in 0..n / 16 {
            let mut r0 = 0;
            while r0 < rows {
                let rt = (rows - r0).min(4);
                let (q, a, wb) = (
                    qp.add(r0 * k),
                    ap.add(r0 * n + ob * 16),
                    wp.add(ob * bstride),
                );
                match rt {
                    1 => tile_avx2::<1>(q, k, wb, a, n),
                    2 => tile_avx2::<2>(q, k, wb, a, n),
                    3 => tile_avx2::<3>(q, k, wb, a, n),
                    _ => tile_avx2::<4>(q, k, wb, a, n),
                }
                r0 += rt;
            }
        }
    }

    #[target_feature(enable = "avx2,avxvnni")]
    pub(super) unsafe fn gemm_vnni256(
        q: &[u8],
        rows: usize,
        w: &Packed,
        blocked: &[i8],
        acc: &mut [i32],
    ) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), blocked.as_ptr(), k * 16);
        tiles256(rows, n / 16, |rt, r0, ob, c| {
            let (q, a, wb) = (
                qp.add(r0 * k),
                ap.add(r0 * n + ob * 16),
                wp.add(ob * bstride),
            );
            by_rows256!(rt, c, tile256, q, k, wb, bstride, a, n)
        });
    }
}

/// Per row: u8 codes of `v` (asymmetric, the range widened to hold 0 exactly), its scale and zero point.
#[inline(always)]
pub(crate) fn quantize_row(v: &[f32], q: &mut [u8]) -> (f32, i32) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx512() {
        return unsafe { x86::quantize_row(v, q) };
    }
    quantize_row_scalar(v, q)
}

#[inline(always)]
pub(super) fn quantize_row_scalar(v: &[f32], q: &mut [u8]) -> (f32, i32) {
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
    let (scale, zp) = quantize_params(lo, hi);
    quantize_tail(v, q, 1.0 / scale, zp);
    (scale, zp as i32)
}

#[inline(always)]
fn quantize_params(lo: f32, hi: f32) -> (f32, f32) {
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let zp = ((-lo / scale).clamp(0.0, 255.0) + 0.5).floor();
    (scale, zp)
}

#[inline(always)]
fn quantize_tail(v: &[f32], q: &mut [u8], inv: f32, zp: f32) {
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
}

/// erf for exact GELU, branch-free f32 (vectorizes): one rational function on [-4, 4], an odd polynomial of degree 13
/// over an even one of degree 8 (the f32 erf of Eigen and XLA), and +-1 to f32 precision outside; 12 multiply-adds
/// and a division, 3.5x faster in the experts' loop than the exponential form it replaced (Abramowitz & Stegun 7.1.26
/// with a polynomial e^x), within 5e-7 of it.
#[inline(always)]
fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let p = ERF_P[1..].iter().fold(ERF_P[0], |p, &c| x2 * p + c);
    let q = ERF_Q[1..].iter().fold(ERF_Q[0], |q, &c| x2 * q + c);
    x * p / q
}

#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

const ERF_P: [f32; 7] = [
    -2.726_142_3e-10,
    2.770_681_4e-8,
    -2.101_024e-6,
    -5.692_506_4e-5,
    -7.349_906_3e-4,
    -2.954_599_8e-3,
    -1.609_603_3e-2,
];
const ERF_Q: [f32; 5] = [
    -1.456_607_2e-5,
    -2.133_740_6e-4,
    -1.682_827e-3,
    -7.373_329_2e-3,
    -1.426_473_9e-2,
];

/// In-place GELU, respecting the engine's CPU cap.
pub(crate) fn gelu_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx512() {
        return unsafe { x86::gelu_slice(x) };
    }
    for v in x {
        *v = gelu(*v);
    }
}

/// sum a[i] * b[i] in 16 partial sums (vectorizes).
#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) struct Scratch {
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

/// A token-routed mixture of experts on int8 Linears (the research export's `TokenMoE`: per frame the router's fp32
/// logits, the top-`k` experts with their softmax renormalised over the k, each expert's two Linears with exact GELU
/// between them, symmetric int8 per output row, on the frames routed to it; the weighted outputs summed), prepacked
/// for the engine's level. A model trained without QAT has them (ONNX Runtime's dynamic int8 export: a reading
/// corrector's).
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

    /// y (frames, d) for x (frames, d): every frame through its top-k experts, one expert after another.
    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32], s: &mut Scratch) {
        #[cfg(target_arch = "x86_64")]
        match self.isa {
            // AVX-512 VNNI CPUs: the routing, quantization and GELU loops 16 wide
            Isa::Vnni512 => return unsafe { self.forward_avx512(x, y, s) },
            // every AVX-VNNI CPU has AVX2 and FMA, and an AVX2 one: 8 wide
            Isa::Vnni256 | Isa::Avx2 => return unsafe { self.forward_avx2(x, y, s) },
            Isa::Avx | Isa::Dot | Isa::Plain => {}
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
        let d = self.d;
        let n = x.len() / d;
        y.fill(0.0);
        route(x, &self.router, self.e, d, self.k, &mut s.rows);
        // the frames' u8 codes, once for every expert they go to
        s.xq.resize(n * d, 0);
        s.xs.clear();
        for r in 0..n {
            let qs = quantize_row(&x[r * d..(r + 1) * d], &mut s.xq[r * d..(r + 1) * d]);
            s.xs.push(qs);
        }
        let Scratch {
            rows,
            xq,
            xs,
            out,
            es,
        } = s;
        for (ex, routed) in rows.iter().enumerate() {
            if !routed.is_empty() {
                out.resize(routed.len() * d, 0.0);
                self.expert(ex, routed, xq, xs, out, es);
                add_routed(y, routed, out, d);
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
            Isa::Vnni256 | Isa::Avx2 => {
                return unsafe { self.expert_avx2(ex, routed, xq, xs, out, s) }
            }
            Isa::Avx | Isa::Dot | Isa::Plain => {}
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
                *hv = sx * sc * (a - zx * c) as f32 + b;
            }
        }
        s.hq.resize(m * f, 0);
        s.hs.clear();
        for i in 0..m {
            let h = &mut s.h[i * f..(i + 1) * f];
            gelu_slice(h);
            let qs = quantize_row(h, &mut s.hq[i * f..(i + 1) * f]);
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

/// Route a contiguous slice of `rows` rows of `x` (`rows * d`) into `out` (`rows * k`), writing
/// each row's top-`k` `(expert_id, weight)` pairs with zero heap allocation (thread-safe for
/// disjoint row ranges).
#[allow(dead_code)]
pub(crate) fn route_rows_slice(
    x: &[f32],
    router: &[f32],
    e: usize,
    d: usize,
    k: usize,
    out: &mut [(usize, f32)],
) {
    assert!((1..=e).contains(&k));
    assert!(router.len() >= span(e, d));
    let mut stack_logits = [0.0; 64];
    let mut heap_logits;
    let logits = if e <= stack_logits.len() {
        &mut stack_logits[..e]
    } else {
        heap_logits = vec![0.0; e];
        &mut heap_logits[..]
    };
    #[cfg(target_arch = "x86_64")]
    let avx512 = crate::cpu::avx512();
    for (r, xr) in x.chunks_exact(d).enumerate() {
        let scalar_logits = |logits: &mut [f32]| {
            for (l, wr) in logits.iter_mut().zip(router.chunks_exact(d)) {
                *l = dot(xr, wr);
            }
        };
        #[cfg(target_arch = "x86_64")]
        if avx512 {
            unsafe { x86::route_logits(xr, router, logits) };
        } else {
            scalar_logits(logits);
        }
        #[cfg(not(target_arch = "x86_64"))]
        scalar_logits(logits);

        let top = &mut out[r * k..(r + 1) * k];
        let mut used = 0u64;
        let mut z = 0.0;
        let mut m = 0.0;
        for i in 0..k {
            let mut best = None;
            for j in 0..e {
                if if e <= 64 {
                    used & (1u64 << j) != 0
                } else {
                    top[..i].iter().any(|&(id, _)| id == j)
                } {
                    continue;
                }
                match best {
                    Some(b) if logits[b] >= logits[j] => {}
                    _ => best = Some(j),
                }
            }
            let j = best.unwrap();
            if e <= 64 {
                used |= 1u64 << j;
            }
            if i == 0 {
                m = logits[j];
            }
            let p = (logits[j] - m).exp();
            top[i] = (j, p);
            z += p;
        }
        for (_, p) in top.iter_mut() {
            *p /= z;
        }
    }
}

/// Each frame of `x` (n, d) routed to the top-`k` of its router logits (`router` (e, d)), with the softmax over the k
/// as its weights: `rows[j]` the (frame, weight) pairs of expert j, in frame order (cleared first).
#[inline(always)]
pub(crate) fn route(
    x: &[f32],
    router: &[f32],
    e: usize,
    d: usize,
    k: usize,
    rows: &mut Vec<Vec<(usize, f32)>>,
) {
    assert!((1..=e).contains(&k));
    assert!(router.len() >= span(e, d));
    rows.resize_with(e, Vec::new);
    rows.iter_mut().for_each(Vec::clear);
    let mut stack_logits = [0.0; 64];
    let mut stack_top = [(0, 0.0); 16];
    let mut heap_logits;
    let mut heap_top;
    let logits = if e <= stack_logits.len() {
        &mut stack_logits[..e]
    } else {
        heap_logits = vec![0.0; e];
        &mut heap_logits[..]
    };
    let top = if k <= stack_top.len() {
        &mut stack_top[..k]
    } else {
        heap_top = vec![(0, 0.0); k];
        &mut heap_top[..]
    };
    #[cfg(target_arch = "x86_64")]
    let avx512 = crate::cpu::avx512();
    for (r, xr) in x.chunks_exact(d).enumerate() {
        let scalar_logits = |logits: &mut [f32]| {
            for (l, wr) in logits.iter_mut().zip(router.chunks_exact(d)) {
                *l = dot(xr, wr);
            }
        };
        #[cfg(target_arch = "x86_64")]
        if avx512 {
            unsafe { x86::route_logits(xr, router, logits) };
        } else {
            scalar_logits(logits);
        }
        #[cfg(not(target_arch = "x86_64"))]
        scalar_logits(logits);

        // Fast path for small routers: a bit per selected expert. Keep the original
        // comparison (including its behavior for ties, infinities and NaNs).
        let mut used = 0u64;
        let mut z = 0.0;
        let mut m = 0.0;
        for i in 0..k {
            let mut best = None;
            for j in 0..e {
                if if e <= 64 {
                    used & (1u64 << j) != 0
                } else {
                    top[..i].iter().any(|&(id, _)| id == j)
                } {
                    continue;
                }
                match best {
                    Some(b) if logits[b] >= logits[j] => {}
                    _ => best = Some(j),
                }
            }
            let j = best.unwrap();
            if e <= 64 {
                used |= 1u64 << j;
            }
            if i == 0 {
                m = logits[j];
            }
            let p = (logits[j] - m).exp();
            top[i] = (j, p);
            z += p;
        }
        for &(j, p) in top.iter() {
            rows[j].push((r, p / z));
        }
    }
}

/// y[r] += w * out[i] for the i-th routed frame (r, w) of an expert.
#[inline(always)]
pub(crate) fn add_routed(y: &mut [f32], routed: &[(usize, f32)], out: &[f32], d: usize) {
    for (i, &(r, wt)) in routed.iter().enumerate() {
        for (yv, &v) in y[r * d..(r + 1) * d]
            .iter_mut()
            .zip(&out[i * d..(i + 1) * d])
        {
            *yv += wt * v;
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
    fn simd_quantization_gelu_and_routing_keep_scalar_bits() {
        let mut rng = Rng(438147);
        for len in [1, 15, 16, 17, 48, 63, 64, 65, 192, 389] {
            let mut x: Vec<_> = (0..len).map(|_| rng.next() * 25.0 - 12.5).collect();
            for special in [
                0.0,
                -0.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(1),
            ] {
                x[len / 3] = special;
                let mut want = vec![1; len];
                let mut got = vec![2; len];
                let a = quantize_row_scalar(&x, &mut want);
                let b = quantize_row(&x, &mut got);
                assert_eq!((a.0.to_bits(), a.1), (b.0.to_bits(), b.1));
                assert_eq!(got, want, "len {len}, special {special}");
                let mut h = x.clone();
                gelu_slice(&mut h);
                for (&x, y) in x.iter().zip(h) {
                    let v = gelu(x);
                    assert!(if v.is_nan() {
                        y.is_nan()
                    } else {
                        v.to_bits() == y.to_bits()
                    });
                }
            }
        }
        for (e, d, k) in [(16, 192, 2), (7, 36, 3), (65, 19, 18)] {
            let x: Vec<_> = (0..3 * d).map(|_| rng.next() * 2.0 - 1.0).collect();
            let mut router: Vec<_> = (0..e * d).map(|_| rng.next() * 2.0 - 1.0).collect();
            router.copy_within(0..d, d); // tie between the first two experts
            for with_nan in [false, true] {
                if with_nan {
                    router[4 * d] = f32::NAN;
                }
                let mut want = vec![Vec::new(); e];
                for (r, xr) in x.chunks_exact(d).enumerate() {
                    let logits: Vec<_> = router.chunks_exact(d).map(|w| dot(xr, w)).collect();
                    let mut top = Vec::new();
                    for _ in 0..k {
                        top.push(
                            (0..e)
                                .filter(|id| !top.contains(id))
                                .reduce(|a, b| if logits[a] >= logits[b] { a } else { b })
                                .unwrap(),
                        );
                    }
                    let m = logits[top[0]];
                    let z: f32 = top.iter().map(|&id| (logits[id] - m).exp()).sum();
                    for id in top {
                        want[id].push((r, ((logits[id] - m).exp() / z).to_bits()));
                    }
                }
                let mut got = Vec::new();
                route(&x, &router, e, d, k, &mut got);
                let got: Vec<Vec<_>> = got
                    .iter()
                    .map(|p| p.iter().map(|&(r, w)| (r, w.to_bits())).collect())
                    .collect();
                for (a, b) in got.iter().zip(want) {
                    assert_eq!(a.len(), b.len());
                    for (&(r, a), (s, b)) in a.iter().zip(b) {
                        assert_eq!(r, s);
                        assert!(
                            a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn column_gemm_and_fused_reducer_match_full_logits() {
        let mut rng = Rng(17493);
        for (rows, out, inp, packed) in [
            (7, 517usize, 28, true),
            (3, 8006, 16, true),
            (5, 261, 13, false),
        ] {
            let n = if packed {
                out.next_multiple_of(16)
            } else {
                out
            };
            let mut raw: Vec<_> = (0..n * inp)
                .map(|_| (rng.next() * 255.0 - 128.0) as i8)
                .collect();
            raw[..2 * inp].fill(0);
            let q: Vec<_> = (0..rows * inp)
                .map(|_| (rng.next() * 255.0) as u8)
                .collect();
            let xs: Vec<_> = (0..rows)
                .map(|i| (0.01 + rng.next() * 0.02, 100 + i as i32))
                .collect();
            let col: Vec<_> = (0..out).map(|_| 0.002 + rng.next() * 0.003).collect();
            let mut bias: Vec<_> = (0..out).map(|_| rng.next() * 2.0 - 1.0).collect();
            bias[0] = 29.0;
            bias[1] = 29.0; // same logit, lower ID wins
            bias[out / 2] = 35.0; // a blacklisted max still contributes to the LSE
            bias[out - 1] = 27.0; // the masked tail is eligible for top-k
            let blank = (out - 2) as u32;
            let blocked = [17, (out / 2) as u32, (out + 13) as u32];
            let stack = Epilogue {
                xs: &xs,
                col: Some(&col),
                scale: 0.08,
                bias: Some(&bias),
                out,
            };
            for isa in [
                Isa::Plain,
                Isa::Avx,
                Isa::Avx2,
                Isa::Vnni256,
                Isa::Vnni512,
                Isa::Dot,
            ] {
                if !supported(isa) {
                    continue;
                }
                let w = Packed::new_for(&raw, n, inp, packed, isa);
                let mut logits = vec![0.0; rows * out];
                gemm_u8i8_f32(isa, &q, rows, &w, &stack, &mut Vec::new(), &mut logits);
                for splits in [1, 4, 8] {
                    let mut separate = vec![f32::NAN; rows * out];
                    let mut parts = Vec::new();
                    let mut greedy = Vec::new();
                    // Deliberately merge the parts in descending column order.
                    for i in (0..splits).rev() {
                        let start = i * w.blocks() / splits;
                        let end = (i + 1) * w.blocks() / splits;
                        unsafe {
                            gemm_u8i8_f32_cols_ptr(
                                isa,
                                &q,
                                rows,
                                &w,
                                &stack,
                                separate.as_mut_ptr(),
                                start,
                                end,
                            )
                        };
                        if i == splits - 1 {
                            for row in separate.chunks_exact(out) {
                                assert!(row[..start * 16].iter().all(|v| v.is_nan()));
                            }
                        }
                        parts.push(gemm_u8i8_reduce_cols(
                            isa, &q, rows, &w, &stack, start, end, &blocked, blank, 12, false,
                        ));
                        let p = gemm_u8i8_reduce_cols(
                            isa, &q, rows, &w, &stack, start, end, &blocked, blank, 12, true,
                        );
                        assert!(p.iter().all(|p| p.top.len() <= 2 && p.sum_exp == 0.0));
                        greedy.push(p);
                    }
                    assert!(
                        logits
                            .iter()
                            .zip(&separate)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{isa:?} splits={splits}"
                    );
                    let result = merge_partial_reduce(&parts, &blocked, blank, 12, false).unwrap();
                    let greedy = merge_partial_reduce(&greedy, &blocked, blank, 12, true).unwrap();
                    for (r, row) in logits.chunks_exact(out).enumerate() {
                        let lse = crate::pipeline::lse::peaked(row);
                        let mut want: Vec<_> = row
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| !blocked.contains(&(*i as u32)))
                            .map(|(i, v)| (i as u32, *v))
                            .collect();
                        want.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                        let (top, blank_lp, id, top1, top2) = &result[r];
                        let check = |a: f32, b: f32| {
                            assert!(
                                (a - b).abs() <= 1e-5,
                                "{isa:?} n={n} splits={splits}: {a} vs {b}"
                            )
                        };
                        assert_eq!(top.len(), 12);
                        for (&(id, val), &(j, want)) in top.iter().zip(&want) {
                            assert_eq!(id, j);
                            check(val, want - lse);
                        }
                        assert_eq!(*id, want[0].0);
                        check(*top1, want[0].1 - lse);
                        check(*top2, want[1].1 - lse);
                        check(*blank_lp, row[blank as usize] - lse);
                        assert_eq!(greedy[r].2, want[0].0);
                        assert_eq!(greedy[r].3, 0.0);
                        check(greedy[r].4, want[1].1 - want[0].1);
                        check(greedy[r].1, row[blank as usize] - want[0].1);
                    }
                }
                let mut bad_bias = bias.clone();
                bad_bias[17] = f32::NAN; // even when blacklisted, this requests fallback
                bad_bias[out - 1] = f32::INFINITY;
                let stack = Epilogue {
                    bias: Some(&bad_bias),
                    ..stack
                };
                let bad = gemm_u8i8_reduce_cols(
                    isa,
                    &q,
                    rows,
                    &w,
                    &stack,
                    0,
                    w.blocks(),
                    &blocked,
                    blank,
                    12,
                    false,
                );
                assert!(bad
                    .iter()
                    .all(|p| p.has_non_finite && p.max == f32::INFINITY));
                assert!(merge_partial_reduce(&[bad], &blocked, blank, 12, false).is_none());
            }
        }
    }

    #[test]
    #[ignore]
    fn timing_head() {
        let isa = detect_isa();
        if isa == Isa::Plain {
            return;
        }
        let (rows, out, inp) = (30usize, 8006usize, 192usize);
        let n = out.next_multiple_of(16);
        let mut rng = Rng(41);
        let w: Vec<_> = (0..n * inp)
            .map(|_| (rng.next() * 255.0 - 128.0) as i8)
            .collect();
        let w = Packed::new(&w, n, inp, true);
        let q: Vec<_> = (0..rows * inp)
            .map(|_| (rng.next() * 255.0) as u8)
            .collect();
        let xs = vec![(0.012, 128); rows];
        let bias: Vec<_> = (0..out)
            .map(|i| {
                if i == 256 {
                    35.0
                } else {
                    rng.next() * 2.0 - 1.0
                }
            })
            .collect();
        let e = Epilogue {
            xs: &xs,
            col: None,
            scale: 0.001,
            bias: Some(&bias),
            out,
        };
        let mut acc = Vec::new();
        let mut y = vec![0.0; rows * out];
        for (variant, greedy) in [
            ("logits", false),
            ("logits + top-k", false),
            ("fused top-k", false),
            ("fused greedy", true),
        ] {
            let mut ns = [0; 500];
            for t in &mut ns {
                let start = std::time::Instant::now();
                if variant.starts_with("logits") {
                    gemm_u8i8_f32(isa, &q, rows, &w, &e, &mut acc, &mut y);
                    std::hint::black_box(&y);
                    if variant == "logits + top-k" {
                        std::hint::black_box(crate::pipeline::proposal::log_probs_from_logits(
                            &y,
                            rows,
                            out,
                            &[],
                            0,
                            16,
                        ));
                    }
                } else {
                    let p = gemm_u8i8_reduce_cols(
                        isa,
                        &q,
                        rows,
                        &w,
                        &e,
                        0,
                        w.blocks(),
                        &[],
                        0,
                        16,
                        greedy,
                    );
                    std::hint::black_box(merge_partial_reduce(&[p], &[], 0, 16, greedy));
                }
                *t = start.elapsed().as_nanos();
            }
            ns.sort_unstable();
            println!(
                "int8 head {rows}x{inp} -> {out} ({isa:?}, {variant}): {:.3} us",
                ns[ns.len() / 2] as f64 / 1000.0
            );
        }
    }

    /// `gemm_u8i8_f32` on the CPU's ISA against its plain pass, the same bits: both epilogues (one scale with a
    /// bias or none, a scale an output with a bias), one pair of scales for all rows or one a row, outputs not in 16s
    /// (the last block masked), 1 to 13 rows.
    #[test]
    fn gemm_f32_gives_the_plain_epilogues_bits() {
        let mut rng = Rng(23);
        let isa = detect_isa();
        for (out, inp) in [(70usize, 64usize), (8006, 192), (16, 4), (33, 96)] {
            let stride = out.div_ceil(16) * 16;
            let w: Vec<f32> = (0..out * inp).map(|_| rng.next()).collect();
            let (mut q8, _) = quantize_rows(&w, out, inp);
            q8.resize(stride * inp, 0);
            let packed = Packed::new(&q8, stride, inp, isa != Isa::Plain);
            let wide = Packed::new_wide(&q8, stride, inp);
            let col: Vec<f32> = (0..out).map(|_| 0.01 + rng.next().abs() * 0.02).collect();
            let bias: Vec<f32> = (0..out).map(|_| rng.next()).collect();
            for rows in [1usize, 6, 7, 13] {
                let q: Vec<u8> = (0..rows * inp)
                    .map(|_| ((rng.next() + 1.0) * 127.5) as u8)
                    .collect();
                let xs: Vec<(f32, i32)> = (0..rows)
                    .map(|_| {
                        (
                            0.02 + rng.next().abs() * 0.01,
                            (rng.next().abs() * 255.0) as i32,
                        )
                    })
                    .collect();
                let cases: [Epilogue; 4] = [
                    Epilogue {
                        xs: &xs[..1],
                        col: None,
                        scale: 0.013,
                        bias: None,
                        out,
                    },
                    Epilogue {
                        xs: &xs[..1],
                        col: None,
                        scale: 0.013,
                        bias: Some(&bias),
                        out,
                    },
                    Epilogue {
                        xs: &xs,
                        col: Some(&col),
                        scale: 0.0,
                        bias: Some(&bias),
                        out,
                    },
                    Epilogue {
                        xs: &xs,
                        col: Some(&col),
                        scale: 0.0,
                        bias: None,
                        out,
                    },
                ];
                for (i, e) in cases.iter().enumerate() {
                    let (mut got, mut want) = (vec![f32::NAN; rows * out], vec![0f32; rows * out]);
                    let (mut a1, mut a2) = (Vec::new(), Vec::new());
                    gemm_u8i8_f32(isa, &q, rows, &packed, e, &mut a1, &mut got);
                    gemm_u8i8_f32(Isa::Plain, &q, rows, &packed, e, &mut a2, &mut want);
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "out {out} inp {inp} rows {rows} case {i}"
                    );
                    // the AVX2 level's kernel from the 16-bit copy, where this CPU has AVX2
                    #[cfg(target_arch = "x86_64")]
                    if isa != Isa::Plain && supported(Isa::Avx2) {
                        let mut got = vec![f32::NAN; rows * out];
                        gemm_u8i8_f32(Isa::Avx2, &q, rows, &wide, e, &mut a1, &mut got);
                        assert_eq!(
                            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "AVX2 from 16 bits, out {out} inp {inp} rows {rows} case {i}"
                        );
                    }
                    // and the AVX level's (128-bit integers)
                    #[cfg(target_arch = "x86_64")]
                    if isa != Isa::Plain && supported(Isa::Avx) {
                        let mut got = vec![f32::NAN; rows * out];
                        gemm_u8i8_f32(Isa::Avx, &q, rows, &wide, e, &mut a1, &mut got);
                        assert_eq!(
                            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "AVX from 16 bits, out {out} inp {inp} rows {rows} case {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_isa_gives_the_same_integer_sums() {
        let mut rng = Rng(7);
        let (out, inp) = (80, 40);
        let w: Vec<i8> = (0..out * inp).map(|_| (rng.next() * 127.0) as i8).collect();
        let (raw, p, pw) = (
            Packed::new(&w, out, inp, false),
            Packed::new_blocked(&w, out, inp),
            Packed::new_wide(&w, out, inp),
        );
        assert!(
            matches!(raw.weights, WeightLayout::Raw(_))
                && matches!(p.weights, WeightLayout::Blocked(_))
                && matches!(pw.weights, WeightLayout::Wide(_)),
            "one layout each"
        );
        assert!(matches!(
            Packed::new_for(&w, out, inp, true, Isa::Avx2).weights,
            WeightLayout::Wide(_)
        ));
        assert!(matches!(
            Packed::new_for(&w, out, inp, true, Isa::Avx).weights,
            WeightLayout::Wide(_)
        ));
        assert!(matches!(
            Packed::new_for(&w, out, inp, true, Isa::Vnni512).weights,
            WeightLayout::Blocked(_)
        ));
        assert!(matches!(
            Packed::new_for(&w, out, inp, true, Isa::Plain).weights,
            WeightLayout::Blocked(_)
        ));
        for rows in [1, 3, 8, 9, 13] {
            let q: Vec<u8> = (0..rows * inp)
                .map(|_| ((rng.next() + 1.0) * 127.0) as u8)
                .collect();
            let mut want = vec![0i32; rows * out];
            gemm_plain(&q, rows, &raw, &mut want);
            // the plain path on the blocked and wide layouts must agree
            let mut got = vec![0i32; rows * out];
            gemm_plain(&q, rows, &p, &mut got);
            assert_eq!(got, want, "plain on blocked, rows {rows}");
            let mut got = vec![0i32; rows * out];
            gemm_plain(&q, rows, &pw, &mut got);
            assert_eq!(got, want, "plain on wide, rows {rows}");
            for isa in [Isa::Vnni512, Isa::Vnni256, Isa::Avx2, Isa::Dot] {
                if detect_isa() == Isa::Plain
                    || (isa == Isa::Vnni512 && detect_isa() != Isa::Vnni512)
                    || (isa == Isa::Dot) != (detect_isa() == Isa::Dot)
                {
                    continue;
                }
                #[cfg(target_arch = "x86_64")]
                if isa == Isa::Vnni256 && !std::arch::is_x86_feature_detected!("avxvnni") {
                    continue;
                }
                #[cfg(target_arch = "x86_64")]
                if isa == Isa::Avx2
                    && !(std::arch::is_x86_feature_detected!("avx2")
                        && std::arch::is_x86_feature_detected!("fma"))
                {
                    continue;
                }
                let mut got = vec![0i32; rows * out];
                gemm_u8i8(isa, &q, rows, &p, &mut got);
                assert_eq!(got, want, "{isa:?} rows {rows}");
                if isa == Isa::Avx2 {
                    // the 16-bit copy's kernel too (a CPU at the AVX2 level packs it)
                    let mut got = vec![0i32; rows * out];
                    gemm_u8i8(isa, &q, rows, &pw, &mut got);
                    assert_eq!(got, want, "AVX2 from 16 bits, rows {rows}");
                    // and the AVX level's kernel on it
                    let mut got = vec![0i32; rows * out];
                    gemm_u8i8(Isa::Avx, &q, rows, &pw, &mut got);
                    assert_eq!(got, want, "AVX from 16 bits, rows {rows}");
                }
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
}
