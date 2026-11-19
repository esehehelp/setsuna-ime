//! The int8 matrix products the encoder's kernels share, and the helpers of its token-routed experts.
//!
//! A row of activations is quantized to u8 per row (asymmetric, zero exact: `quantize_row`) and multiplied with int8
//! weights prepacked in blocks (`Packed`) as u8 x i8 dot products of four (`gemm_u8i8`, and `gemm_u8i8_f32` with its
//! epilogue: the scales, the bias and GELU), with AVX-512 VNNI, AVX-VNNI, AVX2, AVX or plain Rust, the same integer
//! sums on every path. `route` and `add_routed` are a token-routed mixture of experts' router and sum (per frame the
//! router's fp32 logits, the top-`k` experts with their softmax renormalised over the `k`, the weighted outputs
//! summed), the experts' Linears being `group_q`'s.

/// An int8 matrix (out, in) prepacked for the u8 x i8 dot products: 16 output rows x 4 inputs a 64-byte block (lane j
/// = row 16 * b + j), blocks along the inputs; `colsum` the sum of each row (for the activations' zero point). Only
/// one layout is kept:
/// `blocked` for the VNNI paths, else `raw` (row-major, the plain path).
pub(crate) struct Packed {
    out: usize,
    inp: usize,
    blocked: Blocked,
    raw: Vec<i8>,
    colsum: Vec<i32>,
    /// the AVX2 kernel's 16-bit copy of `blocked` (`widen`), made when the engine's level is AVX2; else empty
    wide: Vec<i16>,
}

/// The blocked layout, packed here.
enum Blocked {
    Owned(Vec<i8>),
}

impl std::ops::Deref for Blocked {
    type Target = [i8];

    fn deref(&self) -> &[i8] {
        match self {
            Blocked::Owned(v) => v,
        }
    }
}

impl Packed {
    /// `blocked`: pack for the VNNI paths (when the shape allows 16 x 4 blocks), else keep the matrix as it is.
    pub(crate) fn new(w: &[i8], out: usize, inp: usize, blocked: bool) -> Self {
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
                wide: Vec::new(),
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
        let wide = if matches!(detect_isa(), Isa::Avx2 | Isa::Avx) {
            widen(&blocked)
        } else {
            Vec::new()
        };
        Packed {
            out,
            inp,
            blocked: Blocked::Owned(blocked),
            raw: Vec::new(),
            colsum,
            wide,
        }
    }

    /// Tests only: `new` with the AVX2 kernel's 16-bit copy whatever the CPU.
    #[cfg(test)]
    pub(crate) fn with_wide(mut self) -> Self {
        if !self.blocked.is_empty() {
            self.wide = widen(&self.blocked);
        }
        self
    }
}

/// The AVX2 kernel's copy of a blocked int8 matrix in 16 bits: for each block of 16 outputs and each 4 inputs, 4
/// registers of 16 values, outputs 0-7 with inputs 0 and 1 (output after output, its two inputs side by side), then
/// with inputs 2 and 3, then outputs 8-15 likewise; so one 16-bit multiply-add of a pair of inputs gives 8 outputs'
/// partial sums (`x86::tile_avx2w`).
fn widen(blocked: &[i8]) -> Vec<i16> {
    let mut wide = Vec::with_capacity(blocked.len());
    for b in blocked.chunks_exact(64) {
        for half in 0..2 {
            for pair in 0..2 {
                for o in 0..8 {
                    let at = (half * 8 + o) * 4 + pair * 2;
                    wide.push(b[at] as i16);
                    wide.push(b[at + 1] as i16);
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
    Plain,
}

/// The kernel for the engine's level (`crate::cpu`; SETSUNA_CPU_LEVEL caps it).
pub(crate) fn detect_isa() -> Isa {
    use crate::cpu::Level;
    match crate::cpu::level() {
        Level::Avx512 => Isa::Vnni512,
        Level::AvxVnni => Isa::Vnni256,
        Level::Avx2 => Isa::Avx2,
        Level::Avx => Isa::Avx,
        Level::Plain => Isa::Plain,
    }
}

/// The integer sums acc[r][o] = sum_i q[r][i] * w[o][i] of `rows` u8 rows (stride `w.inp`) with the int8 matrix.
pub(crate) fn gemm_u8i8(isa: Isa, q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
    debug_assert!(q.len() >= rows * w.inp && acc.len() >= rows * w.out);
    if w.blocked.is_empty() {
        return gemm_plain(q, rows, w, acc);
    }
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni512 => unsafe { x86::gemm_vnni512(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni256 => unsafe { x86::gemm_vnni256(q, rows, w, acc) },
        // AVX2: the 16-bit copy's kernel when the matrix has it (the engine's level was AVX2 when it was packed)
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 if !w.wide.is_empty() => unsafe { x86::gemm_avx2w(q, rows, w, acc) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { x86::gemm_avx2(q, rows, w, acc) },
        // AVX (128-bit integers): the 16-bit copy's kernel (a matrix packed at the AVX level has it)
        #[cfg(target_arch = "x86_64")]
        Isa::Avx if !w.wide.is_empty() => unsafe { x86::gemm_avxw(q, rows, w, acc) },
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
    assert!(
        per_row && e.xs.len() >= rows || e.xs.len() == 1,
        "{} scales for {rows} rows",
        e.xs.len()
    );
    assert!(e.out <= w.out && y.len() >= rows * e.out && w.colsum.len() >= e.out);
    assert!(e.col.is_none_or(|c| c.len() >= e.out) && e.bias.is_none_or(|b| b.len() >= e.out));
    #[cfg(target_arch = "x86_64")]
    {
        let raw = x86::Epi {
            xs: e.xs.as_ptr(),
            per_row,
            cs: w.colsum.as_ptr(),
            col: e.col.map_or(std::ptr::null(), <[f32]>::as_ptr),
            scale: e.scale,
            bias: e.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            out: e.out,
        };
        if isa == Isa::Vnni512 && !w.blocked.is_empty() {
            // SAFETY: the CPU has AVX-512 VNNI (`isa`); the slices hold what the asserts say
            return unsafe { x86::gemm_vnni512_f32(q, rows, w, &raw, y) };
        }
        if matches!(isa, Isa::Avx2 | Isa::Avx) && !w.wide.is_empty() {
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
                    x86::gemm_avx2w_f32(acc, rows, w, &raw, y)
                } else {
                    x86::gemm_avxw_f32(acc, rows, w, &raw, y)
                }
            };
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
            acc[r * n + o] = if w.raw.is_empty() {
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

    /// RT rows x C output blocks of int8 weights as `tile512`, the sums turned to `gemm_u8i8_f32`'s outputs in
    /// registers: row r's outputs at `y + r * ystride`, block c's at column `col0 + 16 c`, the columns past `out`
    /// left unwritten (masked).
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
                _mm512_mask_storeu_ps(y.add(r * ystride + o), m, f);
            }
        }
    }

    /// `gemm_u8i8_f32` with AVX-512 VNNI (`gemm_vnni512`'s tiles).
    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) unsafe fn gemm_vnni512_f32(
        q: &[u8],
        rows: usize,
        w: &Packed,
        e: &Epi,
        y: &mut [f32],
    ) {
        let (n, k, out) = (w.out, w.inp, e.out);
        assert!(y.len() >= rows * out && w.colsum.len() >= n);
        let (qp, yp, wp, bstride) = (q.as_ptr(), y.as_mut_ptr(), w.blocked.as_ptr(), k * 16);
        tiles(rows, n / 16, |rt, r0, ob, c| {
            let (q, yr, wb) = (qp.add(r0 * k), yp.add(r0 * out), wp.add(ob * bstride));
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
    pub(super) unsafe fn gemm_avx2w(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.wide.as_ptr(), k * 16);
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
        e: &Epi,
        y: &mut [f32],
    ) {
        let (n, k, out) = (w.out, w.inp, e.out);
        assert!(
            y.len() >= rows * out
                && w.colsum.len() >= n
                && w.wide.len() == n * k
                && pairs.len() >= rows * k / 2
        );
        let (xp, yp, wp, bstride) = (pairs.as_ptr(), y.as_mut_ptr(), w.wide.as_ptr(), k * 16);
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
    pub(super) unsafe fn gemm_avxw(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.wide.as_ptr(), k * 16);
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
        e: &Epi,
        y: &mut [f32],
    ) {
        let (n, k, out) = (w.out, w.inp, e.out);
        assert!(
            y.len() >= rows * out
                && w.colsum.len() >= n
                && w.wide.len() == n * k
                && pairs.len() >= rows * k / 2
        );
        let (xp, yp, wp, bstride) = (pairs.as_ptr(), y.as_mut_ptr(), w.wide.as_ptr(), k * 16);
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
    pub(super) unsafe fn gemm_avx2(q: &[u8], rows: usize, w: &Packed, acc: &mut [i32]) {
        let (n, k) = (w.out, w.inp);
        let (qp, ap, wp, bstride) = (q.as_ptr(), acc.as_mut_ptr(), w.blocked.as_ptr(), k * 16);
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
}

/// Per row: u8 codes of `v` (asymmetric, the range widened to hold 0 exactly), its scale and zero point.
#[inline(always)]
pub(crate) fn quantize_row(v: &[f32], q: &mut [u8]) -> (f32, i32) {
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
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
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
    let n = x.len() / d;
    rows.resize_with(e, Vec::new);
    rows.iter_mut().for_each(Vec::clear);
    let mut logits = vec![0.0f32; e];
    let mut top: Vec<usize> = Vec::with_capacity(k);
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        for (j, l) in logits.iter_mut().enumerate() {
            *l = dot(xr, &router[j * d..(j + 1) * d]);
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
            rows[j].push((r, (logits[j] - m).exp() / z));
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
            let wide = Packed::new(&q8, stride, inp, true).with_wide();
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
                    if isa != Isa::Plain && std::arch::is_x86_feature_detected!("avx2") {
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
                    if isa != Isa::Plain && std::arch::is_x86_feature_detected!("avx") {
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
            for isa in [Isa::Vnni512, Isa::Vnni256, Isa::Avx2] {
                if detect_isa() == Isa::Plain
                    || (isa == Isa::Vnni512 && detect_isa() != Isa::Vnni512)
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
                    let pw = Packed::new(&w, out, inp, true).with_wide();
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
