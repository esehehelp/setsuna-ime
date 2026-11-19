//! Log-sum-exp of a logits row, vectorized.
//!
//! Every frame's log-softmax needs the log-sum-exp of the whole vocabulary
//! row, and the scalar `f32::exp` loop was the engine's largest cost after
//! ONNX Runtime (~0.9 ms of a request at T≈60, V=4801). Here exp is the
//! Cephes `expf` scheme (range reduction by ln 2, degree-6 polynomial,
//! ~1 ulp) written so it vectorizes, over 16 lanes, compiled for AVX-512F
//! and AVX2 and chosen at run time. The arithmetic (no FMA contraction, a
//! fixed lane order) is the same on every path, so the result does not
//! depend on the CPU.
//!
//! The value differs from the scalar loop (`exact`) in the last bits. It is
//! a constant shift within a frame, so the ranking of a frame's tokens never
//! changes; only hypotheses whose scores are that close can swap.

// the x86 kernels and what only they read are unused elsewhere
#![cfg_attr(
    not(target_arch = "x86_64"),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

const LANES: usize = 16;

/// e^x (the Cephes scheme above), branch-free so a loop of it vectorizes; also the AR decoder's softmaxes and the
/// native encoder's softmaxes and sigmoids (`ar_native`, `native_encoder`).
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    // exp(-87.3) ~ 1e-38: a row's smallest terms, far below its sum. The clamp by compares (NaN fails the first and
    // becomes -87.3, a term of ~0), so fx below is always an integer in [-126, 127].
    let x = if x > -87.3 { x } else { -87.3 };
    let x = if x < 88.3 { x } else { 88.3 };
    let fx = (x * LOG2E + 0.5).floor();
    let r = x - fx * C1 - fx * C2;
    let p = ((((1.987_569_2e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    // SAFETY: fx is a finite integer in [-126, 127] (the clamp above), within i32. `fx as i32` (its saturating
    // checks) kept the loop scalar: 5.4 us for a row of 8006 on the 9950X3D (2026-11-12); the value is the same.
    let e = unsafe { fx.to_int_unchecked::<i32>() };
    y * f32::from_bits(((e + 127) << 23) as u32)
}

#[inline(always)]
fn max_body(row: &[f32]) -> f32 {
    let chunks = row.chunks_exact(LANES);
    let rest = chunks.remainder();
    let mut m = [f32::NEG_INFINITY; LANES];
    for c in chunks {
        for i in 0..LANES {
            // a compare-select, which vectorizes (`f32::max` did not); NaN is
            // skipped as `f32::max` skips it, and a signed zero chosen either
            // way gives the same sum and log-sum-exp
            m[i] = if c[i] > m[i] { c[i] } else { m[i] };
        }
    }
    rest.iter().fold(
        m.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        |a, &b| a.max(b),
    )
}

#[inline(always)]
fn lse_body(row: &[f32]) -> f32 {
    let max = max_body(row);
    if !max.is_finite() {
        return exact(row);
    }
    let chunks = row.chunks_exact(LANES);
    let rest = chunks.remainder();
    let mut s = [0f32; LANES];
    for c in chunks {
        for i in 0..LANES {
            s[i] += exp(c[i] - max);
        }
    }
    let mut sum = 0f32;
    for v in s {
        sum += v;
    }
    for &v in rest {
        sum += exp(v - max);
    }
    max + sum.ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn lse_avx512(row: &[f32]) -> f32 {
    lse_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn lse_avx2(row: &[f32]) -> f32 {
    lse_body(row)
}

/// The AVX level's build (no AVX2 or FMA).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn lse_avx1(row: &[f32]) -> f32 {
    lse_body(row)
}

/// Log-sum-exp of `row`, vectorized for the running CPU.
pub fn row(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::avx512() {
            // SAFETY: the CPU supports the target features.
            return unsafe { lse_avx512(row) };
        }
        if crate::cpu::avx2() {
            // SAFETY: as above.
            return unsafe { lse_avx2(row) };
        }
        if crate::cpu::avx() {
            // SAFETY: as above.
            return unsafe { lse_avx1(row) };
        }
    }
    lse_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn max_avx512(row: &[f32]) -> f32 {
    max_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_avx2(row: &[f32]) -> f32 {
    max_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn max_avx1(row: &[f32]) -> f32 {
    max_body(row)
}

/// The largest value of `row` (NaN skipped; -inf for an empty row or one of NaN and -inf), vectorized for the running
/// CPU: the reading corrector's argmax (`proposal::greedy_scores`).
pub fn max(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::avx512() {
            // SAFETY: the CPU supports the target features.
            return unsafe { max_avx512(row) };
        }
        if crate::cpu::avx2() {
            // SAFETY: as above.
            return unsafe { max_avx2(row) };
        }
        if crate::cpu::avx() {
            // SAFETY: as above.
            return unsafe { max_avx1(row) };
        }
    }
    max_body(row)
}

/// Whether the AVX-512 paths run (`crate::cpu`: the CPU has AVX-512 and SETSUNA_CPU_LEVEL does not cap it below).
pub(crate) fn avx512f() -> bool {
    crate::cpu::avx512()
}

/// e^x of 16 lanes: [`exp`]'s operations in its order (no fused multiply-adds), so the same bits.
///
/// # Safety
/// The CPU has AVX-512 F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn exp16(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // NaN becomes -87.3 as in `exp` (vmaxps gives its second operand for a NaN)
    let x = _mm512_max_ps(x, _mm512_set1_ps(-87.3));
    let x = _mm512_min_ps(x, _mm512_set1_ps(88.3));
    let fx = _mm512_add_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)),
        _mm512_set1_ps(0.5),
    );
    let fx = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
    let r = _mm512_sub_ps(
        _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(0.693_359_4))),
        _mm512_mul_ps(fx, _mm512_set1_ps(-2.121_944_4e-4)),
    );
    let mut p = _mm512_set1_ps(1.987_569_2e-4);
    for c in [
        1.398_199_9e-3f32,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        5.000_000_1e-1,
    ] {
        p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(c));
    }
    let y = _mm512_add_ps(
        _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(p, r), r), r),
        _mm512_set1_ps(1.0),
    );
    let e = _mm512_cvttps_epi32(fx);
    let two = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
        e,
        _mm512_set1_epi32(127),
    )));
    _mm512_mul_ps(y, two)
}

/// [`exp16`] of lanes at most 0 (or NaN): the same bits in 4 operations fewer, its upper clamp left out (a no-op
/// there) and the scaling by 2^fx one `vscalefps` (the same single rounding of y 2^fx as the multiply by the power
/// of two). A lane above 0 gives a value not to be used (the script terms' lanes of another class, masked).
///
/// # Safety
/// The CPU has AVX-512 F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) unsafe fn exp16_le0(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let x = _mm512_max_ps(x, _mm512_set1_ps(-87.3));
    let fx = _mm512_add_ps(
        _mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)),
        _mm512_set1_ps(0.5),
    );
    let fx = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
    let r = _mm512_sub_ps(
        _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(0.693_359_4))),
        _mm512_mul_ps(fx, _mm512_set1_ps(-2.121_944_4e-4)),
    );
    let mut p = _mm512_set1_ps(1.987_569_2e-4);
    for c in [
        1.398_199_9e-3f32,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        5.000_000_1e-1,
    ] {
        p = _mm512_add_ps(_mm512_mul_ps(p, r), _mm512_set1_ps(c));
    }
    let y = _mm512_add_ps(
        _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(p, r), r), r),
        _mm512_set1_ps(1.0),
    );
    _mm512_scalef_ps(y, fx)
}

/// e^x of 8 lanes with AVX2: [`exp16`]'s operations, the same bits as [`exp`].
///
/// # Safety
/// The CPU has AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // NaN becomes -87.3 as in `exp` (vmaxps gives its second operand for a NaN)
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.3));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.3));
    let fx = _mm256_floor_ps(_mm256_add_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        _mm256_set1_ps(0.5),
    ));
    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693_359_4))),
        _mm256_mul_ps(fx, _mm256_set1_ps(-2.121_944_4e-4)),
    );
    let mut p = _mm256_set1_ps(1.987_569_2e-4);
    for c in [
        1.398_199_9e-3f32,
        8.333_452e-3,
        4.166_579_6e-2,
        1.666_666_5e-1,
        5.000_000_1e-1,
    ] {
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(c));
    }
    let y = _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, r), r), r),
        _mm256_set1_ps(1.0),
    );
    let e = _mm256_cvttps_epi32(fx);
    let two = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        e,
        _mm256_set1_epi32(127),
    )));
    _mm256_mul_ps(y, two)
}

/// Log-sum-exp of a logits row, the 16-wide chunks with nothing within 24 nats of the row's best left out (each of
/// their terms is below e^-24 of the best's, all of them together below 3e-7 of the sum for 8006 tokens): a row of a
/// peaked distribution (a CTC frame, most of an AR row) takes the exponentials of a few chunks instead of all of
/// them. With AVX-512 the chunks' test is one compare.
pub fn peaked(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx512f() {
        // SAFETY: the CPU has AVX-512 F
        return unsafe { peaked_avx512(row) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx2() {
        // SAFETY: the CPU has AVX2 and FMA
        return unsafe { peaked_avx2(row) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx() {
        // SAFETY: the CPU has AVX
        return unsafe { peaked_avx1(row) };
    }
    peaked_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn peaked_avx2(row: &[f32]) -> f32 {
    peaked_body(row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn peaked_avx1(row: &[f32]) -> f32 {
    peaked_body(row)
}

#[inline(always)]
fn peaked_body(row: &[f32]) -> f32 {
    let max = max_body(row);
    if !max.is_finite() {
        return exact(row);
    }
    let lim = max - 24.0;
    let mut sum = [0f32; LANES];
    let chunks = row.chunks_exact(LANES);
    let rest = chunks.remainder();
    for c in chunks {
        let mut hit = false;
        for &x in c {
            hit |= x > lim;
        }
        if hit {
            for i in 0..LANES {
                sum[i] += exp(c[i] - max);
            }
        }
    }
    let mut total = 0f32;
    for v in sum {
        total += v;
    }
    for &x in rest {
        total += exp(x - max);
    }
    max + total.ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn peaked_avx512(row: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len();
    let whole = n - n % LANES;
    let p = row.as_ptr();
    let mut m = _mm512_set1_ps(f32::NEG_INFINITY);
    for j in (0..whole).step_by(LANES) {
        // NaN skipped (vmaxps gives its second operand)
        m = _mm512_max_ps(_mm512_loadu_ps(p.add(j)), m);
    }
    let mut max = _mm512_reduce_max_ps(m);
    for &x in &row[whole..] {
        max = if x > max { x } else { max };
    }
    if !max.is_finite() {
        return exact(row);
    }
    let (vmax, lim) = (_mm512_set1_ps(max), _mm512_set1_ps(max - 24.0));
    let mut s = _mm512_setzero_ps();
    for j in (0..whole).step_by(LANES) {
        let v = _mm512_loadu_ps(p.add(j));
        if _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, lim) != 0 {
            s = _mm512_add_ps(s, exp16(_mm512_sub_ps(v, vmax)));
        }
    }
    let mut total = _mm512_reduce_add_ps(s);
    for &x in &row[whole..] {
        total += exp(x - max);
    }
    max + total.ln()
}

/// The scalar log-sum-exp the engine used before: `f32::exp` in row order.
pub fn exact(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in row {
        sum += (v - max).exp();
    }
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn reference(row: &[f32]) -> f64 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        max + row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln()
    }

    #[test]
    fn exp_is_accurate_to_about_an_ulp() {
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let rel = ((exp(x) as f64 - (x as f64).exp()) / (x as f64).exp()).abs();
            worst = worst.max(rel);
            x += 0.013;
        }
        assert!(worst < 3e-7, "worst relative error {worst:e}");
    }

    #[test]
    fn matches_a_double_precision_reference() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for n in [1usize, 5, 15, 16, 17, 100, 4801] {
            for scale in [1.0f32, 10.0, 60.0] {
                let row: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 2.0 * scale).collect();
                let want = reference(&row);
                let got = super::row(&row) as f64;
                let tol = 4e-7 * (1.0 + want.abs());
                assert!(
                    (got - want).abs() <= tol,
                    "n={n} scale={scale}: {got} vs {want}"
                );
                // and no worse than the scalar loop's own error
                let scalar = exact(&row) as f64;
                assert!((got - want).abs() <= (scalar - want).abs() + tol);
            }
        }
    }

    /// `peaked` within the left-out terms of the double-precision log-sum-exp, on flat and peaked rows of the vocabulary's
    /// length and short ones, the 16-lane path and the plain one.
    #[test]
    fn peaked_is_the_log_sum_exp() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for n in [1usize, 15, 17, 100, 8006] {
            for scale in [1.0f32, 10.0, 60.0] {
                let mut row: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 2.0 * scale).collect();
                for peak in [None, Some(40.0f32)] {
                    if let Some(p) = peak {
                        row[n / 2] = p;
                    }
                    let want = reference(&row);
                    let tol = 1e-6 * (1.0 + want.abs());
                    let (got, plain) = (peaked(&row) as f64, peaked_body(&row) as f64);
                    assert!(
                        (got - want).abs() <= tol,
                        "n={n} scale={scale}: {got} vs {want}"
                    );
                    assert!(
                        (plain - want).abs() <= tol,
                        "n={n} scale={scale}: {plain} vs {want}"
                    );
                }
            }
        }
        let none = [f32::NEG_INFINITY; 20];
        assert_eq!(peaked(&none).is_nan(), exact(&none).is_nan());
    }

    #[test]
    fn extreme_rows() {
        // saturated, -inf and -1e30 entries, a single element
        let mut row = vec![-1e30f32; 4801];
        row[7] = 1e4;
        row[8] = f32::NEG_INFINITY;
        assert!((super::row(&row) - 1e4).abs() < 1e-2);
        assert_eq!(super::row(&[3.5]), exact(&[3.5]));
        // no finite entry: the scalar loop's answer (NaN) either way
        let none = [f32::NEG_INFINITY; 20];
        assert_eq!(super::row(&none).is_nan(), exact(&none).is_nan());
    }
}
