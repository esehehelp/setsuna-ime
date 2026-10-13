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

const LANES: usize = 16;

#[inline(always)]
fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    // exp(-87.3) ~ 1e-38: a row's smallest terms, far below its sum
    let x = x.clamp(-87.3, 88.3);
    let fx = (x * LOG2E + 0.5).floor();
    let r = x - fx * C1 - fx * C2;
    let p = ((((1.987_569_2e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((fx as i32 + 127) << 23) as u32)
}

#[inline(always)]
fn max_body(row: &[f32]) -> f32 {
    let chunks = row.chunks_exact(LANES);
    let rest = chunks.remainder();
    let mut m = [f32::NEG_INFINITY; LANES];
    for c in chunks {
        for i in 0..LANES {
            m[i] = m[i].max(c[i]);
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

/// Log-sum-exp of `row`, vectorized for the running CPU.
pub fn row(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU supports the target features.
            return unsafe { lse_avx512(row) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: as above.
            return unsafe { lse_avx2(row) };
        }
    }
    lse_body(row)
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
