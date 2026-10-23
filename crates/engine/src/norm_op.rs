//! `neo_ime.AddLayerNorm`: LayerNorm over the last axis of x + skip, and the sum itself, as one CPU kernel.
//!
//! ONNX Runtime's CPU LayerNormalization and com.microsoft SkipLayerNormalization (1.28) run each row as one scalar
//! loop (Welford's update, a division a value): about 0.45 us a 512-wide row, 7% of a Shiran-v2 int4 conversion on
//! one thread (28 norms). Here a row is four vector passes: the sum, its mean, the squared deviations, the output.
//!
//! Inputs: `x` f32 (..., h); `skip` f32 shaped as `x` or as its trailing axes (leading 1s allowed: a (1, T, h) skip
//! under a batch), optional (absent: the norm of `x`); `gamma`, `beta` f32 (h).
//! Attribute `epsilon`. Outputs: the norm, f32 shaped as `x`; the sum `x + skip`, optional (a pre-norm residual
//! stream reads it on). The research export writes it in place of both ORT norms.

use ort::operator::{
    ComputeContext, Kernel, KernelContext, Operator, OperatorInput, OperatorOutput,
};
use ort::value::TensorElementType;

pub struct AddLayerNorm;

impl Operator for AddLayerNorm {
    type Kernel<'attr> = NormKernel;

    fn name(&self) -> &str {
        "AddLayerNorm"
    }

    fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
        use TensorElementType::Float32;
        [
            OperatorInput::required(Float32),
            OperatorInput::optional(Float32),
            OperatorInput::required(Float32),
            OperatorInput::required(Float32),
        ]
    }

    fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
        use TensorElementType::Float32;
        [
            OperatorOutput::required(Float32),
            OperatorOutput::optional(Float32),
        ]
    }

    fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> ort::Result<NormKernel> {
        let eps = kctx
            .get::<f32>("epsilon")
            .ok_or_else(|| ort::Error::new("AddLayerNorm: attribute epsilon missing"))?;
        Ok(NormKernel {
            eps,
            isa: Isa::detect(),
        })
    }
}

pub struct NormKernel {
    eps: f32,
    isa: Isa,
}

impl Kernel for NormKernel {
    fn compute(&mut self, ctx: &ComputeContext) -> ort::Result<()> {
        let missing = |what: &str| ort::Error::new(format!("AddLayerNorm: {what} missing"));
        let xv = ctx.input(0)?.ok_or_else(|| missing("input x"))?;
        let (shape, x) = xv.try_extract_tensor::<f32>()?;
        let shape = shape.to_vec();
        let sv = ctx.input(1)?;
        let skip = match &sv {
            Some(v) => {
                // shaped as x, or as its trailing axes with leading 1s (a (1, T, H) skip under a batch), as
                // SkipLayerNormalization takes it
                let (s, d) = v.try_extract_tensor::<f32>()?;
                let lead = s.iter().take_while(|&&n| n == 1).count();
                let trailing = &s[lead.min(s.len().saturating_sub(1))..];
                if trailing.len() > shape.len() || !shape.ends_with(trailing) {
                    return Err(ort::Error::new(format!(
                        "AddLayerNorm: skip {:?} not shaped as x {shape:?}",
                        &s[..]
                    )));
                }
                Some(d)
            }
            None => None,
        };
        let gv = ctx.input(2)?.ok_or_else(|| missing("input gamma"))?;
        let (_, gamma) = gv.try_extract_tensor::<f32>()?;
        let bv = ctx.input(3)?.ok_or_else(|| missing("input beta"))?;
        let (_, beta) = bv.try_extract_tensor::<f32>()?;
        let h = shape.last().copied().unwrap_or(0) as usize;
        if h == 0 || gamma.len() != h || beta.len() != h {
            return Err(ort::Error::new(format!(
                "AddLayerNorm: gamma {} / beta {} for a last axis of {h}",
                gamma.len(),
                beta.len()
            )));
        }
        let mut yv = ctx
            .output(0, shape.clone())?
            .ok_or_else(|| missing("output"))?;
        let (_, y) = yv.try_extract_tensor_mut::<f32>()?;
        let mut sumv = if ctx.num_outputs()? > 1 {
            ctx.output(1, shape)?
        } else {
            None
        };
        let sum = match sumv.as_mut() {
            Some(v) => Some(v.try_extract_tensor_mut::<f32>()?.1),
            None => None,
        };
        add_layer_norm(self.isa, x, skip, gamma, beta, self.eps, y, sum);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Isa {
    Avx512,
    Avx2,
    Plain,
}

impl Isa {
    fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Isa::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Isa::Avx2;
            }
        }
        Isa::Plain
    }
}

/// Each row of h = gamma.len() values: y = (s - mean) / sqrt(var + eps) * gamma + beta for s = x + skip (x alone
/// without a skip), s also written to `sum` when given; mean and (biased) variance over the row, as torch's LayerNorm.
#[allow(clippy::too_many_arguments)]
fn add_layer_norm(
    isa: Isa,
    x: &[f32],
    skip: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    y: &mut [f32],
    sum: Option<&mut [f32]>,
) {
    #[cfg(target_arch = "x86_64")]
    match isa {
        // SAFETY: the features were detected
        Isa::Avx512 => return unsafe { norm_avx512(x, skip, gamma, beta, eps, y, sum) },
        Isa::Avx2 => return unsafe { norm_avx2(x, skip, gamma, beta, eps, y, sum) },
        Isa::Plain => {}
    }
    norm_body(x, skip, gamma, beta, eps, y, sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn norm_avx512(
    x: &[f32],
    skip: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    y: &mut [f32],
    sum: Option<&mut [f32]>,
) {
    norm_body(x, skip, gamma, beta, eps, y, sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_avx2(
    x: &[f32],
    skip: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    y: &mut [f32],
    sum: Option<&mut [f32]>,
) {
    norm_body(x, skip, gamma, beta, eps, y, sum)
}

#[inline(always)]
fn norm_body(
    x: &[f32],
    skip: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    y: &mut [f32],
    mut sum: Option<&mut [f32]>,
) {
    let h = gamma.len();
    // a skip of fewer rows repeats over the leading axes (`compute` checked its shape)
    let skip_rows = skip.map_or(1, |s| s.len() / h);
    for (r, (xr, yr)) in x.chunks_exact(h).zip(y.chunks_exact_mut(h)).enumerate() {
        // the sum in y's row, then normed there
        match skip {
            Some(s) => {
                let sr = r % skip_rows;
                for ((o, &a), &b) in yr.iter_mut().zip(xr).zip(&s[sr * h..(sr + 1) * h]) {
                    *o = a + b;
                }
            }
            None => yr.copy_from_slice(xr),
        }
        if let Some(s) = sum.as_deref_mut() {
            s[r * h..(r + 1) * h].copy_from_slice(yr);
        }
        let mean = lanes_sum(yr, |v| v) / h as f32;
        let var = lanes_sum(yr, |v| (v - mean) * (v - mean)) / h as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for ((o, &g), &b) in yr.iter_mut().zip(gamma).zip(beta) {
            *o = (*o - mean) * rstd * g + b;
        }
    }
}

/// sum f(v) over `v` in 16 partial sums (vectorizes).
#[inline(always)]
fn lanes_sum(v: &[f32], f: impl Fn(f32) -> f32) -> f32 {
    let mut acc = [0f32; 16];
    let chunks = v.chunks_exact(16);
    let tail: f32 = chunks.remainder().iter().map(|&a| f(a)).sum();
    for c in chunks {
        for i in 0..16 {
            acc[i] += f(c[i]);
        }
    }
    acc.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 3.0 + 0.5
            })
            .collect()
    }

    /// Every path against an f64 LayerNorm of x + skip: with and without a skip and a sum output, row widths off the
    /// 16 lanes too.
    #[test]
    fn add_layer_norm_matches_f64() {
        let mut isas = vec![Isa::Plain];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                isas.push(Isa::Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                isas.push(Isa::Avx512);
            }
        }
        for h in [8usize, 37, 512] {
            let n = 7;
            let (x, skip) = (rows(3, n * h), rows(5, n * h));
            let (gamma, beta) = (rows(7, h), rows(11, h));
            let eps = 1e-5f32;
            for with_skip in [true, false] {
                let mut want = vec![0f64; n * h];
                for r in 0..n {
                    let s: Vec<f64> = (0..h)
                        .map(|i| {
                            x[r * h + i] as f64
                                + if with_skip {
                                    skip[r * h + i] as f64
                                } else {
                                    0.0
                                }
                        })
                        .collect();
                    let mean = s.iter().sum::<f64>() / h as f64;
                    let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64;
                    for i in 0..h {
                        want[r * h + i] = (s[i] - mean) / (var + eps as f64).sqrt()
                            * gamma[i] as f64
                            + beta[i] as f64;
                    }
                }
                for &isa in &isas {
                    let (mut y, mut sum) = (vec![0f32; n * h], vec![0f32; n * h]);
                    let sk = with_skip.then_some(&skip[..]);
                    add_layer_norm(isa, &x, sk, &gamma, &beta, eps, &mut y, Some(&mut sum));
                    let worst = y
                        .iter()
                        .zip(&want)
                        .map(|(&a, &b)| (a as f64 - b).abs())
                        .fold(0f64, f64::max);
                    assert!(
                        worst < 2e-5,
                        "{isa:?} h {h} skip {with_skip}: worst {worst}"
                    );
                    for i in 0..n * h {
                        let s = x[i] + if with_skip { skip[i] } else { 0.0 };
                        assert_eq!(sum[i].to_bits(), s.to_bits(), "{isa:?} sum {i}");
                    }
                    let mut y2 = vec![0f32; n * h];
                    add_layer_norm(isa, &x, sk, &gamma, &beta, eps, &mut y2, None);
                    assert_eq!(y, y2, "{isa:?}: the sum output changes nothing");
                }
            }
        }
    }

    /// A skip of fewer rows (a (1, T, h) skip under a batch of 2) repeats: the same as the skip tiled to x's rows.
    #[test]
    fn a_shorter_skip_repeats() {
        let (h, t) = (40, 3);
        let (x, skip) = (rows(3, 2 * t * h), rows(5, t * h));
        let (gamma, beta) = (rows(7, h), rows(11, h));
        let tiled: Vec<f32> = skip.iter().chain(&skip).copied().collect();
        let isa = Isa::detect();
        let (mut a, mut b) = (vec![0f32; 2 * t * h], vec![0f32; 2 * t * h]);
        add_layer_norm(isa, &x, Some(&skip), &gamma, &beta, 1e-5, &mut a, None);
        add_layer_norm(isa, &x, Some(&tiled), &gamma, &beta, 1e-5, &mut b, None);
        assert_eq!(a, b);
    }

    /// The kernel on Shiran-v2's rows against ONNX Runtime's (1.28, one thread, measured apart: SkipLayerNormalization
    /// 32 us on (1, 60, 512), 14 us on (1, 20, 512)).
    /// cargo test -p setsuna-engine --release --lib norm_op::tests::norm_time -- --ignored --nocapture
    #[test]
    #[ignore]
    fn norm_time() {
        let isa = Isa::detect();
        let h = 512;
        for t in [20usize, 60] {
            let (x, skip) = (rows(3, t * h), rows(5, t * h));
            let (gamma, beta) = (rows(7, h), rows(11, h));
            let (mut y, mut sum) = (vec![0f32; t * h], vec![0f32; t * h]);
            let mut times = Vec::new();
            for _ in 0..2000 {
                let a = std::time::Instant::now();
                add_layer_norm(
                    isa,
                    &x,
                    Some(&skip),
                    &gamma,
                    &beta,
                    1e-5,
                    &mut y,
                    Some(&mut sum),
                );
                times.push(a.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{isa:?} (1, {t}, {h}) with the sum: {:.2} us ({})",
                times[times.len() / 2] * 1e6,
                y[5]
            );
        }
    }
}
