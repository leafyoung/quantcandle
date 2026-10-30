//! Tensor microbenchmark: vectorized GBM path simulation (the arithmetic
//! behind the family's MC estimators) on Device::Cpu vs Device::Cuda.
//!
//! Measures whether candle's CUDA backend actually accelerates the kind of
//! tensor work the MC estimators would delegate to a GPU. Note the pricing
//! modules themselves are scalar-f64 by design — this benchmarks the tensor
//! layer only.

use candle_core::{Device, Tensor};
use std::time::Instant;

fn gbm_log_paths(
    dev: &Device,
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n: usize,
    steps: usize,
    seed: u64,
) -> candle_core::Result<Tensor> {
    // Deterministic pseudo-draws generated on CPU, then moved to the device
    // (transfer-inclusive timing, as a real workload would pay it).
    let mut z = Vec::with_capacity(n * steps);
    let mut state = seed | 1;
    for _ in 0..n * steps {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = ((state >> 11) as f64) / (1u64 << 53) as f64;
        let u2 = ((state.wrapping_mul(0x9E3779B97F4A7C15) >> 11) as f64) / (1u64 << 53) as f64;
        z.push((u1 + u2 - 1.0) * 1.7);
    }
    let zz = Tensor::from_vec(z, (n, steps), dev)?;
    // cumulative sum of z via lower-triangular ones matmul: W_j = sum_{k<=j} z_k
    let tri = Tensor::from_vec(
        (0..steps)
            .flat_map(|i| (0..steps).map(move |j| if j <= i { 1.0 } else { 0.0 }))
            .collect(),
        (steps, steps),
        dev,
    )?;
    let dt = t / steps as f64;
    // log S_j = log S0 + (r - sigma^2/2) t_j + sigma * sqrt(dt) * W_j
    let w = zz.matmul(&tri.t()?)? * (sigma * dt.sqrt());
    let drift_row = Tensor::from_vec(
        (0..steps)
            .map(|j| (r - 0.5 * sigma * sigma) * (j + 1) as f64 * dt)
            .collect(),
        (1, steps),
        dev,
    )?;
    let log_s0 = Tensor::full(s0.ln(), (n, 1), dev)?;
    let drift_bc = drift_row.broadcast_as((n, steps))?;
    let log_s0_col = log_s0.broadcast_as((n, steps))?;
    Ok(((w + drift_bc)? + log_s0_col)?)
}

fn bench(dev: &Device, name: &str, n: usize, steps: usize) {
    let t0 = Instant::now();
    let log_p = gbm_log_paths(dev, 100.0, 0.05, 0.2, 1.0, n, steps, 42).unwrap();
    let s = log_p.exp().unwrap();
    let mean = s.mean_all().unwrap().to_scalar::<f64>().unwrap();
    println!(
        "{name}: {n}x{steps} paths in {:.3}s (mean S={mean:.4})",
        t0.elapsed().as_secs_f64()
    );
}

fn main() {
    let (n, steps) = (500_000usize, 64usize);
    println!("GBM path tensor benchmark: {n} paths x {steps} steps");
    bench(&Device::Cpu, "cpu", n, steps);
    match Device::new_cuda(0) {
        Ok(dev) => {
            bench(&dev, "cuda(warmup)", 10_000, 8);
            bench(&dev, "cuda", n, steps);
        }
        Err(e) => println!("cuda unavailable: {e}"),
    }
}
