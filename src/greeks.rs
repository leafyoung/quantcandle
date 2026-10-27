//! Port of `greeks/greeks.py` (`MalliavinGreeks`, re-exported by
//! `greeks/malliavin.py` as `Greeks`). Monte-Carlo option greeks via
//! Malliavin-calculus weights.
//!
//! Stochasticity: the Python draws `torch.randn` tensors; here all standard
//! normals come from [`crate::rng::Xorshift64Star`] (Box-Muller, bit-exact
//! against `scripts/refgen.py`). Draw order replicates the row-major fill of
//! the torch tensors: for an (num_paths, num_steps) draw, path-major then
//! step-minor. Tests use seed 3001 (module index 1 of the 3000-block).
//!
//! Math is done in `f64`. Expected values come from the float64 reference
//! generator `scripts/gen_greeks.py` (same normals, same seed); tolerance
//! 1e-6 relative (1e-4 for the digital greeks, where the 1e-4-wide Gaussian
//! Dirac approximation makes the estimators large and ill-conditioned).

use crate::lmm::cholesky;
use crate::rng::Xorshift64Star;

/// Mirror of the Python `MalliavinGreeks` class (a name-space handle; all
/// numeric work is in the free functions below).
#[derive(Debug, Clone, Copy, Default)]
pub struct MalliavinGreeks;

/// Alias mirroring the Python `Greeks = MalliavinGreeks` re-export.
pub type Greeks = MalliavinGreeks;

/// Greeks bundle for European options (Delta, Gamma, Vega, Theta, Rho).
#[derive(Debug, Clone, Copy)]
pub struct EuropeanGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

/// Greeks bundle for digital / barrier / basket / Asian options.
#[derive(Debug, Clone, Copy)]
pub struct Greeks3 {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

/// Malliavin greeks for a European call (single-step GBM, exact terminal
/// simulation).
pub fn european_option_greeks(
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    num_paths: usize,
    seed: u64,
) -> EuropeanGreeks {
    let mut rng = Xorshift64Star::new(seed);
    let dt = t;
    let sqrt_dt = dt.sqrt();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let discount = (-r * t).exp();

    let mut acc = [0.0_f64; 6]; // payoff*Dt, payoff*D2, payoff*Dsigma, payoff*S, payoff*Dr, payoff
    for _ in 0..num_paths {
        let z = rng.next_std_normal();
        let st = s0 * (drift + sigma * sqrt_dt * z).exp();
        let payoff = (st - k).max(0.0);
        let dt_st = st * z / sqrt_dt;
        let d2t_st = dt_st * (z / sqrt_dt - sigma * sqrt_dt);
        let d_sigma_st = st * (z * sqrt_dt - sigma * dt);
        let d_r_st = st * t;
        acc[0] += payoff * dt_st;
        acc[1] += payoff * d2t_st;
        acc[2] += payoff * d_sigma_st;
        acc[3] += payoff * st;
        acc[4] += payoff * d_r_st;
        acc[5] += payoff;
    }
    let n = num_paths as f64;
    EuropeanGreeks {
        delta: discount * (acc[0] / n) / (sigma * s0),
        gamma: discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega: discount * (acc[2] / n) / sigma,
        theta: -discount * ((r - 0.5 * sigma * sigma) * acc[3] / n + 0.5 * sigma * acc[0] / n) / t,
        rho: discount * (acc[4] / n) - t * discount * (acc[5] / n),
    }
}

/// Malliavin greeks for a digital (cash-or-nothing) call, with the Dirac
/// delta approximated by a narrow Gaussian of width `epsilon`.
#[allow(clippy::too_many_arguments)]
pub fn digital_option_greeks(
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    num_paths: usize,
    epsilon: f64,
    seed: u64,
) -> Greeks3 {
    let mut rng = Xorshift64Star::new(seed);
    let dt = t;
    let sqrt_dt = dt.sqrt();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let discount = (-r * t).exp();
    let normalizer = epsilon * (2.0 * std::f64::consts::PI).sqrt();

    let mut acc = [0.0_f64; 3];
    for _ in 0..num_paths {
        let z = rng.next_std_normal();
        let st = s0 * (drift + sigma * sqrt_dt * z).exp();
        let delta_approx = (1.0 / normalizer) * (-0.5 * ((st - k) / epsilon).powi(2)).exp();
        let dt_st = st * z / sqrt_dt;
        let d2t_st = dt_st * (z / sqrt_dt - sigma * sqrt_dt);
        let d_sigma_st = st * (z * sqrt_dt - sigma * dt);
        let w = q * delta_approx;
        acc[0] += w * dt_st;
        acc[1] += w * d2t_st;
        acc[2] += w * d_sigma_st;
    }
    let n = num_paths as f64;
    Greeks3 {
        delta: discount * (acc[0] / n) / (sigma * s0),
        gamma: discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega: discount * (acc[2] / n) / sigma,
    }
}

/// Simulate GBM paths row-major from an already-drawn normal matrix
/// `z[path][step]`; returns S[0..=steps] per path.
fn simulate_paths(
    s0: f64,
    t: f64,
    r: f64,
    sigma: f64,
    num_steps: usize,
    z: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let dt = t / num_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    z.iter()
        .map(|row| {
            let mut path = Vec::with_capacity(num_steps + 1);
            path.push(s0);
            for &zi in row {
                let prev = *path.last().unwrap();
                path.push(prev * (drift + vol * zi).exp());
            }
            path
        })
        .collect()
}

/// Malliavin greeks for a knock-out barrier call (`up-and-out` or
/// `down-and-out`), monitored at each simulation step.
#[allow(clippy::too_many_arguments)]
pub fn barrier_option_greeks(
    s0: f64,
    k: f64,
    h: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: &str,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Result<Greeks3, String> {
    if option_type != "up-and-out" && option_type != "down-and-out" {
        return Err("Invalid option_type. Use 'up-and-out' or 'down-and-out'.".to_string());
    }
    let up = option_type == "up-and-out";
    let mut rng = Xorshift64Star::new(seed);
    let dt = t / num_steps as f64;
    let sqrt_dt = dt.sqrt();

    // Draw the (num_paths, num_steps) normal matrix in row-major order.
    let z: Vec<Vec<f64>> = (0..num_paths)
        .map(|_| (0..num_steps).map(|_| rng.next_std_normal()).collect())
        .collect();
    let paths = simulate_paths(s0, t, r, sigma, num_steps, &z);

    let discount = (-r * t).exp();
    let mut acc = [0.0_f64; 3];
    for (zi, srow) in z.iter().zip(paths.iter()) {
        let st = *srow.last().unwrap();
        let hit_barrier = srow[1..].iter().any(|&s| if up { s >= h } else { s <= h });
        if hit_barrier {
            continue; // payoff and all Malliavin weights are zeroed by (~hit_barrier)
        }
        let payoff = (st - k).max(0.0);
        let z_sum: f64 = zi.iter().sum();
        let dt_st = st * z_sum / sqrt_dt;
        let d2t_st = dt_st * (z_sum / sqrt_dt - sigma * sqrt_dt);
        let z2m1: f64 = zi.iter().map(|v| v * v - 1.0).sum();
        let d_sigma_st = st * (z2m1 * sqrt_dt);
        acc[0] += payoff * dt_st;
        acc[1] += payoff * d2t_st;
        acc[2] += payoff * d_sigma_st;
    }
    let n = num_paths as f64;
    Ok(Greeks3 {
        delta: discount * (acc[0] / n) / (sigma * s0),
        gamma: discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega: discount * (acc[2] / n) / sigma,
    })
}

/// Malliavin delta for a floating-strike lookback call.
pub fn lookback_option_delta(
    s0: f64,
    t: f64,
    r: f64,
    sigma: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> f64 {
    let mut rng = Xorshift64Star::new(seed);
    let dt = t / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let z: Vec<Vec<f64>> = (0..num_paths)
        .map(|_| (0..num_steps).map(|_| rng.next_std_normal()).collect())
        .collect();
    let paths = simulate_paths(s0, t, r, sigma, num_steps, &z);

    let discount = (-r * t).exp();
    let mut acc = 0.0_f64;
    for (zi, srow) in z.iter().zip(paths.iter()) {
        let st = *srow.last().unwrap();
        // S_min over columns [0, num_steps) (final column excluded, as in
        // the Python `S[:, :-1]`).
        let prefix = &srow[..num_steps];
        let s_min = prefix.iter().cloned().fold(f64::INFINITY, f64::min);
        let payoff = (st - s_min).max(0.0);
        let dt_st = st * zi.iter().sum::<f64>() / sqrt_dt;
        // indicator: columns equal to the minimum; average their Malliavin
        // derivatives (0 when none — mirrors the Python NaN guard).
        let mut dt_smin = 0.0;
        let mut count = 0.0_f64;
        for (j, &s) in prefix.iter().enumerate() {
            if s == s_min {
                dt_smin += s * zi[j] / sqrt_dt;
                count += 1.0;
            }
        }
        let dt_smin = if count > 0.0 { dt_smin / count } else { 0.0 };
        if payoff > 0.0 {
            acc += (dt_st - dt_smin) / (sigma * s0);
        }
    }
    discount * acc / num_paths as f64
}

/// Malliavin greeks for a European basket (arithmetic-weighted) call on
/// correlated GBM assets; returns (Delta, Gamma, Vega), one entry per asset.
pub fn basket_option_greeks(
    s0: &[f64],
    k: f64,
    t: f64,
    r: f64,
    sigma: &[f64],
    rho: &[Vec<f64>],
    weights: &[f64],
    num_paths: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let num_assets = s0.len();
    let l = cholesky(rho);
    let mut rng = Xorshift64Star::new(seed);
    let dt = t;
    let sqrt_dt = dt.sqrt();
    let discount = (-r * t).exp();

    let mut acc = vec![[0.0_f64; 3]; num_assets];

    for _ in 0..num_paths {
        // Independent normals, then correlate: zc = z @ L.T.
        let z: Vec<f64> = (0..num_assets).map(|_| rng.next_std_normal()).collect();
        let st: Vec<f64> = (0..num_assets)
            .map(|j| {
                let zc: f64 = (0..num_assets).map(|kk| z[kk] * l[j][kk]).sum();
                s0[j] * ((r - 0.5 * sigma[j] * sigma[j]) * t + sigma[j] * sqrt_dt * zc).exp()
            })
            .collect();
        let basket: f64 = (0..num_assets).map(|j| weights[j] * st[j]).sum();
        let payoff = (basket - k).max(0.0);
        for j in 0..num_assets {
            let zc_j: f64 = (0..num_assets).map(|kk| z[kk] * l[j][kk]).sum();
            let dt_st_j = st[j] * zc_j / sqrt_dt;
            let d2t_st_j = dt_st_j * (zc_j / sqrt_dt - sigma[j] * sqrt_dt);
            let d_sigma_st_j = st[j] * (zc_j * sqrt_dt - sigma[j] * dt);
            let w = payoff * weights[j];
            acc[j][0] += w * dt_st_j;
            acc[j][1] += w * d2t_st_j;
            acc[j][2] += w * d_sigma_st_j;
        }
    }
    let n = num_paths as f64;
    let delta: Vec<f64> = (0..num_assets)
        .map(|j| discount * (acc[j][0] / n) / (sigma[j] * s0[j]))
        .collect();
    let gamma: Vec<f64> = (0..num_assets)
        .map(|j| discount * (acc[j][1] / n) / (sigma[j] * sigma[j] * s0[j] * s0[j]))
        .collect();
    let vega: Vec<f64> = (0..num_assets)
        .map(|j| discount * (acc[j][2] / n) / sigma[j])
        .collect();
    (delta, gamma, vega)
}

/// Malliavin greeks for an arithmetic-average Asian call.
pub fn asian_option_greeks(
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Greeks3 {
    let mut rng = Xorshift64Star::new(seed);
    let dt = t / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let z: Vec<Vec<f64>> = (0..num_paths)
        .map(|_| (0..num_steps).map(|_| rng.next_std_normal()).collect())
        .collect();
    let paths = simulate_paths(s0, t, r, sigma, num_steps, &z);

    let discount = (-r * t).exp();
    let mut acc = [0.0_f64; 3];
    for (zi, srow) in z.iter().zip(paths.iter()) {
        // Average over columns [1, num_steps], as the Python `S[:, 1:]`.
        let n_st = num_steps as f64;
        let s_mean: f64 = srow[1..].iter().sum::<f64>() / n_st;
        let payoff = (s_mean - k).max(0.0);
        let mut dt_s_mean = 0.0;
        let mut d2t_s_mean = 0.0;
        let mut d_sigma_s_mean = 0.0;
        for (j, &zj) in zi.iter().enumerate() {
            let sj = srow[j + 1];
            let dt_s = sj * zj / sqrt_dt;
            let d2t_s = dt_s * (zj / sqrt_dt - sigma * sqrt_dt);
            let d_sigma_s = sj * (zj * sqrt_dt - sigma * dt);
            dt_s_mean += dt_s;
            d2t_s_mean += d2t_s;
            d_sigma_s_mean += d_sigma_s;
        }
        dt_s_mean /= n_st;
        d2t_s_mean /= n_st;
        d_sigma_s_mean /= n_st;
        acc[0] += payoff * dt_s_mean;
        acc[1] += payoff * d2t_s_mean;
        acc[2] += payoff * d_sigma_s_mean;
    }
    let n = num_paths as f64;
    Greeks3 {
        delta: discount * (acc[0] / n) / (sigma * s0),
        gamma: discount * (acc[1] / n) / (sigma * sigma * s0 * s0),
        vega: discount * (acc[2] / n) / sigma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 3001; // 3000 + module index
    const TOL: f64 = 1e-6; // relative

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(1e-300) || (a - b).abs() < 1e-12
    }

    #[test]
    fn european_matches_reference() {
        let g = european_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, SEED);
        // scripts/gen_greeks.py: european
        assert!(close(g.delta, 93.97332927208662, TOL), "delta {}", g.delta);
        assert!(close(g.gamma, 7.268677237984516, TOL), "gamma {}", g.gamma);
        assert!(close(g.vega, 7991.189624942042, TOL), "vega {}", g.vega);
        assert!(close(g.theta, -230.13095761217176, TOL), "theta {}", g.theta);
        assert!(close(g.rho, 1395.5863679889742, TOL), "rho {}", g.rho);
    }

    #[test]
    fn digital_matches_reference() {
        let g = digital_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 10.0, 20_000, 2.0, SEED);
        // scripts/gen_greeks.py: digital (epsilon=2.0; the Python default
        // epsilon=1e-4 makes every estimator numerically zero)
        assert!(close(g.delta, -0.14213961143402573, 1e-6), "delta {}", g.delta);
        assert!(close(g.gamma, 0.0029487745130569966, 1e-6), "gamma {}", g.gamma);
        assert!(close(g.vega, -33.29994785023318, 1e-6), "vega {}", g.vega);
    }

    #[test]
    fn barrier_matches_reference() {
        let g = barrier_option_greeks(
            100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "up-and-out", 20_000, 25, SEED,
        )
        .unwrap();
        // scripts/gen_greeks.py: barrier (up-and-out)
        assert!(close(g.delta, 169.99793934820894, TOL), "delta {}", g.delta);
        assert!(close(g.gamma, 183.63245445936846, TOL), "gamma {}", g.gamma);
        assert!(close(g.vega, -174.05787950305043, TOL), "vega {}", g.vega);

        let g2 = barrier_option_greeks(
            100.0, 90.0, 80.0, 1.0, 0.05, 0.2, "down-and-out", 20_000, 25, SEED,
        )
        .unwrap();
        assert!(close(g2.delta, 2903.5653727366225, TOL), "delta {}", g2.delta);
        assert!(close(g2.gamma, 6034.218595945937, TOL), "gamma {}", g2.gamma);
        assert!(close(g2.vega, 1552.748714302966, TOL), "vega {}", g2.vega);
    }

    #[test]
    fn barrier_invalid_type() {
        assert!(barrier_option_greeks(100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "knock-in", 10, 5, SEED).is_err());
    }

    #[test]
    fn lookback_matches_reference() {
        let d = lookback_option_delta(100.0, 1.0, 0.05, 0.2, 20_000, 25, SEED);
        // scripts/gen_greeks.py: lookback
        assert!(close(d, 17.295340929617094, TOL), "delta {d}");
    }

    #[test]
    fn basket_matches_reference() {
        let s0 = [100.0, 90.0, 80.0];
        let sigma = [0.2, 0.25, 0.3];
        let rho = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let weights = [0.4, 0.3, 0.3];
        let (delta, gamma, vega) =
            basket_option_greeks(&s0, 95.0, 1.0, 0.05, &sigma, &rho, &weights, 20_000, SEED);
        // scripts/gen_greeks.py: basket
        let d_exp = [22.549108947872366, 14.51060287552215, 11.97730651682888];
        let g_exp = [1.804147411523855, 0.9926433261158537, 0.765858818081831];
        let v_exp = [1882.6283128582918, 1042.089221101072, 718.1441026308789];
        for j in 0..3 {
            assert!(close(delta[j], d_exp[j], TOL), "delta[{j}] {}", delta[j]);
            assert!(close(gamma[j], g_exp[j], TOL), "gamma[{j}] {}", gamma[j]);
            assert!(close(vega[j], v_exp[j], TOL), "vega[{j}] {}", vega[j]);
        }
    }

    #[test]
    fn asian_matches_reference() {
        let g = asian_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 20_000, 25, SEED);
        // scripts/gen_greeks.py: asian
        assert!(close(g.delta, 41.90190821064969, TOL), "delta {}", g.delta);
        assert!(close(g.gamma, 46.23101947253118, TOL), "gamma {}", g.gamma);
        assert!(close(g.vega, 139.4347463474697, TOL), "vega {}", g.vega);
    }

    /// Cholesky sanity: L @ L.T == input.
    #[test]
    fn cholesky_reconstruction() {
        let m = vec![
            vec![4.0, 2.0, 1.2],
            vec![2.0, 5.0, 1.5],
            vec![1.2, 1.5, 6.0],
        ];
        let l = cholesky(&m);
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for kk in 0..3 {
                    s += l[i][kk] * l[j][kk];
                }
                assert!((s - m[i][j]).abs() < 1e-12);
            }
        }
    }
}
