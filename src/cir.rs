//! Port of `torchquantlib/models/interest_rate/cir.py` (Cox-Ingersoll-Ross
//! interest rate model). Closed-form zero-coupon bond price/yield plus the
//! Gaussian stationary-approximation terminal sampler used by `simulate`
//! (the Python matches first two moments of the exact CIR transition and
//! applies a ReLU floor). Module RNG seed: 2003.

/// Python `CIR.zero_coupon_bond_price` with the same internal clamps.
pub fn zero_coupon_bond_price(kappa: f64, theta: f64, sigma: f64, r0: f64, maturity: f64) -> f64 {
    if maturity <= 0.0 {
        return 1.0;
    }
    let kappa = kappa.max(1e-12);
    let theta = theta.max(0.0);
    let sigma = sigma.max(1e-12);
    let short_rate = r0.max(0.0);

    let gamma = (kappa * kappa + 2.0 * sigma * sigma).sqrt();
    let exp_gamma_t = (gamma * maturity).exp();
    let denominator = (gamma + kappa) * (exp_gamma_t - 1.0) + 2.0 * gamma;
    let denom = denominator.max(1e-12);
    let b_term = 2.0 * (exp_gamma_t - 1.0) / denom;
    let a_term = ((2.0 * gamma * ((kappa + gamma) * maturity / 2.0).exp() / denom).max(1e-12))
        .powf(2.0 * kappa * theta / (sigma * sigma));
    a_term * (-b_term * short_rate).exp()
}

/// Python `CIR.zero_coupon_yield`.
pub fn zero_coupon_yield(kappa: f64, theta: f64, sigma: f64, r0: f64, maturity: f64) -> f64 {
    let price = zero_coupon_bond_price(kappa, theta, sigma, r0, maturity);
    -price.clamp(1e-12, f64::MAX).ln() / maturity.max(1e-12)
}

/// Python `CIR.simulate` (including the Feller condition adjustment the
/// constructor applies when `2*kappa*theta <= sigma^2`: theta is bumped to
/// `sigma^2/(2*kappa) + 1e-6`). Normals from `crate::rng::Xorshift64Star`
/// with `seed` (2003 in tests).
pub fn simulate(kappa: f64, theta: f64, sigma: f64, r0: f64, t: f64, n: usize, seed: u64) -> Vec<f64> {
    // Constructor-time Feller fix (matches __init__ in the Python).
    let mut theta = theta.max(0.0);
    let kappa_c = kappa.max(1e-12);
    let feller_gap = 2.0 * kappa_c * theta - sigma * sigma;
    if feller_gap <= 0.0 {
        theta = sigma * sigma / kappa_c.max(1e-12) / 2.0 + 1e-6;
    }

    let kappa = kappa.max(0.0);
    let sigma = sigma.max(0.0);
    let r0 = r0.max(0.0);
    let mean_reversion = (-kappa * t).exp();
    let mut mean = theta + (r0 - theta) * mean_reversion;
    if kappa * t >= 1.0 {
        mean = theta;
    }
    if sigma == 0.0 {
        return vec![mean.max(0.0); n];
    }
    let stationary_variance = sigma * sigma * theta / (2.0 * kappa).max(1e-12);
    let variance = stationary_variance * (1.0 - (-2.0 * kappa * t).exp());
    let std = variance.max(0.0).sqrt();
    let mut rng = crate::rng::Xorshift64Star::new(seed);
    (0..n)
        .map(|_| (mean + std * rng.next_std_normal()).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KAPPA: f64 = 0.15;
    const THETA: f64 = 0.05;
    const SIGMA: f64 = 0.02;
    const R0: f64 = 0.03;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Closed-form bond price; reference from scripts/gen_cir.py (pure
    /// float64), tolerance 1e-11 relative (powf across implementations).
    #[test]
    fn bond_price_closed_form() {
        let p1 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 1.0);
        let p5 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 5.0);
        let p10 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 10.0);
        let y5 = zero_coupon_yield(KAPPA, THETA, SIGMA, R0, 5.0);
        assert!(rel(p1, 0.9690627652294498) < 1e-11, "p1={p1}");
        assert!(rel(p5, 0.8357016192777666) < 1e-11, "p5={p5}");
        assert!(rel(p10, 0.6733398275379303) < 1e-11, "p10={p10}");
        assert!(rel(y5, 0.035896728869234854) < 1e-10, "y5={y5}");
        assert_eq!(zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 0.0), 1.0);
    }

    /// Stochastic sampler; expected values from scripts/gen_cir.py with
    /// refgen normals, seed 2003, tolerance 1e-12 relative.
    #[test]
    fn simulate_matches_reference() {
        let r = simulate(KAPPA, THETA, SIGMA, R0, 2.0, 8, 2003);
        let expected = [
            0.027491557755358782,
            0.03929694074081416,
            0.038447389418432246,
            0.04658625647593316,
            0.036818660016410695,
            0.042299964951907305,
            0.03919273215255722,
            0.021893950585852563,
        ];
        for (got, want) in r.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn feller_violation_bumps_theta() {
        // 2*kappa*theta = 0.02 < sigma^2 = 0.09 -> theta bumped to
        // sigma^2/(2*kappa) + 1e-6, which inflates the stationary variance.
        // Reference (t=1.0) from scripts/gen_cir.py, seed 2003.
        let r = simulate(0.1, 0.1, 0.3, 0.03, 1.0, 4, 2003);
        let expected = [0.0, 0.21366040800835318, 0.18398263109246554, 0.4683015115614581];
        for (got, want) in r.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15, "got={got} want={want}");
        }
    }

    #[test]
    fn simulate_relu_floor() {
        // Large sigma with short horizon: draws below zero must be floored.
        let r = simulate(0.1, 0.001, 0.5, 0.001, 0.1, 64, 2003);
        assert!(r.iter().all(|v| *v >= 0.0));
    }
}
