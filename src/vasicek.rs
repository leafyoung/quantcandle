//! Port of `torchquantlib/models/interest_rate/vasicek.py` (Vasicek interest
//! rate model). Closed-form zero-coupon bond price/yield plus the terminal
//! short-rate distribution sampler used by `simulate` (exact Gaussian
//! transition over the full horizon). Module RNG seed: 2002.

/// Python `Vasicek.zero_coupon_bond_price` with the same internal clamps.
pub fn zero_coupon_bond_price(kappa: f64, theta: f64, sigma: f64, r0: f64, maturity: f64) -> f64 {
    if maturity <= 0.0 {
        return 1.0;
    }
    let kappa = kappa.max(1e-12);
    let sigma = sigma.max(0.0);
    let b_term = (1.0 - (-kappa * maturity).exp()) / kappa;
    let a_term = ((theta - sigma * sigma / (2.0 * kappa * kappa)) * (b_term - maturity)
        - sigma * sigma * b_term * b_term / (4.0 * kappa))
        .exp();
    a_term * (-b_term * r0).exp()
}

/// Python `Vasicek.zero_coupon_yield`.
pub fn zero_coupon_yield(kappa: f64, theta: f64, sigma: f64, r0: f64, maturity: f64) -> f64 {
    let price = zero_coupon_bond_price(kappa, theta, sigma, r0, maturity);
    -price.clamp(1e-12, f64::MAX).ln() / maturity.max(1e-12)
}

/// Python `Vasicek.simulate`: N draws from the exact terminal short-rate
/// distribution r(T) ~ N(mean, variance), with the Python's special cases
/// (kappa*T >= 1 collapses the mean to theta; sigma == 0 is deterministic).
/// Normals come from `crate::rng::Xorshift64Star` with `seed` (2002 in tests).
pub fn simulate(kappa: f64, theta: f64, sigma: f64, r0: f64, t: f64, n: usize, seed: u64) -> Vec<f64> {
    let kappa = kappa.max(0.0);
    let sigma = sigma.max(0.0);
    let mean_reversion = (-kappa * t).exp();
    let mut mean = theta + (r0 - theta) * mean_reversion;
    if kappa * t >= 1.0 {
        mean = theta;
    }
    if sigma == 0.0 {
        return vec![mean; n];
    }
    let variance = sigma * sigma * (1.0 - (-2.0 * kappa * t).exp()) / (2.0 * kappa).max(1e-12);
    let std = variance.max(0.0).sqrt();
    let mut rng = crate::rng::Xorshift64Star::new(seed);
    (0..n).map(|_| mean + std * rng.next_std_normal()).collect()
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

    /// Closed-form bond price; reference from scripts/gen_vasicek.py (pure
    /// float64 math), tolerance 1e-12 relative.
    #[test]
    fn bond_price_closed_form() {
        let p1 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 1.0);
        let p5 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 5.0);
        let p10 = zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 10.0);
        let y1 = zero_coupon_yield(KAPPA, THETA, SIGMA, R0, 1.0);
        assert!(rel(p1, 0.9691188082746262) < 1e-12, "p1={p1}");
        assert!(rel(p5, 0.8396917673474292) < 1e-12, "p5={p5}");
        assert!(rel(p10, 0.6897352359349652) < 1e-12, "p10={p10}");
        assert!(rel(y1, 0.03136806544871844) < 1e-11, "y1={y1}");
        assert_eq!(zero_coupon_bond_price(KAPPA, THETA, SIGMA, R0, 0.0), 1.0);
    }

    /// Stochastic terminal-distribution sampler; expected values from
    /// scripts/gen_vasicek.py drawing normals from refgen with seed 2002,
    /// tolerance 1e-12 relative (bit-identical normals, f64 math).
    #[test]
    fn simulate_matches_reference() {
        let r = simulate(KAPPA, THETA, SIGMA, R0, 2.0, 8, 2002);
        let expected = [
            0.02613496077664776,
            -0.012153713624497658,
            0.03323136734270169,
            0.02382083956028588,
            0.02579654700137396,
            0.041537151891176356,
            0.059147298795526104,
            0.025514955997793482,
        ];
        for (got, want) in r.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn simulate_sigma_zero_is_degenerate() {
        let r = simulate(KAPPA, THETA, 0.0, R0, 0.5, 4, 2002);
        // kappa*T = 0.075 < 1, mean = theta + (r0-theta)*exp(-kappa T).
        let mean = THETA + (R0 - THETA) * (-KAPPA * 0.5_f64).exp();
        for v in r {
            assert!((v - mean).abs() < 1e-15);
        }
    }

    #[test]
    fn simulate_long_horizon_mean_collapses_to_theta() {
        let r = simulate(KAPPA, THETA, 0.0, R0, 10.0, 4, 2002);
        // kappa*T = 1.5 >= 1 -> mean = theta.
        for v in r {
            assert!((v - THETA).abs() < 1e-15);
        }
    }
}
