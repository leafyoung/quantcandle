//! Port of `torchquantlib/models/interest_rate/black_karasinski.py`
//! (Black-Karasinski short-rate model: lognormal Ornstein-Uhlenbeck).
//! Includes the initial-discount-curve interpolation, the trapezoidal
//! zero-coupon bond price over the expected short-rate path, and the
//! terminal lognormal sampler used by `simulate`. Module RNG seed: 2005.

/// Python `_linear_interpolate` (same as the Hull-White file's copy).
pub fn linear_interpolate(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.len() == 1 {
        return fp[0];
    }
    let clamped = x.clamp(xp[0], xp[xp.len() - 1]);
    let mut upper = xp.partition_point(|&v| v < clamped);
    upper = upper.clamp(1, xp.len() - 1);
    let lower = upper - 1;
    let weight = (clamped - xp[lower]) / (xp[upper] - xp[lower]).max(1e-12);
    fp[lower] + weight * (fp[upper] - fp[lower])
}

/// Black-Karasinski model state: speed `a`, volatility `sigma`, initial
/// short rate `r0`, and an optional fitted initial discount curve.
#[derive(Debug, Clone)]
pub struct BlackKarasinski {
    pub a: f64,
    pub sigma: f64,
    pub r0: f64,
    pub curve_maturities: Option<Vec<f64>>,
    pub curve_discount_factors: Option<Vec<f64>>,
}

impl BlackKarasinski {
    pub fn new(a: f64, sigma: f64, r0: f64) -> Self {
        Self { a, sigma, r0, curve_maturities: None, curve_discount_factors: None }
    }

    /// Python `fit_initial_discount_curve`.
    pub fn fit_initial_discount_curve(&mut self, maturities: Vec<f64>, discount_factors: Vec<f64>) {
        assert_eq!(
            maturities.len(),
            discount_factors.len(),
            "maturities and discount_factors must be one-dimensional tensors of the same shape."
        );
        assert!(!maturities.is_empty());
        self.curve_maturities = Some(maturities);
        self.curve_discount_factors = Some(discount_factors);
    }

    /// Python `initial_discount_factor` (note the r0 clamp at 1e-8).
    pub fn initial_discount_factor(&self, maturity: f64) -> f64 {
        match (&self.curve_maturities, &self.curve_discount_factors) {
            (Some(xp), Some(fp)) => linear_interpolate(maturity, xp, fp),
            _ => (-self.r0.max(1e-8) * maturity).exp(),
        }
    }

    /// Python `instantaneous_forward_rate` (central difference, bump 1e-4).
    pub fn instantaneous_forward_rate(&self, maturity: f64) -> f64 {
        let bump = 1e-4;
        let t_up = maturity + bump;
        let t_down = (maturity - bump).max(0.0);
        let log_up = -self.initial_discount_factor(t_up).clamp(1e-12, f64::MAX).ln();
        let log_down = -self.initial_discount_factor(t_down).clamp(1e-12, f64::MAX).ln();
        (log_up - log_down) / (t_up - t_down).max(1e-12)
    }

    /// Python `zero_coupon_bond_price`: exp(-trapz E[r_t] dt) with E[r_t] =
    /// exp(log(r0) e^{-a t} + variance(t)/2) over `integration_steps` points
    /// (default 256). When no explicit `r_t` is given and a curve is fitted,
    /// the interpolated discount factor is returned instead.
    pub fn zero_coupon_bond_price(&self, maturity: f64, r_t: Option<f64>, integration_steps: usize) -> f64 {
        if maturity <= 0.0 {
            return 1.0;
        }
        if r_t.is_none() && self.curve_maturities.is_some() {
            return self.initial_discount_factor(maturity);
        }
        let a = self.a.max(1e-12);
        let sigma = self.sigma.max(0.0);
        let short_rate = r_t.unwrap_or(self.r0).max(1e-8);

        let n = integration_steps.max(2);
        let dt = maturity / (n - 1) as f64;
        // torch.linspace over [0, maturity], trapezoid rule.
        let mut integral = 0.0;
        for i in 0..n {
            let t = dt * i as f64;
            let mean_log_r = short_rate.ln() * (-a * t).exp();
            let variance_log_r =
                sigma * sigma * (1.0 - (-2.0 * a * t).exp()) / (2.0 * a).max(1e-12);
            let expected_short_rate = (mean_log_r + 0.5 * variance_log_r).exp();
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            integral += w * expected_short_rate * dt;
        }
        (-integral).exp()
    }

    /// Python `zero_coupon_yield`.
    pub fn zero_coupon_yield(&self, maturity: f64, r_t: Option<f64>, integration_steps: usize) -> f64 {
        let price = self.zero_coupon_bond_price(maturity, r_t, integration_steps);
        -price.clamp(1e-12, f64::MAX).ln() / maturity.max(1e-12)
    }

    /// Python `simulate`: N draws of exp(mean + sqrt(var) z) where the
    /// log-rate is Gaussian. Normals from `crate::rng::Xorshift64Star` with
    /// `seed` (2005 in tests).
    pub fn simulate(&self, t: f64, n: usize, seed: u64) -> Vec<f64> {
        let a = self.a.max(0.0);
        let sigma = self.sigma.max(0.0);
        let r0 = self.r0.max(1e-8);
        let log_r0 = r0.ln();
        let mean = log_r0 * (-a * t).exp();
        if sigma == 0.0 {
            return vec![mean.exp(); n];
        }
        let variance = sigma * sigma * (1.0 - (-2.0 * a * t).exp()) / (2.0 * a).max(1e-12);
        let std = variance.max(0.0).sqrt();
        let mut rng = crate::rng::Xorshift64Star::new(seed);
        (0..n)
            .map(|_| (mean + std * rng.next_std_normal()).exp())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Closed-form (trapezoid-integrated) bond price; reference from
    /// scripts/gen_black_karasinski.py, tol 1e-11 relative.
    #[test]
    fn bond_price_matches_reference() {
        let bk = BlackKarasinski::new(0.1, 0.01, 0.03);
        let p1 = bk.zero_coupon_bond_price(1.0, None, 256);
        let p5 = bk.zero_coupon_bond_price(5.0, None, 256);
        let p10 = bk.zero_coupon_bond_price(10.0, None, 256);
        let y1 = bk.zero_coupon_yield(1.0, None, 256);
        assert!(rel(p1, 0.9649184371120622) < 1e-11, "p1={p1}");
        assert!(rel(p5, 0.7105610577190904) < 1e-11, "p5={p5}");
        assert!(rel(p10, 0.27069253076250477) < 1e-11, "p10={p10}");
        assert!(rel(y1, 0.03571170234265963) < 1e-10, "y1={y1}");
        assert_eq!(bk.zero_coupon_bond_price(0.0, None, 256), 1.0);
    }

    #[test]
    fn bond_price_uses_fitted_curve() {
        let mut bk = BlackKarasinski::new(0.1, 0.01, 0.03);
        bk.fit_initial_discount_curve(vec![1.0, 5.0], vec![0.97, 0.86]);
        assert!(rel(bk.zero_coupon_bond_price(3.0, None, 256), 0.915) < 1e-12);
    }

    /// Stochastic sampler; expected values from scripts/gen_black_karasinski.py
    /// with refgen normals, seed 2005, tol 1e-12 relative.
    #[test]
    fn simulate_matches_reference() {
        let bk = BlackKarasinski::new(0.1, 0.01, 0.03);
        let r = bk.simulate(2.0, 8, 2005);
        let expected = [
            0.05636124422394425,
            0.05628683234533506,
            0.05687669858060467,
            0.056972978601515185,
            0.05591941959828365,
            0.05674199927477411,
            0.05616423664753451,
            0.05642739581981856,
        ];
        for (got, want) in r.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    fn simulate_sigma_zero_is_degenerate() {
        let bk = BlackKarasinski::new(0.1, 0.0, 0.03);
        let r = bk.simulate(2.0, 4, 2005);
        let expected = 0.05664640303882101;
        for v in r {
            assert!((v - expected).abs() < 1e-15);
        }
    }
}
