//! Port of `torchquantlib/models/interest_rate/hull_white.py`
//! (Hull-White short-rate model). Includes the initial-discount-curve
//! interpolation machinery, the numerical instantaneous forward rate and
//! theta(t), the closed-form zero-coupon bond price, and the terminal
//! Gaussian sampler used by `simulate`. Module RNG seed: 2004.

/// Python `_linear_interpolate` (clamp to endpoints, searchsorted mid-rule).
pub fn linear_interpolate(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if xp.len() == 1 {
        return fp[0];
    }
    let clamped = x.clamp(xp[0], xp[xp.len() - 1]);
    // torch.searchsorted(xp, clamped, right=False): first index with
    // xp[idx] >= clamped, clamped into [1, len-1].
    let mut upper = xp.partition_point(|&v| v < clamped);
    upper = upper.clamp(1, xp.len() - 1);
    let lower = upper - 1;
    let weight = (clamped - xp[lower]) / (xp[upper] - xp[lower]).max(1e-12);
    fp[lower] + weight * (fp[upper] - fp[lower])
}

/// Hull-White model state: mean-reversion speed `a`, volatility `sigma`,
/// initial short rate `r0`, and an optional fitted initial discount curve.
#[derive(Debug, Clone)]
pub struct HullWhiteModel {
    pub a: f64,
    pub sigma: f64,
    pub r0: f64,
    pub curve_maturities: Option<Vec<f64>>,
    pub curve_discount_factors: Option<Vec<f64>>,
}

impl HullWhiteModel {
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

    /// Python `initial_discount_factor`.
    pub fn initial_discount_factor(&self, maturity: f64) -> f64 {
        match (&self.curve_maturities, &self.curve_discount_factors) {
            (Some(xp), Some(fp)) => linear_interpolate(maturity, xp, fp),
            _ => (-self.r0 * maturity).exp(),
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

    /// Python `theta(t)`.
    pub fn theta(&self, maturity: f64) -> f64 {
        let a = self.a.max(1e-12);
        let sigma = self.sigma.max(0.0);
        let bump = 1e-4;
        let fwd_up = self.instantaneous_forward_rate(maturity + bump);
        let fwd_down = self.instantaneous_forward_rate((maturity - bump).max(0.0));
        let dfdt = (fwd_up - fwd_down) / (2.0 * bump).max(1e-12);
        let forward = self.instantaneous_forward_rate(maturity);
        dfdt + a * forward + (sigma * sigma / (2.0 * a)) * (1.0 - (-2.0 * a * maturity).exp())
    }

    /// Python `zero_coupon_bond_price` — curve-consistent Hull-White ZCB
    /// (Brigo-Mercurio (3.39)) at t=0: P = P^M(0,T)·exp(B·(f^M(0,0) − r_t))
    /// with B = (1−e^{−aT})/a and f^M(0,0) the one-sided finite-difference
    /// initial forward (bump 1e-4). Replaces the previous zero-drift closed
    /// form, which did not reproduce the initial curve. When no explicit
    /// `r_t` is given and a curve has been fitted, the interpolated discount
    /// factor is returned.
    pub fn zero_coupon_bond_price(&self, maturity: f64, r_t: Option<f64>) -> f64 {
        if maturity <= 0.0 {
            return 1.0;
        }
        if r_t.is_none() && self.curve_maturities.is_some() {
            return self.initial_discount_factor(maturity);
        }
        let a = self.a.max(1e-12);
        let short_rate = r_t.unwrap_or(self.r0);
        let b_term = (1.0 - (-a * maturity).exp()) / a;
        let market_price = self.initial_discount_factor(maturity);
        let bump = 1e-4;
        let f0 = (-self.initial_discount_factor(bump).clamp(1e-12, f64::MAX).ln()
            + self.initial_discount_factor(0.0).clamp(1e-12, f64::MAX).ln())
            / bump;
        let a_term = market_price * (b_term * f0).exp();
        a_term * (-b_term * short_rate).exp()
    }

    /// Python `zero_coupon_yield`.
    pub fn zero_coupon_yield(&self, maturity: f64, r_t: Option<f64>) -> f64 {
        let price = self.zero_coupon_bond_price(maturity, r_t);
        -price.clamp(1e-12, f64::MAX).ln() / maturity.max(1e-12)
    }

    /// Python `simulate`: N draws from the exact terminal Gaussian of the
    /// Ornstein-Uhlenbeck short rate. Normals from
    /// `crate::rng::Xorshift64Star` with `seed` (2004 in tests).
    pub fn simulate(&self, t: f64, n: usize, seed: u64) -> Vec<f64> {
        let a = self.a.max(0.0);
        let sigma = self.sigma.max(0.0);
        let mean = self.r0 * (-a * t).exp();
        if sigma == 0.0 {
            return vec![mean; n];
        }
        let variance = sigma * sigma * (1.0 - (-2.0 * a * t).exp()) / (2.0 * a).max(1e-12);
        let std = variance.max(0.0).sqrt();
        let mut rng = crate::rng::Xorshift64Star::new(seed);
        (0..n).map(|_| mean + std * rng.next_std_normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Interpolation; reference from scripts/gen_hull_white.py, tol 1e-12.
    #[test]
    fn linear_interpolate_matches_reference() {
        let xp = [0.5, 1.0, 2.0, 5.0];
        let fp = [0.9753, 0.9512, 0.9048, 0.7788];
        assert!(rel(linear_interpolate(1.25, &xp, &fp), 0.9396) < 1e-12);
        assert!(rel(linear_interpolate(0.2, &xp, &fp), 0.9753) < 1e-12); // clamped low
        assert!(rel(linear_interpolate(7.0, &xp, &fp), 0.7788) < 1e-12); // clamped high
        assert!(rel(linear_interpolate(3.7, &xp, &fp), 0.8334) < 1e-12);
    }

    /// Curve-consistent bond price without a fitted curve (flat default):
    /// reproduces exp(-r0*T) to FD noise; reference from
    /// scripts/gen_hull_white.py, tol 1e-12.
    #[test]
    fn bond_price_closed_form() {
        let hw = HullWhiteModel::new(0.1, 0.01, 0.03);
        let p1 = hw.zero_coupon_bond_price(1.0, None);
        let p5 = hw.zero_coupon_bond_price(5.0, None);
        let p10 = hw.zero_coupon_bond_price(10.0, None);
        assert!(rel(p1, 0.9704455335486447) < 1e-12, "p1={p1}");
        assert!(rel(p5, 0.8607079764255583) < 1e-12, "p5={p5}");
        assert!(rel(p10, 0.7408182206824101) < 1e-12, "p10={p10}");
        // sanity: curve-consistent means flat curve -> exp(-r0*T)
        assert!(rel(p1, (-0.03_f64).exp()) < 1e-11, "p1 vs exp(-r0T)={p1}");
    }

    /// With a flat fitted curve the price equals the discount factor.
    #[test]
    fn bond_price_uses_fitted_curve() {
        let mut hw = HullWhiteModel::new(0.1, 0.01, 0.03);
        hw.fit_initial_discount_curve(vec![1.0, 5.0], vec![0.97, 0.86]);
        assert!(rel(hw.zero_coupon_bond_price(2.5, None), 0.92875) < 1e-12);
    }

    /// Forward rate and theta(t) on a flat r0 curve: forward == r0 and
    /// theta == a*r0 + (sigma^2/2a)(1-e^{-2at}); reference from
    /// scripts/gen_hull_white.py, tol 1e-9 (finite-difference round-off).
    #[test]
    fn forward_and_theta_flat_curve() {
        let hw = HullWhiteModel::new(0.1, 0.01, 0.03);
        let f = hw.instantaneous_forward_rate(2.0);
        let th = hw.theta(2.0);
        assert!(rel(f, 0.03) < 1e-9, "f={f}");
        // Reference from scripts/gen_hull_white.py (same finite-difference
        // recipe); tiny deviation from the analytic value comes from the
        // bump arithmetic.
        assert!(rel(th, 0.0031648418990315058) < 1e-12, "th={th}");
    }

    /// Stochastic sampler; expected values from scripts/gen_hull_white.py
    /// with refgen normals, seed 2004, tol 1e-12 relative.
    #[test]
    fn simulate_matches_reference() {
        let hw = HullWhiteModel::new(0.1, 0.01, 0.03);
        let r = hw.simulate(2.0, 8, 2004);
        let expected = [
            0.01823480828541723,
            0.04722897441969502,
            0.030845099769675633,
            0.040971256455761565,
            0.06182941469944912,
            -0.0029258789730068797,
            0.019132912054679173,
            0.03848256979958703,
        ];
        for (got, want) in r.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }
}
