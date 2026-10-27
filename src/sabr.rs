//! Port of `torchquantlib/models/stochastic_volatility/sabr.py` (SABR model
//! with Hagan implied-volatility asymptotics, Black-76 and Bachelier
//! pricers, plus the Euler terminal-forward sampler from `simulate`).
//! Module RNG seed: 2008.

const EPS: f64 = 1e-8;
const ATM_TOL: f64 = 1e-7;

/// Gauss error function (portable; `f64::erf` is still unstable).
/// Maclaurin series for |x| < 3, DLMF 7.9 continued fraction beyond.
pub(crate) fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let ax = x.abs();
    let r = if ax < 3.0 {
        // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1))
        let x2 = x * x;
        let mut term = ax;
        let mut sum = ax;
        let mut n = 1u32;
        loop {
            term *= -x2 / n as f64;
            let add = term / (2.0 * n as f64 + 1.0);
            sum += add;
            if add.abs() <= 1e-18 * sum.abs() || n > 300 {
                break;
            }
            n += 1;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        1.0 - erfc(ax)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// Complementary error function via the Laplace continued fraction
/// (DLMF 7.9): sqrt(pi) e^{x^2} erfc(x) = x + (1/2)/(x + 1/(x + (3/2)/(x+...))).
fn erfc(x: f64) -> f64 {
    let mut val = x;
    for n in (1..=200).rev() {
        val = x + (n as f64 / 2.0) / val;
    }
    (-x * x).exp() / (std::f64::consts::PI.sqrt() * val)
}

/// Standard normal CDF.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Python `_black76_price`.
pub fn black76_price(
    option_type: &str,
    forward: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
) -> f64 {
    let discount_factor = (-rate * expiry).exp();
    let intrinsic_value = if option_type == "call" {
        (forward - strike).max(0.0)
    } else {
        (strike - forward).max(0.0)
    };
    if expiry <= 0.0 || volatility <= 0.0 {
        return discount_factor * intrinsic_value;
    }
    let sqrt_expiry = expiry.sqrt();
    let volatility_term = volatility * sqrt_expiry;
    let d1 = ((forward / strike).ln() + 0.5 * volatility * volatility * expiry) / volatility_term;
    let d2 = d1 - volatility_term;
    if option_type == "call" {
        discount_factor * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    } else {
        discount_factor * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))
    }
}

/// Python `_bachelier_price`.
pub fn bachelier_price(
    option_type: &str,
    forward: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
) -> f64 {
    let discount_factor = (-rate * expiry).exp();
    let intrinsic_value = if option_type == "call" {
        (forward - strike).max(0.0)
    } else {
        (strike - forward).max(0.0)
    };
    if expiry <= 0.0 || volatility <= 0.0 {
        return discount_factor * intrinsic_value;
    }
    let sqrt_expiry = expiry.sqrt();
    let denominator = (volatility * sqrt_expiry).max(EPS);
    let d = (forward - strike) / denominator;
    let call_price = discount_factor
        * ((forward - strike) * normal_cdf(d) + volatility * sqrt_expiry * normal_pdf(d));
    if option_type == "call" {
        call_price
    } else {
        call_price - discount_factor * (forward - strike)
    }
}

/// SABR parameters (already-constrained, as `constrained_param` returns).
#[derive(Debug, Clone, Copy)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    pub f0: f64,
    pub shift: f64,
}

impl Default for SabrParams {
    fn default() -> Self {
        Self { alpha: 0.2, beta: 0.5, rho: 0.0, nu: 0.3, f0: 100.0, shift: 0.0 }
    }
}

fn resolve_shift(params: &SabrParams, shift: Option<f64>) -> f64 {
    shift.unwrap_or(params.shift)
}

/// Python `_base_forward_from_state`: F0 (or given level), grown by the
/// cost of carry when a rate is supplied.
pub fn base_forward(params: &SabrParams, base_level: Option<f64>, expiry: f64, rate: Option<f64>, dividend: f64) -> f64 {
    let forward = base_level.unwrap_or(params.f0);
    match rate {
        None => forward,
        Some(r) => forward * ((r - dividend) * expiry).exp(),
    }
}

/// Python `_atm_lognormal_implied_volatility`.
pub fn atm_lognormal_implied_volatility(p: &SabrParams, forward: f64, expiry: f64) -> f64 {
    let forward = forward.max(EPS);
    let one_minus_beta = 1.0 - p.beta;
    let forward_power = forward.powf(one_minus_beta);
    let correction = ((one_minus_beta * one_minus_beta) / 24.0)
        * (p.alpha * p.alpha / (forward.powf(2.0 * one_minus_beta)).max(EPS))
        + (p.rho * p.beta * p.nu * p.alpha) / (4.0 * forward_power.max(EPS))
        + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu;
    ((p.alpha / forward_power.max(EPS)) * (1.0 + correction * expiry)).max(EPS)
}

/// Python `_hagan_lognormal_implied_volatility`.
pub fn hagan_lognormal_implied_volatility(
    p: &SabrParams,
    strike: f64,
    expiry: f64,
    forward: Option<f64>,
    shift: Option<f64>,
) -> f64 {
    if expiry <= 0.0 {
        return 0.0;
    }
    let base_forward = base_forward(p, Some(forward.unwrap_or(p.f0)), expiry, None, 0.0);
    let shift_v = resolve_shift(p, shift);
    let shifted_forward = (base_forward + shift_v).max(EPS);
    let shifted_strike = (strike + shift_v).max(EPS);

    let log_fk = (shifted_forward / shifted_strike).ln();
    if log_fk.abs() < ATM_TOL {
        return atm_lognormal_implied_volatility(p, shifted_forward, expiry);
    }

    let one_minus_beta = 1.0 - p.beta;
    let fk_beta = ((shifted_forward * shifted_strike).powf(0.5 * one_minus_beta)).max(EPS);
    let log_fk_sq = log_fk * log_fk;
    let log_fk_4 = log_fk_sq * log_fk_sq;

    let z = (p.nu / p.alpha.max(EPS)) * fk_beta * log_fk;
    let sqrt_term = (1.0 - 2.0 * p.rho * z + z * z).max(EPS).sqrt();
    let x_z = ((sqrt_term + z - p.rho) / (1.0 - p.rho)).max(EPS).ln();

    let log_denominator =
        1.0 + ((one_minus_beta * one_minus_beta) / 24.0) * log_fk_sq
            + ((one_minus_beta * one_minus_beta * one_minus_beta * one_minus_beta) / 1920.0) * log_fk_4;
    let prefactor = p.alpha / (fk_beta * log_denominator).max(EPS);
    let correction = ((one_minus_beta * one_minus_beta) / 24.0)
        * (p.alpha * p.alpha / (shifted_forward * shifted_strike).powf(one_minus_beta).max(EPS))
        + (p.rho * p.beta * p.nu * p.alpha) / (4.0 * fk_beta.max(EPS))
        + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu;

    let ratio = if x_z.abs() < ATM_TOL { 1.0 } else { z / x_z };
    (prefactor * ratio * (1.0 + correction * expiry)).max(EPS)
}

/// Python `_atm_normal_implied_volatility`.
pub fn atm_normal_implied_volatility(p: &SabrParams, forward: f64, expiry: f64) -> f64 {
    let forward = forward.max(EPS);
    let forward_beta = forward.powf(p.beta);
    let one_minus_beta = 1.0 - p.beta;
    let correction = -p.beta * (2.0 - p.beta) * p.alpha * p.alpha
        / (24.0 * (forward.powf(2.0 * one_minus_beta)).max(EPS))
        + (p.rho * p.beta * p.nu * p.alpha) / (4.0 * (forward.powf(one_minus_beta)).max(EPS))
        + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu;
    (p.alpha * forward_beta * (1.0 + correction * expiry)).max(EPS)
}

/// Python `_hagan_normal_implied_volatility`.
pub fn hagan_normal_implied_volatility(
    p: &SabrParams,
    strike: f64,
    expiry: f64,
    forward: Option<f64>,
    shift: Option<f64>,
) -> f64 {
    if expiry <= 0.0 {
        return 0.0;
    }
    let base_forward = base_forward(p, Some(forward.unwrap_or(p.f0)), expiry, None, 0.0);
    let shift_v = resolve_shift(p, shift);
    let shifted_forward = base_forward + shift_v;
    let shifted_strike = strike + shift_v;

    let diff = shifted_forward - shifted_strike;
    if diff.abs() < ATM_TOL {
        return atm_normal_implied_volatility(p, shifted_forward.max(EPS), expiry);
    }

    let fk_beta = ((shifted_forward * shifted_strike).max(EPS).powf(0.5 * p.beta)).max(EPS);
    let zeta = (p.nu / p.alpha.max(EPS)) * diff / fk_beta;
    let sqrt_term = (1.0 - 2.0 * p.rho * zeta + zeta * zeta).max(EPS).sqrt();
    let x_zeta = ((sqrt_term + zeta - p.rho) / (1.0 - p.rho)).max(EPS).ln();

    let prod = (shifted_forward * shifted_strike).max(EPS);
    let correction = -p.beta * (2.0 - p.beta) * p.alpha * p.alpha
        / (24.0 * (prod.powf(1.0 - p.beta)).max(EPS))
        + (p.rho * p.beta * p.nu * p.alpha) / (4.0 * (prod.powf(0.5 * (1.0 - p.beta))).max(EPS))
        + ((2.0 - 3.0 * p.rho * p.rho) / 24.0) * p.nu * p.nu;
    let ratio = if x_zeta.abs() < ATM_TOL { 1.0 } else { diff / x_zeta };
    (p.alpha * ratio * (1.0 + correction * expiry)).max(EPS)
}

/// Python `SABR.implied_volatility`.
pub fn implied_volatility(
    p: &SabrParams,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    model: &str,
    shift: Option<f64>,
) -> f64 {
    let forward = base_forward(p, None, t, Some(r), q);
    match model.to_ascii_lowercase().as_str() {
        "lognormal" => hagan_lognormal_implied_volatility(p, k, t, Some(forward), shift),
        "normal" => hagan_normal_implied_volatility(p, k, t, Some(forward), shift),
        _ => panic!("model must be 'lognormal' or 'normal'"),
    }
}

/// Python `SABR.simulate`: Euler-style SABR sampler for terminal forwards.
/// Per step draws a block of N spot normals followed by N auxiliary normals
/// (matching the Python order). Seed 2008 in tests.
pub fn simulate(p: &SabrParams, s0: Option<f64>, t: f64, n: usize, steps: usize, seed: u64) -> Vec<f64> {
    let steps = steps.max(1);
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_residual = (1.0 - p.rho * p.rho).clamp(EPS, f64::MAX).sqrt();

    let mut forward = vec![(base_forward(p, s0, t, None, 0.0)).max(EPS); n];
    let mut alpha_t = vec![p.alpha; n];
    let mut rng = crate::rng::Xorshift64Star::new(seed);

    for _ in 0..steps {
        let z1: Vec<f64> = (0..n).map(|_| rng.next_std_normal()).collect();
        let z2: Vec<f64> = (0..n).map(|_| rng.next_std_normal()).collect();
        for i in 0..n {
            let z_alpha = p.rho * z1[i] + rho_residual * z2[i];
            let forward_base = forward[i].max(EPS);
            let diffusion = alpha_t[i] * forward_base.powf(p.beta) * sqrt_dt * z1[i];
            forward[i] = (forward[i] + diffusion).max(EPS);
            alpha_t[i] = (alpha_t[i] * (-0.5 * p.nu * p.nu * dt + p.nu * sqrt_dt * z_alpha).exp()).max(EPS);
        }
    }
    forward
}

/// Python `SABR.option_price`.
pub fn option_price(
    p: &SabrParams,
    k: f64,
    t: f64,
    r: f64,
    option_type: &str,
    q: f64,
    model: &str,
    shift: Option<f64>,
) -> f64 {
    let option_style = option_type.to_ascii_lowercase();
    assert!(option_style == "call" || option_style == "put", "option_type must be 'call' or 'put'");
    let forward = base_forward(p, None, t, Some(r), q);
    match model.to_ascii_lowercase().as_str() {
        "lognormal" => {
            let shift_v = resolve_shift(p, shift);
            let shifted_forward = forward + shift_v;
            let shifted_strike = k + shift_v;
            let iv = hagan_lognormal_implied_volatility(p, k, t, Some(forward), shift);
            black76_price(&option_style, shifted_forward, shifted_strike, t, iv, r)
        }
        "normal" => {
            let shift_v = resolve_shift(p, shift);
            let shifted_forward = forward + shift_v;
            let shifted_strike = k + shift_v;
            let iv = hagan_normal_implied_volatility(p, k, t, Some(forward), shift);
            bachelier_price(&option_style, shifted_forward, shifted_strike, t, iv, r)
        }
        _ => panic!("model must be 'lognormal' or 'normal'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SabrParams {
        SabrParams::default()
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Hagan implied vols; reference from scripts/gen_sabr.py (pure float64
    /// replication of the torch code), tol 1e-12 relative.
    #[test]
    fn implied_vols_match_reference() {
        let p = params();
        let iv_ln = implied_volatility(&p, 110.0, 1.0, 0.0, 0.0, "lognormal", None);
        let iv_ln_atm = implied_volatility(&p, 100.0, 1.0, 0.0, 0.0, "lognormal", None);
        let iv_n = implied_volatility(&p, 110.0, 1.0, 0.0, 0.0, "normal", None);
        let iv_n_atm = implied_volatility(&p, 100.0, 1.0, 0.0, 0.0, "normal", None);
        assert!(rel(iv_ln, 0.02452109699988402) < 1e-12, "iv_ln={iv_ln}");
        assert!(rel(iv_ln_atm, 0.020150083333333332) < 1e-12, "iv_ln_atm={iv_ln_atm}");
        assert!(rel(iv_n, 1.7148575695087824) < 1e-12, "iv_n={iv_n}");
        assert!(rel(iv_n_atm, 2.014975) < 1e-12, "iv_n_atm={iv_n_atm}");
    }

    /// Option prices (Black-76 / Bachelier on shifted inputs), with a rate
    /// and a shift; reference from scripts/gen_sabr.py, tol 1e-12.
    #[test]
    fn option_prices_match_reference() {
        let p = params();
        let call_ln = option_price(&p, 105.0, 1.0, 0.03, "call", 0.0, "lognormal", None);
        let put_ln = option_price(&p, 95.0, 1.0, 0.03, "put", 0.0, "lognormal", None);
        let call_n = option_price(&p, 105.0, 1.0, 0.03, "call", 0.0, "normal", None);
        let shifted = option_price(&p, 0.01, 1.0, 0.0, "call", 0.0, "lognormal", Some(0.05));
        assert!(rel(call_ln, 0.1890807767303359) < 1e-12, "call_ln={call_ln}");
        assert!(rel(put_ln, 0.00020839856139982785) < 1e-8, "put_ln={put_ln}");
        assert!(rel(call_n, 0.04857832011877204) < 1e-12, "call_n={call_n}");
        assert!(rel(shifted, 99.99) < 1e-12, "shifted={shifted}");
    }

    /// Stochastic terminal forwards; expected values from
    /// scripts/gen_sabr.py with refgen normals, seed 2008, tol 1e-12.
    #[test]
    fn simulate_matches_reference() {
        let p = params();
        let out = simulate(&p, None, 1.0, 6, 32, 2008);
        let expected = [
            104.35503149972567,
            99.53200457084024,
            99.22584645372046,
            101.30415922762192,
            99.03245350317181,
            100.59543369067899,
        ];
        for (got, want) in out.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-11, "got={got} want={want}");
        }
    }
}
