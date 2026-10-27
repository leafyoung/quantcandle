//! Port of `black_scholes_merton.py` (torchquantlib core/asset_pricing/option).
//!
//! Closed-form scalar math in f64. Also provides the crate-shared
//! high-accuracy `erf` / `normal_cdf` (double precision: Taylor series for
//! |x| < 2.5, trapezoid-quadrature erfc tail for larger |x|), used by the other
//! option modules.

/// Error function, double precision (~1e-15). Series for |x| < 2.5,
/// quadrature erfc tail for larger |x|.
pub fn erf(x: f64) -> f64 {
    let sqrt_pi = std::f64::consts::PI.sqrt();
    if x.abs() < 2.5 {
        // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! * (2n+1))
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0u64;
        while n < 300 {
            n += 1;
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-18 * (sum.abs() + 1.0) || term.abs() < 1e-300 {
                break;
            }
        }
        2.0 / sqrt_pi * sum
    } else if x > 0.0 {
        1.0 - erfc_tail(x)
    } else {
        erfc_tail(-x) - 1.0
    }
}

/// erfc(x) for x >= 2.5 via the substitution
/// erfc(x) = 2*exp(-x^2)/sqrt(pi) * int_0^inf exp(-2*x*s - s^2) ds,
/// evaluated with a trapezoid rule (spectrally accurate for this
/// super-exponentially decaying integrand; ~1e-16 relative).
fn erfc_tail(x: f64) -> f64 {
    let h = 0.01;
    let smax = 30.0 / x;
    let n = (smax / h).ceil() as usize;
    let mut sum = 0.0;
    for k in 0..=n {
        let s = k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        sum += w * (-2.0 * x * s - s * s).exp();
    }
    2.0 * (-x * x).exp() / std::f64::consts::PI.sqrt() * sum * h
}

/// Standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Zero-volatility value: discounted max(forward - strike, 0) style payoff.
fn deterministic_option_value(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    dividend: f64,
) -> f64 {
    let forward_spot = spot * ((rate - dividend) * expiry).exp();
    let discount_factor = (-rate * expiry).exp();
    if option_type == "call" {
        discount_factor * (forward_spot - strike).max(0.0)
    } else {
        discount_factor * (strike - forward_spot).max(0.0)
    }
}

/// Price a Black-Scholes-Merton option.
///
/// `option_style = "american"` uses the same lower-bounded approximation as
/// the Python source: max(european price, intrinsic value).
pub fn black_scholes_merton(
    option_type: &str,
    option_style: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    dividend: f64,
) -> f64 {
    if option_type != "call" && option_type != "put" {
        panic!("option_type must be either 'call' or 'put'.");
    }
    if option_style != "european" && option_style != "american" {
        panic!("option_style must be either 'european' or 'american'.");
    }

    let european_price = if volatility <= 0.0 {
        deterministic_option_value(option_type, spot, strike, expiry, rate, dividend)
    } else {
        let sqrt_expiry = expiry.sqrt();
        let d1 = ((spot / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * expiry)
            / (volatility * sqrt_expiry);
        let d2 = d1 - volatility * sqrt_expiry;
        let nd1 = normal_cdf(d1);
        let nd2 = normal_cdf(d2);
        if option_type == "call" {
            spot * (-dividend * expiry).exp() * nd1 - strike * (-rate * expiry).exp() * nd2
        } else {
            strike * (-rate * expiry).exp() * (1.0 - nd2)
                - spot * (-dividend * expiry).exp() * (1.0 - nd1)
        }
    };

    if option_style == "european" {
        return european_price;
    }
    let intrinsic_value = if option_type == "call" {
        (spot - strike).max(0.0)
    } else {
        (strike - spot).max(0.0)
    };
    european_price.max(intrinsic_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_black_scholes_merton.py (pure f64).
    #[test]
    fn european_call_put() {
        assert!(
            (black_scholes_merton("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
                - 9.227005508154036)
                .abs()
                < 1e-9
        );
        assert!(
            (black_scholes_merton("put", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
                - 6.330080627549918)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn american_style_is_max_of_european_and_intrinsic() {
        // ITM put where intrinsic dominates.
        let v = black_scholes_merton("put", "american", 80.0, 100.0, 1.0, 0.1, 0.05, 0.0);
        assert!((v - 20.0).abs() < 1e-9);
        // OTM call: american == european (intrinsic = 0 < european).
        let c = black_scholes_merton("call", "american", 90.0, 100.0, 1.0, 0.3, 0.02, 0.0);
        assert!((c - 7.64638494703226).abs() < 1e-9);
    }

    #[test]
    fn zero_volatility_deterministic() {
        let v = black_scholes_merton("call", "european", 100.0, 90.0, 2.0, 0.0, 0.03, 0.01);
        assert!((v - 13.261059308093147).abs() < 1e-9);
        let p = black_scholes_merton("put", "european", 100.0, 100.0, 2.0, 0.0, 0.03, 0.01);
        assert!(p.abs() < 1e-12);
    }

    #[test]
    fn normal_cdf_accuracy() {
        // Reference: scipy.stats.norm.cdf.
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-14);
        assert!((normal_cdf(1.0) - 0.8413447460685429).abs() < 1e-14);
        assert!((normal_cdf(-2.5) - 0.006209665325776132).abs() < 1e-13);
        assert!((normal_cdf(6.0) - 0.9999999990134123).abs() < 1e-12);
    }
}
