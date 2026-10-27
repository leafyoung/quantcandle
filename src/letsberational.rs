//! Port of `torchquantlib/core/asset_pricing/letsberational.py`.
//!
//! Black-Scholes helpers: normal CDF/PDF, Black-Scholes price, and implied
//! volatility via robust bisection on [1e-5, 5.0] with price tolerance 1e-8.
//!
//! Note (faithful port of a source quirk): the bisection always prices a CALL
//! internally, even when `is_call` is false — `is_call` only affects the
//! no-arbitrage upper bound check. All math is scalar `f64`.

/// Standard normal CDF: re-exported from [`crate::black_scholes_merton`]
/// (high-accuracy erf, ~1e-15) instead of the former local Abramowitz-Stegun
/// 7.1.26 approximation (~1.5e-7), mirroring the Python `torch.erf` accuracy.
pub use crate::black_scholes_merton::normal_cdf;

/// Standard normal probability density function (coefficient 0.3989 as in the
/// Python source).
pub fn normal_pdf(x: f64) -> f64 {
    0.3989 * (-0.5 * x * x).exp()
}

fn black_scholes_price(
    volatility: f64,
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    is_call: bool,
) -> f64 {
    let sqrt_expiry = expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * expiry)
        / (volatility * sqrt_expiry);
    let d2 = d1 - volatility * sqrt_expiry;
    let discount_factor = (-rate * expiry).exp();

    if is_call {
        spot * normal_cdf(d1) - strike * discount_factor * normal_cdf(d2)
    } else {
        strike * discount_factor * normal_cdf(-d2) - spot * normal_cdf(-d1)
    }
}

/// Estimate implied volatility with a robust bisection search.
///
/// Returns `Err(String)` on invalid inputs (non-positive price/spot/strike/
/// expiry, price outside no-arbitrage bounds, or non-convergence), mirroring
/// the `RuntimeError`s in the Python source.
pub fn implied_volatility(
    option_price: f64,
    spot: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
    is_call: bool,
) -> Result<f64, String> {
    if option_price <= 0.0 {
        return Err("option_price must be positive.".into());
    }
    if spot <= 0.0 || strike <= 0.0 || expiry <= 0.0 {
        return Err("spot, strike, and expiry must be positive.".into());
    }

    let discount_factor = (-rate * expiry).exp();
    let max_price = if is_call { spot } else { strike * discount_factor };
    if option_price >= max_price {
        return Err("option_price is outside no-arbitrage bounds.".into());
    }

    let mut lower = 1e-5f64;
    let mut upper = 5.0f64;
    let tolerance = 1e-8;

    let lower_price = black_scholes_price(lower, spot, strike, expiry, rate, true);
    if option_price <= lower_price {
        return Ok(lower);
    }

    for _ in 0..100 {
        let mid = 0.5 * (lower + upper);
        let model_price = black_scholes_price(mid, spot, strike, expiry, rate, true);
        let price_diff = model_price - option_price;

        if price_diff.abs() < tolerance {
            return Ok(mid);
        }

        if price_diff > 0.0 {
            upper = mid;
        } else {
            lower = mid;
        }
    }

    Err("implied_volatility failed to converge.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from scripts/gen_misc_pricers.py (pure float64).
    // normal_cdf now re-uses the high-accuracy crate::black_scholes_merton
    // erf (~1e-15 vs math.erf), so the implied-vol bisection follows the same
    // midpoint sequence as the pure-f64 Python reference and results agree to
    // ~1e-12; tolerances are correspondingly tight.

    #[test]
    fn test_normal_cdf() {
        let actual = normal_cdf(0.3);
        let expected = 0.6179114221889526;
        // High-accuracy shared erf: agrees with math.erf to ~1e-15.
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual {actual} vs expected {expected}"
        );
    }

    #[test]
    fn test_normal_pdf() {
        let actual = normal_pdf(0.3);
        let expected = 0.38134739550322355;
        assert!(
            (actual - expected).abs() <= 1e-8,
            "actual {actual} vs expected {expected}"
        );
    }

    #[test]
    fn test_implied_volatility_call() {
        let iv = implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, true).unwrap();
        let expected = 0.18797164959634072;
        assert!(
            (iv - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "actual {iv} vs expected {expected}"
        );
        // Round-trip: price at recovered vol reproduces the market price.
        let price = black_scholes_price(iv, 100.0, 100.0, 1.0, 0.05, true);
        assert!((price - 10.0).abs() <= 1e-8, "price {price}");
    }

    #[test]
    fn test_implied_volatility_put() {
        // Ported quirk: the bisection prices a call internally even for puts,
        // so this matches the Python reference value exactly.
        let iv = implied_volatility(8.0, 100.0, 95.0, 0.5, 0.03, false).unwrap();
        let expected = 0.14666327622914455;
        assert!(
            (iv - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "actual {iv} vs expected {expected}"
        );
    }

    #[test]
    fn test_implied_volatility_bounds_error() {
        assert!(implied_volatility(0.0, 100.0, 100.0, 1.0, 0.05, true).is_err());
        assert!(implied_volatility(200.0, 100.0, 100.0, 1.0, 0.05, true).is_err());
    }
}
