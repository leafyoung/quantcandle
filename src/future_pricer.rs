//! Port of `torchquantlib/core/asset_pricing/money/future_pricer.py`.
//!
//! Theoretical futures price via cost of carry:
//! `F = spot * exp(carry_rate * expiry)` where `carry_rate = rate` unless a
//! `foreign_rate` is supplied, in which case `carry_rate = rate - foreign_rate`.

/// Theoretical futures price. Optional `foreign_rate` for cost-of-carry style
/// use cases (rate minus foreign rate).
pub fn future_pricer(spot: f64, rate: f64, expiry: f64, foreign_rate: Option<f64>) -> f64 {
    let carry_rate = match foreign_rate {
        None => rate,
        Some(f) => rate - f,
    };
    spot * (carry_rate * expiry).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from scripts/gen_misc_pricers.py (pure float64).
    #[test]
    fn test_future_pricer_no_foreign() {
        let actual = future_pricer(100.0, 0.05, 1.0, None);
        let expected = 105.12710963760242;
        assert!(
            (actual - expected).abs() <= 1e-8,
            "actual {actual} vs expected {expected}"
        );
    }

    #[test]
    fn test_future_pricer_with_foreign() {
        let actual = future_pricer(100.0, 0.05, 0.5, Some(0.02));
        let expected = 101.51130646157189;
        assert!(
            (actual - expected).abs() <= 1e-8,
            "actual {actual} vs expected {expected}"
        );
    }
}
