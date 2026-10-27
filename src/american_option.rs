//! Port of `american_option.py` (torchquantlib core/asset_pricing/option).
//!
//! Thin wrapper around [`crate::option_utils::price_binomial_tree`] with
//! `option_style = "american"`; requires positive volatility, matching the
//! Python `RuntimeError` with a panic.

/// Price an American option with a CRR binomial tree.
pub fn american_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
) -> f64 {
    if volatility <= 0.0 {
        panic!("volatility must be positive for american_option.");
    }
    crate::option_utils::price_binomial_tree(
        option_type, "american", spot, strike, expiry, volatility, rate, steps, None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_american_option.py (pure f64).
    #[test]
    fn american_put() {
        // put, S=100 K=100 T=1 vol=0.2 r=0.05, 50 steps
        assert!((american_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 50) - 6.073727985724901).abs() < 1e-9);
    }

    #[test]
    fn american_call() {
        // call, S=100 K=100 T=1 vol=0.2 r=0.05, 50 steps
        assert!((american_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 50) - 10.410691540732644).abs() < 1e-9);
    }

    #[test]
    fn deep_itm_put_hits_intrinsic() {
        // With few steps and deep ITM put, early exercise approximates intrinsic.
        let v = american_option("put", 80.0, 100.0, 1.0, 0.2, 0.05, 10);
        assert!((v - 20.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "volatility must be positive")]
    fn zero_volatility_panics() {
        american_option("put", 100.0, 100.0, 1.0, 0.0, 0.05, 10);
    }
}
