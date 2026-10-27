//! Port of `torchquantlib/models/binomial_tree.py` (plus the
//! `price_binomial_tree` core it calls from
//! `torchquantlib/core/asset_pricing/option/utils.py`).
//!
//! Prices European / American / Bermudan options with a CRR binomial tree.
//! The tree core is the single shared implementation in
//! [`crate::option_utils`] (the Rust home of the Python `utils.py` this
//! module calls into), re-exported here under the module's Python names.
//! The Python batch wrapper loops over 1-D batch elements independently;
//! `binomial_tree_batch` does the same.

pub use crate::option_utils::calculate_binomial_tree_params;

/// Python `option_payoff` (scalar form used by this module).
pub fn option_payoff(option_type: &str, terminal_prices: f64, strike: f64) -> f64 {
    match option_type {
        "call" => (terminal_prices - strike).max(0.0),
        "put" => (strike - terminal_prices).max(0.0),
        _ => panic!("Invalid option type. Use 'call' or 'put'."),
    }
}

/// Python `price_binomial_tree` for a scalar spot. `exercise_dates` holds
/// step indices at which early exercise is allowed (Bermudan); ignored for
/// European/American styles, matching the Python.
pub fn price_binomial_tree(
    option_type: &str,
    option_style: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
    exercise_dates: Option<&[usize]>,
) -> f64 {
    crate::option_utils::price_binomial_tree(
        option_type,
        option_style,
        spot,
        strike,
        expiry,
        volatility,
        rate,
        steps,
        exercise_dates,
    )
}

/// Python `binomial_tree` batch wrapper: evaluate each batch element
/// independently. All slices must have the same length (scalar expansion is
/// handled by the caller in Python via broadcasting; here we accept either a
/// single-element slice broadcast across the batch or a full-length slice).
pub fn binomial_tree_batch(
    option_type: &str,
    option_style: &str,
    spot: &[f64],
    strike: &[f64],
    expiry: &[f64],
    volatility: &[f64],
    rate: &[f64],
    steps: usize,
) -> Vec<f64> {
    crate::option_utils::binomial_tree_batch(
        option_type, option_style, spot, strike, expiry, volatility, rate, steps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic closed-form tree results. Reference values from the
    /// torch implementation (`price_binomial_tree`), f64 re-computation in
    /// `scripts/gen_binomial_tree_model.py`; tolerance 1e-12 relative.
    #[test]
    fn european_call_put() {
        let call = price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100, None);
        let put = price_binomial_tree("put", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100, None);
        assert!((call - 10.430611662249113).abs() < 1e-11, "call={call}");
        assert!((put - 5.553554112321353).abs() < 1e-11, "put={put}");
    }

    #[test]
    fn american_put_early_exercise_premium() {
        let euro = price_binomial_tree("put", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 100, None);
        let amer = price_binomial_tree("put", "american", 100.0, 100.0, 1.0, 0.2, 0.05, 100, None);
        assert!((amer - 6.082354409142444).abs() < 1e-11, "amer={amer}");
        assert!(amer > euro); // early-exercise premium
    }

    #[test]
    fn bermudan_between_euro_and_american() {
        let dates: Vec<usize> = (0..100).step_by(10).collect();
        let berm = price_binomial_tree(
            "put", "bermudan", 100.0, 100.0, 1.0, 0.2, 0.05, 100, Some(&dates),
        );
        assert!((berm - 6.025795352002811).abs() < 1e-11, "berm={berm}");
    }

    #[test]
    fn zero_volatility_deterministic() {
        let v = price_binomial_tree("call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 50, None);
        let expected = (-0.05_f64).exp() * (100.0 * 0.05_f64.exp() - 90.0);
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn zero_expiry_intrinsic() {
        let v = price_binomial_tree("call", "european", 110.0, 100.0, 0.0, 0.2, 0.05, 50, None);
        assert!((v - 10.0).abs() < 1e-12);
    }

    #[test]
    fn batch_matches_scalar() {
        let out = binomial_tree_batch(
            "call",
            "european",
            &[100.0, 110.0],
            &[100.0],
            &[1.0],
            &[0.2],
            &[0.05],
            100,
        );
        assert_eq!(out.len(), 2);
        assert!((out[0] - 10.430611662249113).abs() < 1e-11);
    }
}
