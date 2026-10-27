//! Port of `bermudan_option.py` (torchquantlib core/asset_pricing/option).
//!
//! Thin wrapper around [`crate::option_utils::price_binomial_tree`] with
//! `option_style = "bermudan"`; `exercise_dates` are the step indices
//! (0 <= step < steps) at which early exercise is allowed.

/// Price a Bermudan option with specified exercise steps.
pub fn bermudan_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
    exercise_dates: &[usize],
) -> f64 {
    if volatility <= 0.0 {
        panic!("volatility must be positive for bermudan_option.");
    }
    crate::option_utils::price_binomial_tree(
        option_type,
        "bermudan",
        spot,
        strike,
        expiry,
        volatility,
        rate,
        steps,
        Some(exercise_dates),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_bermudan_option.py (pure f64).
    #[test]
    fn bermudan_put_quarterly_exercise() {
        // put, S=100 K=100 T=1 vol=0.2 r=0.05, 50 steps, exercise at
        // steps 12, 24, 36.
        assert!(
            (bermudan_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 50, &[12, 24, 36])
                - 5.93608249055333)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn out_of_range_dates_are_filtered() {
        // Steps >= steps or negative-equivalent entries are ignored, as in the
        // Python filter 0 <= step < steps.
        let v_full = bermudan_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 50, &[12, 24, 36]);
        let v_extra = bermudan_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 50, &[12, 24, 36, 50, 99]);
        assert!((v_full - v_extra).abs() < 1e-12);
        // No valid dates -> European value.
        let v_euro = bermudan_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 50, &[]);
        assert!((v_euro - 5.533633990803824).abs() < 1e-9);
    }

    #[test]
    fn bermudan_call() {
        assert!(
            (bermudan_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 50, &[10, 20, 30, 40])
                - 10.410691540732644)
                .abs()
                < 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "volatility must be positive")]
    fn zero_volatility_panics() {
        bermudan_option("put", 100.0, 100.0, 1.0, 0.0, 0.05, 10, &[5]);
    }
}
