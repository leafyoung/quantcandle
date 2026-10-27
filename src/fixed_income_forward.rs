//! Port of `torchquantlib/core/asset_pricing/bond/fixed_income_forward.py`.
//!
//! Forward price of a fixed income security under continuous compounding:
//! `F = face_value * exp((forward_rate - rate) * time_to_maturity)`.

/// Forward price of a fixed income security (continuous compounding).
pub fn fixed_income_forward(
    face_value: f64,
    rate: f64,
    time_to_maturity: f64,
    forward_rate: f64,
) -> f64 {
    face_value * ((forward_rate - rate) * time_to_maturity).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected value from scripts/gen_misc_pricers.py (pure float64).
    #[test]
    fn test_fixed_income_forward() {
        let actual = fixed_income_forward(100.0, 0.05, 2.0, 0.03);
        let expected = 96.07894391523232;
        assert!(
            (actual - expected).abs() <= 1e-8,
            "actual {actual} vs expected {expected}"
        );
    }
}
