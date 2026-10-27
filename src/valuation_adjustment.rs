//! Port of `valuation_adjustment/valuation_adjustment.py`.
//!
//! X-value adjustments (CVA/DVA/FVA/MVA), all closed-form products of their
//! inputs, with the same non-negative / bounded validation as the Python.

fn validate_non_negative(name: &str, value: f64) -> Result<(), String> {
    if value < 0.0 {
        return Err(format!("{name} must be non-negative."));
    }
    Ok(())
}

fn validate_recovery_rate(recovery_rate: f64) -> Result<(), String> {
    validate_non_negative("recovery_rate", recovery_rate)?;
    if recovery_rate > 1.0 {
        return Err("recovery_rate must be between 0 and 1.".to_string());
    }
    Ok(())
}

/// Credit valuation adjustment.
pub fn calculate_cva(exposure: f64, default_prob: f64, recovery_rate: f64) -> Result<f64, String> {
    validate_non_negative("exposure", exposure)?;
    validate_non_negative("default_prob", default_prob)?;
    validate_recovery_rate(recovery_rate)?;
    Ok(exposure * (1.0 - recovery_rate) * default_prob)
}

/// Debit valuation adjustment (same formula as CVA in the Python source).
pub fn calculate_dva(exposure: f64, default_prob: f64, recovery_rate: f64) -> Result<f64, String> {
    calculate_cva(exposure, default_prob, recovery_rate)
}

/// Funding valuation adjustment.
pub fn calculate_fva(exposure: f64, funding_spread: f64, maturity: f64) -> Result<f64, String> {
    validate_non_negative("exposure", exposure)?;
    validate_non_negative("funding_spread", funding_spread)?;
    validate_non_negative("maturity", maturity)?;
    Ok(exposure * funding_spread * maturity)
}

/// Margin valuation adjustment.
pub fn calculate_mva(exposure: f64, funding_cost: f64, maturity: f64) -> Result<f64, String> {
    validate_non_negative("exposure", exposure)?;
    validate_non_negative("funding_cost", funding_cost)?;
    validate_non_negative("maturity", maturity)?;
    Ok(exposure * funding_cost * maturity)
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 from the Python source.
    #[test]
    fn matches_python_reference() {
        let got = super::calculate_cva(1e6, 0.03, 0.4).unwrap();
        assert!((got - 18000.0).abs() < 1e-6);
        let got = super::calculate_dva(1e6, 0.03, 0.4).unwrap();
        assert!((got - 18000.0).abs() < 1e-6);
        let got = super::calculate_fva(5e5, 0.012, 3.0).unwrap();
        assert!((got - 18000.0).abs() < 1e-6);
        let got = super::calculate_mva(2e5, 0.008, 2.5).unwrap();
        assert!((got - 4000.0).abs() < 1e-6);
    }

    /// Validation errors mirror the Python checks.
    #[test]
    fn validation() {
        assert!(super::calculate_cva(-1.0, 0.03, 0.4).is_err());
        assert!(super::calculate_cva(1e6, -0.03, 0.4).is_err());
        assert!(super::calculate_cva(1e6, 0.03, 1.5).is_err());
        assert!(super::calculate_fva(1.0, -0.1, 1.0).is_err());
        assert!(super::calculate_mva(1.0, 0.1, -1.0).is_err());
    }
}
