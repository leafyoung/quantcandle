//! Port of `market_risk/var.py`.
//!
//! Historical-simulation VaR: losses = max(-returns, 0), then a linearly
//! interpolated (torch.quantile default) quantile over the last dimension.

/// Validates the confidence level is strictly between 0 and 1.
pub fn validate_confidence_level(confidence_level: f64) -> Result<(), String> {
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err("Confidence level must be between 0 and 1".to_string());
    }
    Ok(())
}

/// Validates a non-empty return series.
pub fn validate_returns(returns: &[f64]) -> Result<(), String> {
    if returns.is_empty() {
        return Err("returns must not be empty.".to_string());
    }
    Ok(())
}

/// torch.quantile default ("linear") interpolation on ascending input:
/// position q*(n-1) with linear interpolation between neighbors.
fn quantile_linear(mut xs: Vec<f64>, q: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    xs[lo] + frac * (xs[hi] - xs[lo])
}

/// Historical VaR for a 1-D return series.
pub fn calculate_var(returns: &[f64], confidence_level: f64) -> Result<f64, String> {
    validate_confidence_level(confidence_level)?;
    validate_returns(returns)?;
    let losses: Vec<f64> = returns.iter().map(|r| (-r).max(0.0)).collect();
    Ok(quantile_linear(losses, confidence_level))
}

/// Historical VaR along the last dimension of a 2-D return matrix.
pub fn calculate_var_2d(returns: &[Vec<f64>], confidence_level: f64) -> Result<Vec<f64>, String> {
    validate_confidence_level(confidence_level)?;
    returns
        .iter()
        .map(|row| calculate_var(row, confidence_level))
        .collect()
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 `calculate_var` from the Python source.
    #[test]
    fn matches_python_reference() {
        let r: Vec<f64> = vec![
            0.01, -0.02, 0.015, -0.005, 0.03, -0.012, 0.008, -0.025, 0.001, -0.018,
        ];
        let cases = [
            (0.90_f64, 0.020499999999999997_f64),
            (0.95, 0.022749999999999996),
            (0.99, 0.024550000000000002),
        ];
        for (q, expected) in cases {
            let got = super::calculate_var(&r, q).unwrap();
            assert!((got - expected).abs() < 1e-12, "q={q}: {got} vs {expected}");
        }
        let r2 = vec![
            vec![0.01, -0.05, 0.02, -0.03],
            vec![0.04, -0.01, -0.02, 0.005],
        ];
        let got = super::calculate_var_2d(&r2, 0.75).unwrap();
        assert!((got[0] - 0.035).abs() < 1e-12);
        assert!((got[1] - 0.0125).abs() < 1e-12);
    }

    /// Positive-return-only samples map to zero VaR.
    #[test]
    fn positive_returns_zero_var() {
        let got = super::calculate_var(&[0.01, 0.02, 0.005], 0.95).unwrap();
        assert!(got.abs() < 1e-15);
    }

    /// Validation errors mirror the Python checks.
    #[test]
    fn validation() {
        assert!(super::calculate_var(&[0.1], 1.0).is_err());
        assert!(super::calculate_var(&[0.1], 0.0).is_err());
        assert!(super::calculate_var(&[], 0.95).is_err());
    }
}
