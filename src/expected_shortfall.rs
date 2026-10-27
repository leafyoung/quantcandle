//! Port of `market_risk/expected_shortfall.py`.
//!
//! Expected Shortfall / CVaR: mean of losses at or beyond historical VaR
//! (plus a tiny 1e-6 * VaR stabilizer exactly as in the Python source);
//! zero when no loss reaches VaR.

use crate::var::{calculate_var, validate_confidence_level, validate_returns};

/// Expected Shortfall for a 1-D return series.
pub fn calculate_es(returns: &[f64], confidence_level: f64) -> Result<f64, String> {
    validate_confidence_level(confidence_level)?;
    validate_returns(returns)?;
    let losses: Vec<f64> = returns.iter().map(|r| (-r).max(0.0)).collect();
    let var = calculate_var(returns, confidence_level)?;
    let tail: Vec<f64> = losses.iter().copied().filter(|l| *l >= var).collect();
    if tail.is_empty() {
        return Ok(0.0);
    }
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    Ok(mean + 1e-6 * var)
}

/// Expected Shortfall along the last dimension of a 2-D return matrix.
pub fn calculate_es_2d(returns: &[Vec<f64>], confidence_level: f64) -> Result<Vec<f64>, String> {
    validate_confidence_level(confidence_level)?;
    returns
        .iter()
        .map(|row| calculate_es(row, confidence_level))
        .collect()
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 `calculate_es` from the Python source.
    #[test]
    fn matches_python_reference() {
        let r: Vec<f64> = vec![
            0.01, -0.02, 0.015, -0.005, 0.03, -0.012, 0.008, -0.025, 0.001, -0.018,
        ];
        let cases = [
            (0.90_f64, 0.0250000205_f64),
            (0.95, 0.02500002275),
            (0.99, 0.025000024550000003),
        ];
        for (q, expected) in cases {
            let got = super::calculate_es(&r, q).unwrap();
            assert!((got - expected).abs() < 1e-14, "q={q}: {got} vs {expected}");
        }
        let r2 = vec![
            vec![0.01, -0.05, 0.02, -0.03],
            vec![0.04, -0.01, -0.02, 0.005],
        ];
        let got = super::calculate_es_2d(&r2, 0.75).unwrap();
        assert!((got[0] - 0.050000035000000005).abs() < 1e-14);
        assert!((got[1] - 0.0200000125).abs() < 1e-14);
    }

    /// No losses at/above VaR -> zero (Python returns a zero scalar).
    #[test]
    fn positive_returns_zero_es() {
        let got = super::calculate_es(&[0.01, 0.02, 0.005], 0.95).unwrap();
        assert_eq!(got, 0.0);
    }
}
