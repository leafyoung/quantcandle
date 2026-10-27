//! Port of `market_risk/stress_testing.py`.
//!
//! Applies percentage stress scenarios: portfolio * (1 + scenario).

/// Single portfolio value under a percentage stress scenario.
pub fn stress_test(portfolio_value: f64, stress_scenario: f64) -> f64 {
    portfolio_value * (1.0 + stress_scenario)
}

/// A vector of portfolio values under a vector of percentage scenarios.
pub fn stress_test_vec(portfolio_values: &[f64], stress_scenarios: &[f64]) -> Vec<f64> {
    portfolio_values
        .iter()
        .zip(stress_scenarios.iter())
        .map(|(v, s)| v * (1.0 + s))
        .collect()
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 `stress_test` from the Python source.
    #[test]
    fn matches_python_reference() {
        let got = super::stress_test(1e6, -0.35);
        assert!((got - 650000.0).abs() < 1e-6);
        let got = super::stress_test_vec(&[1e6, 2e6], &[0.2, -0.3]);
        assert!((got[0] - 1200000.0).abs() < 1e-6);
        assert!((got[1] - 1400000.0).abs() < 1e-6);
    }
}
