//! Port of `market_risk/scenario_analysis.py`.
//!
//! Applies scenario multipliers to a portfolio value (elementwise product).

/// Single portfolio value times a single scenario multiplier.
pub fn scenario_analysis(portfolio_value: f64, scenario: f64) -> f64 {
    portfolio_value * scenario
}

/// A vector of portfolio values times a vector of scenario multipliers
/// (elementwise product, as torch broadcasting over equal shapes).
pub fn scenario_analysis_vec(portfolio_values: &[f64], scenarios: &[f64]) -> Vec<f64> {
    portfolio_values
        .iter()
        .zip(scenarios.iter())
        .map(|(v, s)| v * s)
        .collect()
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 `scenario_analysis` from the Python source.
    #[test]
    fn matches_python_reference() {
        let got = super::scenario_analysis_vec(&[1e6, 2.5e5, 4e5], &[0.1, -0.2, 0.05]);
        assert!((got[0] - 100000.0).abs() < 1e-8);
        assert!((got[1] - (-50000.0)).abs() < 1e-8);
        assert!((got[2] - 20000.0).abs() < 1e-8);
        let got = super::scenario_analysis(1e6, -0.15);
        assert!((got - (-150000.0)).abs() < 1e-6);
    }
}
