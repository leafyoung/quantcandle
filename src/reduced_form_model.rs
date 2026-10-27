//! Port of `credit_risk/reduced_form_model.py`.
//!
//! Constant-hazard-rate reduced-form expected loss:
//! survival = exp(-lambda * t);
//! expected_loss = (1 - recovery) * (1 - survival).
//! `lambda_0` is unused in the Python implementation (kept for parity).

/// Expected loss over `time` under a constant default intensity.
pub fn reduced_form_model(
    _lambda_0: f64,
    default_intensity: f64,
    recovery_rate: f64,
    time: f64,
) -> f64 {
    let survival_prob = (-default_intensity * time).exp();
    (1.0 - recovery_rate) * (1.0 - survival_prob)
}

#[cfg(test)]
mod tests {
    /// Reference: torch float64 from the Python source.
    #[test]
    fn matches_python_reference() {
        let cases = [
            (0.03_f64, 0.02_f64, 0.4_f64, 5.0_f64, 0.05709754917842429_f64),
            (0.03, 0.05, 0.25, 10.0, 0.29510200521552493),
            (0.03, 0.10, 1.0, 7.0, 0.0),
        ];
        for (l0, lam, rec, t, expected) in cases {
            let got = super::reduced_form_model(l0, lam, rec, t);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1e-300),
                "got {got}, expected {expected}"
            );
        }
    }
}
