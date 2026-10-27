//! Port of `torchquantlib/utils/yield_curve_construction.py`.
//!
//! Simple yield bootstrapping from cash-flow totals and market prices, plus a
//! Nelson-Siegel yield curve with lambda fixed at 1. All math is scalar `f64`.
//! The 2D `(n_instruments, n_cashflows)` and batched 3D
//! `(batch, n_instruments, n_cashflows)` forms of the Python source are kept
//! as separate functions.

/// Errors mirroring the Python source's `RuntimeError`s.
#[derive(Debug, PartialEq)]
pub enum YieldCurveError {
    NonPositivePrice,
    ShapeMismatch,
}

/// Construct simple yields from cash-flow totals and market prices.
///
/// `cash_flows[i]` holds the cash flows of instrument `i`; instrument `i`
/// discounts at exponent `1/(i+1)` (periods are 1-based).
pub fn bootstrap_yield_curve(
    cash_flows: &[Vec<f64>],
    prices: &[f64],
) -> Result<Vec<f64>, YieldCurveError> {
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(YieldCurveError::NonPositivePrice);
    }
    if cash_flows.len() != prices.len() {
        return Err(YieldCurveError::ShapeMismatch);
    }

    let num_instruments = cash_flows.len();
    let mut yields = Vec::with_capacity(num_instruments);
    for (i, flows) in cash_flows.iter().enumerate() {
        let total: f64 = flows.iter().sum();
        let exponent = 1.0 / (i + 1) as f64;
        yields.push((total / prices[i]).powf(exponent) - 1.0);
    }
    Ok(yields)
}

/// Batched (3D) form: one curve per batch element, with matching price rows.
pub fn bootstrap_yield_curve_batched(
    cash_flows: &[Vec<Vec<f64>>],
    prices: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, YieldCurveError> {
    if cash_flows.len() != prices.len()
        || cash_flows
            .iter()
            .zip(prices)
            .any(|(cf, p)| cf.len() != p.len())
    {
        return Err(YieldCurveError::ShapeMismatch);
    }
    let mut out = Vec::with_capacity(cash_flows.len());
    for (cf, p) in cash_flows.iter().zip(prices) {
        out.push(bootstrap_yield_curve(cf, p)?);
    }
    Ok(out)
}

/// Nelson-Siegel yield curve with lambda fixed at 1:
/// `y(tau) = beta0 + beta1 * (1 - e^-tau)/tau + beta2 * ((1 - e^-tau)/tau - e^-tau)`.
pub fn nelson_siegel_yield_curve(tau: &[f64], beta0: f64, beta1: f64, beta2: f64) -> Vec<f64> {
    tau.iter()
        .map(|&t| {
            let exp_term = (-t).exp();
            let factor1 = (1.0 - exp_term) / t;
            let factor2 = factor1 - exp_term;
            beta0 + beta1 * factor1 + beta2 * factor2
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from scripts/gen_misc_pricers.py (pure float64).
    #[test]
    fn test_bootstrap_yield_curve_2d() {
        let cash_flows = vec![vec![50.0, 1050.0], vec![60.0, 1060.0], vec![1080.0, 0.0]];
        let prices = vec![980.0, 995.0, 900.0];
        let actual = bootstrap_yield_curve(&cash_flows, &prices).unwrap();
        let expected = [0.12244897959183665, 0.0609562388258611, 0.06265856918261115];
        for (a, e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() <= 1e-8 * e.abs().max(1.0),
                "actual {a} vs expected {e}"
            );
        }
    }

    #[test]
    fn test_bootstrap_yield_curve_batched() {
        let cash_flows = vec![
            vec![vec![50.0, 1050.0], vec![60.0, 1060.0]],
            vec![vec![40.0, 1040.0], vec![70.0, 1070.0]],
        ];
        let prices = vec![vec![975.0, 990.0], vec![960.0, 1005.0]];
        let actual = bootstrap_yield_curve_batched(&cash_flows, &prices).unwrap();
        let expected = [
            [0.1282051282051282, 0.06363204695662072],
            [0.125, 0.0650485238752998],
        ];
        for (batch, exp_batch) in actual.iter().zip(&expected) {
            for (a, e) in batch.iter().zip(exp_batch) {
                assert!(
                    (a - e).abs() <= 1e-8 * e.abs().max(1.0),
                    "actual {a} vs expected {e}"
                );
            }
        }
    }

    #[test]
    fn test_bootstrap_errors() {
        let cash_flows = vec![vec![50.0, 1050.0]];
        assert_eq!(
            bootstrap_yield_curve(&cash_flows, &[0.0]),
            Err(YieldCurveError::NonPositivePrice)
        );
        assert_eq!(
            bootstrap_yield_curve(&cash_flows, &[1.0, 2.0]),
            Err(YieldCurveError::ShapeMismatch)
        );
    }

    #[test]
    fn test_nelson_siegel() {
        let tau = [0.5, 1.0, 2.0, 5.0];
        let actual = nelson_siegel_yield_curve(&tau, 0.03, -0.02, 0.01);
        let expected = [
            0.016065306597126332,
            0.02,
            0.024323323583816937,
            0.027946096424007316,
        ];
        for (a, e) in actual.iter().zip(&expected) {
            assert!(
                (a - e).abs() <= 1e-8,
                "actual {a} vs expected {e}"
            );
        }
    }
}
