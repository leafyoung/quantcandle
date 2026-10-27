//! Port of `torchquantlib/models/interest_rate/lmm.py` (Libor Market Model).
//! One-step terminal update of forward rates under a correlated Gaussian
//! shock (Cholesky factor of the correlation matrix). Module RNG seed: 2006.

/// Cholesky factor (lower triangular) of a symmetric positive-definite
/// matrix, matching `torch.linalg.cholesky`. Asserts positive-definiteness.
/// Shared crate-wide helper (also used by `exotics` and `greeks`).
pub(crate) fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                assert!(sum >= 0.0, "correlation matrix is not positive definite");
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}

/// Libor Market Model state.
#[derive(Debug, Clone)]
pub struct LMM {
    pub forward_rates: Vec<f64>,
    pub volatilities: Vec<f64>,
    pub correlation_matrix: Vec<Vec<f64>>,
}

impl LMM {
    pub fn new(forward_rates: Vec<f64>, volatilities: Vec<f64>, correlations: Vec<Vec<f64>>) -> Self {
        assert_eq!(
            forward_rates.len(),
            volatilities.len(),
            "forward_rates_init and volatilities_init must have the same length."
        );
        let n = forward_rates.len();
        assert_eq!(
            (correlations.len(), correlations.first().map_or(0, |r| r.len())),
            (n, n),
            "correlations_init must be a square matrix matching the number of rates."
        );
        Self { forward_rates, volatilities, correlation_matrix: correlations }
    }

    /// Python `LMM.simulate`: N one-step terminal scenarios at horizon `t`.
    ///
    /// - draws N x num_rates iid normals (row-major) from
    ///   `crate::rng::Xorshift64Star` with `seed` (2006 in tests);
    /// - correlates them with the Cholesky factor: noise = z @ L^T;
    /// - if all forward rates are non-negative, scales the shock by the mean
    ///   forward rate and floors the result at zero; otherwise plain additive.
    pub fn simulate(&self, t: f64, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let num_rates = self.forward_rates.len();
        let vols: Vec<f64> = self.volatilities.iter().map(|v| v.max(0.0)).collect();
        let chol = cholesky(&self.correlation_matrix);
        let sqrt_t = t.max(0.0).sqrt();

        let all_nonneg = self.forward_rates.iter().all(|&r| r >= 0.0);
        let base_scale = if all_nonneg {
            self.forward_rates.iter().sum::<f64>() / num_rates as f64
        } else {
            0.0
        };

        let mut rng = crate::rng::Xorshift64Star::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // Draw one row of iid normals, then correlate.
            let z: Vec<f64> = (0..num_rates).map(|_| rng.next_std_normal()).collect();
            let mut row = Vec::with_capacity(num_rates);
            for j in 0..num_rates {
                let correlated: f64 = (0..num_rates).map(|k| z[k] * chol[j][k]).sum();
                let scale = if all_nonneg { base_scale } else { 1.0 } * vols[j] * sqrt_t * correlated;
                let value = self.forward_rates[j] + scale;
                row.push(if all_nonneg { value.max(0.0) } else { value });
            }
            out.push(row);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Cholesky vs the Python `torch.linalg.cholesky` reference; tol 1e-12.
    #[test]
    fn cholesky_matches_reference() {
        let a = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.3],
            vec![0.2, 0.3, 1.0],
        ];
        let l = cholesky(&a);
        let expected = [
            [1.0, 0.0, 0.0],
            [0.5, 0.8660254037844386, 0.0],
            [0.2, 0.23094010767585033, 0.9521904571390466],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(rel(l[i][j], expected[i][j]) < 1e-12, "l[{i}][{j}]={}", l[i][j]);
            }
        }
    }

    /// Stochastic scenarios; expected values from scripts/gen_lmm.py with
    /// refgen normals, seed 2006, tol 1e-12 relative.
    #[test]
    fn simulate_matches_reference_nonneg() {
        let lmm = LMM::new(
            vec![0.03, 0.035, 0.04],
            vec![0.1, 0.15, 0.2],
            vec![
                vec![1.0, 0.5, 0.2],
                vec![0.5, 1.0, 0.3],
                vec![0.2, 0.3, 1.0],
            ],
        );
        let out = lmm.simulate(1.0, 3, 2006);
        let expected = [
            [0.039661145010229715, 0.039854038648551325, 0.04403412931163772],
            [0.035034923857970836, 0.03288614717875651, 0.04410208300180206],
            [0.03318410437062085, 0.040648702167959086, 0.04815742006517208],
        ];
        for (row, want) in out.iter().zip(expected) {
            for (got, w) in row.iter().zip(want) {
                assert!(rel(*got, w) < 1e-12, "got={got} want={w}");
            }
        }
    }

    /// Negative forward rates disable the scaling/floor branch.
    #[test]
    fn simulate_negative_branch() {
        let lmm = LMM::new(
            vec![-0.01, 0.02],
            vec![0.1, 0.2],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let out = lmm.simulate(1.0, 2, 2006);
        // Uncorrelated noise: value = fwd + vol * sqrt(T) * z.
        let expected = [
            [0.26603271457799194, -0.08521279876984011],
            [0.0053045047355977665, 0.30770993474119074],
        ];
        for (row, want) in out.iter().zip(expected) {
            for (got, w) in row.iter().zip(want) {
                assert!(rel(*got, w) < 1e-12, "got={got} want={w}");
            }
        }
    }
}
