//! Port of `utils.py` (torchquantlib core/asset_pricing/option): binomial-tree
//! building blocks shared by the American/Bermudan option pricers.
//!
//! All math in f64, mirroring the Python numerics (formulas and tree layout
//! identical: `price_tree[step][downs]`, backward induction over steps).

/// (dt, u, d, p) parameters of the CRR binomial tree.
pub fn calculate_binomial_tree_params(expiry: f64, volatility: f64, rate: f64, steps: usize) -> (f64, f64, f64, f64) {
    let dt = expiry / steps as f64;
    if dt <= 0.0 {
        return (dt, 1.0, 1.0, 1.0);
    }
    let u = (volatility * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = if (u - d).abs() < 1e-12 * u.abs().max(1.0) {
        1.0
    } else {
        ((rate * dt).exp() - d) / (u - d)
    };
    (dt, u, d, p)
}

/// Payoff of a call/put at terminal (or any) prices.
pub fn option_payoff(option_type: &str, terminal_prices: &[f64], strike: f64) -> Vec<f64> {
    match option_type {
        "call" => terminal_prices.iter().map(|&s| (s - strike).max(0.0)).collect(),
        "put" => terminal_prices.iter().map(|&s| (strike - s).max(0.0)).collect(),
        _ => panic!("Invalid option type. Use 'call' or 'put'."),
    }
}

/// Price tree `tree[step][down]` = spot * u^(step-down) * d^down.
pub fn build_price_tree(spot: f64, u: f64, d: f64, steps: usize) -> Vec<Vec<f64>> {
    let mut tree = vec![vec![0.0; steps + 1]; steps + 1];
    for step in 0..=steps {
        for down in 0..=step {
            let ups = (step - down) as f64;
            tree[step][down] = spot * u.powf(ups) * d.powi(down as i32);
        }
    }
    tree
}

/// Binomial-tree option price. `exercise_dates` are step indices at which
/// early exercise is allowed (Bermudan); ignored for European style, and the
/// full range 0..steps is used for American style.
pub fn price_binomial_tree(
    option_type: &str,
    option_style: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
    exercise_dates: Option<&[usize]>,
) -> f64 {
    if option_type != "call" && option_type != "put" {
        panic!("Invalid option type. Use 'call' or 'put'.");
    }
    if option_style != "european" && option_style != "american" && option_style != "bermudan" {
        panic!("Invalid option style. Use 'european', 'american', or 'bermudan'.");
    }
    if steps == 0 {
        panic!("steps must be a positive integer.");
    }
    if expiry <= 0.0 {
        let payoff = if option_type == "call" {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
        return payoff;
    }
    if volatility < 0.0 {
        panic!("volatility must be non-negative.");
    }
    if volatility.abs() < 1e-12 {
        let deterministic_terminal = spot * (rate * expiry).exp();
        let disc = (-rate * expiry).exp();
        let payoff = if option_type == "call" {
            (deterministic_terminal - strike).max(0.0)
        } else {
            (strike - deterministic_terminal).max(0.0)
        };
        return disc * payoff;
    }

    let (dt, u, d, p) = calculate_binomial_tree_params(expiry, volatility, rate, steps);
    let _ = dt;
    let discount = (-rate * dt).exp();
    let price_tree = build_price_tree(spot, u, d, steps);
    let mut value_tree = vec![vec![0.0; steps + 1]; steps + 1];
    for down in 0..=steps {
        value_tree[steps][down] = if option_type == "call" {
            (price_tree[steps][down] - strike).max(0.0)
        } else {
            (strike - price_tree[steps][down]).max(0.0)
        };
    }

    let exercise_steps: Vec<usize> = if option_style == "american" {
        (0..steps).collect()
    } else if option_style == "bermudan" {
        exercise_dates
            .unwrap_or(&[])
            .iter()
            .copied()
            .filter(|&s| s < steps)
            .collect()
    } else {
        Vec::new()
    };

    for step in (0..steps).rev() {
        let can_exercise = exercise_steps.contains(&step);
        for down in 0..=step {
            let continuation = (p * value_tree[step + 1][down] + (1.0 - p) * value_tree[step + 1][down + 1])
                * discount;
            if can_exercise {
                let exercise = if option_type == "call" {
                    (price_tree[step][down] - strike).max(0.0)
                } else {
                    (strike - price_tree[step][down]).max(0.0)
                };
                value_tree[step][down] = exercise.max(continuation);
            } else {
                value_tree[step][down] = continuation;
            }
        }
    }

    value_tree[0][0]
}

/// Batch wrapper: evaluate each batch element independently. A single-element
/// slice is broadcast across the batch (Python broadcasting semantics);
/// otherwise slices must be full-length.
pub fn binomial_tree_batch(
    option_type: &str,
    option_style: &str,
    spot: &[f64],
    strike: &[f64],
    expiry: &[f64],
    volatility: &[f64],
    rate: &[f64],
    steps: usize,
) -> Vec<f64> {
    let n = spot.len();
    let pick = |v: &[f64], i: usize| -> f64 {
        if v.len() == 1 {
            v[0]
        } else {
            v[i]
        }
    };
    (0..n)
        .map(|i| {
            price_binomial_tree(
                option_type,
                option_style,
                spot[i],
                pick(strike, i),
                pick(expiry, i),
                pick(volatility, i),
                pick(rate, i),
                steps,
                None,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_option_utils.py (pure f64).
    #[test]
    fn european_call_tree() {
        let v = price_binomial_tree("call", "european", 100.0, 100.0, 1.0, 0.2, 0.05, 5, None);
        assert!((v - 10.805933920410084).abs() < 1e-9);
    }

    #[test]
    fn american_put_tree() {
        let v = price_binomial_tree("put", "american", 100.0, 100.0, 1.0, 0.2, 0.05, 10, None);
        assert!((v - 6.0042590201829515).abs() < 1e-9);
    }

    #[test]
    fn bermudan_put_tree() {
        let v = price_binomial_tree(
            "put", "bermudan", 100.0, 100.0, 1.0, 0.2, 0.05, 10, Some(&[2, 5, 8]),
        );
        assert!((v - 5.856695576439613).abs() < 1e-9);
    }

    #[test]
    fn zero_volatility() {
        let v = price_binomial_tree("call", "european", 100.0, 90.0, 1.0, 0.0, 0.05, 5, None);
        assert!((v - 14.38935179493575).abs() < 1e-9);
    }

    #[test]
    fn zero_expiry_returns_intrinsic() {
        let v = price_binomial_tree("put", "european", 95.0, 100.0, 0.0, 0.2, 0.05, 5, None);
        assert!((v - 5.0).abs() < 1e-12);
    }

    #[test]
    fn params_match_python() {
        let (dt, u, d, p) = calculate_binomial_tree_params(1.0, 0.2, 0.05, 4);
        assert!((dt - 0.25).abs() < 1e-12);
        assert!((u - 1.1051709180756477).abs() < 1e-12);
        assert!((d - 0.9048374180359595).abs() < 1e-12);
        assert!((p - 0.537808371956414).abs() < 1e-12);
    }
}
