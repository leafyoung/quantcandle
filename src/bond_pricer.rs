//! Port of `torchquantlib/core/asset_pricing/bond/bond_pricer.py`.
//!
//! Closed-form and lattice bond pricing: zero-coupon, coupon bonds,
//! callable/putable bonds on a short-rate lattice, convertible bonds on a
//! stock binomial tree (with call/put exercise schedules), and a
//! stochastic-rate coupon bond. All math is scalar `f64` (pure lattice /
//! closed-form, no tensors needed).

const EPSILON: f64 = 1e-8;

fn period_count(periods: i64) -> usize {
    periods.max(1) as usize
}

fn short_rate_tree(initial_rate: f64, count: usize, rate_volatility: f64) -> Vec<Vec<f64>> {
    let sigma = rate_volatility.max(0.0);
    if sigma == 0.0 {
        return (0..count).map(|n| vec![initial_rate; n + 1]).collect();
    }
    let up = sigma.exp();
    (0..count)
        .map(|n| {
            let exponents: Vec<i32> = (-((n as i32))..=((n as i32))).step_by(2).collect();
            exponents
                .into_iter()
                .map(|e| (initial_rate * up.powi(e)).max(0.0))
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn embedded_option_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    exercise_price: Option<f64>,
    exercise_period: Option<i64>,
    exercise_kind: Option<&str>,
    rate_volatility: Option<f64>,
) -> f64 {
    let period_count = period_count(periods);
    let rate_tree = short_rate_tree(rate, period_count, rate_volatility.unwrap_or(0.0));
    let coupon = face_value * coupon_rate;
    let mut continuation = vec![face_value; period_count + 1];

    let exercise_index = exercise_period
        .map(|p| (p.max(1) as usize).min(period_count));

    for step in (0..period_count).rev() {
        let expected_next: Vec<f64> = continuation
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect();
        let mut node_values: Vec<f64> = expected_next
            .iter()
            .zip(&rate_tree[step])
            .map(|(&v, &r)| (coupon + v) / (1.0 + r).max(EPSILON))
            .collect();
        if let (Some(idx), Some(price)) = (exercise_index, exercise_price) {
            if step + 1 == idx {
                match exercise_kind {
                    Some("call") => node_values.iter_mut().for_each(|v| *v = v.min(price)),
                    Some("put") => node_values.iter_mut().for_each(|v| *v = v.max(price)),
                    _ => {}
                }
            }
        }
        continuation = node_values;
    }
    continuation[0]
}

pub fn zero_coupon_bond(face_value: f64, rate: f64, maturity: f64) -> f64 {
    face_value / (1.0 + rate).powf(maturity).max(EPSILON)
}

pub fn coupon_bond(face_value: f64, coupon_rate: f64, rate: f64, periods: i64) -> f64 {
    let n = period_count(periods);
    let coupon_cashflow = face_value * coupon_rate;
    let mut total = 0.0;
    for t in 1..=n {
        total += coupon_cashflow / (1.0 + rate).powi(t as i32).max(EPSILON);
    }
    total + face_value / (1.0 + rate).powi(n as i32).max(EPSILON)
}

pub fn callable_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    call_price: f64,
    call_period: i64,
    rate_volatility: Option<f64>,
) -> f64 {
    let lattice_value = embedded_option_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        Some(call_price),
        Some(call_period),
        Some("call"),
        rate_volatility,
    );
    lattice_value.min(coupon_bond(face_value, coupon_rate, rate, periods))
}

/// Alias kept for parity with the Python API (`callable_bond` in the source
/// simply forwards to `callable_bond_lattice`).
pub fn callable_bond(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    call_price: f64,
    call_period: i64,
) -> f64 {
    callable_bond_lattice(face_value, coupon_rate, rate, periods, call_price, call_period, None)
}

pub fn putable_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    put_price: f64,
    put_period: i64,
    rate_volatility: Option<f64>,
) -> f64 {
    let lattice_value = embedded_option_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        Some(put_price),
        Some(put_period),
        Some("put"),
        rate_volatility,
    );
    lattice_value.max(coupon_bond(face_value, coupon_rate, rate, periods))
}

/// Alias kept for parity with the Python API (`putable_bond` in the source
/// simply forwards to `putable_bond_lattice`).
pub fn putable_bond(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    put_price: f64,
    put_period: i64,
) -> f64 {
    putable_bond_lattice(face_value, coupon_rate, rate, periods, put_price, put_period, None)
}

pub fn convertible_bond(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    conversion_ratio: f64,
    conversion_price: f64,
) -> f64 {
    let straight_value = coupon_bond(face_value, coupon_rate, rate, periods);
    let conversion_value = conversion_ratio * conversion_price;
    straight_value.max(conversion_value)
}

type Schedule = std::collections::BTreeMap<usize, f64>;

fn normalize_schedule(
    period_count: usize,
    schedule: Option<&[(usize, f64)]>,
    single_price: Option<f64>,
    single_period: Option<i64>,
) -> Schedule {
    let mut out = Schedule::new();
    if let Some(sched) = schedule {
        for &(p, price) in sched {
            let idx = (p.max(1)).min(period_count);
            out.insert(idx, price);
        }
    }
    if let (Some(price), Some(p)) = (single_price, single_period) {
        let idx = (p.max(1) as usize).min(period_count);
        out.insert(idx, price);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn convertible_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    conversion_ratio: f64,
    stock_price: f64,
    stock_volatility: f64,
    dividend_yield: Option<f64>,
    call_price: Option<f64>,
    call_period: Option<i64>,
    put_price: Option<f64>,
    put_period: Option<i64>,
    call_schedule: Option<&[(usize, f64)]>,
    put_schedule: Option<&[(usize, f64)]>,
) -> f64 {
    let period_count = period_count(periods);
    let coupon = face_value * coupon_rate;
    let stock0 = stock_price;
    let volatility = stock_volatility.max(0.0);
    let dividend = dividend_yield.unwrap_or(0.0);
    let dt = 1.0f64;

    let call_schedule_map =
        normalize_schedule(period_count, call_schedule, call_price, call_period);
    let put_schedule_map = normalize_schedule(period_count, put_schedule, put_price, put_period);

    if volatility == 0.0 {
        let growth = ((rate - dividend) * dt).exp();
        let terminal_stock = stock0 * growth.powi(period_count as i32);
        let mut continuation = vec![face_value.max(conversion_ratio * terminal_stock); period_count + 1];
        if let Some(&p) = call_schedule_map.get(&period_count) {
            continuation.iter_mut().for_each(|v| *v = v.min(p));
        }
        if let Some(&p) = put_schedule_map.get(&period_count) {
            continuation.iter_mut().for_each(|v| *v = v.max(p));
        }
        for step in (0..period_count).rev() {
            continuation = continuation
                .windows(2)
                .map(|w| (coupon + w[0]) / (1.0 + rate).max(EPSILON))
                .collect();
            let exercise_date = step + 1;
            if let Some(&p) = call_schedule_map.get(&exercise_date) {
                continuation.iter_mut().for_each(|v| *v = v.min(p));
            }
            if let Some(&p) = put_schedule_map.get(&exercise_date) {
                continuation.iter_mut().for_each(|v| *v = v.max(p));
            }
        }
        return continuation[0];
    }

    let up = (volatility * dt.sqrt()).exp();
    let down = 1.0 / up;
    let growth = ((rate - dividend) * dt).exp();
    let probability = ((growth - down) / (up - down).max(EPSILON)).clamp(0.0, 1.0);

    let stock_nodes = |step: usize| -> Vec<f64> {
        (0..=step)
            .map(|j| stock0 * up.powi((step - j) as i32) * down.powi(j as i32))
            .collect()
    };

    let mut continuation: Vec<f64> = stock_nodes(period_count)
        .into_iter()
        .map(|s| face_value.max(conversion_ratio * s))
        .collect();
    if let Some(&p) = call_schedule_map.get(&period_count) {
        continuation.iter_mut().for_each(|v| *v = v.min(p));
    }
    if let Some(&p) = put_schedule_map.get(&period_count) {
        continuation.iter_mut().for_each(|v| *v = v.max(p));
    }

    for step in (0..period_count).rev() {
        let nodes = stock_nodes(step);
        let expected_next: Vec<f64> = continuation
            .windows(2)
            .map(|w| probability * w[0] + (1.0 - probability) * w[1])
            .collect();
        let mut node_value: Vec<f64> = expected_next
            .iter()
            .zip(&nodes)
            .map(|(&h, &s)| ((coupon + h) / (1.0 + rate).max(EPSILON)).max(conversion_ratio * s))
            .collect();
        let exercise_date = step + 1;
        if let Some(&p) = call_schedule_map.get(&exercise_date) {
            node_value.iter_mut().for_each(|v| *v = v.min(p));
        }
        if let Some(&p) = put_schedule_map.get(&exercise_date) {
            node_value.iter_mut().for_each(|v| *v = v.max(p));
        }
        continuation = node_value;
    }

    continuation[0]
}

#[allow(clippy::too_many_arguments)]
pub fn callable_convertible_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    conversion_ratio: f64,
    stock_price: f64,
    stock_volatility: f64,
    call_price: f64,
    call_period: i64,
    dividend_yield: Option<f64>,
    call_schedule: Option<&[(usize, f64)]>,
) -> f64 {
    convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield,
        Some(call_price),
        Some(call_period),
        None,
        None,
        call_schedule,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn putable_convertible_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    conversion_ratio: f64,
    stock_price: f64,
    stock_volatility: f64,
    put_price: f64,
    put_period: i64,
    dividend_yield: Option<f64>,
    put_schedule: Option<&[(usize, f64)]>,
) -> f64 {
    convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield,
        None,
        None,
        Some(put_price),
        Some(put_period),
        None,
        put_schedule,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn callable_putable_convertible_bond_lattice(
    face_value: f64,
    coupon_rate: f64,
    rate: f64,
    periods: i64,
    conversion_ratio: f64,
    stock_price: f64,
    stock_volatility: f64,
    call_price: f64,
    call_period: i64,
    put_price: f64,
    put_period: i64,
    dividend_yield: Option<f64>,
    call_schedule: Option<&[(usize, f64)]>,
    put_schedule: Option<&[(usize, f64)]>,
) -> f64 {
    convertible_bond_lattice(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        stock_price,
        stock_volatility,
        dividend_yield,
        Some(call_price),
        Some(call_period),
        Some(put_price),
        Some(put_period),
        call_schedule,
        put_schedule,
    )
}

/// Rates slice is the per-period short rate path `rate[:period_count]`.
pub fn stochastic_rate_bond(face_value: f64, coupon_rate: f64, rates: &[f64]) -> f64 {
    let coupon = face_value * coupon_rate;
    let mut discount = 1.0f64;
    let mut total = 0.0;
    for &r in rates {
        discount *= 1.0 + r;
        total += coupon / discount;
    }
    total + face_value / discount
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol.max(tol * expected.abs()),
            "actual {actual} vs expected {expected} (diff {diff})"
        );
    }

    // Expected values from scripts/gen_bond_pricer.py (pure float64).
    #[test]
    fn test_zero_coupon_bond() {
        assert_close(zero_coupon_bond(1000.0, 0.05, 3.0), 863.837598531476, TOL);
    }

    #[test]
    fn test_coupon_bond() {
        assert_close(coupon_bond(1000.0, 0.05, 0.04, 5), 1044.5182233101618, TOL);
    }

    #[test]
    fn test_callable_bond_lattice() {
        assert_close(
            callable_bond_lattice(1000.0, 0.06, 0.05, 5, 1050.0, 3, Some(0.1)),
            1041.0660615540885,
            TOL,
        );
        assert_close(
            callable_bond(1000.0, 0.06, 0.05, 5, 1050.0, 3),
            callable_bond_lattice(1000.0, 0.06, 0.05, 5, 1050.0, 3, None),
            TOL,
        );
    }

    #[test]
    fn test_putable_bond_lattice() {
        assert_close(
            putable_bond_lattice(1000.0, 0.06, 0.05, 5, 950.0, 3, Some(0.1)),
            1043.2947667063079,
            TOL,
        );
    }

    #[test]
    fn test_convertible_bond() {
        assert_close(convertible_bond(100.0, 0.05, 0.05, 4, 2.0, 60.0), 120.0, TOL);
    }

    #[test]
    fn test_convertible_bond_lattice_with_vol() {
        let calls = [(3usize, 110.0f64)];
        let puts = [(2usize, 90.0f64)];
        assert_close(
            convertible_bond_lattice(
                100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.2, Some(0.01), Some(110.0), Some(3),
                Some(90.0), Some(2), Some(&calls), Some(&puts),
            ),
            127.7103299966812,
            TOL,
        );
    }

    #[test]
    fn test_convertible_bond_lattice_zero_vol() {
        let calls = [(3usize, 110.0f64)];
        let puts = [(2usize, 90.0f64)];
        assert_close(
            convertible_bond_lattice(
                100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.0, Some(0.01), Some(110.0), Some(3),
                Some(90.0), Some(2), Some(&calls), Some(&puts),
            ),
            109.07029478458048,
            TOL,
        );
    }

    #[test]
    fn test_stochastic_rate_bond() {
        assert_close(
            stochastic_rate_bond(1000.0, 0.04, &[0.03, 0.035, 0.04, 0.045]),
            1010.083333337649,
            TOL,
        );
    }
}
