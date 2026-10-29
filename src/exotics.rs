//! Port of `exotics.py` (torchquantlib core/asset_pricing/option).
//!
//! Monte-Carlo pricers for digital, rainbow, barrier, lookback, Asian,
//! quanto and basket options, plus Curran's Asian approximation and a
//! closed-form chooser option.
//!
//! Stochasticity: torch `randn` calls are replaced by
//! `crate::rng::Xorshift64Star` normals with an explicit `seed` parameter per
//! function (documented deviation: the Python API relies on
//! `torch.manual_seed`). Draw order replicates torch tensor fills: 1-D
//! `(num_paths,)` per call / per time step, 2-D `(num_paths, n_assets)`
//! row-major. Correlated draws use the Cholesky factor as `z @ L.T`.

use crate::black_scholes_merton::normal_cdf;
use crate::lmm::cholesky;
use crate::rng::Xorshift64Star;

/// Validate volatility/expiry/num_paths like `ValidationMixin::validate_parameters`.
fn validate_parameters(volatility: f64, expiry: f64, num_paths: Option<usize>) {
    if volatility <= 0.0 || expiry <= 0.0 {
        panic!("Volatility and expiry must be positive.");
    }
    if let Some(n) = num_paths {
        if n == 0 {
            panic!("Number of paths must be a positive integer.");
        }
    }
}

fn validate_option_type(option_type: &str) {
    if option_type != "call" && option_type != "put" {
        panic!("Option type must be either 'call' or 'put'.");
    }
}

/// Price a digital (binary cash-or-nothing) option by Monte Carlo.
pub fn digital_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    payout: f64,
    num_paths: usize,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));
    validate_option_type(option_type);

    let mut rng = Xorshift64Star::new(seed);
    let z = rng.normals(num_paths);
    let mut sum = 0.0;
    for &zi in &z {
        let asset_price = spot
            * (((rate - 0.5 * volatility * volatility) * expiry
                + volatility * expiry.sqrt() * zi)
                .exp());
        let payoff = if (option_type == "call" && asset_price >= strike)
            || (option_type == "put" && asset_price <= strike)
        {
            payout
        } else {
            0.0
        };
        sum += payoff * (-rate * expiry).exp();
    }
    sum / num_paths as f64
}

/// Price a rainbow (weighted-basket call) option by Monte Carlo with
/// correlated assets via Cholesky.
pub fn rainbow_option(
    spots: &[f64],
    weights: &[f64],
    strike: f64,
    expiry: f64,
    volatilities: &[f64],
    correlations: &[Vec<f64>],
    rate: f64,
    num_paths: usize,
    seed: u64,
) -> f64 {
    let n_assets = spots.len();
    let l = cholesky(correlations);
    let mut rng = Xorshift64Star::new(seed);
    let mut sum = 0.0;
    for _ in 0..num_paths {
        // z row (n_assets), then corr_rand = z @ L.T -> (L @ z)
        let z: Vec<f64> = (0..n_assets).map(|_| rng.next_std_normal()).collect();
        let mut prices = Vec::with_capacity(n_assets);
        for a in 0..n_assets {
            let corr: f64 = (0..n_assets).map(|k| l[a][k] * z[k]).sum();
            let p = spots[a]
                * (((rate - 0.5 * volatilities[a] * volatilities[a]) * expiry
                    + volatilities[a] * expiry.sqrt() * corr)
                    .exp());
            prices.push(p);
        }
        let portfolio: f64 = prices
            .iter()
            .zip(weights.iter())
            .map(|(&p, &w)| p * w)
            .sum();
        sum += (portfolio - strike).max(0.0) * (-rate * expiry).exp();
    }
    sum / num_paths as f64
}

/// Price a barrier option by Monte Carlo (discrete barrier monitoring).
pub fn barrier_option(
    option_type: &str,
    barrier_type: &str,
    spot: f64,
    strike: f64,
    barrier: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));
    validate_option_type(option_type);
    if !matches!(
        barrier_type,
        "up-and-out" | "down-and-out" | "up-and-in" | "down-and-in"
    ) {
        panic!("Invalid barrier type");
    }

    let dt = expiry / num_steps as f64;
    let mut rng = Xorshift64Star::new(seed);
    let drift = (rate - 0.5 * volatility * volatility) * dt;
    let sig_sqrt_dt = volatility * dt.sqrt();
    let disc = (-rate * expiry).exp();

    let mut sum = 0.0;
    for _ in 0..num_paths {
        let mut price = spot;
        let mut hit = false;
        for _ in 0..num_steps {
            let z = rng.next_std_normal();
            price *= (drift + sig_sqrt_dt * z).exp();
            let crossed = match barrier_type {
                "up-and-out" => price > barrier,
                "down-and-out" => price < barrier,
                "up-and-in" => price >= barrier,
                _ => price <= barrier,
            };
            if crossed {
                hit = true;
            }
        }
        let vanilla = if option_type == "call" {
            (price - strike).max(0.0)
        } else {
            (strike - price).max(0.0)
        };
        let payoff = if barrier_type.ends_with("out") {
            if hit { 0.0 } else { vanilla }
        } else if hit {
            vanilla
        } else {
            0.0
        };
        sum += payoff * disc;
    }
    sum / num_paths as f64
}

/// Shared lookback Monte-Carlo core. `bgk_up` / `bgk_dn` multiply the running
/// maximum / minimum before the payoff; both are 1.0 for discrete monitoring
/// (`x * 1.0` is exact in IEEE-754, so the discrete path is unchanged).
fn lookback_mc(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    num_paths: usize,
    strike_type: &str,
    seed: u64,
    bgk_up: f64,
    bgk_dn: f64,
) -> f64 {
    let dt = expiry / 252.0;
    let num_steps = (expiry / dt) as usize;
    let mut rng = Xorshift64Star::new(seed);
    let drift = (rate - 0.5 * volatility * volatility) * dt;
    let sig_sqrt_dt = volatility * dt.sqrt();
    let disc = (-rate * expiry).exp();

    let mut sum = 0.0;
    for _ in 0..num_paths {
        let mut price = spot;
        let mut max_p = spot;
        let mut min_p = spot;
        for _ in 0..num_steps {
            let z = rng.next_std_normal();
            price *= (drift + sig_sqrt_dt * z).exp();
            if price > max_p {
                max_p = price;
            }
            if price < min_p {
                min_p = price;
            }
        }
        let payoff = if option_type == "call" {
            if strike_type == "fixed" {
                (max_p * bgk_up - strike).max(0.0)
            } else {
                (price - max_p * bgk_up).max(0.0)
            }
        } else if strike_type == "fixed" {
            (strike - min_p * bgk_dn).max(0.0)
        } else {
            (min_p * bgk_dn - price).max(0.0)
        };
        sum += payoff * disc;
    }
    sum / num_paths as f64
}

/// Price a lookback option by Monte Carlo with daily steps (252 per year,
/// matching the Python hard-coded `dt = expiry / 252`).
pub fn lookback_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    num_paths: usize,
    strike_type: &str,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));
    validate_option_type(option_type);
    if strike_type != "fixed" && strike_type != "floating" {
        panic!("Invalid strike type");
    }
    lookback_mc(
        option_type, spot, strike, expiry, volatility, rate, num_paths, strike_type, seed, 1.0, 1.0,
    )
}

/// Broadie-Glasserman-Kou (1997) continuity-correction constant
/// `beta = -zeta(1/2) / sqrt(2*pi) ~= 0.5826`.
const BGK_BETA: f64 = 0.5825971579390108;

/// Price a lookback option under CONTINUOUS monitoring by Monte Carlo, via
/// the Broadie-Glasserman-Kou (1997) continuity correction applied to the
/// discrete-monitoring estimator: in distribution the continuously monitored
/// extremum satisfies `max_cont ~ max_disc * e^{+beta*sigma*sqrt(dt)}` and
/// `min_cont ~ min_disc * e^{-beta*sigma*sqrt(dt)}`, so the running maximum is
/// inflated (resp. minimum deflated) by that factor before the payoff.
/// Verified against QuantLib 1.44 `AnalyticContinuousFixedLookbackEngine`
/// (fixed call S=100 K=100 r=5% q=0 vol=20% T=1 -> 19.16762525733232; the
/// uncorrected discrete estimator sits ~4.6% low at 252 steps).
/// Seed convention: same `Xorshift64Star` as `lookback_option`; the embedded
/// tests use seeds 1244 (call) / 1245 (put), next in the module's sequence.
pub fn lookback_option_continuous(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    num_paths: usize,
    strike_type: &str,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));
    validate_option_type(option_type);
    if strike_type != "fixed" && strike_type != "floating" {
        panic!("Invalid strike type");
    }
    let dt = expiry / 252.0;
    let up = (BGK_BETA * volatility * dt.sqrt()).exp();
    let dn = (-BGK_BETA * volatility * dt.sqrt()).exp();
    lookback_mc(
        option_type, spot, strike, expiry, volatility, rate, num_paths, strike_type, seed, up, dn,
    )
}

/// Price an Asian (average-price) option by Monte Carlo.
pub fn asian_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    num_paths: usize,
    num_steps: usize,
    average_type: &str,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));
    validate_option_type(option_type);
    if average_type != "arithmetic" && average_type != "geometric" {
        panic!("Invalid average type");
    }

    let dt = expiry / num_steps as f64;
    let mut rng = Xorshift64Star::new(seed);
    let drift = (rate - 0.5 * volatility * volatility) * dt;
    let sig_sqrt_dt = volatility * dt.sqrt();
    let disc = (-rate * expiry).exp();

    let mut sum = 0.0;
    for _ in 0..num_paths {
        let mut price = spot;
        let mut log_sum = 0.0f64;
        let mut arith_sum = 0.0f64;
        // Include the initial spot in the average, as paths[:, 0] = spot and
        // the Python mean runs over all num_steps + 1 columns.
        arith_sum += spot;
        log_sum += spot.ln();
        for _ in 0..num_steps {
            let z = rng.next_std_normal();
            price *= (drift + sig_sqrt_dt * z).exp();
            arith_sum += price;
            log_sum += price.ln();
        }
        let avg_price = if average_type == "arithmetic" {
            arith_sum / (num_steps + 1) as f64
        } else {
            (log_sum / (num_steps + 1) as f64).exp()
        };
        let payoff = if option_type == "call" {
            (avg_price - strike).max(0.0)
        } else {
            (strike - avg_price).max(0.0)
        };
        sum += payoff * disc;
    }
    sum / num_paths as f64
}

/// Price a quanto option by Monte Carlo (asset in foreign currency, payoff
/// converted at simulated FX rate).
pub fn quanto_option(
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    fx_volatility: f64,
    correlation: f64,
    domestic_rate: f64,
    foreign_rate: f64,
    fx_rate: f64,
    num_paths: usize,
    seed: u64,
) -> f64 {
    validate_parameters(volatility, expiry, Some(num_paths));

    let mut rng = Xorshift64Star::new(seed);
    let mut sum = 0.0;
    for _ in 0..num_paths {
        let z1 = rng.next_std_normal();
        let z2 = correlation * z1 + (1.0 - correlation * correlation).sqrt() * rng.next_std_normal();
        let asset_price = spot
            * (((foreign_rate - 0.5 * volatility * volatility) * expiry
                + volatility * expiry.sqrt() * z1)
                .exp());
        let fx = fx_rate
            * (((domestic_rate - foreign_rate - 0.5 * fx_volatility * fx_volatility) * expiry
                + fx_volatility * expiry.sqrt() * z2)
                .exp());
        sum += (asset_price - strike).max(0.0) * fx * (-domestic_rate * expiry).exp();
    }
    sum / num_paths as f64
}

/// Price a basket option by Monte Carlo with correlated assets.
pub fn basket_option(
    spots: &[f64],
    weights: &[f64],
    strike: f64,
    expiry: f64,
    volatilities: &[f64],
    correlations: &[Vec<f64>],
    rate: f64,
    option_type: &str,
    num_paths: usize,
    seed: u64,
) -> f64 {
    validate_parameters(
        volatilities.iter().cloned().fold(f64::INFINITY, f64::min),
        expiry,
        Some(num_paths),
    );
    validate_option_type(option_type);

    let n_assets = spots.len();
    let l = cholesky(correlations);
    let mut rng = Xorshift64Star::new(seed);
    let disc = (-rate * expiry).exp();
    let mut sum = 0.0;
    for _ in 0..num_paths {
        let z: Vec<f64> = (0..n_assets).map(|_| rng.next_std_normal()).collect();
        let mut basket_value = 0.0;
        for a in 0..n_assets {
            let corr: f64 = (0..n_assets).map(|k| l[a][k] * z[k]).sum();
            let p = spots[a]
                * (((rate - 0.5 * volatilities[a] * volatilities[a]) * expiry
                    + volatilities[a] * expiry.sqrt() * corr)
                    .exp());
            basket_value += p * weights[a];
        }
        let payoff = if option_type == "call" {
            (basket_value - strike).max(0.0)
        } else {
            (strike - basket_value).max(0.0)
        };
        sum += payoff * disc;
    }
    sum / num_paths as f64
}

/// Curran's approximation for an Asian option (port of `CurranAsianOption`).
pub struct CurranAsianOption {
    pub option_type: String,
    pub average_type: String,
}

impl CurranAsianOption {
    pub fn new(option_type: &str, average_type: &str) -> Self {
        validate_option_type(option_type);
        Self {
            option_type: option_type.to_string(),
            average_type: average_type.to_string(),
        }
    }

    /// Price via Curran's approximation.
    pub fn forward(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        volatility: f64,
        rate: f64,
        num_steps: usize,
    ) -> f64 {
        validate_parameters(volatility, expiry, None);
        let adjusted_vol =
            volatility * ((num_steps + 1) as f64 / (6.0 * num_steps as f64)).sqrt();

        if self.average_type == "arithmetic" {
            let d1 = ((spot / strike).ln() + (rate + 0.5 * adjusted_vol * adjusted_vol) * expiry)
                / (adjusted_vol * expiry.sqrt());
            let d2 = d1 - adjusted_vol * expiry.sqrt();
            if self.option_type == "call" {
                spot * (-rate * expiry).exp() * normal_cdf(d1)
                    - strike * (-rate * expiry).exp() * normal_cdf(d2)
            } else {
                strike * (-rate * expiry).exp() * normal_cdf(-d2)
                    - spot * (-rate * expiry).exp() * normal_cdf(-d1)
            }
        } else {
            // Analytical solution for the geometric average.
            let adjusted_rate = rate - 0.5 * volatility * volatility / 3.0;
            crate::black_scholes_merton::black_scholes_merton(
                &self.option_type,
                "european",
                spot,
                strike,
                expiry,
                adjusted_vol,
                adjusted_rate,
                0.0,
            )
        }
    }
}

/// Price a chooser option with the Black-Scholes max(call, put) formula,
/// discounted over the choose time.
pub fn chooser_option(
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    choose_time: f64,
) -> f64 {
    validate_parameters(volatility, expiry, None);

    let t = expiry - choose_time;
    let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * t)
        / (volatility * t.sqrt());
    let d2 = d1 - volatility * t.sqrt();

    let call_price = spot * normal_cdf(d1) - strike * (-rate * t).exp() * normal_cdf(d2);
    let put_price = strike * (-rate * t).exp() * normal_cdf(-d2) - spot * normal_cdf(-d1);

    call_price.max(put_price) * (-rate * choose_time).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_exotics.py (pure f64, same seeds).
    #[test]
    fn digital_call() {
        let v = digital_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 10.0, 1000, 1234);
        assert!((v - 5.345909365693983).abs() < 1e-9);
    }

    #[test]
    fn rainbow_call() {
        let spots = vec![100.0, 90.0, 80.0];
        let weights = vec![0.5, 0.3, 0.2];
        let vols = vec![0.2, 0.25, 0.3];
        let corr = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let v = rainbow_option(&spots, &weights, 90.0, 1.0, &vols, &corr, 0.05, 500, 1235);
        assert!((v - 11.414732712382342).abs() < 1e-9);
    }

    #[test]
    fn barrier_up_and_out_call() {
        let v = barrier_option(
            "call", "up-and-out", 100.0, 100.0, 120.0, 1.0, 0.2, 0.05, 500, 50, 1236,
        );
        assert!((v - 1.4789899008215022).abs() < 1e-9);
    }

    #[test]
    fn barrier_down_and_in_put() {
        let v = barrier_option(
            "put", "down-and-in", 100.0, 100.0, 85.0, 1.0, 0.2, 0.05, 500, 50, 1237,
        );
        assert!((v - 4.46778044843966).abs() < 1e-9);
    }

    #[test]
    fn lookback_fixed_call() {
        let v = lookback_option("call", 100.0, 100.0, 0.5, 0.2, 0.05, 200, "fixed", 1238);
        assert!((v - 12.790728299383302).abs() < 1e-9);
    }

    #[test]
    fn lookback_fixed_put() {
        // Note: the floating-strike branches (both call and put) have payoffs
        // max(S_T - max(S), 0) / max(min(S) - S_T, 0) that are identically 0
        // — a quirk of the Python source (they use the wrong extremum), which
        // the port preserves. The fixed-strike put is the only other
        // non-degenerate combination, so it replaces the old vacuous
        // floating-put test. Reference: scripts/gen_exotics.py, seed 1239.
        let v = lookback_option("put", 100.0, 100.0, 0.5, 0.2, 0.05, 200, "fixed", 1239);
        assert!((v - 9.323267890110916).abs() < 1e-9);
    }

    #[test]
    fn lookback_continuous_fixed_call() {
        // Reference: scripts/gen_exotics.py lookback_option_continuous,
        // seed 1244 (first seed of the continuous variant).
        let v = lookback_option_continuous("call", 100.0, 100.0, 0.5, 0.2, 0.05, 200, "fixed", 1244);
        assert!((v - 11.879877441725975).abs() < 1e-9);
    }

    #[test]
    fn lookback_continuous_fixed_call_ql_golden() {
        // 200k paths, 252 steps, seed 1244. QuantLib 1.44 golden:
        // ContinuousFixedLookbackOption + AnalyticContinuousFixedLookbackEngine,
        // S=100 K=100 r=5% q=0 vol=20% T=1 -> 19.16762525733232.
        // BGK-corrected MC lands ~2.4e-3 relative below the golden at this
        // seed (MC noise at 200k paths is of the same order); band 3e-2.
        let v = lookback_option_continuous("call", 100.0, 100.0, 1.0, 0.2, 0.05, 200_000, "fixed", 1244);
        assert!((v - 19.12157821818025).abs() < 1e-9);
        let golden = 19.16762525733232_f64;
        assert!((v - golden).abs() / golden <= 3e-2);
    }

    #[test]
    fn lookback_continuous_fixed_put_ql_golden() {
        // 200k paths, 252 steps, seed 1245. QL golden (same setup, put):
        // 12.339744687432269. Verifies the minimum-deflation direction
        // (min_cont ~ min_disc * e^{-beta*sigma*sqrt(dt)}).
        let v = lookback_option_continuous("put", 100.0, 100.0, 1.0, 0.2, 0.05, 200_000, "fixed", 1245);
        assert!((v - 12.324399072817435).abs() < 1e-9);
        let golden = 12.339744687432269_f64;
        assert!((v - golden).abs() / golden <= 3e-2);
    }

    #[test]
    fn lookback_continuous_vs_discrete_bgk_gap() {
        // Same seed (1244): the BGK-adjusted price exceeds the discrete price
        // by ~4.6% here. The extremum itself is inflated only by
        // e^{beta*sigma*sqrt(dt)} ~ 1.00737; the price gap is leveraged
        // through (f*E[M] - K) / (E[M] - K) with E[M] ~ 119, K = 100.
        let d = lookback_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 200_000, "fixed", 1244);
        let c = lookback_option_continuous("call", 100.0, 100.0, 1.0, 0.2, 0.05, 200_000, "fixed", 1244);
        let ratio = c / d;
        assert!((ratio - 1.0456898499452234).abs() < 1e-9);
        assert!(ratio > 1.03 && ratio < 1.06);
    }

    #[test]
    fn asian_mc_arithmetic() {
        let v = asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 1000, 50, "arithmetic", 1240);
        assert!((v - 5.455662023386994).abs() < 1e-9);
    }

    #[test]
    fn asian_mc_geometric() {
        let v = asian_option("put", 100.0, 100.0, 1.0, 0.2, 0.05, 1000, 50, "geometric", 1241);
        assert!((v - 3.222490145778699).abs() < 1e-9);
    }

    #[test]
    fn quanto_call() {
        let v = quanto_option(100.0, 100.0, 1.0, 0.2, 0.1, 0.3, 0.05, 0.02, 7.8, 1000, 1242);
        assert!((v - 70.22137345027403).abs() < 1e-9);
    }

    #[test]
    fn basket_put() {
        let spots = vec![100.0, 50.0];
        let weights = vec![0.6, 0.4];
        let vols = vec![0.2, 0.3];
        let corr = vec![vec![1.0, 0.4], vec![0.4, 1.0]];
        let v = basket_option(&spots, &weights, 75.0, 1.0, &vols, &corr, 0.05, "put", 1000, 1243);
        assert!((v - 2.5309404121395453).abs() < 1e-9);
    }

    #[test]
    fn curran_arithmetic_call() {
        let c = CurranAsianOption::new("call", "arithmetic");
        let v = c.forward(100.0, 100.0, 1.0, 0.2, 0.05, 252);
        assert!((v - 2.575286024338297).abs() < 1e-9);
    }

    #[test]
    fn curran_geometric_put() {
        let c = CurranAsianOption::new("put", "geometric");
        let v = c.forward(100.0, 100.0, 1.0, 0.2, 0.05, 252);
        assert!((v - 1.5108036415253139).abs() < 1e-9);
    }

    #[test]
    fn chooser_call_dominates() {
        let v = chooser_option(100.0, 100.0, 1.0, 0.2, 0.05, 0.5);
        assert!((v - 6.71864526308475).abs() < 1e-9);
    }

    #[test]
    fn chooser_put_dominates() {
        let v = chooser_option(80.0, 100.0, 1.0, 0.2, 0.05, 0.5);
        assert!((v - 17.54304177655884).abs() < 1e-9);
    }
}
