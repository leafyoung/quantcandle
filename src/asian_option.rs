//! Port of `asian_option.py` (torchquantlib core/asset_pricing/option).
//!
//! Analytical approximations for Asian options (geometric closed form and
//! lognormal arithmetic moment-matching), an `AsianOption` struct with Monte
//! Carlo (delegating to [`crate::exotics::asian_option`] with the shared
//! `Xorshift64Star` normals) and finite-difference greeks.
//!
//! Monte-Carlo seeding: the Python code calls `torch.manual_seed(seed)` with
//! `seed = 42` by default; here the seed is an explicit parameter of
//! [`monte_carlo_price`] (default 42).

use crate::black_scholes_merton::normal_cdf;
use crate::exotics;

const EPSILON: f64 = 1e-8;

fn sampling_times(expiry: f64, steps: usize) -> Vec<f64> {
    (1..=steps).map(|k| expiry * k as f64 / steps as f64).collect()
}

fn average_forward_mean(spot: f64, expiry: f64, rate: f64, steps: usize) -> f64 {
    let times = sampling_times(expiry, steps);
    let mean: f64 = times.iter().map(|&t| (rate * t).exp()).sum::<f64>() / steps as f64;
    spot * mean
}

fn discounted_intrinsic(option_type: &str, average_level: f64, strike: f64, expiry: f64, rate: f64) -> f64 {
    let payoff = if option_type == "call" {
        (average_level - strike).max(0.0)
    } else {
        (strike - average_level).max(0.0)
    };
    (-rate * expiry).exp() * payoff
}

fn lognormal_option_price(
    option_type: &str,
    mean_level: f64,
    second_moment: f64,
    strike: f64,
    expiry: f64,
    rate: f64,
) -> f64 {
    let mean_level = mean_level.max(EPSILON);
    let second_moment = second_moment.max(mean_level * mean_level);
    let log_variance = (second_moment / (mean_level * mean_level).max(EPSILON))
        .max(1.0)
        .ln();

    if log_variance <= EPSILON {
        return discounted_intrinsic(option_type, mean_level, strike, expiry, rate);
    }

    let sqrt_log_variance = log_variance.sqrt();
    let d1 = ((mean_level / strike).ln() + 0.5 * log_variance) / sqrt_log_variance;
    let d2 = d1 - sqrt_log_variance;
    let discount_factor = (-rate * expiry).exp();

    if option_type == "call" {
        discount_factor * (mean_level * normal_cdf(d1) - strike * normal_cdf(d2))
    } else {
        discount_factor * (strike * normal_cdf(-d2) - mean_level * normal_cdf(-d1))
    }
}

fn geometric_asian_price(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
) -> f64 {
    let times = sampling_times(expiry, steps);
    let n = steps as f64;
    let mean_t = times.iter().sum::<f64>() / n;
    let log_spot_mean = spot.max(EPSILON).ln() + (rate - 0.5 * volatility * volatility) * mean_t;
    // mean over the full n x n covariance matrix min(t_i, t_j)
    let mut cov_mean = 0.0;
    for &ti in &times {
        for &tj in &times {
            cov_mean += ti.min(tj);
        }
    }
    cov_mean /= n * n;
    let log_variance = volatility * volatility * cov_mean;
    let second_moment = (2.0 * log_spot_mean + 2.0 * log_variance).exp();
    let mean_level = (log_spot_mean + 0.5 * log_variance).exp();
    lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)
}

fn arithmetic_asian_lognormal_price(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
) -> f64 {
    let times = sampling_times(expiry, steps);
    let n = steps as f64;
    let mean_level = average_forward_mean(spot, expiry, rate, steps);
    let mut second_moment = 0.0;
    for &ti in &times {
        for &tj in &times {
            second_moment += (rate * (ti + tj) + volatility * volatility * ti.min(tj)).exp();
        }
    }
    second_moment = spot * spot * second_moment / (n * n);
    lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)
}

/// Asian option specification and pricing (port of the `AsianOption` dataclass).
#[derive(Clone)]
pub struct AsianOption {
    pub underlying: f64,
    pub strike: f64,
    pub expiry: f64,
    pub rf_rate: f64,
    pub volatility: f64,
    pub option_type: String,
    pub averaging_type: String,
    pub steps: usize,
}

impl AsianOption {
    pub fn new(
        underlying: f64,
        strike: f64,
        expiry: f64,
        rf_rate: f64,
        volatility: f64,
        option_type: &str,
        averaging_type: &str,
        steps: usize,
    ) -> Self {
        Self {
            underlying,
            strike: strike.max(EPSILON),
            expiry,
            rf_rate,
            volatility,
            option_type: option_type.to_string(),
            averaging_type: averaging_type.to_string(),
            steps,
        }
    }

    pub fn price(&self, method: &str, num_simulations: usize, num_steps: Option<usize>, seed: u64) -> f64 {
        if self.option_type != "call" && self.option_type != "put" {
            panic!("Option type must be either 'call' or 'put'.");
        }
        if self.averaging_type != "arithmetic" && self.averaging_type != "geometric" {
            panic!("average_type must be either 'arithmetic' or 'geometric'.");
        }
        let steps = num_steps.unwrap_or(self.steps);
        match method {
            "monte_carlo" => self.monte_carlo_price(num_simulations, Some(steps), seed),
            "analytical_approximation" => self.analytical_approximation_price(Some(steps)),
            _ => panic!("method must be 'monte_carlo' or 'analytical_approximation'."),
        }
    }

    pub fn monte_carlo_price(&self, num_simulations: usize, num_steps: Option<usize>, seed: u64) -> f64 {
        let steps = num_steps.unwrap_or(self.steps);
        if self.volatility <= 0.0 {
            let average_level = average_forward_mean(self.underlying, self.expiry, self.rf_rate, steps);
            return discounted_intrinsic(&self.option_type, average_level, self.strike, self.expiry, self.rf_rate);
        }
        exotics::asian_option(
            &self.option_type,
            self.underlying,
            self.strike,
            self.expiry,
            self.volatility,
            self.rf_rate,
            num_simulations,
            steps,
            &self.averaging_type,
            seed,
        )
    }

    pub fn analytical_approximation_price(&self, steps: Option<usize>) -> f64 {
        let steps = steps.unwrap_or(self.steps);
        if self.volatility <= 0.0 {
            let average_level = average_forward_mean(self.underlying, self.expiry, self.rf_rate, steps);
            return discounted_intrinsic(&self.option_type, average_level, self.strike, self.expiry, self.rf_rate);
        }
        if self.averaging_type == "geometric" {
            geometric_asian_price(
                &self.option_type,
                self.underlying,
                self.strike,
                self.expiry,
                self.volatility,
                self.rf_rate,
                steps,
            )
        } else {
            arithmetic_asian_lognormal_price(
                &self.option_type,
                self.underlying,
                self.strike,
                self.expiry,
                self.volatility,
                self.rf_rate,
                steps,
            )
        }
    }

    fn finite_difference_greek(&self, field: GreekField, bump_size: f64, sign: f64) -> f64 {
        let base = self.field_value(field);
        let up = self.bumped(field, base + bump_size);
        let down = self.bumped(field, base - bump_size);
        sign * (up - down) / (2.0 * bump_size)
    }

    fn bumped(&self, field: GreekField, value: f64) -> f64 {
        let mut o = self.clone();
        match field {
            GreekField::Underlying => o.underlying = value,
            GreekField::Expiry => o.expiry = value,
            GreekField::Volatility => o.volatility = value,
            GreekField::Rate => o.rf_rate = value,
        }
        o.analytical_approximation_price(None)
    }

    fn field_value(&self, field: GreekField) -> f64 {
        match field {
            GreekField::Underlying => self.underlying,
            GreekField::Expiry => self.expiry,
            GreekField::Volatility => self.volatility,
            GreekField::Rate => self.rf_rate,
        }
    }

    pub fn delta(&self) -> f64 {
        self.finite_difference_greek(GreekField::Underlying, 1e-2, 1.0)
    }

    pub fn gamma(&self) -> f64 {
        let bump = 1e-2;
        let price_center = self.analytical_approximation_price(None);
        let price_up = self.bumped(GreekField::Underlying, self.underlying + bump);
        let price_down = self.bumped(GreekField::Underlying, self.underlying - bump);
        (price_up - 2.0 * price_center + price_down) / (bump * bump)
    }

    pub fn theta(&self) -> f64 {
        self.finite_difference_greek(GreekField::Expiry, 1e-4, -1.0)
    }

    pub fn vega(&self) -> f64 {
        self.finite_difference_greek(GreekField::Volatility, 1e-4, 1.0)
    }

    pub fn rho(&self) -> f64 {
        self.finite_difference_greek(GreekField::Rate, 1e-4, 1.0)
    }
}

#[derive(Clone, Copy)]
enum GreekField {
    Underlying,
    Expiry,
    Volatility,
    Rate,
}

/// Free function matching the Python `asian_option` entry point:
/// `num_paths = None` -> analytical approximation, otherwise Monte Carlo.
pub fn asian_option(
    option_type: &str,
    spot: f64,
    strike: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    steps: usize,
    average_type: &str,
    num_paths: Option<usize>,
    seed: u64,
) -> f64 {
    if option_type != "call" && option_type != "put" {
        panic!("Option type must be either 'call' or 'put'.");
    }
    let option = AsianOption::new(spot, strike, expiry, rate, volatility, option_type, average_type, steps);
    match num_paths {
        None => option.analytical_approximation_price(Some(steps)),
        Some(n) => option.monte_carlo_price(n, Some(steps), seed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values from scripts/gen_asian_option.py (pure f64; MC uses
    /// the shared refgen normals with the same seeds).
    #[test]
    fn analytical_geometric_call() {
        let o = AsianOption::new(100.0, 100.0, 1.0, 0.05, 0.2, "call", "geometric", 252);
        assert!((o.analytical_approximation_price(None) - 5.565508831349255).abs() < 1e-9);
    }

    #[test]
    fn analytical_arithmetic_call() {
        let o = AsianOption::new(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 252);
        assert!((o.analytical_approximation_price(None) - 5.801498938949076).abs() < 1e-9);
    }

    #[test]
    fn analytical_arithmetic_put() {
        let o = AsianOption::new(100.0, 110.0, 1.0, 0.05, 0.2, "put", "arithmetic", 52);
        assert!((o.analytical_approximation_price(None) - 9.087690947769374).abs() < 1e-9);
    }

    #[test]
    fn analytical_zero_volatility() {
        let o = AsianOption::new(100.0, 100.0, 1.0, 0.05, 0.0, "call", "arithmetic", 252);
        // mean forward level over times k/252, k=1..252, discounted payoff
        assert!((o.analytical_approximation_price(None) - 2.4278855699852664).abs() < 1e-9);
    }

    #[test]
    fn monte_carlo_call() {
        let o = AsianOption::new(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 50);
        assert!((o.monte_carlo_price(1000, None, 42) - 5.343075879923734).abs() < 1e-9);
    }

    #[test]
    fn free_function_entry_points() {
        assert!(
            (asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 252, "geometric", None, 42)
                - 5.565508831349255)
                .abs()
                < 1e-9
        );
        assert!(
            (asian_option("call", 100.0, 100.0, 1.0, 0.2, 0.05, 50, "arithmetic", Some(1000), 42)
                - 5.343075879923734)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn greeks() {
        // Finite differences over ~1e-2/1e-4 bumps cancel ~1e-4 of the price,
        // amplifying cross-libm (Python vs Rust exp) ulp differences; the
        // convention's 1e-6 relative tolerance applies here.
        let o = AsianOption::new(100.0, 100.0, 1.0, 0.05, 0.2, "call", "arithmetic", 252);
        let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
        assert!(rel(o.delta(), 0.5931734274305711) < 1e-6);
        assert!(rel(o.gamma(), 0.032109755547793384) < 1e-6);
        assert!(rel(o.theta(), -3.432807050329778) < 1e-6);
        assert!(rel(o.vega(), 21.947043359307195) < 1e-6);
        assert!(rel(o.rho(), 24.762053642102266) < 1e-6);
    }
}
