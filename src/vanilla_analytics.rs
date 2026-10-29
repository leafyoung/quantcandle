//! Analytic Black-Scholes barrier and binary option prices, ported from
//! tf-quant-finance `black_scholes/vanilla_prices.py` (`barrier_price` and
//! `binary_price`).
//!
//! `barrier_price_analytic` is the continuous-monitoring single-barrier
//! closed form (Reiner-Rubinstein / Merton-style, the Clewlow-Strickland /
//! Haug mask formulation used by tff): 12 terms, each an algebraic factor
//! times a normal CDF, selected by barrier type and by whether the strike is
//! above the barrier. `binary_price` is the cash-or-nothing binary paying one
//! unit of cash, with the dividend-corrected forward
//! `F = S * exp(-q*T) / DF` (the FIXED tff version; ignoring q mis-prices
//! binaries when q > 0).
//!
//! Pure f64 scalar math mirroring the tff formulas branch-for-branch,
//! including the rebate terms and the `sqrt_var > 0` clamp of the binary
//! price. Values verified against QuantLib (AnalyticBarrierEngine on
//! BarrierOption, CashOrNothingPayoff + AnalyticEuropeanEngine) and against
//! tff itself in float64; embedded literals below come from those runs.
//!
//! Rebate conventions: by default (`barrier_price_analytic`, tff semantics)
//! a knock-out rebate is paid at hit and a knock-in rebate at expiry.
//! `barrier_price_analytic_expiry_rebate` instead pays a knock-out rebate at
//! expiry iff the barrier is never touched (the standard Reiner-Rubinstein
//! knock-out-rebate-at-expiry term `R * DF * P(survival)`, obtained by
//! re-weighting the same survival-probability terms the knock-in rows use);
//! knock-in results never depend on the choice.
//!
//! QuantLib cross-check (AnalyticBarrierEngine, QL 1.44): QL's knock-in
//! rebate equals the survival term above to <=2e-15, and QL's knock-out
//! rebate is paid at hit with the Haug `lam = sqrt(mu^2 + 2r/sigma^2)`
//! parameterization (matches QL to <=2e-15), which differs from tff's
//! `b`-parameterized hit-time term (e.g. by +0.0298 on the S=100 K=100 B=85
//! R=3 vol=25% T=1 case). QL knock-out totals therefore match neither path;
//! the expiry-rebate test literals are composed from pure QL numbers,
//! `QL KO(R=0) + QL KI rebate term`, which equals main + survival term
//! exactly (engine agreement <=5.3e-15).

use crate::black_scholes_merton::normal_cdf;

/// Mask rows for strike > barrier, indexed by
/// `(is_barrier_down << 2) + (is_knock_out << 1) + is_call`
/// (same rows and order as tff's `mask_matrix_greater_strike`).
const MASK_GREATER_STRIKE: [[f64; 12]; 8] = [
    [1., 1., -1., -1., 0., 0., 1., 1., 1., 1., 0., 0.], // up and in put
    [1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],   // up and in call
    [0., 0., 1., 1., 0., 0., -1., -1., 0., 0., 1., 1.], // up and out put
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],   // up and out call
    [0., 0., 1., 1., -1., -1., 1., 1., 0., 0., 1., 1.], // down and in put
    [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],   // down and in call
    [1., 1., -1., -1., 1., 1., -1., -1., 0., 0., 1., 1.], // down and out put
    [1., 1., 0., 0., -1., -1., 0., 0., 0., 0., 1., 1.], // down and out call
];

/// Mask rows for strike <= barrier (tff's `mask_matrix_lower_strike`).
const MASK_LOWER_STRIKE: [[f64; 12]; 8] = [
    [0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.], // up and in put
    [0., 0., 1., 1., -1., -1., 1., 1., 1., 1., 0., 0.], // up and in call
    [1., 1., 0., 0., -1., -1., 0., 0., 0., 0., 1., 1.], // up and out put
    [1., 1., -1., -1., 1., 1., -1., -1., 0., 0., 1., 1.], // up and out call
    [1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.], // down and in put
    [1., 1., -1., -1., 0., 0., 1., 1., 1., 1., 0., 0.], // down and in call
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.], // down and out put
    [0., 0., 1., 1., 0., 0., -1., -1., 0., 0., 1., 1.], // down and out call
];

/// Price a continuously-monitored single-barrier option in closed form
/// (Black-Scholes, tff `barrier_price` semantics).
///
/// Arguments follow the tff keyword names: `discount_rate` is the risk-free
/// rate r, `dividend_rate` the continuous dividend yield q. The barrier must
/// not be breached at inception (down barrier below `spot`, up barrier above
/// `spot`), matching the formula's domain. `rebate` is the fixed cash payout
/// if a knock-out barrier is breached (paid at hit) or if a knock-in barrier
/// is never breached (paid at expiry).
pub fn barrier_price_analytic(
    volatility: f64,
    strike: f64,
    expiry: f64,
    spot: f64,
    barrier: f64,
    rebate: f64,
    discount_rate: f64,
    dividend_rate: f64,
    is_barrier_down: bool,
    is_knock_out: bool,
    is_call: bool,
) -> f64 {
    barrier_price_analytic_impl(
        volatility,
        strike,
        expiry,
        spot,
        barrier,
        rebate,
        discount_rate,
        dividend_rate,
        is_barrier_down,
        is_knock_out,
        is_call,
        false,
    )
}

/// Same as [`barrier_price_analytic`], but a knock-out rebate is paid at
/// expiry iff the barrier is never touched (value `R * DF * P(survival)`,
/// the standard Reiner-Rubinstein knock-out-rebate-at-expiry term) instead
/// of at hit. Knock-in results are identical under both functions because
/// knock-in rebates are paid at expiry in both conventions.
pub fn barrier_price_analytic_expiry_rebate(
    volatility: f64,
    strike: f64,
    expiry: f64,
    spot: f64,
    barrier: f64,
    rebate: f64,
    discount_rate: f64,
    dividend_rate: f64,
    is_barrier_down: bool,
    is_knock_out: bool,
    is_call: bool,
) -> f64 {
    barrier_price_analytic_impl(
        volatility,
        strike,
        expiry,
        spot,
        barrier,
        rebate,
        discount_rate,
        dividend_rate,
        is_barrier_down,
        is_knock_out,
        is_call,
        true,
    )
}

fn barrier_price_analytic_impl(
    volatility: f64,
    strike: f64,
    expiry: f64,
    spot: f64,
    barrier: f64,
    rebate: f64,
    discount_rate: f64,
    dividend_rate: f64,
    is_barrier_down: bool,
    is_knock_out: bool,
    is_call: bool,
    rebate_paid_at_expiry: bool,
) -> f64 {
    let index =
        ((is_barrier_down as usize) << 2) + ((is_knock_out as usize) << 1) + (is_call as usize);
    let masks = if strike > barrier {
        MASK_GREATER_STRIKE[index]
    } else {
        MASK_LOWER_STRIKE[index]
    };
    // Knock-out with the rebate paid at expiry: move the rebate weight from
    // the hit-time terms (columns 11-12) onto the survival-probability terms
    // (columns 9-10) that the knock-in rows use. Knock-in rows are unaffected.
    let masks = if is_knock_out && rebate_paid_at_expiry {
        let mut m = masks;
        m[8] = m[10];
        m[9] = m[11];
        m[10] = 0.0;
        m[11] = 0.0;
        m
    } else {
        masks
    };

    let call_or_put = if is_call { 1.0 } else { -1.0 };
    let below_or_above = if is_barrier_down { 1.0 } else { -1.0 };

    // Parameters for the six integrals (tff names, minus the unicode).
    let sqrt_var = volatility * expiry.sqrt();
    let mu = (discount_rate - dividend_rate) - volatility * volatility / 2.0;
    let lamda = 1.0 + mu / (volatility * volatility);
    let x = (spot / strike).ln() / sqrt_var + lamda * sqrt_var;
    let x1 = (spot / barrier).ln() / sqrt_var + lamda * sqrt_var;
    let y = (barrier * barrier / (spot * strike)).ln() / sqrt_var + lamda * sqrt_var;
    let y1 = (barrier / spot).ln() / sqrt_var + lamda * sqrt_var;
    let b = (mu * mu + 2.0 * volatility * volatility * discount_rate) / (volatility * volatility);
    let z = (barrier / spot).ln() / sqrt_var + b * sqrt_var;
    let a = mu / (volatility * volatility);

    let discount_factors = (-discount_rate * expiry).exp();
    let barriers_ratio = barrier / spot;
    let spots_term = call_or_put * spot * (-dividend_rate * expiry).exp();
    let strikes_term = call_or_put * strike * discount_factors;

    let terms = [
        spots_term,
        -strikes_term,
        spots_term,
        -strikes_term,
        spots_term * barriers_ratio.powf(2.0 * lamda),
        -strikes_term * barriers_ratio.powf(2.0 * lamda - 2.0),
        spots_term * barriers_ratio.powf(2.0 * lamda),
        -strikes_term * barriers_ratio.powf(2.0 * lamda - 2.0),
        rebate * discount_factors,
        -rebate * discount_factors * barriers_ratio.powf(2.0 * lamda - 2.0),
        rebate * barriers_ratio.powf(a + b),
        rebate * barriers_ratio.powf(a - b),
    ];
    let cdfs = [
        normal_cdf(call_or_put * x),
        normal_cdf(call_or_put * (x - sqrt_var)),
        normal_cdf(call_or_put * x1),
        normal_cdf(call_or_put * (x1 - sqrt_var)),
        normal_cdf(below_or_above * y),
        normal_cdf(below_or_above * (y - sqrt_var)),
        normal_cdf(below_or_above * y1),
        normal_cdf(below_or_above * (y1 - sqrt_var)),
        normal_cdf(below_or_above * (x1 - sqrt_var)),
        normal_cdf(below_or_above * (y1 - sqrt_var)),
        normal_cdf(below_or_above * z),
        normal_cdf(below_or_above * (z - 2.0 * b * sqrt_var)),
    ];

    let mut price = 0.0;
    for k in 0..12 {
        price += masks[k] * terms[k] * cdfs[k];
    }
    price
}

/// Price a cash-or-nothing binary option paying one unit of cash at expiry
/// (Black-Scholes, tff `binary_price` semantics with the dividend-corrected
/// forward `F = S * exp(-q*T) / DF`).
///
/// The call pays 1 if the spot ends above the strike, the put 1 if below, so
/// the price is the discounted risk-neutral probability of ending in the
/// money. Zero volatility collapses to the discounted step payoff (tff's
/// `sqrt_var > 0` clamp).
pub fn binary_price(
    volatility: f64,
    strike: f64,
    expiry: f64,
    spot: f64,
    discount_rate: f64,
    dividend_rate: f64,
    is_call: bool,
) -> f64 {
    let discount_factors = (-discount_rate * expiry).exp();
    let forwards = spot * (-dividend_rate * expiry).exp() / discount_factors;
    let sqrt_var = volatility * expiry.sqrt();
    let undiscounted_calls = if sqrt_var > 0.0 {
        let d2 = (forwards / strike).ln() / sqrt_var - sqrt_var / 2.0;
        normal_cdf(d2)
    } else if forwards > strike {
        1.0
    } else {
        0.0
    };
    if is_call {
        discount_factors * undiscounted_calls
    } else {
        discount_factors * (1.0 - undiscounted_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes_merton::black_scholes_merton;

    const R: f64 = 0.05;
    const Q: f64 = 0.02;

    /// Golden values verified against QuantLib AnalyticBarrierEngine
    /// (tf-quant-finance cross-check harness, 2026-10).
    #[test]
    fn barrier_golden_values() {
        // down-out B=85 K=100 call vol=20% T=1, S=100
        let v = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 85.0, 0.0, R, Q, true, true, true);
        assert!((v - 8.759460458735848).abs() < 1e-10);
        // up-out B=125 K=100 put vol=25% T=1
        let v = barrier_price_analytic(
            0.25, 100.0, 1.0, 100.0, 125.0, 0.0, R, Q, false, true, false,
        );
        assert!((v - 7.902600854338467).abs() < 1e-10);
        // down-in B=85 K=110 call vol=25% T=1
        let v = barrier_price_analytic(0.25, 110.0, 1.0, 100.0, 85.0, 0.0, R, Q, true, false, true);
        assert!((v - 0.5479573225762513).abs() < 1e-10);
        // up-in B=125 K=90 put vol=20% T=1
        let v = barrier_price_analytic(
            0.20, 90.0, 1.0, 100.0, 125.0, 0.0, R, Q, false, false, false,
        );
        assert!((v - 0.013422992543647733).abs() < 1e-12);
    }

    /// Extra grid points covering both mask tables (K > B and K < B), all
    /// four knock combinations, deep moneyness and S != 100; each value
    /// reproduced QuantLib to <= 1.1e-14.
    #[test]
    fn barrier_extra_ql_grid() {
        // down-out call, K > B, vol=30%
        let v = barrier_price_analytic(0.30, 95.0, 1.0, 100.0, 90.0, 0.0, R, Q, true, true, true);
        assert!((v - 9.711137714182236).abs() < 1e-10);
        // up-in call, deep OTM K=140
        let v = barrier_price_analytic(
            0.25, 140.0, 1.0, 100.0, 120.0, 0.0, R, Q, false, false, true,
        );
        assert!((v - 1.5111745977832367).abs() < 1e-11);
        // up-out put, K > B
        let v = barrier_price_analytic(
            0.25, 130.0, 1.0, 100.0, 125.0, 0.0, R, Q, false, true, false,
        );
        assert!((v - 24.5605437578551).abs() < 1e-10);
        // down-in put, K < B, T=2, vol=30%
        let v = barrier_price_analytic(0.30, 75.0, 2.0, 100.0, 85.0, 0.0, R, Q, true, false, false);
        assert!((v - 3.925672050043495).abs() < 1e-11);
        // down-out put, K < B: barrier below both spot and strike region -> 0
        let v = barrier_price_analytic(0.25, 80.0, 1.0, 100.0, 85.0, 0.0, R, Q, true, true, false);
        assert!(v.abs() < 1e-12);
        // up-in call, S=90, T=2, vol=40%
        let v =
            barrier_price_analytic(0.40, 110.0, 2.0, 90.0, 105.0, 0.0, R, Q, false, false, true);
        assert!((v - 14.800183620042578).abs() < 1e-11);
    }

    /// Knock-in + knock-out = vanilla European (rebate = 0), matching the
    /// BSM vanilla price of this crate.
    #[test]
    fn barrier_in_plus_out_equals_vanilla() {
        // down barrier, call
        let pin =
            barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, R, Q, true, false, true);
        let pout =
            barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, R, Q, true, true, true);
        let vanilla = black_scholes_merton("call", "european", 100.0, 100.0, 1.0, 0.20, R, Q);
        assert!((pin + pout - vanilla).abs() < 1e-10);

        // up barrier, put, T=2
        let pin = barrier_price_analytic(
            0.25, 100.0, 2.0, 100.0, 120.0, 0.0, R, Q, false, false, false,
        );
        let pout = barrier_price_analytic(
            0.25, 100.0, 2.0, 100.0, 120.0, 0.0, R, Q, false, true, false,
        );
        let vanilla = black_scholes_merton("put", "european", 100.0, 100.0, 2.0, 0.25, R, Q);
        assert!((pin + pout - vanilla).abs() < 1e-10);
    }

    /// Rebate handling (tff semantics). The knock-in rebate value is also
    /// QuantLib-exact; the knock-out one is tff-f64-verified because QL pays
    /// knock-out rebates at a different time (expiry vs hit).
    #[test]
    fn barrier_with_rebate() {
        // down-out call, rebate 3 paid at hit (tff semantics)
        let v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 3.0, R, Q, true, true, true);
        assert!((v - 11.475189709298022).abs() < 1e-10);
        // down-in call, rebate 1.5 paid at expiry if never knocked in
        let v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 1.5, R, Q, true, false, true);
        assert!((v - 1.8890581732854028).abs() < 1e-11);
    }

    /// Golden binary values verified against QuantLib CashOrNothingPayoff +
    /// AnalyticEuropeanEngine (and tff f64).
    #[test]
    fn binary_golden_values() {
        // ATM call, K=100, T=1, vol=20%
        let v = binary_price(0.20, 100.0, 1.0, 100.0, R, Q, true);
        assert!((v - 0.49458109105322323).abs() < 1e-12);
        // put, K=110, T=2, vol=30%
        let v = binary_price(0.30, 110.0, 2.0, 100.0, R, Q, false);
        assert!((v - 0.557506659561213).abs() < 1e-12);
    }

    /// Extra QL-verified grid points: deep ITM/OTM, high vol.
    #[test]
    fn binary_extra_ql_grid() {
        // deep ITM call, K=70
        let v = binary_price(0.20, 70.0, 1.0, 100.0, R, Q, true);
        assert!((v - 0.9194836337641996).abs() < 1e-12);
        // deep OTM call, K=140
        let v = binary_price(0.20, 140.0, 1.0, 100.0, R, Q, true);
        assert!((v - 0.04879969669114354).abs() < 1e-12);
        // put, K=70, T=2, vol=30%
        let v = binary_price(0.30, 70.0, 2.0, 100.0, R, Q, false);
        assert!((v - 0.1996576342569846).abs() < 1e-12);
        // ATM call, vol=50%
        let v = binary_price(0.50, 100.0, 1.0, 100.0, R, Q, true);
        assert!((v - 0.4039439177288213).abs() < 1e-12);
    }

    /// call + put binaries sum to the discount factor; zero volatility
    /// collapses to the discounted step payoff (tff's clamp).
    #[test]
    fn binary_put_call_sum_and_zero_vol() {
        let c = binary_price(0.25, 105.0, 1.5, 100.0, R, Q, true);
        let p = binary_price(0.25, 105.0, 1.5, 100.0, R, Q, false);
        assert!((c + p - (-R * 1.5f64).exp()).abs() < 1e-14);
        // forward 100*e^{0.03} > 90 -> call pays DF; < 110 -> put pays DF
        assert!((binary_price(0.0, 90.0, 1.0, 100.0, R, Q, true) - (-R).exp()).abs() < 1e-14);
        assert!((binary_price(0.0, 110.0, 1.0, 100.0, R, Q, false) - (-R).exp()).abs() < 1e-14);
        assert!(binary_price(0.0, 110.0, 1.0, 100.0, R, Q, true).abs() < 1e-14);
    }

    /// Knock-out rebate paid at expiry iff the barrier is never touched
    /// (`R * DF * P(survival)`). QL's AnalyticBarrierEngine pays KO rebates
    /// at hit (Haug lam parameterization), so these literals are composed
    /// from pure QL numbers: QL KO(R=0) + QL KI rebate term (== the survival
    /// term); the engine reproduces them to <= 5.3e-15.
    #[test]
    fn barrier_ko_rebate_paid_at_expiry() {
        // known case: down-out call S=100 K=100 B=85 R=3 vol=25% T=1
        let v = barrier_price_analytic_expiry_rebate(
            0.25, 100.0, 1.0, 100.0, 85.0, 3.0, R, Q, true, true, true,
        );
        assert!((v - 11.300833094739227).abs() < 1e-10);
        // up-out put S=100 K=90 B=112 R=4 vol=25% T=0.6, r=3% q=1%
        let v = barrier_price_analytic_expiry_rebate(
            0.25, 90.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01, false, true, false,
        );
        assert!((v - 4.509289715741195).abs() < 1e-10);
    }

    /// Knock-in results are invariant under the rebate-timing choice: KI
    /// rebates are paid at expiry in both conventions (bit-identical over
    /// all eight KI mask rows).
    #[test]
    fn barrier_rebate_flag_ki_invariance() {
        let cases = [
            (0.25, 130.0, 1.0, 100.0, 125.0, 2.0, R, Q, false, false, false), // up-in put, K > B
            (0.25, 140.0, 1.0, 100.0, 120.0, 1.0, R, Q, false, false, true),  // up-in call, K > B
            (0.22, 95.0, 0.8, 100.0, 88.0, 2.0, 0.06, 0.01, true, false, false), // down-in put, K > B
            (0.25, 110.0, 1.0, 100.0, 85.0, 1.5, R, Q, true, false, true),    // down-in call, K > B
            (0.25, 90.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01, false, false, false), // up-in put, K < B
            (0.25, 100.0, 0.6, 100.0, 112.0, 4.0, 0.03, 0.01, false, false, true), // up-in call, K < B
            (0.22, 80.0, 0.8, 100.0, 88.0, 2.0, 0.06, 0.01, true, false, false), // down-in put, K < B
            (0.25, 80.0, 1.0, 100.0, 85.0, 1.5, R, Q, true, false, true),     // down-in call, K < B
        ];
        for (vol, k, t, s, bar, reb, r, q, down, _, call) in cases {
            let a = barrier_price_analytic(vol, k, t, s, bar, reb, r, q, down, false, call);
            let b =
                barrier_price_analytic_expiry_rebate(vol, k, t, s, bar, reb, r, q, down, false, call);
            assert_eq!(a, b, "KI invariance failed: vol={vol} k={k} t={t} down={down} call={call}");
        }
    }

    /// The default (hit-timing) path keeps the tff convention untouched;
    /// the expiry-rebate variant genuinely moves the KO rebate term.
    #[test]
    fn barrier_rebate_flag_default_unchanged() {
        let hit = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 3.0, R, Q, true, true, true);
        assert!((hit - 11.475189709298022).abs() < 1e-10);
        let expiry = barrier_price_analytic_expiry_rebate(
            0.25, 100.0, 1.0, 100.0, 85.0, 3.0, R, Q, true, true, true,
        );
        assert!((expiry - 11.300833094739227).abs() < 1e-10);
        assert!((hit - expiry).abs() > 0.1);
    }
}
