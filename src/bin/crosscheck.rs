//! Cross-check runner: reads a case file (JSON) and prices every case with
//! quantcandle, emitting results as JSON on stdout.
//!
//! Case format (runtime-agnostic, shared with the cross-check suite):
//! { "cases": [ { "id": "...", "model": "<model>", "params": { ... } } ] }
//! Output:
//! { "backend": "...", "results": [ { "id", "ok", "price"? , "values"?, "error"? } ] }
//!
//! Models: bsm, binomial, heston_cf, heston_mc, vasicek_zcb, cir_zcb,
//! hull_white_zcb, black_karasinski_zcb, american_option, bermudan_option,
//! asian_option, black76, sabr_iv, sabr_price, digital_option,
//! barrier_option, lookback_option, chooser_option, bond_zero_coupon,
//! bond_coupon, bond_callable, bond_putable, greeks_european, var, es.
//!
//! Usage: crosscheck [--backend cpu|mkl] < cases.json

use quantcandle::{
    american_option::american_option,
    asian_option::asian_option,
    bermudan_option::bermudan_option,
    binomial_tree_model::price_binomial_tree,
    black_scholes_merton::black_scholes_merton,
    bond_pricer::{callable_bond, coupon_bond, putable_bond, zero_coupon_bond},
    cir::zero_coupon_bond_price as cir_zcb,
    exotics,
    expected_shortfall::calculate_es,
    greeks::european_option_greeks,
    heston::{characteristic_function_price, monte_carlo_option_price, HestonParams},
    hull_white::HullWhiteModel,
    sabr::{black76_price, hagan_lognormal_implied_volatility, option_price as sabr_option_price, SabrParams},
    vasicek::zero_coupon_bond_price as vasicek_zcb,
    var::calculate_var,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Read;

#[derive(Deserialize)]
struct CaseFile {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    model: String,
    params: serde_json::Value,
}

#[derive(serde::Serialize)]
struct CaseResult {
    id: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    price: Option<f64>,
    /// Multi-output cases (e.g. greeks); empty for scalar cases.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    values: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn scalar(price: f64) -> Result<CaseResult, String> {
    Ok(CaseResult { id: String::new(), ok: true, price: Some(price), values: BTreeMap::new(), error: None })
}

fn f64_of(v: &serde_json::Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("missing or non-numeric param '{key}'"))
}

fn usize_of(v: &serde_json::Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("missing or invalid integer param '{key}'"))
}

fn i64_of(v: &serde_json::Value, key: &str) -> Result<i64, String> {
    v.get(key)
        .and_then(|x| x.as_i64())
        .ok_or_else(|| format!("missing or invalid integer param '{key}'"))
}

fn str_of<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("missing or non-string param '{key}'"))
}

fn array_of(v: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        .ok_or_else(|| format!("missing or non-array param '{key}'"))
}

fn opt_f64(v: &serde_json::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

fn opt_usize(v: &serde_json::Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
}

fn opt_str<'a>(v: &'a serde_json::Value, key: &str, default: &'a str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or(default)
}

fn opt_shift(v: &serde_json::Value) -> Option<f64> {
    v.get("shift").and_then(|x| x.as_f64())
}

fn opt_seed(v: &serde_json::Value) -> u64 {
    v.get("seed").and_then(|x| x.as_u64()).unwrap_or(2007)
}

fn sabr_params(p: &serde_json::Value) -> Result<SabrParams, String> {
    Ok(SabrParams {
        alpha: f64_of(p, "alpha")?,
        beta: f64_of(p, "beta")?,
        rho: f64_of(p, "rho")?,
        nu: f64_of(p, "nu")?,
        f0: f64_of(p, "f0")?,
        shift: opt_f64(p, "shift", 0.0),
    })
}

fn price_heston_params(p: &serde_json::Value) -> Result<HestonParams, String> {
    Ok(HestonParams {
        kappa: f64_of(p, "kappa")?,
        theta: f64_of(p, "theta")?,
        sigma_v: f64_of(p, "sigma_v")?,
        rho: f64_of(p, "rho")?,
        v0: f64_of(p, "v0")?,
        mu: opt_f64(p, "mu", 0.0),
    })
}

fn price_case(model: &str, p: &serde_json::Value) -> Result<CaseResult, String> {
    match model {
        "bsm" => scalar(black_scholes_merton(
            str_of(p, "option_type")?,
            str_of(p, "option_style")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            f64_of(p, "dividend")?,
        )),
        "binomial" => scalar(price_binomial_tree(
            str_of(p, "option_type")?,
            str_of(p, "option_style")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            usize_of(p, "steps")?,
            None,
        )),
        "heston_cf" => {
            let params = price_heston_params(p)?;
            scalar(characteristic_function_price(
                params,
                f64_of(p, "spot")?,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                f64_of(p, "rate")?,
                str_of(p, "option_type")?,
                f64_of(p, "dividend")?,
                opt_f64(p, "integration_limit", 100.0),
                opt_usize(p, "integration_points", 2000),
            ))
        }
        "heston_mc" => {
            let params = price_heston_params(p)?;
            scalar(monte_carlo_option_price(
                params,
                f64_of(p, "spot")?,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                f64_of(p, "rate")?,
                str_of(p, "option_type")?,
                f64_of(p, "dividend")?,
                usize_of(p, "n_paths")?,
                usize_of(p, "steps")?,
                opt_seed(p),
            ))
        }
        "vasicek_zcb" => scalar(vasicek_zcb(
            f64_of(p, "kappa")?,
            f64_of(p, "theta")?,
            f64_of(p, "sigma")?,
            f64_of(p, "r0")?,
            f64_of(p, "maturity")?,
        )),
        "cir_zcb" => scalar(cir_zcb(
            f64_of(p, "kappa")?,
            f64_of(p, "theta")?,
            f64_of(p, "sigma")?,
            f64_of(p, "r0")?,
            f64_of(p, "maturity")?,
        )),
        "hull_white_zcb" => {
            let hw = HullWhiteModel::new(f64_of(p, "a")?, f64_of(p, "sigma")?, f64_of(p, "r0")?);
            scalar(hw.zero_coupon_bond_price(f64_of(p, "maturity")?, None))
        }
        "black_karasinski_zcb" => {
            let bk = quantcandle::black_karasinski::BlackKarasinski::new(
                f64_of(p, "a")?,
                f64_of(p, "sigma")?,
                f64_of(p, "r0")?,
            );
            scalar(bk.zero_coupon_bond_price(f64_of(p, "maturity")?, None, opt_usize(p, "integration_steps", 100)))
        }
        "american_option" => scalar(american_option(
            str_of(p, "option_type")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            usize_of(p, "steps")?,
        )),
        "bermudan_option" => {
            let dates: Vec<usize> = p
                .get("exercise_dates")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                .unwrap_or_default();
            scalar(bermudan_option(
                str_of(p, "option_type")?,
                f64_of(p, "spot")?,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                f64_of(p, "volatility")?,
                f64_of(p, "rate")?,
                usize_of(p, "steps")?,
                &dates,
            ))
        }
        "asian_option" => scalar(asian_option(
            str_of(p, "option_type")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            usize_of(p, "steps")?,
            opt_str(p, "average_type", "arithmetic"),
            p.get("num_paths").and_then(|x| x.as_u64()).map(|x| x as usize),
            opt_seed(p),
        )),
        "black76" => scalar(black76_price(
            str_of(p, "option_type")?,
            f64_of(p, "forward")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
        )),
        "sabr_iv" => {
            let sp = sabr_params(p)?;
            scalar(hagan_lognormal_implied_volatility(
                &sp,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                p.get("forward").and_then(|x| x.as_f64()),
                opt_shift(p),
            ))
        }
        "sabr_price" => {
            let sp = sabr_params(p)?;
            scalar(sabr_option_price(
                &sp,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                f64_of(p, "rate")?,
                str_of(p, "option_type")?,
                opt_f64(p, "dividend", 0.0),
                opt_str(p, "model", "hagan"),
                opt_shift(p),
            ))
        }
        "digital_option" => scalar(exotics::digital_option(
            str_of(p, "option_type")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            f64_of(p, "payout")?,
            usize_of(p, "n_paths")?,
            opt_seed(p),
        )),
        "barrier_option" => scalar(exotics::barrier_option(
            str_of(p, "option_type")?,
            str_of(p, "barrier_type")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "barrier")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            usize_of(p, "n_paths")?,
            usize_of(p, "steps")?,
            opt_seed(p),
        )),
        "lookback_option" => scalar(exotics::lookback_option(
            str_of(p, "option_type")?,
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            usize_of(p, "n_paths")?,
            opt_str(p, "strike_type", "fixed"),
            opt_seed(p),
        )),
        "chooser_option" => scalar(exotics::chooser_option(
            f64_of(p, "spot")?,
            f64_of(p, "strike")?,
            f64_of(p, "expiry")?,
            f64_of(p, "volatility")?,
            f64_of(p, "rate")?,
            f64_of(p, "choose_time")?,
        )),
        "bond_zero_coupon" => scalar(zero_coupon_bond(
            f64_of(p, "face_value")?,
            f64_of(p, "rate")?,
            f64_of(p, "maturity")?,
        )),
        "bond_coupon" => scalar(coupon_bond(
            f64_of(p, "face_value")?,
            f64_of(p, "coupon_rate")?,
            f64_of(p, "rate")?,
            i64_of(p, "periods")?,
        )),
        "bond_callable" => scalar(callable_bond(
            f64_of(p, "face_value")?,
            f64_of(p, "coupon_rate")?,
            f64_of(p, "rate")?,
            i64_of(p, "periods")?,
            f64_of(p, "call_price")?,
            i64_of(p, "call_period")?,
        )),
        "bond_putable" => scalar(putable_bond(
            f64_of(p, "face_value")?,
            f64_of(p, "coupon_rate")?,
            f64_of(p, "rate")?,
            i64_of(p, "periods")?,
            f64_of(p, "put_price")?,
            i64_of(p, "put_period")?,
        )),
        "greeks_european" => {
            let g = european_option_greeks(
                f64_of(p, "spot")?,
                f64_of(p, "strike")?,
                f64_of(p, "expiry")?,
                f64_of(p, "rate")?,
                f64_of(p, "volatility")?,
                opt_usize(p, "n_paths", 65536),
                opt_seed(p),
            );
            Ok(CaseResult {
                id: String::new(),
                ok: true,
                price: None,
                values: BTreeMap::from([
                    ("delta".into(), g.delta),
                    ("gamma".into(), g.gamma),
                    ("vega".into(), g.vega),
                    ("theta".into(), g.theta),
                    ("rho".into(), g.rho),
                ]),
                error: None,
            })
        }
        "var" => scalar(calculate_var(&array_of(p, "returns")?, f64_of(p, "confidence_level")?)?),
        "es" => scalar(calculate_es(&array_of(p, "returns")?, f64_of(p, "confidence_level")?)?),
        other => Err(format!("unknown model '{other}'")),
    }
}

fn main() {
    let mut backend = "cpu".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--backend" {
            backend = args.next().unwrap_or_else(|| "cpu".into());
        }
    }

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    let file: CaseFile = serde_json::from_str(&input).expect("parse case JSON");

    let results: Vec<CaseResult> = file
        .cases
        .iter()
        .map(|c| {
            let mut r = match price_case(&c.model, &c.params) {
                Ok(r) => r,
                Err(e) => CaseResult {
                    id: String::new(),
                    ok: false,
                    price: None,
                    values: BTreeMap::new(),
                    error: Some(e),
                },
            };
            r.id = c.id.clone();
            r
        })
        .collect();

    let out = serde_json::json!({ "backend": backend, "results": results });
    println!("{}", serde_json::to_string_pretty(&out).expect("serialize"));
}
