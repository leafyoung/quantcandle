//! Port of `credit_risk/structural_model.py` (Merton structural model).
//!
//! Equity is priced as a Black-Scholes call option on the firm's assets:
//! `E = V * N(d1) - D * exp(-r*T) * N(d2)` with the usual d1/d2.

/// Standard normal CDF, computed from a self-contained double-precision
/// `erfc` (series for small args, incomplete-gamma series/continued fraction
/// for the tails; verified to ~1e-15 against Python `math.erfc`).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

const A: f64 = 0.5; // incomplete-gamma shape parameter
const GLN: f64 = 0.5723649429247001; // ln Gamma(0.5) = 0.5 * ln(pi)
const SQRT_PI: f64 = 1.7724538509055159;

/// erfc for z >= 0, accurate to ~1e-15.
fn erfc_pos(z: f64) -> f64 {
    if z < 1.0 {
        // erf Taylor series (alternating, fast for |z| < 1).
        let x2 = z * z;
        let mut term = z;
        let mut s = z;
        for n in 1..80 {
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            s += add;
            if add.abs() < 1e-18 * s.abs() {
                break;
            }
        }
        1.0 - s * 2.0 / SQRT_PI
    } else {
        let xx = z * z;
        if xx < A + 1.0 {
            // Regularized lower incomplete gamma P(0.5, z^2) by series.
            let mut ap = A;
            let mut s = 1.0 / A;
            let mut de = s;
            for _ in 0..1000 {
                ap += 1.0;
                de *= xx / ap;
                s += de;
                if de.abs() < s.abs() * 1e-16 {
                    break;
                }
            }
            1.0 - s * (-xx + A * xx.ln() - GLN).exp()
        } else {
            // Regularized upper incomplete gamma Q(0.5, z^2) by the
            // modified Lentz continued fraction (Numerical Recipes gcf).
            let fmin = 1e-300;
            let mut b = xx + 1.0 - A;
            let mut c = 1.0 / fmin;
            let mut d = 1.0 / b;
            let mut h = d;
            for i in 1..1000 {
                let an = -(i as f64) * (i as f64 - A);
                b += 2.0;
                d = an * d + b;
                if d.abs() < fmin {
                    d = fmin;
                }
                c = b + an / c;
                if c.abs() < fmin {
                    c = fmin;
                }
                d = 1.0 / d;
                let del = d * c;
                h *= del;
                if (del - 1.0).abs() < 1e-16 {
                    break;
                }
            }
            (-xx + A * xx.ln() - GLN).exp() * h
        }
    }
}

/// Complementary error function.
pub fn erfc(x: f64) -> f64 {
    let q = erfc_pos(x.abs());
    if x >= 0.0 {
        q
    } else {
        2.0 - q
    }
}

/// Merton structural model: equity value of the firm.
pub fn merton_model(asset_value: f64, debt: f64, volatility: f64, rate: f64, expiry: f64) -> f64 {
    let sqrt_t = expiry.sqrt();
    let d1 = ((asset_value / debt).ln() + (rate + 0.5 * volatility * volatility) * expiry)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    asset_value * norm_cdf(d1) - debt * (-rate * expiry).exp() * norm_cdf(d2)
}

#[cfg(test)]
mod tests {
    /// Reference: torch (float64) `merton_model` from the Python source,
    /// e.g. merton_model(tensor(100.0), tensor(80.0), tensor(0.25),
    /// tensor(0.05), tensor(1.0)).item() == 25.412511998314308.
    #[test]
    fn matches_python_reference() {
        let cases = [
            (100.0, 80.0, 0.25, 0.05, 1.0, 25.412511998314308),
            (100.0, 120.0, 0.3, 0.03, 2.0, 12.113095785829557),
            (50.0, 40.0, 0.2, 0.0, 0.5, 10.154557237945262),
        ];
        for (v, d, vol, r, t, expected) in cases {
            let got = super::merton_model(v, d, vol, r, t);
            assert!(
                (got - expected).abs() <= 1e-8 * expected.abs(),
                "got {got}, expected {expected}"
            );
        }
    }

    /// erfc accuracy sanity vs math.erfc reference values.
    #[test]
    fn erfc_accuracy() {
        let cases = [
            (0.0_f64, 1.0_f64),
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (1.5, 0.033894853524689274),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-5),
            (-1.2, 1.9103139782296354),
        ];
        for (x, expected) in cases {
            assert!(
                (super::erfc(x) - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "erfc({x})"
            );
        }
    }
}
