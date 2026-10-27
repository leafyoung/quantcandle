//! Port of `torchquantlib/models/local_volatility/dupire_local_volatility.py`
//! (Dupire local volatility model). Euler logspot simulation where the local
//! volatility surface `sigma(S, t)` is supplied as a function. Module RNG
//! seed: 2009.

/// Python `DupireLocalVol.simulate`: `steps` Euler steps updating
/// `spot *= exp(rate*dt + sigma(spot, t)*sqrt(dt)*z)` with `t` evaluated at
/// `(step + 1) * dt`. Negativity from the surface raises. Normals come from
/// `crate::rng::Xorshift64Star` with `seed` (2009 in tests).
pub fn simulate(
    local_vol_func: &dyn Fn(f64, f64) -> f64,
    s0: f64,
    t: f64,
    n: usize,
    steps: usize,
    rate: f64,
    seed: u64,
) -> Vec<f64> {
    let steps = steps.max(1);
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut spot = vec![s0; n];
    let mut rng = crate::rng::Xorshift64Star::new(seed);
    for step in 0..steps {
        let time = (step + 1) as f64 * dt;
        let mut diffusion = Vec::with_capacity(n);
        for s in &spot {
            let sigma = local_vol_func(*s, time);
            if sigma < 0.0 {
                panic!("Negative volatility encountered");
            }
            diffusion.push(sigma * sqrt_dt);
        }
        for (i, s) in spot.iter_mut().enumerate() {
            let z = rng.next_std_normal();
            *s *= (rate * dt + diffusion[i] * z).exp();
        }
    }
    spot
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Constant surface degenerates to geometric Brownian motion; expected
    /// values from scripts/gen_dupire_local_volatility.py with refgen
    /// normals, seed 2009, tol 1e-12 relative.
    #[test]
    fn simulate_constant_surface_matches_reference() {
        let out = simulate(&|_s, _t| 0.2, 100.0, 1.0, 6, 32, 0.05, 2009);
        let expected = [
            92.02458413250487,
            87.40917483785537,
            95.42764799048972,
            93.92498775669627,
            125.67360115565047,
            117.31017140023587,
        ];
        for (got, want) in out.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    /// Surface depending on spot and time is honoured (deterministic
    /// spot-independent check embedded in the gen script too).
    #[test]
    fn simulate_state_dependent_surface_matches_reference() {
        let surface = |s: f64, t: f64| 0.1 + 0.2 * (s / 100.0).min(2.0) * (1.0 / (1.0 + t));
        let out = simulate(&surface, 100.0, 0.5, 4, 16, 0.02, 2009);
        let expected = [
            96.02933173574249,
            96.63160341139788,
            93.29985284075049,
            142.54048922865078,
        ];
        for (got, want) in out.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    #[test]
    #[should_panic(expected = "Negative volatility encountered")]
    fn negative_volatility_panics() {
        simulate(&|_s, _t| -0.1, 100.0, 1.0, 2, 4, 0.0, 2009);
    }
}
