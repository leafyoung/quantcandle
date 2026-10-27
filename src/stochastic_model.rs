//! Port of `torchquantlib/models/stochastic_model.py`.
//!
//! The Python file is a PyTorch base class (`StochasticModel`) providing
//! parameter storage/movement, device handling, gradient "anchor" hooks and
//! parameter constraint clamping — training-harness machinery with no reusable
//! numeric tensor core (rule 7). The numeric core that survives the port is
//! the parameter-constraint machinery: per-parameter min/max bounds
//! (`parameter_constraints`), `constrained_param` (clamped read) and
//! `_apply_constraints` (in-place clamp with a `nextafter`-adjusted lower
//! bound so a strict `min` bound stays attainable).

/// Constraint bounds for a single parameter, mirroring the Python
/// `{"min": ..., "max": ...}` dicts.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParamBounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ParamBounds {
    pub fn new(min: Option<f64>, max: Option<f64>) -> Self {
        Self { min, max }
    }
}

/// Shared base for stochastic simulation models: a named parameter store plus
/// per-parameter constraint bounds.
#[derive(Debug, Clone, Default)]
pub struct StochasticModel {
    params: std::collections::HashMap<String, f64>,
    constraints: std::collections::HashMap<String, ParamBounds>,
}

impl StochasticModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_param(&mut self, name: &str, value: f64) {
        self.params.insert(name.to_string(), value);
    }

    pub fn param(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }

    pub fn set_constraints(&mut self, name: &str, bounds: ParamBounds) {
        self.constraints.insert(name.to_string(), bounds);
    }

    /// `torch.nextafter(min, inf)` in the Python `_safe_lower_bound`: the
    /// smallest f64 strictly greater than `min`.
    fn safe_lower_bound(min: f64) -> f64 {
        let bits = min.to_bits();
        if min >= 0.0 {
            f64::from_bits(bits + 1)
        } else {
            f64::from_bits(bits - 1)
        }
    }

    /// Python `constrained_param`: clamp the parameter into its declared
    /// bounds (no-op when unconstrained).
    pub fn constrained_param(&self, name: &str) -> f64 {
        let value = self.params.get(name).copied().unwrap_or(0.0);
        match self.constraints.get(name) {
            None => value,
            Some(b) => {
                let lo = b.min.map(Self::safe_lower_bound);
                let hi = b.max;
                let mut v = value;
                if let Some(lo) = lo {
                    if v < lo {
                        v = lo;
                    }
                }
                if let Some(hi) = hi {
                    if v > hi {
                        v = hi;
                    }
                }
                v
            }
        }
    }

    /// Python `_apply_constraints`: clamp the stored parameter in place
    /// (lower bound via `_safe_lower_bound`, upper bound as declared).
    pub fn apply_constraints(&mut self) {
        let names: Vec<String> = self.constraints.keys().cloned().collect();
        for name in names {
            let bounds = self.constraints[&name];
            if let Some(min) = bounds.min {
                let lower = Self::safe_lower_bound(min);
                if let Some(v) = self.params.get_mut(&name) {
                    if *v < lower {
                        *v = lower;
                    }
                    if let Some(max) = bounds.max {
                        if *v > max {
                            *v = max;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic constraint behaviour (no RNG).
    #[test]
    fn constrained_param_clamps_to_bounds() {
        let mut m = StochasticModel::new();
        m.set_param("kappa", -0.5);
        m.set_param("rho", 1.5);
        m.set_param("theta", 0.04);
        m.set_constraints("kappa", ParamBounds::new(Some(1e-6), None));
        m.set_constraints("rho", ParamBounds::new(Some(-0.9989), Some(0.9989)));

        // Unconstrained passes through.
        assert!((m.constrained_param("theta") - 0.04).abs() < 1e-12);
        // Below min -> clamped to nextafter(min, inf), strictly above min.
        let k = m.constrained_param("kappa");
        assert!(k > 1e-6 && k - 1e-6 < 1e-18);
        // Above max -> clamped.
        assert!((m.constrained_param("rho") - 0.9989).abs() < 1e-12);
    }

    #[test]
    fn apply_constraints_clamps_in_place() {
        let mut m = StochasticModel::new();
        m.set_param("sigma", -1.0);
        m.set_param("rho", -2.0);
        m.set_constraints("sigma", ParamBounds::new(Some(1e-6), None));
        m.set_constraints("rho", ParamBounds::new(Some(-0.9989), Some(0.9989)));
        m.apply_constraints();
        let expected_lower = 1e-6_f64.next_up();
        assert!((m.param("sigma").unwrap() - expected_lower).abs() < 1e-24);
        assert!((m.param("rho").unwrap() - (-0.9989)).abs() < 1e-12);
    }

    #[test]
    fn safe_lower_bound_is_nextafter() {
        assert_eq!(StochasticModel::safe_lower_bound(1e-6), 1e-6_f64.next_up());
        assert_eq!(StochasticModel::safe_lower_bound(0.0), 5e-324);
    }
}
