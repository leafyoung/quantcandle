//! Deterministic normal sampler shared by all modules, so Rust results and the
//! Python reference harness (scripts/refgen.py) draw identical standard
//! normals and agree to floating-point error.

pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in (0, 1), same bit recipe as the Python reference harness.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller, returning in the same order as the Python harness.
    pub fn next_std_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn normals(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_std_normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity with scripts/refgen.py (verified: identical bits).
    #[test]
    fn matches_python_reference() {
        let mut r = Xorshift64Star::new(42);
        assert!((r.next_std_normal() - 0.2960330150871692).abs() < 1e-12);
        assert!((r.next_std_normal() - 0.6443779794453868).abs() < 1e-12);
    }
}
