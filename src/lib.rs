//! quantcandle — Candle-framework port of `torchquantlib` (Python/PyTorch),
//! mirroring quantburn's module layout and numerics so the three libraries can
//! be cross-checked numerically.
//!
//! Conventions (following quantburn's CONVENTIONS.md):
//! - closed-form scalar math in f64, identical formulas to the Python source;
//! - stochastic modules draw standard normals from `crate::rng::Xorshift64Star`
//!   (bit-identical to quantburn's sampler and its Python reference harness);
//! - Candle tensors (`candle_core::Tensor`, f32) are used for vector work —
//!   path simulation — on the selected backend device (CPU, or Vulkan with the
//!   `vulkan` feature), so backend parity is actually exercised;
//! - each module carries embedded tests whose expected values match quantburn's
//!   (same RNG, same f64 math).

pub mod black_scholes_merton;
pub mod binomial_tree_model;
pub mod cir;
pub mod american_option;
pub mod asian_option;
pub mod bermudan_option;
pub mod black_karasinski;
pub mod bond_pricer;
pub mod dupire_local_volatility;
pub mod exotics;
pub mod expected_shortfall;
pub mod fixed_income_forward;
pub mod future_pricer;
pub mod gans;
pub mod greeks;
pub mod letsberational;
pub mod lmm;
pub mod malliavin;
pub mod reduced_form_model;
pub mod sabr;
pub mod scenario_analysis;
pub mod stochastic_model;
pub mod stress_testing;
pub mod structural_model;
pub mod valuation_adjustment;
pub mod vanilla_analytics;
pub mod var;
pub mod yield_curve_construction;
pub mod device;
pub mod heston;
pub mod hull_white;
pub mod option_utils;
pub mod rng;
pub mod vasicek;

pub use device::make_device;
