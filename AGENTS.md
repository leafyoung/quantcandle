# AGENTS.md — quantcandle

Candle-framework port of torchquantlib, mirroring `../quantburn` module-for-module
(same f64 numerics, same `Xorshift64Star` RNG, same embedded expected values).
Read `../quantburn/CONVENTIONS.md` before converting further models.

## Commands

```bash
cargo test                 # 123 embedded tests (CPU)
cargo test --features mkl  # MKL-accelerated CPU variant
cargo run --bin crosscheck -- [--backend cpu|mkl] < cases.json
```

## Rules

- Coverage now mirrors quantburn 1:1 (same module names; lib.rs groups mirror
  quantburn's). When quantburn gains a module, port it here identically.
- Closed-form math in f64, formulas identical to the Python source and to
  quantburn — do not "improve" numerics, cross-checking depends on parity.
- Candle tensors (f32) only for vector work (path simulation) via
  `simulate_terminal_paths_tensor`; keep the scalar f64 path alongside as the
  parity reference.
- Stochastic code must use `crate::rng::Xorshift64Star` (bit-compatible with
  quantburn and its Python reference) — never float RNG.
- Backend reality: candle 0.9.2 has NO Vulkan backend upstream (CUDA/Metal/
  accelerate/MKL only; ROCm PR huggingface/candle#3801 still open). CPU and
  CPU+MKL are the supported configurations here; do not add a fake vulkan path.
