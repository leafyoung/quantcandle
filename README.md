# quantcandle

Candle-framework port of `torchquantlib` (Python/PyTorch), mirroring
`../quantburn` module-for-module so the three libraries can be cross-checked
numerically. See `../quantburn/CONVENTIONS.md` for the conversion conventions
followed (f64 closed forms, shared Xorshift64Star RNG, embedded parity tests).

## Modules
Full coverage of quantburn / torchquantlib: options (BSM, American, Asian,
Bermudan, exotics incl. digital/barrier/lookback/chooser/rainbow/quanto/basket,
let's-be-rational implied vol), models (binomial CRR, Heston with candle-tensor
Euler path, SABR + Hagan IV, Vasicek, CIR, Hull-White, Black-Karasinski, LMM,
Dupire local vol, GAN, stochastic model), bonds (zero/coupon/callable/putable/
convertible lattices, forwards, futures), risk (Greeks, Malliavin, VaR, ES,
scenario/stress), credit (structural + reduced-form), XVA, yield-curve
construction. 123 embedded tests, expected values identical to quantburn's.

## Backends
Candle 0.9.2 has **no Vulkan backend upstream** (only CUDA/Metal/accelerate/MKL;
the ROCm PR huggingface/candle#3801 is still open). Configurations here:
- `cargo build` — CPU
- `cargo build --features mkl` — CPU + MKL kernels (the accelerated variant)
Burn keeps Vulkan via burn-wgpu; that pairing is quantburn-only.

## Cross-check runner
    cargo run --bin crosscheck [-- --backend cpu|mkl] < cases.json
Reads the runtime-agnostic case JSON (shared with the cross-check suite at
`../cross-check`) and writes JSON results to stdout. Supported models: bsm, binomial, heston_cf, heston_mc, vasicek_zcb, cir_zcb,
hull_white_zcb, black_karasinski_zcb, american_option, bermudan_option,
asian_option, black76, sabr_iv, sabr_price, digital_option, barrier_option,
lookback_option, chooser_option, bond_zero_coupon, bond_coupon, bond_callable,
bond_putable, greeks_european (values map), var, es.

## Tests
35 embedded tests; expected values are the same literals as quantburn's tests
(identical RNG draws and f64 math), plus a tensor-vs-scalar f32 parity test.
