//! Backend device selection.
//!
//! Candle (as of 0.9.2) ships only CUDA / Metal / accelerate / MKL backends —
//! there is no Vulkan backend upstream (unlike Burn, whose wgpu backend
//! provides Vulkan for quantburn). On this AMD/Linux box the realistic
//! configurations are therefore:
//! - CPU (default)
//! - CPU + MKL (`mkl` feature) as the accelerated-CPU variant
//! The upstream ROCm PR (huggingface/candle#3801) is still open; revisit if it
//! lands.

use candle_core::Device;

/// Build a Candle device for a named backend (`"cpu"` or `"mkl"`; MKL only
/// exists as a compile-time feature, so it selects CPU with MKL-enabled
/// kernels when built with `--features mkl`).
pub fn make_device(backend: &str) -> Result<Device, String> {
    match backend {
        "cpu" | "mkl" => Ok(Device::Cpu),
        "vulkan" => Err("candle has no vulkan backend upstream (0.9.2); use burn/quantburn for vulkan".to_string()),
        other => Err(format!("unknown backend '{other}' (use 'cpu' or 'mkl')")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_works() {
        assert!(make_device("cpu").is_ok());
        assert!(make_device("vulkan").is_err());
        assert!(make_device("gpu").is_err());
    }
}
