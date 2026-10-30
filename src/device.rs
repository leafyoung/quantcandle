//! Backend device selection.
//!
//! Candle ships CUDA / Metal / accelerate / MKL backends — there is no Vulkan
//! backend upstream (unlike Burn, whose wgpu backend provides Vulkan for
//! quantburn), and ROCm is still an open PR (huggingface/candle#3801).
//! Configurations:
//! - CPU (default)
//! - CPU + MKL (`mkl` feature) as the accelerated-CPU variant
//! - NVIDIA GPU (`cuda` feature; requires the CUDA toolkit to build, and a
//!   CUDA GPU at runtime). `make_device("cuda")` is only available when the
//!   crate is built with the `cuda` feature; otherwise it returns an error.

use candle_core::Device;

/// Build a Candle device for a named backend (`"cpu"`, `"mkl"`, or `"cuda"`;
/// MKL only exists as a compile-time feature, so it selects CPU with
/// MKL-enabled kernels when built with `--features mkl`; CUDA requires the
/// `cuda` compile-time feature and a CUDA GPU at runtime).
pub fn make_device(backend: &str) -> Result<Device, String> {
    match backend {
        "cpu" | "mkl" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0).map_err(|e| format!("cuda device init failed: {e}")),
        "vulkan" => Err("candle has no vulkan backend upstream; use burn/quantburn for vulkan".to_string()),
        other => Err(format!("unknown backend '{other}' (use 'cpu', 'mkl' or 'cuda')")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_works() {
        assert!(make_device("cpu").is_ok());
        assert!(make_device("mkl").is_ok());
        assert!(make_device("vulkan").is_err());
        assert!(make_device("gpu").is_err());
    }

    /// GPU smoke test (only compiled with the `cuda` feature; requires a
    /// CUDA GPU at runtime): tensor arithmetic on the CUDA device must match
    /// the CPU result.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_smoke() {
        let dev = make_device("cuda").expect("cuda device init");
        let a = candle_core::Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &dev)
            .unwrap();
        let b = candle_core::Tensor::from_vec(vec![5.0f64, 6.0, 7.0, 8.0], (2, 2), &dev)
            .unwrap();
        let c = (a + b).unwrap();
        let vals = c.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(vals, vec![6.0, 8.0, 10.0, 12.0]);
        // matmul (exercises cublas)
        let m = (a.matmul(&b.transpose(0, 1).unwrap()).unwrap())
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        assert_eq!(m, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
