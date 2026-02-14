//! # runner_tch
//!
//! TorchScript inference runner using the `tch-rs` crate (LibTorch bindings).
//! Supports both CPU and CUDA inference.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use runner_core::{ModelInfo, Runner, RunnerConfig};
use tch::{CModule, Device, Tensor};

/// A runner backed by LibTorch (TorchScript models).
pub struct TchRunner {
    module: CModule,
    device: Device,
    config: RunnerConfig,
}

impl Runner for TchRunner {
    fn from_config(config: &RunnerConfig) -> Result<Self> {
        let device = if config.use_cuda && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let module = CModule::load_on_device(&config.model_path, device)?;
        Ok(Self {
            module,
            device,
            config: config.clone(),
        })
    }

    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let input_shape: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let flat: Vec<f32> = input.iter().cloned().collect();
        let t = Tensor::from_slice(&flat)
            .view(input_shape.as_slice())
            .to_device(self.device);

        let out = self.module.forward_ts(&[t])?;
        let out = out.to_device(Device::Cpu);

        let out_shape: Vec<usize> = out.size().iter().map(|&d| d as usize).collect();
        let numel: usize = out_shape.iter().product();
        let mut out_vec = vec![0f32; numel];
        out.copy_data(&mut out_vec, numel);

        Ok(ArrayD::from_shape_vec(IxDyn(&out_shape), out_vec)?)
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model_path.clone(),
            backend: "tch".to_string(),
            input_shape: self.config.input_shape.clone(),
            output_shape: vec![], // determined at runtime
        }
    }
}

/// Legacy convenience methods for backward compatibility.
impl TchRunner {
    /// Load a TorchScript model from a file path.
    pub fn load(path: &str, use_cuda: bool) -> Result<Self> {
        let config = RunnerConfig {
            model_path: path.to_string(),
            use_cuda,
            ..RunnerConfig::default()
        };
        Self::from_config(&config)
    }

    /// Run inference with a 4D input array (backward-compatible).
    pub fn run_4d(&self, input: ndarray::Array4<f32>) -> Result<ArrayD<f32>> {
        let dyn_input = input.into_dyn();
        self.run(&dyn_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_config_default_no_cuda() {
        let config = RunnerConfig {
            model_path: "nonexistent.pt".to_string(),
            use_cuda: false,
            ..RunnerConfig::default()
        };
        // Loading will fail because the file doesn't exist,
        // but we can test that config is accepted
        let result = TchRunner::from_config(&config);
        assert!(result.is_err()); // expected: file not found
    }
}
