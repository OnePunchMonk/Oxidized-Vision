//! # runner_ort
//!
//! ONNX inference runner backed by Microsoft's ONNX Runtime (via the `ort` crate).
//!
//! Compared to the pure-Rust `tract` backend, ONNX Runtime ships hand-tuned,
//! hardware-specific kernels for the ops that dominate vision model latency
//! (grouped/depthwise convolution, layer norm, attention, resize) and applies
//! aggressive graph-level fusion (Conv+BatchNorm+Activation, MatMul+Add, GELU,
//! LayerNorm) at `GraphOptimizationLevel::Level3`. For standard CNN/ViT vision
//! backbones this typically yields lower latency and higher throughput than
//! `tract`, especially on CPU (oneDNN-backed kernels) and CUDA/TensorRT GPUs.

use anyhow::{anyhow, Result};
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
use ort::{
    ep::CPU,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use runner_core::tracing::{debug, info, warn};
use runner_core::{ModelInfo, Runner, RunnerConfig};
use std::sync::Mutex;

/// A runner backed by ONNX Runtime, with full graph-level operator fusion
/// applied ahead of time (`Level3`), giving vision backbones fused
/// Conv+BN+Activation / MatMul+Add / LayerNorm kernels instead of running
/// each op individually.
pub struct OrtRunner {
    // `Session::run` takes `&mut self` in `ort`; wrap in a `Mutex` so
    // `OrtRunner` can still satisfy `Runner: Send + Sync` for use behind
    // an `Arc` in multi-threaded servers.
    session: Mutex<Session>,
    config: RunnerConfig,
    output_shape: Mutex<Vec<usize>>,
}

impl Runner for OrtRunner {
    fn from_config(config: &RunnerConfig) -> Result<Self> {
        info!(
            model_path = %config.model_path,
            input_shape = ?config.input_shape,
            use_cuda = config.use_cuda,
            optimize = config.optimize,
            "Loading ONNX model with ONNX Runtime"
        );

        let opt_level = if config.optimize {
            GraphOptimizationLevel::Level3
        } else {
            GraphOptimizationLevel::Disable
        };

        let mut builder = Session::builder()
            .map_err(|e| anyhow!("runner_ort: failed to create session builder: {e}"))?
            .with_optimization_level(opt_level)
            .map_err(|e| anyhow!("runner_ort: failed to set optimization level: {e}"))?;

        #[cfg(feature = "cuda")]
        if config.use_cuda {
            match builder
                .clone()
                .with_execution_providers([CUDA::default().build()])
            {
                Ok(b) => builder = b,
                Err(e) => warn!(
                    error = %e,
                    "Failed to register CUDA execution provider, falling back to CPU"
                ),
            }
        }
        #[cfg(not(feature = "cuda"))]
        if config.use_cuda {
            warn!(
                "use_cuda=true but runner_ort was built without the `cuda` feature; falling back to CPU"
            );
        }

        builder = builder
            .with_execution_providers([CPU::default().build()])
            .map_err(|e| anyhow!("runner_ort: failed to register CPU execution provider: {e}"))?;

        let session = builder.commit_from_file(&config.model_path)?;

        info!("Model loaded and optimized (Level3 fusion) successfully");

        Ok(Self {
            session: Mutex::new(session),
            config: config.clone(),
            output_shape: Mutex::new(Vec::new()),
        })
    }

    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        debug!(input_shape = ?input.shape(), "Running ONNX Runtime inference");

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("runner_ort: session mutex poisoned"))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| anyhow!("runner_ort: model has no inputs"))?;

        let in_shape: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
        let in_data: Vec<f32> = input.iter().copied().collect();
        let input_value = Tensor::from_array((in_shape, in_data))?;

        let outputs = session.run(ort::inputs![input_name => input_value])?;

        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        let out_shape: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let output = ArrayD::from_shape_vec(IxDyn(&out_shape), out_data.to_vec())?;

        *self
            .output_shape
            .lock()
            .map_err(|_| anyhow!("runner_ort: output_shape mutex poisoned"))? = out_shape;

        debug!(output_shape = ?output.shape(), "Inference complete");
        Ok(output)
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model_path.clone(),
            backend: "ort".to_string(),
            input_shape: self.config.input_shape.clone(),
            output_shape: self
                .output_shape
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default(),
        }
    }
}

impl OrtRunner {
    /// Load an ONNX model from a file path (backward-compatible convenience).
    pub fn load(path: &str) -> Result<Self> {
        Self::from_config(&RunnerConfig {
            model_path: path.to_string(),
            ..RunnerConfig::default()
        })
    }

    /// Load an ONNX model from a file path with a custom input shape.
    pub fn load_with_shape(path: &str, input_shape: &[usize]) -> Result<Self> {
        Self::from_config(&RunnerConfig {
            model_path: path.to_string(),
            input_shape: input_shape.to_vec(),
            ..RunnerConfig::default()
        })
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
    fn test_load_missing_file_errors() {
        let config = RunnerConfig {
            model_path: "nonexistent.onnx".to_string(),
            ..RunnerConfig::default()
        };
        let result = OrtRunner::from_config(&config);
        assert!(result.is_err());
    }
}
