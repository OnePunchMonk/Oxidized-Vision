//! # runner_tract
//!
//! ONNX inference runner using the `tract` crate (pure-Rust ONNX engine).
//! Supports CPU inference and WebAssembly targets.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use runner_core::tracing::{debug, info, warn};
use runner_core::{ModelInfo, Runner, RunnerConfig};
use tract_onnx::prelude::*;

/// A runner backed by the tract ONNX engine.
pub struct TractRunner {
    model: TypedRunnableModel<TypedModel>,
    config: RunnerConfig,
}

impl Runner for TractRunner {
    fn from_config(config: &RunnerConfig) -> Result<Self> {
        info!(
            model_path = %config.model_path,
            input_shape = ?config.input_shape,
            optimize = config.optimize,
            "Loading ONNX model with tract"
        );

        let input_shape: Vec<usize> = config.input_shape.clone();
        let fact = f32::fact(&input_shape);

        let mut model = tract_onnx::onnx().model_for_path(&config.model_path)?;
        model = model.with_input_fact(0, fact.into())?;

        let model = if config.optimize {
            debug!("Applying tract graph optimizations");
            model.into_optimized()?
        } else {
            model.into_typed()?
        };

        let model = model.into_runnable()?;
        info!("Model loaded successfully");

        Ok(Self {
            model,
            config: config.clone(),
        })
    }

    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        debug!(input_shape = ?input.shape(), "Running tract inference");
        let tract_input: tract_onnx::prelude::Tensor = input.clone().into();
        let result = self.model.run(tvec!(tract_input.into()))?;
        let output = result[0].to_array_view::<f32>()?.to_owned();
        debug!(output_shape = ?output.shape(), "Inference complete");
        Ok(output)
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model_path.clone(),
            backend: "tract".to_string(),
            input_shape: self.config.input_shape.clone(),
            output_shape: vec![], // determined at runtime
        }
    }
}

impl TractRunner {
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

    /// Load an ONNX model from raw bytes (useful for WASM targets).
    pub fn load_from_bytes(data: &[u8], input_shape: &[usize]) -> Result<Self> {
        info!(bytes = data.len(), input_shape = ?input_shape, "Loading model from bytes");
        let fact = f32::fact(input_shape);
        let mut cursor = std::io::Cursor::new(data);

        let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
        model = model.with_input_fact(0, fact.into())?;
        let model = model.into_optimized()?.into_runnable()?;

        Ok(Self {
            model,
            config: RunnerConfig {
                model_path: "<bytes>".to_string(),
                input_shape: input_shape.to_vec(),
                ..RunnerConfig::default()
            },
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
    fn test_runner_config_default() {
        let config = RunnerConfig {
            model_path: "nonexistent.onnx".to_string(),
            ..RunnerConfig::default()
        };
        let result = TractRunner::from_config(&config);
        assert!(result.is_err()); // expected: file not found
    }

    #[test]
    fn test_load_from_bytes_invalid() {
        let data = b"not a valid onnx model";
        let result = TractRunner::load_from_bytes(data, &[1, 3, 256, 256]);
        assert!(result.is_err());
    }
}
