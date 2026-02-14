//! # runner_tensorrt
//!
//! TensorRT inference runner (experimental).
//!
//! This crate provides a TensorRT-based runner for NVIDIA GPU inference.
//! Since there are no stable Rust bindings for TensorRT, this implementation
//! uses a subprocess-based approach to call the `trtexec` CLI tool, or can
//! be extended with custom FFI bindings.
//!
//! **Requirements:**
//! - NVIDIA TensorRT SDK installed and `trtexec` on PATH.
//! - An ONNX model file as input.

use anyhow::{bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use runner_core::tracing::{debug, info, warn};
use runner_core::{ModelInfo, Runner, RunnerConfig};
use std::path::Path;
use std::process::Command;

/// A runner backed by NVIDIA TensorRT.
///
/// This runner converts ONNX models to TensorRT engines and runs inference.
/// Currently uses a file-based I/O approach for maximum compatibility.
pub struct TensorRTRunner {
    engine_path: String,
    config: RunnerConfig,
}

impl Runner for TensorRTRunner {
    fn from_config(config: &RunnerConfig) -> Result<Self> {
        let model_path = &config.model_path;

        // Verify the model file exists
        if !Path::new(model_path).exists() {
            bail!("Model file not found: {}", model_path);
        }

        // Check if trtexec is available
        let trtexec_available = Command::new("trtexec")
            .arg("--help")
            .output()
            .is_ok();

        if !trtexec_available {
            bail!(
                "TensorRT runner requires `trtexec` to be installed and on PATH. \
                 Install the NVIDIA TensorRT SDK from https://developer.nvidia.com/tensorrt"
            );
        }

        // Build the TensorRT engine from the ONNX model
        let engine_path = format!("{}.engine", model_path.trim_end_matches(".onnx"));

        if !Path::new(&engine_path).exists() {
            info!(
                model_path = %model_path,
                engine_path = %engine_path,
                "Building TensorRT engine from ONNX model"
            );

            let shape_str = config
                .input_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");

            let status = Command::new("trtexec")
                .args(&[
                    &format!("--onnx={}", model_path),
                    &format!("--saveEngine={}", engine_path),
                    &format!("--optShapes=input:{}", shape_str),
                    "--fp16", // use FP16 by default for performance
                ])
                .status()
                .context("Failed to run trtexec")?;

            if !status.success() {
                bail!("trtexec failed to build engine");
            }
            info!("TensorRT engine built successfully");
        } else {
            info!(engine_path = %engine_path, "Using cached TensorRT engine");
        }

        Ok(Self {
            engine_path,
            config: config.clone(),
        })
    }

    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        debug!(input_shape = ?input.shape(), "Running TensorRT inference");

        // Write input to a temporary file
        let input_path = format!("{}.input.bin", self.engine_path);
        let output_path = format!("{}.output.bin", self.engine_path);

        let flat: Vec<f32> = input.iter().cloned().collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&input_path, &bytes)?;

        // Run inference using trtexec
        let status = Command::new("trtexec")
            .args(&[
                &format!("--loadEngine={}", self.engine_path),
                &format!("--loadInputs=input:{}", input_path),
                &format!("--saveOutput={}", output_path),
            ])
            .status()
            .context("Failed to run trtexec inference")?;

        if !status.success() {
            bail!("trtexec inference failed");
        }

        // Read output
        let output_bytes = std::fs::read(&output_path)
            .context("Failed to read trtexec output")?;

        let output_floats: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Clean up temporary files
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);

        // We use the input shape as a heuristic for output shape.
        // In production, you'd parse the engine metadata for the actual output shape.
        let output_shape = input.shape().to_vec();
        debug!(output_shape = ?output_shape, "TensorRT inference complete");
        Ok(ArrayD::from_shape_vec(IxDyn(&output_shape), output_floats)?)
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model_path.clone(),
            backend: "tensorrt".to_string(),
            input_shape: self.config.input_shape.clone(),
            output_shape: vec![],
        }
    }
}

impl TensorRTRunner {
    /// Load from path (backward-compatible convenience).
    pub fn load(path: &str) -> Result<Self> {
        Self::from_config(&RunnerConfig {
            model_path: path.to_string(),
            use_cuda: true,
            ..RunnerConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorrt_missing_file() {
        let config = RunnerConfig {
            model_path: "nonexistent.onnx".to_string(),
            ..RunnerConfig::default()
        };
        let result = TensorRTRunner::from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensorrt_info() {
        // We can't fully test without TensorRT installed,
        // but we can test that the struct is constructible
        let runner = TensorRTRunner {
            engine_path: "test.engine".to_string(),
            config: RunnerConfig {
                model_path: "test.onnx".to_string(),
                ..RunnerConfig::default()
            },
        };
        let info = runner.info();
        assert_eq!(info.backend, "tensorrt");
    }
}
