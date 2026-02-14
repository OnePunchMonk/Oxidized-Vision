//! # runner_core
//!
//! Shared trait and types for OxidizedVision inference runners.
//! All backend crates (`runner_tch`, `runner_tract`, `runner_tensorrt`)
//! implement the [`Runner`] trait defined here, enabling generic usage
//! in applications like web servers and CLI tools.

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Metadata describing a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Human-readable name of the model.
    pub name: String,
    /// Backend used for inference (e.g. "tch", "tract", "tensorrt").
    pub backend: String,
    /// Expected input shape (e.g. `[1, 3, 256, 256]`).
    pub input_shape: Vec<usize>,
    /// Expected output shape (may be empty if unknown until first run).
    pub output_shape: Vec<usize>,
}

/// Configuration for loading a runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Path to the model file.
    pub model_path: String,
    /// Expected input shape.
    pub input_shape: Vec<usize>,
    /// Whether to use CUDA (applicable to GPU-capable backends).
    pub use_cuda: bool,
    /// Whether to apply backend-specific optimizations.
    pub optimize: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            input_shape: vec![1, 3, 256, 256],
            use_cuda: false,
            optimize: true,
        }
    }
}

/// The core trait that all inference backends must implement.
///
/// This enables applications to be generic over the choice of runtime.
///
/// # Example
/// ```ignore
/// let runner = TractRunner::from_config(&config)?;
/// let input = ndarray::ArrayD::<f32>::zeros(IxDyn(&[1, 3, 256, 256]));
/// let output = runner.run(&input)?;
/// println!("Output shape: {:?}", output.shape());
/// ```
pub trait Runner: Send + Sync {
    /// Create a runner from a configuration.
    fn from_config(config: &RunnerConfig) -> Result<Self>
    where
        Self: Sized;

    /// Run inference on the given input tensor.
    /// Input and output are dynamic-dimensional arrays to support arbitrary shapes.
    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>>;

    /// Return metadata about the loaded model.
    fn info(&self) -> ModelInfo;
}

/// Helper: convert a shape slice `&[usize]` into `IxDyn`.
pub fn shape_to_ix(shape: &[usize]) -> IxDyn {
    IxDyn(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_serialize() {
        let info = ModelInfo {
            name: "test_model".to_string(),
            backend: "mock".to_string(),
            input_shape: vec![1, 3, 256, 256],
            output_shape: vec![1, 64, 256, 256],
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("test_model"));
    }

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.input_shape, vec![1, 3, 256, 256]);
        assert!(!config.use_cuda);
        assert!(config.optimize);
    }

    #[test]
    fn test_shape_to_ix() {
        let ix = shape_to_ix(&[1, 3, 256, 256]);
        assert_eq!(ix.as_array_view().len(), 4);
    }
}
