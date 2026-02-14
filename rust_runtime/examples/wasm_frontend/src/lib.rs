//! # wasm_frontend
//!
//! WebAssembly frontend for running OxidizedVision models in the browser.
//! Uses the tract runner to load ONNX models from bytes and run inference.

use wasm_bindgen::prelude::*;
use runner_core::Runner;
use runner_tract::TractRunner;
use ndarray::{ArrayD, IxDyn};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Greet function for testing WASM module loading.
#[wasm_bindgen]
pub fn greet() -> String {
    console_error_panic_hook::set_once();
    "Hello from OxidizedVision WASM! 🚀".to_string()
}

/// Run inference on an ONNX model loaded from bytes.
///
/// # Arguments
/// * `model_data` - Raw bytes of the ONNX model file
/// * `input_data` - Flattened f32 input data
/// * `batch_size` - Batch dimension
/// * `channels` - Number of channels
/// * `height` - Input height
/// * `width` - Input width
///
/// # Returns
/// A JSON string containing the output shape and flattened data.
#[wasm_bindgen]
pub fn run_inference_wasm(
    model_data: &[u8],
    input_data: &[f32],
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> String {
    console_error_panic_hook::set_once();

    let input_shape = vec![batch_size, channels, height, width];
    let expected_len: usize = input_shape.iter().product();

    if input_data.len() != expected_len {
        return format!(
            "{{\"error\": \"Input data length {} doesn't match shape {:?} (expected {})\"}}",
            input_data.len(), input_shape, expected_len
        );
    }

    console_log!("Loading model from {} bytes...", model_data.len());

    let runner = match TractRunner::load_from_bytes(model_data, &input_shape) {
        Ok(r) => r,
        Err(e) => return format!("{{\"error\": \"Failed to load model: {}\"}}", e),
    };

    console_log!("Model loaded. Running inference...");

    let input = match ArrayD::<f32>::from_shape_vec(IxDyn(&input_shape), input_data.to_vec()) {
        Ok(arr) => arr,
        Err(e) => return format!("{{\"error\": \"Failed to create input array: {}\"}}", e),
    };

    match runner.run(&input) {
        Ok(output) => {
            let output_shape: Vec<String> = output.shape().iter().map(|d| d.to_string()).collect();
            let output_data: Vec<String> = output.iter().map(|v| format!("{:.6}", v)).collect();
            format!(
                "{{\"status\": \"success\", \"output_shape\": [{}], \"data_preview\": [{}]}}",
                output_shape.join(", "),
                output_data.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
            )
        }
        Err(e) => format!("{{\"error\": \"Inference failed: {}\"}}", e),
    }
}

/// Get model info after loading from bytes.
#[wasm_bindgen]
pub fn get_model_info(model_data: &[u8], height: usize, width: usize) -> String {
    console_error_panic_hook::set_once();

    let input_shape = vec![1, 3, height, width];
    match TractRunner::load_from_bytes(model_data, &input_shape) {
        Ok(runner) => {
            let info = runner.info();
            format!(
                "{{\"name\": \"{}\", \"backend\": \"{}\", \"input_shape\": {:?}}}",
                info.name, info.backend, info.input_shape
            )
        }
        Err(e) => format!("{{\"error\": \"{}\"}}", e),
    }
}
