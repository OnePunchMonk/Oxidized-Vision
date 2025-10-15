# OxidizedVision Architecture

This document provides a detailed overview of the OxidizedVision architecture, components, and workflows.

## 1. High-Level Overview

OxidizedVision is designed as a pipeline that transforms PyTorch models into efficient Rust artifacts. The core philosophy is to provide a seamless and automated workflow for developers, from a trained model in Python to a deployed inference service in Rust.

The main stages of the pipeline are:

1.  **Conversion**: A PyTorch model is converted to TorchScript and then to the ONNX format.
2.  **Optimization**: The ONNX model is optimized for inference.
3.  **Packaging**: The optimized model is packaged into a Rust crate with a chosen runtime.
4.  **Deployment**: The Rust crate can be built into a binary, a library, or a WebAssembly module.

```mermaid
graph TD
    subgraph "Python: Conversion Pipeline"
        A[PyTorch Model] -- "torch.jit.trace" --> B(TorchScript .pt);
        B -- "torch.onnx.export" --> C(ONNX .onnx);
        C -- "onnx-simplifier" --> D(Optimized .onnx);
    end

    subgraph "Rust: Inference Runtimes"
        B --> E("tch-rs Runtime");
        D --> F("tract Runtime");
    end

    E -- "Build" --> G[Native Binary / Library];
    F -- "Build" --> H[Native Binary or WASM Module];
```

## 2. Core Components

### 2.1. Python Client (`python_client/`)

The Python client is the main entry point for users. It provides:

-   **A Command-Line Interface (CLI)**: Built with `Typer` (based on `Click`), it offers commands like `convert`, `validate`, `package`, and `serve`.
-   **Conversion Utilities**: Python scripts that handle the conversion from PyTorch to TorchScript and ONNX. It uses `torch.jit.trace` for tracing and `torch.onnx.export` for ONNX conversion.
-   **Validation**: A module to compare the outputs of the original PyTorch model, the TorchScript model, and the ONNX model to ensure correctness.
-   **Configuration Management**: Uses `Pydantic` and YAML for strongly-typed configurations, making the conversion process reproducible.

### 2.2. Rust Runtimes (`rust_runtime/`)

The Rust part of the project is a workspace containing several crates. The key components are the runners, which are responsible for loading and executing the models.

#### Runner Trait

To ensure interoperability, we define a common `Runner` trait that each runtime implements. This allows the application layer (e.g., a web server) to be generic over the choice of runtime.

```rust
pub trait Runner {
    fn load_model(path: &str) -> Result<Self> where Self: Sized;
    fn run(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array4<f32>>;
    fn info(&self) -> ModelInfo;
}
```

#### `runner_tch`

-   **Backend**: Uses the `tch-rs` crate, which provides bindings to LibTorch (the C++ backend of PyTorch).
-   **Model Format**: Loads TorchScript models (`.pt`).
-   **Pros**:
    -   Guarantees 100% parity with PyTorch.
    -   Supports GPU inference via CUDA.
-   **Cons**:
    -   Requires LibTorch to be installed, which is a large dependency.
    -   Less portable than the `tract` runtime.

#### `runner_tract`

-   **Backend**: Uses the `tract` crate, a pure-Rust ONNX inference engine.
-   **Model Format**: Loads ONNX models (`.onnx`).
-   **Pros**:
    -   Lightweight and has no external dependencies (besides a C compiler).
    -   Highly portable, including support for WebAssembly (WASM).
    -   Excellent performance on CPUs.
-   **Cons**:
    -   May not support all ONNX operators, especially those from very new or complex models.
    -   GPU support is experimental.

### 2.3. Example Applications (`rust_runtime/examples/`)

To demonstrate how to use the packaged models, OxidizedVision includes several example applications:

-   `image_server`: An `actix-web` based server that exposes a `/predict` endpoint for model inference.
-   `denoiser_cli`: A command-line tool that takes an image, runs it through a model, and saves the output.
-   `wasm_frontend`: An example of how to compile a model and its runner to WebAssembly and use it in a simple web page.

## 3. The Conversion Pipeline in Detail

The `oxidizedvision convert` command orchestrates the following steps:

1.  **Load PyTorch Model**: The user provides a PyTorch model class and a checkpoint file. The tool instantiates the model and loads the weights.
2.  **Trace to TorchScript**: The model is traced using `torch.jit.trace` with a dummy input of a specified shape. This produces a TorchScript model (`.pt` file). Tracing is preferred over scripting for its simplicity, but scripting can be used for models with data-dependent control flow.
3.  **Export to ONNX**: The TorchScript model is then exported to ONNX format using `torch.onnx.export`. This step requires specifying an `opset_version`.
4.  **Optimize ONNX (Optional)**: The generated ONNX graph can be optimized using tools like `onnx-simplifier` to fuse operations and simplify the graph, which often leads to faster inference.
5.  **Validate**: Throughout the process, validation checks are performed to ensure that the output of each model format is numerically close to the original PyTorch model's output.

This entire process is configured via a single YAML file, ensuring that the conversion is reproducible.
