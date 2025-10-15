# OxidizedVision

**Compile your vision models to Rust for ultra-fast inference.**

<p align="center">
  <a href="https://github.com/your-username/oxidizedvision/actions/workflows/ci.yml"><img src="https://github.com/your-username/oxidizedvision/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/oxidizedvision/"><img src="https://img.shields.io/pypi/v/oxidizedvision" alt="PyPI"></a>
  <a href="https://crates.io/crates/oxidizedvision_run"><img src="https://img.shields.io/crates/v/oxidizedvision_run" alt="Crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

OxidizedVision is an end-to-end toolkit for converting PyTorch vision models into lightweight, production-ready Rust artifacts. It's designed for performance, portability, and ease of use, enabling you to deploy your models on servers, edge devices, and even in the browser with WebAssembly.

## Why OxidizedVision?

-   **Peak Performance**: Leverage Rust's bare-metal performance to get the lowest possible latency and a minimal memory footprint for your models.
-   **Rock-Solid Reliability**: Eliminate a whole class of Python runtime errors. Rust's compile-time checks and strong type system mean your inference service is robust and production-ready.
-   **Simplified Deployment**: Deploy a single, self-contained binary without needing to manage Python environments, `pip` dependencies, or bulky libraries on your server.
-   **Run Anywhere**: With backends for both LibTorch (`tch-rs`) and a pure-Rust ONNX runtime (`tract`), you can deploy to powerful servers with GPUs or to resource-constrained edge devices.
-   **WebAssembly Support**: Compile your models to WASM using the `tract` backend and run them directly in the browser for interactive web applications.

## Key Features

-   **End-to-End Conversion**: A seamless pipeline to convert PyTorch models to TorchScript, then to ONNX, and finally into a Rust-based runtime.
-   **Choice of Runtimes**:
    -   `tch-rs` (LibTorch backend) for full compatibility and GPU support.
    -   `tract` (ONNX backend) for lightweight, portable, and WASM-friendly deployments.
-   **Simple CLI**: An intuitive command-line interface to manage the entire conversion and packaging process.
-   **Ready-to-use Examples**: Get started quickly with pre-configured examples for common vision tasks.

## Getting Started

### Prerequisites

-   Python 3.7+
-   Rust (install via [rustup](https://rustup.rs/))
-   A C++ compiler (for the `tch` crate)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/onepunchmonk/oxidizedvision.git
    cd oxidizedvision
    ```

2.  **Install the Python client:**
    Create a virtual environment and install the necessary packages.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -e ./python_client
    ```

### Quickstart: Convert and Run a Model

Let's convert a pre-trained U-Net model and run it in a Rust-based web server.

1.  **Convert the model:**
    This command uses the configuration in `examples/example_unet/config.yml` to convert a PyTorch model to TorchScript and then to ONNX.
    ```bash
    oxidizedvision convert --config examples/example_unet/config.yml
    ```
    The output models (`model.pt` and `model.onnx`) will be saved in the `out/` directory.

2.  **Package the model into a Rust crate:**
    This command takes the ONNX model and generates a new Rust crate in `rust_runtime/packaged/unet_tract` that can run the model using the `tract` runtime.
    ```bash
    oxidizedvision package --onnx out/model.onnx --runner tract --out rust_runtime/packaged/unet_tract
    ```

3.  **Build and run the Rust server:**
    ```bash
    cd rust_runtime/packaged/unet_tract
    cargo build --release
    ./target/release/image_server --model model.onnx --port 8080
    ```
    The server is now running! You can send it an image for inference.

## Project Structure

The repository is a monorepo containing the Python CLI and the Rust runtimes.

```
OxidizedVision/
├─ python_client/  # Python package for the CLI and conversion tools
├─ rust_runtime/   # Rust workspace for the inference runtimes and examples
├─ examples/       # Example models, notebooks, and configurations
├─ tools/          # Helper scripts for the conversion pipeline
└─ docs/           # Documentation and architecture diagrams
```

For more details on the architecture, see `docs/architecture.md`.

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
