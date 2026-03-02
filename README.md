# 🚀 OxidizedVision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Compile your PyTorch models to Rust for ultra-fast, memory-safe inference.**

OxidizedVision is a production-grade toolkit that bridges the gap between Python-based model training and Rust-based deployment. It provides a seamless pipeline to **convert**, **optimize**, **validate**, **benchmark**, **profile**, and **package** your models — from a trained PyTorch `nn.Module` to a deployable Rust binary, REST API, or WebAssembly module.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Model Conversion** | PyTorch → TorchScript → ONNX with a single command |
| ⚡ **Optimization** | ONNX graph simplification, constant folding, INT8/FP16 quantization |
| ✅ **Validation** | Numerical consistency checks (MAE, RMSE, Cosine Similarity) across formats |
| 📊 **Benchmarking** | Latency (avg, p50, p95, p99), throughput, and memory profiling |
| 🔬 **Profiling** | Parameter count, model size, per-layer breakdown |
| 📦 **Packaging** | Auto-generate a deployable Rust crate (server or CLI) |
| 🌐 **Multi-Backend** | `tract` (pure Rust), `tch` (LibTorch), `tensorrt` (NVIDIA GPU) |
| 🧩 **WASM Support** | Run models in the browser via WebAssembly |
| 📋 **Model Registry** | Track all converted models and their metadata locally |
| 🎨 **Rich CLI** | Beautiful terminal output with progress indicators and tables |
| 🔀 **Multi-Model Server** | Serve multiple models from a single Rust server instance |
| ⏱️ **Dynamic Batching** | Configurable request batching for efficient inference |
| 📝 **Structured Logging** | `tracing` (Rust) + Rich/JSON (Python) for full observability |
| 📈 **Metrics Endpoint** | `/metrics` for monitoring request counts and server health |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python Client (CLI)                   │
│  convert │ validate │ benchmark │ optimize │ profile    │
│  package │ serve    │ list      │ info                   │
│                                                         │
│  Global: --verbose  │  --json-log                       │
└────────────────────────┬────────────────────────────────┘
                         │ Generates
┌────────────────────────▼────────────────────────────────┐
│                    Rust Runtimes                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ runner_tch   │  │ runner_tract │  │ runner_tensorrt │ │
│  │ (LibTorch)   │  │ (Pure Rust)  │  │ (GPU / TensorRT)│ │
│  └──────┬──────┘  └──────┬──────┘  └───────┬─────────┘ │
│         │   All implement Runner trait      │           │
│  ┌──────▼──────────────────▼────────────────▼─────────┐ │
│  │              runner_core (Shared Trait)             │ │
│  │          + tracing structured logging              │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │ Deploys to
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Native Binary    REST API Server    WASM Module
                    (multi-model,
                     batching,
                     /metrics)
```

---

## ⚡ Quickstart

### 1. Install

```bash
# From PyPI
pip install oxidizedvision

# From source (development)
pip install -e "./python_client[dev]"
```

### 2. Create a Config

```yaml
# config.yml
model:
  path: examples/example_unet/model.py
  class_name: UNet
  input_shape: [1, 3, 256, 256]

export:
  output_dir: out
  model_name: unet

validate:
  tolerance_mae: 1e-4
  tolerance_cos_sim: 0.999

benchmark:
  iters: 100
  device: cpu
```

### 3. Run the Pipeline

```bash
# Convert PyTorch → TorchScript + ONNX
oxidizedvision convert config.yml

# Validate numerical consistency
oxidizedvision validate config.yml

# Optimize the ONNX model
oxidizedvision optimize out/unet.onnx --quantize int8

# Benchmark performance
oxidizedvision benchmark out/unet.pt --runners torchscript,tract

# Profile the model
oxidizedvision profile config.yml

# Package into a Rust crate
oxidizedvision package out/unet.onnx --runner tract --template server

# List registered models
oxidizedvision list
```

### 4. Debug with Structured Logging

```bash
# Verbose mode (DEBUG level)
oxidizedvision --verbose convert config.yml

# JSON log output (for CI / log aggregation)
oxidizedvision --json-log convert config.yml
```

---

## 📖 CLI Reference

| Command | Description | Example |
|---|---|---|
| `convert` | Convert PyTorch → TorchScript + ONNX | `oxidizedvision convert config.yml` |
| `validate` | Check numerical consistency | `oxidizedvision validate config.yml --num-tests 5` |
| `benchmark` | Measure inference performance | `oxidizedvision benchmark out/model.pt --runners torchscript,tract` |
| `optimize` | Optimize an ONNX model | `oxidizedvision optimize out/model.onnx --quantize fp16` |
| `profile` | Analyze model parameters and layers | `oxidizedvision profile config.yml` |
| `package` | Generate deployable Rust crate | `oxidizedvision package out/model.onnx --template server` |
| `serve` | Start inference server | `oxidizedvision serve ./binary --port 8080` |
| `list` | List registered models | `oxidizedvision list` |
| `info` | Detailed model information | `oxidizedvision info unet` |

### Global Options

| Flag | Description |
|---|---|
| `--verbose` / `-v` | Enable DEBUG-level logging |
| `--json-log` | Emit logs as JSON lines (for CI / production) |

---

## 🦀 Rust Runtimes

### Shared Runner Trait

All backends implement a common `Runner` trait:

```rust
pub trait Runner: Send + Sync {
    fn from_config(config: &RunnerConfig) -> Result<Self> where Self: Sized;
    fn run(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>>;
    fn info(&self) -> ModelInfo;
}
```

### Available Backends

| Backend | Model Format | GPU | WASM | Dependencies |
|---|---|---|---|---|
| `runner_tract` | ONNX | ❌ | ✅ | None (pure Rust) |
| `runner_tch` | TorchScript | ✅ | ❌ | LibTorch |
| `runner_tensorrt` | ONNX → Engine | ✅ | ❌ | TensorRT SDK |

---

## 🖥️ Inference Server

The built-in `image_server` example provides a production-ready REST API:

```bash
# Single model
cargo run -p image_server -- --model model.onnx --port 8080

# Multi-model (serve multiple models simultaneously)
cargo run -p image_server -- \
  --model segmenter=models/seg.onnx \
  --model classifier=models/cls.onnx \
  --port 8080

# With dynamic batching
cargo run -p image_server -- \
  --model model.onnx \
  --max-batch-size 8 \
  --max-wait-ms 50

# JSON structured logs
cargo run -p image_server -- --model model.onnx --log-format json
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Inference on the default model |
| `POST` | `/predict/{model_name}` | Inference on a named model |
| `GET` | `/health` | Health check with per-model status |
| `GET` | `/metrics` | Request counts, error counts, batch status |
| `GET` | `/models` | List all loaded models |

---

## 🗂️ Project Structure

```
Oxidized-Vision/
├── python_client/             # Python CLI & pipeline
│   ├── oxidizedvision/
│   │   ├── cli.py             # Typer CLI entry point
│   │   ├── config.py          # Pydantic config models
│   │   ├── convert.py         # Model conversion
│   │   ├── validate.py        # Numerical validation
│   │   ├── benchmark.py       # Performance measurement
│   │   ├── optimize.py        # ONNX optimization
│   │   ├── profile.py         # Model profiling
│   │   ├── registry.py        # Model registry
│   │   └── logging.py         # Structured logging (Rich / JSON)
│   └── tests/                 # pytest test suite
├── rust_runtime/              # Rust inference runtimes
│   ├── crates/
│   │   ├── runner_core/       # Shared Runner trait + tracing
│   │   ├── runner_tch/        # LibTorch backend
│   │   ├── runner_tract/      # tract (ONNX) backend
│   │   └── runner_tensorrt/   # TensorRT backend
│   └── examples/
│       ├── image_server/      # Multi-model REST API with batching
│       ├── denoiser_cli/      # Image denoising CLI
│       └── wasm_frontend/     # Browser inference demo
├── tools/                     # Standalone scripts
├── benchmarks/                # Benchmark infrastructure
├── examples/                  # User-facing examples
│   └── example_unet/         # Complete UNet example
├── docs/                      # Architecture docs
└── .github/workflows/         # CI/CD + PyPI auto-deploy
```

---

## 🧪 Testing

```bash
# Python tests
pytest python_client/tests/ -v --cov=oxidizedvision

# Rust tests
cargo test --workspace
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## 📦 Publishing to PyPI

Releases are automatically published to PyPI when a GitHub Release is created with a `v*` tag (e.g., `v1.0.2`). See [`.github/workflows/publish.yml`](.github/workflows/publish.yml) for details.

To publish manually:

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing instructions, and PR guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
