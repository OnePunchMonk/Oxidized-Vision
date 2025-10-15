# Quickstart

This guide will walk you through converting a model and running it with OxidizedVision.

## 1. Setup

Create a virtualenv and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e python_client/.[dev]
```

## 2. Run conversion

Run the conversion using the example config:

```bash
python -m oxidizedvision.cli convert --config examples/example_unet/config.yml
```

## 3. Package for tract runtime

Package the ONNX model for the `tract` runtime:

```bash
python -m oxidizedvision.cli package --onnx out/model.onnx --runner tract --out rust_runtime/packaged/unet_tract
```

## 4. Build the Rust example server

Build the Rust server:

```bash
cd rust_runtime/packaged/unet_tract
cargo build --release
./target/release/unet_server --model model.onnx --port 8080
```
