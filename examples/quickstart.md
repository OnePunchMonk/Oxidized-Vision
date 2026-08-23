# Quickstart

This guide will walk you through converting a model and running it with OxidizedVision.

## 1. Setup

Create a virtualenv and install dependencies (from the repo root):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Run conversion

Run the conversion using an example config. Three real, working example
architectures are included, each a different vision modality:

```bash
# 2D segmentation: a UNet with skip connections
python -m oxidizedvision.cli convert examples/example_unet/config.yml

# 2D object detection: a compact YOLO-style single-stage detector
python -m oxidizedvision.cli convert examples/example_detector/config.yml

# 3D point clouds: a PointNet-style classifier (LiDAR/depth-sensor input)
python -m oxidizedvision.cli convert examples/example_pointnet/config.yml
```

Each produces `out/<model_name>.pt` (TorchScript) and `out/<model_name>.onnx`.

> **Note:** `example_pointnet`'s final `Linear` layer exports to an ONNX
> `Gemm` node that the pure-Rust `tract` backend currently fails to load
> (`Failed analyse for node ".../Gemm"`). Use `--runner ort` when packaging
> or serving that model; `runner_ort` (ONNX Runtime) handles it fine, and so
> does the Python-side `onnx`/`ort` benchmark path.

## 3. Validate and benchmark

```bash
python -m oxidizedvision.cli validate examples/example_detector/config.yml
python -m oxidizedvision.cli benchmark out/detector.pt --runners torchscript,tract,ort --input-shape 1,3,320,320
```

## 4. Package for a Rust runtime

Package the ONNX model for the `ort` runtime (or `tract`, `tch`, `tensorrt`):

```bash
python -m oxidizedvision.cli package out/detector.onnx --runner ort --out rust_runtime/packaged/detector_ort
```

## 5. Build and run the packaged Rust crate

```bash
cd rust_runtime/packaged/detector_ort
cargo build --release
./target/release/detector_ort --model model.onnx --port 8080
```

Or run the full multi-model `image_server` example directly instead of a
packaged single-model crate — it supports both raw-tensor (`/predict`) and
raw-image-upload (`/predict/image`, with SIMD decode/resize/normalize)
endpoints:

```bash
cd rust_runtime
cargo run -p image_server -- --model detector=../out/detector.onnx --backend ort --port 8080
```
