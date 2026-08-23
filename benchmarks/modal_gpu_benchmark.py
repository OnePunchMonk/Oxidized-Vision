"""
OxidizedVision — GPU benchmark suite on Modal.

Runs the example UNet across PyTorch / TorchScript / ONNX Runtime (CPU + CUDA
execution providers) on a real GPU, so the numbers in the README/leaderboard
reflect actual hardware rather than a CPU-only laptop run.

Usage:
    modal run benchmarks/modal_gpu_benchmark.py

Requires a Modal account (`modal token new`) with GPU access on the workspace.
"""

import json

import modal

app = modal.App("oxidizedvision-gpu-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime-gpu",
        "numpy",
        "psutil",
        "rich",
        "pyyaml",
        "pydantic",
        "onnx-simplifier",
        "onnxscript",
    )
    .add_local_dir(
        "../python_client/oxidizedvision",
        remote_path="/root/oxidizedvision",
    )
    .add_local_dir(
        "../examples/example_unet",
        remote_path="/root/example_unet",
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_gpu_benchmarks(iters: int = 100, input_shape: list = None) -> list:
    import sys

    sys.path.insert(0, "/root")

    import torch

    from oxidizedvision.benchmark import run_benchmarks
    from oxidizedvision.convert import convert_model
    from oxidizedvision.config import Config

    input_shape = input_shape or [1, 3, 256, 256]

    cfg = Config(
        **{
            "model": {
                "path": "/root/example_unet/model.py",
                "class_name": "UNet",
                "input_shape": input_shape,
            },
            "export": {"output_dir": "/tmp/out", "model_name": "unet"},
        }
    )
    ts_path, onnx_path = convert_model(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"

    results = run_benchmarks(
        model_path=ts_path,
        runners=["pytorch", "torchscript", "onnx"],
        iters=iters,
        batch_size=1,
        input_shape=input_shape,
        device=device,
        model_source_path="/root/example_unet/model.py",
        model_class_name="UNet",
    )

    for r in results:
        r["hardware"] = gpu_name

    return results


@app.local_entrypoint()
def main(iters: int = 100, out: str = "results/gpu_leaderboard.json"):
    results = run_gpu_benchmarks.remote(iters=iters)
    print(json.dumps(results, indent=2))
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
