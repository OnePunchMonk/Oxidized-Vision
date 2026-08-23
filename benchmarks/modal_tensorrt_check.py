"""
OxidizedVision — verify runner_tensorrt against a real GPU + TensorRT SDK.

runner_tensorrt shells out to `trtexec`, which isn't available on a stock
CI/dev machine — this app builds it in a Modal container with the NVIDIA
TensorRT SDK installed (via NVIDIA's public CUDA apt repo, no NGC login
required) on a real GPU, and runs an actual model through it end-to-end
(export -> build engine -> run inference -> check output shape).

This caught two real trtexec CLI incompatibilities that runner_tensorrt
had baked in from an older TensorRT release: the bare `--fp16` build flag
and `--saveOutput` for raw-binary inference output were both removed in
TensorRT 10+ (strongly-typed networks, JSON-based `--exportOutput`
instead) — fixed in runner_tensorrt with version-tolerant fallbacks.

Usage:
    modal run benchmarks/modal_tensorrt_check.py
"""

import modal

app = modal.App("oxidizedvision-tensorrt-check")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "gnupg", "curl", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y tensorrt",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .pip_install("torch", "onnx", "onnxscript")
    .add_local_dir("../rust_runtime/crates/runner_tensorrt", remote_path="/root/runner_tensorrt")
    .add_local_dir("../rust_runtime/crates/runner_core", remote_path="/root/runner_core")
)


@app.function(image=image, gpu="T4", timeout=900)
def check_tensorrt_runner():
    import os
    import shutil
    import subprocess

    os.environ["PATH"] = "/root/.cargo/bin:" + os.environ["PATH"]

    # Export a small real model to ONNX to exercise the runner against.
    # Dynamic batch axis matches this repo's own convert.py, so the
    # exported model actually accepts --optShapes/--shapes the way
    # runner_tensorrt's trtexec invocations assume.
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    model = Tiny().eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        "/tmp/tiny.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Exported /tmp/tiny.onnx")

    # Build a tiny binary that loads the model and runs inference through
    # TensorRTRunner, in a throwaway workspace alongside the real crates.
    shutil.copytree("/root/runner_tensorrt", "/tmp/runner_tensorrt")
    shutil.copytree("/root/runner_core", "/tmp/runner_core")

    with open("/tmp/Cargo.toml", "w") as f:
        f.write(
            '[workspace]\nmembers = ["runner_core", "runner_tensorrt", "trt_check"]\n'
            'resolver = "2"\n'
        )

    os.makedirs("/tmp/trt_check/src", exist_ok=True)
    with open("/tmp/trt_check/Cargo.toml", "w") as f:
        f.write(
            """[package]
name = "trt_check"
version = "0.1.0"
edition = "2021"

[dependencies]
runner_core = { path = "../runner_core" }
runner_tensorrt = { path = "../runner_tensorrt" }
ndarray = "0.15"
anyhow = "1.0"
"""
        )
    with open("/tmp/trt_check/src/main.rs", "w") as f:
        f.write(
            """use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use runner_core::{Runner, RunnerConfig};
use runner_tensorrt::TensorRTRunner;

fn main() -> Result<()> {
    let config = RunnerConfig {
        model_path: "/tmp/tiny.onnx".to_string(),
        input_shape: vec![1, 3, 64, 64],
        use_cuda: true,
        optimize: true,
    };
    let runner = TensorRTRunner::from_config(&config)?;
    let input = ArrayD::<f32>::zeros(IxDyn(&[1, 3, 64, 64]));
    let output = runner.run(&input)?;
    println!("SUCCESS: output shape = {:?}", output.shape());
    Ok(())
}
"""
        )

    build = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", "/tmp/trt_check/Cargo.toml"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("BUILD returncode:", build.returncode)
    print("BUILD tail:", (build.stdout + build.stderr)[-500:])
    if build.returncode != 0:
        return {"build_ok": False}

    run = subprocess.run(
        ["/tmp/target/release/trt_check"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Only print the tail — the verbose trtexec dump can exceed Modal's
    # per-log-line size; the compiled binary's own "SUCCESS: ..." println
    # is the last line of its stdout, which is what we actually need.
    print("RUN returncode:", run.returncode)
    print("RUN stdout tail:", run.stdout[-500:])
    print("RUN stderr tail:", run.stderr[-500:])

    return {"build_ok": True, "run_ok": run.returncode == 0}


@app.local_entrypoint()
def main():
    result = check_tensorrt_runner.remote()
    print(result)
