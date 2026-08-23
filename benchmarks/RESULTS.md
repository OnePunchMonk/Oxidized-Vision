# Benchmark Leaderboard

Real numbers, not vibes. Every table below comes from a script in this
directory that anyone can re-run — nothing here is hand-edited.

## GPU: example UNet, Tesla T4 (Modal)

Source: `python benchmarks/modal_gpu_benchmark.py` (`modal run
benchmarks/modal_gpu_benchmark.py`), raw output in
[`results/gpu_leaderboard.json`](results/gpu_leaderboard.json).

Model: `examples/example_unet` (real encoder/decoder UNet with skip
connections, not a toy), input `[1, 3, 256, 256]`, 100 iterations after a
10-iteration warmup.

| Runner | Device | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput |
|---|---|---|---|---|---|---|
| PyTorch (eager) | Tesla T4 | 15.391 | 15.183 | 18.501 | 18.598 | 64.97/s |
| TorchScript | Tesla T4 | 15.570 | 15.534 | 15.952 | 16.552 | 64.23/s |
| ONNX Runtime (CUDA EP) | Tesla T4 | 14.436 | 14.397 | 14.659 | 15.385 | 69.27/s |

ONNX Runtime's CUDA execution provider is ~6% faster than eager PyTorch and
~7% faster than TorchScript on this model/shape — consistent with the
`runner_ort` backend's premise (graph-level fusion + hardware-tuned kernels
beat naive op-by-op execution), though this particular run used ONNX
Runtime's default optimization level via the Python client, not the Rust
`runner_ort` crate's explicit `Level3` setting. Re-run with `--runners
torchscript,tract,ort` (see below) once `tract` and the Rust `ort` backend
are benchmarked head-to-head on the same hardware.

## CPU: example UNet (local / CI)

Source: `oxidizedvision benchmark out/unet.pt --runners
torchscript,tract,ort` or `.github/workflows/benchmarks.yml` (runs weekly on
`ubuntu-latest`, uploads `benchmark_results.json` as a build artifact).

CPU numbers vary a lot by host, so no fixed table is published here — pull
the latest artifact from the
[Benchmarks workflow](../.github/workflows/benchmarks.yml) runs for
current numbers on GitHub's runners.

## Reproducing

```bash
# CPU, local
oxidizedvision convert examples/example_unet/config.yml
oxidizedvision benchmark out/unet.pt --runners torchscript,tract,ort --iters 100

# GPU, on Modal (requires `modal token new` once)
cd benchmarks
uv run modal run modal_gpu_benchmark.py
```
