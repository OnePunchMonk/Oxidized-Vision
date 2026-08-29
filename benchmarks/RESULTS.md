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

## CPU: `kernel_vision` fused vision kernels (ViT-scale synthetic tokens)

Source: `cargo bench -p kernel_vision`
(`rust_runtime/crates/kernel_vision/benches/kernels.rs`), single run on an
Apple M5 Pro (arm64), criterion default sampling (100 samples/case). These
are CPU-only micro-benchmarks of the kernels in isolation (synthetic
tokens, not a full model forward pass) — see
[`docs/architecture.md`](../docs/architecture.md#6-vision-specific-fused-kernels-kernel_vision)
for what the kernels do and their accuracy/compute tradeoffs.

| Kernel | Grid | Naive baseline | Fused/windowed | Speedup |
|---|---|---|---|---|
| Self-attention (d=96, window=7) | 14x14 (196 tok) | 448 µs (full O(n²)) | 146 µs | ~3.1x |
| Self-attention (d=96, window=7) | 28x28 (784 tok) | 6.83 ms (full O(n²)) | 287 µs | ~23.8x |
| Token merge, r=n/4 (d=384) | 196 tokens | — | 226 µs | n/a (no baseline; see caveat below) |
| Token merge, r=n/4 (d=384) | 784 tokens | — | 975 µs | n/a |

The attention speedup grows with grid size because windowing changes the
complexity class (`O(n²)` -> `O(n · window²)`), not just the constant
factor — expected, and consistent with why Swin-style windowed attention
scales to higher resolutions where full ViT attention doesn't. Token
merging has no "naive baseline" row because it doesn't accelerate a single
op — its win is downstream: every block *after* a merge point sees `n - r`
tokens instead of `n`, and that compounding effect across the rest of the
network isn't visible in a micro-benchmark of the merge op itself. Measure
it end-to-end (full model latency with vs. without `TokenMerge` layers
inserted) before citing a number for it; this table is not that number.

## GPU: `kernel_vision`-style vision kernels, Tesla T4 (Modal)

Source: `modal run benchmarks/modal_kernel_vision_gpu_benchmark.py`, raw
output in
[`results/kernel_vision_gpu_leaderboard.json`](results/kernel_vision_gpu_leaderboard.json).
Same shapes as the CPU table above (ViT-B-ish 768-dim tokens, 12 heads,
batch 8, window 7). Uses PyTorch's fused
`scaled_dot_product_attention` (dispatches to a flash-attention/
memory-efficient CUDA kernel on GPU) as the execution backend for both the
"full" and "windowed" cases — the windowing/merging logic is this repo's,
the underlying attention math kernel is PyTorch's, not a hand-written CUDA
kernel (see the "what this doesn't cover" note in
[`docs/architecture.md`](../docs/architecture.md#6-vision-specific-fused-kernels-kernel_vision)).

| Case | Grid/shape | Full attention | Windowed (w=7) | Speedup |
|---|---|---|---|---|
| Self-attention | 14x14 (196 tok) | 0.986 ms | 0.181 ms | ~5.5x |
| Self-attention | 28x28 (784 tok) | 5.701 ms | 0.603 ms | ~9.5x |

| `TokenMerge` (r=49, n=196) | Time |
|---|---|
| Merge op itself | 0.761 ms |
| Downstream attention, before merge (196 tok) | 0.498 ms |
| Downstream attention, after merge (147 tok) | 0.311 ms (~1.6x) |

`TokenMerge`'s own op cost (0.76ms) is *larger* than the attention time it
saves per block (0.19ms) at this single-block scale on a T4 — its win only
pays off once it feeds *multiple* downstream blocks (attention +
MLP) at the reduced token count, and once the model is deep enough that the
one-time merge cost is amortized. Don't cite this as a net win in isolation;
measure end-to-end model latency with `TokenMerge` layers actually inserted
before claiming a speedup from it.

## Reproducing

```bash
# CPU, local
oxidizedvision convert examples/example_unet/config.yml
oxidizedvision benchmark out/unet.pt --runners torchscript,tract,ort --iters 100

# GPU, on Modal (requires `modal token new` once)
cd benchmarks
uv run modal run modal_gpu_benchmark.py
uv run modal run modal_kernel_vision_gpu_benchmark.py

# kernel_vision CPU micro-benchmarks (Rust)
cd rust_runtime
cargo bench -p kernel_vision

# Verify runner_tensorrt against a real TensorRT SDK + GPU (no stock
# CI/dev machine has trtexec installed, so this is the only way to
# actually exercise that backend end-to-end)
uv run modal run modal_tensorrt_check.py
```
