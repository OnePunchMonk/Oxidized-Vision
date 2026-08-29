"""
OxidizedVision — GPU benchmark for the vision-specific kernels
(`oxidizedvision.kernels.TokenMerge`, and windowed vs. full attention).

Runs on a real GPU via Modal so the numbers reflect actual hardware, not a
CPU laptop run — see `benchmarks/RESULTS.md` for the CPU-only
`cargo bench -p kernel_vision` numbers that motivated writing this.

Honest scope note: this measures the *windowing/merging technique* using
PyTorch's built-in fused attention kernel
(`torch.nn.functional.scaled_dot_product_attention`, which itself dispatches
to a flash-attention/efficient-attention CUDA kernel on GPU) as the
execution backend — it does not benchmark a hand-written CUDA kernel. See
`docs/architecture.md` §6 for why a from-scratch CUDA kernel isn't in this
PR's scope.

Usage:
    modal run benchmarks/modal_kernel_vision_gpu_benchmark.py
"""

import json

import modal

app = modal.App("oxidizedvision-kernel-vision-gpu-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "onnx-simplifier",
        "psutil",
        "rich",
        "pyyaml",
        "pydantic",
        "typer",
    )
    .add_local_dir(
        "../python_client/oxidizedvision",
        remote_path="/root/oxidizedvision",
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_kernel_benchmarks() -> dict:
    import time

    import torch
    import torch.nn.functional as F

    device = "cuda"
    torch.manual_seed(0)

    def timed(fn, iters=200, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e3  # ms/iter

    results = {"device": torch.cuda.get_device_name(0)}

    # --- 1. Full vs. windowed self-attention, ViT-Base-ish scale ---------
    # 14x14=196 tokens (ViT-B/16 @ 224px), d=768, 12 heads, batch=8, window=7
    # (Swin-T's window size), matching the CPU kernel_vision bench's shape.
    for (h, w), label in [((14, 14), "14x14_196tok"), ((28, 28), "28x28_784tok")]:
        n, d, heads, batch, window = h * w, 768, 12, 8, 7
        x = torch.randn(batch, n, d, device=device)
        q = x.view(batch, n, heads, d // heads).transpose(1, 2)  # [b, heads, n, dh]

        def full_attn(q=q):
            return F.scaled_dot_product_attention(q, q, q)

        # Windowed: reshape the h*w grid into (h/window * w/window) windows,
        # treat windows as extra batch dim, run SDPA restricted to each
        # window's window*window tokens only.
        wins_y, wins_x = h // window, w // window
        q_win = (
            q.view(batch, heads, wins_y, window, wins_x, window, d // heads)
            .permute(0, 2, 4, 1, 3, 5, 6)
            .reshape(batch * wins_y * wins_x, heads, window * window, d // heads)
        )

        def windowed_attn(q_win=q_win):
            return F.scaled_dot_product_attention(q_win, q_win, q_win)

        results[f"attention_{label}_full_ms"] = timed(full_attn)
        results[f"attention_{label}_windowed_w7_ms"] = timed(windowed_attn)

    # --- 2. TokenMerge op cost + downstream attention savings ------------
    import sys

    sys.path.insert(0, "/root")
    from oxidizedvision.kernels import token_merge

    n, d, heads, batch, r = 196, 768, 12, 8, 49  # merge away 25% of tokens
    x = torch.randn(batch, n, d, device=device)

    def merge_op(x=x):
        return token_merge(x, r)

    results["token_merge_op_ms"] = timed(merge_op)

    merged = token_merge(x, r)
    q_before = x.view(batch, n, heads, d // heads).transpose(1, 2)
    q_after = merged.view(batch, n - r, heads, d // heads).transpose(1, 2)

    results["attention_before_merge_ms"] = timed(
        lambda: F.scaled_dot_product_attention(q_before, q_before, q_before)
    )
    results["attention_after_merge_ms"] = timed(
        lambda: F.scaled_dot_product_attention(q_after, q_after, q_after)
    )

    return results


@app.local_entrypoint()
def main():
    results = run_kernel_benchmarks.remote()
    print(json.dumps(results, indent=2))
    with open("results/kernel_vision_gpu_leaderboard.json", "w") as f:
        json.dump(results, f, indent=2)
