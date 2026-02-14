#!/usr/bin/env python3
"""
OxidizedVision — Standalone benchmark tool.

Usage:
    python benchmarks/run_benchmark.py --model out/model.pt --runners torchscript,tract
"""

import argparse
import sys
import os
import json

# Add the python_client directory to the path so we can import oxidizedvision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_client"))

from oxidizedvision.benchmark import run_benchmarks
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model inference performance."
    )
    parser.add_argument("--model", required=True, help="Path to the model file.")
    parser.add_argument(
        "--runners",
        required=True,
        help="Comma-separated list of runners (e.g., torchscript,tract,pytorch).",
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--input-shape",
        default=None,
        help="Input shape as comma-separated dims (e.g., '1,3,256,256').",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device: 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--model-source", default=None, help="Model source .py file (for pytorch runner)."
    )
    parser.add_argument(
        "--model-class", default=None, help="Model class name (for pytorch runner)."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    args = parser.parse_args()

    runners = [r.strip() for r in args.runners.split(",")]
    shape = None
    if args.input_shape:
        shape = [int(d.strip()) for d in args.input_shape.split(",")]

    results = run_benchmarks(
        model_path=args.model,
        runners=runners,
        iters=args.iters,
        batch_size=args.batch_size,
        input_shape=shape,
        device=args.device,
        model_source_path=args.model_source,
        model_class_name=args.model_class,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        table = Table(title="Benchmark Results")
        table.add_column("Runner", style="cyan")
        table.add_column("Avg (ms)", style="magenta")
        table.add_column("p50 (ms)", style="blue")
        table.add_column("p95 (ms)", style="yellow")
        table.add_column("p99 (ms)", style="red")
        table.add_column("Throughput", style="green")
        table.add_column("Mem Δ (MB)", style="dim")

        for r in results:
            table.add_row(
                r["runner"],
                str(r["avg_latency_ms"]),
                str(r["p50_latency_ms"]),
                str(r["p95_latency_ms"]),
                str(r["p99_latency_ms"]),
                f"{r['throughput_per_sec']}/s",
                str(r["memory_delta_mb"]),
            )
        console.print(table)


if __name__ == "__main__":
    main()
