"""
OxidizedVision — Benchmarking module.

Measures latency, throughput, and memory usage across different model runners.
"""

import time
import numpy as np
import torch
import onnxruntime as ort
import psutil
import os
from typing import List, Dict, Any, Optional
from rich.console import Console

from .convert import _import_model_from_path

console = Console()


def _warmup(run_fn, iters: int = 10):
    """Run warmup iterations."""
    for _ in range(iters):
        run_fn()


def _benchmark_pytorch(
    model_source_path: str,
    model_class_name: str,
    input_shape: List[int],
    checkpoint: Optional[str],
    iters: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Benchmark a native PyTorch model."""
    model_class = _import_model_from_path(model_source_path, model_class_name)
    model = model_class()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    torch_device = torch.device(device)
    model = model.to(torch_device)
    dummy_input = torch.randn(*input_shape, device=torch_device)

    def run_fn():
        with torch.no_grad():
            model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()

    return _measure(run_fn, iters)


def _benchmark_torchscript(
    model_path: str,
    input_shape: List[int],
    iters: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Benchmark a TorchScript model."""
    torch_device = torch.device(device)
    model = torch.jit.load(model_path, map_location=torch_device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=torch_device)

    def run_fn():
        with torch.no_grad():
            model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize()

    return _measure(run_fn, iters)


def _benchmark_onnx(
    model_path: str,
    input_shape: List[int],
    iters: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Benchmark an ONNX model via onnxruntime."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    def run_fn():
        session.run(None, {input_name: dummy_input})

    return _measure(run_fn, iters)


def _measure(run_fn, iters: int, warmup_iters: int = 10) -> Dict[str, Any]:
    """Core measurement: warmup, then time `iters` runs and measure memory."""
    process = psutil.Process(os.getpid())

    _warmup(run_fn, warmup_iters)

    # Collect per-iteration latencies for percentile stats
    latencies = []
    mem_before = process.memory_info().rss

    for _ in range(iters):
        start = time.perf_counter()
        run_fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    mem_after = process.memory_info().rss

    total_time_s = sum(latencies) / 1000.0
    avg_latency_ms = np.mean(latencies)
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)
    std_ms = np.std(latencies)
    throughput = iters / total_time_s if total_time_s > 0 else 0
    memory_delta_mb = (mem_after - mem_before) / (1024 * 1024)

    return {
        "avg_latency_ms": round(avg_latency_ms, 3),
        "p50_latency_ms": round(p50_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "p99_latency_ms": round(p99_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "throughput_per_sec": round(throughput, 2),
        "memory_delta_mb": round(memory_delta_mb, 2),
        "total_time_s": round(total_time_s, 3),
        "iters": iters,
    }


def measure_performance(
    model_path: str,
    runner: str,
    iters: int,
    batch_size: int,
    input_shape: Optional[List[int]] = None,
    device: str = "cpu",
    model_source_path: Optional[str] = None,
    model_class_name: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure latency, throughput, and memory for a given model and runner.
    
    Args:
        model_path: Path to the model file (.pt or .onnx).
        runner: Runner name ('pytorch', 'torchscript', 'tract', 'onnx').
        iters: Number of benchmark iterations.
        batch_size: Batch size for the input tensor.
        input_shape: Full input shape. If None, uses [batch_size, 3, 256, 256].
        device: Device to run on ('cpu' or 'cuda').
        model_source_path: For 'pytorch' runner — path to the .py file.
        model_class_name: For 'pytorch' runner — class name.
        model_checkpoint: For 'pytorch' runner — optional checkpoint path.
        
    Returns:
        Dict with benchmark results.
    """
    if input_shape is None:
        input_shape = [batch_size, 3, 256, 256]
    else:
        input_shape = list(input_shape)
        input_shape[0] = batch_size

    if runner == "pytorch":
        if not model_source_path or not model_class_name:
            raise ValueError(
                "PyTorch runner requires 'model_source_path' and 'model_class_name'. "
                "Pass these via --model-source and --model-class CLI flags."
            )
        metrics = _benchmark_pytorch(
            model_source_path, model_class_name, input_shape, model_checkpoint, iters, device
        )
    elif runner == "torchscript":
        metrics = _benchmark_torchscript(model_path, input_shape, iters, device)
    elif runner in ("tract", "onnx"):
        metrics = _benchmark_onnx(model_path, input_shape, iters, device)
    else:
        raise ValueError(f"Unknown runner: {runner}")

    return {
        "runner": runner,
        "model_path": model_path,
        "device": device,
        "batch_size": batch_size,
        "input_shape": input_shape,
        **metrics,
    }


def run_benchmarks(
    model_path: str,
    runners: List[str],
    iters: int,
    batch_size: int,
    input_shape: Optional[List[int]] = None,
    device: str = "cpu",
    model_source_path: Optional[str] = None,
    model_class_name: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run benchmarks for a list of runners.
    
    Args:
        model_path: Path to the model file (.pt or .onnx).
        runners: List of runner names.
        iters: Number of benchmark iterations.
        batch_size: Batch size.
        input_shape: Full input shape.
        device: Device to run on.
        model_source_path: For 'pytorch' runner.
        model_class_name: For 'pytorch' runner.
        model_checkpoint: For 'pytorch' runner.
        
    Returns:
        List of benchmark result dicts.
    """
    results = []
    for runner in runners:
        console.print(f"\n📊 Benchmarking [bold cyan]{runner}[/bold cyan]...")

        # Adjust model path for ONNX-based runners
        current_model_path = model_path
        if runner in ("tract", "onnx"):
            current_model_path = model_path.replace(".pt", ".onnx")
            if not os.path.exists(current_model_path):
                console.print(f"  [yellow]Warning: ONNX model not found at {current_model_path}. Skipping.[/yellow]")
                continue

        try:
            result = measure_performance(
                current_model_path,
                runner,
                iters,
                batch_size,
                input_shape=input_shape,
                device=device,
                model_source_path=model_source_path,
                model_class_name=model_class_name,
                model_checkpoint=model_checkpoint,
            )
            results.append(result)
            console.print(f"  ✅ Done — avg: {result['avg_latency_ms']}ms, throughput: {result['throughput_per_sec']}/s")
        except Exception as e:
            console.print(f"  [red]Error benchmarking {runner}: {e}[/red]")

    return results
