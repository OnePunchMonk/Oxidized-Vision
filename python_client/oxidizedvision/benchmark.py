import time
import numpy as np
import torch
import onnxruntime as ort
import psutil
import subprocess
import os
from typing import List, Dict, Any

def measure_performance(model_path: str, runner: str, iters: int, batch_size: int) -> Dict[str, Any]:
    """Measures latency, throughput, and memory for a given model and runner."""
    process = psutil.Process(os.getpid())
    
    if runner == "pytorch":
        # This is a placeholder for loading a standard PyTorch model
        # In a real scenario, you would load it from a .py file or similar
        raise NotImplementedError("PyTorch runner not fully implemented yet. Requires model definition.")

    elif runner == "torchscript":
        model = torch.jit.load(model_path)
        model.eval()
        input_shape = [batch_size, 3, 256, 256]  # Assuming this shape
        dummy_input = torch.randn(*input_shape)
        
        # Warmup
        for _ in range(10):
            model(dummy_input)
            
        # Timed and memory-measured run
        mem_before = process.memory_info().rss
        start_time = time.time()
        for _ in range(iters):
            model(dummy_input)
        end_time = time.time()
        mem_after = process.memory_info().rss
        
    elif runner == "tract":  # Simulating with onnxruntime
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_shape = [batch_size, 3, 256, 256]  # Assuming this shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Timed and memory-measured run
        mem_before = process.memory_info().rss
        start_time = time.time()
        for _ in range(iters):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        mem_after = process.memory_info().rss
        
    else:
        raise ValueError(f"Unknown runner: {runner}")

    total_time = end_time - start_time
    avg_latency_ms = (total_time / iters) * 1000
    throughput = (iters * batch_size) / total_time
    memory_usage_mb = (mem_after - mem_before) / (1024 * 1024)

    return {
        "runner": runner,
        "model_path": model_path,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_images_per_sec": round(throughput, 2),
        "memory_usage_mb": round(memory_usage_mb, 2),
    }

def run_benchmarks(model_path: str, runners: List[str], iters: int, batch_size: int) -> List[Dict[str, Any]]:
    """Run benchmarks for a list of runners."""
    results = []
    for runner in runners:
        print(f"Running benchmark for {runner}...")
        
        # Adjust model path for ONNX-based runners
        current_model_path = model_path
        if runner in ["tract", "onnx"]:
            current_model_path = model_path.replace(".pt", ".onnx")
            if not os.path.exists(current_model_path):
                print(f"Warning: ONNX model not found at {current_model_path}. Skipping.")
                continue

        try:
            result = measure_performance(current_model_path, runner, iters, batch_size)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {runner}: {e}")
            
    return results
