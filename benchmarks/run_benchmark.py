import argparse
import time
import numpy as np
import torch
import onnxruntime as ort

def run_benchmark(model_path, runner, iters, batch_size):
    if runner == "torchscript":
        model = torch.jit.load(model_path)
        model.eval()
        input_shape = [batch_size, 3, 256, 256] # Assuming this shape
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        for _ in range(10):
            model(dummy_input)
            
        # Timed run
        start_time = time.time()
        for _ in range(iters):
            model(dummy_input)
        end_time = time.time()
        
    elif runner == "tract": # Simulating with onnxruntime
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_shape = [batch_size, 3, 256, 256] # Assuming this shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Timed run
        start_time = time.time()
        for _ in range(iters):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
    else:
        raise ValueError(f"Unknown runner: {runner}")

    total_time = end_time - start_time
    avg_latency = total_time / iters
    throughput = iters * batch_size / total_time
    
    print(f"Runner: {runner}")
    print(f"  Avg Latency: {avg_latency * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--runners', required=True, help="comma-separated list, e.g., torchscript,tract")
    parser.add_argument('--input', help="dummy input path")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    runners = args.runners.split(',')
    for runner in runners:
        model_file = args.model
        if runner == "tract":
            # In a real scenario, you'd use the Rust binary.
            # Here we use the onnx file with onnxruntime as a proxy.
            model_file = model_file.replace(".pt", ".onnx")
        run_benchmark(model_file, runner, args.iters, args.batch_size)

if __name__ == '__main__':
    main()
