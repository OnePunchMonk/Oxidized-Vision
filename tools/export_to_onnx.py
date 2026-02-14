#!/usr/bin/env python3
"""
OxidizedVision — Standalone TorchScript → ONNX exporter.

This is a lightweight tool for converting a TorchScript model to ONNX.
For the full pipeline (PyTorch → TorchScript + ONNX), use:
    oxidizedvision convert config.yml

Usage:
    python tools/export_to_onnx.py --input out/model.pt --output out/model.onnx --input-shape "1,3,256,256"
    python tools/export_to_onnx.py --input out/model.pt --output out/model.onnx --input-shape "1,3,256,256" --simplify --opset 14
"""

import torch
import onnx
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Export a TorchScript model to ONNX format."
    )
    parser.add_argument("--input", required=True, help="Input TorchScript model path (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX model path (.onnx)")
    parser.add_argument(
        "--input-shape", required=True, help='Input shape, e.g., "1,3,256,256"'
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    parser.add_argument(
        "--simplify", action="store_true", help="Apply onnx-simplifier after export"
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="Enable dynamic batch dimension"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input model not found: {args.input}")
        return 1

    shape = tuple(map(int, args.input_shape.split(",")))

    print(f"Loading TorchScript model from: {args.input}")
    model = torch.jit.load(args.input)

    dummy = torch.randn(shape)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if args.simplify:
        try:
            from onnxsim import simplify

            print("Simplifying ONNX model...")
            model_onnx = onnx.load(args.output)
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, args.output)
                print("Simplification successful.")
            else:
                print("Warning: Simplified model validation failed. Using original.")
        except ImportError:
            print("Warning: onnx-simplifier not installed. Skipping simplification.")

    file_size = os.path.getsize(args.output) / 1024
    print(f"Model saved to {args.output} ({file_size:.1f} KB)")


if __name__ == "__main__":
    main()
