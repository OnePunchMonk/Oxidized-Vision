#!/usr/bin/env python3
"""
OxidizedVision — Standalone ONNX optimization tool.

This script wraps the `oxidizedvision optimize` command for standalone use
without installing the package.

Usage:
    python tools/optimize_onnx.py --input models/model.onnx --output models/model_optimized.onnx
    python tools/optimize_onnx.py --input models/model.onnx --quantize int8
"""

import argparse
import sys
import os

# Add the python_client directory to the path so we can import oxidizedvision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_client"))

from oxidizedvision.optimize import optimize_model


def main():
    parser = argparse.ArgumentParser(
        description="Optimize an ONNX model (simplify, quantize, fold constants)."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input ONNX model."
    )
    parser.add_argument(
        "--output", default=None, help="Path for the output optimized model."
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnx-simplifier pass.",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8", "fp16"],
        default=None,
        help="Quantization mode.",
    )
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Skip constant folding pass.",
    )
    args = parser.parse_args()

    result_path = optimize_model(
        input_path=args.input,
        output_path=args.output,
        simplify=not args.no_simplify,
        quantize=args.quantize,
        constant_folding=not args.no_constant_folding,
    )
    print(f"\nOptimized model: {result_path}")


if __name__ == "__main__":
    main()
