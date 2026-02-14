#!/usr/bin/env python3
"""
OxidizedVision — Standalone conversion tool.

This script wraps the `oxidizedvision convert` command for standalone use
without installing the package.

Usage:
    python tools/convert.py --config examples/example_unet/config.yml
"""

import argparse
import sys
import os

# Add the python_client directory to the path so we can import oxidizedvision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_client"))

from oxidizedvision.config import load_config
from oxidizedvision.convert import convert_model


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch model to TorchScript and ONNX."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ts_path, onnx_path = convert_model(config)

    print(f"\nGenerated files:")
    print(f"  TorchScript: {ts_path}")
    print(f"  ONNX:        {onnx_path}")


if __name__ == "__main__":
    main()
