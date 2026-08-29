"""
OxidizedVision — Production-grade, Rust-native inference toolkit for PyTorch models.

A complete pipeline to convert, optimize, validate, benchmark, and serve
machine learning models with the speed and safety of Rust.
"""

__version__ = "1.0.1"
__author__ = "Avaya Aggarwal"
__email__ = "aggarwal.avaya27@gmail.com"

from .benchmark import measure_performance, run_benchmarks
from .config import Config, ExportConfig, ModelConfig, load_config, save_config
from .convert import convert_model, convert_to_onnx, convert_to_torchscript, load_model
from .kernels import TokenMerge, token_merge
from .logging import configure_logging, get_logger
from .optimize import optimize_model, quantize_onnx, simplify_onnx
from .profile import print_profile, profile_model
from .registry import get_model_info, list_models, register_model
from .validate import calculate_cosine_similarity, calculate_mae, validate_models

__all__ = [
    # Config
    "Config",
    "load_config",
    "save_config",
    "ModelConfig",
    "ExportConfig",
    # Convert
    "convert_model",
    "load_model",
    "convert_to_torchscript",
    "convert_to_onnx",
    # Kernels
    "TokenMerge",
    "token_merge",
    # Validate
    "validate_models",
    "calculate_mae",
    "calculate_cosine_similarity",
    # Benchmark
    "run_benchmarks",
    "measure_performance",
    # Optimize
    "optimize_model",
    "simplify_onnx",
    "quantize_onnx",
    # Profile
    "profile_model",
    "print_profile",
    # Registry
    "register_model",
    "list_models",
    "get_model_info",
    # Logging
    "configure_logging",
    "get_logger",
]
