"""
OxidizedVision — Production-grade, Rust-native inference toolkit for PyTorch models.

A complete pipeline to convert, optimize, validate, benchmark, and serve
machine learning models with the speed and safety of Rust.
"""

__version__ = "1.0.1"
__author__ = "Avaya Aggarwal"
__email__ = "aggarwal.avaya27@gmail.com"

from .config import Config, load_config, save_config, ModelConfig, ExportConfig
from .convert import convert_model, load_model, convert_to_torchscript, convert_to_onnx
from .validate import validate_models, calculate_mae, calculate_cosine_similarity
from .benchmark import run_benchmarks, measure_performance
from .optimize import optimize_model, simplify_onnx, quantize_onnx
from .profile import profile_model, print_profile
from .registry import register_model, list_models, get_model_info

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
]
