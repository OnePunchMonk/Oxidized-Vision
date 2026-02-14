"""
OxidizedVision — Configuration management.

Provides strongly-typed Pydantic models for the YAML configuration schema,
ensuring reproducible and validated conversion pipelines.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


class ModelConfig(BaseModel):
    """Configuration for the source PyTorch model."""
    path: str = Field(..., description="Path to the Python file containing the model class.")
    class_name: str = Field(..., description="Name of the nn.Module class to instantiate.")
    input_shape: List[int] = Field(..., description="Input tensor shape, e.g. [1, 3, 256, 256].")
    checkpoint: Optional[str] = Field(None, description="Optional path to a .pt checkpoint file.")

    @validator("input_shape")
    def validate_input_shape(cls, v):
        if len(v) < 2:
            raise ValueError("input_shape must have at least 2 dimensions")
        if any(d <= 0 for d in v):
            raise ValueError("All dimensions in input_shape must be positive")
        return v


class ExportConfig(BaseModel):
    """Configuration for the ONNX export step."""
    opset_version: int = Field(14, description="ONNX opset version.")
    do_constant_folding: bool = Field(True, description="Whether to apply constant folding.")
    output_dir: str = Field("out", description="Directory to save exported models.")
    model_name: str = Field("model", description="Base filename for exported models.")


class RunnerConfig(BaseModel):
    """Configuration for a single inference runner."""
    name: str = Field(..., description="Runner name: 'tch', 'tract', or 'tensorrt'.")
    optimize: bool = Field(True, description="Whether to apply backend-specific optimizations.")
    use_cuda: Optional[bool] = Field(None, description="Whether to use CUDA (GPU runners only).")


class ValidateConfig(BaseModel):
    """Configuration for the validation step."""
    num_tests: int = Field(1, description="Number of random inputs to validate with.")
    tolerance_mae: float = Field(1e-5, description="Maximum acceptable Mean Absolute Error.")
    tolerance_cos_sim: float = Field(0.999, description="Minimum acceptable Cosine Similarity.")


class OptimizeConfig(BaseModel):
    """Configuration for the optimization step."""
    simplify: bool = Field(True, description="Apply onnx-simplifier.")
    quantize: Optional[str] = Field(None, description="Quantization mode: 'int8', 'fp16', or None.")
    constant_folding: bool = Field(True, description="Apply constant folding optimization.")


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking."""
    iters: int = Field(100, description="Number of benchmark iterations.")
    batch_size: int = Field(1, description="Batch size for benchmarking.")
    warmup_iters: int = Field(10, description="Number of warmup iterations.")
    device: str = Field("cpu", description="Device: 'cpu' or 'cuda'.")


class Config(BaseModel):
    """Top-level configuration for the OxidizedVision pipeline."""
    model: ModelConfig
    export: ExportConfig = ExportConfig()
    runners: List[RunnerConfig] = []
    validation: ValidateConfig = ValidateConfig()
    optimize: OptimizeConfig = OptimizeConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()


def load_config(config_path: str) -> Config:
    """Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Validated Config object.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValidationError: If the config has invalid values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return Config(**raw)


def save_config(config: Config, config_path: str) -> None:
    """Save a Config object to a YAML file.
    
    Args:
        config: Config object to save.
        config_path: Path to write the YAML file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False)
