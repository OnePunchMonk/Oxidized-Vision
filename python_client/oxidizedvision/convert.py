"""
OxidizedVision — Model conversion module.

Converts PyTorch models to TorchScript and ONNX formats.
"""

import torch
import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

from .config import Config, load_config

console = Console()


def _import_model_from_path(model_path: str, model_class_name: str) -> type:
    """Dynamically import a model class from a given file path.
    
    Args:
        model_path: Path to the .py file containing the model class.
        model_class_name: Name of the nn.Module subclass.
        
    Returns:
        The model class (not an instance).
        
    Raises:
        ImportError: If the module or class can't be loaded.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location(model_class_name, str(path))
    if spec is None:
        raise ImportError(f"Could not load spec for module at {model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, model_class_name):
        raise ImportError(
            f"Class '{model_class_name}' not found in {model_path}. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    model_class = getattr(module, model_class_name)
    return model_class


def load_model(config: Config) -> torch.nn.Module:
    """Load and instantiate a PyTorch model from config.
    
    Args:
        config: Validated Config object.
        
    Returns:
        Instantiated and eval-mode PyTorch model.
    """
    model_config = config.model

    console.print(f"📦 Importing [cyan]{model_config.class_name}[/cyan] from [dim]{model_config.path}[/dim]")
    model_class = _import_model_from_path(model_config.path, model_config.class_name)
    model = model_class()

    if model_config.checkpoint:
        checkpoint_path = Path(model_config.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_config.checkpoint}")
        console.print(f"📥 Loading checkpoint from [dim]{model_config.checkpoint}[/dim]")
        model.load_state_dict(torch.load(model_config.checkpoint, map_location="cpu"))

    model.eval()
    return model


def convert_to_torchscript(
    model: torch.nn.Module,
    input_shape: list,
    output_dir: str,
    model_name: str,
) -> str:
    """Convert a PyTorch model to TorchScript via tracing.
    
    Args:
        model: PyTorch model in eval mode.
        input_shape: Shape of the dummy input tensor.
        output_dir: Directory to save the .pt file.
        model_name: Base filename for the output.
        
    Returns:
        Path to the saved TorchScript model.
    """
    os.makedirs(output_dir, exist_ok=True)
    dummy_input = torch.randn(input_shape)

    console.print(f"🔄 Tracing model with input shape {input_shape}...")
    traced_model = torch.jit.trace(model, dummy_input)

    traced_path = os.path.join(output_dir, f"{model_name}.pt")
    traced_model.save(traced_path)
    console.print(f"✅ TorchScript model saved to [green]{traced_path}[/green]")

    return traced_path


def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: list,
    output_dir: str,
    model_name: str,
    opset_version: int = 14,
    do_constant_folding: bool = True,
) -> str:
    """Convert a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model in eval mode.
        input_shape: Shape of the dummy input tensor.
        output_dir: Directory to save the .onnx file.
        model_name: Base filename for the output.
        opset_version: ONNX opset version.
        do_constant_folding: Whether to apply constant folding.
        
    Returns:
        Path to the saved ONNX model.
    """
    os.makedirs(output_dir, exist_ok=True)
    dummy_input = torch.randn(input_shape)

    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    console.print(f"🔄 Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    console.print(f"✅ ONNX model saved to [green]{onnx_path}[/green]")

    return onnx_path


def convert_model(config: dict) -> Tuple[str, str]:
    """Full conversion pipeline: PyTorch → TorchScript + ONNX.
    
    Args:
        config: Either a Config object or a raw dict (for backward compatibility).
        
    Returns:
        Tuple of (torchscript_path, onnx_path).
    """
    if isinstance(config, dict):
        cfg = Config(**config)
    else:
        cfg = config

    model = load_model(cfg)

    output_dir = cfg.export.output_dir
    model_name = cfg.export.model_name
    input_shape = cfg.model.input_shape

    ts_path = convert_to_torchscript(model, input_shape, output_dir, model_name)
    onnx_path = convert_to_onnx(
        model,
        input_shape,
        output_dir,
        model_name,
        opset_version=cfg.export.opset_version,
        do_constant_folding=cfg.export.do_constant_folding,
    )

    console.print(f"\n🎉 Conversion complete! Models saved to [bold cyan]{output_dir}/[/bold cyan]")
    return ts_path, onnx_path
