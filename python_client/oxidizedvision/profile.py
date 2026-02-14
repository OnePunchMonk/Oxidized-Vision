"""
OxidizedVision — Model profiling module.

Provides model summary, parameter count, and FLOPs estimation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.table import Table

from .convert import _import_model_from_path

console = Console()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the total and trainable parameters of a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dict with 'total', 'trainable', and 'non_trainable' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate the model size in megabytes (assuming float32).
    
    Args:
        model: PyTorch model.
        
    Returns:
        Estimated size in MB.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def get_layer_summary(model: torch.nn.Module, input_shape: List[int]) -> List[Dict[str, Any]]:
    """Get a per-layer summary of the model.
    
    Args:
        model: PyTorch model.
        input_shape: Input tensor shape.
        
    Returns:
        List of dicts with layer info.
    """
    layers = []
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            output_shape = list(output.shape) if isinstance(output, torch.Tensor) else "N/A"
            layers.append({
                "name": name,
                "type": module.__class__.__name__,
                "output_shape": output_shape,
                "params": params,
            })
        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(hook_fn(name)))

    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    return layers


def profile_model(
    model_source_path: str,
    model_class_name: str,
    input_shape: List[int],
    checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Profile a PyTorch model: parameters, size, and per-layer breakdown.
    
    Args:
        model_source_path: Path to the .py file containing the model class.
        model_class_name: Name of the nn.Module class.
        input_shape: Input tensor shape.
        checkpoint: Optional checkpoint path.
        
    Returns:
        Profiling results dict.
    """
    model_class = _import_model_from_path(model_source_path, model_class_name)
    model = model_class()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)
    layers = get_layer_summary(model, input_shape)

    return {
        "model_class": model_class_name,
        "input_shape": input_shape,
        "parameters": params,
        "estimated_size_mb": round(size_mb, 3),
        "layers": layers,
    }


def print_profile(profile: Dict[str, Any]) -> None:
    """Pretty-print a model profile using rich tables.
    
    Args:
        profile: Profile dict from profile_model().
    """
    console.print(f"\n📊 [bold]Model Profile: {profile['model_class']}[/bold]")
    console.print(f"   Input shape: {profile['input_shape']}")
    console.print(f"   Total parameters: {profile['parameters']['total']:,}")
    console.print(f"   Trainable parameters: {profile['parameters']['trainable']:,}")
    console.print(f"   Estimated size: {profile['estimated_size_mb']:.3f} MB")

    if profile["layers"]:
        table = Table(title="Layer Summary")
        table.add_column("Layer", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Output Shape", style="green")
        table.add_column("Parameters", style="yellow", justify="right")

        for layer in profile["layers"]:
            table.add_row(
                layer["name"],
                layer["type"],
                str(layer["output_shape"]),
                f"{layer['params']:,}" if layer["params"] > 0 else "-",
            )

        total_in_layers = sum(l["params"] for l in profile["layers"])
        table.add_row(
            "[bold]Total[/bold]", "", "",
            f"[bold]{total_in_layers:,}[/bold]",
        )
        console.print(table)
