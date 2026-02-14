"""
OxidizedVision — Model registry module.

Tracks converted models and their metadata in a local JSON-based registry.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

console = Console()

REGISTRY_FILENAME = ".oxidizedvision_registry.json"


def _get_registry_path(base_dir: str = ".") -> str:
    """Get the path to the registry file."""
    return os.path.join(base_dir, REGISTRY_FILENAME)


def _load_registry(base_dir: str = ".") -> Dict[str, Any]:
    """Load the registry from disk."""
    path = _get_registry_path(base_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"models": {}, "version": "1.0"}


def _save_registry(registry: Dict[str, Any], base_dir: str = ".") -> None:
    """Save the registry to disk."""
    path = _get_registry_path(base_dir)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_model(
    model_name: str,
    model_paths: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
    base_dir: str = ".",
) -> None:
    """Register a converted model in the local registry.
    
    Args:
        model_name: Name of the model.
        model_paths: Dict mapping format to path (e.g. {'torchscript': 'out/model.pt'}).
        config: Optional config dict used during conversion.
        base_dir: Base directory for the registry file.
    """
    registry = _load_registry(base_dir)

    file_sizes = {}
    for fmt, path in model_paths.items():
        if os.path.exists(path):
            file_sizes[fmt] = os.path.getsize(path)

    registry["models"][model_name] = {
        "paths": model_paths,
        "file_sizes": file_sizes,
        "config": config,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    _save_registry(registry, base_dir)
    console.print(f"📦 Registered model [bold cyan]{model_name}[/bold cyan] in registry.")


def list_models(base_dir: str = ".") -> List[Dict[str, Any]]:
    """List all registered models.
    
    Args:
        base_dir: Base directory for the registry file.
        
    Returns:
        List of model info dicts.
    """
    registry = _load_registry(base_dir)
    models = []
    for name, info in registry.get("models", {}).items():
        models.append({"name": name, **info})
    return models


def get_model_info(model_name: str, base_dir: str = ".") -> Optional[Dict[str, Any]]:
    """Get info for a specific model.
    
    Args:
        model_name: Name of the model.
        base_dir: Base directory for the registry file.
        
    Returns:
        Model info dict, or None if not found.
    """
    registry = _load_registry(base_dir)
    if model_name in registry.get("models", {}):
        return {"name": model_name, **registry["models"][model_name]}
    return None


def remove_model(model_name: str, base_dir: str = ".") -> bool:
    """Remove a model from the registry (does not delete files).
    
    Args:
        model_name: Name of the model to remove.
        base_dir: Base directory for the registry file.
        
    Returns:
        True if removed, False if not found.
    """
    registry = _load_registry(base_dir)
    if model_name in registry.get("models", {}):
        del registry["models"][model_name]
        _save_registry(registry, base_dir)
        console.print(f"🗑️  Removed [bold cyan]{model_name}[/bold cyan] from registry.")
        return True
    console.print(f"[yellow]Model '{model_name}' not found in registry.[/yellow]")
    return False


def print_model_list(base_dir: str = ".") -> None:
    """Pretty-print the list of registered models."""
    models = list_models(base_dir)
    if not models:
        console.print("[dim]No models registered. Run 'oxidizedvision convert' first.[/dim]")
        return

    table = Table(title="Registered Models")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Formats", style="magenta")
    table.add_column("Total Size", style="green", justify="right")
    table.add_column("Created", style="yellow")

    for model in models:
        formats = ", ".join(model.get("paths", {}).keys())
        total_size = sum(model.get("file_sizes", {}).values())
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024 * 1024):.1f} MB"
        created = model.get("created_at", "N/A")[:19]
        table.add_row(model["name"], formats, size_str, created)

    console.print(table)


def print_model_info(model_name: str, base_dir: str = ".") -> None:
    """Pretty-print detailed info for a specific model."""
    info = get_model_info(model_name, base_dir)
    if not info:
        console.print(f"[yellow]Model '{model_name}' not found in registry.[/yellow]")
        return

    console.print(f"\n📦 [bold]Model: {info['name']}[/bold]")
    console.print(f"   Created: {info.get('created_at', 'N/A')}")
    console.print(f"   Updated: {info.get('updated_at', 'N/A')}")

    if info.get("paths"):
        console.print("\n   📁 Files:")
        for fmt, path in info["paths"].items():
            size = info.get("file_sizes", {}).get(fmt, 0)
            exists = "✅" if os.path.exists(path) else "❌"
            console.print(f"      {exists} [{fmt}] {path} ({size / 1024:.1f} KB)")

    if info.get("config"):
        console.print(f"\n   ⚙️  Config:")
        config = info["config"]
        if "model" in config:
            console.print(f"      Class: {config['model'].get('class_name', 'N/A')}")
            console.print(f"      Input shape: {config['model'].get('input_shape', 'N/A')}")
