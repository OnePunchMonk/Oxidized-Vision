"""
OxidizedVision — Model validation module.

Compares outputs across PyTorch, TorchScript, and ONNX to ensure numerical consistency.
"""

import torch
import numpy as np
import onnxruntime as ort
import itertools
from typing import Dict, Optional, List
from rich.table import Table
from rich.console import Console

from .config import Config
from .convert import _import_model_from_path

console = Console()


def calculate_mae(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    """Calculate Mean Absolute Error between two tensors."""
    return float(np.mean(np.abs(tensor1 - tensor2)))


def calculate_cosine_similarity(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    """Calculate Cosine Similarity between two flattened tensors."""
    t1 = tensor1.flatten()
    t2 = tensor2.flatten()
    dot_product = np.dot(t1, t2)
    norm_1 = np.linalg.norm(t1)
    norm_2 = np.linalg.norm(t2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return float(dot_product / (norm_1 * norm_2))


def calculate_max_abs_error(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    """Calculate Maximum Absolute Error between two tensors."""
    return float(np.max(np.abs(tensor1 - tensor2)))


def calculate_rmse(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    """Calculate Root Mean Squared Error between two tensors."""
    return float(np.sqrt(np.mean((tensor1 - tensor2) ** 2)))


def _get_pytorch_output(
    model_path: str,
    class_name: str,
    input_tensor: torch.Tensor,
    checkpoint: Optional[str] = None,
) -> np.ndarray:
    """Run inference with a native PyTorch model.
    
    Args:
        model_path: Path to the .py file containing the model class.
        class_name: Name of the nn.Module class.
        input_tensor: Input tensor for inference.
        checkpoint: Optional checkpoint path.
        
    Returns:
        Output as a numpy array.
    """
    model_class = _import_model_from_path(model_path, class_name)
    model = model_class()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy()


def _get_torchscript_output(model_path: str, input_tensor: torch.Tensor) -> np.ndarray:
    """Run inference with a TorchScript model."""
    ts_model = torch.jit.load(model_path)
    ts_model.eval()
    with torch.no_grad():
        output = ts_model(input_tensor)
    return output.numpy()


def _get_onnx_output(model_path: str, input_numpy: np.ndarray) -> np.ndarray:
    """Run inference with an ONNX model via onnxruntime."""
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: input_numpy})
    return ort_output[0]


def validate_models(
    model_paths: dict,
    input_shape: Optional[List[int]] = None,
    tolerance_mae: float = 1e-5,
    tolerance_cos_sim: float = 0.999,
    num_tests: int = 1,
    model_source_path: Optional[str] = None,
    model_class_name: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
) -> bool:
    """Validate that different model formats produce consistent outputs.
    
    Args:
        model_paths: Dict mapping format name to file path.
                     Keys: 'pytorch', 'torchscript', 'onnx'.
        input_shape: Input tensor shape. Defaults to [1, 3, 256, 256].
        tolerance_mae: Maximum acceptable MAE.
        tolerance_cos_sim: Minimum acceptable Cosine Similarity.
        num_tests: Number of random inputs to test.
        model_source_path: Path to .py file (for 'pytorch' validation).
        model_class_name: Class name (for 'pytorch' validation).
        model_checkpoint: Checkpoint path (for 'pytorch' validation).
        
    Returns:
        True if all comparisons pass, False otherwise.
    """
    if input_shape is None:
        input_shape = [1, 3, 256, 256]

    all_passed = True

    for test_idx in range(num_tests):
        if num_tests > 1:
            console.print(f"\n--- Test {test_idx + 1}/{num_tests} ---")

        # Generate random input
        dummy_input_torch = torch.randn(*input_shape)
        dummy_input_numpy = dummy_input_torch.numpy()

        outputs: Dict[str, np.ndarray] = {}

        # --- Get PyTorch output ---
        if "pytorch" in model_paths and model_source_path and model_class_name:
            try:
                out = _get_pytorch_output(
                    model_source_path, model_class_name, dummy_input_torch, model_checkpoint
                )
                outputs["pytorch"] = out
                console.print(f"✅ Ran [cyan]PyTorch[/cyan] model — output shape: {out.shape}")
            except Exception as e:
                console.print(f"[red]Error with PyTorch model: {e}[/red]")

        # --- Get TorchScript output ---
        if "torchscript" in model_paths:
            try:
                out = _get_torchscript_output(model_paths["torchscript"], dummy_input_torch)
                outputs["torchscript"] = out
                console.print(f"✅ Ran [cyan]TorchScript[/cyan] model — output shape: {out.shape}")
            except Exception as e:
                console.print(f"[red]Error with TorchScript model: {e}[/red]")

        # --- Get ONNX output ---
        if "onnx" in model_paths:
            try:
                out = _get_onnx_output(model_paths["onnx"], dummy_input_numpy)
                outputs["onnx"] = out
                console.print(f"✅ Ran [cyan]ONNX[/cyan] model — output shape: {out.shape}")
            except Exception as e:
                console.print(f"[red]Error with ONNX model: {e}[/red]")

        if len(outputs) < 2:
            console.print("[red]Need at least two models to compare. Aborting validation.[/red]")
            return False

        # --- Compare outputs ---
        table = Table(title=f"Validation Report{f' (Test {test_idx + 1})' if num_tests > 1 else ''}")
        table.add_column("Comparison", justify="center", style="cyan")
        table.add_column("MAE", style="magenta")
        table.add_column("RMSE", style="blue")
        table.add_column("Max Error", style="yellow")
        table.add_column("Cosine Sim", style="green")
        table.add_column("MAE Status", style="bold")
        table.add_column("CosSim Status", style="bold")

        for (name1, out1), (name2, out2) in itertools.combinations(outputs.items(), 2):
            mae = calculate_mae(out1, out2)
            rmse = calculate_rmse(out1, out2)
            max_err = calculate_max_abs_error(out1, out2)
            cos_sim = calculate_cosine_similarity(out1, out2)

            mae_passed = mae <= tolerance_mae
            cosim_passed = cos_sim >= tolerance_cos_sim

            if not (mae_passed and cosim_passed):
                all_passed = False

            table.add_row(
                f"{name1} vs {name2}",
                f"{mae:.2e}",
                f"{rmse:.2e}",
                f"{max_err:.2e}",
                f"{cos_sim:.6f}",
                "[green]PASS[/green]" if mae_passed else "[red]FAIL[/red]",
                "[green]PASS[/green]" if cosim_passed else "[red]FAIL[/red]",
            )

        console.print(table)

    if all_passed:
        console.print(
            f"\n✅ [bold green]All models are consistent within tolerance "
            f"(MAE ≤ {tolerance_mae:.2e}, CosSim ≥ {tolerance_cos_sim}).[/bold green]"
        )
    else:
        console.print(
            f"\n❌ [bold red]Inconsistency detected! Check the report above.[/bold red]"
        )

    return all_passed
