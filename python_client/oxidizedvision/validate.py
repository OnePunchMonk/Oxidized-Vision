import torch
import onnxruntime as ort
import numpy as np
from rich.table import Table
from rich.console import Console
import itertools

def calculate_mae(tensor1, tensor2):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(tensor1 - tensor2))

def calculate_cosine_similarity(tensor1, tensor2):
    """Calculates Cosine Similarity."""
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    dot_product = np.dot(tensor1, tensor2)
    norm_1 = np.linalg.norm(tensor1)
    norm_2 = np.linalg.norm(tensor2)
    return dot_product / (norm_1 * norm_2)

def validate_models(model_paths: dict, tolerance_mae: float = 1e-5, tolerance_cos_sim: float = 0.999):
    """
    Validates that different model formats produce similar outputs.
    """
    console = Console()
    outputs = {}
    
    # Create a dummy input
    input_shape = [1, 3, 256, 256]  # Assuming this shape
    dummy_input_torch = torch.randn(*input_shape)
    dummy_input_numpy = dummy_input_torch.numpy()

    # --- Get PyTorch output (if available) ---
    if "pytorch" in model_paths:
        try:
            # This part is tricky as it requires loading the model definition.
            # For now, we'll assume a placeholder. A better implementation
            # would dynamically import the model class.
            console.print("[yellow]Warning: PyTorch model validation is a placeholder.[/yellow]")
            # outputs["pytorch"] = dummy_input_numpy # Placeholder
        except Exception as e:
            console.print(f"[red]Error loading PyTorch model: {e}[/red]")

    # --- Get TorchScript output ---
    if "torchscript" in model_paths:
        try:
            ts_model = torch.jit.load(model_paths["torchscript"])
            ts_model.eval()
            ts_output = ts_model(dummy_input_torch)
            outputs["torchscript"] = ts_output.detach().numpy()
            console.print(f"✅ Loaded and ran [cyan]TorchScript[/cyan] model.")
        except Exception as e:
            console.print(f"[red]Error with TorchScript model: {e}[/red]")

    # --- Get ONNX output ---
    if "onnx" in model_paths:
        try:
            ort_session = ort.InferenceSession(model_paths["onnx"])
            input_name = ort_session.get_inputs()[0].name
            ort_output = ort_session.run(None, {input_name: dummy_input_numpy})
            outputs["onnx"] = ort_output[0]
            console.print(f"✅ Loaded and ran [cyan]ONNX[/cyan] model.")
        except Exception as e:
            console.print(f"[red]Error with ONNX model: {e}[/red]")

    if len(outputs) < 2:
        console.print("[red]Need at least two models to compare. Aborting validation.[/red]")
        return

    # --- Compare outputs ---
    table = Table(title="Validation Report")
    table.add_column("Comparison", justify="center", style="cyan")
    table.add_column("MAE", style="magenta")
    table.add_column("Cosine Similarity", style="green")
    table.add_column("MAE Status", style="yellow")
    table.add_column("CosSim Status", style="yellow")

    all_passed = True
    for (name1, out1), (name2, out2) in itertools.combinations(outputs.items(), 2):
        mae = calculate_mae(out1, out2)
        cos_sim = calculate_cosine_similarity(out1, out2)
        
        mae_passed = mae <= tolerance_mae
        cosim_passed = cos_sim >= tolerance_cos_sim
        
        if not (mae_passed and cosim_passed):
            all_passed = False

        table.add_row(
            f"{name1} vs {name2}",
            f"{mae:.2e}",
            f"{cos_sim:.6f}",
            f"[green]PASS[/green]" if mae_passed else f"[red]FAIL[/red]",
            f"[green]PASS[/green]" if cosim_passed else f"[red]FAIL[/red]",
        )
        
    console.print(table)
    
    if all_passed:
        console.print(f"✅ [bold green]All models are consistent within tolerance (MAE <= {tolerance_mae:.2e}, CosSim >= {tolerance_cos_sim}).[/bold green]")
    else:
        console.print(f"❌ [bold red]Inconsistency detected. Check the report above.[/bold red]")
