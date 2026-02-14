"""
OxidizedVision — ONNX optimization module.

Provides graph simplification, quantization, and operator fusion for ONNX models.
"""

import os
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


def simplify_onnx(input_path: str, output_path: Optional[str] = None) -> str:
    """Simplify an ONNX model using onnx-simplifier.
    
    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to save the simplified model. If None, overwrites input.
        
    Returns:
        Path to the simplified model.
    """
    import onnx
    from onnxsim import simplify

    if output_path is None:
        output_path = input_path

    console.print(f"🔧 Simplifying ONNX model: [dim]{input_path}[/dim]")

    model = onnx.load(input_path)
    model_simp, check = simplify(model)

    if not check:
        console.print("[yellow]Warning: Simplified model failed validation check. Using original.[/yellow]")
        return input_path

    onnx.save(model_simp, output_path)

    # Report size reduction
    original_size = os.path.getsize(input_path)
    simplified_size = os.path.getsize(output_path)
    reduction = (1 - simplified_size / original_size) * 100 if original_size > 0 else 0

    console.print(
        f"✅ Simplified model saved to [green]{output_path}[/green] "
        f"({original_size / 1024:.1f} KB → {simplified_size / 1024:.1f} KB, "
        f"{reduction:.1f}% reduction)"
    )
    return output_path


def quantize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "int8",
) -> str:
    """Quantize an ONNX model.
    
    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to save the quantized model. If None, appends '_quantized'.
        mode: Quantization mode — 'int8' (dynamic) or 'fp16'.
        
    Returns:
        Path to the quantized model.
    """
    if output_path is None:
        stem = Path(input_path).stem
        suffix = Path(input_path).suffix
        output_path = str(Path(input_path).parent / f"{stem}_quantized{suffix}")

    console.print(f"🔧 Quantizing ONNX model ({mode}): [dim]{input_path}[/dim]")

    if mode == "int8":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
    elif mode == "fp16":
        import onnx
        from onnxruntime.transformers import float16
        model = onnx.load(input_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}. Expected 'int8' or 'fp16'.")

    original_size = os.path.getsize(input_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0

    console.print(
        f"✅ Quantized model saved to [green]{output_path}[/green] "
        f"({original_size / 1024:.1f} KB → {quantized_size / 1024:.1f} KB, "
        f"{reduction:.1f}% reduction)"
    )
    return output_path


def optimize_model(
    input_path: str,
    output_path: Optional[str] = None,
    simplify: bool = True,
    quantize: Optional[str] = None,
    constant_folding: bool = True,
) -> str:
    """Full optimization pipeline for an ONNX model.
    
    Args:
        input_path: Path to the input ONNX model.
        output_path: Path for the optimized model. If None, appends '_optimized'.
        simplify: Whether to apply onnx-simplifier.
        quantize: Quantization mode ('int8', 'fp16', or None).
        constant_folding: Whether to apply constant folding (included in simplify).
        
    Returns:
        Path to the optimized model.
    """
    import onnx

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"ONNX model not found: {input_path}")

    if output_path is None:
        stem = Path(input_path).stem
        suffix = Path(input_path).suffix
        output_path = str(Path(input_path).parent / f"{stem}_optimized{suffix}")

    current_path = input_path
    console.print(f"\n🚀 Starting ONNX optimization pipeline for [bold cyan]{input_path}[/bold cyan]")

    # Step 1: Validate the model
    console.print("  📋 Validating input model...")
    model = onnx.load(input_path)
    onnx.checker.check_model(model)
    console.print("  ✅ Model is valid ONNX")

    # Step 2: Simplify
    if simplify:
        current_path = simplify_onnx(current_path, output_path)

    # Step 3: Constant folding (via onnxruntime optimization)
    if constant_folding and not simplify:
        # onnx-simplifier already does constant folding, so only do this
        # if simplification is disabled
        import onnxruntime as ort
        console.print("  🔧 Applying constant folding...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_path
        ort.InferenceSession(current_path, sess_options)
        current_path = output_path
        console.print(f"  ✅ Constant folding applied")

    # Step 4: Quantize
    if quantize:
        current_path = quantize_onnx(current_path, output_path, mode=quantize)

    console.print(f"\n🎉 Optimization complete: [bold green]{current_path}[/bold green]")
    return current_path
