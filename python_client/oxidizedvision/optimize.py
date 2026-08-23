"""
OxidizedVision — ONNX optimization module.

Provides graph simplification, quantization, and operator fusion for ONNX models.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
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
        console.print(
            "[yellow]Warning: Simplified model failed validation check. Using original.[/yellow]"
        )
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


class _RandomCalibrationDataReader:
    """Feeds random-normal input tensors for static (calibration-based) quantization.

    Static PTQ needs representative activations to compute per-tensor/per-channel
    quantization ranges, unlike dynamic quantization (which only quantizes
    weights and computes activation ranges on the fly at every inference —
    cheaper to set up, but leaves accuracy/speed on the table for CNN/ViT
    vision backbones where activation distributions are well-behaved enough
    for static ranges to pay off). Random-normal data is a reasonable default
    calibration set when no real dataset is supplied; pass `calibration_data`
    for a tighter calibration against actual input distributions.
    """

    def __init__(
        self,
        input_name: str,
        input_shape: list[int],
        num_samples: int,
        calibration_data: Optional[np.ndarray] = None,
    ):
        self._input_name = input_name
        if calibration_data is not None:
            self._samples = iter(calibration_data)
        else:
            rng = np.random.default_rng(seed=0)
            sample_shape = (num_samples, *input_shape[1:])
            self._samples = iter(rng.standard_normal(sample_shape).astype(np.float32))

    def get_next(self):
        sample = next(self._samples, None)
        if sample is None:
            return None
        return {self._input_name: np.expand_dims(sample, 0).astype(np.float32)}


def quantize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "int8",
    input_shape: Optional[list[int]] = None,
    num_calibration_samples: int = 32,
    calibration_data: Optional[np.ndarray] = None,
) -> str:
    """Quantize an ONNX model.

    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to save the quantized model. If None, appends '_quantized'.
        mode: Quantization mode — 'int8' (dynamic, weight-only), 'static_int8'
            (calibration-based, quantizes activations too — generally better
            accuracy/speed tradeoff for CNN/ViT vision backbones), or 'fp16'.
        input_shape: Required for 'static_int8' — the model's input shape,
            used to generate calibration data if `calibration_data` isn't given.
        num_calibration_samples: Number of calibration samples for 'static_int8'
            when generating synthetic (random-normal) calibration data.
        calibration_data: Optional array of shape [N, *input_shape[1:]] with
            real representative inputs for 'static_int8' calibration — passing
            actual data (rather than random noise) gives tighter, more accurate
            quantization ranges.

    Returns:
        Path to the quantized model.
    """
    if output_path is None:
        stem = Path(input_path).stem
        suffix = Path(input_path).suffix
        output_path = str(Path(input_path).parent / f"{stem}_quantized{suffix}")

    console.print(f"🔧 Quantizing ONNX model ({mode}): [dim]{input_path}[/dim]")

    if mode == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
    elif mode == "static_int8":
        if input_shape is None:
            raise ValueError("static_int8 quantization requires input_shape.")

        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_static

        input_name = (
            ort.InferenceSession(input_path, providers=["CPUExecutionProvider"])
            .get_inputs()[0]
            .name
        )

        reader = _RandomCalibrationDataReader(
            input_name, input_shape, num_calibration_samples, calibration_data
        )

        quantize_static(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
    elif mode == "fp16":
        import onnx
        from onnxruntime.transformers import float16

        model = onnx.load(input_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
    else:
        raise ValueError(
            f"Unknown quantization mode: {mode}. Expected 'int8', 'static_int8', or 'fp16'."
        )

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
    input_shape: Optional[list[int]] = None,
    num_calibration_samples: int = 32,
    calibration_data: Optional[np.ndarray] = None,
) -> str:
    """Full optimization pipeline for an ONNX model.

    Args:
        input_path: Path to the input ONNX model.
        output_path: Path for the optimized model. If None, appends '_optimized'.
        simplify: Whether to apply onnx-simplifier.
        quantize: Quantization mode ('int8', 'static_int8', 'fp16', or None).
        constant_folding: Whether to apply constant folding (included in simplify).
        input_shape: Required when quantize='static_int8'.
        num_calibration_samples: Synthetic calibration sample count for 'static_int8'.
        calibration_data: Optional real calibration data for 'static_int8'.

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
    console.print(
        f"\n🚀 Starting ONNX optimization pipeline for [bold cyan]{input_path}[/bold cyan]"
    )

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
        console.print("  ✅ Constant folding applied")

    # Step 4: Quantize
    if quantize:
        current_path = quantize_onnx(
            current_path,
            output_path,
            mode=quantize,
            input_shape=input_shape,
            num_calibration_samples=num_calibration_samples,
            calibration_data=calibration_data,
        )

    console.print(f"\n🎉 Optimization complete: [bold green]{current_path}[/bold green]")
    return current_path
