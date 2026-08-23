"""
OxidizedVision — Model pruning module.

Zeroes out low-magnitude weights in a PyTorch model's Conv2d/Linear layers
before export, as a model-compression step ahead of quantization.

**Honest caveat on speed, not just size:** unstructured pruning (the
default here) zeros individual weights without changing tensor shapes, so
it reduces the *stored* parameter count and can shrink a checkpoint on
disk, but a dense CPU/GPU inference kernel (ONNX Runtime, tract, LibTorch)
still multiplies through the zeros — it does **not** speed up inference by
itself, only sparse-aware runtimes would benefit, and none of this repo's
backends currently are. Structured pruning (`pruning_structured=True`)
zeros whole output channels rather than individual weights, which is the
prerequisite for actually removing those channels (shrinking the tensor
and thus the FLOP count) — but this step only zeros the channels, it
doesn't yet re-slice the model to drop them, so it doesn't reduce compute
either. Use pruning here for its real, delivered benefit today — a smaller,
more compressible checkpoint, and a foundation to build real channel
removal on top of — not as a standalone latency win.
"""

import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from rich.console import Console

console = Console()


def _prunable_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            yield name, module


def compute_sparsity(model: nn.Module) -> float:
    """Fraction of zero-valued weights across all Conv/Linear layers."""
    total = 0
    zeros = 0
    for _, module in _prunable_modules(model):
        weight = module.weight.data
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    return zeros / total if total > 0 else 0.0


def prune_model(
    model: nn.Module,
    amount: float,
    structured: bool = False,
) -> tuple[nn.Module, float]:
    """Prune a PyTorch model's Conv/Linear layers in place.

    Args:
        model: The model to prune (modified in place, and returned).
        amount: Fraction (0-1) of weights (unstructured) or channels
            (structured) to zero out per layer.
        structured: If True, zero whole output channels (ln_structured,
            L2 norm, dim=0) instead of individual weights (global L1
            magnitude across all prunable layers). See module docstring
            for what this does and doesn't speed up.

    Returns:
        (pruned_model, sparsity) — sparsity is the resulting fraction of
        zero-valued weights, as measured after pruning.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"pruning amount must be in (0, 1), got {amount}")

    modules = list(_prunable_modules(model))
    if not modules:
        console.print("[yellow]Warning: no Conv2d/Conv1d/Linear layers found to prune.[/yellow]")
        return model, 0.0

    if structured:
        console.print(
            f"🔧 Structured channel pruning ({amount:.0%} of output channels per layer)..."
        )
        for _, module in modules:
            torch_prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            torch_prune.remove(module, "weight")
    else:
        console.print(f"🔧 Unstructured global magnitude pruning ({amount:.0%} of weights)...")
        params_to_prune = [(module, "weight") for _, module in modules]
        torch_prune.global_unstructured(
            params_to_prune,
            pruning_method=torch_prune.L1Unstructured,
            amount=amount,
        )
        for module in [m for _, m in modules]:
            torch_prune.remove(module, "weight")

    sparsity = compute_sparsity(model)
    console.print(f"✅ Pruning complete — {sparsity:.1%} of weights are now zero.")
    return model, sparsity
