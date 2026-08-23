"""Tests for the pruning module."""

import torch.nn as nn
from oxidizedvision.prune import compute_sparsity, prune_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class TestPruneModel:
    def test_unstructured_pruning_reaches_target_sparsity(self):
        model = TinyModel()
        assert compute_sparsity(model) == 0.0

        pruned, sparsity = prune_model(model, amount=0.3, structured=False)
        assert pruned is model
        assert abs(sparsity - 0.3) < 0.05
        assert abs(compute_sparsity(model) - sparsity) < 1e-9

    def test_structured_pruning_zeros_whole_channels(self):
        model = TinyModel()
        _, sparsity = prune_model(model, amount=0.5, structured=True)
        assert sparsity > 0.0

        # At least one full output channel of the conv should be all zeros.
        weight = model.conv.weight.data
        zeroed_channels = (weight.view(weight.shape[0], -1).abs().sum(dim=1) == 0).sum().item()
        assert zeroed_channels > 0

    def test_invalid_amount_raises(self):
        model = TinyModel()
        for bad in (0.0, 1.0, -0.1, 1.5):
            try:
                prune_model(model, amount=bad)
                assert False, f"expected ValueError for amount={bad}"
            except ValueError:
                pass

    def test_no_prunable_layers_returns_zero_sparsity(self):
        model = nn.ReLU()
        _, sparsity = prune_model(model, amount=0.3)
        assert sparsity == 0.0
