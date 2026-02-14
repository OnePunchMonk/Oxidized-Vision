"""Shared test fixtures for OxidizedVision tests."""

import pytest
import os
import torch
from pathlib import Path


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a temporary directory with a simple PyTorch model."""
    model_file = tmp_path / "model.py"
    model_file.write_text(
        '''
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
'''
    )
    return tmp_path


@pytest.fixture
def simple_config(tmp_model_dir):
    """Generate a test config dict."""
    return {
        "model": {
            "path": str(tmp_model_dir / "model.py"),
            "class_name": "SimpleModel",
            "input_shape": [1, 3, 32, 32],
        },
        "export": {
            "opset_version": 14,
            "do_constant_folding": True,
            "output_dir": str(tmp_model_dir / "out"),
            "model_name": "test_model",
        },
    }


@pytest.fixture
def simple_config_yaml(tmp_model_dir, simple_config):
    """Write a YAML config file and return its path."""
    import yaml

    config_path = tmp_model_dir / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(simple_config, f)
    return str(config_path)
