"""Tests for the profiling module."""

import pytest
import torch
from oxidizedvision.profile import (
    count_parameters,
    estimate_model_size_mb,
    get_layer_summary,
    profile_model,
)


class TestCountParameters:
    def test_known_model(self):
        model = torch.nn.Linear(10, 5)
        params = count_parameters(model)
        # 10*5 weights + 5 bias = 55
        assert params["total"] == 55
        assert params["trainable"] == 55
        assert params["non_trainable"] == 0

    def test_frozen_params(self):
        model = torch.nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        params = count_parameters(model)
        assert params["trainable"] == 0
        assert params["non_trainable"] == 55


class TestEstimateModelSize:
    def test_size_estimate(self):
        model = torch.nn.Linear(1000, 1000)
        size_mb = estimate_model_size_mb(model)
        # ~4MB for 1M float32 params
        assert 3.0 < size_mb < 5.0


class TestProfileModel:
    def test_profile_model(self, tmp_model_dir):
        result = profile_model(
            model_source_path=str(tmp_model_dir / "model.py"),
            model_class_name="SimpleModel",
            input_shape=[1, 3, 32, 32],
        )
        assert result["model_class"] == "SimpleModel"
        assert result["parameters"]["total"] > 0
        assert result["estimated_size_mb"] > 0
        assert len(result["layers"]) > 0

    def test_layer_summary_types(self, tmp_model_dir):
        result = profile_model(
            model_source_path=str(tmp_model_dir / "model.py"),
            model_class_name="SimpleModel",
            input_shape=[1, 3, 32, 32],
        )
        layer_types = [l["type"] for l in result["layers"]]
        assert "Conv2d" in layer_types
        assert "ReLU" in layer_types
