"""Tests for the validation module."""

import pytest
import numpy as np
import os
from oxidizedvision.validate import (
    calculate_mae,
    calculate_cosine_similarity,
    calculate_max_abs_error,
    calculate_rmse,
    validate_models,
)
from oxidizedvision.convert import convert_model
from oxidizedvision.config import Config


class TestMetrics:
    def test_mae_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert calculate_mae(a, a) == 0.0

    def test_mae_different(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        assert abs(calculate_mae(a, b) - 0.1) < 1e-7

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(calculate_cosine_similarity(a, a) - 1.0) < 1e-7

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(calculate_cosine_similarity(a, b)) < 1e-7

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert calculate_cosine_similarity(a, b) == 0.0

    def test_max_abs_error(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.5, 3.0])
        assert abs(calculate_max_abs_error(a, b) - 0.5) < 1e-7

    def test_rmse_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(a, a) == 0.0

    def test_rmse_different(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert abs(calculate_rmse(a, b) - 1.0) < 1e-7


class TestValidateModels:
    def test_validate_torchscript_vs_onnx(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)

        model_paths = {
            "torchscript": ts_path,
            "onnx": onnx_path,
        }

        result = validate_models(
            model_paths,
            input_shape=cfg.model.input_shape,
            tolerance_mae=1e-3,
            tolerance_cos_sim=0.99,
        )
        assert result is True

    def test_validate_with_pytorch(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)

        model_paths = {
            "pytorch": cfg.model.path,
            "torchscript": ts_path,
            "onnx": onnx_path,
        }

        result = validate_models(
            model_paths,
            input_shape=cfg.model.input_shape,
            tolerance_mae=1e-3,
            tolerance_cos_sim=0.99,
            model_source_path=cfg.model.path,
            model_class_name=cfg.model.class_name,
        )
        assert result is True

    def test_validate_insufficient_models(self):
        result = validate_models(
            {"torchscript": "nonexistent.pt"},
            input_shape=[1, 3, 32, 32],
        )
        assert result is False

    def test_validate_multiple_tests(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)

        result = validate_models(
            {"torchscript": ts_path, "onnx": onnx_path},
            input_shape=cfg.model.input_shape,
            tolerance_mae=1e-3,
            tolerance_cos_sim=0.99,
            num_tests=3,
        )
        assert result is True
