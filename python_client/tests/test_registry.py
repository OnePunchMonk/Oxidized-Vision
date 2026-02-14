"""Tests for the model registry module."""

import pytest
import json
from oxidizedvision.registry import (
    register_model,
    list_models,
    get_model_info,
    remove_model,
    _load_registry,
    _save_registry,
)


class TestRegistry:
    def test_register_and_list(self, tmp_path):
        register_model(
            "test_model",
            {"torchscript": "out/model.pt", "onnx": "out/model.onnx"},
            base_dir=str(tmp_path),
        )
        models = list_models(str(tmp_path))
        assert len(models) == 1
        assert models[0]["name"] == "test_model"

    def test_get_model_info(self, tmp_path):
        register_model(
            "my_model",
            {"torchscript": "out/model.pt"},
            config={"model": {"class_name": "UNet"}},
            base_dir=str(tmp_path),
        )
        info = get_model_info("my_model", str(tmp_path))
        assert info is not None
        assert info["name"] == "my_model"
        assert "torchscript" in info["paths"]

    def test_get_nonexistent_model(self, tmp_path):
        info = get_model_info("nonexistent", str(tmp_path))
        assert info is None

    def test_remove_model(self, tmp_path):
        register_model("to_remove", {"onnx": "model.onnx"}, base_dir=str(tmp_path))
        assert remove_model("to_remove", str(tmp_path)) is True
        assert get_model_info("to_remove", str(tmp_path)) is None

    def test_remove_nonexistent(self, tmp_path):
        assert remove_model("nonexistent", str(tmp_path)) is False

    def test_multiple_models(self, tmp_path):
        for i in range(5):
            register_model(
                f"model_{i}",
                {"onnx": f"out/model_{i}.onnx"},
                base_dir=str(tmp_path),
            )
        models = list_models(str(tmp_path))
        assert len(models) == 5

    def test_registry_persistence(self, tmp_path):
        register_model("persistent", {"onnx": "m.onnx"}, base_dir=str(tmp_path))

        # Load independently
        registry = _load_registry(str(tmp_path))
        assert "persistent" in registry["models"]
