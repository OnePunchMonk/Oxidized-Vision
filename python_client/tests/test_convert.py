"""Tests for the conversion module."""

import pytest
import os
import torch
from oxidizedvision.convert import convert_model, load_model, _import_model_from_path
from oxidizedvision.config import Config


class TestImportModel:
    def test_import_valid_model(self, tmp_model_dir):
        model_class = _import_model_from_path(
            str(tmp_model_dir / "model.py"), "SimpleModel"
        )
        assert model_class is not None
        model = model_class()
        assert isinstance(model, torch.nn.Module)

    def test_import_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _import_model_from_path("nonexistent.py", "MyModel")

    def test_import_nonexistent_class(self, tmp_model_dir):
        with pytest.raises(ImportError, match="not found"):
            _import_model_from_path(
                str(tmp_model_dir / "model.py"), "NonExistentClass"
            )


class TestConvertModel:
    def test_full_conversion(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)

        assert os.path.exists(ts_path)
        assert os.path.exists(onnx_path)
        assert ts_path.endswith(".pt")
        assert onnx_path.endswith(".onnx")

    def test_torchscript_output_is_valid(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, _ = convert_model(cfg)

        model = torch.jit.load(ts_path)
        dummy = torch.randn(1, 3, 32, 32)
        output = model(dummy)
        assert output.shape == (1, 10)

    def test_onnx_output_is_valid(self, simple_config):
        cfg = Config(**simple_config)
        _, onnx_path = convert_model(cfg)

        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
        output = session.run(None, {input_name: dummy})
        assert output[0].shape == (1, 10)

    def test_output_dir_is_created(self, simple_config):
        simple_config["export"]["output_dir"] = os.path.join(
            simple_config["export"]["output_dir"], "nested", "deep"
        )
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)
        assert os.path.exists(ts_path)

    def test_conversion_with_dict(self, simple_config):
        """Backward compatibility: passing a raw dict."""
        ts_path, onnx_path = convert_model(simple_config)
        assert os.path.exists(ts_path)
        assert os.path.exists(onnx_path)


class TestLoadModel:
    def test_load_model(self, simple_config):
        cfg = Config(**simple_config)
        model = load_model(cfg)
        assert isinstance(model, torch.nn.Module)
        assert not model.training  # should be in eval mode
