"""Tests for the configuration module."""

import pytest
import yaml
import os
from oxidizedvision.config import (
    Config,
    ModelConfig,
    ExportConfig,
    ValidateConfig,
    OptimizeConfig,
    BenchmarkConfig,
    load_config,
    save_config,
)


class TestModelConfig:
    def test_valid_model_config(self):
        cfg = ModelConfig(
            path="model.py",
            class_name="MyModel",
            input_shape=[1, 3, 256, 256],
        )
        assert cfg.path == "model.py"
        assert cfg.class_name == "MyModel"
        assert cfg.input_shape == [1, 3, 256, 256]
        assert cfg.checkpoint is None

    def test_invalid_input_shape_too_short(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            ModelConfig(path="model.py", class_name="M", input_shape=[1])

    def test_invalid_input_shape_negative(self):
        with pytest.raises(ValueError, match="positive"):
            ModelConfig(path="model.py", class_name="M", input_shape=[1, -3, 256, 256])

    def test_with_checkpoint(self):
        cfg = ModelConfig(
            path="model.py",
            class_name="MyModel",
            input_shape=[1, 3, 256, 256],
            checkpoint="weights.pt",
        )
        assert cfg.checkpoint == "weights.pt"


class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.opset_version == 14
        assert cfg.do_constant_folding is True
        assert cfg.output_dir == "out"
        assert cfg.model_name == "model"

    def test_custom_values(self):
        cfg = ExportConfig(
            opset_version=11,
            output_dir="custom_dir",
            model_name="my_model",
        )
        assert cfg.opset_version == 11
        assert cfg.output_dir == "custom_dir"


class TestConfig:
    def test_minimal_config(self):
        cfg = Config(
            model=ModelConfig(
                path="model.py",
                class_name="MyModel",
                input_shape=[1, 3, 256, 256],
            )
        )
        # Defaults should be populated
        assert cfg.export.opset_version == 14
        assert cfg.validation.tolerance_mae == 1e-5
        assert cfg.optimize.simplify is True
        assert cfg.benchmark.iters == 100

    def test_full_config(self):
        cfg = Config(
            model=ModelConfig(
                path="model.py",
                class_name="MyModel",
                input_shape=[1, 3, 256, 256],
            ),
            export=ExportConfig(opset_version=11),
            validation=ValidateConfig(num_tests=5, tolerance_mae=1e-4),
        )
        assert cfg.export.opset_version == 11
        assert cfg.validation.num_tests == 5


class TestLoadSaveConfig:
    def test_load_config(self, tmp_path):
        config_data = {
            "model": {
                "path": "model.py",
                "class_name": "MyModel",
                "input_shape": [1, 3, 256, 256],
            },
            "export": {
                "opset_version": 14,
                "output_dir": "out",
            },
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(str(config_path))
        assert isinstance(cfg, Config)
        assert cfg.model.class_name == "MyModel"

    def test_load_nonexistent_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yml")

    def test_save_and_reload(self, tmp_path):
        cfg = Config(
            model=ModelConfig(
                path="model.py",
                class_name="MyModel",
                input_shape=[1, 3, 256, 256],
            )
        )
        config_path = str(tmp_path / "saved_config.yml")
        save_config(cfg, config_path)

        assert os.path.exists(config_path)

        loaded = load_config(config_path)
        assert loaded.model.class_name == cfg.model.class_name
        assert loaded.model.input_shape == cfg.model.input_shape
