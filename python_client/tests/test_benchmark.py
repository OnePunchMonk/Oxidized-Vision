"""Tests for the benchmark module."""

import pytest
import os
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from oxidizedvision.benchmark import (
    measure_performance,
    run_benchmarks,
    _measure,
)
from oxidizedvision.convert import convert_model
from oxidizedvision.config import Config


class TestMeasure:
    def test_measure_returns_all_keys(self):
        """Test that _measure returns all expected metric keys."""
        counter = {"n": 0}

        def dummy_fn():
            counter["n"] += 1

        result = _measure(dummy_fn, iters=10, warmup_iters=2)

        assert "avg_latency_ms" in result
        assert "p50_latency_ms" in result
        assert "p95_latency_ms" in result
        assert "p99_latency_ms" in result
        assert "std_latency_ms" in result
        assert "throughput_per_sec" in result
        assert "memory_delta_mb" in result
        assert "total_time_s" in result
        assert "iters" in result
        assert result["iters"] == 10

    def test_measure_latency_is_positive(self):
        def dummy_fn():
            pass

        result = _measure(dummy_fn, iters=5, warmup_iters=1)
        assert result["avg_latency_ms"] >= 0


class TestMeasurePerformance:
    def test_torchscript_benchmark(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, _ = convert_model(cfg)

        result = measure_performance(
            model_path=ts_path,
            runner="torchscript",
            iters=5,
            batch_size=1,
            input_shape=[1, 3, 32, 32],
        )
        assert result["runner"] == "torchscript"
        assert result["avg_latency_ms"] > 0
        assert result["throughput_per_sec"] > 0

    def test_onnx_benchmark(self, simple_config):
        cfg = Config(**simple_config)
        _, onnx_path = convert_model(cfg)

        result = measure_performance(
            model_path=onnx_path,
            runner="onnx",
            iters=5,
            batch_size=1,
            input_shape=[1, 3, 32, 32],
        )
        assert result["runner"] == "onnx"
        assert result["avg_latency_ms"] > 0

    def test_pytorch_benchmark(self, simple_config, tmp_model_dir):
        result = measure_performance(
            model_path="",
            runner="pytorch",
            iters=5,
            batch_size=1,
            input_shape=[1, 3, 32, 32],
            model_source_path=str(tmp_model_dir / "model.py"),
            model_class_name="SimpleModel",
        )
        assert result["runner"] == "pytorch"
        assert result["avg_latency_ms"] > 0

    def test_pytorch_runner_requires_source(self):
        with pytest.raises(ValueError, match="model_source_path"):
            measure_performance(
                model_path="",
                runner="pytorch",
                iters=5,
                batch_size=1,
            )

    def test_unknown_runner(self):
        with pytest.raises(ValueError, match="Unknown runner"):
            measure_performance(
                model_path="model.pt",
                runner="unknown",
                iters=5,
                batch_size=1,
            )


class TestRunBenchmarks:
    def test_run_multiple_runners(self, simple_config):
        cfg = Config(**simple_config)
        ts_path, onnx_path = convert_model(cfg)

        results = run_benchmarks(
            model_path=ts_path,
            runners=["torchscript", "tract"],
            iters=5,
            batch_size=1,
            input_shape=[1, 3, 32, 32],
        )
        assert len(results) == 2
        runner_names = [r["runner"] for r in results]
        assert "torchscript" in runner_names
        assert "tract" in runner_names
