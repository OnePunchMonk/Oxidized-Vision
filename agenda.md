# OxidizedVision — Agent Agenda

> **Generated**: 2026-02-14
> **Last Updated**: 2026-02-14
> **Scope**: Full codebase audit — stubs, placeholders, missing features, and improvements.
> **Goal**: Make OxidizedVision a **production-grade, comprehensive** PyTorch → Rust inference toolkit.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🟡 | Partially implemented / has stubs |
| ✅ | Complete |

---

## Sprint 1 — Core Architecture & Shared Abstractions ✅

These are foundational items that many other tasks depend on.

### 1.1 ✅ Define a shared `Runner` trait crate (`runner_core`)

- [x] Created `rust_runtime/crates/runner_core/` with shared trait.
- [x] Defined `Runner` trait with `from_config()`, `run()`, `info()` using `ArrayD<f32>`.
- [x] Added `ModelInfo`, `RunnerConfig`, `Send + Sync`, helper functions, unit tests.
- [x] `runner_tch`, `runner_tract`, and `runner_tensorrt` all implement the trait.
- [x] Added to workspace in both `Cargo.toml` files.
- [x] Updated all examples to use the trait.

### 1.2 ✅ Support dynamic output tensor shapes in `runner_tch`

- [x] Now uses `ArrayD<f32>` / `IxDyn` instead of locked `Array4`.
- [x] Backward-compatible `run_4d` preserved.

### 1.3 ✅ Support dynamic input shapes in `runner_tract`

- [x] Accepts input shape via `RunnerConfig`.
- [x] No more hard-coded `[1, 3, 256, 256]`.

### 1.4 ✅ Add `load_from_bytes` to `runner_tract` for WASM support

- [x] Added `TractRunner::load_from_bytes()`.
- [x] WASM frontend now uses actual inference.

---

## Sprint 2 — Python Client: Stubs & Placeholders ✅

### 2.1 ✅ Implement PyTorch runner in benchmark module

- [x] Accepts model source path + class name.
- [x] Dynamically imports and instantiates the model (reuses `_import_model_from_path`).
- [x] Full warmup + benchmarked iterations with CUDA sync support.
- [x] Added percentile latencies (p50, p95, p99).

### 2.2 ✅ Implement PyTorch validation path

- [x] Dynamically imports model class using config.
- [x] Runs forward pass and captures output for comparison.
- [x] Added RMSE and Max Absolute Error metrics.

### 2.3 ✅ Hard-coded input shape `[1, 3, 256, 256]` — FIXED

- [x] Input shape read from config / CLI args everywhere.
- [x] All hard-coded assumptions removed.

### 2.4 ✅ Hard-coded output directory `"out/"` in conversion — FIXED

- [x] Reads `output_dir` and `model_name` from config.
- [x] CLI `convert` and `validate` are now consistent.

### 2.5 ✅ Add `optimize` CLI command

- [x] Created `optimize.py` module with graph simplification, INT8/FP16 quantization, constant folding.
- [x] Added `optimize` command to CLI.
- [x] Reports size reduction.

### 2.6 ✅ Improve `serve` CLI command

- [x] Auto-build Rust server binary via `cargo build --release`.
- [x] Informative error messages if binary doesn't exist.
- [x] Support passing model path and runner type.

---

## Sprint 3 — Rust: TensorRT Runner & Missing Implementations ✅

### 3.1 ✅ Implement TensorRT runner

- [x] Uses `trtexec` subprocess approach (no FFI needed).
- [x] Implements `Runner` trait.
- [x] Builds engines from ONNX, handles inference via file I/O.
- [x] Added to workspace.

### 3.2 ✅ Complete denoiser_cli image pre/post-processing

- [x] Image resizing, normalization, HWC→CHW conversion.
- [x] Tensor→image conversion, denormalization, clamp, CHW→HWC, save.
- [x] CLI args for dimensions and normalization parameters.

### 3.3 ✅ Complete WASM frontend example

- [x] Uses `TractRunner::load_from_bytes` for inference.
- [x] Model upload, input shape config, styled results display.
- [x] `run_inference_wasm` and `get_model_info` implemented.

### 3.4 ✅ Fix `image_server` inference endpoint

- [x] Uses `Arc<TractRunner>` instead of Mutex.
- [x] Accepts JSON data with configurable shape.
- [x] `/health` and `/predict` endpoints.
- [x] Structured JSON responses and error handling.

---

## Sprint 4 — Tools: Placeholder Scripts ✅

### 4.1 ✅ Implement `tools/convert.py`

- [x] Standalone conversion script wrapping `oxidizedvision.convert`.
- [x] Accepts `--config` CLI arg.

### 4.2 ✅ Implement `tools/optimize_onnx.py`

- [x] Standalone optimization script wrapping `oxidizedvision.optimize`.
- [x] Accepts `--input`, `--output`, `--quantize`, `--no-simplify` flags.

---

## Sprint 5 — Testing & CI ✅

### 5.1 ✅ Fix and expand Python tests

- [x] Created `conftest.py` with proper fixtures (tmp_model_dir, simple_config, YAML config).
- [x] `test_convert.py` — full pipeline, TorchScript/ONNX validation, backward compat.
- [x] `test_validate.py` — all metrics, edge cases, multi-format validation.
- [x] `test_config.py` — Pydantic validation, defaults, load/save round-trip.
- [x] `test_benchmark.py` — measurement mechanics, all runner types, error handling.
- [x] `test_cli.py` — CLI integration tests with CliRunner.
- [x] `test_profile.py` — parameter counting, size estimation, layer summary.
- [x] `test_registry.py` — register, list, get, remove, persistence.

### 5.2 ✅ Add Rust unit tests

- [x] `runner_core`: Tests for serialization, default config, shape conversion.
- [x] `runner_tch`: Tests for config and error handling.
- [x] `runner_tract`: Tests for config and error handling.
- [x] `runner_tensorrt`: Tests for config and trtexec detection.

### 5.3 ✅ Make CI actually run tests

- [x] Python: `pytest` with coverage + `codecov`.
- [x] Multi-version matrix (3.9, 3.10, 3.11, 3.12).
- [x] Linting: `black`, `isort`, `ruff`.
- [x] Updated to `actions/checkout@v4`, `actions/setup-python@v5`, `dtolnay/rust-toolchain`.
- [x] Rust: `rustfmt`, `clippy`, build, test.

### 5.4 ✅ Make benchmarks CI actually functional

- [x] Converts example model as setup step.
- [x] Runs benchmarks with JSON output.
- [x] Stores results as CI artifacts.

---

## Sprint 6 — Config & Packaging Improvements ✅

### 6.1 ✅ Validate and unify configuration schema

- [x] All CLI commands use unified `Config` object via `load_config()`.
- [x] Added `OptimizeConfig`, `BenchmarkConfig` sub-models.
- [x] Validate/benchmark configs now propagated properly.
- [x] Input validation with helpful error messages.

### 6.2 ✅ Improve packaging (`package` command)

- [x] Removed erroneous `std = "1.0"` dependency.
- [x] `--template server|cli` support.
- [x] Proper `main.rs` generation from templates.
- [x] `.gitignore` and `README.md` generated.
- [x] ONNX existence validation.
- [x] `--runner tch|tract|tensorrt` support.

### 6.3 ✅ Fix `pyproject.toml` / packaging discrepancy

- [x] Removed `click` from dependencies.
- [x] Added `psutil` and `rich` to dependencies.
- [x] Updated `requires-python` to `>=3.9`.
- [x] Added `[dev]` dependencies group.
- [x] Added tool configs for `black`, `isort`, `ruff`, `pytest`, `mypy`.

---

## Sprint 7 — New Features (Roadmap Items)

### 7.1 ✅ GPU benchmarking support

- [x] `--device cpu|cuda` flag in benchmark CLI.
- [x] `torch.cuda.synchronize()` for accurate GPU timing.
- [x] GPU memory delta reporting.

### 7.2 ✅ Model profiling / summary command

- [x] `profile` CLI command with parameter count, size, per-layer breakdown.
- [x] Output in table format (rich) and JSON.

### 7.3 ✅ Dynamic batching support in Rust server

- [x] Implement request batching in `image_server` via `DynamicBatcher` struct.
- [x] Add configurable `--max-batch-size` and `--max-wait-ms` CLI flags.

### 7.4 ✅ Model versioning and registry

- [x] Local model registry tracking converted models and metadata.
- [x] `oxidizedvision list` to see all models.
- [x] `oxidizedvision info <model>` to inspect metadata.

### 7.5 ✅ Logging and observability

- [x] Added `tracing` to all Rust crates (`runner_core`, `runner_tch`, `runner_tract`, `runner_tensorrt`).
- [x] Added `tracing-actix-web` for per-request tracing in the Rust server.
- [x] Added `/metrics` endpoint for server observability.
- [x] Added Python `logging` module (`oxidizedvision.logging`) with Rich and JSON formatters.
- [x] Added `--verbose` / `--json-log` global CLI options.

### 7.6 ✅ Multi-model support

- [x] Allow Rust server to serve multiple models via `--model name=path` (repeatable).
- [x] Route requests by name: `POST /predict/{model_name}`.
- [x] Added `GET /models` endpoint to list loaded models.
- [ ] Support model hot-reloading (future).

---

## Sprint 8 — Documentation & Developer Experience ✅

### 8.1 ✅ Expand `CONTRIBUTING.md`

- [x] Development setup instructions (Python + Rust).
- [x] Project structure documentation.
- [x] Testing instructions.
- [x] PR review process.
- [x] Code of conduct.

### 8.2 ✅ Add API reference documentation

- [x] Docstrings on all Python functions and classes.
- [x] `rustdoc` comments on Rust public items.

### 8.3 ✅ Add more example models

- [x] Real U-Net implementation (encoder-decoder with skip connections, BatchNorm, bilinear upsampling).
- [x] Updated example config to match unified schema.

### 8.4 ✅ Update `CODEOWNERS`

- [x] Replaced `@github_username` with actual `@OnePunchMonk`.

### 8.5 ✅ Expand architecture documentation

- [x] Mermaid diagrams for pipeline and sequence flows.
- [x] Full config schema reference.
- [x] Troubleshooting guide.

---

## Sprint 9 — Code Quality & Housekeeping ✅

### 9.1 ✅ Add `__init__.py` exports

- [x] Exports `__version__`, key classes and functions.
- [x] Package-level docstring.

### 9.2 ✅ Add type annotations throughout Python code

- [x] Complete type annotations in all modules.
- [x] `mypy` added to CI.

### 9.3 ✅ Error handling improvements

- [x] `rich.console.Console` used consistently.
- [x] Proper `typer.Exit(code=1)` exit codes.
- [x] Input validation before expensive operations.
- [x] All CLI commands wrapped in exception handlers.

### 9.4 ✅ Remove duplicate / dead code (resolved)

- [x] `tools/` scripts refactored to import from `oxidizedvision` package.
- [x] `benchmarks/run_benchmark.py` refactored to import from package.
- [x] `tools/export_to_onnx.py` updated with proper flags.

### 9.5 ✅ Add pre-commit hooks

- [x] Created `.pre-commit-config.yaml` (black, isort, ruff, mypy, rustfmt, clippy, general hooks).
- [x] Documented setup in `CONTRIBUTING.md`.

---

## Summary: Completion Status

| Sprint | Description | Status |
|--------|-------------|--------|
| 1 | Core Architecture & Shared Abstractions | ✅ Complete |
| 2 | Python Client: Stubs & Placeholders | ✅ Complete |
| 3 | Rust: TensorRT Runner & Missing Implementations | ✅ Complete |
| 4 | Tools: Placeholder Scripts | ✅ Complete |
| 5 | Testing & CI | ✅ Complete |
| 6 | Config & Packaging Improvements | ✅ Complete |
| 7 | New Features (Roadmap Items) | ✅ Complete (5/6; hot-reload deferred) |
| 8 | Documentation & Developer Experience | ✅ Complete |
| 9 | Code Quality & Housekeeping | ✅ Complete |

### Remaining Items (Future Work)

- **Sprint 7.6**: Model hot-reloading (partial — multi-model serving is done, reload is deferred)
