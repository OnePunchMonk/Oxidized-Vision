---
description: Agent objectives, agenda, and working principles for OxidizedVision
---

# OxidizedVision — Agent Agenda & Objectives

> **Project**: OxidizedVision — PyTorch → Rust inference toolkit
> **Repository**: https://github.com/OnePunchMonk/Oxidized-Vision
> **Author**: Avaya Aggarwal

---

## 🎯 Mission Statement

OxidizedVision bridges the gap between Python-based model training and Rust-based deployment.
The agent's overarching goal is to make this a **production-grade, comprehensive** toolkit with a seamless
pipeline: **Convert → Optimize → Validate → Benchmark → Profile → Package → Deploy**.

---

## 🧭 Core Objectives

### 1. Maintain & Extend the Python Client (`python_client/`)

The Python CLI is the user-facing entry point. Keep it polished and feature-complete.

- **CLI (`cli.py`)**: Typer-based; commands: `convert`, `validate`, `benchmark`, `optimize`, `profile`, `package`, `serve`, `list`, `info`.
- **Config (`config.py`)**: Pydantic models for YAML config with strong validation.
- **Conversion (`convert.py`)**: PyTorch → TorchScript + ONNX with dynamic axes.
- **Validation (`validate.py`)**: Cross-format comparison (MAE, RMSE, Max Error, Cosine Similarity).
- **Benchmarking (`benchmark.py`)**: Latency percentiles, throughput, memory profiling, CPU/CUDA.
- **Optimization (`optimize.py`)**: ONNX graph simplification, constant folding, INT8/FP16 quantization.
- **Profiling (`profile.py`)**: Parameter count, model size, per-layer breakdown.
- **Registry (`registry.py`)**: Local JSON-based model tracking and metadata.

### 2. Maintain & Extend the Rust Runtimes (`rust_runtime/`)

All Rust backends implement the shared `Runner` trait from `runner_core`.

| Crate | Backend | Format | GPU | WASM |
|---|---|---|---|---|
| `runner_tract` | tract (pure Rust) | ONNX | ❌ | ✅ |
| `runner_tch` | tch-rs (LibTorch) | TorchScript | ✅ | ❌ |
| `runner_tensorrt` | TensorRT (subprocess) | ONNX → Engine | ✅ | ❌ |

- Always implement changes via the `Runner` trait to keep backends interchangeable.
- Keep `runner_core` lean; only shared types and the trait belong there.
- Maintain `Send + Sync` bounds for async server compatibility.

### 3. Ensure Quality via Tests & CI

- **Python**: `pytest` suite in `python_client/tests/` covering convert, validate, benchmark, config, CLI, profile, registry.
- **Rust**: Unit tests in each crate for config, serialization, error handling.
- **CI**: GitHub Actions for Python (multi-version 3.9–3.12, linting, coverage) and Rust (fmt, clippy, build, test).

### 4. Keep Documentation Accurate

- `README.md` — high-level overview and quickstart.
- `docs/architecture.md` — deep-dive architecture, diagrams, config schema, troubleshooting.
- `CONTRIBUTING.md` — dev setup, project structure, testing, PR process.
- Inline docstrings (Python) and `rustdoc` comments (Rust) on all public items.

---

## 📋 Current Completion Status

| Sprint | Area | Status |
|---|---|---|
| 1 | Core Architecture & Shared Abstractions | ✅ Complete |
| 2 | Python Client: Stubs & Placeholders | ✅ Complete |
| 3 | Rust: TensorRT Runner & Missing Implementations | ✅ Complete |
| 4 | Tools: Placeholder Scripts | ✅ Complete |
| 5 | Testing & CI | ✅ Complete |
| 6 | Config & Packaging Improvements | ✅ Complete |
| 7 | New Features (Roadmap) | ✅ Complete (hot-reload deferred) |
| 8 | Documentation & Developer Experience | ✅ Complete |
| 9 | Code Quality & Housekeeping | ✅ Complete |

---

## 🔴 Remaining Work (Priority Order)

### P0 — High Priority (DONE ✅)

1. ~~**Dynamic Batching in Rust Server** (Sprint 7.3)~~ ✅
   - `DynamicBatcher` struct with configurable `--max-batch-size` and `--max-wait-ms`.

2. ~~**Structured Logging & Observability** (Sprint 7.5)~~ ✅
   - `tracing` in all Rust crates, `tracing-actix-web` for request tracing.
   - `/metrics` endpoint, Python `logging` module with Rich/JSON output.
   - `--verbose` / `--json-log` global CLI options.

### P1 — Medium Priority (DONE ✅)

3. ~~**Multi-Model Support** (Sprint 7.6)~~ ✅
   - `--model name=path` (repeatable), `POST /predict/{name}`, `GET /models`.
   - Hot-reloading deferred to future work.

4. ~~**Pre-commit Hooks** (Sprint 9.5)~~ ✅
   - `.pre-commit-config.yaml` with full Python + Rust + general hooks.
   - Documented in `CONTRIBUTING.md`.

### P2 — Stretch Goals

5. **ONNX Runtime backend** — Add `runner_ort` using the `ort` crate for broader hardware acceleration.
6. **Model compression** — Pruning and knowledge distillation tools in the Python client.
7. **Docker images** — Pre-built images for easy deployment with each runtime backend.
8. **Benchmark dashboard** — CI-driven historical performance tracking with visualization.

---

## 🛠️ Working Principles

### When Modifying Python Code

1. Use Pydantic models for any new config structures.
2. Use `rich` for all terminal output (console, tables, progress bars).
3. Use `typer` for any new CLI commands; register them in `cli.py`.
4. Add type annotations on every function signature.
5. Write `pytest` tests for new functionality in `python_client/tests/`.
6. Reuse helpers like `_import_model_from_path()` from `convert.py`.

### When Modifying Rust Code

1. New backends must implement the `Runner` trait from `runner_core`.
2. Use `anyhow::Result` for error propagation.
3. Use `ndarray::ArrayD<f32>` for tensor I/O (not fixed-dimension arrays).
4. Pass `rustfmt` and `clippy` before committing.
5. Add unit tests for config parsing and error paths at minimum.

### When Modifying CI / Build

1. Python CI matrix: 3.9, 3.10, 3.11, 3.12.
2. Keep actions pinned to latest stable versions (`@v4`, `@v5`).
3. Always run linting before tests.
4. Benchmark CI should store JSON results as artifacts.

### General Rules

- **Never hard-code** input shapes, output directories, or model paths. Everything comes from config.
- **Always validate inputs** before expensive operations (conversion, benchmarking).
- **Keep `tools/` scripts thin** — they should import from `oxidizedvision` package, not duplicate logic.
- **Update documentation** (`README.md`, `architecture.md`) whenever adding new features or commands.
- **Run the full test suite** (`pytest` + `cargo test`) before considering work complete.

---

## 📁 Key File Map

```
Oxidized-Vision/
├── .agent/workflows/agenda.md    ← This file
├── agenda.md                     ← Sprint tracker (detailed task list)
├── Cargo.toml                    ← Rust workspace root
├── pyproject.toml                ← Python package config
├── python_client/
│   ├── oxidizedvision/           ← Python source modules
│   └── tests/                    ← pytest suite
├── rust_runtime/
│   ├── crates/                   ← Rust crates (runner_core, runner_tch, runner_tract, runner_tensorrt)
│   └── examples/                 ← Example apps (image_server, denoiser_cli, wasm_frontend)
├── tools/                        ← Standalone scripts (convert.py, optimize_onnx.py, export_to_onnx.py)
├── examples/example_unet/        ← Reference UNet model + config
├── benchmarks/                   ← Benchmark infrastructure
├── docs/architecture.md          ← Architecture deep-dive
├── CONTRIBUTING.md               ← Contributor guide
└── .github/workflows/            ← CI/CD pipelines
```

---

## ✅ Verification Checklist

Before marking any task as complete, ensure:

- [ ] All new Python code has type annotations and docstrings.
- [ ] `pytest python_client/tests/ -v` passes.
- [ ] `cargo test --workspace` passes (if Rust was touched).
- [ ] `cargo clippy --workspace` reports no warnings (if Rust was touched).
- [ ] `black` + `isort` + `ruff` pass on Python code.
- [ ] Documentation is updated if any public APIs changed.
- [ ] The root `agenda.md` sprint tracker is updated with progress.
