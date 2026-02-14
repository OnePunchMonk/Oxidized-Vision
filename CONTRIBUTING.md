# Contributing to OxidizedVision

Thank you for your interest in contributing to OxidizedVision! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code contributions.

## 🛠️ Development Setup

### Prerequisites

- **Python 3.9+** with `pip`
- **Rust stable** (latest) with `cargo`
- **Git**

### Setting Up the Python Environment

```bash
# Clone the repository
git clone https://github.com/OnePunchMonk/Oxidized-Vision.git
cd Oxidized-Vision

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install in development mode with dev dependencies
pip install -e "./python_client[dev]"
```

### Setting Up the Rust Environment

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the workspace
cargo build --workspace

# Run Rust tests
cargo test --workspace
```

## 📁 Project Structure

```
Oxidized-Vision/
├── python_client/         # Python CLI & pipeline
│   ├── oxidizedvision/    # Main package
│   │   ├── cli.py         # CLI entry point (typer)
│   │   ├── config.py      # Pydantic configuration models
│   │   ├── convert.py     # PyTorch → TorchScript/ONNX
│   │   ├── validate.py    # Numerical consistency checks
│   │   ├── benchmark.py   # Performance measurement
│   │   ├── optimize.py    # ONNX optimization & quantization
│   │   ├── profile.py     # Model profiling (params, FLOPs)
│   │   └── registry.py    # Local model registry
│   └── tests/             # pytest test suite
├── rust_runtime/          # Rust inference runtimes
│   ├── crates/
│   │   ├── runner_core/   # Shared Runner trait
│   │   ├── runner_tch/    # LibTorch (TorchScript) backend
│   │   ├── runner_tract/  # tract (ONNX) backend
│   │   └── runner_tensorrt/  # TensorRT (GPU) backend
│   └── examples/
│       ├── image_server/  # REST API server
│       ├── denoiser_cli/  # Image processing CLI
│       └── wasm_frontend/ # Browser inference demo
├── tools/                 # Standalone utility scripts
├── benchmarks/            # Benchmark infrastructure
├── examples/              # User-facing examples
└── docs/                  # Documentation
```

## 🧪 Running Tests

### Python Tests

```bash
# Run all tests
pytest python_client/tests/ -v

# Run with coverage
pytest python_client/tests/ -v --cov=oxidizedvision --cov-report=html

# Run a specific test file
pytest python_client/tests/test_convert.py -v
```

### Rust Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p runner_core
```

## 📝 Code Style

### Python

We use `black` for formatting, `isort` for import sorting, and `ruff` for linting:

```bash
black python_client/
isort python_client/
ruff check python_client/ --fix
```

### Rust

We use `rustfmt` and `clippy`:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

## 🔀 Making a Pull Request

1. **Fork** the repository and create a feature branch.
2. **Write tests** for any new functionality.
3. **Run the full test suite** and ensure all tests pass.
4. **Format your code** using the tools described above.
5. **Write clear commit messages** using [conventional commits](https://www.conventionalcommits.org/):
   - `feat: add quantization support to optimize module`
   - `fix: correct cosine similarity calculation for zero vectors`
   - `docs: update quickstart guide`
   - `test: add validation edge case tests`
6. **Submit the PR** against the `main` branch with a clear description.

## 🐛 Reporting Issues

When reporting bugs, please include:

- **Python/Rust version** and OS
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages / stack traces**
- Your **config.yml** (if applicable)

## 🤝 Code of Conduct

Be respectful, constructive, and welcoming. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
