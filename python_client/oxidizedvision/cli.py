import typer
from . import convert as convert_module
from . import validate as validate_module
from . import config as config_module
import yaml
import os
import shutil
import subprocess

app = typer.Typer()

@app.command()
def convert(config_path: str):
    """
    Run the full conversion pipeline.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting conversion process...")
    convert_module.convert_model(config)
    print("Conversion process finished.")

@app.command()
def validate(torchscript: str, onnx: str):
    """
    Validate outputs of TorchScript and ONNX models.
    """
    print(f"Validating {torchscript} and {onnx}...")
    validate_module.validate_models(torchscript, onnx)
    print("Validation finished.")

@app.command()
def package(onnx: str, runner: str, out: str):
    """
    Package the ONNX model into a Rust crate.
    """
    print(f"Packaging {onnx} with runner {runner} to {out}...")
    os.makedirs(out, exist_ok=True)
    
    # Copy the model into the new crate
    shutil.copy(onnx, os.path.join(out, "model.onnx"))

    # Create a minimal Cargo.toml
    cargo_toml = f"""
[package]
name = "{os.path.basename(out)}"
version = "0.1.0"
edition = "2021"

[dependencies]
runner_{runner} = {{ path = "../../crates/runner_{runner}" }}
ndarray = "0.15"
anyhow = "1.0"
image = "0.24"
clap = {{ version = "3.1", features = ["derive"] }}
actix-web = "4"
serde = {{ version = "1.0", features = ["derive"] }}
std = "1.0"
"""
    with open(os.path.join(out, "Cargo.toml"), "w") as f:
        f.write(cargo_toml)

    # Create a main.rs that is a copy of the image_server example
    server_main_rs_path = os.path.join(os.path.dirname(__file__), "../../../rust_runtime/examples/image_server/main.rs")
    os.makedirs(os.path.join(out, "src"), exist_ok=True)
    shutil.copy(server_main_rs_path, os.path.join(out, "src/main.rs"))
    
    print(f"Rust crate created at {out}")
    print("To build, run `cargo build --release` in that directory.")


@app.command()
def serve(model: str, port: int = 8080):
    """
    Start the example server (by calling the Rust binary).
    """
    print(f"Serving model {model} on port {port}...")
    subprocess.run([model, "--port", str(port)])
