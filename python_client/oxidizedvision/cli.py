import typer
from . import convert as convert_module
from . import validate as validate_module
from . import config as config_module
from . import benchmark as benchmark_module
import yaml
import os
import shutil
import subprocess
import json
from rich.console import Console
from rich.table import Table

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
def validate(
    config_path: str,
    tolerance_mae: float = 1e-5,
    tolerance_cos_sim: float = 0.999,
):
    """
    Validate outputs of TorchScript and ONNX models based on a config file.
    """
    console = Console()
    console.print(f"🔎 Validating models from [bold cyan]{config_path}[/bold cyan]...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_paths = {}
    output_dir = config.get("output_dir", "models")
    model_name = config.get("model_name", "model")

    # Gather model paths
    ts_path = os.path.join(output_dir, f"{model_name}.pt")
    if os.path.exists(ts_path):
        model_paths["torchscript"] = ts_path

    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    if os.path.exists(onnx_path):
        model_paths["onnx"] = onnx_path

    if len(model_paths) < 2:
        console.print("[red]Could not find at least two models (TorchScript and ONNX) to compare.[/red]")
        return

    validate_module.validate_models(
        model_paths,
        tolerance_mae=tolerance_mae,
        tolerance_cos_sim=tolerance_cos_sim,
    )
    console.print("✅ Validation finished.")

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
def benchmark(
    model_path: str,
    runners: str,
    iters: int = 100,
    batch_size: int = 1,
    output_format: str = "table",
):
    """
    Benchmark model performance across different runners.
    
    --runners: Comma-separated list of runners (e.g., torchscript,tract)
    --output-format: Output format (table, json)
    """
    console = Console()
    console.print(f"🚀 Starting benchmark for [bold cyan]{model_path}[/bold cyan]...")
    
    runner_list = [r.strip() for r in runners.split(',')]
    
    results = benchmark_module.run_benchmarks(
        model_path=model_path,
        runners=runner_list,
        iters=iters,
        batch_size=batch_size
    )
    
    if output_format == "json":
        console.print(json.dumps(results, indent=2))
    else:
        table = Table(title="Benchmark Results")
        table.add_column("Runner", justify="right", style="cyan", no_wrap=True)
        table.add_column("Avg Latency (ms)", style="magenta")
        table.add_column("Throughput (img/s)", style="green")
        table.add_column("Memory (MB)", style="yellow")

        for result in results:
            table.add_row(
                result["runner"],
                str(result["avg_latency_ms"]),
                str(result["throughput_images_per_sec"]),
                str(result["memory_usage_mb"]),
            )
        console.print(table)
        
    console.print("✅ Benchmark finished.")

@app.command()
def serve(model: str, port: int = 8080):
    """
    Start the example server (by calling the Rust binary).
    """
    print(f"Serving model {model} on port {port}...")
    subprocess.run([model, "--port", str(port)])
