"""
OxidizedVision — Command-Line Interface.

The main entry point for the OxidizedVision toolkit. All pipeline operations
(convert, validate, benchmark, optimize, package, profile, serve) are
accessible through this CLI.
"""

import typer
import yaml
import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from . import convert as convert_module
from . import validate as validate_module
from . import benchmark as benchmark_module
from . import optimize as optimize_module
from . import profile as profile_module
from . import registry as registry_module
from .config import load_config, Config
from .logging import configure_logging, get_logger

app = typer.Typer(
    name="oxidizedvision",
    help="🚀 OxidizedVision — Compile PyTorch models to Rust for ultra-fast inference.",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


# ──────────────────────────────── callback ────────────────────────────────


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    json_log: bool = typer.Option(False, "--json-log", help="Emit logs as JSON lines."),
):
    """Global options applied before any subcommand."""
    configure_logging(
        level="DEBUG" if verbose else "INFO",
        json_output=json_log,
        verbose=verbose,
    )


# ──────────────────────────────── convert ────────────────────────────────


@app.command()
def convert(
    config_path: str = typer.Argument(..., help="Path to the YAML config file."),
):
    """Convert a PyTorch model to TorchScript and ONNX formats."""
    try:
        cfg = load_config(config_path)
        logger.info("Starting conversion from %s", config_path)
        console.print(f"\n🚀 Starting conversion from [bold cyan]{config_path}[/bold cyan]")
        ts_path, onnx_path = convert_module.convert_model(cfg)

        # Register in model registry
        model_name = cfg.export.model_name
        registry_module.register_model(
            model_name,
            {"torchscript": ts_path, "onnx": onnx_path},
            config=cfg.dict(),
        )
        logger.info(
            "Conversion complete: ts=%s, onnx=%s", ts_path, onnx_path
        )

    except Exception as e:
        logger.error("Conversion failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Conversion failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── validate ────────────────────────────────


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to the YAML config file."),
    tolerance_mae: float = typer.Option(1e-5, help="Maximum MAE tolerance."),
    tolerance_cos_sim: float = typer.Option(0.999, help="Minimum cosine similarity tolerance."),
    num_tests: int = typer.Option(1, help="Number of random inputs to validate."),
):
    """Validate numerical consistency between TorchScript and ONNX outputs."""
    try:
        cfg = load_config(config_path)
        logger.info("Validating models from %s", config_path)
        console.print(f"\n🔎 Validating models from [bold cyan]{config_path}[/bold cyan]...")

        output_dir = cfg.export.output_dir
        model_name = cfg.export.model_name

        model_paths = {}
        ts_path = os.path.join(output_dir, f"{model_name}.pt")
        if os.path.exists(ts_path):
            model_paths["torchscript"] = ts_path

        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        if os.path.exists(onnx_path):
            model_paths["onnx"] = onnx_path

        # Include PyTorch source for direct comparison
        model_paths["pytorch"] = cfg.model.path

        if len(model_paths) < 2:
            console.print("[red]Could not find at least two models to compare.[/red]")
            raise typer.Exit(code=1)

        # Use config-level tolerances as defaults
        t_mae = tolerance_mae or cfg.validation.tolerance_mae
        t_cos = tolerance_cos_sim or cfg.validation.tolerance_cos_sim
        n_tests = num_tests or cfg.validation.num_tests

        passed = validate_module.validate_models(
            model_paths,
            input_shape=cfg.model.input_shape,
            tolerance_mae=t_mae,
            tolerance_cos_sim=t_cos,
            num_tests=n_tests,
            model_source_path=cfg.model.path,
            model_class_name=cfg.model.class_name,
            model_checkpoint=cfg.model.checkpoint,
        )

        if passed:
            logger.info("Validation passed")
        else:
            logger.warning("Validation failed")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Validation failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Validation failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── benchmark ────────────────────────────────


@app.command()
def benchmark(
    model_path: str = typer.Argument(..., help="Path to the model file (.pt or .onnx)."),
    runners: str = typer.Option("torchscript,tract", help="Comma-separated list of runners."),
    iters: int = typer.Option(100, help="Number of benchmark iterations."),
    batch_size: int = typer.Option(1, help="Batch size."),
    output_format: str = typer.Option("table", help="Output format: 'table' or 'json'."),
    device: str = typer.Option("cpu", help="Device: 'cpu' or 'cuda'."),
    input_shape: Optional[str] = typer.Option(None, help="Input shape as comma-separated dims (e.g., '1,3,256,256')."),
    model_source: Optional[str] = typer.Option(None, help="Model source .py file (for 'pytorch' runner)."),
    model_class: Optional[str] = typer.Option(None, help="Model class name (for 'pytorch' runner)."),
):
    """Benchmark model performance across different runners."""
    try:
        logger.info("Starting benchmark for %s (runners=%s, iters=%d)", model_path, runners, iters)
        console.print(f"\n🚀 Starting benchmark for [bold cyan]{model_path}[/bold cyan]...")

        runner_list = [r.strip() for r in runners.split(",")]

        shape = None
        if input_shape:
            shape = [int(d.strip()) for d in input_shape.split(",")]

        results = benchmark_module.run_benchmarks(
            model_path=model_path,
            runners=runner_list,
            iters=iters,
            batch_size=batch_size,
            input_shape=shape,
            device=device,
            model_source_path=model_source,
            model_class_name=model_class,
        )

        if output_format == "json":
            console.print(json.dumps(results, indent=2))
        else:
            table = Table(title="Benchmark Results")
            table.add_column("Runner", justify="right", style="cyan", no_wrap=True)
            table.add_column("Device", style="dim")
            table.add_column("Avg (ms)", style="magenta")
            table.add_column("p50 (ms)", style="blue")
            table.add_column("p95 (ms)", style="yellow")
            table.add_column("p99 (ms)", style="red")
            table.add_column("Throughput", style="green")
            table.add_column("Mem Δ (MB)", style="dim")

            for result in results:
                table.add_row(
                    result["runner"],
                    result.get("device", "cpu"),
                    str(result["avg_latency_ms"]),
                    str(result["p50_latency_ms"]),
                    str(result["p95_latency_ms"]),
                    str(result["p99_latency_ms"]),
                    f"{result['throughput_per_sec']}/s",
                    str(result["memory_delta_mb"]),
                )
            console.print(table)

        logger.info("Benchmark finished (%d results)", len(results))
        console.print("\n✅ Benchmark finished.")

    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Benchmark failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── optimize ────────────────────────────────


@app.command()
def optimize(
    input_path: str = typer.Argument(..., help="Path to the ONNX model to optimize."),
    output_path: Optional[str] = typer.Option(None, help="Output path. Defaults to '<input>_optimized.onnx'."),
    simplify: bool = typer.Option(True, help="Apply onnx-simplifier."),
    quantize: Optional[str] = typer.Option(None, help="Quantization mode: 'int8' or 'fp16'."),
    constant_folding: bool = typer.Option(True, help="Apply constant folding."),
):
    """Optimize an ONNX model (simplify, quantize, fold constants)."""
    try:
        logger.info("Optimizing %s (simplify=%s, quantize=%s)", input_path, simplify, quantize)
        optimize_module.optimize_model(
            input_path=input_path,
            output_path=output_path,
            simplify=simplify,
            quantize=quantize,
            constant_folding=constant_folding,
        )
        logger.info("Optimization complete")
    except Exception as e:
        logger.error("Optimization failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Optimization failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── profile ────────────────────────────────


@app.command()
def profile(
    config_path: str = typer.Argument(..., help="Path to the YAML config file."),
    output_format: str = typer.Option("table", help="Output format: 'table' or 'json'."),
):
    """Profile a PyTorch model: parameter count, size, and layer breakdown."""
    try:
        cfg = load_config(config_path)
        logger.info("Profiling model from %s", config_path)
        result = profile_module.profile_model(
            model_source_path=cfg.model.path,
            model_class_name=cfg.model.class_name,
            input_shape=cfg.model.input_shape,
            checkpoint=cfg.model.checkpoint,
        )

        if output_format == "json":
            console.print(json.dumps(result, indent=2, default=str))
        else:
            profile_module.print_profile(result)

        logger.info("Profiling complete")

    except Exception as e:
        logger.error("Profiling failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Profiling failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── package ────────────────────────────────


MAIN_RS_SERVER_TEMPLATE = '''\
use actix_web::{{post, get, web, App, HttpServer, Responder, HttpResponse}};
use serde::{{Deserialize, Serialize}};
use runner_core::{{Runner, RunnerConfig}};
use runner_{runner}::{runner_struct};
use ndarray::{{ArrayD, IxDyn}};
use std::sync::Arc;
use clap::Parser;

struct AppState {{
    runner: Arc<{runner_struct}>,
}}

#[derive(Parser, Debug)]
struct Args {{
    #[clap(short, long)]
    model: String,
    #[clap(short, long, default_value_t = 8080)]
    port: u16,
}}

#[derive(Deserialize)]
struct InferenceRequest {{
    data: Option<Vec<f32>>,
    shape: Option<Vec<usize>>,
}}

#[derive(Serialize)]
struct InferenceResponse {{
    status: String,
    output_shape: Vec<usize>,
}}

#[get("/health")]
async fn health() -> impl Responder {{
    HttpResponse::Ok().json(serde_json::json!({{"status": "healthy"}}))
}}

#[post("/predict")]
async fn predict(req: web::Json<InferenceRequest>, data: web::Data<AppState>) -> impl Responder {{
    let shape = req.shape.clone().unwrap_or(vec![{input_shape}]);
    let numel: usize = shape.iter().product();
    let input_data = req.data.clone().unwrap_or(vec![0.0f32; numel]);
    let input = ArrayD::<f32>::from_shape_vec(IxDyn(&shape), input_data).unwrap();
    match data.runner.run(&input) {{
        Ok(output) => HttpResponse::Ok().json(InferenceResponse {{
            status: "success".to_string(),
            output_shape: output.shape().to_vec(),
        }}),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error: {{}}", e)),
    }}
}}

#[actix_web::main]
async fn main() -> std::io::Result<()> {{
    let args = Args::parse();
    let config = RunnerConfig {{
        model_path: args.model.clone(),
        input_shape: vec![{input_shape}],
        ..RunnerConfig::default()
    }};
    let runner = {runner_struct}::from_config(&config).expect("Failed to load model");
    let app_state = web::Data::new(AppState {{ runner: Arc::new(runner) }});
    println!("Server starting at http://127.0.0.1:{{}}", args.port);
    HttpServer::new(move || App::new().app_data(app_state.clone()).service(health).service(predict))
        .bind(("127.0.0.1", args.port))?.run().await
}}
'''

MAIN_RS_CLI_TEMPLATE = '''\
use clap::Parser;
use runner_core::{{Runner, RunnerConfig}};
use runner_{runner}::{runner_struct};
use ndarray::{{ArrayD, IxDyn}};

#[derive(Parser, Debug)]
struct Args {{
    #[clap(short, long)]
    model: String,
    #[clap(short, long)]
    input: String,
    #[clap(short, long)]
    output: String,
}}

fn main() -> anyhow::Result<()> {{
    let args = Args::parse();
    let config = RunnerConfig {{
        model_path: args.model.clone(),
        input_shape: vec![{input_shape}],
        ..RunnerConfig::default()
    }};
    let runner = {runner_struct}::from_config(&config)?;
    let numel = vec![{input_shape}].iter().product::<usize>();
    let input = ArrayD::<f32>::zeros(IxDyn(&[{input_shape}]));
    let output = runner.run(&input)?;
    println!("Output shape: {{:?}}", output.shape());
    Ok(())
}}
'''


@app.command()
def package(
    onnx: str = typer.Argument(..., help="Path to the ONNX model file."),
    runner: str = typer.Option("tract", help="Runner backend: 'tract', 'tch', or 'tensorrt'."),
    out: str = typer.Option("./packaged", help="Output directory for the Rust crate."),
    template: str = typer.Option("server", help="Template: 'server' or 'cli'."),
    input_shape: str = typer.Option("1,3,256,256", help="Input shape."),
):
    """Package an ONNX model into a deployable Rust crate."""
    try:
        if not os.path.exists(onnx):
            console.print(f"[red]ONNX model not found: {onnx}[/red]")
            raise typer.Exit(code=1)

        logger.info("Packaging %s with runner=%s, template=%s", onnx, runner, template)
        console.print(f"\n📦 Packaging [bold cyan]{onnx}[/bold cyan] with runner [cyan]{runner}[/cyan]...")
        os.makedirs(out, exist_ok=True)

        # Copy the model
        shutil.copy(onnx, os.path.join(out, "model.onnx"))

        # Runner struct mapping
        runner_structs = {
            "tract": "TractRunner",
            "tch": "TchRunner",
            "tensorrt": "TensorRTRunner",
        }
        runner_struct = runner_structs.get(runner, "TractRunner")

        # Generate Cargo.toml
        deps = {
            "tract": 'runner_tract = { path = "../../crates/runner_tract" }',
            "tch": 'runner_tch = { path = "../../crates/runner_tch" }',
            "tensorrt": 'runner_tensorrt = { path = "../../crates/runner_tensorrt" }',
        }

        extra_deps = ""
        if template == "server":
            extra_deps = '\nactix-web = "4"\nserde_json = "1.0"'

        cargo_toml = f"""[package]
name = "{os.path.basename(out)}"
version = "0.1.0"
edition = "2021"

[dependencies]
runner_core = {{ path = "../../crates/runner_core" }}
{deps.get(runner, deps['tract'])}
ndarray = "0.15"
anyhow = "1.0"
clap = {{ version = "3.1", features = ["derive"] }}
serde = {{ version = "1.0", features = ["derive"] }}{extra_deps}
"""
        with open(os.path.join(out, "Cargo.toml"), "w") as f:
            f.write(cargo_toml)

        # Generate main.rs
        os.makedirs(os.path.join(out, "src"), exist_ok=True)
        shape_str = input_shape

        if template == "server":
            main_rs = MAIN_RS_SERVER_TEMPLATE.format(
                runner=runner,
                runner_struct=runner_struct,
                input_shape=shape_str,
            )
        else:
            main_rs = MAIN_RS_CLI_TEMPLATE.format(
                runner=runner,
                runner_struct=runner_struct,
                input_shape=shape_str,
            )

        with open(os.path.join(out, "src/main.rs"), "w") as f:
            f.write(main_rs)

        # Generate .gitignore
        with open(os.path.join(out, ".gitignore"), "w") as f:
            f.write("/target\n*.onnx\n")

        # Generate README
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(f"# {os.path.basename(out)}\n\n")
            f.write(f"Auto-generated by OxidizedVision.\n\n")
            f.write(f"## Build\n\n```bash\ncargo build --release\n```\n\n")
            if template == "server":
                f.write(f"## Run\n\n```bash\n./target/release/{os.path.basename(out)} --model model.onnx --port 8080\n```\n")
            else:
                f.write(f"## Run\n\n```bash\n./target/release/{os.path.basename(out)} --model model.onnx --input input.png --output output.png\n```\n")

        logger.info("Rust crate created at %s", out)
        console.print(f"\n✅ Rust crate created at [bold green]{out}[/bold green]")
        console.print(f"   To build: [dim]cd {out} && cargo build --release[/dim]")

    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Packaging failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Packaging failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── serve ────────────────────────────────


@app.command()
def serve(
    model: str = typer.Argument(..., help="Path to Rust server binary or ONNX model."),
    port: int = typer.Option(8080, help="Port to serve on."),
    build: bool = typer.Option(False, help="Build the Rust binary before serving."),
    crate_dir: Optional[str] = typer.Option(None, help="Rust crate directory (for --build)."),
):
    """Start the inference server."""
    try:
        if build and crate_dir:
            logger.info("Building Rust binary in %s", crate_dir)
            console.print(f"🔧 Building Rust binary in [dim]{crate_dir}[/dim]...")
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=crate_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Build failed: %s", result.stderr)
                console.print(f"[red]Build failed:\n{result.stderr}[/red]")
                raise typer.Exit(code=1)

            binary_name = os.path.basename(crate_dir)
            if os.name == "nt":
                binary_name += ".exe"
            model = os.path.join(crate_dir, "target", "release", binary_name)
            logger.info("Built binary: %s", model)
            console.print(f"✅ Built: {model}")

        if not os.path.exists(model):
            console.print(f"[red]Binary or model not found: {model}[/red]")
            console.print("[dim]Hint: use --build --crate-dir <path> to build first.[/dim]")
            raise typer.Exit(code=1)

        logger.info("Starting server on port %d", port)
        console.print(f"\n🚀 Starting server on port [bold cyan]{port}[/bold cyan]...")
        subprocess.run([model, "--port", str(port)])

    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Serve failed: %s", e, exc_info=True)
        console.print(f"\n[red]❌ Serve failed: {e}[/red]")
        raise typer.Exit(code=1)


# ──────────────────────────────── list / info ────────────────────────────────


@app.command(name="list")
def list_models():
    """List all registered models."""
    registry_module.print_model_list()


@app.command()
def info(
    model_name: str = typer.Argument(..., help="Name of the model to inspect."),
):
    """Show detailed info about a registered model."""
    registry_module.print_model_info(model_name)


# ──────────────────────────────── entry ────────────────────────────────


def main():
    app()


if __name__ == "__main__":
    main()
