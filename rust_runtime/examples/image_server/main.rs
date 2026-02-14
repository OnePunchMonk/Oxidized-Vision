use actix_web::{post, get, web, App, HttpServer, Responder, HttpResponse};
use serde::{Deserialize, Serialize};
use runner_core::{Runner, RunnerConfig};
use runner_tract::TractRunner;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;
use clap::Parser;

struct AppState {
    runner: Arc<TractRunner>,
    config: RunnerConfig,
}

#[derive(Parser, Debug)]
#[clap(author, version, about = "OxidizedVision inference server")]
struct Args {
    /// Path to the ONNX model file
    #[clap(short, long)]
    model: String,

    /// Port to listen on
    #[clap(short, long, default_value_t = 8080)]
    port: u16,

    /// Input shape as comma-separated values (e.g., "1,3,256,256")
    #[clap(long, default_value = "1,3,256,256")]
    input_shape: String,
}

#[derive(Deserialize)]
struct InferenceRequest {
    /// Flattened input data as a list of f32 values.
    /// If empty, zeros will be used.
    data: Option<Vec<f32>>,

    /// Input shape (e.g. [1, 3, 256, 256]).
    /// If not provided, uses the server's default.
    shape: Option<Vec<usize>>,
}

#[derive(Serialize)]
struct InferenceResponse {
    status: String,
    output_shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    backend: String,
    input_shape: Vec<usize>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[get("/health")]
async fn health(data: web::Data<AppState>) -> impl Responder {
    let info = data.runner.info();
    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        model: info.name,
        backend: info.backend,
        input_shape: info.input_shape,
    })
}

#[post("/predict")]
async fn predict(req: web::Json<InferenceRequest>, data: web::Data<AppState>) -> impl Responder {
    let shape = req.shape.clone().unwrap_or_else(|| data.config.input_shape.clone());

    if shape.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "Input shape cannot be empty".to_string(),
        });
    }

    let numel: usize = shape.iter().product();

    let input_data = match &req.data {
        Some(d) => {
            if d.len() != numel {
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        d.len(), shape, numel
                    ),
                });
            }
            d.clone()
        }
        None => vec![0.0f32; numel],
    };

    let input = match ArrayD::<f32>::from_shape_vec(IxDyn(&shape), input_data) {
        Ok(arr) => arr,
        Err(e) => {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: format!("Failed to create input array: {}", e),
            });
        }
    };

    match data.runner.run(&input) {
        Ok(output) => {
            let output_shape = output.shape().to_vec();
            let output_data: Vec<f32> = output.iter().cloned().collect();
            HttpResponse::Ok().json(InferenceResponse {
                status: "success".to_string(),
                output_shape,
                data: output_data,
            })
        }
        Err(e) => {
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Inference failed: {}", e),
            })
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let input_shape: Vec<usize> = args
        .input_shape
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("Invalid input shape dimension"))
        .collect();

    let config = RunnerConfig {
        model_path: args.model.clone(),
        input_shape: input_shape.clone(),
        use_cuda: false,
        optimize: true,
    };

    let runner = TractRunner::from_config(&config).expect("Failed to load model");
    let runner = Arc::new(runner);

    let app_state = web::Data::new(AppState {
        runner: runner.clone(),
        config: config.clone(),
    });

    println!("🚀 OxidizedVision server starting at http://127.0.0.1:{}", args.port);
    println!("   Model: {}", args.model);
    println!("   Input shape: {:?}", input_shape);
    println!("   Endpoints:");
    println!("     POST /predict  - Run inference");
    println!("     GET  /health   - Health check");

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(health)
            .service(predict)
    })
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}
