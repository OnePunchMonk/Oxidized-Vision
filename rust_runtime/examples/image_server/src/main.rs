//! OxidizedVision Inference Server
//!
//! A production-grade REST API server with:
//! - **Dynamic batching**: Collects requests and batches them for efficient GPU/CPU inference.
//! - **Multi-model support**: Serve multiple models simultaneously, routed by name.
//! - **Structured logging**: Full `tracing` integration with JSON output and request tracing.
//! - **Prometheus metrics**: `/metrics` endpoint for observability.
//! - **Health checking**: `/health` endpoint with per-model status.

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use ndarray::{ArrayD, IxDyn};
use runner_core::{Runner, RunnerConfig};
use runner_ort::OrtRunner;
use runner_tract::TractRunner;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{debug, error, info, instrument};
use tracing_actix_web::TracingLogger;

mod preprocess;

// ─────────────────────────────── Configuration ───────────────────────────────

#[derive(Parser, Debug)]
#[clap(author, version, about = "OxidizedVision inference server")]
struct Args {
    /// Path(s) to ONNX model files (comma-separated for multi-model).
    /// Format: "name=path" or just "path" (name defaults to filename stem).
    #[clap(short, long)]
    model: Vec<String>,

    /// Port to listen on
    #[clap(short, long, default_value_t = 8080)]
    port: u16,

    /// Input shape as comma-separated values (e.g., "1,3,256,256")
    #[clap(long, default_value = "1,3,256,256")]
    input_shape: String,

    /// Maximum batch size for dynamic batching (0 = disabled)
    #[clap(long, default_value_t = 0)]
    max_batch_size: usize,

    /// Maximum wait time in milliseconds before flushing a partial batch
    #[clap(long, default_value_t = 50)]
    max_wait_ms: u64,

    /// Log format: 'pretty' or 'json'
    #[clap(long, default_value = "pretty")]
    log_format: String,

    /// Inference backend: 'tract' (pure Rust, portable) or 'ort' (ONNX Runtime,
    /// fused vision kernels — generally faster on CPU/GPU for CNN/ViT models).
    #[clap(long, default_value = "ort")]
    backend: String,

    /// Run inference on GPU via CUDA (only applies to the 'ort' backend, and
    /// only if runner_ort was built with the `cuda` feature).
    #[clap(long)]
    use_cuda: bool,
}

// ─────────────────────────────── Request / Response types ─────────────────────

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
    model: String,
    output_shape: Vec<usize>,
    data: Vec<f32>,
    latency_ms: f64,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    models: Vec<ModelHealthEntry>,
    total_requests: u64,
    total_errors: u64,
}

#[derive(Serialize)]
struct ModelHealthEntry {
    name: String,
    backend: String,
    input_shape: Vec<usize>,
    status: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    request_id: String,
}

#[derive(Serialize)]
struct MetricsResponse {
    total_requests: u64,
    total_errors: u64,
    models_loaded: usize,
    batching_enabled: bool,
    max_batch_size: usize,
    pending_batch_items: usize,
}

// ─────────────────────────────── Dynamic Batcher ─────────────────────────────

struct BatchItem {
    input: ArrayD<f32>,
    responder: oneshot::Sender<Result<ArrayD<f32>, String>>,
}

struct DynamicBatcher {
    queue: Mutex<Vec<BatchItem>>,
    max_batch_size: usize,
    max_wait: Duration,
}

impl DynamicBatcher {
    fn new(max_batch_size: usize, max_wait_ms: u64) -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
            max_batch_size,
            max_wait: Duration::from_millis(max_wait_ms),
        }
    }

    fn pending_count(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Submit a request to the batcher and wait for the result.
    async fn submit(
        &self,
        input: ArrayD<f32>,
        runner: &Arc<dyn Runner>,
    ) -> Result<ArrayD<f32>, String> {
        if self.max_batch_size <= 1 {
            // Batching disabled — run directly.
            return runner
                .run(&input)
                .map_err(|e| format!("Inference failed: {}", e));
        }

        let (tx, rx) = oneshot::channel();
        let should_flush;

        {
            let mut queue = self.queue.lock().unwrap();
            queue.push(BatchItem {
                input,
                responder: tx,
            });
            should_flush = queue.len() >= self.max_batch_size;
        }

        if should_flush {
            self.flush(runner).await;
        } else {
            // Wait for the batch timer to flush
            let runner_clone = runner.clone();
            let max_wait = self.max_wait;
            // Note: In a real production system, you'd use a shared timer.
            // Here we spawn a delayed flush per partial batch as a simple approach.
            let batcher_ptr = self as *const DynamicBatcher as usize;
            tokio::spawn(async move {
                tokio::time::sleep(max_wait).await;
                // Safety: The batcher lives for the entire server lifetime via AppState.
                // This is safe because AppState is wrapped in web::Data (Arc).
                let batcher = unsafe { &*(batcher_ptr as *const DynamicBatcher) };
                batcher.flush(&runner_clone).await;
            });
        }

        rx.await.map_err(|_| "Batch channel closed".to_string())?
    }

    /// Flush all pending items in the queue — runs each through the runner.
    async fn flush(&self, runner: &Arc<dyn Runner>) {
        let items: Vec<BatchItem> = {
            let mut queue = self.queue.lock().unwrap();
            std::mem::take(&mut *queue)
        };

        if items.is_empty() {
            return;
        }

        debug!(batch_size = items.len(), "Flushing batch");

        // Process each item. In a future version with true batched inference,
        // inputs could be concatenated along dim 0 and run as a single batch.
        for item in items {
            let result = runner
                .run(&item.input)
                .map_err(|e| format!("Inference failed: {}", e));
            let _ = item.responder.send(result);
        }
    }
}

// ─────────────────────────────── Server State ────────────────────────────────

struct ModelEntry {
    runner: Arc<dyn Runner>,
    config: RunnerConfig,
    batcher: DynamicBatcher,
}

struct AppState {
    models: HashMap<String, ModelEntry>,
    default_model: String,
    metrics: ServerMetrics,
}

struct ServerMetrics {
    total_requests: AtomicU64,
    total_errors: AtomicU64,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        }
    }
}

// ─────────────────────────────── Endpoints ───────────────────────────────────

#[get("/health")]
#[instrument(skip(data))]
async fn health(data: web::Data<AppState>) -> impl Responder {
    let models: Vec<ModelHealthEntry> = data
        .models
        .iter()
        .map(|(name, entry)| {
            let info = entry.runner.info();
            ModelHealthEntry {
                name: name.clone(),
                backend: info.backend,
                input_shape: info.input_shape,
                status: "ready".to_string(),
            }
        })
        .collect();

    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        models,
        total_requests: data.metrics.total_requests.load(Ordering::Relaxed),
        total_errors: data.metrics.total_errors.load(Ordering::Relaxed),
    })
}

/// Prometheus scrape endpoint (text exposition format, not JSON — a
/// `/metrics` path is conventionally expected to be Prometheus-scrapable,
/// so this emits the real exposition format rather than an ad-hoc JSON
/// shape a scraper can't parse). Use `/metrics.json` for the same numbers
/// as JSON, e.g. for scripting.
#[get("/metrics")]
#[instrument(skip(data))]
async fn metrics(data: web::Data<AppState>) -> impl Responder {
    let stats = collect_metrics(&data);

    let mut body = String::new();
    body.push_str("# HELP oxidizedvision_requests_total Total inference requests received.\n");
    body.push_str("# TYPE oxidizedvision_requests_total counter\n");
    body.push_str(&format!(
        "oxidizedvision_requests_total {}\n",
        stats.total_requests
    ));

    body.push_str("# HELP oxidizedvision_errors_total Total inference requests that errored.\n");
    body.push_str("# TYPE oxidizedvision_errors_total counter\n");
    body.push_str(&format!(
        "oxidizedvision_errors_total {}\n",
        stats.total_errors
    ));

    body.push_str("# HELP oxidizedvision_models_loaded Number of models currently loaded.\n");
    body.push_str("# TYPE oxidizedvision_models_loaded gauge\n");
    body.push_str(&format!(
        "oxidizedvision_models_loaded {}\n",
        stats.models_loaded
    ));

    body.push_str(
        "# HELP oxidizedvision_batching_enabled Whether dynamic batching is enabled (1) or not (0).\n",
    );
    body.push_str("# TYPE oxidizedvision_batching_enabled gauge\n");
    body.push_str(&format!(
        "oxidizedvision_batching_enabled {}\n",
        stats.batching_enabled as u8
    ));

    body.push_str("# HELP oxidizedvision_max_batch_size Configured maximum batch size.\n");
    body.push_str("# TYPE oxidizedvision_max_batch_size gauge\n");
    body.push_str(&format!(
        "oxidizedvision_max_batch_size {}\n",
        stats.max_batch_size
    ));

    body.push_str(
        "# HELP oxidizedvision_pending_batch_items Requests currently queued awaiting a batch flush.\n",
    );
    body.push_str("# TYPE oxidizedvision_pending_batch_items gauge\n");
    body.push_str(&format!(
        "oxidizedvision_pending_batch_items {}\n",
        stats.pending_batch_items
    ));

    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4; charset=utf-8")
        .body(body)
}

/// Same numbers as `/metrics`, as JSON — for scripting/tooling that would
/// rather not parse the Prometheus text format.
#[get("/metrics.json")]
#[instrument(skip(data))]
async fn metrics_json(data: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().json(collect_metrics(&data))
}

fn collect_metrics(data: &web::Data<AppState>) -> MetricsResponse {
    let total_pending: usize = data
        .models
        .values()
        .map(|e| e.batcher.pending_count())
        .sum();
    let any_batching = data.models.values().any(|e| e.batcher.max_batch_size > 1);
    let max_bs = data
        .models
        .values()
        .map(|e| e.batcher.max_batch_size)
        .max()
        .unwrap_or(0);

    MetricsResponse {
        total_requests: data.metrics.total_requests.load(Ordering::Relaxed),
        total_errors: data.metrics.total_errors.load(Ordering::Relaxed),
        models_loaded: data.models.len(),
        batching_enabled: any_batching,
        max_batch_size: max_bs,
        pending_batch_items: total_pending,
    }
}

/// Predict on the default model.
#[post("/predict")]
#[instrument(skip(req, data), fields(model = %data.default_model))]
async fn predict(req: web::Json<InferenceRequest>, data: web::Data<AppState>) -> impl Responder {
    let model_name = &data.default_model;
    run_inference(model_name, &req, &data).await
}

/// Predict on a specific named model.
#[post("/predict/{model_name}")]
#[instrument(skip(req, data), fields(model = %model_name))]
async fn predict_named(
    model_name: web::Path<String>,
    req: web::Json<InferenceRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    run_inference(&model_name, &req, &data).await
}

async fn run_inference(
    model_name: &str,
    req: &InferenceRequest,
    data: &web::Data<AppState>,
) -> HttpResponse {
    let request_id = uuid::Uuid::new_v4().to_string();

    let entry = match get_model_entry(model_name, data, &request_id) {
        Ok(e) => e,
        Err(resp) => return *resp,
    };

    let shape = req
        .shape
        .clone()
        .unwrap_or_else(|| entry.config.input_shape.clone());

    if shape.is_empty() {
        data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "Input shape cannot be empty".to_string(),
            request_id,
        });
    }

    let numel: usize = shape.iter().product();

    let input_data = match &req.data {
        Some(d) => {
            if d.len() != numel {
                data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        d.len(),
                        shape,
                        numel
                    ),
                    request_id,
                });
            }
            d.clone()
        }
        None => vec![0.0f32; numel],
    };

    let input = match ArrayD::<f32>::from_shape_vec(IxDyn(&shape), input_data) {
        Ok(arr) => arr,
        Err(e) => {
            data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: format!("Failed to create input array: {}", e),
                request_id,
            });
        }
    };

    run_and_respond(model_name, entry, input, data, request_id).await
}

/// Predict on the default model from a raw image upload (JPEG/PNG/etc).
/// Decodes, SIMD-resizes to the model's configured input shape, and
/// normalizes in one fused pass (see `preprocess::decode_resize_normalize`)
/// instead of requiring the caller to pre-process into a flat float array.
#[post("/predict/image")]
#[instrument(skip(body, data), fields(model = %data.default_model))]
async fn predict_image(body: web::Bytes, data: web::Data<AppState>) -> impl Responder {
    let model_name = data.default_model.clone();
    run_inference_from_image(&model_name, &body, &data).await
}

/// Predict on a named model from a raw image upload.
#[post("/predict/image/{model_name}")]
#[instrument(skip(body, data), fields(model = %model_name))]
async fn predict_image_named(
    model_name: web::Path<String>,
    body: web::Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    run_inference_from_image(&model_name, &body, &data).await
}

async fn run_inference_from_image(
    model_name: &str,
    image_bytes: &[u8],
    data: &web::Data<AppState>,
) -> HttpResponse {
    let request_id = uuid::Uuid::new_v4().to_string();

    let entry = match get_model_entry(model_name, data, &request_id) {
        Ok(e) => e,
        Err(resp) => return *resp,
    };

    let shape = &entry.config.input_shape;
    if shape.len() != 4 {
        data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!(
                "Model '{}' has input shape {:?}; image preprocessing requires a 4D \
                 [N, C, H, W] shape",
                model_name, shape
            ),
            request_id,
        });
    }
    let (target_h, target_w) = (shape[2] as u32, shape[3] as u32);

    let input = match preprocess::decode_resize_normalize(
        image_bytes,
        target_h,
        target_w,
        preprocess::NormalizeStats::default(),
    ) {
        Ok(arr) => arr,
        Err(e) => {
            data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: format!("Failed to preprocess image: {}", e),
                request_id,
            });
        }
    };

    run_and_respond(model_name, entry, input, data, request_id).await
}

/// Shared inference tail: run the model on a prepared input tensor and
/// build the HTTP response. Used by both the raw-tensor and image-upload
/// predict paths.
async fn run_and_respond(
    model_name: &str,
    entry: &ModelEntry,
    input: ArrayD<f32>,
    data: &web::Data<AppState>,
    request_id: String,
) -> HttpResponse {
    data.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

    let start = Instant::now();
    let result = entry.batcher.submit(input, &entry.runner).await;
    let latency = start.elapsed();
    let latency_ms = latency.as_secs_f64() * 1000.0;

    match result {
        Ok(output) => {
            let output_shape = output.shape().to_vec();
            let output_data: Vec<f32> = output.iter().cloned().collect();
            debug!(
                model = model_name,
                latency_ms = latency_ms,
                output_shape = ?output_shape,
                "Inference complete"
            );
            HttpResponse::Ok().json(InferenceResponse {
                status: "success".to_string(),
                model: model_name.to_string(),
                output_shape,
                data: output_data,
                latency_ms,
            })
        }
        Err(e) => {
            data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
            error!(model = model_name, error = %e, "Inference failed");
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: e,
                request_id,
            })
        }
    }
}

/// Look up a model by name, returning a structured 404 response on miss.
fn get_model_entry<'a>(
    model_name: &str,
    data: &'a web::Data<AppState>,
    request_id: &str,
) -> Result<&'a ModelEntry, Box<HttpResponse>> {
    data.models.get(model_name).ok_or_else(|| {
        data.metrics.total_errors.fetch_add(1, Ordering::Relaxed);
        let available: Vec<&String> = data.models.keys().collect();
        Box::new(HttpResponse::NotFound().json(ErrorResponse {
            error: format!(
                "Model '{}' not found. Available models: {:?}",
                model_name, available
            ),
            request_id: request_id.to_string(),
        }))
    })
}

/// List all loaded models.
#[get("/models")]
#[instrument(skip(data))]
async fn list_models(data: web::Data<AppState>) -> impl Responder {
    let models: Vec<serde_json::Value> = data
        .models
        .iter()
        .map(|(name, entry)| {
            let info = entry.runner.info();
            serde_json::json!({
                "name": name,
                "backend": info.backend,
                "input_shape": info.input_shape,
                "batching_enabled": entry.batcher.max_batch_size > 1,
                "max_batch_size": entry.batcher.max_batch_size,
            })
        })
        .collect();

    HttpResponse::Ok().json(serde_json::json!({ "models": models }))
}

// ─────────────────────────────── Main ────────────────────────────────────────

fn parse_model_arg(arg: &str) -> (String, String) {
    if let Some(idx) = arg.find('=') {
        let name = arg[..idx].to_string();
        let path = arg[idx + 1..].to_string();
        (name, path)
    } else {
        let path = arg.to_string();
        let name = std::path::Path::new(arg)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string();
        (name, path)
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true);

    match args.log_format.as_str() {
        "json" => {
            subscriber.json().init();
        }
        _ => {
            subscriber.init();
        }
    }

    let input_shape: Vec<usize> = args
        .input_shape
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .expect("Invalid input shape dimension")
        })
        .collect();

    // Load models
    let mut models = HashMap::new();
    let mut first_model_name = String::new();

    if args.model.is_empty() {
        error!("No model specified. Use --model <path> or --model <name>=<path>");
        std::process::exit(1);
    }

    for model_arg in &args.model {
        let (name, path) = parse_model_arg(model_arg);

        info!(
            name = %name,
            path = %path,
            input_shape = ?input_shape,
            max_batch_size = args.max_batch_size,
            "Loading model"
        );

        let config = RunnerConfig {
            model_path: path.clone(),
            input_shape: input_shape.clone(),
            use_cuda: args.use_cuda,
            optimize: true,
        };

        let runner: Arc<dyn Runner> = match args.backend.as_str() {
            "tract" => Arc::new(TractRunner::from_config(&config).unwrap_or_else(|e| {
                error!(name = %name, path = %path, error = %e, "Failed to load model");
                std::process::exit(1);
            })),
            "ort" => Arc::new(OrtRunner::from_config(&config).unwrap_or_else(|e| {
                error!(name = %name, path = %path, error = %e, "Failed to load model");
                std::process::exit(1);
            })),
            other => {
                error!(backend = %other, "Unknown backend. Supported: 'tract', 'ort'");
                std::process::exit(1);
            }
        };

        let batcher = DynamicBatcher::new(args.max_batch_size, args.max_wait_ms);

        if first_model_name.is_empty() {
            first_model_name = name.clone();
        }

        models.insert(
            name.clone(),
            ModelEntry {
                runner,
                config,
                batcher,
            },
        );

        info!(name = %name, "Model loaded successfully");
    }

    let app_state = web::Data::new(AppState {
        models,
        default_model: first_model_name.clone(),
        metrics: ServerMetrics::new(),
    });

    info!(
        port = args.port,
        models = app_state.models.len(),
        default_model = %first_model_name,
        batching = args.max_batch_size > 0,
        "Starting OxidizedVision server"
    );

    println!();
    println!(
        "🚀 OxidizedVision server starting at http://127.0.0.1:{}",
        args.port
    );
    println!("   Models loaded: {}", app_state.models.len());
    for (name, entry) in app_state.models.iter() {
        let info = entry.runner.info();
        println!(
            "     📦 {} ({}), input: {:?}",
            name, info.backend, info.input_shape
        );
    }
    println!(
        "   Dynamic batching: {}",
        if args.max_batch_size > 0 {
            format!(
                "enabled (max_batch_size={}, max_wait={}ms)",
                args.max_batch_size, args.max_wait_ms
            )
        } else {
            "disabled".to_string()
        }
    );
    println!("   Endpoints:");
    println!("     POST /predict                    - Inference from a raw tensor (default model)");
    println!("     POST /predict/<model_name>       - Inference from a raw tensor (named model)");
    println!(
        "     POST /predict/image              - Inference from a raw image upload (default model)"
    );
    println!(
        "     POST /predict/image/<model_name> - Inference from a raw image upload (named model)"
    );
    println!("     GET  /health               - Health check");
    println!("     GET  /metrics              - Server metrics");
    println!("     GET  /models               - List loaded models");
    println!();

    HttpServer::new(move || {
        App::new()
            .wrap(TracingLogger::default())
            .app_data(app_state.clone())
            .service(health)
            .service(metrics)
            .service(metrics_json)
            .service(predict_image)
            .service(predict_image_named)
            .service(predict)
            .service(predict_named)
            .service(list_models)
    })
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}
