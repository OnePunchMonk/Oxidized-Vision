use actix_web::{web, App, HttpServer, Responder, post};
use serde::Deserialize;
use runner_tract::TractRunner; // Assuming tract for the server example
use ndarray::Array4;
use std::sync::Mutex;
use clap::Parser;

struct AppState {
    runner: Mutex<TractRunner>,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long)]
    model: String,
    #[clap(short, long, default_value_t = 8080)]
    port: u16,
}

#[derive(Deserialize)]
struct InferenceRequest {
    // In a real app, you'd likely send image data
    // For simplicity, we'll just use a dummy array shape
    shape: Vec<usize>,
}

#[post("/predict")]
async fn predict(req: web::Json<InferenceRequest>, data: web::Data<AppState>) -> impl Responder {
    let shape = &req.shape;
    if shape.len() != 4 {
        return web::Json("Input shape must have 4 dimensions".to_string());
    }
    
    let input = Array4::<f32>::zeros((shape[0], shape[1], shape[2], shape[3]));
    
    let mut runner = data.runner.lock().unwrap();
    let result = runner.run(input);

    match result {
        Ok(output) => web::Json(format!("Inference successful, output shape: {:?}", output.shape())),
        Err(e) => web::Json(format!("Inference failed: {}", e)),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let runner = TractRunner::load(&args.model).expect("Failed to load model");
    let app_state = web::Data::new(AppState {
        runner: Mutex::new(runner),
    });

    println!("Starting server at http://127.0.0.1:{}", args.port);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(predict)
    })
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}
