// rust_runtime/examples/wasm_frontend/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet() -> String {
    "Hello from wasm!".to_string()
}
