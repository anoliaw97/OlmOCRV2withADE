mod llama;
mod ocr;

use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct AppInfo {
    app: String,
    version: String,
    stack: String,
}

#[derive(Serialize)]
struct PipelineResult {
    answer: String,
    mode: String,
    latency_ms: u128,
}

#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        app: "OCRLLMVLMMLPETROAFRO Desktop".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        stack: "Tauri2 + Solid + TypeScript".to_string(),
    }
}

#[tauri::command]
fn run_pipeline(question: String, mode: Option<String>) -> PipelineResult {
    let t0 = Instant::now();
    let m = mode.unwrap_or_else(|| "balanced".to_string());
    let normalized = question.trim();

    let answer = if normalized.is_empty() {
        "No input provided.".to_string()
    } else {
        format!(
            "Pipeline accepted your request in '{}' mode.\n\nRuntime modules are now available through:\n- ingest_ocr\n- run_llama_cpp\n\nInput:\n{}",
            m, normalized
        )
    };

    PipelineResult {
        answer,
        mode: m,
        latency_ms: t0.elapsed().as_millis(),
    }
}

#[tauri::command]
fn ingest_ocr(
    file_path: String,
    language: Option<String>,
    tesseract_path: Option<String>,
) -> Result<ocr::OcrIngestionResult, String> {
    ocr::ingest(file_path, language, tesseract_path)
}

#[tauri::command]
fn run_llama_cpp(
    prompt: String,
    model_path: String,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    context: Option<String>,
    llama_cli_path: Option<String>,
) -> Result<llama::LlamaRunResult, String> {
    llama::generate(
        prompt,
        model_path,
        system_prompt,
        max_tokens,
        temperature,
        context,
        llama_cli_path,
    )
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            app_info,
            run_pipeline,
            ingest_ocr,
            run_llama_cpp
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
