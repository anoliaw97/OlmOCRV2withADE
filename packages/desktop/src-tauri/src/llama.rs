use serde::Serialize;
use std::{path::PathBuf, process::Command, time::Instant};

#[derive(Serialize)]
pub struct LlamaRunResult {
    pub answer: String,
    pub model_path: String,
    pub runtime: String,
    pub latency_ms: u128,
}

pub fn generate(
    prompt: String,
    model_path: String,
    system_prompt: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    context: Option<String>,
    llama_cli_path: Option<String>,
) -> Result<LlamaRunResult, String> {
    let t0 = Instant::now();
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err("Prompt is required for llama.cpp generation".to_string());
    }

    let model = PathBuf::from(model_path.trim());
    if !model.exists() {
        return Err(format!("Model file not found: {}", model.display()));
    }

    let executable = resolve_llama_cli(llama_cli_path)?;
    let composed_prompt = compose_prompt(prompt, system_prompt, context);
    let token_limit = max_tokens.unwrap_or(384).max(32);
    let temp = temperature.unwrap_or(0.2).clamp(0.0, 2.0);

    let output = Command::new(&executable)
        .arg("-m")
        .arg(&model)
        .arg("-p")
        .arg(&composed_prompt)
        .arg("-n")
        .arg(token_limit.to_string())
        .arg("--temp")
        .arg(temp.to_string())
        .arg("--no-display-prompt")
        .output()
        .map_err(|e| {
            format!(
                "Failed to execute llama.cpp runtime '{}': {e}",
                executable.display()
            )
        })?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("llama.cpp invocation failed: {detail}"));
    }

    let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if answer.is_empty() {
        return Err("llama.cpp returned an empty response".to_string());
    }

    Ok(LlamaRunResult {
        answer,
        model_path: model.to_string_lossy().to_string(),
        runtime: executable.to_string_lossy().to_string(),
        latency_ms: t0.elapsed().as_millis(),
    })
}

fn compose_prompt(prompt: &str, system_prompt: Option<String>, context: Option<String>) -> String {
    let mut full = String::new();

    if let Some(system) = system_prompt {
        let system = system.trim();
        if !system.is_empty() {
            full.push_str("[SYSTEM]\n");
            full.push_str(system);
            full.push_str("\n\n");
        }
    }

    if let Some(ctx) = context {
        let ctx = ctx.trim();
        if !ctx.is_empty() {
            full.push_str("[CONTEXT]\n");
            full.push_str(ctx);
            full.push_str("\n\n");
        }
    }

    full.push_str("[USER]\n");
    full.push_str(prompt);
    full.push_str("\n\n[ASSISTANT]\n");
    full
}

fn resolve_llama_cli(llama_cli_path: Option<String>) -> Result<PathBuf, String> {
    if let Some(raw) = llama_cli_path {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("Provided llama_cli_path is empty".to_string());
        }
        let explicit = PathBuf::from(raw);
        if explicit.exists() {
            return Ok(explicit);
        }

        return Err(format!(
            "Provided llama.cpp executable was not found: {raw}"
        ));
    }

    for candidate in ["llama-cli.exe", "llama-cli"] {
        if command_exists(candidate) {
            return Ok(PathBuf::from(candidate));
        }
    }

    Err(
        "Could not find llama-cli executable. Provide llama_cli_path or add llama-cli to PATH."
            .to_string(),
    )
}

fn command_exists(command_name: &str) -> bool {
    Command::new(command_name)
        .arg("--help")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}
