use serde::Serialize;
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

#[derive(Serialize)]
pub struct OcrIngestionResult {
    pub file_path: String,
    pub extractor: String,
    pub text: String,
    pub preview: String,
    pub char_count: usize,
    pub elapsed_ms: u128,
}

pub fn ingest(
    file_path: String,
    language: Option<String>,
    tesseract_path: Option<String>,
) -> Result<OcrIngestionResult, String> {
    let t0 = Instant::now();
    let normalized = file_path.trim();
    if normalized.is_empty() {
        return Err("OCR file path is required".to_string());
    }

    let input_path = PathBuf::from(normalized);
    if !input_path.exists() {
        return Err(format!("OCR file not found: {}", input_path.display()));
    }

    let language_code = language
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "eng".to_string());

    let (text, extractor) = if is_plain_text_file(&input_path) {
        let text = fs::read_to_string(&input_path)
            .map_err(|e| format!("Failed to read text file '{}': {e}", input_path.display()))?;
        (text, "native-text-read".to_string())
    } else {
        let executable = resolve_tesseract(tesseract_path)?;
        let output = Command::new(&executable)
            .arg(&input_path)
            .arg("stdout")
            .arg("-l")
            .arg(&language_code)
            .output()
            .map_err(|e| {
                format!(
                    "Failed to execute tesseract '{}': {e}",
                    executable.display()
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let detail = if stderr.is_empty() {
                "tesseract returned a non-zero exit code".to_string()
            } else {
                stderr
            };
            return Err(format!(
                "OCR failed for '{}': {detail}",
                input_path.display()
            ));
        }

        let extracted = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if extracted.is_empty() {
            return Err(format!(
                "OCR completed but no text was extracted from '{}'",
                input_path.display()
            ));
        }

        (
            extracted,
            format!("tesseract:{}", executable.to_string_lossy()),
        )
    };

    let preview: String = text.chars().take(600).collect();
    Ok(OcrIngestionResult {
        file_path: input_path.to_string_lossy().to_string(),
        extractor,
        char_count: text.chars().count(),
        text,
        preview,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

fn resolve_tesseract(tesseract_path: Option<String>) -> Result<PathBuf, String> {
    if let Some(raw) = tesseract_path {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("Provided tesseract path is empty".to_string());
        }

        let explicit = PathBuf::from(raw);
        if explicit.exists() {
            return Ok(explicit);
        }

        return Err(format!(
            "Provided tesseract executable was not found: {raw}"
        ));
    }

    for candidate in ["tesseract.exe", "tesseract"] {
        if command_exists(candidate) {
            return Ok(PathBuf::from(candidate));
        }
    }

    Err(
        "Could not find tesseract executable. Install Tesseract OCR or provide tesseract_path."
            .to_string(),
    )
}

fn command_exists(command_name: &str) -> bool {
    Command::new(command_name)
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn is_plain_text_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .map(OsString::from)
        .unwrap_or_else(OsString::new)
        .to_string_lossy()
        .to_ascii_lowercase();

    matches!(
        ext.as_str(),
        "txt" | "md" | "csv" | "json" | "yaml" | "yml" | "log" | "tsv"
    )
}
