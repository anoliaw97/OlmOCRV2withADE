from __future__ import annotations

from huggingface_hub import snapshot_download


MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "allenai/olmOCR-2-7B-1025-FP8",
    "Qwen/Qwen2.5-3B-Instruct",
]


def main() -> None:
    print("Downloading required Hugging Face models (resume enabled)...")
    for model_id in MODELS:
        print(f"\\n==> {model_id}")
        snapshot_download(repo_id=model_id, resume_download=True)
    print("\\nAll model downloads completed.")


if __name__ == "__main__":
    main()
