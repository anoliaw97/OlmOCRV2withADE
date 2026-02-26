"""
Download and verify allenai/olmOCR-2-7B-1025-FP8 model.

Run once before launching the GUI:
    python download_model.py

This script:
1. Installs compressed-tensors (required for FP8)
2. Downloads the FP8 model from HuggingFace
3. Verifies the quantization_config is FP8 (not the full-precision model)
4. Prints the local cache path for confirmation
"""

import subprocess
import sys
import json
import os


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


print("=" * 60)
print("olmOCR FP8 Model Downloader")
print("=" * 60)

# Step 1: Ensure compressed-tensors is installed
print("\n[1/3] Checking compressed-tensors...")
try:
    import compressed_tensors
    print(f"  OK — compressed_tensors {compressed_tensors.__version__}")
except ImportError:
    print("  Not found — installing...")
    install("compressed-tensors")
    import compressed_tensors
    print(f"  Installed — compressed_tensors {compressed_tensors.__version__}")

# Step 2: Download model
print("\n[2/3] Downloading allenai/olmOCR-2-7B-1025-FP8...")
print("  (skipped if already cached)\n")

from huggingface_hub import snapshot_download

model_id = "allenai/olmOCR-2-7B-1025-FP8"
local_dir = snapshot_download(repo_id=model_id)
print(f"\n  Cached at: {local_dir}")

# Step 3: Verify it is the FP8 model
print("\n[3/3] Verifying FP8 quantization config...")
config_path = os.path.join(local_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

q_config = config.get("quantization_config", None)
if q_config is None:
    print("  ERROR: No quantization_config found!")
    print("  This may be the full-precision model, not the FP8 version.")
    sys.exit(1)

# Check it's float8
groups = q_config.get("config_groups", {})
found_fp8 = False
for group in groups.values():
    weights = group.get("weights", {})
    if weights.get("num_bits") == 8 and weights.get("type") == "float":
        found_fp8 = True
        break

if found_fp8:
    print("  OK — FP8 (float8) quantization confirmed")
else:
    print(f"  WARNING: Unexpected quantization config: {q_config}")

torch_dtype = config.get("torch_dtype", "unknown")
print(f"  torch_dtype in config: {torch_dtype}")

print("\n" + "=" * 60)
print("Model ready. You can now launch the GUI:")
print("  python olmocr_agentic_gui.py")
print("=" * 60)
