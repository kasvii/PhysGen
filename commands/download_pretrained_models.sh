#!/usr/bin/env bash
set -euo pipefail

REPO_ID="kasvii/PhysGen"
LOCAL_ROOT="."

python -m pip install -U huggingface_hub

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "kasvii/PhysGen"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=".",
    allow_patterns=[
        "outputs/DragDec/*",
        "outputs/PhysDec/*",
        "outputs/ShapeVAE/*",
        "outputs/FinetuneAll/*",
        "outputs/Dora-VAE-1.1/*",
    ],
    local_dir_use_symlinks=False,
)

print("Done. Selected checkpoints downloaded.")
PY