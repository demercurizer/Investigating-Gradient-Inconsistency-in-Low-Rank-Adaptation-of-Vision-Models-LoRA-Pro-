#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from datasets import load_dataset

dataset_name = "timm/eurosat-rgb"
for split in ("train", "validation", "test"):
    ds = load_dataset(dataset_name, split=split)
    print(f"{dataset_name}:{split}: {len(ds)} examples")
PY
