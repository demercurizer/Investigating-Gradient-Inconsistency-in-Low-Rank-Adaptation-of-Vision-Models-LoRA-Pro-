#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
import transformers
import datasets
import peft
import yaml

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("imports: ok")
PY

python scripts/train.py --config configs/baseline_linear_probe.yaml --dry_run
