#!/usr/bin/env bash
set -euo pipefail

python -V
nvidia-smi || true

python -m pip install -U pip
python -m pip install -r requirements.txt

mkdir -p /workspace/data
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs
