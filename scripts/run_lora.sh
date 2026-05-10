#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config configs/lora_train.yaml
