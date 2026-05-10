#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config configs/baseline_linear_probe.yaml
