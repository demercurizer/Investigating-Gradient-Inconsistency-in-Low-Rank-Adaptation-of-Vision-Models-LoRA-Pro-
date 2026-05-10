# Vast.ai Runbook

This runbook prepares and runs the LoRA-Pro CLIP-ViT baselines on a Vast.ai
PyTorch CUDA template.

## 1. Clone

```bash
git clone git@github.com:demercurizer/Investigating-Gradient-Inconsistency-in-Low-Rank-Adaptation-of-Vision-Models-LoRA-Pro-.git
cd Investigating-Gradient-Inconsistency-in-Low-Rank-Adaptation-of-Vision-Models-LoRA-Pro-
```

If SSH is not configured on the instance, use the HTTPS clone URL from GitHub.

## 2. Prepare Environment

Safe setup plus dry-run sanity check:

```bash
bash run.sh
```

Manual setup command:

```bash
bash scripts/prepare_env.sh
```

This checks Python and GPU visibility, installs `requirements.txt`, and creates:

```text
/workspace/data
/workspace/checkpoints
/workspace/logs
```

## 3. Optional Dataset Cache

```bash
bash scripts/download_data.sh
```

This caches EuroSAT RGB splits through Hugging Face Datasets. It is optional:
the training scripts will also download/cache the dataset on first use.

## 4. Dry-Run Sanity Check

```bash
bash scripts/sanity_check.sh
```

This is the first command to run after setup. It checks:

- PyTorch and CUDA visibility
- GPU name
- imports for `torch`, `transformers`, `datasets`, `peft`, `yaml`
- CLIP model creation
- EuroSAT batch loading
- forward pass
- cross-entropy loss
- backward pass
- dry-run metrics/checkpoint path creation

The dry-run uses:

```text
--dry_run
max_train_batches: 2
max_val_batches: 2
num_workers: 0
```

## 5. Zero-Shot Baseline

```bash
bash scripts/run_zero_shot.sh
```

Output:

```text
/workspace/checkpoints/baseline_zero_shot_eurosat/metrics.json
```

## 6. Linear Probe Baseline

```bash
bash scripts/run_linear_probe.sh
```

Output:

```text
/workspace/checkpoints/baseline_linear_probe_eurosat/metrics.json
/workspace/checkpoints/baseline_linear_probe_eurosat/checkpoint.pt
```

## 7. LoRA Baseline

```bash
bash scripts/run_lora.sh
```

Output:

```text
/workspace/checkpoints/lora_eurosat_r8/metrics.json
/workspace/checkpoints/lora_eurosat_r8/checkpoint.pt
```

## Configs

- `configs/baseline_zero_shot.yaml`
- `configs/baseline_linear_probe.yaml`
- `configs/lora_train.yaml`

All Vast paths are configured under:

```yaml
paths:
  data_dir: /workspace/data
  output_dir: /workspace/checkpoints/...
  log_dir: /workspace/logs
```

## Dry-Run Commands

```bash
python scripts/train.py --config configs/baseline_linear_probe.yaml --dry_run
python scripts/train.py --config configs/lora_train.yaml --dry_run
python scripts/zero_shot.py --config configs/baseline_zero_shot.yaml --dry_run
```

## Real Run Commands

```bash
bash scripts/run_zero_shot.sh
bash scripts/run_linear_probe.sh
bash scripts/run_lora.sh
```

## Do Not Commit

The following should stay out of git:

- datasets
- checkpoints
- logs
- `wandb/`
- generated PDFs unless explicitly needed for submission
