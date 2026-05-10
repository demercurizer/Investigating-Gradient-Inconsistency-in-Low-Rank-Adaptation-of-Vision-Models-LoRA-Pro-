# LoRA-Pro for CLIP-ViT Image Classification

## Goal

We investigate whether LoRA-Pro reduces the gap between LoRA and full fine-tuning
for CLIP-ViT-B/16 image classification.

## Methods

- Zero-shot CLIP
- Linear probe
- LoRA
- DoRA
- LoRA-Pro

## Main setup

- Backbone: `openai/clip-vit-base-patch16`
- Datasets: EuroSAT, CIFAR-100
- Main rank: `r=8`
- Target modules: `q_proj`, `v_proj`

## Environment

```bash
conda env create -f environment.yml
conda activate lorapro-vit