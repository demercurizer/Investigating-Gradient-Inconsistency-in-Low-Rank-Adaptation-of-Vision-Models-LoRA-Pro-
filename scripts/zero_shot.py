"""Zero-shot CLIP evaluation on EuroSAT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


EUROSAT_LABELS = [
    "annual crop land",
    "forest",
    "herbaceous vegetation land",
    "highway or road",
    "industrial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings",
    "river",
    "sea or lake",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pooled_output(output):
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, tuple):
        return output[1]
    raise TypeError(f"Unsupported CLIP output type: {type(output)!r}")


def get_text_embeddings(model: CLIPModel, inputs: dict) -> torch.Tensor:
    features = pooled_output(model.get_text_features(**inputs))
    if features.shape[-1] != model.projection_dim:
        features = model.text_projection(features)
    return features


def get_image_embeddings(model: CLIPModel, inputs: dict) -> torch.Tensor:
    features = pooled_output(model.get_image_features(**inputs))
    if features.shape[-1] != model.projection_dim:
        features = model.visual_projection(features)
    return features


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    dry_run = bool(args.dry_run or cfg.get("dry_run", {}).get("enabled", False))
    output_dir = Path(cfg.get("paths", {}).get("output_dir", cfg["experiment"].get("output_dir", "results/zero_shot")))
    log_dir = Path(cfg.get("paths", {}).get("log_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["model"].get("pretrained_model_name_or_path", cfg["model"].get("name"))
    batch_size = int(cfg["data"].get("batch_size", 128))
    max_eval_batches = cfg.get("dry_run", {}).get("max_eval_batches") if dry_run else None

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Dry run: {dry_run}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    dataset_name = cfg["data"].get("dataset", "timm/eurosat-rgb")
    split = cfg["data"].get("split", "test")
    dataset = load_dataset(dataset_name, split=split)
    prompts = [f"a satellite photo of {label}" for label in EUROSAT_LABELS]

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_features = get_text_embeddings(model, text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    for batch_idx, start in enumerate(tqdm(range(0, len(dataset), batch_size))):
        if max_eval_batches is not None and batch_idx >= int(max_eval_batches):
            break
        batch = dataset[start : start + batch_size]
        labels = torch.tensor(batch["label"], device=device)

        image_inputs = processor(images=batch["image"], return_tensors="pt").to(device)
        image_features = get_image_embeddings(model, image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    metrics = {
        "method": "zero-shot",
        "dataset": "EuroSAT",
        "model": model_name,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "device": device,
        "batch_size": batch_size,
        "dry_run": dry_run,
        "prompts": prompts,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
