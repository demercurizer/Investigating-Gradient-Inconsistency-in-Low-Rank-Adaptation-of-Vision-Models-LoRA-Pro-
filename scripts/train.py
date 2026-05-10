"""Training and dry-run entrypoint for CLIP linear probe and LoRA baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - handled at runtime for non-LoRA runs.
    LoraConfig = None
    TaskType = None
    get_peft_model = None


NUM_CLASSES = 10


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


def get_image_embeddings(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    features = pooled_output(model.get_image_features(pixel_values=pixel_values))
    if features.shape[-1] != model.projection_dim:
        features = model.visual_projection(features)
    return features


class ClipClassifier(nn.Module):
    def __init__(self, clip: CLIPModel, num_classes: int) -> None:
        super().__init__()
        self.clip = clip
        self.classifier = nn.Linear(clip.projection_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = get_image_embeddings(self.clip, pixel_values)
        return self.classifier(image_features)


def collate_fn(processor: CLIPProcessor):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

    return collate


def build_clip(cfg: dict) -> CLIPModel:
    model_name = cfg["model"].get("pretrained_model_name_or_path", cfg["model"].get("name"))
    clip = CLIPModel.from_pretrained(model_name)
    method = cfg["model"]["method"]

    if method == "linear_probe":
        for param in clip.parameters():
            param.requires_grad = False
        return clip

    if method == "lora":
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("peft is required for LoRA training. Install requirements.txt first.")
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(cfg["model"]["rank"]),
            lora_alpha=int(cfg["model"]["alpha"]),
            lora_dropout=float(cfg["model"].get("dropout", 0.0)),
            target_modules=list(cfg["model"].get("target_modules", ["q_proj", "v_proj"])),
        )
        return get_peft_model(clip, lora_cfg)

    raise ValueError(f"Unsupported training method: {method}")


def evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None) -> dict:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            loss_sum += loss.item()

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": total,
    }


def train(cfg: dict, dry_run: bool) -> dict:
    set_seed(int(cfg["experiment"].get("seed", 42)))

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", cfg["experiment"].get("output_dir", "outputs/train")))
    log_dir = Path(paths.get("log_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["model"].get("pretrained_model_name_or_path", cfg["model"].get("name"))
    processor = CLIPProcessor.from_pretrained(model_name)
    clip = build_clip(cfg)
    model = ClipClassifier(clip, NUM_CLASSES).to(device)

    data_cfg = cfg["data"]
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "validation")
    dataset_name = data_cfg.get("dataset", "timm/eurosat-rgb")
    num_workers = int(cfg.get("dry_run", {}).get("num_workers", 0) if dry_run else data_cfg.get("num_workers", 4))
    batch_size = int(data_cfg.get("batch_size", 64))

    train_data = load_dataset(dataset_name, split=train_split)
    val_data = load_dataset(dataset_name, split=val_split)
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "collate_fn": collate_fn(processor)}
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    max_train_batches = cfg.get("training", {}).get("max_train_batches")
    max_val_batches = cfg.get("training", {}).get("max_val_batches")
    if dry_run:
        max_train_batches = cfg.get("dry_run", {}).get("max_train_batches", 2)
        max_val_batches = cfg.get("dry_run", {}).get("max_val_batches", 2)

    epochs = 1 if dry_run else int(cfg["training"].get("epochs", 1))
    train_loss = 0.0
    steps = 0

    model.train()
    for _epoch in range(epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="train")):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

    val_metrics = evaluate(model, val_loader, device, int(max_val_batches) if max_val_batches is not None else None)
    checkpoint_path = output_dir / ("dry_run_checkpoint.pt" if dry_run else "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "dry_run": dry_run,
        },
        checkpoint_path,
    )

    metrics = {
        "method": cfg["model"]["method"],
        "dataset": data_cfg.get("dataset", "timm/eurosat-rgb"),
        "model": model_name,
        "dry_run": dry_run,
        "device": device,
        "train_loss": train_loss / max(steps, 1),
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_total": val_metrics["total"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path = output_dir / ("dry_run_metrics.json" if dry_run else "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dry_run = bool(args.dry_run or cfg.get("dry_run", {}).get("enabled", False))
    train(cfg, dry_run=dry_run)


if __name__ == "__main__":
    main()
