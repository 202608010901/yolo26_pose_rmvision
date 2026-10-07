"""Reusable configuration, device, and dataset helpers for YOLO-Pose training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load the training YAML and resolve paths relative to the YAML file."""
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def resolve(value: str | None) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()

    config["data"] = resolve(config["data"])
    config["model"] = resolve(config["model"])
    config["resume"] = resolve(config.get("resume"))
    config["output"]["project"] = resolve(config["output"]["project"])
    return config


def select_device(requested: str) -> str:
    """Select CUDA or CPU, either automatically or from an explicit choice."""
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境没有可用 CUDA；请检查 GPU 环境。")
    if requested in ("cuda", "cpu"):
        return "0" if requested == "cuda" else "cpu"
    raise ValueError(f"不支持的设备设置：{requested}，请选择 auto、cuda 或 cpu。")


def read_final_metrics(csv_path: Path) -> dict[str, float]:
    """Read available box and pose mAP values from the final results.csv row."""
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    final = {key.strip(): value for key, value in rows[-1].items() if key is not None}
    metrics: dict[str, float] = {}
    for key in (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
    ):
        try:
            metrics[key] = float(final[key])
        except (KeyError, TypeError, ValueError):
            continue
    return metrics


def dataset_label_cache_paths(data_yaml: Path) -> list[Path]:
    """Find YOLO label-cache paths for the configured train and validation splits."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    cache_paths: list[Path] = []
    for split in ("train", "val"):
        split_value = config.get(split)
        if not split_value:
            continue
        image_path = Path(split_value)
        if not image_path.is_absolute():
            image_path = root / image_path
        cache_path = image_path.resolve().parent / "labels.cache"
        if cache_path not in cache_paths:
            cache_paths.append(cache_path)
    return cache_paths
