from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent


def read_results(csv_path: Path) -> tuple[list[int], dict[str, list[float]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {key.strip(): value for key, value in row.items() if key is not None}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"训练结果为空：{csv_path}")

    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[key] = values
    epochs = list(range(1, len(rows) + 1))
    return epochs, columns


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制训练损失和框/Pose mAP 曲线。")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到训练 CSV：{csv_path}")
    epochs, columns = read_results(csv_path)
    marker = "o" if len(epochs) <= 2 else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    loss_keys = (
        ("train/box_loss", "Train box loss"),
        ("val/box_loss", "Validation box loss"),
        ("train/pose_loss", "Train keypoint loss"),
        ("val/pose_loss", "Validation keypoint loss"),
    )
    loss_count = 0
    for key, label in loss_keys:
        if key in columns:
            axes[0].plot(epochs, columns[key], linewidth=2, marker=marker, label=label)
            loss_count += 1
    if not loss_count:
        raise ValueError(f"{csv_path} 中没有识别到 box/pose loss 列。")
    axes[0].set_title("Training and validation losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    metric_keys = (
        ("metrics/mAP50-95(B)", "Box mAP50-95"),
        ("metrics/mAP50-95(P)", "Keypoint mAP50-95"),
    )
    metric_count = 0
    for key, label in metric_keys:
        if key in columns:
            axes[1].plot(epochs, columns[key], linewidth=2.2, marker=marker, label=label)
            metric_count += 1
    if not metric_count:
        raise ValueError(f"{csv_path} 中没有识别到 box/pose mAP 列。")
    axes[1].set_title("Validation mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    output_path = run_dir / "training_curves_custom.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"自定义训练曲线：{output_path}")
    print(f"Ultralytics 原生曲线：{run_dir / 'results.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
