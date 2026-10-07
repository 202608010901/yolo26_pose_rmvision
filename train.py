from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from train_utils import (
    dataset_label_cache_paths,
    load_config,
    read_final_metrics,
    select_device,
)

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 YAML 配置训练 YOLO26-Pose。")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "train_config.yaml")
    config = load_config(parser.parse_args().config.resolve())
    train = config["train"]
    output = config["output"]
    data_yaml = config["data"]
    run_dir = output["project"] / output["name"]

    device = select_device(config["device"])
    output["project"].mkdir(parents=True, exist_ok=True)
    label_cache_paths = dataset_label_cache_paths(data_yaml)
    existing_label_caches = {path for path in label_cache_paths if path.exists()}
    model = YOLO(str(config["resume"] or config["model"]))
    epoch_starts: dict[int, float] = {}
    epoch_seconds: list[float] = []

    def on_train_epoch_start(trainer: Any) -> None:
        epoch_starts[int(trainer.epoch)] = time.perf_counter()

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(trainer.epoch)
        start = epoch_starts.get(epoch)
        if start is not None:
            duration = time.perf_counter() - start
            epoch_seconds.append(duration)
            progress = {
                "epochs_completed": epoch + 1,
                "epochs_requested": train["epochs"],
                "latest_epoch_seconds": duration,
                "average_epoch_seconds": statistics.mean(epoch_seconds),
                "validation_metrics": {key: float(value) for key, value in trainer.metrics.items()},
                "best_epoch": trainer.stopper.best_epoch,
            }
            progress_path = Path(trainer.save_dir) / "progress.json"
            progress_tmp = progress_path.with_suffix(".tmp")
            progress_tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
            progress_tmp.replace(progress_path)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    total_started = time.perf_counter()
    training_error: Exception | None = None
    try:
        model.train(
            data=str(data_yaml), device=device, resume=bool(config["resume"]),
            project=str(output["project"]), name=output["name"],
            exist_ok=output["exist_ok"], **train,
        )
    except Exception as exc:
        training_error = exc
    finally:
        elapsed = time.perf_counter() - total_started
        for cache_path in label_cache_paths:
            if cache_path not in existing_label_caches and cache_path.exists():
                try:
                    cache_path.unlink()
                except OSError as exc:
                    print(f"无法删除临时标签缓存 {cache_path}：{exc}")

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", run_dir)).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    measured_epochs = len(epoch_seconds)
    avg_epoch_seconds = (
        statistics.mean(epoch_seconds)
        if epoch_seconds
        else (elapsed / max(measured_epochs, 1))
    )
    projected_hours = avg_epoch_seconds * train["epochs"] / 3600
    summary = {
        "model": str(config["model"]),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "0" else None,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "imgsz": train["imgsz"],
        "batch": train["batch"],
        "workers": train["workers"],
        "epochs_requested": train["epochs"],
        "epochs_measured": measured_epochs,
        "epoch_seconds": epoch_seconds,
        "average_epoch_seconds": avg_epoch_seconds,
        "elapsed_seconds": elapsed,
        "estimated_max_hours_without_early_stop": projected_hours,
        "final_epoch_metrics": read_final_metrics(save_dir / "results.csv"),
        "best_validation_metrics": {
            key: float(value)
            for key, value in getattr(getattr(model, "metrics", None), "results_dict", {}).items()
        },
        "training_error": repr(training_error) if training_error is not None else None,
        "note": "训练时长受早停、系统负载和验证耗时影响。",
    }
    summary_path = save_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"训练输出：{save_dir}")
    print(f"耗时：{elapsed / 60:.1f} 分钟；训练摘要：{summary_path}")
    if training_error is not None:
        print(f"训练未完成：{training_error!r}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
