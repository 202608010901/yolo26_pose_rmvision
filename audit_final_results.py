"""Audit the saved best model at an explicit operating confidence threshold."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs" / "pose" / "final_cuda_640_b16_20261007"
OUTPUT = BASE / "outputs" / "full_training" / RUN.name


def main() -> None:
    cache = BASE.parent / "数据集" / "装甲板" / "dataset" / "val" / "labels.cache"
    cache_existed = cache.exists()
    try:
        model = YOLO(str(RUN / "weights" / "best.pt"))
        metrics = model.val(data=str(BASE / "armor_pose.yaml"), device=0, imgsz=640,
                            batch=32, workers=4, conf=0.25, half=True, plots=True,
                            project=str(BASE / "runs" / "pose"),
                            name="audit_conf025_20261007", exist_ok=False, fraction=1.0)
        matrix = metrics.confusion_matrix.matrix
        names = [model.names[i] for i in range(len(model.names))] + ["background"]
        with (OUTPUT / "confusion_matrix_conf025.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["predicted / true", *names])
            for name, row in zip(names, matrix):
                writer.writerow([name, *[int(x) for x in row]])
        diagonal = sum(matrix[i, i] for i in range(len(model.names)))
        total_gt = matrix[:, :-1].sum()
        total_pred = matrix[:-1, :].sum()
        result = {"confidence_threshold": 0.25, "matching_iou_threshold": 0.45,
                  "validation_images": 1409, "validation_instances": int(total_gt),
                  "correct_class_matches": int(diagonal),
                  "all_predictions": int(total_pred),
                  "micro_precision_at_conf025_iou045": float(diagonal / total_pred),
                  "micro_recall_at_conf025_iou045": float(diagonal / total_gt),
                  "metrics_with_confidence_prefilter": {k: float(v) for k, v in metrics.results_dict.items()},
                  "note": "固定阈值混淆矩阵审计。截断低置信度预测后的 mAP 不能替代标准完整 PR 验证指标。",
                  "per_class_confusion": [{"class": names[i], "gt_instances": int(matrix[:, i].sum()),
                                           "correct": int(matrix[i, i]),
                                           "recall": float(matrix[i, i] / matrix[:, i].sum())}
                                          for i in range(len(model.names))]}
        (OUTPUT / "operating_threshold_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if not cache_existed and cache.exists():
            cache.unlink()


if __name__ == "__main__":
    main()
