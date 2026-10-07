"""Build a reproducible Markdown/JSON summary and compact training delivery archive."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

BASE = Path(__file__).resolve().parent
NAME = "final_cuda_640_b16_20261007"
RUN = BASE / "runs" / "pose" / NAME
OUT = BASE / "outputs" / "full_training" / NAME


def main() -> None:
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    resource = json.loads((RUN / "resource_summary.json").read_text(encoding="utf-8"))
    with (RUN / "results.csv").open(encoding="utf-8", newline="") as f:
        epochs = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    if status["status"] != "completed" or len(epochs) != 100 or resource["training_error"]:
        raise RuntimeError("Training is not successfully complete; refuse a completed report.")
    rows = {}
    pattern = re.compile(r"^\s+(B[1-5OS]|R[1-5OS])\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    columns = ["class", "images", "instances", "box_precision", "box_recall", "box_map50",
               "box_map50_95", "pose_precision", "pose_recall", "pose_map50", "pose_map50_95"]
    for line in (OUT / "train.log").read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            values = match.groups()
            rows[values[0]] = dict(zip(columns, [values[0], int(values[1]), int(values[2]),
                                               *[float(v) for v in values[3:]]]))
    if len(rows) != 14 or sum(r["instances"] for r in rows.values()) != 2362:
        raise RuntimeError("Final per-class validation records are incomplete.")
    with (OUT / "per_class_validation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())
    metric = resource["best_validation_metrics"]
    best_epoch = int(max(epochs, key=lambda r: r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(P)"])["epoch"])
    sha = hashlib.sha256((RUN / "weights" / "best.pt").read_bytes()).hexdigest()
    audit = json.loads((OUT / "operating_threshold_audit.json").read_text(encoding="utf-8"))
    pipeline_seconds = (datetime.fromisoformat(status["finished_at"]) - datetime.fromisoformat(status["started_at"])).total_seconds()
    summary = {"status": "completed", "run": NAME, "best_epoch": best_epoch,
               "epochs_completed": len(epochs), "started_at": status["started_at"], "finished_at": status["finished_at"],
               "train_images_used": 5636, "corrupt_train_images_skipped": 1, "validation_images": 1409,
               "validation_instances": 2362, "independent_test_set": False,
               "config": status["config"], "gpu": resource["gpu_name"],
               "training_seconds": resource["elapsed_seconds"], "pipeline_seconds": pipeline_seconds,
               "best_validation_metrics": metric, "per_class_metrics_rounded_from_log": list(rows.values()),
               "best_weights": str(RUN / "weights" / "best.pt"), "best_weights_sha256": sha,
               "peak_cuda_allocated_gib": resource["peak_cuda_allocated_gib"],
               "peak_cuda_reserved_gib": resource["peak_cuda_reserved_gib"],
               "main_process_rss_peak_gib": resource["peak_process_rss_gib_sampled"],
               "operating_threshold_audit": audit}
    (OUT / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = lambda x: f"{100*x:.2f}%"
    box_gain = (epochs[-1]["metrics/mAP50-95(B)"] - epochs[79]["metrics/mAP50-95(B)"]) * 100
    pose_gain = (epochs[-1]["metrics/mAP50-95(P)"] - epochs[79]["metrics/mAP50-95(P)"]) * 100
    lines = ["# 装甲板 YOLO26-Pose 全量 GPU 训练结果", "",
             "完整完成 100/100 轮，未触发早停，训练无异常；最佳轮次为第 100 轮。最佳权重已完成全验证集评估和图片预测。", "",
             f"- 开始时间：{status['started_at']}；全部处理完成：{status['finished_at']}（北京时间）。",
             f"- 训练与最终验证耗时：{resource['elapsed_seconds']/60:.2f} 分钟；整个流程：{pipeline_seconds/60:.2f} 分钟。",
             f"- GPU：{resource['gpu_name']}；模型 YOLO26n-Pose，14 类，每个目标 4 个角点。",
             "- 数据：5,636 张可读训练图片，1,409 张验证图片 / 2,362 个装甲板实例；跳过 1 张 0 字节坏图。",
             "- 原 train/val 划分保留，fraction=1.0；没有把验证集并入训练。",
             "- 参数：640 输入，batch 16，100 epoch，patience 30，workers 4，seed 42，CUDA + AMP。",
             f"- CUDA 张量分配峰值 {resource['peak_cuda_allocated_gib']:.2f} GiB，缓存保留峰值 {resource['peak_cuda_reserved_gib']:.2f} GiB；主训练进程 RSS 峰值 {resource['peak_process_rss_gib_sampled']:.2f} GiB。进程内存不含数据加载子进程，CUDA 数字不含桌面等其他程序。", "",
             "## 最佳模型的标准验证指标", "",
             "指标取自训练结束后的 best.pt 全验证集复评。mAP 不等于分类准确率；Precision/Recall 为验证器曲线选取的统计值，并非固定 conf=0.25 的值。", "",
             "| 项目 | Precision | Recall | mAP50 | mAP50–95 |", "| --- | ---: | ---: | ---: | ---: |"]
    for task, label in [("B", "装甲板框"), ("P", "四角点 Pose")]:
        lines.append("| " + label + " | " + " | ".join(pct(metric[f"metrics/{k}({task})"]) for k in ["precision", "recall", "mAP50", "mAP50-95"]) + " |")
    lines.extend(["", "Pose mAP 使用关键点 OKS 评估，与框 IoU mAP 的尺度不同；两者数值不能直接比较。这里只能报告验证集表现，当前没有独立测试集，也未验证实际相机视频泛化。", "",
                  "## 收敛情况", "",
                  "前 20 轮提升明显，60–80 轮后趋于平台；训练/验证损失总体下降，曲线没有显示明显持续恶化的验证损失。",
                  f"第 80→100 轮，框 mAP50–95 增加 {box_gain:.3f} 个百分点，Pose 增加 {pose_gain:.3f} 个百分点；继续同配置长训的增益可能有限。第 91 轮关闭 mosaic 后训练损失下降，不能把这一变化单独解释为泛化提升。", "",
                  f"![训练曲线]({RUN.as_posix()}/training_curves_custom.png)", "",
                  "## 各类别验证表现", "",
                  "类别数据来自最终日志，保留日志三位小数精度；以下按 Pose mAP50–95 从低到高排序。", "",
                  "| 类别 | 验证实例 | 框 mAP50–95 | Pose mAP50–95 |", "| --- | ---: | ---: | ---: |"])
    for r in sorted(rows.values(), key=lambda r: r["pose_map50_95"]):
        lines.append(f"| {r['class']} | {r['instances']} | {pct(r['box_map50_95'])} | {pct(r['pose_map50_95'])} |")
    lines.extend(["", "角点表现的主要短板为 R1（56.4%）、B1（57.1%）、B5（61.7%）和 B3（61.8%）。RS、BO、RO 的角点指标较高，其中 BO 只有 36 个验证实例，不能把少量样本上的高分理解为充分泛化。", "",
                  "## 固定置信度混淆矩阵复核", "",
                  "原训练的混淆矩阵沿用验证 conf=0.001，按几何 IoU 跨类别匹配预测；大量低置信度候选会影响其对角线，不能直接当作标准 mAP 或固定部署阈值的分类准确率。保留原图，另用 best.pt 对全验证集做 conf=0.25 的审计。", "",
                  f"conf=0.25、框匹配 IoU>0.45：正确类别匹配 {audit['correct_class_matches']} / GT {audit['validation_instances']}，预测共 {audit['all_predictions']}；微平均 Precision {pct(audit['micro_precision_at_conf025_iou045'])}、Recall {pct(audit['micro_recall_at_conf025_iou045'])}。这项审计的阈值与标准 mAP 评估不同，不替代上表。", "",
                  f"![固定阈值混淆矩阵]({BASE.as_posix()}/runs/pose/audit_conf025_20261007/confusion_matrix_normalized.png)", "",
                  "## 可交付文件与使用", "",
                  f"- 最佳模型：`{RUN / 'weights' / 'best.pt'}`（{(RUN / 'weights' / 'best.pt').stat().st_size / 1024**2:.2f} MiB）。",
                  f"- SHA-256：`{sha}`。",
                  "- `last.pt` 为最终模型；训练结束时优化器已由框架剥离，作为模型快照保留，不能按未完成训练的 last.pt 方式恢复优化器状态。",
                  "- 本目录包含结构化汇总 JSON、类别 CSV、固定阈值审计 JSON/CSV；运行目录保留逐轮 CSV、参数、曲线和资源摘要。", "",
                  "推理示例（项目目录内）：", "", "```powershell",
                  f".venv-cuda\\Scripts\\python.exe predict.py --weights runs/pose/{NAME}/weights/best.pt --source D:/your_camera_image.jpg --device cuda --conf 0.25",
                  "```", "",
                  "验证批处理日志报告纯网络推理约 0.9 ms/图；单张示例报告 12.4 ms/图，且有预处理与后处理开销。它们不是包含相机采集、PnP、跟踪等链路的实时帧率测试。", "",
                  "## 后续优先顺序", "",
                  "1. 用实际相机或未参与训练/验证的视频检查远距离、运动模糊、遮挡，以及角点用于 PnP 后的稳定性，建立独立测试数据。",
                  "2. 优先补查 R1/B1 的角点标注与目标尺寸，分析低分样本；现有验证数值不能证明低分原因。",
                  "3. 若实际问题集中在小目标，比较 960 输入与当前 640 模型的角点误差和端到端时延，再决定是否进行更高分辨率训练。", ""])
    report = OUT / "training_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    package = OUT / "armor_pose_cuda_training_results.zip"
    files = {"model/best.pt": RUN / "weights" / "best.pt", "training/args.yaml": RUN / "args.yaml",
             "training/results.csv": RUN / "results.csv", "training/resource_summary.json": RUN / "resource_summary.json",
             "figures/training_curves.png": RUN / "training_curves_custom.png", "figures/results.png": RUN / "results.png",
             "figures/confusion_matrix_conf025.png": BASE / "runs" / "pose" / "audit_conf025_20261007" / "confusion_matrix_normalized.png",
             "examples/1004.jpg": BASE / "outputs" / "predictions" / NAME / "1004.jpg",
             "report/training_report.md": report, "report/training_summary.json": OUT / "training_summary.json",
             "report/per_class_validation.csv": OUT / "per_class_validation.csv",
             "report/operating_threshold_audit.json": OUT / "operating_threshold_audit.json",
             "report/confusion_matrix_conf025.csv": OUT / "confusion_matrix_conf025.csv",
             "reproduce/requirements-cuda.txt": BASE / "requirements-cuda.txt", "reproduce/train.py": BASE / "train.py",
             "reproduce/armor_pose.yaml": BASE / "armor_pose.yaml", "reproduce/predict.py": BASE / "predict.py"}
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if name == "report/training_report.md":
                portable = path.read_text(encoding="utf-8")
                portable = portable.replace(f"{RUN.as_posix()}/training_curves_custom.png", "../figures/training_curves.png")
                portable = portable.replace(f"{BASE.as_posix()}/runs/pose/audit_conf025_20261007/confusion_matrix_normalized.png", "../figures/confusion_matrix_conf025.png")
                archive.writestr(name, portable)
            else:
                archive.write(path, name)
        archive.writestr("README.txt", "装甲板 GPU 训练交付包\n\n最佳模型：model/best.pt\n结果报告：report/training_report.md\n训练曲线与固定阈值混淆矩阵：figures/\n结构化指标与参数：report/、training/\n\n本包不包含原始数据集，所有指标为验证集表现。\nreproduce/ 提供核心训练与推理代码；更换机器时请修改 armor_pose.yaml 的数据集路径，并准备 YOLO26n-Pose 预训练权重后复现训练。\n")
    print(json.dumps({"report": str(report), "package": str(package), "package_mib": package.stat().st_size/1024**2,
                      "best_epoch": best_epoch, "best_weights_sha256": sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
