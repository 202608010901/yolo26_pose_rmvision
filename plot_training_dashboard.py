"""Export training curves from measured CSV/JSON records as PNG and vector PDF."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

BASE = Path(__file__).resolve().parent
BLUE = "#2563A6"
ORANGE = "#E78424"
TEAL = "#138A82"
RED = "#B65166"
INK = "#20334B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Empty results.csv")
    columns = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    epochs = columns["epoch"]
    if not np.all(np.diff(epochs) > 0):
        raise ValueError("Epoch records must be strictly increasing")
    summary_path = run / "training_summary.json"
    if not summary_path.exists():
        summary_path = run / "resource_summary.json"  # Older runs sampled memory here too.
    resource = json.loads(summary_path.read_text(encoding="utf-8"))
    out = BASE / "outputs" / "full_training" / run.name / "curves"
    out.mkdir(parents=True, exist_ok=True)
    pdf_dir = BASE / "output" / "pdf" / run.name
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "training_curves.pdf"
    for file in ("msyh.ttc", "msyhbd.ttc"):
        font = Path("C:/Windows/Fonts") / file
        if font.is_file():
            font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 12,
                         "axes.titlesize": 14, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": "#5B6878", "ytick.color": "#5B6878",
                         "axes.edgecolor": "#CAD2DB", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    saved = []
    smooth_window = 5
    close_epoch = int(epochs[-1]) - 10 + 1

    def average(values):
        return np.array([np.mean(values[max(0, i-smooth_window+1):i+1]) for i in range(len(values))])

    def setup(ax, title, ylabel="Loss", percent=False):
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("训练轮次 / Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, int(epochs[-1]))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.grid(axis="both", color="#E4E8EE", linewidth=0.65)
        ax.set_axisbelow(True)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if len(epochs) >= 20:
            ax.axvspan(close_epoch - 0.5, epochs[-1], facecolor="#EEF3F9", alpha=0.7, zorder=0)
            ax.axvline(close_epoch, color="#A3AFBD", linewidth=1, linestyle=":")

    def curve(ax, key, label, color, style="-", smooth=True):
        values = columns[key]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in {key}")
        if smooth:
            ax.plot(epochs, values, color=color, alpha=0.25, linewidth=1.1, linestyle=style)
            ax.plot(epochs, average(values), color=color, linewidth=2.0, linestyle=style, label=label)
        else:
            ax.plot(epochs, values, color=color, linewidth=1.8, linestyle=style, label=label)
        ax.scatter(epochs[-1], values[-1], color=color, s=20, zorder=5, clip_on=False)

    def loss(ax, key, title):
        setup(ax, title)
        curve(ax, f"train/{key}", "训练集", BLUE)
        curve(ax, f"val/{key}", "验证集", ORANGE)
        ax.legend(frameon=False, fontsize=10)

    def maps(ax, task, title):
        setup(ax, title, "验证集 mAP", True)
        curve(ax, f"metrics/mAP50({task})", "mAP50", BLUE)
        curve(ax, f"metrics/mAP50-95({task})", "mAP50-95", TEAL)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.text(0.97, 0.08, "末轮原始值\n" + "\n".join(
            f"{key}: {columns[f'metrics/{key}({task})'][-1]:.2%}" for key in ["mAP50", "mAP50-95"]),
            transform=ax.transAxes, ha="right", fontsize=10, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 6})

    def figure(title, subtitle, shape=(2, 2), size=(14, 8.8)):
        fig, axes = plt.subplots(*shape, figsize=size)
        fig.subplots_adjust(left=0.065, right=0.97, bottom=0.12, top=0.83, hspace=0.48, wspace=0.3)
        fig.text(0.065, 0.957, title, fontsize=23, weight="bold", color=INK)
        fig.text(0.065, 0.904, subtitle, fontsize=10.5, color="#66758A")
        fig.text(0.065, 0.035, "淡线：每轮原始值；粗线：向后 5 轮均值。浅色区域：第 91-100 轮关闭 Mosaic。",
                 fontsize=9.5, color="#66758A")
        return fig, np.asarray(axes).ravel()

    def save(fig, name, pdf):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=300)
        pdf.savefig(fig)
        saved.append(path)
        plt.close(fig)

    subtitle = "YOLO26n-Pose | RTX 5060 Laptop | 640 / batch 16 | 100 epochs | train 5,636 / val 1,409"
    with PdfPages(pdf_path, metadata={"Title": "装甲板全量 GPU 训练曲线", "Author": "Training records",
                                    "Subject": run.name}) as pdf:
        fig, ax = figure("装甲板训练曲线总览", subtitle, (2, 3), (16, 9.6))
        loss(ax[0], "box_loss", "01  装甲板框损失")
        loss(ax[1], "pose_loss", "02  四角点损失")
        loss(ax[2], "cls_loss", "03  类别损失")
        maps(ax[3], "B", "04  装甲板框 mAP")
        maps(ax[4], "P", "05  四角点 Pose mAP")
        setup(ax[5], "06  验证集 Precision / Recall", "指标", True)
        for task, prefix, color in [("B", "框", BLUE), ("P", "角点", TEAL)]:
            curve(ax[5], f"metrics/precision({task})", prefix + " Precision", color)
            curve(ax[5], f"metrics/recall({task})", prefix + " Recall", color, "--")
        ax[5].set_ylim(0, 1.02)
        ax[5].legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
        save(fig, "01_training_overview", pdf)

        fig, ax = figure("训练与验证损失", subtitle)
        for axis, key, title in zip(ax, ["box_loss", "pose_loss", "cls_loss", "l1_loss"],
                                    ["装甲板框 / Box loss", "四角点 / Pose loss", "类别 / Classification loss", "坐标回归 / L1 loss"]):
            loss(axis, key, title)
        save(fig, "02_training_validation_losses", pdf)

        fig, ax = figure("验证集检测与角点指标", "所有指标来自每轮验证集记录；Pose 的 OKS 与框 IoU 评估尺度不同，数值不能直接比较。")
        maps(ax[0], "B", "装甲板框 mAP")
        maps(ax[1], "P", "四角点 Pose mAP")
        for axis, task, title in [(ax[2], "B", "装甲板框 Precision / Recall"), (ax[3], "P", "四角点 Precision / Recall")]:
            setup(axis, title, "指标", True)
            curve(axis, f"metrics/precision({task})", "Precision", BLUE)
            curve(axis, f"metrics/recall({task})", "Recall", ORANGE)
            axis.set_ylim(0, 1.02)
            axis.legend(frameon=False, fontsize=10, loc="lower right")
        save(fig, "03_validation_metrics", pdf)

        fig, ax = figure("训练后期收敛细节", "放大第 60-100 轮的原始记录；纵轴使用局部范围，便于观察微小变化。")
        fig.texts[-1].remove()
        fig.text(0.065, 0.035, "本页全部为原始值，无平滑。浅色区域：第 91-100 轮关闭 Mosaic。", fontsize=9.5,
                 color="#66758A", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        for axis, task, label in [(ax[0], "B", "框 mAP50-95"), (ax[1], "P", "角点 mAP50-95")]:
            key = f"metrics/mAP50-95({task})"
            setup(axis, label, "验证集 mAP", True)
            curve(axis, key, label, BLUE if task == "B" else TEAL, smooth=False)
            axis.set_xlim(60, 100)
            axis.set_xticks([60, 70, 80, 90, 100])
            tail = columns[key][epochs >= 60]
            span = max(np.ptp(tail), 0.004)
            axis.set_ylim(tail.min()-span*0.18, tail.max()+span*0.25)
            gain = 100 * (columns[key][-1] - columns[key][np.where(epochs == 80)[0][0]])
            axis.text(0.04, 0.92, f"80 → 100 轮：+{gain:.3f} 个百分点", transform=axis.transAxes, fontsize=10)
        for axis, key, title in [(ax[2], "box_loss", "后期框损失"), (ax[3], "pose_loss", "后期角点损失")]:
            setup(axis, title)
            for prefix, label, color in [("train", "训练集", BLUE), ("val", "验证集", ORANGE)]:
                curve(axis, f"{prefix}/{key}", label, color, smooth=False)
            axis.set_xlim(60, 100)
            axis.set_xticks([60, 70, 80, 90, 100])
            tail = np.concatenate([columns[f"{prefix}/{key}"][epochs >= 60] for prefix in ["train", "val"]])
            span = max(np.ptp(tail), 0.04)
            axis.set_ylim(tail.min()-span*0.1, tail.max()+span*0.2)
            axis.legend(frameon=False, fontsize=10)
        save(fig, "04_late_training_convergence", pdf)

        durations = np.asarray(resource["epoch_seconds"], dtype=float)
        if len(durations) != len(epochs) or np.any(durations <= 0):
            raise ValueError("Epoch timing does not match results.csv")
        fig, ax = figure("学习率与训练耗时", subtitle)
        fig.texts[-1].remove()
        fig.text(0.065, 0.035, "学习率与时间均使用原始记录。单轮耗时包括训练、验证和回调记录开销。", fontsize=9.5,
                 color="#66758A", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        setup(ax[0], "学习率调度", "Learning rate")
        lr_keys = [key for key in columns if key.startswith("lr/")]
        if all(np.array_equal(columns[key], columns[lr_keys[0]]) for key in lr_keys):
            curve(ax[0], lr_keys[0], "参数组 0 / 1 / 2（相同）", BLUE, smooth=False)
        else:
            for key, color in zip(lr_keys, [BLUE, TEAL, ORANGE]):
                curve(ax[0], key, key, color, smooth=False)
        ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[0].legend(frameon=False, fontsize=10)
        setup(ax[1], "每轮训练与验证耗时", "秒 / epoch")
        ax[1].plot(epochs, durations, color=BLUE, linewidth=1.5)
        median = np.median(durations)
        ax[1].axhline(median, color=ORANGE, linestyle="--", label=f"中位数 {median:.1f} 秒")
        ax[1].legend(frameon=False, fontsize=10)
        setup(ax[2], "已记录轮次的累计耗时", "分钟")
        ax[2].plot(epochs, np.cumsum(durations)/60, color=TEAL, linewidth=2)
        ax[2].text(0.04, 0.88, f"训练流程总耗时：{resource['elapsed_seconds']/60:.2f} 分钟", transform=ax[2].transAxes, fontsize=10)
        setup(ax[3], "验证 mAP 随训练时间的变化", "验证集 mAP50-95", True)
        ax[3].set_xlabel("逐轮累计时间 / 分钟")
        minutes = np.cumsum(durations)/60
        for task, label, color in [("B", "框", BLUE), ("P", "角点", TEAL)]:
            ax[3].plot(minutes, columns[f"metrics/mAP50-95({task})"], color=color, linewidth=1.8, label=label)
        ax[3].set_xlim(0, minutes[-1])
        # This panel uses time, so epoch-specific annotations must be removed.
        for patch in list(ax[3].patches):
            patch.remove()
        for line in list(ax[3].lines):
            if line.get_linestyle() == ":":
                line.remove()
        ax[3].set_ylim(0, 1.02)
        ax[3].legend(frameon=False, fontsize=10, loc="lower right")
        save(fig, "05_learning_rate_and_timing", pdf)

    shutil.copy2(run / "results.csv", out / "results.csv")
    metadata = {"run": str(run), "epochs": len(epochs), "smoothing": "trailing mean, window=5; raw values retained",
                "metric_source": "per-epoch results.csv; differs slightly from best.pt final reevaluation",
                "timing_source": "training_summary.json epoch_seconds", "pdf": str(pdf_path),
                "png": [str(path) for path in saved]}
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text("完整训练曲线（100 轮）\n\n01 总览；02 训练/验证损失；03 验证指标；04 后期收敛；05 学习率与时间。\nPNG 为 300 DPI，PDF 保留矢量曲线。淡线为每轮原始数据，粗线为向后 5 轮均值；后期细节页、学习率和时间页不平滑。末轮标注取原始 CSV 值，与 best.pt 最终复评有轻微差异。\n关闭 Mosaic 标记对应第 91-100 轮。kobj_loss、rle_loss 在本次记录均为 0，不作为有变化的损失曲线绘制。\n", encoding="utf-8")
    package = out / "training_curves_bundle.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in saved + [out / "results.csv", out / "plot_metadata.json", out / "README.txt", pdf_path, Path(__file__)]:
            archive.write(path, path.name)
    print(json.dumps(metadata | {"zip": str(package)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
