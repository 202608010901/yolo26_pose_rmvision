from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"


def select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，推理改用 CPU。")
            return "cpu"
        return "0" if requested == "cuda" else "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="在图片或图片目录上运行装甲板 Pose 推理。")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--name", default="validation_examples")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"找不到权重文件：{args.weights}")
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"找不到推理输入：{source}")

    model = YOLO(str(args.weights.resolve()))
    device = select_device(args.device)
    results = model.predict(
        source=str(source.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        project=str(OUTPUT_DIR),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    print(f"设备：{device}")
    print(f"已处理图片：{len(results)}")
    print(f"可视化结果：{OUTPUT_DIR / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
