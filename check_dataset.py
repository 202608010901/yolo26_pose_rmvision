from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = BASE_DIR / "armor_pose.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_FIELDS = 13


def resolve_dataset(data_yaml: Path) -> tuple[dict, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"数据 YAML 格式不正确：{data_yaml}")

    configured_root = Path(data.get("path", data_yaml.parent))
    root = configured_root if configured_root.is_absolute() else data_yaml.parent / configured_root
    return data, root.resolve()


def image_dir_for(root: Path, split_value: str | Path) -> Path:
    split_path = Path(split_value)
    return (split_path if split_path.is_absolute() else root / split_path).resolve()


def parse_label_file(
    label_path: Path,
    class_count: int,
    class_instances: Counter[int],
    class_images: defaultdict[int, set[str]],
) -> list[str]:
    errors: list[str] = []
    for row_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != EXPECTED_FIELDS:
            errors.append(
                f"{label_path}:{row_number}: 应为 13 列，实际 {len(fields)} 列"
            )
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            errors.append(f"{label_path}:{row_number}: 存在非数字字段")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{label_path}:{row_number}: 存在非有限数值")
            continue
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            errors.append(
                f"{label_path}:{row_number}: 类别编号 {values[0]} 不在 0 到 {class_count - 1} 之间"
            )
            continue
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            errors.append(f"{label_path}:{row_number}: 坐标不在 0 到 1 之间")
            continue
        if values[3] <= 0 or values[4] <= 0:
            errors.append(f"{label_path}:{row_number}: 边界框宽高必须大于 0")
            continue
        class_instances[class_id] += 1
        class_images[class_id].add(label_path.stem)
    return errors


def inspect_split(
    split_name: str,
    image_dir: Path,
    names: list[str],
) -> tuple[list[str], int, int, Counter[int], defaultdict[int, set[str]]]:
    label_dir = image_dir.parent / "labels"
    errors: list[str] = []
    warnings: list[str] = []
    class_instances: Counter[int] = Counter()
    class_images: defaultdict[int, set[str]] = defaultdict(set)
    image_paths = sorted(
        (path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path),
    )
    label_paths = sorted(label_dir.rglob("*.txt"), key=lambda path: str(path))
    images_by_stem = {path.stem: path for path in image_paths}
    labels_by_stem = {path.stem: path for path in label_paths}

    for stem in sorted(images_by_stem.keys() - labels_by_stem.keys()):
        errors.append(f"{split_name}: 图片缺少同名标签：{images_by_stem[stem]}")
    for stem in sorted(labels_by_stem.keys() - images_by_stem.keys()):
        errors.append(f"{split_name}: 标签缺少同名图片：{labels_by_stem[stem]}")

    valid_images = 0
    corrupt_images = 0
    for stem, image_path in images_by_stem.items():
        try:
            with Image.open(image_path) as image:
                image.verify()
            valid_images += 1
        except Exception as exc:
            corrupt_images += 1
            warnings.append(f"{split_name}: 无法读取图片，将由训练框架跳过：{image_path} ({exc})")
            continue

        label_path = labels_by_stem.get(stem)
        if label_path is None:
            continue
        row_errors = parse_label_file(
            label_path, len(names), class_instances, class_images
        )
        errors.extend(row_errors)

    print(
        f"{split_name}: 图片 {len(image_paths)}，可读 {valid_images}，"
        f"损坏 {corrupt_images}，标签 {len(label_paths)}"
    )
    return warnings + errors, len(image_paths), valid_images, class_instances, class_images


def main() -> int:
    parser = argparse.ArgumentParser(description="只读检查 RoboMaster YOLO-Pose 数据。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    args = parser.parse_args()

    data, root = resolve_dataset(args.data.resolve())
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    else:
        names = [str(raw_names[index]) for index in range(len(raw_names))]
    if not names:
        raise ValueError("data.yaml 中没有 names 类别列表")
    if data.get("kpt_shape") != [4, 2]:
        raise ValueError(f"kpt_shape 应为 [4, 2]，实际为 {data.get('kpt_shape')}")

    all_errors: list[str] = []
    valid_total = 0
    for split in ("train", "val"):
        if split not in data:
            raise ValueError(f"data.yaml 缺少 {split} 路径")
        image_dir = image_dir_for(root, data[split])
        if not image_dir.is_dir():
            raise FileNotFoundError(f"找不到 {split} 图片目录：{image_dir}")
        split_messages, _, valid_images, _, _ = inspect_split(split, image_dir, names)
        valid_total += valid_images
        all_errors.extend(split_messages)

    hard_errors = [message for message in all_errors if "应为 13 列" in message or "类别编号" in message or "坐标不在" in message or "宽高必须" in message or "图片缺少" in message or "标签缺少" in message or "非数字" in message or "非有限数值" in message]
    warnings = [message for message in all_errors if message not in hard_errors]
    for message in warnings:
        print(f"警告：{message}")
    for message in hard_errors:
        print(f"错误：{message}")

    print(f"类别数：{len(names)}；关键点形状：{data['kpt_shape']}")
    print(f"坐标顺序：类别、框 cx cy w h、四个关键点的 x y，共 {EXPECTED_FIELDS} 列")
    if hard_errors:
        print(f"检查失败：{len(hard_errors)} 项结构或标签错误。")
        return 1
    print(f"检查完成：原数据未修改；可读图片共 {valid_total} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
