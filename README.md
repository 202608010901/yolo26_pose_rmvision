# RoboMaster 装甲板 YOLO26-Pose 项目

基于 YOLO26n-Pose 微调，预测装甲板类别、边界框和四个角点。数据集保持原位置，训练参数与路径统一由 YAML 配置管理。

## 环境依赖配置

要求 Python 3.10+。依赖分两套，按运行设备选择：

| 文件 | 适用场景 | 说明 |
| --- | --- | --- |
| `requirements-cuda.txt` | NVIDIA GPU | torch/torchvision 为 cu130 wheel，显卡驱动需支持对应 CUDA 版本；不匹配时修改文件中的 `--extra-index-url` |
| `requirements.txt` | CPU / macOS（MPS） | 通用安装 |
| `requirements-lock.txt` | 复现 | 完整锁定版本，按需使用 |

CUDA（Windows 示例）：

```powershell
python -m venv .venv-cuda
.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

CPU / macOS：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

数据与权重准备：

- 编辑 `armor_pose.yaml`，把 `path` 指向数据集根目录（绝对或相对路径均可），`train`/`val` 为图片子目录；标签放在与 `images/` 平级的 `labels/` 目录。
- 标签每行 13 列：`class cx cy w h x1 y1 x2 y2 x3 y3 x4 y4`，坐标均为 0–1 归一化值，四角点顺序为左上、左下、右下、右上。
- `kpt_shape: [4, 2]` 表示 4 个关键点；`flip_idx` 定义水平翻转时四点对应的交换关系；`names` 为类别列表。
- 首次运行会自动下载 `yolo26n-pose.pt` 预训练权重，也可提前放到项目根目录。
- `train_config.yaml` 的 `device` 可选 `auto` / `cuda` / `cpu`；`auto` 在无 CUDA 时自动回退 CPU。

## 代码树说明

```text
yolo26_pose_tutorial/
├─ armor_pose.yaml             # 数据集定义：路径、类别、关键点形状与翻转索引
├─ train_config.yaml           # 训练配置：数据/模型/设备/输出目录以及全部训练与增强参数
├─ train.py                    # 训练入口：读配置、选设备、逐轮记录耗时、写训练摘要、清理临时标签缓存
├─ train_utils.py              # 公共工具：配置解析、设备选择、指标读取、标签缓存路径
├─ run_full_training.py        # 一键流水线：数据检查 → 训练 → 自定义曲线 → 最佳权重示例预测；写状态并在训练期间保持系统唤醒
├─ check_dataset.py            # 只读数据检查：列数、类别、坐标范围、图片可读性、图片与标签配对
├─ analyze_dataset.py          # 生成类别分布、目标尺寸分布和验证集标注预览图
├─ predict.py                  # 用指定权重对图片或目录推理，保存可视化结果
├─ plot_results.py             # 由 results.csv 绘制损失与框/角点 mAP 曲线
├─ plot_training_dashboard.py  # 由 results.csv 与 training_summary.json 导出中文训练看板（PNG + 矢量 PDF）
├─ summarize_training.py       # 汇总某次已完成训练：Markdown 报告、逐类别 CSV、交付 zip
├─ audit_final_results.py      # 用固定 conf=0.25 复核 best.pt 的混淆矩阵与微平均指标
├─ requirements*.txt           # 依赖清单
└─ README.md
```

注意：`plot_training_dashboard.py`、`summarize_training.py`、`audit_final_results.py` 针对具体运行写死了 run 名或路径，换实验时需修改脚本顶部常量。

## 快速开始

```powershell
# 1. 检查数据（只读）
.venv-cuda\Scripts\python.exe check_dataset.py

# 2. 数据分析图（可选）
.venv-cuda\Scripts\python.exe analyze_dataset.py

# 3. 编辑 train_config.yaml：每次新实验设置唯一的 output.name；续训把 resume 指向对应 last.pt

# 4. 训练（或使用一键流水线 run_full_training.py）
.venv-cuda\Scripts\python.exe -u train.py --config train_config.yaml
.venv-cuda\Scripts\python.exe -u run_full_training.py --config train_config.yaml

# 5. 绘制曲线与推理
.venv-cuda\Scripts\python.exe plot_results.py --run-dir runs/pose/<run-name>
.venv-cuda\Scripts\python.exe predict.py --weights runs/pose/<run-name>/weights/best.pt --source <图片或目录>
```

训练配置说明：`train_config.yaml` 中 `output.project` / `output.name` 决定运行目录；`train` 段包含轮数、早停、图像尺寸、批量、数据增强等全部参数（文件内有逐项注释）；`patience` 为连续无提升的早停轮数；`fraction: 1.0` 表示使用完整训练集。在线增强只在训练时作用于 batch，不会写入或扩充原始数据。

## 输出说明

### `runs/pose/<run-name>/` — 训练运行目录

Ultralytics 原生输出加本项目扩展：

- `weights/best.pt`、`weights/last.pt`：最佳与最终权重
- `results.csv`、`results.png`：逐轮损失与指标记录
- `Box*_curve.png`、`Pose*_curve.png`、`confusion_matrix*.png`、`labels.jpg`、`train_batch*/`、`val_batch*_*`：框架自动生成的曲线、混淆矩阵与批次可视化
- `args.yaml`：本次训练完整参数快照
- `progress.json`：本项目回调写入的完成轮数、单轮耗时、验证指标与最佳轮次
- `training_summary.json`：环境版本、设备、耗时、最终与最佳指标汇总
- `training_curves_custom.png`：`plot_results.py` 生成的自定义曲线

### `outputs/` — 项目输出

- `analysis/`：`class_distribution.png`、`target_size_distribution.png`、`annotation_preview.jpg`
- `predictions/<name>/`：`predict.py` 保存的推理可视化图片
- `full_training/<run-name>/`：`run_full_training.py` 流水线产物
  - `status.json`：当前步骤、状态、起止时间
  - `dataset_check.log`、`train.log`、`plot.log`、`predict.log`：各步骤完整日志
  - `training_report.md`、`training_summary.json`、`per_class_validation.csv`：汇总报告与结构化指标
  - `operating_threshold_audit.json`、`confusion_matrix_conf025.csv`、`threshold_audit.log`：固定置信度审计结果
  - `curves/`：训练看板 PNG、`results.csv` 副本、元数据与打包 zip
  - `armor_pose_cuda_training_results.zip`：模型、参数、图表、报告的交付包
- `video/`：演示视频脚本

### `output/pdf/<run-name>/`

`plot_training_dashboard.py` 生成的矢量 PDF 训练曲线（目录是单数 `output`，与 `outputs/` 区分）。

## 已完成训练

项目已完成一次 100 轮全量训练，运行名为 `final_cuda_640_b16_20261007`：640 输入、batch 16、patience 30、完整 train/val 划分，未触发早停。`best.pt` 全验证集复评指标：

| 项目 | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 装甲板框 | 93.79% | 90.78% | 95.20% | 67.70% |
| 四角点 Pose | 86.09% | 83.89% | 83.30% | 76.47% |

详细报告见 `outputs/full_training/final_cuda_640_b16_20261007/training_report.md`。Pose mAP 使用关键点 OKS 评估，与框 IoU mAP 尺度不同，两者不可直接比较；本项目只有 train/val，没有独立测试集，以上均为验证集指标。

## 其他说明

- `runs/`、`outputs/`、`*.pt`、虚拟环境目录已在 `.gitignore` 中忽略，仓库只保留代码与配置。
- 训练时 Ultralytics 会在数据目录生成 `labels.cache` 标签缓存；`train.py` 只清理本次新建的缓存，不会修改图片和标签。
- 数据检查发现损坏图片时只报警告，Ultralytics 数据校验会跳过该图片及其标签。
