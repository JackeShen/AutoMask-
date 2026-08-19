# AutoMask — YOLO 自动标注系统

基于 **PyQt5 + ultralytics (YOLO)** 的图形化自动标注工具。加载训练好的 YOLO
权重（`.pt`），对图片（单张或文件夹）做推理，自动生成标注并保存。

> 本程序在 `goal` conda 环境中开发与运行，默认 **CPU 推理**。

## 功能

1. **加载模型**：选择 YOLO 权重 `.pt`（可额外用逗号填写类别名覆盖）。
2. **选择图片**：支持单张图片或整个文件夹（批量）。
3. **选择标注保存路径**：指定标注文件输出目录。
4. **运行推理**：单张运行 / 批量运行，预览框选与（分割模型的）掩码。
5. **标注复核**：打开复核窗口，人工修正/增删检测框与分割多边形（移动、改类、加点删点）。
6. **导出标注**（可多选）：
   - YOLO txt（检测：`class cx cy w h`；分割：`class + 归一化多边形`）
   - Pascal VOC xml
   - COCO json（批量时汇总为 `annotations.coco.json`）
   - 可视化图片（带框/标签/掩码）
   - `classes.txt` / `dataset.yaml`

## 运行

```bash
conda activate goal
python main.py
```

操作流程：

1. 点【加载模型】→ 选 `.pt` 权重（类别名留空则使用模型自带 names）。
2. 点【选择图片】或【选择文件夹】导入待标注图片。
3. 点【浏览】选【标注保存路径】。
4. 勾选需要的导出格式与阈值（置信度 / IOU）。
5. 【运行当前图片】或【批量运行全部】。
6. 预览区可滚轮缩放；【保存当前结果】可单独保存当前图。

## 项目结构

```
automask/
├── main.py                  # 入口
├── build.bat               # PyInstaller 打包（CPU 版）
├── requirements_goal.txt   # 依赖记录
├── run.bat                 # 一键启动脚本（免 activate）
├── src/
│   ├── yolo_model.py       # YOLO 模型加载与推理封装（与 UI 解耦）
│   ├── annotation_io.py    # 标注保存（txt/xml/coco/可视化）
│   ├── main_window.py      # PyQt5 主界面（自动标注）
│   └── review_window.py    # 标注复核窗口（人工修正）
└── README.md
```

## 输出格式说明

- **YOLO txt**：每行一个目标。检测：`class cx cy w h`（归一化）；
  分割：`class x1 y1 x2 y2 ...`（归一化多边形顶点）。
- **VOC xml**：标准 Pascal VOC 目标检测格式。
- **COCO json**：批量时所有图片汇总到一个文件，含 bbox 与 polygon。
- **可视化**：`原图名_vis.jpg`，叠加检测框、类别标签（含置信度）与分割掩码。

## 打包为独立 exe

在 `goal` 环境中执行：

```bash
build.bat
```

生成的 `dist\AutoMask\AutoMask.exe` 可独立运行，无需 conda 环境。

**缩小体积（可选）**：若所用环境里的 torch 带 CUDA，exe 会很大（2~3GB）。
本项目的 `goal` 环境为纯 CPU 版 torch（2.x+cpu），直接打包体积约 1.3GB；
如需进一步缩小，可新建仅含 CPU 版 torch 的干净环境再打包：

```bash
conda create -n automask_cpu python=3.10 -y
conda activate automask_cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics PyQt5 opencv-python numpy Pillow PyYAML
# 然后将本仓库拷入该环境，执行 build.bat
```

## 备注

- 分割模型（YOLOv8-seg 等）会自动导出多边形标注；检测模型导出框标注。
- 设备下拉默认 `CPU`；选择 `自动` 会检测 CUDA 可用性，`CUDA:0` 强制用 GPU。
