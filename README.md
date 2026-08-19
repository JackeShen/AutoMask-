# AutoMask — YOLO 自动标注系统
<img width="1913" height="982" alt="8105e5e2c19b8c5e2f7971ca4b5e6e98" src="https://github.com/user-attachments/assets/55c715f0-6700-4c51-9fd3-b89cf2b9b536" />
<img width="1916" height="985" alt="eeea38ccdfe6b7e86ecc65f3f883ac80" src="https://github.com/user-attachments/assets/606386f1-dd03-48f0-a5af-1382cec7ea21" />


基于 **PyQt5 + ultralytics (YOLO)** 的图形化自动标注工具：加载训练好的 YOLO 权重（`.pt`），对单张图片或整个文件夹做推理，自动生成检测/分割标注，并支持**人工复核修正**。

> 在 `goal` conda 环境中开发与运行，默认 **CPU 推理**（纯 CPU 版 torch，无需 GPU）。

## 功能特性

- **加载模型**：任意 YOLO 权重 `.pt`（检测 / 分割 / 分类均支持），类别名自动读取，也可手动覆盖
- **自动标注**：单张运行 / 批量运行，CPU 推理
- **多格式导出**（可多选）：
  - YOLO txt —— 检测：`class cx cy w h`；分割：`class x1 y1 x2 y2 …`（归一化多边形）
  - Pascal VOC xml
  - COCO json —— 批量时汇总为 `annotations.coco.json`（含 bbox + 多边形）
  - 可视化图片 —— 叠加框 / 标签 / 置信度 / 分割掩码（透明度可调）
  - `classes.txt` / `dataset.yaml`
- **推理参数可调**：置信度阈值、IOU 阈值
- **标注复核**：打开复核窗口人工修正 —— 移动/缩放检测框、增删目标、拖拽多边形顶点、增删顶点、框选批量改类，保存自动备份 `.bak`
- **界面**：深色主题、卡片式三栏布局、预览滚轮缩放、手绘图标（无外部资源）

## 环境要求

- Python 3.10（推荐 conda 环境）
- 依赖见 `requirements_goal.txt`：`ultralytics`、`PyQt5`、`opencv-python`、`numpy`、`PyYAML`

## 安装与启动

方式一（Windows，双击启动，免激活环境）：

```
双击 run.bat
```

方式二（命令行）：

```bash
conda activate goal
cd <本目录>
python main.py
```

## 使用流程

1. 点【加载模型】→ 选 YOLO 权重 `.pt`（选完自动加载，状态栏变为绿色"已加载"）
2. 点【选图片】或【选文件夹】导入待标注图片
3. 点【保存路径】选标注输出目录
4. 按需勾选导出格式、调节置信度 / IOU
5. 【运行当前】或【批量运行】；预览区滚轮缩放
6. 需要人工修正时，左栏【标注复核】打开复核窗口

## 输出格式说明

- **YOLO txt**：每行一个目标。检测：`class cx cy w h`（归一化中心点+宽高）；分割：`class x1 y1 x2 y2 ...`（归一化多边形顶点，只写多边形不写框）
- **VOC xml**：标准 Pascal VOC 目标检测格式（`bndbox`）
- **COCO json**：批量时全部图片汇总一个文件；含 `bbox` 与 `segmentation` 多边形
- **可视化**：`原图名_vis.jpg`，叠加检测框、类别标签（含置信度）与分割掩码

## 标注复核（review_window）

- 加载图片文件夹 + labels 文件夹（YOLO txt）+ `classes.txt`
- 模式自动识别：检测（框）/ 分割（多边形）
- 编辑：拖动移动、四角缩框、右键删多边形顶点、双击边插入顶点、Delete 删除目标
- 改类：选中目标（可框选多个）后点击右侧类别图例即批量改类
- 保存：覆盖原 txt（首次自动备份 `.bak`），支持"保存并下一张"

## 打包为独立程序

### Windows

```bash
conda activate goal
build.bat
```

产物：`dist\AutoMask\AutoMask.exe`（普通文件夹版，约 1.3GB，含 torch 依赖），目标机器无需安装 conda/Python。

### macOS

代码可直接运行（`pip install ultralytics pyqt5 opencv-python` 后 `python main.py`）。
打包需**在 macOS 上**用 PyInstaller 打 `.app`（不支持跨平台交叉编译）：

```bash
pip install pyinstaller
pyinstaller --windowed --name AutoMask --collect-all ultralytics --collect-all torch main.py
```

## 项目结构

```
automask/
├── main.py                  # 入口
├── build.bat               # Windows 打包脚本（PyInstaller）
├── run.bat                 # 一键启动脚本（免 activate）
├── requirements_goal.txt   # 依赖记录
├── src/
│   ├── yolo_model.py       # YOLO 模型加载与推理封装（与 UI 解耦）
│   ├── annotation_io.py    # 标注保存（txt/xml/coco/可视化）
│   ├── main_window.py      # PyQt5 主界面（自动标注）
│   └── review_window.py    # 标注复核窗口（人工修正）
└── README.md
```

## 技术栈

PyQt5 · ultralytics (YOLOv8/11) · OpenCV · NumPy · PyInstaller

## 备注

- 分割模型（YOLO-seg）自动导出多边形标注；检测模型导出框标注
- 设备下拉默认 `CPU`；`自动` 会检测 CUDA，`CUDA:0` 强制 GPU（需显卡环境）
- 标注 txt 坐标均为归一化（0~1），与图片尺寸无关
