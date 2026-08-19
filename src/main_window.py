"""PyQt5 main window for the AutoMask YOLO automatic annotation tool.

Core actions:
  1. 加载模型            (Load YOLO weights .pt)
  2. 选择图片 / 文件夹    (Select image or folder)
  3. 选择标注保存路径      (Select annotation save directory)

Plus: run inference (single / batch), preview image + boxes/masks overlay,
and export annotations (YOLO txt / VOC xml / COCO json / visualization).

UI philosophy: premium 3-column layout — left nav + settings rail, central
preview stage, right legend + log. Cohesive design system (gradient brand
bar, card shadows, hand-drawn line icons, custom scrollbars, dark/light theme).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF, QSize, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QDesktopServices, QIcon, QImage, QPainter, QPainterPath,
    QPen, QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGridLayout,
    QGraphicsDropShadowEffect, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from src.annotation_io import CocoWriter, color_palette, save_annotation
from src.yolo_model import ModelLoadError, YoloModel

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Accent used for hand-drawn nav icons (reads well on both themes).
ICON_ACCENT = "#5b8cff"


# ---------------------------------------------------------------------------- #
# Hand-drawn line icons (no external assets, no emoji)
# ---------------------------------------------------------------------------- #
def _make_icon(kind: str, color: str = ICON_ACCENT, size: int = 22) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    s = size
    pen = QPen(QColor(color))
    pen.setWidthF(1.9)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    if kind == "model":
        for i in range(3):
            y = 7 + i * 4.5
            p.drawLine(QLineF(4, y, s / 2, y - 3.2))
            p.drawLine(QLineF(s / 2, y - 3.2, s - 4, y))
            p.drawLine(QLineF(s - 4, y, s / 2, y + 3.2))
            p.drawLine(QLineF(s / 2, y + 3.2, 4, y))
    elif kind == "image":
        p.drawRoundedRect(QRectF(3, 4, s - 6, s - 8), 2.5, 2.5)
        p.drawLine(QLineF(6, s - 5, 10, s - 10))
        p.drawLine(QLineF(10, s - 10, 13, s - 7))
        p.drawLine(QLineF(13, s - 7, 16, s - 11))
        p.drawLine(QLineF(16, s - 11, s - 4, s - 5))
        p.drawEllipse(QRectF(s - 10, 6, 4, 4))
    elif kind == "folder":
        path = QPainterPath()
        path.moveTo(3, s - 3)
        path.lineTo(3, 6)
        path.lineTo(8, 6)
        path.lineTo(10.5, 4)
        path.lineTo(s - 3, 4)
        path.lineTo(s - 3, s - 3)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "export":
        path = QPainterPath()
        path.moveTo(3, s - 3)
        path.lineTo(3, 6)
        path.lineTo(8, 6)
        path.lineTo(10.5, 4)
        path.lineTo(s - 3, 4)
        path.lineTo(s - 3, s - 3)
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(QLineF(s / 2, s - 6, s / 2, 7))
        p.drawLine(QLineF(s / 2 - 3.2, 10, s / 2, 6.5))
        p.drawLine(QLineF(s / 2 + 3.2, 10, s / 2, 6.5))
    elif kind == "play":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        tri = QPainterPath()
        tri.moveTo(7, 5)
        tri.lineTo(7, s - 5)
        tri.lineTo(s - 5, s / 2)
        tri.closeSubpath()
        p.drawPath(tri)
    elif kind == "batch":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        t1 = QPainterPath()
        t1.moveTo(4, 5)
        t1.lineTo(4, s - 5)
        t1.lineTo(11, s / 2)
        t1.closeSubpath()
        p.drawPath(t1)
        t2 = QPainterPath()
        t2.moveTo(12, 5)
        t2.lineTo(12, s - 5)
        t2.lineTo(s - 2, s / 2)
        t2.closeSubpath()
        p.drawPath(t2)
    elif kind == "save":
        p.drawRoundedRect(QRectF(4, 4, s - 8, s - 8), 2, 2)
        p.drawRect(QRectF(s - 11, 4, 7, 5))
        p.drawLine(QLineF(8, 10, s - 8, 10))
        p.drawRect(QRectF(8, 13, s - 16, s - 17))
    elif kind == "check":
        path = QPainterPath()
        path.moveTo(4, s / 2 + 1)
        path.lineTo(s / 2 - 2, s - 5)
        path.lineTo(s - 3, 5)
        p.drawPath(path)
    elif kind == "sun":
        c = s / 2
        p.drawEllipse(QRectF(c - 3.6, c - 3.6, 7.2, 7.2))
        for k in range(8):
            a = math.radians(k * 45)
            x1 = c + math.cos(a) * 5.6
            y1 = c + math.sin(a) * 5.6
            x2 = c + math.cos(a) * 8.6
            y2 = c + math.sin(a) * 8.6
            p.drawLine(QLineF(x1, y1, x2, y2))
    elif kind == "moon":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(4, 3, s - 9, s - 9))
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        p.drawEllipse(QRectF(10, 1, s - 9, s - 9))
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    elif kind == "check":
        # checkmark in a rounded badge
        p.drawRoundedRect(QRectF(3, 3, s - 6, s - 6), 3, 3)
        path = QPainterPath()
        path.moveTo(6, s / 2)
        path.lineTo(s / 2 - 1, s - 6)
        path.lineTo(s - 5, 6)
        pen2 = QPen(QColor(color))
        pen2.setWidthF(2.4)
        pen2.setCapStyle(Qt.RoundCap)
        pen2.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen2)
        p.drawPath(path)
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------------------- #
# Design system (theme palettes)
# ---------------------------------------------------------------------------- #
def _build_qss(dark: bool) -> str:
    if dark:
        c = dict(
            bg="#0e1014", surface="#161a22", surface2="#1d222c",
            input="#11141a", border="#2a313d", border2="#333b48",
            text="#e8ecf4", dim="#8b93a7", accent="#5b8cff", accent2="#8a5cff",
            success="#3ad29f", danger="#f87272", shadow="rgba(0,0,0,0.45)",
        )
    else:
        c = dict(
            bg="#eaeef5", surface="#ffffff", surface2="#f4f7fb",
            input="#ffffff", border="#e2e8f0", border2="#cdd6e4",
            text="#1f2430", dim="#6b7280", accent="#3b6cf6", accent2="#7c5cff",
            success="#16a34a", danger="#dc2626", shadow="rgba(40,55,90,0.18)",
        )
    tpl = """
    QWidget {
        color: %(text)s;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        font-size: 10pt;
        background: %(bg)s;
    }
    QMainWindow { background: %(bg)s; }

    /* ---- scrollbars ---- */
    QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
    QScrollBar::handle:vertical {
        background: %(border2)s; border-radius: 5px; min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: %(accent)s; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal { background: transparent; height: 9px; }
    QScrollBar::handle:horizontal { background: %(border2)s; border-radius: 5px; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

    /* ---- sidebar nav buttons (compact 2-col grid) ---- */
    QPushButton#nav {
        background: %(surface2)s; color: %(text)s;
        border: 1px solid %(border)s; border-radius: 10px;
        padding: 8px 10px; text-align: left; font-size: 9.5pt;
        min-height: 34px;
    }
    QPushButton#nav:hover {
        border-color: %(accent)s; background: %(surface)s;
    }
    QPushButton#nav:pressed { border-color: %(accent)s; }

    /* ---- generic buttons ---- */
    QPushButton {
        background: %(surface2)s; color: %(text)s;
        border: 1px solid %(border)s; border-radius: 10px;
        padding: 9px 14px;
    }
    QPushButton:hover { border-color: %(accent)s; }
    QPushButton:pressed { border-color: %(accent)s; }
    QPushButton:disabled { color: %(dim)s; background: %(surface)s;
        border-color: %(border)s; }

    QPushButton#primary {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 %(accent)s, stop:1 %(accent2)s);
        color: #ffffff; border: none; font-weight: 600;
    }
    QPushButton#primary:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 %(accent)s, stop:1 %(accent2)s); }
    QPushButton#primary:disabled { color: rgba(255,255,255,0.55);
        background: %(border2)s; }

    QPushButton#ghost {
        background: %(surface)s; color: %(text)s;
        border: 1px solid %(border)s; border-radius: 10px; padding: 9px 14px;
    }
    QPushButton#ghost:hover { border-color: %(accent)s; }

    /* ---- inputs ---- */
    QLineEdit, QComboBox {
        background: %(input)s; border: 1px solid %(border)s;
        border-radius: 10px; padding: 9px 11px; color: %(text)s;
    }
    QLineEdit:focus, QComboBox:focus { border-color: %(accent)s; }
    QLineEdit::placeholder { color: %(dim)s; }
    QComboBox::drop-down { border: none; width: 24px; }
    QComboBox QAbstractItemView { background: %(input)s; color: %(text)s;
        selection-background-color: %(accent)s; outline: 0; border-radius: 8px; }

    /* ---- cards / group boxes ---- */
    QGroupBox {
        background: %(surface)s; border: 1px solid %(border)s;
        border-radius: 12px; margin-top: 13px; padding: 11px 11px 11px 11px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 11px; padding: 0 6px; color: %(accent)s;
        font-weight: 600; font-size: 10pt;
    }

    /* ---- lists ---- */
    QListWidget {
        background: %(input)s; border: 1px solid %(border)s;
        border-radius: 11px; padding: 6px; outline: 0;
    }
    QListWidget::item { padding: 7px 9px; border-radius: 7px; color: %(text)s; }
    QListWidget::item:selected { background: %(accent)s; color: #fff; }
    QListWidget::item:hover { background: %(surface2)s; }

    /* ---- checkboxes ---- */
    QCheckBox { spacing: 9px; padding: 4px; }
    QCheckBox::indicator {
        width: 17px; height: 17px; border-radius: 5px;
        border: 1px solid %(border2)s; background: %(input)s;
    }
    QCheckBox::indicator:checked {
        background: %(accent)s; border-color: %(accent)s;
    }

    /* ---- sliders ---- */
    QSlider::groove:horizontal { height: 6px; background: %(input)s;
        border-radius: 3px; border: 1px solid %(border)s; }
    QSlider::sub-page:horizontal { height: 6px;
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 %(accent2)s, stop:1 %(accent)s);
        border-radius: 3px; }
    QSlider::handle:horizontal {
        width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
        background: %(text)s; border: 2px solid %(accent)s;
    }
    QSlider::handle:horizontal:hover { background: #fff; }

    /* ---- graphics view (preview stage) ---- */
    QGraphicsView {
        background: %(surface2)s; border: 1px solid %(border)s;
        border-radius: 16px;
    }

    /* ---- log ---- */
    QPlainTextEdit {
        background: %(input)s; border: 1px solid %(border)s;
        border-radius: 11px; color: %(dim)s;
        font-family: 'Cascadia Code', 'Consolas', monospace; padding: 10px;
    }

    /* ---- progress ---- */
    QProgressBar {
        background: %(input)s; border: 1px solid %(border)s;
        border-radius: 9px; text-align: center; color: %(text)s;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 %(accent2)s, stop:1 %(accent)s);
        border-radius: 9px;
    }

    /* ---- status bar ---- */
    QStatusBar { background: %(surface)s; border-top: 1px solid %(border)s;
        color: %(dim)s; padding: 3px 8px; }
    QStatusBar::item { border: none; }
    """
    return tpl % c


# ---------------------------------------------------------------------------- #
# Workers
# ---------------------------------------------------------------------------- #
class LoadModelWorker(QThread):
    sig_loaded = pyqtSignal(list)
    sig_error = pyqtSignal(str)
    sig_finished = pyqtSignal()

    def __init__(self, mm: YoloModel, weights: str, device: str,
                 names_override: Optional[List[str]]):
        super().__init__()
        self.mm, self.weights = mm, weights
        self.device, self.names_override = device, names_override

    def run(self):
        try:
            self.mm.load(self.weights, self.device, self.names_override)
            self.sig_loaded.emit(self.mm.names)
        except Exception as e:
            self.sig_error.emit(str(e))
        self.sig_finished.emit()


class InferenceWorker(QThread):
    sig_progress = pyqtSignal(int, int, str)
    sig_image_done = pyqtSignal(dict)
    sig_saved = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_finished = pyqtSignal()

    def __init__(self, mm: YoloModel, paths: List[str], mode: str,
                 out_dir: str, opts: Dict):
        super().__init__()
        self.mm, self.paths = mm, paths
        self.mode, self.out_dir, self.opts = mode, out_dir, opts

    def run(self):
        try:
            total = len(self.paths)
            coco = None
            if self.mode == "batch" and self.opts.get("save_coco"):
                coco = CocoWriter(self.mm.names)
            for i, p in enumerate(self.paths):
                if self.isInterruptionRequested():
                    break
                stem = os.path.splitext(os.path.basename(p))[0]
                fname = os.path.basename(p)
                try:
                    img = cv2.imread(p)
                    if img is None:
                        self.sig_error.emit(f"无法读取图片: {p}")
                        continue
                    conf = float(self.opts.get("conf", 0.25))
                    iou = float(self.opts.get("iou", 0.7))
                    result = self.mm.predict(img, conf=conf, iou=iou)
                except Exception as e:
                    # One bad image must not abort the whole batch.
                    self.sig_error.emit(f"跳过 {stem}: {e}")
                    continue
                if self.mode == "single":
                    self.sig_image_done.emit({
                        "image_bgr": img, "result": result,
                        "stem": stem, "path": p,
                    })
                else:
                    # Batch writes the COCO summary once at the end; skip
                    # per-image .coco.json here to avoid duplicate noise.
                    img_opts = dict(self.opts)
                    img_opts["save_coco"] = False
                    try:
                        written = save_annotation(
                            img, result, self.out_dir, stem, img_opts,
                            image_name=fname)
                    except Exception as e:
                        self.sig_error.emit(f"保存失败 {stem}: {e}")
                        continue
                    if coco is not None:
                        h, w = result["shape"]
                        coco.add(result["boxes"], result["masks"],
                                 stem, h, w, file_name=fname)
                    self.sig_saved.emit(
                        f"已保存: {stem} -> "
                        f"{', '.join(os.path.basename(x) for x in written.values())}")
                self.sig_progress.emit(i + 1, total, stem)
            if coco is not None and not self.isInterruptionRequested():
                cp = os.path.join(self.out_dir, "annotations.coco.json")
                coco.write(cp)
                self.sig_saved.emit(f"COCO 汇总已保存: {cp}")
            self.sig_finished.emit()
        except Exception as e:
            self.sig_error.emit(f"推理出错: {e}")
            self.sig_finished.emit()


# ---------------------------------------------------------------------------- #
# Image view with zoom
# ---------------------------------------------------------------------------- #
class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def show_pixmap(self, pix: QPixmap):
        self.scene.clear()
        self.scene.addItem(QGraphicsPixmapItem(pix))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


# ---------------------------------------------------------------------------- #
# Shadow helper
# ---------------------------------------------------------------------------- #
def _shadow(widget, blur=22, offset=4, color="rgba(0,0,0,0.40)"):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(blur)
    sh.setOffset(0, offset)
    sh.setColor(QColor(color))
    widget.setGraphicsEffect(sh)


# ---------------------------------------------------------------------------- #
# Main window
# ---------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mm = YoloModel()
        self.images: List[str] = []
        self.results_cache: "OrderedDict[str, dict]" = OrderedDict()
        self.current: Optional[dict] = None
        self.model_load_worker = None
        self.infer_worker = None

        self.setWindowTitle("AutoMask · YOLO 自动标注")
        self.resize(1340, 860)
        self.setContentsMargins(0, 0, 0, 0)

        self._build_ui()
        self.setStyleSheet(_build_qss(True))
        self._set_running_state(False)
        self.log("就绪 · 先【加载模型】→ 选【图片】→【运行当前】或【批量运行】。")

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ============================ LEFT RAIL ============================
        rail = QWidget()
        rail.setFixedWidth(440)
        rail.setStyleSheet("background: #161a22;")
        rv = QVBoxLayout(rail)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(10)

        # brand
        brand = QWidget()
        brand.setFixedHeight(54)
        brand.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #5b8cff, stop:1 #8a5cff);"
            "border-radius: 12px;")
        bv = QHBoxLayout(brand)
        bv.setContentsMargins(14, 0, 14, 0)
        btitle = QLabel("AutoMask")
        btitle.setStyleSheet("color:#fff; font-size:15pt; font-weight:700;")
        bsub = QLabel("YOLO 自动标注")
        bsub.setStyleSheet("color:rgba(255,255,255,0.85); font-size:9pt;")
        bcol = QVBoxLayout()
        bcol.setSpacing(1)
        bcol.addWidget(btitle)
        bcol.addWidget(bsub)
        bv.addLayout(bcol)
        bv.addStretch(1)
        rv.addWidget(brand)

        # primary nav actions — compact 2-column grid (saves vertical space)
        nav_grid = QGridLayout()
        nav_grid.setSpacing(8)
        self.btn_nav_model = QPushButton(" 加载模型")
        self.btn_nav_model.setObjectName("nav")
        self.btn_nav_model.setIcon(_make_icon("model"))
        self.btn_nav_model.clicked.connect(self.pick_and_load_model)
        self.btn_nav_img = QPushButton(" 选择图片")
        self.btn_nav_img.setObjectName("nav")
        self.btn_nav_img.setIcon(_make_icon("image"))
        self.btn_nav_img.clicked.connect(self.select_image)
        self.btn_nav_folder = QPushButton(" 选文件夹")
        self.btn_nav_folder.setObjectName("nav")
        self.btn_nav_folder.setIcon(_make_icon("folder"))
        self.btn_nav_folder.clicked.connect(self.select_folder)
        self.btn_nav_out = QPushButton(" 保存路径")
        self.btn_nav_out.setObjectName("nav")
        self.btn_nav_out.setIcon(_make_icon("export"))
        self.btn_nav_out.clicked.connect(self.select_out)
        self.btn_nav_review = QPushButton(" 标注复核")
        self.btn_nav_review.setObjectName("nav")
        self.btn_nav_review.setIcon(_make_icon("check"))
        self.btn_nav_review.clicked.connect(self.open_review)
        nav_grid.addWidget(self.btn_nav_model, 0, 0)
        nav_grid.addWidget(self.btn_nav_img, 0, 1)
        nav_grid.addWidget(self.btn_nav_folder, 1, 0)
        nav_grid.addWidget(self.btn_nav_out, 1, 1)
        nav_grid.addWidget(self.btn_nav_review, 2, 0, 1, 2)
        nav_wrap = QWidget()
        nav_wrap.setLayout(nav_grid)
        rv.addWidget(nav_wrap)

        # scrollable settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        settings = QWidget()
        sv = QVBoxLayout(settings)
        sv.setContentsMargins(0, 2, 0, 2)
        sv.setSpacing(9)
        sv.addWidget(self._card_model())
        sv.addWidget(self._card_image())
        sv.addWidget(self._card_output())
        sv.addWidget(self._card_thresholds())
        sv.addStretch(1)
        scroll.setWidget(settings)
        rv.addWidget(scroll, stretch=1)

        root.addWidget(rail)

        # ============================ CENTER ============================
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(14, 14, 14, 14)
        cv.setSpacing(12)

        # top action bar
        top = QWidget()
        top.setFixedHeight(58)
        top.setStyleSheet(
            "background: #161a22; border: 1px solid #2a313d; border-radius: 14px;")
        _shadow(top, blur=18, offset=3)
        tv = QHBoxLayout(top)
        tv.setContentsMargins(14, 0, 14, 0)
        self.lbl_file = QLabel("未选择图片")
        self.lbl_file.setStyleSheet("font-size:11pt; color:#e8ecf4;")
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("font-size:10pt; color:#8b93a7;")
        tv.addWidget(self.lbl_file)
        tv.addWidget(self.lbl_count)
        tv.addStretch(1)
        self.act_run_one = QPushButton(" 运行当前")
        self.act_run_one.setObjectName("primary")
        self.act_run_one.setIcon(_make_icon("play", "#ffffff"))
        self.act_run_one.clicked.connect(self.run_current)
        self.act_run_all = QPushButton(" 批量运行")
        self.act_run_all.setIcon(_make_icon("batch"))
        self.act_run_all.clicked.connect(self.run_batch)
        self.act_save = QPushButton(" 保存当前")
        self.act_save.setIcon(_make_icon("save"))
        self.act_save.clicked.connect(self.save_current)
        for b in (self.act_run_one, self.act_run_all, self.act_save):
            tv.addWidget(b)
        cv.addWidget(top)

        # preview stage
        self.view = ImageView()
        cv.addWidget(self.view, stretch=1)

        root.addWidget(center, stretch=1)

        # ============================ RIGHT ============================
        right = QWidget()
        right.setFixedWidth(268)
        rw = QVBoxLayout(right)
        rw.setContentsMargins(14, 14, 14, 14)
        rw.setSpacing(12)

        legend_card = QGroupBox("类别图例")
        _shadow(legend_card, blur=18, offset=3)
        lcv = QVBoxLayout(legend_card)
        self.legend = QListWidget()
        lcv.addWidget(self.legend)
        rw.addWidget(legend_card, stretch=2)

        log_card = QGroupBox("运行日志")
        _shadow(log_card, blur=18, offset=3)
        gcv = QVBoxLayout(log_card)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        gcv.addWidget(self.log_box)
        rw.addWidget(log_card, stretch=3)

        self.btn_open_out = QPushButton("打开输出目录")
        self.btn_open_out.setObjectName("ghost")
        self.btn_open_out.setIcon(_make_icon("export"))
        self.btn_open_out.clicked.connect(self.open_out_dir)
        rw.addWidget(self.btn_open_out)

        root.addWidget(right)

        # status bar + progress
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setRange(0, 100)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("未加载模型")

    # ------------------------------------------------------------------ #
    # settings cards
    # ------------------------------------------------------------------ #
    def _card_model(self) -> QGroupBox:
        g = QGroupBox("1 · 模型")
        fm = QFormLayout(g)
        fm.setVerticalSpacing(6)
        fm.setHorizontalSpacing(8)
        fm.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.le_weights = QLineEdit()
        self.le_weights.setPlaceholderText("选择 YOLO 权重 .pt")
        btn_weights = QPushButton("浏览")
        btn_weights.clicked.connect(self.select_weights)
        row_w = QHBoxLayout()
        row_w.addWidget(self.le_weights)
        row_w.addWidget(btn_weights)
        fm.addRow("权重", row_w)
        self.le_names = QLineEdit()
        self.le_names.setPlaceholderText("留空 = 自动读取模型自带类别名")
        fm.addRow("类别名", self.le_names)
        hint = QLabel("留空即可：类别名会从 .pt 权重自动读取；"
                      "仅在模型未内嵌名称时才需手动填写覆盖。")
        hint.setStyleSheet("color:#8b93a7; font-size:9pt;")
        hint.setWordWrap(True)
        fm.addRow(hint)
        self.cb_device = QComboBox()
        self.cb_device.addItems(["CPU", "自动", "CUDA:0"])
        self.cb_device.setCurrentText("CPU")
        fm.addRow("设备", self.cb_device)
        self.btn_load = QPushButton("加载模型")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self.load_model)
        fm.addRow(self.btn_load)
        self.lbl_model_status = QLabel("未加载")
        self.lbl_model_status.setStyleSheet("color:#8b93a7;")
        fm.addRow(self.lbl_model_status)
        return g

    def _card_image(self) -> QGroupBox:
        g = QGroupBox("2 · 图片")
        iv = QVBoxLayout(g)
        iv.setSpacing(6)
        self.img_list = QListWidget()
        self.img_list.currentRowChanged.connect(self.on_image_selected)
        self.img_list.setMaximumHeight(118)
        iv.addWidget(self.img_list)
        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton("← 上一张")
        self.btn_next = QPushButton("下一张 →")
        self.btn_prev.clicked.connect(lambda: self.step_image(-1))
        self.btn_next.clicked.connect(lambda: self.step_image(1))
        row_nav.addWidget(self.btn_prev)
        row_nav.addWidget(self.btn_next)
        iv.addLayout(row_nav)
        return g

    def _card_output(self) -> QGroupBox:
        g = QGroupBox("3 · 标注输出")
        ov = QVBoxLayout(g)
        ov.setSpacing(7)
        self.le_out = QLineEdit()
        self.le_out.setPlaceholderText("标注文件保存目录")
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(self.select_out)
        row_out = QHBoxLayout()
        row_out.addWidget(self.le_out)
        row_out.addWidget(btn_out)
        ov.addLayout(row_out)

        self.cb_yolo = QCheckBox("YOLO txt（默认）")
        self.cb_yolo.setChecked(True)
        self.cb_voc = QCheckBox("Pascal VOC xml")
        self.cb_vis = QCheckBox("可视化图片")
        self.cb_vis.setChecked(True)
        self.cb_classes = QCheckBox("classes.txt")
        self.cb_classes.setChecked(True)
        self.cb_yaml = QCheckBox("dataset.yaml")
        self.cb_coco = QCheckBox("COCO json（汇总）")
        row_fmt = QHBoxLayout()
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col1.addWidget(self.cb_yolo)
        col1.addWidget(self.cb_vis)
        col1.addWidget(self.cb_yaml)
        col2.addWidget(self.cb_voc)
        col2.addWidget(self.cb_classes)
        col2.addWidget(self.cb_coco)
        row_fmt.addLayout(col1)
        row_fmt.addLayout(col2)
        ov.addLayout(row_fmt)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(40)
        self.slider_alpha.valueChanged.connect(self.on_alpha_changed)
        ov.addWidget(QLabel("可视化透明度"))
        ov.addWidget(self.slider_alpha)
        return g

    def _card_thresholds(self) -> QGroupBox:
        g = QGroupBox("推理阈值")
        thv = QVBoxLayout(g)
        thv.setSpacing(6)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(25)
        self.lbl_conf = QLabel("置信度: 0.25")
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"置信度: {v/100:.2f}"))
        thv.addWidget(self.lbl_conf)
        thv.addWidget(self.slider_conf)
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(10, 95)
        self.slider_iou.setValue(70)
        self.lbl_iou = QLabel("IOU: 0.70")
        self.slider_iou.valueChanged.connect(
            lambda v: self.lbl_iou.setText(f"IOU: {v/100:.2f}"))
        thv.addWidget(self.lbl_iou)
        thv.addWidget(self.slider_iou)
        return g

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def _set_running_state(self, running: bool):
        for w in (self.btn_load, self.act_run_one, self.act_run_all,
                  self.act_save, self.btn_nav_model):
            w.setEnabled(not running)
        if running:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def closeEvent(self, event):
        # Wait for any running worker so the QThread is not destroyed
        # mid-run (which crashes Qt). Disconnect first so queued signals
        # don't fire into a half-destroyed window.
        for worker in (self.model_load_worker, self.infer_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                try:
                    worker.disconnect()
                except Exception:
                    pass
                worker.wait()
        event.accept()

    def resolve_device(self, choice: str) -> str:
        if choice == "CPU":
            return "cpu"
        if choice == "自动":
            try:
                import torch
                return "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return "0"

    def build_opts(self) -> Dict:
        return {
            "save_yolo": self.cb_yolo.isChecked(),
            "save_voc": self.cb_voc.isChecked(),
            "save_vis": self.cb_vis.isChecked(),
            "save_classes": self.cb_classes.isChecked(),
            "save_yaml": self.cb_yaml.isChecked(),
            "save_coco": self.cb_coco.isChecked(),
            "vis_alpha": self.slider_alpha.value() / 100.0,
            "conf": self.slider_conf.value() / 100.0,
            "iou": self.slider_iou.value() / 100.0,
        }

    # ------------------------------------------------------------------ #
    # model
    # ------------------------------------------------------------------ #
    def select_weights(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", "", "PyTorch Weights (*.pt)")
        if p:
            self.le_weights.setText(p)

    def pick_and_load_model(self):
        """左栏【加载模型】：选完权重文件后立即加载，
        避免选了文件却忘了点加载，导致运行时提示“请先加载模型”。"""
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", "", "PyTorch Weights (*.pt)")
        if p:
            self.le_weights.setText(p)
            self.load_model()

    def load_model(self):
        weights = self.le_weights.text().strip()
        if not weights:
            QMessageBox.warning(self, "提示", "请先选择权重文件 (.pt)。")
            return
        device = self.resolve_device(self.cb_device.currentText())
        names = None
        raw = self.le_names.text().strip()
        if raw:
            names = [x.strip() for x in raw.split(",") if x.strip()]
        self.log(f"正在加载模型: {weights}  device={device}")
        self._set_running_state(True)
        self.model_load_worker = LoadModelWorker(
            self.mm, weights, device, names)
        self.model_load_worker.sig_loaded.connect(self.on_model_loaded)
        self.model_load_worker.sig_error.connect(self.on_model_error)
        self.model_load_worker.sig_finished.connect(
            lambda: self._set_running_state(False))
        self.model_load_worker.start()

    def on_model_loaded(self, names):
        self.lbl_model_status.setText(
            f"已加载 ({self.mm.task or '?'})  {len(names)} 类")
        self.lbl_model_status.setStyleSheet("color:#3ad29f;")
        self.statusBar().showMessage(
            f"模型: {os.path.basename(self.mm.weights_path or '')} | "
            f"任务: {self.mm.task} | 设备: {self.mm.device}")
        self.populate_legend(names)
        self.log(f"模型加载成功，类别: {', '.join(names)}")

    def on_model_error(self, msg):
        self.lbl_model_status.setText("加载失败")
        self.lbl_model_status.setStyleSheet("color:#f87272;")
        self.log(f"[错误] {msg}")
        QMessageBox.critical(self, "模型加载失败", msg)

    def populate_legend(self, names):
        self.legend.clear()
        palette = color_palette(len(names) or 1, bgr=False)
        for i, n in enumerate(names):
            item = QListWidgetItem(f"{i}: {n}")
            r, g, b = palette[i]
            pm = QPixmap(16, 16)
            pm.fill(QColor(r, g, b))
            item.setIcon(QIcon(pm))
            self.legend.addItem(item)

    # ------------------------------------------------------------------ #
    # images
    # ------------------------------------------------------------------ #
    def select_image(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if files:
            for f in files:
                if f not in self.images:
                    self.images.append(f)
                    self.img_list.addItem(os.path.basename(f))
            self.img_list.setCurrentRow(len(self.images) - 1)

    def select_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        added = 0
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMAGE_EXTS):
                fp = os.path.join(d, fn)
                if fp not in self.images:
                    self.images.append(fp)
                    self.img_list.addItem(fn)
                    added += 1
        self.log(f"从文件夹添加 {added} 张图片。")
        if self.images:
            self.img_list.setCurrentRow(0)

    def on_image_selected(self, row):
        if row < 0 or row >= len(self.images):
            return
        path = self.images[row]
        self.lbl_file.setText(os.path.basename(path))
        cached = self.results_cache.get(path)
        if cached:
            self.show_result(cached, from_cache=True)
        else:
            img = cv2.imread(path)
            if img is None:
                self.log(f"[警告] 无法读取: {path}")
                return
            self.current = None
            self.lbl_count.setText("")
            self.display_bgr(img)

    def step_image(self, delta):
        if not self.images:
            return
        row = self.img_list.currentRow()
        row = (row + delta) % len(self.images)
        self.img_list.setCurrentRow(row)

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def run_current(self):
        if not self.mm.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        row = self.img_list.currentRow()
        if row < 0 or row >= len(self.images):
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        self._set_running_state(True)
        self.infer_worker = InferenceWorker(
            self.mm, [self.images[row]], "single",
            self.le_out.text().strip(), self.build_opts())
        self._wire_infer(self.infer_worker)
        self.infer_worker.start()

    def run_batch(self):
        if not self.mm.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        if not self.images:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        out = self.le_out.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请先选择标注保存路径。")
            return
        self._set_running_state(True)
        self.infer_worker = InferenceWorker(
            self.mm, list(self.images), "batch", out, self.build_opts())
        self._wire_infer(self.infer_worker)
        self.infer_worker.start()

    def _wire_infer(self, worker):
        worker.sig_image_done.connect(
            lambda r: self.show_result(r, from_cache=False))
        worker.sig_saved.connect(self.log)
        worker.sig_error.connect(lambda m: self.log(f"[错误] {m}"))
        worker.sig_progress.connect(self._on_infer_progress)
        worker.sig_finished.connect(lambda: self._set_running_state(False))

    def _on_infer_progress(self, v, t, name):
        # Switch from indeterminate spinner to a real percentage once the
        # first progress tick arrives.
        self.progress.setRange(0, t)
        self.progress.setValue(v)
        self.log(f"进度 {v}/{t}  {name}")

    def show_result(self, res: dict, from_cache: bool):
        self.current = res
        if not from_cache:
            self.results_cache[res["path"]] = res
            # Bound memory: keep only the most recent 32 single-run results.
            while len(self.results_cache) > 32:
                self.results_cache.popitem(last=False)
        self.display_bgr(self.make_vis(res))
        nb = len(res["result"]["boxes"])
        nm = len(res["result"]["masks"])
        self.lbl_count.setText(f"检测到  框 {nb}  ·  掩码 {nm}")
        self.log(f"推理完成: {res['stem']}  框={nb}  掩码={nm}"
                 f"  任务={res['result']['task']}")
        if nb == 0 and nm == 0:
            self.log("[提示] 当前图片未检测到目标。可尝试调低置信度"
                     "（左侧阈值卡片），或确认图片内容与模型类别匹配。")

    def make_vis(self, res: dict):
        alpha = self.slider_alpha.value() / 100.0
        from src.annotation_io import draw_results
        return draw_results(
            res["image_bgr"], res["result"]["boxes"],
            res["result"]["masks"], res["result"]["names"], alpha)

    def on_alpha_changed(self, _):
        if self.current:
            self.display_bgr(self.make_vis(self.current))

    # ------------------------------------------------------------------ #
    # save / output
    # ------------------------------------------------------------------ #
    def select_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择标注保存目录")
        if d:
            self.le_out.setText(d)

    def open_review(self):
        # Lazy import avoids a circular import (review_window imports
        # helpers from this module; this module only needs ReviewWindow
        # when the user actually opens the review window).
        from src.review_window import ReviewWindow
        self._review_win = ReviewWindow(self)
        self._review_win.show()

    def save_current(self):
        if not self.current:
            QMessageBox.warning(self, "提示", "请先运行当前图片生成结果。")
            return
        out = self.le_out.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请先选择标注保存路径。")
            return
        fname = os.path.basename(self.current.get("path") or "")
        if not fname:
            fname = self.current["stem"] + ".jpg"
        written = save_annotation(
            self.current["image_bgr"], self.current["result"],
            out, self.current["stem"], self.build_opts(),
            image_name=fname)
        self.log("已保存当前结果: " +
                 ", ".join(os.path.basename(x) for x in written.values()))

    def open_out_dir(self):
        out = self.le_out.text().strip()
        if out and os.path.isdir(out):
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            QMessageBox.warning(self, "提示", "请先选择有效的输出目录。")

    # ------------------------------------------------------------------ #
    # display
    # ------------------------------------------------------------------ #
    def display_bgr(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self.view.show_pixmap(QPixmap.fromImage(qimg))


# backward-compatible alias
MainWindow = MainWindow
