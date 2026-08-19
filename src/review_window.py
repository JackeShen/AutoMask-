"""Annotation review window for the AutoMask tool.

Loads an images folder + a labels folder (YOLO txt) + an optional
classes.txt, displays each image with its annotations, and lets the user
manually review/edit them: move/resize boxes, add/delete boxes, edit
segmentation polygon vertices (move/insert/delete), add/delete polygons.
Mode (detect vs segment) is auto-detected per label file. Saving overwrites
the original label file (with a one-time ``.bak`` backup of the very first
version).

Design reuses the main window's design system (``_build_qss`` / ``_make_icon``
/ ``_shadow``) for visual consistency.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import List, Optional

import cv2
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QFileDialog, QGraphicsItem, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QStyle, QStyleOptionGraphicsItem, QVBoxLayout, QWidget,
)

from src.annotation_io import color_palette, read_yolo_txt, save_yolo_txt
from src.main_window import _build_qss, _make_icon, _shadow

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
HANDLE = 9  # px size of edit handles


# ---------------------------------------------------------------------------- #
# geometry helper
# ---------------------------------------------------------------------------- #
def _dist_to_seg(p: QPointF, a: QPointF, b: QPointF) -> float:
    apx, apy = p.x() - a.x(), p.y() - a.y()
    abx, aby = b.x() - a.x(), b.y() - a.y()
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx, cy = a.x() + t * abx, a.y() + t * aby
    return math.hypot(p.x() - cx, p.y() - cy)


# ---------------------------------------------------------------------------- #
# Editable box (detection)
# ---------------------------------------------------------------------------- #
class EditBoxItem(QGraphicsRectItem):
    def __init__(self, cls: int, x1, y1, x2, y2):
        super().__init__(QRectF(min(x1, x2), min(y1, y2),
                                abs(x2 - x1), abs(y2 - y1)))
        self.cls = int(cls)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self._action = None       # 'move' | 'tl'/'tr'/'bl'/'br' | None
        self._start_rect = None
        self._start_pos = None

    def _corner(self, p: QPointF):
        r = self.rect()
        for name, pt in (("tl", r.topLeft()), ("tr", r.topRight()),
                         ("bl", r.bottomLeft()), ("br", r.bottomRight())):
            if (p - pt).manhattanLength() <= HANDLE + 2:
                return name
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            super().mousePressEvent(e)
            return
        c = self._corner(e.pos())
        self._action = c if c else "move"
        self._start_rect = QRectF(self.rect())
        self._start_pos = QPointF(e.scenePos())
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._action:
            super().mouseMoveEvent(e)
            return
        d = e.scenePos() - self._start_pos
        if self._action == "move":
            self.setRect(self._start_rect.translated(d))
        else:
            r = QRectF(self._start_rect)
            if self._action == "tl":
                r.setTopLeft(r.topLeft() + d)
            elif self._action == "tr":
                r.setTopRight(r.topRight() + d)
            elif self._action == "bl":
                r.setBottomLeft(r.bottomLeft() + d)
            elif self._action == "br":
                r.setBottomRight(r.bottomRight() + d)
            self.setRect(r.normalized())
        self._notify()
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._action:
            self._action = None
            self._start_rect = None
            self._start_pos = None
            e.accept()
        else:
            super().mouseReleaseEvent(e)

    def paint(self, painter, option, widget=None):
        opt = QStyleOptionGraphicsItem(option)
        opt.state &= ~QStyle.State_Selected
        super().paint(painter, opt, widget)
        if self.isSelected():
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(QBrush(QColor("#5b8cff")))
            r = self.rect()
            for pt in (r.topLeft(), r.topRight(),
                       r.bottomLeft(), r.bottomRight()):
                painter.drawRect(QRectF(pt.x() - HANDLE / 2,
                                        pt.y() - HANDLE / 2, HANDLE, HANDLE))

    def apply_color(self, rgb):
        r, g, b = rgb
        self.setPen(QPen(QColor(r, g, b), 2))
        self.setBrush(QBrush(Qt.NoBrush))

    def _notify(self):
        cb = getattr(self.scene(), "on_changed", None)
        if cb:
            cb()


# ---------------------------------------------------------------------------- #
# Editable polygon (segmentation)
# ---------------------------------------------------------------------------- #
class EditPolyItem(QGraphicsPolygonItem):
    def __init__(self, cls: int, pts):
        super().__init__(QPolygonF([QPointF(*p) for p in pts]))
        self.cls = int(cls)
        self.pts = [QPointF(*p) for p in pts]
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._action = None     # 'move' | vertex index | None
        self._start_pos = None
        self._start_pts = None

    def _vertex_at(self, p: QPointF):
        for i, pt in enumerate(self.pts):
            if (p - pt).manhattanLength() <= HANDLE + 2:
                return i
        return -1

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            i = self._vertex_at(e.pos())
            if i >= 0:
                self._action = i
            else:
                self._action = "move"
                self._start_pos = QPointF(e.scenePos())
                self._start_pts = [QPointF(p) for p in self.pts]
            e.accept()
            return
        if e.button() == Qt.RightButton:
            i = self._vertex_at(e.pos())
            if i >= 0 and len(self.pts) > 3:
                del self.pts[i]
                self._rebuild()
                self._notify()
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._action is None:
            super().mouseMoveEvent(e)
            return
        if self._action == "move":
            d = e.scenePos() - self._start_pos
            self.pts = [self._start_pts[k] + d for k in range(len(self._start_pts))]
        else:
            self.pts[self._action] = QPointF(e.pos())
        self._rebuild()
        self._notify()
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._action is not None:
            self._action = None
            self._start_pos = None
            self._start_pts = None
            e.accept()
        else:
            super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        p = QPointF(e.pos())
        ei = self._nearest_edge(p)
        if ei is not None:
            self.pts.insert(ei + 1, p)
            self._rebuild()
            self._notify()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def _rebuild(self):
        self.setPolygon(QPolygonF(self.pts))

    def _nearest_edge(self, p):
        best_i, best_d = -1, 1e18
        n = len(self.pts)
        for i in range(n):
            d = _dist_to_seg(p, self.pts[i], self.pts[(i + 1) % n])
            if d < best_d:
                best_d, best_i = d, i
        return best_i if best_d <= HANDLE * 1.6 else None

    def paint(self, painter, option, widget=None):
        opt = QStyleOptionGraphicsItem(option)
        opt.state &= ~QStyle.State_Selected
        super().paint(painter, opt, widget)
        if self.isSelected():
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(QBrush(QColor("#5b8cff")))
            for pt in self.pts:
                painter.drawRect(QRectF(pt.x() - HANDLE / 2,
                                        pt.y() - HANDLE / 2, HANDLE, HANDLE))

    def apply_color(self, rgb):
        r, g, b = rgb
        self.setPen(QPen(QColor(r, g, b), 2))
        self.setBrush(QBrush(QColor(r, g, b, 70)))

    def _notify(self):
        cb = getattr(self.scene(), "on_changed", None)
        if cb:
            cb()


# ---------------------------------------------------------------------------- #
# Review view (draw / pan / zoom)
# ---------------------------------------------------------------------------- #
class ReviewView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self._mode = "select"          # select | draw_box | draw_poly
        self._draw_start = None
        self._temp_rect = None
        self._poly_pts: List[QPointF] = []
        self._poly_preview = None
        self._pixmap_item = None
        self._panning = False
        self._last_pan = None
        # callbacks set by the window
        self.on_changed = None
        self.on_box_drawn = None
        self.on_finalize_poly = None
        self.on_delete_selected = None

    # ----- mode / scene -----
    def set_mode(self, mode: str):
        self._mode = mode
        self.scene.clearSelection()
        self._cancel_draw()
        if mode == "select":
            # Rubber-band drag: dragging on empty space box-selects every
            # item intersecting the rectangle (region select -> batch
            # reclassify via the class dropdown / apply button).
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)

    def _cancel_draw(self):
        if self._temp_rect is not None:
            self.scene.removeItem(self._temp_rect)
            self._temp_rect = None
        if self._poly_preview is not None:
            self.scene.removeItem(self._poly_preview)
            self._poly_preview = None
        self._poly_pts = []
        self._draw_start = None

    def clear_scene(self):
        self.scene.clear()
        self._pixmap_item = None
        self._cancel_draw()

    def set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self.scene.addItem(self._pixmap_item)
        self.scene.setSceneRect(0, 0, w, h)

    def fit(self):
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # ----- mouse -----
    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._mode == "draw_box" and e.button() == Qt.LeftButton:
            sp = self.mapToScene(e.pos())
            self._draw_start = sp
            self._temp_rect = QGraphicsRectItem(QRectF(sp, sp))
            self._temp_rect.setPen(QPen(QColor("#5b8cff"), 2, Qt.DashLine))
            self.scene.addItem(self._temp_rect)
            return
        if self._mode == "draw_poly" and e.button() == Qt.LeftButton:
            self._poly_pts.append(self.mapToScene(e.pos()))
            self._update_poly_preview()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._panning:
            d = e.pos() - self._last_pan
            self._last_pan = e.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            return
        if self._mode == "draw_box" and self._temp_rect is not None:
            self._temp_rect.setRect(QRectF(
                self._draw_start, self.mapToScene(e.pos())).normalized())
            return
        if self._mode == "draw_poly":
            self._update_poly_preview(cursor=self.mapToScene(e.pos()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor
                           if self._mode != "select" else Qt.ArrowCursor)
            return
        if (self._mode == "draw_box" and self._temp_rect is not None
                and e.button() == Qt.LeftButton):
            r = self._temp_rect.rect()
            self.scene.removeItem(self._temp_rect)
            self._temp_rect = None
            self._draw_start = None
            self.set_mode("select")
            if r.width() >= 3 and r.height() >= 3 and self.on_box_drawn:
                self.on_box_drawn(r.left(), r.top(), r.right(), r.bottom())
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._mode == "draw_poly":
            # the double-click's second press appended a near-duplicate;
            # drop it before finalizing.
            if len(self._poly_pts) >= 2:
                a, b = self._poly_pts[-1], self._poly_pts[-2]
                if (a - b).manhattanLength() <= HANDLE + 2:
                    self._poly_pts.pop()
            if len(self._poly_pts) >= 3:
                self._finalize_poly()
            else:
                self.set_mode("select")
            return
        super().mouseDoubleClickEvent(e)

    def _update_poly_preview(self, cursor=None):
        if self._poly_preview is not None:
            self.scene.removeItem(self._poly_preview)
            self._poly_preview = None
        if not self._poly_pts:
            return
        path = QPainterPath()
        path.moveTo(self._poly_pts[0])
        for p in self._poly_pts[1:]:
            path.lineTo(p)
        if cursor is not None:
            path.lineTo(cursor)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor("#5b8cff"), 2, Qt.DashLine))
        self.scene.addItem(item)
        self._poly_preview = item

    def _finalize_poly(self):
        if len(self._poly_pts) < 3:
            self.set_mode("select")
            return
        pts = self._poly_pts
        self._poly_pts = []
        if self._poly_preview is not None:
            self.scene.removeItem(self._poly_preview)
            self._poly_preview = None
        self.set_mode("select")
        if self.on_finalize_poly:
            self.on_finalize_poly([(p.x(), p.y()) for p in pts])

    # ----- keys -----
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            if self._mode != "select":
                self.set_mode("select")
                return
        elif e.key() == Qt.Key_Return and self._mode == "draw_poly":
            self._finalize_poly()
            return
        elif e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            sel = self.scene.selectedItems()
            if sel and self.on_delete_selected:
                self.on_delete_selected(sel)
                return
        super().keyPressEvent(e)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


# ---------------------------------------------------------------------------- #
# Review window
# ---------------------------------------------------------------------------- #
class ReviewWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoMask · 标注复核")
        self.resize(1300, 820)
        self.images: List[str] = []
        self.labels_map: dict = {}
        self.labels_dir = ""
        self.classes: List[str] = []
        self.palette: List[tuple] = []
        self.items: list = []
        self.current_image_path: Optional[str] = None
        self.img_w = 0
        self.img_h = 0
        self.current_mode = "detect"
        self.dirty = False
        self.current_class = 0
        self._build_ui()
        self.setStyleSheet(_build_qss(True))
        self.log("就绪 · 依次选择【图片文件夹】【labels 文件夹】【classes.txt】。")

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ===================== LEFT RAIL =====================
        rail = QWidget()
        rail.setFixedWidth(420)
        rail.setStyleSheet("background:#161a22;")
        rv = QVBoxLayout(rail)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(9)

        title = QLabel("标注复核")
        title.setStyleSheet("color:#fff;font-size:15pt;font-weight:700;"
                            "padding:4px 4px;")
        rv.addWidget(title)

        self.btn_imgs = QPushButton(" 选择图片文件夹")
        self.btn_imgs.setObjectName("nav")
        self.btn_imgs.setIcon(_make_icon("folder"))
        self.btn_imgs.clicked.connect(self.select_images)
        self.btn_labels = QPushButton(" 选择 labels 文件夹")
        self.btn_labels.setObjectName("nav")
        self.btn_labels.setIcon(_make_icon("folder"))
        self.btn_labels.clicked.connect(self.select_labels)
        self.btn_classes = QPushButton(" 选择 classes.txt")
        self.btn_classes.setObjectName("nav")
        self.btn_classes.setIcon(_make_icon("image"))
        self.btn_classes.clicked.connect(self.select_classes)
        rv.addWidget(self.btn_imgs)
        rv.addWidget(self.btn_labels)
        rv.addWidget(self.btn_classes)

        self.img_list = QListWidget()
        self.img_list.setMaximumHeight(140)
        self.img_list.currentRowChanged.connect(self.on_image_selected)
        rv.addWidget(self.img_list)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← 上一张")
        self.btn_next = QPushButton("下一张 ->")
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        rv.addLayout(nav)

        hint = QLabel("改类:选中目标(可框选多个)后,点右侧【类别图例】对应行即改为该类;\n未选中时点图例=设为新增用的当前类别。")
        hint.setStyleSheet("color:#8b93a7; font-size:9pt;")
        hint.setWordWrap(True)
        rv.addWidget(hint)

        self.lbl_mode = QLabel("模式: -")
        self.lbl_mode.setStyleSheet("color:#8b93a7;")
        rv.addWidget(self.lbl_mode)
        self.btn_toggle_mode = QPushButton("切换为分割模式")
        self.btn_toggle_mode.clicked.connect(self.toggle_mode)
        rv.addWidget(self.btn_toggle_mode)

        self.btn_add_box = QPushButton(" 新增框（左键拖动绘制）")
        self.btn_add_box.setIcon(_make_icon("image"))
        self.btn_add_box.clicked.connect(lambda: self.view.set_mode("draw_box"))
        self.btn_add_poly = QPushButton(" 新增多边形（逐点点击）")
        self.btn_add_poly.setIcon(_make_icon("image"))
        self.btn_add_poly.clicked.connect(lambda: self.view.set_mode("draw_poly"))
        self.btn_finish = QPushButton(" 完成多边形（Enter / 双击）")
        self.btn_finish.clicked.connect(self.finish_poly)
        self.btn_del = QPushButton(" 删除选中（Delete）")
        self.btn_del.clicked.connect(self.delete_selected)
        rv.addWidget(self.btn_add_box)
        rv.addWidget(self.btn_add_poly)
        rv.addWidget(self.btn_finish)
        rv.addWidget(self.btn_del)
        rv.addStretch(1)

        self.btn_save = QPushButton(" 保存（覆盖 + .bak 备份）")
        self.btn_save.setObjectName("primary")
        self.btn_save.setIcon(_make_icon("save"))
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save_next = QPushButton(" 保存并下一张")
        self.btn_save_next.clicked.connect(self.save_and_next)
        rv.addWidget(self.btn_save)
        rv.addWidget(self.btn_save_next)

        root.addWidget(rail)

        # ===================== CENTER =====================
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.setSpacing(8)
        self.lbl_file = QLabel("未加载图片")
        self.lbl_file.setStyleSheet("color:#e8ecf4;font-size:11pt;")
        cv.addWidget(self.lbl_file)
        self.view = ReviewView()
        self.view.on_changed = self.on_edit
        self.view.on_box_drawn = self.on_box_drawn
        self.view.on_finalize_poly = self.on_poly_drawn
        self.view.on_delete_selected = self.delete_items
        self.view.scene.selectionChanged.connect(self.on_selection_changed)
        cv.addWidget(self.view, stretch=1)
        root.addWidget(center, stretch=1)

        # ===================== RIGHT =====================
        right = QWidget()
        right.setFixedWidth(240)
        rw = QVBoxLayout(right)
        rw.setContentsMargins(12, 12, 12, 12)
        rw.setSpacing(10)
        leg = QGroupBox("类别图例（点击改类）")
        _shadow(leg, 18, 3)
        lv = QVBoxLayout(leg)
        self.legend = QListWidget()
        self.legend.itemClicked.connect(self.on_legend_clicked)
        lv.addWidget(self.legend)
        rw.addWidget(leg, 2)
        logc = QGroupBox("日志")
        _shadow(logc, 18, 3)
        gv = QVBoxLayout(logc)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        gv.addWidget(self.log_box)
        rw.addWidget(logc, 3)
        root.addWidget(right)

        self.statusBar().showMessage("未加载")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def _color_for(self, cls: int):
        if self.palette:
            return self.palette[cls % len(self.palette)]
        return (91, 140, 255)

    def _refresh_palette(self):
        self.palette = color_palette(len(self.classes) or 1, bgr=False)
        self.legend.clear()
        for i, n in enumerate(self.classes):
            it = QListWidgetItem(f"{i}: {n}")
            r, g, b = self.palette[i]
            pm = QPixmap(16, 16)
            pm.fill(QColor(r, g, b))
            it.setIcon(QIcon(pm))
            self.legend.addItem(it)

    def _recolor_items(self):
        for it in self.items:
            it.apply_color(self._color_for(it.cls))

    def _update_mode_ui(self):
        is_seg = self.current_mode == "segment"
        self.lbl_mode.setText(f"模式: {'分割' if is_seg else '检测'}")
        self.btn_toggle_mode.setEnabled(not self.items)
        self.btn_toggle_mode.setText(
            "切换为检测模式" if is_seg else "切换为分割模式")
        self.btn_add_box.setVisible(not is_seg)
        self.btn_add_poly.setVisible(is_seg)
        self.btn_finish.setVisible(is_seg)

    # ------------------------------------------------------------------ #
    # folder / class loading
    # ------------------------------------------------------------------ #
    def select_images(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        self.images = []
        self.img_list.clear()
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMAGE_EXTS):
                self.images.append(os.path.join(d, fn))
                self.img_list.addItem(fn)
        self.log(f"加载 {len(self.images)} 张图片。")
        if self.images:
            self.img_list.setCurrentRow(0)

    def select_labels(self):
        d = QFileDialog.getExistingDirectory(self, "选择 labels 文件夹")
        if not d:
            return
        self.labels_dir = d
        self.labels_map = {}
        for fn in os.listdir(d):
            if fn.lower().endswith(".txt") and fn.lower() != "classes.txt":
                stem = os.path.splitext(fn)[0]
                self.labels_map[stem] = os.path.join(d, fn)
        self.log(f"加载 {len(self.labels_map)} 个标签文件。")
        if self.current_image_path is not None:
            self.load_image(self.img_list.currentRow())

    def select_classes(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择 classes.txt", "", "Text (*.txt)")
        if not p:
            return
        with open(p, "r", encoding="utf-8") as f:
            self.classes = [ln.strip() for ln in f if ln.strip()]
        self._refresh_palette()
        self._recolor_items()
        self.current_class = 0
        if self.classes:
            self.legend.setCurrentRow(0)
        self.log(f"加载 {len(self.classes)} 个类别。点击右侧图例行可设当前类别 / 改选中目标的类。")

    # ------------------------------------------------------------------ #
    # image navigation
    # ------------------------------------------------------------------ #
    def on_image_selected(self, row):
        if row < 0 or row >= len(self.images):
            return
        if not self._confirm_switch():
            return
        self.load_image(row)

    def _confirm_switch(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(
            self, "未保存", "当前图片有未保存改动，是否保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            self.save_current()
            return True
        if r == QMessageBox.Discard:
            return True
        return False

    def load_image(self, row):
        path = self.images[row]
        img = cv2.imread(path)
        if img is None:
            self.log(f"[警告] 无法读取: {path}")
            return
        self.img_h, self.img_w = img.shape[:2]
        self.current_image_path = path
        stem = os.path.splitext(os.path.basename(path))[0]
        self.lbl_file.setText(os.path.basename(path))
        self.view.clear_scene()
        self.view.set_image(img)
        self.items = []

        boxes, masks, mode = [], [], "detect"
        lp = self.labels_map.get(stem)
        if lp and os.path.exists(lp):
            boxes, masks, mode = read_yolo_txt(lp)
        self.current_mode = mode

        for b in boxes:
            cx, cy, bw, bh = b["xywhn"]
            x1 = (cx - bw / 2) * self.img_w
            y1 = (cy - bh / 2) * self.img_h
            x2 = (cx + bw / 2) * self.img_w
            y2 = (cy + bh / 2) * self.img_h
            it = EditBoxItem(b["cls"], x1, y1, x2, y2)
            self.view.scene.addItem(it)
            self.items.append(it)
        for m in masks:
            pts = [(p[0] * self.img_w, p[1] * self.img_h) for p in m["xyn"]]
            it = EditPolyItem(m["cls"], pts)
            self.view.scene.addItem(it)
            self.items.append(it)
        self._recolor_items()
        self._update_mode_ui()
        self.dirty = False
        self.view.fit()
        self.statusBar().showMessage(
            f"{os.path.basename(path)}  ·  {mode}  ·  {len(self.items)} 个目标")
        self.log(f"载入 {stem}: {mode}, {len(boxes)} 框, {len(masks)} 多边形")

    def step(self, delta):
        if not self.images:
            return
        r = (self.img_list.currentRow() + delta) % len(self.images)
        self.img_list.setCurrentRow(r)

    def toggle_mode(self):
        if self.items:
            QMessageBox.information(self, "提示", "当前已有目标，不能切换模式。")
            return
        self.current_mode = "segment" if self.current_mode == "detect" else "detect"
        self._update_mode_ui()
        self.on_edit()

    # ------------------------------------------------------------------ #
    # selection / class
    # ------------------------------------------------------------------ #
    def on_selection_changed(self):
        sel = [s for s in self.view.scene.selectedItems()
               if isinstance(s, (EditBoxItem, EditPolyItem))]
        if len(sel) > 1:
            self.statusBar().showMessage(
                f"已选 {len(sel)} 个 -> 点右侧【类别图例】对应行批量改类")

    def on_legend_clicked(self, item):
        if not self.classes:
            return
        row = self.legend.row(item)
        if row < 0:
            return
        self.current_class = row
        sel = [s for s in self.view.scene.selectedItems()
               if isinstance(s, (EditBoxItem, EditPolyItem))]
        if not sel:
            self.log(f"当前类别设为 {row}: {self.classes[row]}（用于新增）")
            return
        for it in sel:
            it.cls = row
            it.apply_color(self._color_for(row))
        self.on_edit()
        self.log(f"已将 {len(sel)} 个目标的类别改为 {row}: {self.classes[row]}")

    # ------------------------------------------------------------------ #
    # add / delete
    # ------------------------------------------------------------------ #
    def on_box_drawn(self, x1, y1, x2, y2):
        cls = self.current_class
        it = EditBoxItem(cls, x1, y1, x2, y2)
        it.apply_color(self._color_for(cls))
        self.view.scene.addItem(it)
        self.items.append(it)
        self.view.scene.clearSelection()
        it.setSelected(True)
        self.on_edit()
        self.log("新增框。")

    def on_poly_drawn(self, pts):
        cls = self.current_class
        it = EditPolyItem(cls, pts)
        it.apply_color(self._color_for(cls))
        self.view.scene.addItem(it)
        self.items.append(it)
        self.view.scene.clearSelection()
        it.setSelected(True)
        self.on_edit()
        self.log("新增多边形。")

    def finish_poly(self):
        if self.view._mode == "draw_poly":
            self.view._finalize_poly()

    def delete_selected(self):
        self.delete_items(self.view.scene.selectedItems())

    def delete_items(self, items):
        n = 0
        for it in list(items):
            if isinstance(it, (EditBoxItem, EditPolyItem)) and it in self.items:
                self.items.remove(it)
                self.view.scene.removeItem(it)
                n += 1
        if n:
            self.on_edit()
            self.log(f"删除 {n} 个目标。")

    def on_edit(self):
        self.dirty = True
        self.statusBar().showMessage(
            f"* 未保存  ·  {self.current_mode}  ·  {len(self.items)} 个目标")

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def _collect(self):
        boxes, masks = [], []
        for it in self.items:
            if isinstance(it, EditBoxItem):
                r = it.rect()
                x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
                cx = (x1 + x2) / 2 / self.img_w
                cy = (y1 + y2) / 2 / self.img_h
                bw = (x2 - x1) / self.img_w
                bh = (y2 - y1) / self.img_h
                boxes.append({"cls": it.cls, "xywhn": [cx, cy, bw, bh]})
            elif isinstance(it, EditPolyItem):
                xyn = [[p.x() / self.img_w, p.y() / self.img_h] for p in it.pts]
                masks.append({"cls": it.cls, "xyn": xyn})
        return boxes, masks

    def save_current(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "没有可保存的图片。")
            return
        stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        lp = self.labels_map.get(stem)
        if lp is None:
            if not self.labels_dir:
                QMessageBox.warning(self, "提示", "未选择 labels 文件夹，无法保存。")
                return
            lp = os.path.join(self.labels_dir, stem + ".txt")
            self.labels_map[stem] = lp
        boxes, masks = self._collect()
        # one-time backup of the original (if it exists and not yet backed up)
        if os.path.exists(lp) and not os.path.exists(lp + ".bak"):
            shutil.copy2(lp, lp + ".bak")
        save_yolo_txt(boxes, masks, lp, self.current_mode,
                      shape=(self.img_h, self.img_w))
        self.dirty = False
        self.log(f"已保存: {os.path.basename(lp)} "
                 f"({len(boxes) + len(masks)} 个目标)")
        self.statusBar().showMessage(f"已保存 {os.path.basename(lp)}")

    def save_and_next(self):
        if self.dirty:
            self.save_current()
        self.step(1)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        if self.dirty and not self._confirm_switch():
            event.ignore()
            return
        event.accept()
