"""Annotation I/O for the YOLO-based AutoMask tool.

Supported outputs:
  - YOLO txt  (boxes: ``class cx cy w h`` normalized; seg: ``class`` + normalized polygon)
  - Pascal VOC XML
  - COCO JSON  (accumulated across images via :class:`CocoWriter`)
  - Visualization JPG (boxes + labels, masks filled)
  - classes.txt / dataset.yaml
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


def _yaml_quote(s) -> str:
    """Quote a class name for safe YAML emission (handles : , # etc.)."""
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def color_palette(n: int, bgr: bool = True) -> List[tuple]:
    """Return ``n`` visually distinct colors.

    (B, G, R) tuples by default (OpenCV friendly), or (R, G, B) when
    ``bgr=False``.
    """
    import colorsys

    if n <= 0:
        return []
    palette: List[tuple] = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        if bgr:
            palette.append((int(b * 255), int(g * 255), int(r * 255)))
        else:
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


# --------------------------------------------------------------------------- #
# YOLO txt
# --------------------------------------------------------------------------- #
def save_yolo_txt(boxes, masks, out_path: str, task: str,
                  shape=None) -> str:
    lines: List[str] = []
    if task == "segment" and masks:
        for m in masks:
            cls = m["cls"]
            pts = m.get("xyn")
            if pts is None and shape is not None:
                h, w = shape
                pts = [[x / w, y / h] for x, y in m["poly"]]
            if pts:
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                lines.append(f"{cls} {flat}")
    else:
        for b in boxes:
            cx, cy, bw, bh = b["xywhn"]
            lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(out_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")
    return out_path


def read_yolo_txt(path: str):
    """Read a YOLO label file back into plain structures.

    Returns ``(boxes, masks, mode)`` where ``mode`` is ``'detect'`` or
    ``'segment'`` (auto-detected from the content):

      - a line with 4 values after the class id -> detection box
        ``{'cls': int, 'xywhn': [cx, cy, w, h]}`` (normalized)
      - a line with >=6 values (even count) -> segmentation polygon
        ``{'cls': int, 'xyn': [[x, y], ...]}`` (normalized)

    Empty lines are skipped; malformed lines are skipped silently. A missing
    file returns empty results with mode ``'detect'``.
    """
    boxes: List[dict] = []
    masks: List[dict] = []
    mode = "detect"
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                if len(vals) == 4:
                    boxes.append({"cls": cls, "xywhn": vals})
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    pts = [[vals[i], vals[i + 1]]
                           for i in range(0, len(vals), 2)]
                    masks.append({"cls": cls, "xyn": pts})
                    mode = "segment"
                # else: malformed, skip
    except FileNotFoundError:
        return [], [], "detect"
    return boxes, masks, mode


# --------------------------------------------------------------------------- #
# Pascal VOC XML
# --------------------------------------------------------------------------- #
def save_voc_xml(boxes, names, out_path: str, img_w: int, img_h: int,
                 filename: str) -> str:
    p: List[str] = []
    p.append('<?xml version="1.0" encoding="utf-8"?>')
    p.append("<annotation>")
    p.append("  <folder>images</folder>")
    p.append(f"  <filename>{filename}</filename>")
    p.append(f"  <size><width>{img_w}</width><height>{img_h}</height>"
             f"<depth>3</depth></size>")
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        p.append("  <object>")
        p.append(f"    <name>{names[b['cls']]}</name>")
        p.append("    <pose>Unspecified</pose>"
                 "<truncated>0</truncated><difficult>0</difficult>")
        p.append(f"    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                 f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>")
        p.append("  </object>")
    p.append("</annotation>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(p) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
# COCO JSON (accumulated)
# --------------------------------------------------------------------------- #
class CocoWriter:
    def __init__(self, classes: Sequence[str]):
        self.classes: List[str] = list(classes)
        self.images: List[dict] = []
        self.annotations: List[dict] = []
        self.categories: List[dict] = [
            {"id": i, "name": n, "supercategory": "none"}
            for i, n in enumerate(classes)
        ]
        self._img_id = 0
        self._ann_id = 0

    def add(self, boxes, masks, image_stem: str, height: int, width: int,
            file_name: Optional[str] = None) -> None:
        self._img_id += 1
        img_id = self._img_id
        if file_name is None:
            file_name = f"{image_stem}.jpg"
        self.images.append({
            "id": img_id,
            "file_name": file_name,
            "height": int(height),
            "width": int(width),
        })
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            self._ann_id += 1
            self.annotations.append({
                "id": self._ann_id,
                "image_id": img_id,
                "category_id": b["cls"],
                "bbox": [x1, y1, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
                "segmentation": [],
            })
        for m in masks:
            poly = m.get("poly") or []
            if len(poly) < 3:
                continue
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            pts = [float(c) for coord in poly for c in coord]
            area = float(cv2.contourArea(np.array(poly, dtype=np.float32)))
            self._ann_id += 1
            self.annotations.append({
                "id": self._ann_id,
                "image_id": img_id,
                "category_id": m["cls"],
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "area": area,
                "iscrowd": 0,
                "segmentation": [pts],
            })

    def write(self, out_path: str) -> str:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "images": self.images,
                "annotations": self.annotations,
                "categories": self.categories,
            }, f, ensure_ascii=False, indent=2)
        return out_path


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def draw_results(image_bgr, boxes, masks, names,
                 alpha: float = 0.4, show_conf: bool = True):
    palette = color_palette(len(names) or 1, bgr=True)
    ncol = len(palette)

    def _color(cls: int):
        # Guard against cls out of range (e.g. names override too short);
        # cycle through the palette instead of IndexError-ing.
        return palette[cls % ncol] if ncol else (128, 128, 128)

    out = image_bgr.copy()
    if masks:
        overlay = out.copy()
        for m in masks:
            poly = np.array(m["poly"], dtype=np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(overlay, [poly], _color(m["cls"]))
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        color = _color(b["cls"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = names[b["cls"]] if 0 <= b["cls"] < len(names) else str(b["cls"])
        if show_conf:
            label = f"{label} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 4, 0)),
                      (x1 + tw, max(y1 - 2, 0)), color, -1)
        cv2.putText(out, label, (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def save_vis(image_bgr, boxes, masks, names, out_path: str,
             alpha: float = 0.4) -> str:
    out = draw_results(image_bgr, boxes, masks, names, alpha)
    cv2.imwrite(out_path, out)
    return out_path


# --------------------------------------------------------------------------- #
# class list / yaml
# --------------------------------------------------------------------------- #
def save_classes_txt(names, out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")
    return out_path


def save_dataset_yaml(names, out_path: str, task: str = "detect") -> str:
    quoted = ", ".join(_yaml_quote(n) for n in names)
    body = f"task: {task}\nnc: {len(names)}\nnames: [{quoted}]\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    return out_path


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def save_annotation(image_bgr, result, out_dir: str, image_stem: str,
                    opts: Optional[Dict] = None,
                    image_name: Optional[str] = None) -> Dict[str, str]:
    """Save one image's annotations according to ``opts``.

    opts keys (optional, defaults shown):
        save_yolo (True), save_voc (False), save_vis (True),
        save_classes (True), save_yaml (False), save_coco (False),
        vis_alpha (0.4)
    ``image_name`` is the original filename (with extension) written into
    VOC/COCO metadata; defaults to ``{image_stem}.jpg``.
    Returns a dict mapping kind -> written path.
    """
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    boxes = result.get("boxes", [])
    masks = result.get("masks", [])
    names = result.get("names", [])
    task = result.get("task", "detect")
    h, w = result.get("shape", image_bgr.shape[:2])
    if image_name is None:
        image_name = image_stem + ".jpg"
    written: Dict[str, str] = {}

    if opts.get("save_yolo", True):
        p = os.path.join(out_dir, f"{image_stem}.txt")
        save_yolo_txt(boxes, masks, p, task, shape=(h, w))
        written["yolo"] = p

    if opts.get("save_voc", False):
        p = os.path.join(out_dir, f"{image_stem}.xml")
        save_voc_xml(boxes, names, p, w, h, image_name)
        written["voc"] = p

    if opts.get("save_vis", True):
        p = os.path.join(out_dir, f"{image_stem}_vis.jpg")
        save_vis(image_bgr, boxes, masks, names, p,
                 alpha=float(opts.get("vis_alpha", 0.4)))
        written["vis"] = p

    if opts.get("save_classes", True):
        p = os.path.join(out_dir, "classes.txt")
        save_classes_txt(names, p)
        written["classes"] = p

    if opts.get("save_yaml", False):
        p = os.path.join(out_dir, "dataset.yaml")
        save_dataset_yaml(names, p, task)
        written["yaml"] = p

    if opts.get("save_coco", False):
        writer = CocoWriter(names)
        writer.add(boxes, masks, image_stem, h, w, file_name=image_name)
        p = os.path.join(out_dir, f"{image_stem}.coco.json")
        writer.write(p)
        written["coco"] = p

    return written
