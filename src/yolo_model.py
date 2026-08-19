"""YOLO model manager for the AutoMask automatic annotation tool.

Wraps ``ultralytics.YOLO`` to load a trained weights file (.pt) and run
CPU-friendly inference, returning plain Python / numpy structures so the
rest of the app (UI, annotation I/O) stays independent of ultralytics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ModelLoadError(Exception):
    pass


class YoloModel:
    def __init__(self):
        self.model = None
        self.names: List[str] = []
        self.weights_path: Optional[str] = None
        self.task: Optional[str] = None  # 'detect' | 'segment' | 'classify'
        self.device: str = "cpu"
        self._ultralytics_available = None

    # ------------------------------------------------------------------ #
    def ultralytics_available(self) -> bool:
        if self._ultralytics_available is None:
            try:
                import ultralytics  # noqa: F401
                self._ultralytics_available = True
            except Exception:
                self._ultralytics_available = False
        return self._ultralytics_available

    # ------------------------------------------------------------------ #
    def load(self, weights: str, device: str = "cpu",
             names_override: Optional[List[str]] = None) -> None:
        if not self.ultralytics_available():
            raise ModelLoadError(
                "未检测到 ultralytics，请在 goal 环境中运行本程序。"
            )
        if not _is_file(weights):
            raise ModelLoadError(f"权重文件不存在: {weights}")

        from ultralytics import YOLO

        try:
            self.model = YOLO(weights)
        except Exception as e:  # pragma: no cover - depends on user model
            raise ModelLoadError(f"模型加载失败: {e}") from e

        self.weights_path = weights
        self.device = device

        raw = getattr(self.model, "names", None)
        if isinstance(raw, dict):
            names = [raw[i] for i in sorted(raw.keys())]
        elif isinstance(raw, (list, tuple)):
            names = list(raw)
        else:
            names = []

        if names_override:
            names = [n.strip() for n in names_override if n.strip()]
        if not names:
            names = ["class_0"]
        self.names = names
        self.task = getattr(self.model, "task", None)

    # ------------------------------------------------------------------ #
    def predict(self, image, conf: float = 0.25, iou: float = 0.7) -> Dict:
        if self.model is None:
            raise ModelLoadError("模型尚未加载，请先加载模型。")

        try:
            results = self.model.predict(
                source=image, device=self.device, conf=conf, iou=iou,
                save=False, verbose=False,
            )
        except Exception as e:  # pragma: no cover
            raise ModelLoadError(f"推理失败: {e}") from e

        r = results[0]
        if hasattr(r, "orig_shape"):
            h, w = r.orig_shape
        else:  # pragma: no cover
            h, w = image.shape[:2]

        boxes: List[Dict] = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                cls = int(b.cls[0])
                conf_ = float(b.conf[0]) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
                xywhn = b.xywhn[0].detach().cpu().numpy().tolist()
                boxes.append({
                    "cls": cls, "conf": conf_,
                    "xyxy": xyxy, "xywhn": xywhn,
                })

        masks: List[Dict] = []
        if getattr(r, "masks", None) is not None and len(r.masks):
            for i, poly in enumerate(r.masks.xy):
                cls = int(r.boxes.cls[i]) if r.boxes is not None else 0
                xyn = None
                if hasattr(r.masks, "xyn"):
                    try:
                        # ultralytics 8.x returns numpy arrays (no .detach);
                        # np.asarray handles both ndarray and tensor cases.
                        xyn = np.asarray(r.masks.xyn[i]).tolist()
                    except Exception:
                        xyn = None
                masks.append({
                    "cls": cls,
                    "poly": np.array(poly, dtype=float).tolist(),
                    "xyn": xyn,
                })

        if masks:
            task = "segment"
        elif boxes:
            task = "detect"
        else:
            task = self.task or "classify"

        return {
            "boxes": boxes,
            "masks": masks,
            "names": self.names,
            "task": task,
            "shape": (int(h), int(w)),
        }

    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def num_classes(self) -> int:
        return len(self.names)


def _is_file(p: str) -> bool:
    import os
    return os.path.isfile(p)
