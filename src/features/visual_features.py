from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from data_structures.nodes import FrameFeature


@dataclass(frozen=True)
class VisualFeatureConfig:
    saliency_downsample: int = 8
    object_threshold: int = 180


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return frame.mean(axis=2).astype(np.float32)


def _extract_objects(gray: np.ndarray, threshold: int) -> List[Dict[str, float]]:
    mask = gray > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [{"bbox": [int(x1), int(y1), int(x2), int(y2)], "label": "bright_region", "confidence": 0.5}]


def _extract_faces(frame: np.ndarray) -> List[Dict[str, float]]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    gray_u8 = _to_gray(frame).astype(np.uint8)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected = cascade.detectMultiScale(gray_u8, scaleFactor=1.1, minNeighbors=4)
    faces = []
    for idx, (x, y, w, h) in enumerate(detected):
        faces.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "track_id": f"face_local_{idx}"})
    return faces


def _motion_vectors(prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> List[List[float]]:
    if prev_gray is None:
        return [[0.0, 0.0]]
    delta = curr_gray - prev_gray
    gy, gx = np.gradient(delta)
    return [[float(gx.mean()), float(gy.mean())]]


def _saliency_map(gray: np.ndarray, downsample: int) -> List[List[float]]:
    gy, gx = np.gradient(gray)
    sal = np.sqrt(gx ** 2 + gy ** 2)
    if sal.max() > 0:
        sal = sal / sal.max()
    ds = max(1, downsample)
    small = sal[::ds, ::ds]
    return small.round(4).tolist()


def _composition_metrics(gray: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape
    weights = gray + 1e-6
    y_idx, x_idx = np.indices(gray.shape)
    total = weights.sum()
    cx = float((x_idx * weights).sum() / total)
    cy = float((y_idx * weights).sum() / total)
    return {
        "center_of_mass_x": cx / max(1.0, w),
        "center_of_mass_y": cy / max(1.0, h),
        "balance": abs(0.5 - cx / max(1.0, w)) + abs(0.5 - cy / max(1.0, h)),
    }


def extract_frame_features(
    frame_id: int,
    timestamp: float,
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray] = None,
    config: VisualFeatureConfig | None = None,
) -> FrameFeature:
    cfg = config or VisualFeatureConfig()
    gray = _to_gray(frame)
    prev_gray = _to_gray(prev_frame) if prev_frame is not None else None

    return FrameFeature(
        frame_id=frame_id,
        timestamp=timestamp,
        objects=_extract_objects(gray, cfg.object_threshold),
        faces=_extract_faces(frame),
        motion_vectors=_motion_vectors(prev_gray, gray),
        saliency_map=_saliency_map(gray, cfg.saliency_downsample),
        composition_metrics=_composition_metrics(gray),
    )
