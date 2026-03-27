from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    resolution: Tuple[int, int]
    frame_count: int
    duration: float

    def frame_to_time(self, frame_index: int) -> float:
        return frame_index / self.fps

    def time_to_frame(self, timestamp: float) -> int:
        return int(round(timestamp * self.fps))


class VideoLoader:
    """Deterministic video ingest with frame index mapping."""

    def __init__(self, path: str):
        self.path = path
        self._cv2 = None

    def _require_cv2(self):
        if self._cv2 is None:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for direct video ingest") from exc
            self._cv2 = cv2
        return self._cv2

    def load_metadata(self) -> VideoMetadata:
        cv2 = self._require_cv2()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(fps=fps, resolution=(width, height), frame_count=frame_count, duration=duration)

    def iter_frames(self, stride: int = 1) -> Iterator[Tuple[int, float, "object"]]:
        cv2 = self._require_cv2()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, index / fps, frame
            index += 1
        cap.release()


def ingest_video(path: str) -> Dict[str, object]:
    metadata = VideoLoader(path).load_metadata()
    return {
        "fps": metadata.fps,
        "resolution": metadata.resolution,
        "frame_count": metadata.frame_count,
        "duration": metadata.duration,
        "frame_index_mapping": {
            "frame_to_time": "t = frame_id / fps",
            "time_to_frame": "frame_id = round(t * fps)",
        },
    }
