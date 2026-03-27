from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from data_structures.nodes import ShotNode


@dataclass(frozen=True)
class ShotDetectionConfig:
    hard_cut_threshold: float = 0.45
    soft_min: float = 0.10
    soft_max: float = 0.45
    soft_window: int = 6


def frame_difference_scores(frames: Sequence[np.ndarray]) -> List[float]:
    if len(frames) < 2:
        return []

    scores: List[float] = []
    for prev, curr in zip(frames[:-1], frames[1:]):
        prev_gray = prev.mean(axis=2) if prev.ndim == 3 else prev
        curr_gray = curr.mean(axis=2) if curr.ndim == 3 else curr
        diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        scores.append(float(diff.mean() / 255.0))
    return scores


def detect_shots(
    diff_scores: Iterable[float],
    fps: float,
    duration: float,
    config: ShotDetectionConfig | None = None,
) -> List[ShotNode]:
    cfg = config or ShotDetectionConfig()
    diffs = list(diff_scores)
    if fps <= 0:
        raise ValueError("fps must be positive")

    boundaries = [0]
    transition_labels = {0: ("start", 1.0)}

    i = 0
    while i < len(diffs):
        score = diffs[i]
        if score >= cfg.hard_cut_threshold:
            cut_frame = i + 1
            boundaries.append(cut_frame)
            transition_labels[cut_frame] = ("hard_cut", min(1.0, score))
            i += 1
            continue

        if cfg.soft_min <= score < cfg.soft_max:
            j = i
            window_sum = 0.0
            while j < len(diffs) and cfg.soft_min <= diffs[j] < cfg.soft_max and (j - i) < cfg.soft_window:
                window_sum += diffs[j]
                j += 1
            if (j - i) >= 2:
                cut_frame = j
                avg_score = window_sum / (j - i)
                boundaries.append(cut_frame)
                transition_labels[cut_frame] = ("soft_transition", min(0.99, avg_score + 0.15))
                i = j
                continue
        i += 1

    end_frame = int(round(duration * fps))
    if boundaries[-1] != end_frame:
        boundaries.append(end_frame)

    boundaries = sorted(set(boundaries))
    shots: List[ShotNode] = []

    for idx, (start_f, end_f) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start_time = start_f / fps
        end_time = end_f / fps
        transition_type, confidence = transition_labels.get(start_f, ("continuity", 0.8))
        shots.append(
            ShotNode(
                id=f"shot_{idx:04d}",
                start_time=start_time,
                end_time=end_time,
                duration=max(0.0, end_time - start_time),
                transition_type=transition_type,
                confidence=float(confidence),
            )
        )
    return shots
