from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MemoryRecord:
    token_id: str
    timestamp: float
    modality: str
    state: List[float]
    segment_id: str


@dataclass
class SegmentSummary:
    segment_id: str
    start_time: float
    end_time: float
    summary: List[float]
    entity_ids: List[str]
