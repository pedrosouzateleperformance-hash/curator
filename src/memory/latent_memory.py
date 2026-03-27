from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List

from src.memory.memory_trace import MemoryRecord, SegmentSummary
from src.pipeline.contracts import MemoryTrace


class LatentMemory:
    def __init__(self, recent_capacity: int = 256, salient_capacity: int = 512) -> None:
        self.recent_states: Deque[MemoryRecord] = deque(maxlen=recent_capacity)
        self.salient_states: Deque[MemoryRecord] = deque(maxlen=salient_capacity)
        self.segment_summaries: List[SegmentSummary] = []
        self.entity_pointers: Dict[str, List[str]] = defaultdict(list)
        self.unresolved_links: List[Dict[str, str]] = []

    def add_state(self, record: MemoryRecord, salience: float = 0.0, entity_links: List[str] | None = None) -> None:
        self.recent_states.append(record)
        if salience > 0.6:
            self.salient_states.append(record)
        for entity_id in entity_links or []:
            self.entity_pointers[entity_id].append(record.token_id)

    def add_segment_summary(self, summary: SegmentSummary) -> None:
        self.segment_summaries.append(summary)

    def add_unresolved_link(self, source_token: str, reason: str) -> None:
        self.unresolved_links.append({"source_token": source_token, "reason": reason})

    def query_by_entity(self, entity_id: str) -> List[str]:
        return self.entity_pointers.get(entity_id, [])

    def query_by_segment(self, segment_id: str) -> List[MemoryRecord]:
        return [r for r in self.recent_states if r.segment_id == segment_id]

    def export(self) -> MemoryTrace:
        retained_states = [
            {
                "token_id": r.token_id,
                "timestamp": r.timestamp,
                "modality": r.modality,
                "segment_id": r.segment_id,
                "state": r.state,
            }
            for r in list(self.salient_states) + list(self.recent_states)
        ]
        summary_states = [
            {
                "segment_id": s.segment_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "summary": s.summary,
                "entity_ids": s.entity_ids,
            }
            for s in self.segment_summaries
        ]
        return MemoryTrace(
            retained_states=retained_states,
            summary_states=summary_states,
            entity_pointers=dict(self.entity_pointers),
            unresolved_links=self.unresolved_links.copy(),
        )
