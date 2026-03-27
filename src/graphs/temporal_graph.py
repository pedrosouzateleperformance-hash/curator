from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import BaseGraph, GraphNode, read_field


@dataclass
class TemporalUpdateResult:
    node_id: str
    before_edge: str | None
    overlap_edges: list[str]


class TemporalGraph(BaseGraph):
    def __init__(self) -> None:
        super().__init__(name="temporal")
        self._last_node_id: str | None = None

    def update_from_segment(self, segment: Any) -> TemporalUpdateResult:
        segment_id = str(read_field(segment, "segment_id", read_field(segment, "id", "unknown")))
        start = float(read_field(segment, "timestamp_start", read_field(segment, "start", 0.0)))
        end = float(read_field(segment, "timestamp_end", read_field(segment, "end", start)))
        duration = max(0.0, end - start)
        node = GraphNode(
            node_id=f"segment:{segment_id}",
            node_type="segment",
            timestamp_start=start,
            timestamp_end=end,
            features={"duration": duration, "pacing_hint": read_field(segment, "pacing", "steady")},
            confidence=float(read_field(segment, "confidence", 1.0)),
            salience=float(read_field(segment, "salience", 0.5)),
        )
        self.add_or_update_node(node)

        before_edge = None
        if self._last_node_id and self._last_node_id != node.node_id:
            prev = self.nodes[self._last_node_id]
            gap = start - prev.timestamp_end
            relation = "after" if gap >= 0 else "overlap"
            before_edge = self.add_or_update_edge(
                source=self._last_node_id,
                target=node.node_id,
                relation=relation,
                weight=1.0 / (1.0 + abs(gap)),
                confidence=0.9,
                evidence={"gap": gap},
            )

        overlap_edges: list[str] = []
        for existing_id, existing in list(self.nodes.items()):
            if existing_id == node.node_id:
                continue
            overlap = min(existing.timestamp_end, end) - max(existing.timestamp_start, start)
            if overlap > 0:
                overlap_edges.append(
                    self.add_or_update_edge(
                        source=existing_id,
                        target=node.node_id,
                        relation="overlap",
                        weight=min(1.0, overlap / max(duration, 1e-6)),
                        confidence=0.8,
                        evidence={"overlap": overlap},
                    )
                )

        self._last_node_id = node.node_id
        return TemporalUpdateResult(node_id=node.node_id, before_edge=before_edge, overlap_edges=overlap_edges)

    def current_pacing_profile(self) -> Dict[str, float]:
        durations = [float(node.features.get("duration", 0.0)) for node in self.nodes.values()]
        if not durations:
            return {"avg_duration": 0.0, "tempo": 0.0}
        avg = sum(durations) / len(durations)
        tempo = 1.0 / max(avg, 1e-6)
        return {"avg_duration": avg, "tempo": tempo}
