from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .common import BaseGraph, GraphNode, read_field


@dataclass
class EntityUpdateResult:
    active_entities: List[str]
    interaction_edges: List[str]


class EntityGraph(BaseGraph):
    def __init__(self) -> None:
        super().__init__(name="entity")

    def update_from_segment(self, segment: Any) -> EntityUpdateResult:
        start = float(read_field(segment, "timestamp_start", read_field(segment, "start", 0.0)))
        end = float(read_field(segment, "timestamp_end", read_field(segment, "end", start)))
        entities = read_field(segment, "entities", []) or []
        active: List[str] = []
        for entity in entities:
            if isinstance(entity, dict):
                raw_id = entity.get("id") or entity.get("name")
                entity_type = entity.get("type", "entity")
                attrs = dict(entity)
            else:
                raw_id = str(entity)
                entity_type = "entity"
                attrs = {"label": raw_id}
            node_id = f"entity:{raw_id}"
            self.add_or_update_node(
                GraphNode(
                    node_id=node_id,
                    node_type=entity_type,
                    timestamp_start=start,
                    timestamp_end=end,
                    features=attrs,
                    confidence=0.95,
                    salience=float(attrs.get("salience", 0.5)),
                )
            )
            active.append(node_id)

        interaction_edges: List[str] = []
        for idx, source in enumerate(active):
            for target in active[idx + 1 :]:
                interaction_edges.append(
                    self.add_or_update_edge(
                        source=source,
                        target=target,
                        relation="co_occurrence",
                        weight=0.7,
                        confidence=0.9,
                        evidence={"segment_start": start, "segment_end": end},
                    )
                )
                interaction_edges.append(
                    self.add_or_update_edge(
                        source=target,
                        target=source,
                        relation="co_occurrence",
                        weight=0.7,
                        confidence=0.9,
                        evidence={"segment_start": start, "segment_end": end},
                    )
                )

        return EntityUpdateResult(active_entities=active, interaction_edges=interaction_edges)

    def active_entities_at(self, timestamp: float) -> List[str]:
        return [
            node_id
            for node_id, node in self.nodes.items()
            if node.timestamp_start <= timestamp <= node.timestamp_end
        ]

    def reappearing_entities(self, min_span: float = 10.0) -> List[str]:
        results: List[str] = []
        for node_id, node in self.nodes.items():
            if node.timestamp_end - node.timestamp_start >= min_span:
                results.append(node_id)
        return results
