from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .common import BaseGraph, GraphNode, cosine_similarity, read_field


@dataclass
class CausalUpdateResult:
    node_id: str
    causal_edges: List[str]


class CausalGraph(BaseGraph):
    def __init__(self, causal_threshold: float = 0.55) -> None:
        super().__init__(name="causal")
        self.causal_threshold = causal_threshold
        self._ordered_events: list[str] = []

    def update_from_segment(self, segment: Any) -> CausalUpdateResult:
        event_id = str(read_field(segment, "event_id", read_field(segment, "segment_id", read_field(segment, "id", "unknown"))))
        start = float(read_field(segment, "timestamp_start", read_field(segment, "start", 0.0)))
        end = float(read_field(segment, "timestamp_end", read_field(segment, "end", start)))

        embedding = list(read_field(segment, "causal_embedding", read_field(segment, "semantic_embedding", [])) or [])
        action = read_field(segment, "action", "event")
        states = read_field(segment, "state_changes", []) or []

        node = GraphNode(
            node_id=f"event:{event_id}",
            node_type="event",
            timestamp_start=start,
            timestamp_end=end,
            features={"action": action, "state_changes": states, "embedding": embedding},
            confidence=float(read_field(segment, "causal_confidence", 0.75)),
            salience=float(read_field(segment, "salience", 0.5)),
        )
        self.add_or_update_node(node)

        causal_edges: List[str] = []
        for previous_id in reversed(self._ordered_events[-8:]):
            prev_node = self.nodes[previous_id]
            if prev_node.timestamp_start > node.timestamp_start:
                continue
            prev_embedding = list(prev_node.features.get("embedding", []) or [])
            similarity = cosine_similarity(prev_embedding, embedding)
            if similarity < self.causal_threshold:
                continue
            relation = "causes"
            if "conflict" in str(action) or "conflict" in str(prev_node.features.get("action", "")):
                relation = "conflicts"
            elif states and any("resolved" in str(s) for s in states):
                relation = "resolves"
            elif abs(node.timestamp_start - prev_node.timestamp_end) <= 1.5:
                relation = "enables"

            causal_edges.append(
                self.add_or_update_edge(
                    source=previous_id,
                    target=node.node_id,
                    relation=relation,
                    weight=similarity,
                    confidence=min(0.99, 0.5 + 0.5 * similarity),
                    evidence={"temporal_delta": node.timestamp_start - prev_node.timestamp_end},
                )
            )

        self._ordered_events.append(node.node_id)
        return CausalUpdateResult(node_id=node.node_id, causal_edges=causal_edges)

    def chain_to_event(self, event_node_id: str, max_depth: int = 4) -> list[list[str]]:
        chains: list[list[str]] = []

        def _walk(current: str, depth: int, current_chain: list[str]) -> None:
            if depth == 0:
                chains.append(list(current_chain))
                return
            incoming_edges = self.incoming.get(current, [])
            if not incoming_edges:
                chains.append(list(current_chain))
                return
            for edge_id in incoming_edges:
                edge = self.edges[edge_id]
                if edge.relation not in {"causes", "enables", "resolves", "conflicts"}:
                    continue
                _walk(edge.source, depth - 1, [edge_id] + current_chain)

        _walk(event_node_id, max_depth, [])
        return chains
