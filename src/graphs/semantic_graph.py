from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .common import BaseGraph, GraphNode, cosine_similarity, read_field


@dataclass
class SemanticUpdateResult:
    node_id: str
    similarity_edges: List[str]
    theme_edges: List[str]


class SemanticGraph(BaseGraph):
    def __init__(self, similarity_threshold: float = 0.65) -> None:
        super().__init__(name="semantic")
        self.similarity_threshold = similarity_threshold

    def update_from_segment(self, segment: Any, active_entities: list[str]) -> SemanticUpdateResult:
        segment_id = str(read_field(segment, "segment_id", read_field(segment, "id", "unknown")))
        start = float(read_field(segment, "timestamp_start", read_field(segment, "start", 0.0)))
        end = float(read_field(segment, "timestamp_end", read_field(segment, "end", start)))
        embedding = list(read_field(segment, "semantic_embedding", []) or [])
        concepts = read_field(segment, "concepts", []) or []
        themes = read_field(segment, "themes", []) or []

        node = GraphNode(
            node_id=f"semantic:{segment_id}",
            node_type="semantic_segment",
            timestamp_start=start,
            timestamp_end=end,
            features={"concepts": concepts, "themes": themes, "embedding": embedding},
            confidence=float(read_field(segment, "semantic_confidence", 0.85)),
            salience=float(read_field(segment, "salience", 0.5)),
        )
        self.add_or_update_node(node)

        similarity_edges: List[str] = []
        theme_edges: List[str] = []

        for existing_id, existing in list(self.nodes.items()):
            if existing_id == node.node_id:
                continue
            existing_embedding = list(existing.features.get("embedding", []) or [])
            sim = cosine_similarity(embedding, existing_embedding)
            if sim >= self.similarity_threshold:
                similarity_edges.append(
                    self.add_or_update_edge(
                        source=existing_id,
                        target=node.node_id,
                        relation="similarity",
                        weight=sim,
                        confidence=min(0.99, 0.6 + 0.4 * sim),
                        evidence={"type": "latent_similarity"},
                    )
                )

        for theme in themes:
            theme_id = f"theme:{theme}"
            self.add_or_update_node(
                GraphNode(
                    node_id=theme_id,
                    node_type="theme",
                    timestamp_start=start,
                    timestamp_end=end,
                    features={"theme": theme},
                    confidence=0.8,
                    salience=0.6,
                )
            )
            theme_edges.append(
                self.add_or_update_edge(
                    source=node.node_id,
                    target=theme_id,
                    relation="thematic_reinforcement",
                    weight=0.75,
                    confidence=0.85,
                    evidence={"entity_context": active_entities},
                )
            )

        return SemanticUpdateResult(node_id=node.node_id, similarity_edges=similarity_edges, theme_edges=theme_edges)

    def dominant_theme(self, start: float, end: float) -> str | None:
        scores: dict[str, float] = {}
        for edge in self.edges.values():
            if edge.relation != "thematic_reinforcement":
                continue
            source = self.nodes.get(edge.source)
            target = self.nodes.get(edge.target)
            if not source or not target:
                continue
            if source.timestamp_end < start or source.timestamp_start > end:
                continue
            theme_name = str(target.features.get("theme", target.node_id))
            scores[theme_name] = scores.get(theme_name, 0.0) + edge.weight
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]
