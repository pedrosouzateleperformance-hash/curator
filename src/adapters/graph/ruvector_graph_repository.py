from __future__ import annotations

from dataclasses import asdict

from src.domain.graph import GraphEdge, GraphNode
from src.ports.graph import GraphRepositoryPort


class RuVectorGraphRepository(GraphRepositoryPort):
    """Adapter facade for an external RuVector graph service."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges[edge.id] = edge

    def query(self, query_text: str, *, limit: int = 20) -> list[dict[str, object]]:
        lower = query_text.lower()
        matched = [asdict(node) for node in self._nodes.values() if lower in node.id.lower() or lower in node.type.lower()]
        return matched[:limit]

    def embed(self, text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text)
        return [((seed + i * 31) % 1000) / 1000.0 for i in range(12)]

    def retrieve_subgraph(self, node_ids: list[str], *, depth: int = 1) -> dict[str, object]:
        nodes = [asdict(self._nodes[node_id]) for node_id in node_ids if node_id in self._nodes]
        edges = [
            asdict(edge)
            for edge in self._edges.values()
            if edge.source in node_ids or edge.target in node_ids
        ]
        return {"nodes": nodes, "edges": edges, "depth": depth}
