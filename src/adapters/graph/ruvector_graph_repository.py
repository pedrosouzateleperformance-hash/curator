from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.domain.graph import GraphEdge, GraphNode
from src.ports.graph_repository_port import GraphRepositoryPort


@dataclass
class RuVectorNodeDTO:
    id: str
    node_type: str
    layer: str
    attributes: dict[str, Any]


@dataclass
class RuVectorEdgeDTO:
    id: str
    edge_type: str
    layer: str
    source: str
    target: str
    weight: float
    attributes: dict[str, Any]


class RuVectorClientProtocol(Protocol):
    def upsert_node(self, node: RuVectorNodeDTO) -> None: ...

    def upsert_edge(self, edge: RuVectorEdgeDTO) -> None: ...

    def query_nodes(self, query_text: str, limit: int = 20) -> list[dict[str, object]]: ...

    def embed_text(self, text: str) -> list[float]: ...

    def fetch_subgraph(self, node_ids: list[str], depth: int = 1) -> dict[str, object]: ...


class InMemoryRuVectorClient(RuVectorClientProtocol):
    def __init__(self) -> None:
        self._nodes: dict[str, RuVectorNodeDTO] = {}
        self._edges: dict[str, RuVectorEdgeDTO] = {}

    def upsert_node(self, node: RuVectorNodeDTO) -> None:
        self._nodes[node.id] = node

    def upsert_edge(self, edge: RuVectorEdgeDTO) -> None:
        self._edges[edge.id] = edge

    def query_nodes(self, query_text: str, limit: int = 20) -> list[dict[str, object]]:
        if query_text.startswith("active_entities_at:"):
            timestamp = float(query_text.split(":", 1)[1])
            items = [
                asdict(node)
                for node in self._nodes.values()
                if node.id.startswith("entity:")
                and float(node.attributes.get("start", -1.0)) <= timestamp <= float(node.attributes.get("end", -1.0))
            ]
            return items[:limit]

        if query_text.startswith("unresolved_threads"):
            unresolved: list[dict[str, object]] = []
            for node in self._nodes.values():
                if not node.id.startswith("event:"):
                    continue
                outgoing = [edge for edge in self._edges.values() if edge.source == node.id]
                resolved = any(edge.edge_type == "resolves" for edge in outgoing)
                action = str(node.attributes.get("action", ""))
                if not resolved and action != "resolved":
                    unresolved.append(asdict(node))
            return unresolved[:limit]

        if query_text.startswith("dominant_theme:"):
            _, start_raw, end_raw = query_text.split(":", 2)
            start, end = float(start_raw), float(end_raw)
            scores: dict[str, float] = {}
            for edge in self._edges.values():
                if edge.edge_type != "thematic_reinforcement":
                    continue
                source = self._nodes.get(edge.source)
                target = self._nodes.get(edge.target)
                if not source or not target:
                    continue
                s_start = float(source.attributes.get("start", 0.0))
                s_end = float(source.attributes.get("end", 0.0))
                if s_end < start or s_start > end:
                    continue
                theme = str(target.attributes.get("theme", target.id))
                scores[theme] = scores.get(theme, 0.0) + edge.weight
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return [{"theme": key, "score": value} for key, value in ordered[:limit]]

        lower = query_text.lower()
        matched = [
            asdict(node)
            for node in self._nodes.values()
            if lower in node.id.lower() or lower in node.node_type.lower() or lower in node.layer.lower()
        ]
        return matched[:limit]

    def embed_text(self, text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text)
        return [((seed + i * 31) % 1000) / 1000.0 for i in range(12)]

    def fetch_subgraph(self, node_ids: list[str], depth: int = 1) -> dict[str, object]:
        frontier = set(node_ids)
        visited = set(node_ids)
        depth = max(1, depth)
        for _ in range(depth):
            new_nodes: set[str] = set()
            for edge in self._edges.values():
                if edge.source in frontier or edge.target in frontier:
                    new_nodes.add(edge.source)
                    new_nodes.add(edge.target)
            frontier = new_nodes - visited
            visited |= new_nodes
        nodes = [asdict(self._nodes[node_id]) for node_id in visited if node_id in self._nodes]
        edges = [
            asdict(edge)
            for edge in self._edges.values()
            if edge.source in visited and edge.target in visited
        ]
        return {"nodes": nodes, "edges": edges, "depth": depth}


class RuVectorGraphRepository(GraphRepositoryPort):
    """Graph repository adapter that maps domain entities to RuVector DTOs."""

    def __init__(self, client: RuVectorClientProtocol | None = None) -> None:
        self._client = client or InMemoryRuVectorClient()

    def add_node(self, node: GraphNode) -> None:
        self._client.upsert_node(
            RuVectorNodeDTO(id=node.id, node_type=node.type, layer=node.layer, attributes=dict(node.attributes))
        )

    def add_edge(self, edge: GraphEdge) -> None:
        self._client.upsert_edge(
            RuVectorEdgeDTO(
                id=edge.id,
                edge_type=edge.type,
                layer=edge.layer,
                source=edge.source,
                target=edge.target,
                weight=edge.weight,
                attributes=dict(edge.attributes),
            )
        )

    def query(self, query_text: str, *, limit: int = 20) -> list[dict[str, object]]:
        return self._client.query_nodes(query_text=query_text, limit=limit)

    def embed(self, text: str) -> list[float]:
        return self._client.embed_text(text)

    def retrieve_subgraph(self, node_ids: list[str], *, depth: int = 1) -> dict[str, object]:
        return self._client.fetch_subgraph(node_ids=node_ids, depth=depth)
