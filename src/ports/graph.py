from __future__ import annotations

from typing import Protocol

from src.domain.graph import GraphEdge, GraphNode


class GraphRepositoryPort(Protocol):
    def add_node(self, node: GraphNode) -> None: ...

    def add_edge(self, edge: GraphEdge) -> None: ...

    def query(self, query_text: str, *, limit: int = 20) -> list[dict[str, object]]: ...

    def embed(self, text: str) -> list[float]: ...

    def retrieve_subgraph(self, node_ids: list[str], *, depth: int = 1) -> dict[str, object]: ...
