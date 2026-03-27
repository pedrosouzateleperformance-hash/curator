from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    timestamp_start: float
    timestamp_end: float
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    salience: float = 0.5


@dataclass
class GraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    weight: float
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


class BaseGraph:
    """Traceable graph with explicit node and edge metadata."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.outgoing: Dict[str, List[str]] = {}
        self.incoming: Dict[str, List[str]] = {}

    def add_or_update_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if not existing:
            self.nodes[node.node_id] = node
            self.outgoing.setdefault(node.node_id, [])
            self.incoming.setdefault(node.node_id, [])
            return
        existing.timestamp_end = max(existing.timestamp_end, node.timestamp_end)
        existing.timestamp_start = min(existing.timestamp_start, node.timestamp_start)
        existing.features.update(node.features)
        existing.confidence = max(existing.confidence, node.confidence)
        existing.salience = max(existing.salience, node.salience)

    def _edge_key(self, source: str, target: str, relation: str) -> str:
        return f"{source}->{target}:{relation}"

    def add_or_update_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float,
        confidence: float,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        edge_id = self._edge_key(source, target, relation)
        if edge_id in self.edges:
            edge = self.edges[edge_id]
            edge.weight = max(edge.weight, weight)
            edge.confidence = max(edge.confidence, confidence)
            if evidence:
                edge.evidence.update(evidence)
            return edge_id
        edge = GraphEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            confidence=confidence,
            evidence=evidence or {},
        )
        self.edges[edge_id] = edge
        self.outgoing.setdefault(source, []).append(edge_id)
        self.incoming.setdefault(target, []).append(edge_id)
        return edge_id

    def get_neighbors(self, node_id: str, relations: Optional[Iterable[str]] = None) -> List[GraphNode]:
        rel_set = set(relations or [])
        result: List[GraphNode] = []
        for edge_id in self.outgoing.get(node_id, []):
            edge = self.edges[edge_id]
            if rel_set and edge.relation not in rel_set:
                continue
            target = self.nodes.get(edge.target)
            if target:
                result.append(target)
        return result

    def multi_hop_paths(self, start: str, max_hops: int = 3, min_weight: float = 0.0) -> List[List[str]]:
        paths: List[List[str]] = []

        def _dfs(current: str, hops_left: int, edge_path: List[str], seen: set[str]) -> None:
            if hops_left == 0:
                return
            for edge_id in self.outgoing.get(current, []):
                edge = self.edges[edge_id]
                if edge.weight < min_weight:
                    continue
                if edge.target in seen:
                    continue
                next_path = edge_path + [edge_id]
                paths.append(next_path)
                _dfs(edge.target, hops_left - 1, next_path, seen | {edge.target})

        _dfs(start, max_hops, [], {start})
        return paths


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sqrt(sum(a * a for a in vec_a))
    mag_b = sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def read_field(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)
