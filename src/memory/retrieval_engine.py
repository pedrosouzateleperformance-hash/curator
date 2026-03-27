from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.graphs.causal_graph import CausalGraph
from src.graphs.entity_graph import EntityGraph
from src.graphs.semantic_graph import SemanticGraph
from src.graphs.temporal_graph import TemporalGraph


@dataclass
class RetrievalResult:
    query: str
    node_ids: List[str]
    edge_ids: List[str]
    confidence: float


class RetrievalEngine:
    def __init__(
        self,
        temporal_graph: TemporalGraph,
        entity_graph: EntityGraph,
        semantic_graph: SemanticGraph,
        causal_graph: CausalGraph,
    ) -> None:
        self.temporal_graph = temporal_graph
        self.entity_graph = entity_graph
        self.semantic_graph = semantic_graph
        self.causal_graph = causal_graph

    def what_led_to_event(self, event_node_id: str, max_hops: int = 4) -> RetrievalResult:
        chains = self.causal_graph.chain_to_event(event_node_id, max_depth=max_hops)
        edge_ids = sorted({edge for chain in chains for edge in chain})
        node_ids = sorted(
            {
                self.causal_graph.edges[edge_id].source
                for edge_id in edge_ids
                if edge_id in self.causal_graph.edges
            }
            | {event_node_id}
        )
        confidence = 0.0
        if edge_ids:
            confidence = sum(self.causal_graph.edges[e].confidence for e in edge_ids) / len(edge_ids)
        return RetrievalResult(
            query=f"what_led_to_event:{event_node_id}",
            node_ids=node_ids,
            edge_ids=edge_ids,
            confidence=confidence,
        )

    def active_entities(self, timestamp: float) -> RetrievalResult:
        nodes = self.entity_graph.active_entities_at(timestamp)
        return RetrievalResult(
            query=f"active_entities:{timestamp}",
            node_ids=nodes,
            edge_ids=[],
            confidence=0.9 if nodes else 0.3,
        )

    def unresolved_threads(self) -> RetrievalResult:
        unresolved_nodes: list[str] = []
        supporting_edges: list[str] = []
        for node_id, node in self.causal_graph.nodes.items():
            outgoing = self.causal_graph.outgoing.get(node_id, [])
            resolved = any(self.causal_graph.edges[e].relation == "resolves" for e in outgoing)
            if not resolved and node.features.get("action") != "resolved":
                unresolved_nodes.append(node_id)
                supporting_edges.extend(outgoing)
        conf = 0.8 if unresolved_nodes else 0.4
        return RetrievalResult(
            query="unresolved_threads",
            node_ids=sorted(unresolved_nodes),
            edge_ids=sorted(set(supporting_edges)),
            confidence=conf,
        )

    def dominant_theme(self, start: float, end: float) -> RetrievalResult:
        theme = self.semantic_graph.dominant_theme(start, end)
        nodes = [f"theme:{theme}"] if theme else []
        edges = [
            edge_id
            for edge_id, edge in self.semantic_graph.edges.items()
            if edge.relation == "thematic_reinforcement" and edge.target in nodes
        ]
        return RetrievalResult(
            query=f"dominant_theme:{start}-{end}",
            node_ids=nodes,
            edge_ids=edges,
            confidence=0.85 if theme else 0.2,
        )

    def pacing_profile(self) -> RetrievalResult:
        profile = self.temporal_graph.current_pacing_profile()
        node_ids = sorted(self.temporal_graph.nodes.keys())[-5:]
        confidence = 0.85 if node_ids else 0.3
        return RetrievalResult(
            query=f"pacing_profile:{profile}",
            node_ids=node_ids,
            edge_ids=[],
            confidence=confidence,
        )

    def weighted_path_search(
        self,
        graph_name: str,
        start_node: str,
        relation_filter: Optional[set[str]] = None,
        max_hops: int = 3,
    ) -> RetrievalResult:
        graph_map = {
            "temporal": self.temporal_graph,
            "entity": self.entity_graph,
            "semantic": self.semantic_graph,
            "causal": self.causal_graph,
        }
        graph = graph_map[graph_name]
        paths = graph.multi_hop_paths(start_node, max_hops=max_hops)
        edge_ids: list[str] = []
        for path in paths:
            for edge_id in path:
                edge = graph.edges[edge_id]
                if relation_filter and edge.relation not in relation_filter:
                    continue
                edge_ids.append(edge_id)
        edge_ids = sorted(set(edge_ids))
        node_ids = sorted({start_node} | {graph.edges[e].target for e in edge_ids})
        conf = 0.0
        if edge_ids:
            conf = sum(graph.edges[e].weight for e in edge_ids) / len(edge_ids)
        return RetrievalResult(
            query=f"weighted_path_search:{graph_name}:{start_node}",
            node_ids=node_ids,
            edge_ids=edge_ids,
            confidence=conf,
        )
