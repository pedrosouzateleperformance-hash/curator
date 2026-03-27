from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Deque, Dict, List

from src.domain.graph import GraphEdge, GraphNode
from src.ports.graph_repository_port import GraphRepositoryPort


@dataclass
class RetrievalResult:
    query: str
    node_ids: List[str]
    edge_ids: List[str]
    confidence: float


@dataclass
class MemorySnapshot:
    short_term_segments: List[str]
    long_term_summaries: List[Dict[str, Any]]
    unresolved_threads: List[str]
    salient_nodes: List[str]


@dataclass
class GraphState:
    temporal_node_ids: List[str] = field(default_factory=list)
    entity_node_ids: List[str] = field(default_factory=list)
    semantic_node_ids: List[str] = field(default_factory=list)
    causal_node_ids: List[str] = field(default_factory=list)
    entity_co_occurrence_edges: List[str] = field(default_factory=list)
    causal_edge_ids: List[str] = field(default_factory=list)
    unresolved_thread_ids: List[str] = field(default_factory=list)
    avg_segment_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryOrchestrator:
    def __init__(self, graph_repository: GraphRepositoryPort, active_window_size: int = 12):
        self._graph = graph_repository
        self.active_window_size = active_window_size
        self.active_segments: Deque[str] = deque(maxlen=active_window_size)
        self.segment_store: Dict[str, Any] = {}
        self.long_term_summaries: List[Dict[str, Any]] = []
        self.unresolved_threads: set[str] = set()

    def update(self, sequence_state: Any, segment_state: Any, fusion_trace: Any, memory_trace: Any) -> GraphState:
        segment_id = str(getattr(segment_state, "segment_id", None) or segment_state.get("segment_id") or segment_state.get("id"))
        start = float(segment_state.get("timestamp_start", segment_state.get("start", 0.0)))
        end = float(segment_state.get("timestamp_end", segment_state.get("end", start)))
        duration = max(0.0, end - start)

        self.segment_store[segment_id] = {
            "sequence_state": sequence_state,
            "segment_state": segment_state,
            "fusion_trace": fusion_trace,
            "memory_trace": memory_trace,
        }

        temporal_node_id = f"segment:{segment_id}"
        self._graph.add_node(
            GraphNode(
                id=temporal_node_id,
                type="segment",
                layer="temporal",
                attributes={"start": start, "end": end, "duration": duration},
            )
        )

        entity_nodes: list[str] = []
        entity_edges: list[str] = []
        for entity in segment_state.get("entities", []) or []:
            raw_id = entity.get("id") if isinstance(entity, dict) else str(entity)
            entity_id = f"entity:{raw_id}"
            attrs = dict(entity) if isinstance(entity, dict) else {"label": raw_id}
            attrs.update({"start": start, "end": end})
            self._graph.add_node(GraphNode(id=entity_id, type="entity", layer="entity", attributes=attrs))
            entity_nodes.append(entity_id)

        for source, target in combinations(entity_nodes, 2):
            edge_id = f"{source}->{target}:co_occurrence"
            self._graph.add_edge(
                GraphEdge(
                    id=edge_id,
                    type="co_occurrence",
                    layer="entity",
                    source=source,
                    target=target,
                    weight=0.7,
                    attributes={"segment_id": segment_id},
                )
            )
            entity_edges.append(edge_id)

        semantic_node_id = f"semantic:{segment_id}"
        themes = list(segment_state.get("themes", []) or [])
        concepts = list(segment_state.get("concepts", []) or [])
        semantic_embedding = list(segment_state.get("semantic_embedding", []) or self._graph.embed(segment_id))
        self._graph.add_node(
            GraphNode(
                id=semantic_node_id,
                type="semantic_segment",
                layer="semantic",
                attributes={"start": start, "end": end, "themes": themes, "concepts": concepts, "embedding": semantic_embedding},
            )
        )

        for theme in themes:
            theme_id = f"theme:{theme}"
            self._graph.add_node(GraphNode(id=theme_id, type="theme", layer="semantic", attributes={"theme": theme, "start": start, "end": end}))
            self._graph.add_edge(
                GraphEdge(
                    id=f"{semantic_node_id}->{theme_id}:thematic_reinforcement",
                    type="thematic_reinforcement",
                    layer="semantic",
                    source=semantic_node_id,
                    target=theme_id,
                    weight=0.75,
                    attributes={"entity_context": entity_nodes},
                )
            )

        event_id = str(segment_state.get("event_id", segment_id))
        causal_node_id = f"event:{event_id}"
        action = str(segment_state.get("action", "event"))
        self._graph.add_node(
            GraphNode(
                id=causal_node_id,
                type="event",
                layer="causal",
                attributes={"start": start, "end": end, "action": action, "segment_id": segment_id},
            )
        )

        causal_edges: list[str] = []
        prior_events = [f"event:{self.segment_store[s].get('segment_state', {}).get('event_id', s)}" for s in list(self.segment_store.keys())[:-1]]
        for prior in prior_events[-2:]:
            relation = "conflicts" if "conflict" in action else "causes"
            edge_id = f"{prior}->{causal_node_id}:{relation}"
            self._graph.add_edge(
                GraphEdge(
                    id=edge_id,
                    type=relation,
                    layer="causal",
                    source=prior,
                    target=causal_node_id,
                    weight=0.8,
                    attributes={"temporal_delta": start},
                )
            )
            causal_edges.append(edge_id)

        self.active_segments.append(segment_id)
        self._refresh_unresolved_threads()
        self._compress_if_needed()

        avg_duration = sum(max(0.0, float(self.segment_store[s]["segment_state"].get("timestamp_end", 0.0) - self.segment_store[s]["segment_state"].get("timestamp_start", 0.0))) for s in self.segment_store) / max(1, len(self.segment_store))

        return GraphState(
            temporal_node_ids=[f"segment:{sid}" for sid in self.segment_store.keys()],
            entity_node_ids=entity_nodes,
            semantic_node_ids=[semantic_node_id] + [f"theme:{theme}" for theme in themes],
            causal_node_ids=[causal_node_id],
            entity_co_occurrence_edges=entity_edges,
            causal_edge_ids=causal_edges,
            unresolved_thread_ids=sorted(self.unresolved_threads),
            avg_segment_duration=avg_duration,
            metadata={"active_window": list(self.active_segments), "summary_count": len(self.long_term_summaries)},
        )

    def _refresh_unresolved_threads(self) -> None:
        unresolved = self._graph.query("unresolved_threads", limit=200)
        self.unresolved_threads = {str(item.get("id", "")) for item in unresolved if item.get("id")}

    def _compress_if_needed(self) -> None:
        if len(self.segment_store) <= self.active_window_size * 2:
            return
        stale = sorted(self.segment_store.keys())[:-self.active_window_size]
        if not stale:
            return
        dominant = self._graph.query("dominant_theme:0.0:10000.0", limit=3)
        summary = {
            "segments": stale,
            "dominant_theme": [str(item.get("theme", "")) for item in dominant],
            "unresolved_threads": sorted(self.unresolved_threads),
        }
        self.long_term_summaries.append(summary)
        for seg in stale:
            self.segment_store.pop(seg, None)

    def retrieve_by_time(self, timestamp: float) -> RetrievalResult:
        nodes = self._graph.query(f"active_entities_at:{timestamp}", limit=50)
        node_ids = sorted(str(node.get("id")) for node in nodes if node.get("id"))
        return RetrievalResult(query=f"active_entities:{timestamp}", node_ids=node_ids, edge_ids=[], confidence=0.9 if node_ids else 0.2)

    def retrieve_by_entity(self, entity_id: str) -> RetrievalResult:
        node_id = f"entity:{entity_id}" if not entity_id.startswith("entity:") else entity_id
        graph = self._graph.retrieve_subgraph([node_id], depth=3)
        node_ids = sorted(str(node.get("id")) for node in graph.get("nodes", []) if node.get("id"))
        edge_ids = sorted(str(edge.get("id")) for edge in graph.get("edges", []) if edge.get("id"))
        return RetrievalResult(query=f"entity:{node_id}", node_ids=node_ids, edge_ids=edge_ids, confidence=0.8 if edge_ids else 0.3)

    def retrieve_by_semantic_query(self, start: float, end: float) -> RetrievalResult:
        themes = self._graph.query(f"dominant_theme:{start}:{end}", limit=1)
        node_ids = [f"theme:{themes[0]['theme']}"] if themes else []
        return RetrievalResult(query=f"dominant_theme:{start}-{end}", node_ids=node_ids, edge_ids=[], confidence=0.85 if node_ids else 0.2)

    def retrieve_by_causal_chain(self, event_id: str) -> RetrievalResult:
        node_id = f"event:{event_id}" if not event_id.startswith("event:") else event_id
        graph = self._graph.retrieve_subgraph([node_id], depth=4)
        nodes = sorted(str(node.get("id")) for node in graph.get("nodes", []) if str(node.get("id", "")).startswith("event:"))
        edges = sorted(str(edge.get("id")) for edge in graph.get("edges", []) if str(edge.get("id", "")).endswith(":causes") or str(edge.get("id", "")).endswith(":conflicts"))
        if node_id not in nodes:
            nodes.append(node_id)
        return RetrievalResult(query=f"causal_chain:{node_id}", node_ids=nodes, edge_ids=edges, confidence=0.8 if edges else 0.35)

    def snapshot(self) -> MemorySnapshot:
        salient_semantic = [f"semantic:{seg}" for seg in self.segment_store.keys()]
        salient_entities = [f"entity:{entity}" for entity in sorted({str(item).replace('entity:', '') for seg in self.segment_store.values() for item in seg.get('segment_state', {}).get('entities', []) if not isinstance(item, dict)})]
        return MemorySnapshot(
            short_term_segments=list(self.active_segments),
            long_term_summaries=list(self.long_term_summaries),
            unresolved_threads=sorted(self.unresolved_threads),
            salient_nodes=sorted(set(salient_semantic + salient_entities)),
        )
