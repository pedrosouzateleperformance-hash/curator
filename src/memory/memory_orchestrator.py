from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from src.graphs.causal_graph import CausalGraph
from src.graphs.entity_graph import EntityGraph
from src.graphs.semantic_graph import SemanticGraph
from src.graphs.temporal_graph import TemporalGraph
from src.memory.retrieval_engine import RetrievalEngine, RetrievalResult


@dataclass
class MemorySnapshot:
    short_term_segments: List[str]
    long_term_summaries: List[Dict[str, Any]]
    unresolved_threads: List[str]
    salient_nodes: List[str]


@dataclass
class GraphState:
    temporal_graph: TemporalGraph
    entity_graph: EntityGraph
    semantic_graph: SemanticGraph
    causal_graph: CausalGraph
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryOrchestrator:
    def __init__(self, active_window_size: int = 12):
        self.temporal_graph = TemporalGraph()
        self.entity_graph = EntityGraph()
        self.semantic_graph = SemanticGraph()
        self.causal_graph = CausalGraph()
        self.retrieval_engine = RetrievalEngine(
            temporal_graph=self.temporal_graph,
            entity_graph=self.entity_graph,
            semantic_graph=self.semantic_graph,
            causal_graph=self.causal_graph,
        )

        self.active_window_size = active_window_size
        self.active_segments: Deque[str] = deque(maxlen=active_window_size)
        self.segment_store: Dict[str, Any] = {}
        self.long_term_summaries: List[Dict[str, Any]] = []
        self.unresolved_threads: set[str] = set()

    def update(self, sequence_state: Any, segment_state: Any, fusion_trace: Any, memory_trace: Any) -> GraphState:
        segment_id = str(getattr(segment_state, "segment_id", None) or segment_state.get("segment_id") or segment_state.get("id"))
        self.segment_store[segment_id] = {
            "sequence_state": sequence_state,
            "segment_state": segment_state,
            "fusion_trace": fusion_trace,
            "memory_trace": memory_trace,
        }

        temporal_result = self.temporal_graph.update_from_segment(segment_state)
        entity_result = self.entity_graph.update_from_segment(segment_state)
        semantic_result = self.semantic_graph.update_from_segment(segment_state, entity_result.active_entities)
        causal_result = self.causal_graph.update_from_segment(segment_state)

        self.active_segments.append(segment_id)
        self._refresh_unresolved_threads()
        self._compress_if_needed()

        return GraphState(
            temporal_graph=self.temporal_graph,
            entity_graph=self.entity_graph,
            semantic_graph=self.semantic_graph,
            causal_graph=self.causal_graph,
            metadata={
                "last_update": {
                    "temporal": temporal_result,
                    "entity": entity_result,
                    "semantic": semantic_result,
                    "causal": causal_result,
                },
                "active_window": list(self.active_segments),
                "summary_count": len(self.long_term_summaries),
            },
        )

    def _refresh_unresolved_threads(self) -> None:
        unresolved = self.retrieval_engine.unresolved_threads().node_ids
        self.unresolved_threads = set(unresolved)

    def _compress_if_needed(self) -> None:
        if len(self.segment_store) <= self.active_window_size * 2:
            return
        stale = sorted(self.segment_store.keys())[:-self.active_window_size]
        if not stale:
            return
        summary = {
            "segments": stale,
            "dominant_theme": self.retrieval_engine.dominant_theme(0.0, 10_000.0).node_ids,
            "reappearing_entities": self.entity_graph.reappearing_entities(),
            "unresolved_threads": sorted(self.unresolved_threads),
        }
        self.long_term_summaries.append(summary)
        for seg in stale:
            self.segment_store.pop(seg, None)

    def retrieve_by_time(self, timestamp: float) -> RetrievalResult:
        return self.retrieval_engine.active_entities(timestamp)

    def retrieve_by_entity(self, entity_id: str) -> RetrievalResult:
        node_id = f"entity:{entity_id}" if not entity_id.startswith("entity:") else entity_id
        return self.retrieval_engine.weighted_path_search(
            graph_name="entity",
            start_node=node_id,
            relation_filter={"co_occurrence", "interaction", "transformation", "reappearance"},
            max_hops=3,
        )

    def retrieve_by_semantic_query(self, start: float, end: float) -> RetrievalResult:
        return self.retrieval_engine.dominant_theme(start, end)

    def retrieve_by_causal_chain(self, event_id: str) -> RetrievalResult:
        node_id = f"event:{event_id}" if not event_id.startswith("event:") else event_id
        return self.retrieval_engine.what_led_to_event(node_id)

    def snapshot(self) -> MemorySnapshot:
        salient_nodes = sorted(
            [node_id for node_id, node in self.semantic_graph.nodes.items() if node.salience >= 0.65]
            + [node_id for node_id, node in self.entity_graph.nodes.items() if node.salience >= 0.65]
        )
        return MemorySnapshot(
            short_term_segments=list(self.active_segments),
            long_term_summaries=list(self.long_term_summaries),
            unresolved_threads=sorted(self.unresolved_threads),
            salient_nodes=salient_nodes,
        )
