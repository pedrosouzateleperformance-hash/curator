from __future__ import annotations

from src.domain.core import NarrativeState, SequenceState
from src.ports.graph_repository_port import GraphRepositoryPort
from src.ports.use_cases import ContextReasoningPort


class ContextReasoningAdapter(ContextReasoningPort):
    def __init__(self, graph_repository: GraphRepositoryPort) -> None:
        self._graph = graph_repository

    def reason(self, sequence: SequenceState) -> NarrativeState:
        lookup = self._graph.retrieve_subgraph([sequence.segment_id], depth=2)
        node_count = float(len(lookup.get("nodes", [])))
        edge_count = float(len(lookup.get("edges", [])))
        local = {"node_count": node_count, "edge_count": edge_count}
        global_context = {"transition_score": sequence.transition_score}
        contradiction_count = 1 if edge_count > (node_count + 5.0) else 0
        return NarrativeState(
            segment_id=sequence.segment_id,
            local_context=local,
            global_context=global_context,
            contradiction_count=contradiction_count,
        )
