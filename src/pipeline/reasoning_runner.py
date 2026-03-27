from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.adapters.graph.ruvector_graph_repository import RuVectorGraphRepository
from src.memory.memory_orchestrator import GraphState, MemoryOrchestrator
from src.ports.graph_repository_port import GraphRepositoryPort
from src.reasoning.decision_engine import DecisionCandidate, DecisionProposalEngine
from src.reasoning.explanation_engine import ExplanationEngine, ExplanationTrace
from src.reasoning.narrative_tracker import NarrativeState, NarrativeStateTracker


@dataclass
class ReasoningOutput:
    graph_state: GraphState
    narrative_state: NarrativeState
    decision_candidates: List[DecisionCandidate]
    explanation_traces: List[ExplanationTrace]


class ReasoningRunner:
    """Phase 4 runner for MAGMA-style graph memory and reasoning."""

    def __init__(self, graph_repository: GraphRepositoryPort | None = None, active_window_size: int = 12) -> None:
        repository = graph_repository or RuVectorGraphRepository()
        self.memory = MemoryOrchestrator(graph_repository=repository, active_window_size=active_window_size)
        self.narrative_tracker = NarrativeStateTracker(memory=self.memory)
        self.decision_engine = DecisionProposalEngine()
        self.explanation_engine = ExplanationEngine()

    def process(
        self,
        sequence_state: Any,
        segment_state: Any,
        fusion_trace: Any,
        memory_trace: Any,
    ) -> ReasoningOutput:
        graph_state = self.memory.update(sequence_state, segment_state, fusion_trace, memory_trace)

        timestamp = segment_state.get("timestamp_end", segment_state.get("end", 0.0))
        narrative_state = self.narrative_tracker.update(graph_state, float(timestamp))
        candidates = self.decision_engine.propose(graph_state, narrative_state)
        explanations = self.explanation_engine.explain_many(candidates, graph_state, narrative_state)

        return ReasoningOutput(
            graph_state=graph_state,
            narrative_state=narrative_state,
            decision_candidates=candidates,
            explanation_traces=explanations,
        )
