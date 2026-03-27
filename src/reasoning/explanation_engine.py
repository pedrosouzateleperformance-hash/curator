from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.memory.memory_orchestrator import GraphState
from src.reasoning.decision_engine import DecisionCandidate
from src.reasoning.narrative_tracker import NarrativeState


@dataclass
class ExplanationTrace:
    decision_id: str
    reasoning_path: List[str]
    contributing_nodes: List[str]
    graph_sources: List[str]
    confidence: float
    decision_driver: str


class ExplanationEngine:
    def explain(
        self,
        candidate: DecisionCandidate,
        graph_state: GraphState,
        narrative_state: NarrativeState,
    ) -> ExplanationTrace:
        sources: list[str] = []
        if any(node.startswith("event:") for node in candidate.supporting_nodes):
            sources.append("causal_graph")
        if any(node.startswith("entity:") for node in candidate.supporting_nodes):
            sources.append("entity_graph")
        if any(node.startswith("semantic:") or node.startswith("theme:") for node in candidate.supporting_nodes):
            sources.append("semantic_graph")
        if not sources:
            sources.append("temporal_graph")

        driver = "continuity"
        if narrative_state.emotional_state in {"rising_tension", "high_tension"}:
            driver = "emotion"
        elif narrative_state.pacing_state in {"fast", "slow"}:
            driver = "rhythm"
        elif narrative_state.unresolved_threads:
            driver = "story_progression"

        return ExplanationTrace(
            decision_id=candidate.decision_id,
            reasoning_path=list(candidate.justification_path),
            contributing_nodes=list(candidate.supporting_nodes),
            graph_sources=sources,
            confidence=candidate.confidence,
            decision_driver=driver,
        )

    def explain_many(
        self,
        candidates: List[DecisionCandidate],
        graph_state: GraphState,
        narrative_state: NarrativeState,
    ) -> List[ExplanationTrace]:
        return [self.explain(c, graph_state, narrative_state) for c in candidates]
