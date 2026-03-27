from __future__ import annotations

from dataclasses import dataclass

from src.types import GraphState, NarrativeState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class NarrativeWeights:
    coherence_weight: float = 0.30
    causal_weight: float = 0.30
    entity_weight: float = 0.20
    conflict_weight: float = 0.20


class NarrativeReward:
    def __init__(self, weights: NarrativeWeights | None = None):
        self.weights = weights or NarrativeWeights()

    def components(self, graph_state: GraphState, narrative_state: NarrativeState) -> dict[str, float]:
        coherence_score = _clip01(0.5 * narrative_state.coherence + 0.5 * graph_state.semantic.get("coherence", 0.0))
        causal_consistency = _clip01(graph_state.causal.get("causal_consistency", 0.0))
        entity_persistence = _clip01(graph_state.entity.get("entity_persistence", 0.0))
        conflict_progression = _clip01(0.6 * narrative_state.tension + 0.4 * graph_state.causal.get("conflict_progression", 0.0))
        return {
            "coherence_score": coherence_score,
            "causal_consistency": causal_consistency,
            "entity_persistence": entity_persistence,
            "conflict_progression": conflict_progression,
        }

    def score(self, graph_state: GraphState, narrative_state: NarrativeState) -> float:
        c = self.components(graph_state, narrative_state)
        w = self.weights
        return (
            w.coherence_weight * c["coherence_score"]
            + w.causal_weight * c["causal_consistency"]
            + w.entity_weight * c["entity_persistence"]
            + w.conflict_weight * c["conflict_progression"]
        )
