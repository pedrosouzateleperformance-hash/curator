from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.types import DecisionCandidate, GraphState, NarrativeState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class MurchWeights:
    w_e: float = 0.35
    w_s: float = 0.25
    w_r: float = 0.15
    w_et: float = 0.10
    w_2d: float = 0.10
    w_3d: float = 0.05


class MurchValidator:
    """Rule-based scoring aligned with Walter Murch's cut criteria."""

    def __init__(self, weights: MurchWeights | None = None):
        self.weights = weights or MurchWeights()

    def components(
        self, candidate: DecisionCandidate, graph_state: GraphState, narrative_state: NarrativeState
    ) -> Dict[str, float]:
        ctx = candidate.graph_context
        emotion_alignment = _clip01(0.5 * narrative_state.emotional_intensity + 0.5 * ctx.get("emotion_match", 0.0))
        story_progression = _clip01(0.6 * narrative_state.progression + 0.4 * ctx.get("story_gain", 0.0))
        rhythm_consistency = _clip01(0.7 * graph_state.temporal.get("rhythm_consistency", 0.0) + 0.3 * ctx.get("beat_sync", 0.0))
        eye_trace = _clip01(ctx.get("eye_trace_continuity", 0.0))
        continuity_2d = _clip01(ctx.get("spatial_2d", 0.0))
        continuity_3d = _clip01(ctx.get("spatial_3d", 0.0))
        return {
            "E": emotion_alignment,
            "S": story_progression,
            "R": rhythm_consistency,
            "ET": eye_trace,
            "C2D": continuity_2d,
            "C3D": continuity_3d,
        }

    def score(self, candidate: DecisionCandidate, graph_state: GraphState, narrative_state: NarrativeState) -> float:
        c = self.components(candidate, graph_state, narrative_state)
        w = self.weights
        return (
            w.w_e * c["E"]
            + w.w_s * c["S"]
            + w.w_r * c["R"]
            + w.w_et * c["ET"]
            + w.w_2d * c["C2D"]
            + w.w_3d * c["C3D"]
        )
