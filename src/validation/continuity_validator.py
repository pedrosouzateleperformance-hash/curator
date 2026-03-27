from __future__ import annotations

from dataclasses import dataclass

from src.types import DecisionCandidate, GraphState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ContinuityWeights:
    w_action: float = 0.40
    w_spatial: float = 0.35
    w_temporal: float = 0.25


class ContinuityValidator:
    def __init__(self, weights: ContinuityWeights | None = None):
        self.weights = weights or ContinuityWeights()

    def components(self, candidate: DecisionCandidate, graph_state: GraphState) -> dict[str, float]:
        ctx = candidate.graph_context
        return {
            "action_continuity": _clip01(ctx.get("action_continuity", graph_state.causal.get("action_chain", 0.0))),
            "spatial_consistency": _clip01(ctx.get("spatial_consistency", graph_state.semantic.get("scene_alignment", 0.0))),
            "temporal_smoothness": _clip01(ctx.get("temporal_smoothness", graph_state.temporal.get("temporal_smoothness", 0.0))),
        }

    def score(self, candidate: DecisionCandidate, graph_state: GraphState) -> float:
        c = self.components(candidate, graph_state)
        w = self.weights
        return (
            w.w_action * c["action_continuity"]
            + w.w_spatial * c["spatial_consistency"]
            + w.w_temporal * c["temporal_smoothness"]
        )
