from __future__ import annotations

from dataclasses import dataclass

from src.types import DecisionCandidate


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class GestaltWeights:
    w_balance: float = 0.30
    w_saliency: float = 0.30
    w_symmetry: float = 0.20
    w_tension: float = 0.20


class GestaltValidator:
    def __init__(self, weights: GestaltWeights | None = None):
        self.weights = weights or GestaltWeights()

    def components(self, candidate: DecisionCandidate) -> dict[str, float]:
        ctx = candidate.graph_context
        return {
            "frame_balance": _clip01(ctx.get("frame_balance", 0.0)),
            "saliency_distribution": _clip01(ctx.get("saliency_distribution", 0.0)),
            "symmetry_score": _clip01(ctx.get("symmetry_score", 0.0)),
            "visual_tension": _clip01(ctx.get("visual_tension", 0.0)),
        }

    def score(self, candidate: DecisionCandidate) -> float:
        c = self.components(candidate)
        w = self.weights
        return (
            w.w_balance * c["frame_balance"]
            + w.w_saliency * c["saliency_distribution"]
            + w.w_symmetry * c["symmetry_score"]
            + w.w_tension * c["visual_tension"]
        )
