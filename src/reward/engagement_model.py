from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import pvariance
from typing import Sequence

from src.types import DecisionCandidate, NarrativeState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class EngagementWeights:
    pacing_weight: float = 0.35
    emotional_weight: float = 0.30
    novelty_weight: float = 0.35


class EngagementModel:
    """Proxy engagement reward from pacing entropy, emotional variance, and novelty."""

    def __init__(self, weights: EngagementWeights | None = None):
        self.weights = weights or EngagementWeights()

    @staticmethod
    def pacing_entropy(candidate_actions: Sequence[str]) -> float:
        if not candidate_actions:
            return 0.0
        counts = {}
        for action in candidate_actions:
            counts[action] = counts.get(action, 0) + 1
        total = len(candidate_actions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(max(p, 1e-9), 2)
        max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 1.0
        return _clip01(entropy / max(max_entropy, 1e-6))

    @staticmethod
    def emotional_variance(narrative_states: Sequence[NarrativeState]) -> float:
        if len(narrative_states) < 2:
            return 0.0
        values = [s.emotional_intensity for s in narrative_states]
        return _clip01(pvariance(values))

    @staticmethod
    def novelty_score(candidates: Sequence[DecisionCandidate]) -> float:
        if not candidates:
            return 0.0
        signature_set = {tuple(sorted(c.graph_context.items())) for c in candidates}
        return _clip01(len(signature_set) / max(len(candidates), 1))

    def score(
        self,
        candidate_actions: Sequence[str],
        narrative_states: Sequence[NarrativeState],
        candidates: Sequence[DecisionCandidate],
    ) -> tuple[float, dict[str, float]]:
        pacing = self.pacing_entropy(candidate_actions)
        emotion = self.emotional_variance(narrative_states)
        novelty = self.novelty_score(candidates)
        w = self.weights
        total = w.pacing_weight * pacing + w.emotional_weight * emotion + w.novelty_weight * novelty
        return total, {
            "pacing_entropy": pacing,
            "emotional_variance": emotion,
            "novelty_score": novelty,
        }
