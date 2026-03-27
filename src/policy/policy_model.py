from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence

from src.policy.action_space import ACTION_SPACE, ACTION_TO_INDEX
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState


def _clip(value: float, lo: float = -20.0, hi: float = 20.0) -> float:
    return max(lo, min(hi, value))


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(_clip(v - m)) for v in logits]
    denom = sum(exps) + 1e-12
    return [e / denom for e in exps]


@dataclass
class PolicyConfig:
    feature_dim: int = 10
    seed: int = 7


class PolicyModel:
    """Simple linear stochastic policy over editing actions."""

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()
        rnd = random.Random(self.config.seed)
        self.weights = [
            [rnd.uniform(-0.01, 0.01) for _ in range(self.config.feature_dim)]
            for _ in ACTION_SPACE
        ]

    def featurize(self, graph_state: GraphState, narrative_state: NarrativeState, candidate: DecisionCandidate) -> List[float]:
        ctx = candidate.graph_context
        return [
            narrative_state.progression,
            narrative_state.tension,
            narrative_state.emotional_intensity,
            narrative_state.coherence,
            graph_state.temporal.get("rhythm_consistency", 0.0),
            graph_state.causal.get("causal_consistency", 0.0),
            graph_state.entity.get("entity_persistence", 0.0),
            ctx.get("story_gain", 0.0),
            ctx.get("emotion_match", 0.0),
            1.0,
        ]

    def logits(self, features: Sequence[float]) -> List[float]:
        return [sum(w_i * x_i for w_i, x_i in zip(row, features)) for row in self.weights]

    def action_probs(self, graph_state: GraphState, narrative_state: NarrativeState, candidate: DecisionCandidate) -> List[float]:
        feats = self.featurize(graph_state, narrative_state, candidate)
        return _softmax(self.logits(feats))

    def sample_action(self, graph_state: GraphState, narrative_state: NarrativeState, candidate: DecisionCandidate) -> ActionType:
        probs = self.action_probs(graph_state, narrative_state, candidate)
        r = random.random()
        cdf = 0.0
        for idx, p in enumerate(probs):
            cdf += p
            if r <= cdf:
                return ACTION_SPACE[idx]
        return ACTION_SPACE[-1]

    def update_policy_gradient(
        self,
        features_batch: Sequence[Sequence[float]],
        actions: Sequence[ActionType],
        advantages: Sequence[float],
        lr: float,
        grad_clip: float = 1.0,
    ) -> float:
        if not features_batch:
            return 0.0
        total_norm = 0.0
        for features, action, adv in zip(features_batch, actions, advantages):
            probs = _softmax(self.logits(features))
            chosen_idx = ACTION_TO_INDEX[action]
            for k, _ in enumerate(ACTION_SPACE):
                coeff = (1.0 if k == chosen_idx else 0.0) - probs[k]
                for j, f_j in enumerate(features):
                    grad = adv * coeff * f_j
                    grad = max(-grad_clip, min(grad_clip, grad))
                    self.weights[k][j] += lr * grad
                    total_norm += grad * grad
        return math.sqrt(total_norm)
