from __future__ import annotations

from dataclasses import dataclass
import math

from src.policy.action_space import ACTION_TO_INDEX
from src.policy.policy_model import PolicyModel
from src.types import DecisionCandidate, GraphState, NarrativeState


@dataclass
class PreferenceLearningConfig:
    learning_rate: float = 0.005


class PreferenceLearner:
    """RLAIF/RLHF-style pairwise preference objective over candidate actions."""

    def __init__(self, policy: PolicyModel, config: PreferenceLearningConfig | None = None):
        self.policy = policy
        self.config = config or PreferenceLearningConfig()

    def preference_loss(
        self,
        graph_state: GraphState,
        narrative_state: NarrativeState,
        preferred: DecisionCandidate,
        rejected: DecisionCandidate,
    ) -> float:
        p_pref = self.policy.action_probs(graph_state, narrative_state, preferred)[ACTION_TO_INDEX[preferred.action_type]]
        p_rej = self.policy.action_probs(graph_state, narrative_state, rejected)[ACTION_TO_INDEX[rejected.action_type]]
        return -(math.log(max(p_pref, 1e-9)) - math.log(max(p_rej, 1e-9)))

    def update(
        self,
        graph_state: GraphState,
        narrative_state: NarrativeState,
        preferred: DecisionCandidate,
        rejected: DecisionCandidate,
    ) -> float:
        pref_feats = self.policy.featurize(graph_state, narrative_state, preferred)
        rej_feats = self.policy.featurize(graph_state, narrative_state, rejected)
        actions = [preferred.action_type, rejected.action_type]
        advantages = [1.0, -1.0]
        self.policy.update_policy_gradient([pref_feats, rej_feats], actions, advantages, lr=self.config.learning_rate)
        return self.preference_loss(graph_state, narrative_state, preferred, rejected)
