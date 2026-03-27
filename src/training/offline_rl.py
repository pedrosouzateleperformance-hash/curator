from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from src.policy.policy_model import PolicyModel
from src.reward.reward_function import RewardFunction
from src.types import ActionType, Phase4Input


@dataclass
class OfflineRLConfig:
    algorithm: str = "awr"
    learning_rate: float = 0.01
    batch_size: int = 8


class OfflineRLTrainer:
    def __init__(self, policy: PolicyModel, reward_function: RewardFunction, config: OfflineRLConfig | None = None):
        self.policy = policy
        self.reward_fn = reward_function
        self.config = config or OfflineRLConfig()

    def compute_advantages(self, totals: Sequence[float]) -> List[float]:
        baseline = sum(totals) / max(len(totals), 1)
        return [t - baseline for t in totals]

    def train_step(self, episode: Phase4Input) -> dict[str, float]:
        scored = self.reward_fn.score_phase4_input(episode)
        by_id = {row["candidate_id"]: row for row in scored}

        features_batch, actions, totals = [], [], []
        for candidate in episode.decision_candidates:
            features_batch.append(self.policy.featurize(episode.graph_state, episode.narrative_state, candidate))
            actions.append(candidate.action_type)
            totals.append(by_id[candidate.candidate_id]["R_total"])

        advantages = self.compute_advantages(totals)
        grad_norm = self.policy.update_policy_gradient(
            features_batch=features_batch,
            actions=actions,
            advantages=advantages,
            lr=self.config.learning_rate,
        )
        return {
            "loss_proxy": -sum(advantages) / max(len(advantages), 1),
            "mean_reward": sum(totals) / max(len(totals), 1),
            "grad_norm": grad_norm,
        }
