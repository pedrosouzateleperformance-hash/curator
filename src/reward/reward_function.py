from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.reward.engagement_model import EngagementModel
from src.reward.narrative_reward import NarrativeReward
from src.types import DecisionCandidate, GraphState, NarrativeState, Phase4Input
from src.validation.audio_validator import AudioValidator
from src.validation.continuity_validator import ContinuityValidator
from src.validation.gestalt_validator import GestaltValidator
from src.validation.murch_validator import MurchValidator


@dataclass(frozen=True)
class RewardWeights:
    alpha: float = 0.25
    beta: float = 0.20
    gamma: float = 0.15
    delta: float = 0.15
    epsilon: float = 0.15
    zeta: float = 0.10


class RewardFunction:
    def __init__(
        self,
        weights: RewardWeights | None = None,
        murch_validator: MurchValidator | None = None,
        continuity_validator: ContinuityValidator | None = None,
        audio_validator: AudioValidator | None = None,
        gestalt_validator: GestaltValidator | None = None,
        narrative_reward: NarrativeReward | None = None,
        engagement_model: EngagementModel | None = None,
    ):
        self.weights = weights or RewardWeights()
        self.murch = murch_validator or MurchValidator()
        self.continuity = continuity_validator or ContinuityValidator()
        self.audio = audio_validator or AudioValidator()
        self.gestalt = gestalt_validator or GestaltValidator()
        self.narrative = narrative_reward or NarrativeReward()
        self.engagement = engagement_model or EngagementModel()

    def score_candidate(
        self,
        candidate: DecisionCandidate,
        graph_state: GraphState,
        narrative_state: NarrativeState,
        candidate_pool: Iterable[DecisionCandidate],
    ) -> Dict[str, float]:
        r_murch = self.murch.score(candidate, graph_state, narrative_state)
        r_continuity = self.continuity.score(candidate, graph_state)
        r_audio = self.audio.score(candidate, graph_state, narrative_state)
        r_gestalt = self.gestalt.score(candidate)
        r_narrative = self.narrative.score(graph_state, narrative_state)

        pool = list(candidate_pool)
        r_engagement, engagement_components = self.engagement.score(
            [c.action_type.value for c in pool],
            [narrative_state for _ in pool],
            pool,
        )

        w = self.weights
        total = (
            w.alpha * r_murch
            + w.beta * r_continuity
            + w.gamma * r_audio
            + w.delta * r_gestalt
            + w.epsilon * r_narrative
            + w.zeta * r_engagement
        )

        return {
            "R_murch": r_murch,
            "R_continuity": r_continuity,
            "R_audio": r_audio,
            "R_gestalt": r_gestalt,
            "R_narrative": r_narrative,
            "R_engagement": r_engagement,
            **engagement_components,
            "R_total": total,
        }

    def score_phase4_input(self, batch: Phase4Input) -> List[Dict[str, float]]:
        scored: List[Dict[str, float]] = []
        for c in batch.decision_candidates:
            row = {"candidate_id": c.candidate_id, **self.score_candidate(c, batch.graph_state, batch.narrative_state, batch.decision_candidates)}
            scored.append(row)
        ranked = sorted(scored, key=lambda x: x["R_total"], reverse=True)
        rank_map = {r["candidate_id"]: i + 1 for i, r in enumerate(ranked)}
        for row in scored:
            row["rank"] = rank_map[row["candidate_id"]]
        return scored
