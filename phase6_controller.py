"""Phase 6 Inference-Time Decision Engine and Real-Time Editing Controller.

Graph-grounded, score-based, cost-aware controller for selecting edit actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


class Action(str, Enum):
    CUT = "CUT"
    HOLD = "HOLD"
    TRANSITION = "TRANSITION"
    ACCELERATE = "ACCELERATE"
    DEFER = "DEFER"
    EXPERIMENT = "EXPERIMENT"


RouteMode = Literal["local-only", "hybrid", "cloud-assisted"]
DecisionStatus = Literal["accept", "reject", "revise", "defer"]


@dataclass
class RewardVector:
    R_emotion: float
    R_story: float
    R_rhythm: float
    R_continuity: float
    R_audio_sync: float
    R_balance: float
    R_engagement: float
    R_efficiency: float
    R_surprise: float
    R_compression: float

    def weighted_sum(self, weights: Dict[str, float]) -> float:
        return sum(getattr(self, k) * v for k, v in weights.items())


@dataclass
class InfoMetrics:
    entropy_before: float
    entropy_after: float
    surprise: float
    entropy_excess: float
    KL_shift: float

    @property
    def information_gain(self) -> float:
        return self.entropy_before - self.entropy_after


@dataclass
class ScaleFlow:
    micro: float
    meso: float
    macro: float
    supra: float

    @property
    def scale_consistency(self) -> float:
        # Adjacent-scale agreement.
        pairs = [(self.micro, self.meso), (self.meso, self.macro), (self.macro, self.supra)]
        return sum(1.0 - abs(a - b) for a, b in pairs) / len(pairs)

    @property
    def flow_stability(self) -> float:
        return (self.meso + self.macro + self.scale_consistency) / 3.0

    @property
    def fixed_point_score(self) -> float:
        return (self.flow_stability + self.supra) / 2.0


@dataclass
class Constraints:
    hard_violations: List[str] = field(default_factory=list)
    soft_violations: List[str] = field(default_factory=list)
    rule_conflicts: List[str] = field(default_factory=list)

    @property
    def has_hard_violation(self) -> bool:
        return len(self.hard_violations) > 0


@dataclass
class ValidatorSignals:
    murch: float
    katz: float
    chion: float
    gestalt: float

    @property
    def agreement(self) -> float:
        return (self.murch + self.katz + self.chion + self.gestalt) / 4.0


@dataclass
class CandidateAction:
    action: Action
    reward_vector: RewardVector
    info_metrics: InfoMetrics
    scale_flow: ScaleFlow
    constraints: Constraints
    validators: ValidatorSignals
    graph_path_confidence: float
    latency_cost: float
    compute_cost: float
    uncertainty_cost: float
    hard_rule_penalty: float = 10.0
    explanation_nodes: List[str] = field(default_factory=list)
    explanation_edges: List[str] = field(default_factory=list)
    resolved_conflicts: List[str] = field(default_factory=list)
    dominating_validators: List[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "R_emotion": 0.10,
            "R_story": 0.15,
            "R_rhythm": 0.10,
            "R_continuity": 0.15,
            "R_audio_sync": 0.10,
            "R_balance": 0.08,
            "R_engagement": 0.12,
            "R_efficiency": 0.10,
            "R_surprise": 0.05,
            "R_compression": 0.05,
        }
    )
    alpha_ig: float = 0.4
    beta_useful_surprise: float = 0.2
    gamma_destructive_surprise: float = 0.2
    delta_entropy_excess: float = 0.2
    lambda_latency: float = 0.5
    lambda_compute: float = 0.4
    lambda_risk: float = 0.5
    lambda_violation: float = 1.0
    low_conf_threshold: float = 0.62
    high_impact_threshold: float = 0.75


class Phase6DecisionEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def _useful_surprise(self, surprise: float, continuity: float, story: float) -> float:
        # Surprise is rewarded only when legible and structurally supported.
        gating = (continuity + story) / 2.0
        return surprise * gating

    def _destructive_surprise(self, surprise: float, continuity: float, audio_sync: float) -> float:
        # Penalize novelty that breaks coherence.
        incoherence = 1.0 - ((continuity + audio_sync) / 2.0)
        return surprise * incoherence

    def _r_info(self, candidate: CandidateAction) -> float:
        rv = candidate.reward_vector
        info = candidate.info_metrics
        useful = self._useful_surprise(info.surprise, rv.R_continuity, rv.R_story)
        destructive = self._destructive_surprise(info.surprise, rv.R_continuity, rv.R_audio_sync)
        return (
            self.config.alpha_ig * info.information_gain
            + self.config.beta_useful_surprise * useful
            - self.config.gamma_destructive_surprise * destructive
            - self.config.delta_entropy_excess * info.entropy_excess
        )

    def _route(self, c: CandidateAction) -> Tuple[RouteMode, float]:
        # Confidence blends graph support + validator agreement + scale stability.
        confidence = (
            0.40 * c.graph_path_confidence
            + 0.35 * c.validators.agreement
            + 0.25 * c.scale_flow.flow_stability
        )
        high_impact = c.reward_vector.R_story >= self.config.high_impact_threshold

        if confidence >= 0.8 and c.compute_cost <= 0.4:
            return "local-only", confidence
        if confidence < self.config.low_conf_threshold and high_impact:
            return "cloud-assisted", confidence
        return "hybrid", confidence

    def expected_utility(self, c: CandidateAction) -> float:
        if c.constraints.has_hard_violation:
            return -self.config.lambda_violation * c.hard_rule_penalty

        weighted_reward = c.reward_vector.weighted_sum(self.config.reward_weights)
        info_reward = self._r_info(c)
        scale_reward = 0.5 * c.scale_flow.flow_stability + 0.5 * c.scale_flow.fixed_point_score
        validator_reward = c.validators.agreement
        soft_penalty = 0.1 * len(c.constraints.soft_violations) + 0.1 * len(c.constraints.rule_conflicts)

        return (
            weighted_reward
            + 0.2 * info_reward
            + 0.15 * scale_reward
            + 0.15 * validator_reward
            - self.config.lambda_latency * c.latency_cost
            - self.config.lambda_compute * c.compute_cost
            - self.config.lambda_risk * c.uncertainty_cost
            - soft_penalty
        )

    def decide(self, candidates: List[CandidateAction]) -> Dict[str, object]:
        if not candidates:
            raise ValueError("No candidate actions provided.")

        # 1) Hard constraints first.
        valid = [c for c in candidates if not c.constraints.has_hard_violation]
        if not valid:
            return {
                "action": Action.HOLD.value,
                "final_decision": {
                    "status": "defer",
                    "reason": "All candidates violate hard constraints; preserve continuity and optionality.",
                },
                "constraints": {
                    "hard_violations": [v for c in candidates for v in c.constraints.hard_violations],
                    "soft_violations": [],
                    "rule_conflicts": [],
                },
            }

        # 2-6) Evaluate utility.
        scored = [(c, self.expected_utility(c)) for c in valid]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_u = scored[0]

        # Tie-break: lower risk, higher explainability proxy, lower compute.
        if len(scored) > 1 and abs(scored[0][1] - scored[1][1]) < 0.03:
            top = [scored[0][0], scored[1][0]]
            top.sort(
                key=lambda c: (
                    c.uncertainty_cost,
                    -(len(c.explanation_nodes) + len(c.explanation_edges)),
                    c.compute_cost,
                )
            )
            best = top[0]
            best_u = self.expected_utility(best)

        route_mode, confidence = self._route(best)

        status: DecisionStatus = "accept"
        reason = "Highest expected utility under constraints."
        if confidence < self.config.low_conf_threshold:
            status = "defer" if best.action in {Action.CUT, Action.ACCELERATE, Action.EXPERIMENT} else "revise"
            reason = "Confidence below threshold; conservative action or re-evaluation preferred."

        return {
            "action": best.action.value,
            "reward_vector": best.reward_vector.__dict__,
            "info_metrics": {
                **best.info_metrics.__dict__,
                "information_gain": best.info_metrics.information_gain,
            },
            "scale_flow": {
                "micro": best.scale_flow.micro,
                "meso": best.scale_flow.meso,
                "macro": best.scale_flow.macro,
                "supra": best.scale_flow.supra,
                "scale_consistency": best.scale_flow.scale_consistency,
                "flow_stability": best.scale_flow.flow_stability,
                "fixed_point_score": best.scale_flow.fixed_point_score,
            },
            "constraints": {
                "hard_violations": best.constraints.hard_violations,
                "soft_violations": best.constraints.soft_violations,
                "rule_conflicts": best.constraints.rule_conflicts,
            },
            "routing": {
                "mode": route_mode,
                "latency_estimate": best.latency_cost,
                "compute_estimate": best.compute_cost,
                "confidence": confidence,
            },
            "explanation_path": {
                "activated_nodes": best.explanation_nodes,
                "supporting_edges": best.explanation_edges,
                "resolved_conflicts": best.resolved_conflicts,
                "dominating_validators": best.dominating_validators,
            },
            "final_decision": {
                "status": status,
                "reason": reason,
                "expected_utility": round(best_u, 4),
            },
        }
