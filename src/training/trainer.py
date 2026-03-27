from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from src.policy.policy_model import PolicyModel
from src.reward.reward_function import RewardFunction, RewardWeights
from src.training.offline_rl import OfflineRLTrainer
from src.types import Phase4Input


@dataclass
class StyleProfile:
    style_name: str = "default"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "alpha": 0.25,
        "beta": 0.20,
        "gamma": 0.15,
        "delta": 0.15,
        "epsilon": 0.15,
        "zeta": 0.10,
    })


@dataclass
class TrainerConfig:
    epochs: int = 5
    dynamic_weight_adaptation: bool = True


class Phase5Trainer:
    def __init__(
        self,
        policy: PolicyModel,
        reward_function: RewardFunction,
        config: TrainerConfig | None = None,
        offline_trainer: OfflineRLTrainer | None = None,
    ):
        self.policy = policy
        self.reward_function = reward_function
        self.config = config or TrainerConfig()
        self.offline_trainer = offline_trainer or OfflineRLTrainer(policy, reward_function)

    def adapt_weights(self, mean_components: Dict[str, float]) -> None:
        if not self.config.dynamic_weight_adaptation:
            return
        # Simple stabilization: slightly increase underperforming objectives.
        rw = self.reward_function.weights
        comp_to_weight = {
            "R_murch": "alpha",
            "R_continuity": "beta",
            "R_audio": "gamma",
            "R_gestalt": "delta",
            "R_narrative": "epsilon",
            "R_engagement": "zeta",
        }
        updated = rw.__dict__.copy()
        for comp, key in comp_to_weight.items():
            v = mean_components.get(comp, 0.5)
            if v < 0.5:
                updated[key] *= 1.02
            else:
                updated[key] *= 0.99
        total = sum(updated.values())
        normalized = {k: v / total for k, v in updated.items()}
        self.reward_function.weights = RewardWeights(**normalized)

    def train(self, dataset: Sequence[Phase4Input]) -> tuple[PolicyModel, dict[str, List[float]], StyleProfile]:
        logs = {"mean_reward": [], "loss_proxy": [], "grad_norm": []}
        for _ in range(self.config.epochs):
            component_sum = {
                "R_murch": 0.0,
                "R_continuity": 0.0,
                "R_audio": 0.0,
                "R_gestalt": 0.0,
                "R_narrative": 0.0,
                "R_engagement": 0.0,
            }
            component_count = 0
            for episode in dataset:
                metrics = self.offline_trainer.train_step(episode)
                logs["mean_reward"].append(metrics["mean_reward"])
                logs["loss_proxy"].append(metrics["loss_proxy"])
                logs["grad_norm"].append(metrics["grad_norm"])

                scored = self.reward_function.score_phase4_input(episode)
                for row in scored:
                    for key in component_sum:
                        component_sum[key] += row[key]
                    component_count += 1

            means = {k: v / max(component_count, 1) for k, v in component_sum.items()}
            self.adapt_weights(means)

        style = StyleProfile(style_name="learned_phase5", weights=self.reward_function.weights.__dict__.copy())
        return self.policy, logs, style


def example_training_run() -> dict[str, object]:
    from src.policy.action_space import ACTION_SPACE
    from src.reward.reward_function import RewardFunction
    from src.types import DecisionCandidate, GraphState, NarrativeState, Phase4Input

    policy = PolicyModel()
    reward = RewardFunction()
    trainer = Phase5Trainer(policy=policy, reward_function=reward)

    graph = GraphState(
        semantic={"coherence": 0.8, "dialogue_clarity": 0.7, "scene_alignment": 0.8},
        temporal={"rhythm_consistency": 0.75, "audio_sync": 0.7, "temporal_smoothness": 0.85},
        causal={"causal_consistency": 0.8, "action_chain": 0.7, "conflict_progression": 0.65},
        entity={"entity_persistence": 0.9},
    )
    narrative = NarrativeState(progression=0.7, tension=0.6, emotional_intensity=0.8, coherence=0.85)
    candidates = []
    for i, a in enumerate(ACTION_SPACE):
        candidates.append(
            DecisionCandidate(
                candidate_id=f"c{i}",
                action_type=a,
                timestamp=float(i),
                graph_context={
                    "emotion_match": 0.5 + i * 0.05,
                    "story_gain": 0.4 + i * 0.04,
                    "beat_sync": 0.6,
                    "eye_trace_continuity": 0.7,
                    "spatial_2d": 0.7,
                    "spatial_3d": 0.6,
                    "action_continuity": 0.7,
                    "spatial_consistency": 0.8,
                    "temporal_smoothness": 0.8,
                    "av_sync": 0.75,
                    "dialogue_clarity": 0.7,
                    "audio_emotion_match": 0.8,
                    "frame_balance": 0.6,
                    "saliency_distribution": 0.7,
                    "symmetry_score": 0.5,
                    "visual_tension": 0.6,
                },
                reasoning_path=["phase4", "candidate"],
            )
        )
    dataset = [Phase4Input(graph_state=graph, narrative_state=narrative, decision_candidates=candidates)]
    trained_policy, logs, style = trainer.train(dataset)
    return {"policy": trained_policy, "logs": logs, "style": style}
