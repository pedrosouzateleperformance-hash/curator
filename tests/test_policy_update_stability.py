from src.policy.policy_model import PolicyModel
from src.reward.reward_function import RewardFunction
from src.training.offline_rl import OfflineRLTrainer
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input


def _episode():
    graph = GraphState(
        semantic={"coherence": 0.8, "dialogue_clarity": 0.7, "scene_alignment": 0.9},
        temporal={"rhythm_consistency": 0.8, "audio_sync": 0.9, "temporal_smoothness": 0.7},
        causal={"causal_consistency": 0.8, "action_chain": 0.8, "conflict_progression": 0.7},
        entity={"entity_persistence": 0.85},
    )
    narrative = NarrativeState(progression=0.75, tension=0.6, emotional_intensity=0.8, coherence=0.85)
    candidates = [
        DecisionCandidate(
            candidate_id=str(i),
            action_type=a,
            timestamp=float(i),
            graph_context={
                "emotion_match": 0.6,
                "story_gain": 0.5,
                "beat_sync": 0.8,
                "eye_trace_continuity": 0.8,
                "spatial_2d": 0.8,
                "spatial_3d": 0.7,
                "action_continuity": 0.8,
                "spatial_consistency": 0.9,
                "temporal_smoothness": 0.8,
                "av_sync": 0.9,
                "dialogue_clarity": 0.8,
                "audio_emotion_match": 0.9,
                "frame_balance": 0.7,
                "saliency_distribution": 0.8,
                "symmetry_score": 0.6,
                "visual_tension": 0.7,
            },
            reasoning_path=["x"],
        )
        for i, a in enumerate([ActionType.CUT, ActionType.HOLD, ActionType.TRANSITION])
    ]
    return Phase4Input(graph_state=graph, narrative_state=narrative, decision_candidates=candidates)


def test_policy_gradient_stable_norm():
    policy = PolicyModel()
    trainer = OfflineRLTrainer(policy, RewardFunction())
    result = trainer.train_step(_episode())
    assert result["grad_norm"] < 20
