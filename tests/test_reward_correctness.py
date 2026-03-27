from src.reward.reward_function import RewardFunction
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input


def test_reward_components_and_ranking():
    graph = GraphState(
        semantic={"coherence": 0.8, "dialogue_clarity": 0.7, "scene_alignment": 0.9},
        temporal={"rhythm_consistency": 0.8, "audio_sync": 0.9, "temporal_smoothness": 0.7},
        causal={"causal_consistency": 0.8, "action_chain": 0.8, "conflict_progression": 0.7},
        entity={"entity_persistence": 0.85},
    )
    narrative = NarrativeState(progression=0.75, tension=0.6, emotional_intensity=0.8, coherence=0.85)

    good = DecisionCandidate(
        candidate_id="good",
        action_type=ActionType.CUT,
        timestamp=1.0,
        graph_context={
            "emotion_match": 0.9,
            "story_gain": 0.85,
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

    weak = DecisionCandidate(
        candidate_id="weak",
        action_type=ActionType.HOLD,
        timestamp=2.0,
        graph_context={k: 0.1 for k in good.graph_context.keys()},
        reasoning_path=["x"],
    )

    rf = RewardFunction()
    scored = rf.score_phase4_input(Phase4Input(graph, narrative, [good, weak]))
    by_id = {row["candidate_id"]: row for row in scored}

    assert by_id["good"]["R_total"] > by_id["weak"]["R_total"]
    assert by_id["good"]["rank"] == 1
