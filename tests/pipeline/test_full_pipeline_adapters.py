from src.pipeline.full_pipeline_adapters import _phase6_to_phase7
from src.pipeline.contracts import SegmentState
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input


def test_phase6_to_phase7_builds_sorted_candidates_and_segment():
    decision = DecisionCandidate("c1", ActionType.SHIFT_PACING, 0.0, {}, [], {})
    phase4_input = Phase4Input(
        graph_state=GraphState(semantic={"semantic_density": 0.2}, causal={"causal_consistency": 0.3}),
        narrative_state=NarrativeState(tension=0.4, emotional_intensity=0.4, coherence=0.8),
        decision_candidates=[decision],
    )
    segment = SegmentState("seg-1", 1.0, 2.0, [0.1], ["e1"], ["dialogue"], {"token_density": 0.5, "modality_span": 0.6})

    payload = _phase6_to_phase7(decision, phase4_input, segment)

    candidates = payload["decision_candidates"]
    costs = [item.cost_estimate for item in candidates]
    assert costs == sorted(costs)
    assert payload["segment_state"].segments[0].segment_id == "seg-1"
