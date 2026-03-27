from src.pipeline.full_pipeline_controller import score_phase5_and_select
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input


class FixedReward:
    def score_phase4_input(self, batch: Phase4Input):
        scores = []
        for candidate in batch.decision_candidates:
            value = 0.9 if candidate.candidate_id == "c1" else 0.9
            scores.append({"candidate_id": candidate.candidate_id, "R_total": value})
        return scores


class FixedPolicy:
    def action_probs(self, graph_state, narrative_state, candidate):
        if candidate.candidate_id == "c1":
            return [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
        return [0.3, 0.1, 0.2, 0.1, 0.1, 0.2]


def _batch() -> Phase4Input:
    c1 = DecisionCandidate("c1", ActionType.CUT, 0.0, {}, [], {})
    c2 = DecisionCandidate("c2", ActionType.CUT, 0.0, {}, [], {})
    return Phase4Input(GraphState(), NarrativeState(), [c1, c2])


def test_controller_uses_stable_tie_breaking_and_policy_score():
    selection = score_phase5_and_select([_batch()], FixedReward(), FixedPolicy())
    assert selection.selected_candidate.candidate_id == "c2"
    assert selection.ranking[0]["combined_score"] >= selection.ranking[1]["combined_score"]


def test_controller_fallback_when_no_batches():
    selection = score_phase5_and_select([], FixedReward(), FixedPolicy())
    assert selection.selected_candidate.candidate_id == "fallback_hold"
    assert selection.ranking[0]["candidate_id"] == "fallback_hold"
