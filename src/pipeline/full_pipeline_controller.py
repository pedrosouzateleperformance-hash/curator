from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.policy.action_space import ACTION_SPACE
from src.policy.policy_model import PolicyModel
from src.reward.reward_function import RewardFunction
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input


@dataclass(frozen=True)
class ControllerSelection:
    selected_candidate: DecisionCandidate
    selected_row: Dict[str, Any]
    ranking: List[Dict[str, Any]]


def score_phase5_and_select(
    phase4_inputs: List[Phase4Input],
    reward_function: RewardFunction,
    policy_model: PolicyModel,
) -> ControllerSelection:
    scored_rows, candidate_registry = _score_batches(phase4_inputs, reward_function, policy_model)
    if not scored_rows:
        return _fallback_selection()
    ranking = _stable_rank(scored_rows)
    selected_row = ranking[0]
    selected_candidate = candidate_registry[selected_row["candidate_id"]]
    return ControllerSelection(selected_candidate=selected_candidate, selected_row=selected_row, ranking=ranking)


def _score_batches(
    phase4_inputs: List[Phase4Input],
    reward_function: RewardFunction,
    policy_model: PolicyModel,
) -> Tuple[List[Dict[str, Any]], Dict[str, DecisionCandidate]]:
    scored_rows: List[Dict[str, Any]] = []
    registry: Dict[str, DecisionCandidate] = {}
    for batch in phase4_inputs:
        rows = reward_function.score_phase4_input(batch)
        rows = sorted(rows, key=lambda row: (-float(row["R_total"]), str(row["candidate_id"])))
        for row in rows:
            candidate = _candidate_by_id(batch, row["candidate_id"])
            registry[candidate.candidate_id] = candidate
            row["policy_score"] = _policy_score(policy_model, batch.graph_state, batch.narrative_state, candidate)
            row["combined_score"] = 0.7 * float(row["R_total"]) + 0.3 * float(row["policy_score"])
            scored_rows.append(row)
    return scored_rows, registry


def _policy_score(
    policy_model: PolicyModel,
    graph_state: GraphState,
    narrative_state: NarrativeState,
    candidate: DecisionCandidate,
) -> float:
    probs = policy_model.action_probs(graph_state, narrative_state, candidate)
    return float(probs[ACTION_SPACE.index(candidate.action_type)])


def _candidate_by_id(batch: Phase4Input, candidate_id: str) -> DecisionCandidate:
    for candidate in batch.decision_candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(f"Unknown candidate id: {candidate_id}")


def _stable_rank(scored_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        scored_rows,
        key=lambda row: (-float(row["combined_score"]), -float(row["R_total"]), str(row["candidate_id"])),
    )


def _fallback_selection() -> ControllerSelection:
    candidate = DecisionCandidate(
        candidate_id="fallback_hold",
        action_type=ActionType.HOLD,
        timestamp=0.0,
        graph_context={"story_gain": 0.0, "emotion_match": 0.0},
        reasoning_path=["fallback:no_phase5_candidates"],
        metadata={},
    )
    row = {"candidate_id": candidate.candidate_id, "R_total": 0.0, "policy_score": 0.0, "combined_score": 0.0}
    return ControllerSelection(selected_candidate=candidate, selected_row=row, ranking=[row])
