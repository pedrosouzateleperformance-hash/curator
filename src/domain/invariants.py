from __future__ import annotations

from src.domain.core import CoherenceMetrics, MultiscaleProfile, NarrativeState, SequenceState


def validate_temporal_continuity(previous: SequenceState, current: SequenceState) -> None:
    if previous.segment_id == current.segment_id:
        return
    if current.transition_score < 0.0:
        raise ValueError("Coalgebra transition score must be non-negative")


def validate_context_composition(state: NarrativeState) -> None:
    if state.contradiction_count > 0:
        raise ValueError("Topos context composition failed due to contradictions")


def validate_coherence_smoothness(metrics: CoherenceMetrics, *, threshold: float = -1.0) -> None:
    if metrics.smoothness < threshold:
        raise ValueError("Information-geometry smoothness fell below threshold")


def validate_multiscale_stability(profile: MultiscaleProfile, *, max_shift: float = 1.0) -> None:
    if profile.regime_shift_score > max_shift:
        raise ValueError("Renormalization regime shift exceeded tolerance")


def validate_no_implicit_resets(previous_step: int, current_step: int) -> None:
    if current_step != previous_step + 1:
        raise ValueError("Implicit reset detected: temporal step must advance by exactly one")


def validate_timestamps_monotonic(previous_timestamp: float, current_timestamp: float) -> None:
    if current_timestamp < previous_timestamp:
        raise ValueError("Timestamp monotonicity violated")


def validate_probability_embedding_shape_consistency(previous: tuple[float, ...], current: tuple[float, ...]) -> None:
    if previous and current and len(previous) != len(current):
        raise ValueError("Probability embedding shape inconsistency detected")


def validate_multiscale_hierarchy_monotonicity(frame_count: int, shot_count: int, scene_count: int, act_count: int) -> None:
    if not (frame_count >= shot_count >= scene_count >= act_count >= 0):
        raise ValueError("Multiscale hierarchy must be monotonic (frame->shot->scene->act)")


def validate_system_transition(previous: object, current: object) -> None:
    from src.domain.system_state import SystemState

    if not isinstance(previous, SystemState) or not isinstance(current, SystemState):
        raise TypeError("validate_system_transition expects SystemState values")

    validate_no_implicit_resets(previous.temporal.step, current.temporal.step)
    validate_timestamps_monotonic(previous.temporal.timestamp, current.temporal.timestamp)
    validate_probability_embedding_shape_consistency(previous.probabilistic.embedding, current.probabilistic.embedding)
    validate_multiscale_hierarchy_monotonicity(
        current.multiscale.frame_count,
        current.multiscale.shot_count,
        current.multiscale.scene_count,
        current.multiscale.act_count,
    )
