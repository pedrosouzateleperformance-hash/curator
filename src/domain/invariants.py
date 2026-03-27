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
