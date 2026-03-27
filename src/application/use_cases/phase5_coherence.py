from __future__ import annotations

import math

from src.domain.core import CoherenceMetrics, SequenceState
from src.domain.invariants import validate_coherence_smoothness
from src.domain.system_state import ProbabilisticState, SystemState, TemporalState, TransitionResult


class Phase5CoherenceUseCase:
    def execute(self, state: SystemState, current: SequenceState) -> TransitionResult[CoherenceMetrics]:
        prev = state.probabilistic.embedding
        cur = current.latent_state
        length = min(len(prev), len(cur))
        if length == 0:
            metrics = CoherenceMetrics(segment_id=current.segment_id, kl_divergence=0.0, wasserstein_distance=0.0, cosine_distance=0.0)
            validate_coherence_smoothness(metrics)
            next_state = state.transition(
                temporal=TemporalState(
                    step=state.temporal.step + 1,
                    timestamp=max(state.temporal.timestamp, state.temporal.structured.end_time if state.temporal.structured else state.temporal.timestamp),
                    segment_id=current.segment_id,
                    structured=state.temporal.structured,
                    sequence=current,
                ),
                probabilistic=ProbabilisticState(
                    embedding=cur,
                    previous_embedding=prev,
                    coherence=metrics,
                ),
            )
            return TransitionResult(output=metrics, state=next_state)

        eps = 1e-6
        p = [max(eps, abs(v)) for v in prev[:length]]
        q = [max(eps, abs(v)) for v in cur[:length]]
        p_sum = sum(p)
        q_sum = sum(q)
        p = [v / p_sum for v in p]
        q = [v / q_sum for v in q]

        kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        wasserstein = sum(abs(pi - qi) for pi, qi in zip(p, q)) / length
        dot = sum(pi * qi for pi, qi in zip(p, q))
        p_norm = math.sqrt(sum(pi * pi for pi in p))
        q_norm = math.sqrt(sum(qi * qi for qi in q))
        cosine = 1.0 - (dot / max(eps, p_norm * q_norm))

        metrics = CoherenceMetrics(
            segment_id=current.segment_id,
            kl_divergence=kl,
            wasserstein_distance=wasserstein,
            cosine_distance=cosine,
        )
        validate_coherence_smoothness(metrics)
        next_state = state.transition(
            temporal=TemporalState(
                step=state.temporal.step + 1,
                timestamp=max(state.temporal.timestamp, state.temporal.structured.end_time if state.temporal.structured else state.temporal.timestamp),
                segment_id=current.segment_id,
                structured=state.temporal.structured,
                sequence=current,
            ),
            probabilistic=ProbabilisticState(
                embedding=cur,
                previous_embedding=prev,
                coherence=metrics,
            ),
        )
        return TransitionResult(output=metrics, state=next_state)
