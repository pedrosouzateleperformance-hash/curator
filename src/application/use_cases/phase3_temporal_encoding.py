from __future__ import annotations

from src.domain.core import SequenceState, StructuredData
from src.domain.invariants import validate_temporal_continuity
from src.domain.system_state import ProbabilisticState, SystemState, TemporalState, TransitionResult
from src.ports.use_cases import TemporalEncodingPort


class Phase3TemporalEncodingUseCase:
    def __init__(self, encoder: TemporalEncodingPort) -> None:
        self._encoder = encoder

    def execute(self, state: SystemState, structured: StructuredData) -> TransitionResult[SequenceState]:
        encoded = self._encoder.encode(structured)
        if state.temporal.sequence is not None:
            validate_temporal_continuity(state.temporal.sequence, encoded)

        next_state = state.transition(
            temporal=TemporalState(
                step=state.temporal.step + 1,
                timestamp=float(structured.end_time),
                segment_id=encoded.segment_id,
                structured=structured,
                sequence=encoded,
            ),
            probabilistic=ProbabilisticState(
                embedding=encoded.latent_state,
                previous_embedding=state.probabilistic.embedding,
                coherence=state.probabilistic.coherence,
            ),
        )
        return TransitionResult(output=encoded, state=next_state)
