from __future__ import annotations

from src.domain.core import NarrativeState, SequenceState
from src.domain.invariants import validate_context_composition
from src.domain.system_state import ContextState, SystemState, TemporalState, TransitionResult
from src.ports.use_cases import ContextReasoningPort


class Phase4ContextReasoningUseCase:
    def __init__(self, reasoner: ContextReasoningPort) -> None:
        self._reasoner = reasoner

    def execute(self, state: SystemState, sequence: SequenceState) -> TransitionResult[NarrativeState]:
        narrative = self._reasoner.reason(sequence)
        validate_context_composition(narrative)
        next_state = state.transition(
            context=ContextState(
                segment_id=narrative.segment_id,
                local_context=narrative.local_context,
                global_context=narrative.global_context,
                contradiction_count=narrative.contradiction_count,
                narrative=narrative,
            ),
            temporal=TemporalState(
                step=state.temporal.step + 1,
                timestamp=max(state.temporal.timestamp, (state.temporal.structured.end_time if state.temporal.structured else state.temporal.timestamp)),
                segment_id=sequence.segment_id,
                structured=state.temporal.structured,
                sequence=sequence,
            ),
        )
        return TransitionResult(output=narrative, state=next_state)
