from __future__ import annotations

from src.domain.core import NarrativeState, SequenceState
from src.domain.invariants import validate_context_composition
from src.ports.use_cases import ContextReasoningPort


class Phase4ContextReasoningUseCase:
    def __init__(self, reasoner: ContextReasoningPort) -> None:
        self._reasoner = reasoner

    def execute(self, sequence: SequenceState) -> NarrativeState:
        narrative = self._reasoner.reason(sequence)
        validate_context_composition(narrative)
        return narrative
