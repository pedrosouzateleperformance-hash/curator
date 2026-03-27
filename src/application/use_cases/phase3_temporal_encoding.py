from __future__ import annotations

from src.domain.core import SequenceState, StructuredData
from src.domain.invariants import validate_temporal_continuity
from src.ports.use_cases import TemporalEncodingPort


class Phase3TemporalEncodingUseCase:
    def __init__(self, encoder: TemporalEncodingPort) -> None:
        self._encoder = encoder
        self._previous: SequenceState | None = None

    def execute(self, structured: StructuredData) -> SequenceState:
        encoded = self._encoder.encode(structured)
        if self._previous is not None:
            validate_temporal_continuity(self._previous, encoded)
        self._previous = encoded
        return encoded
