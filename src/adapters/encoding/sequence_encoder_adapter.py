from __future__ import annotations

from src.domain.core import StructuredData
from src.domain.core import SequenceState as DomainSequenceState
from src.ports.use_cases import TemporalEncodingPort


class SequenceEncoderAdapter(TemporalEncodingPort):
    """Small deterministic encoder for the clean-architecture flow."""

    def encode(self, structured: StructuredData) -> DomainSequenceState:
        span = max(1e-6, structured.end_time - structured.start_time)
        entity_factor = float(len(structured.entities)) / (len(structured.entities) + 1.0)
        audio_factor = float(len(structured.audio_events)) / (len(structured.audio_events) + 1.0)
        latent = (
            float(structured.start_time),
            float(structured.end_time),
            entity_factor,
            audio_factor,
            span,
        )
        return DomainSequenceState(segment_id=structured.segment_id, latent_state=latent, transition_score=span)
