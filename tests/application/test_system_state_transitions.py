from __future__ import annotations

import pytest

from src.application.use_cases import (
    Phase3TemporalEncodingUseCase,
    Phase4ContextReasoningUseCase,
    Phase5CoherenceUseCase,
    Phase6MultiscaleUseCase,
    Phase7DecisionExecutionUseCase,
)
from src.domain.core import ExecutionMode, NarrativeState, SequenceState, StructuredData
from src.domain.system_state import SystemState


class _StubEncoder:
    def encode(self, structured: StructuredData) -> SequenceState:
        return SequenceState(segment_id=structured.segment_id, latent_state=(0.2, 0.3, 0.5), transition_score=0.1)


class _StubReasoner:
    def reason(self, sequence: SequenceState) -> NarrativeState:
        return NarrativeState(segment_id=sequence.segment_id, local_context={"a": 1.0}, global_context={"g": 1.0}, contradiction_count=0)


def test_phase_transitions_return_explicit_state() -> None:
    state = SystemState()
    structured = StructuredData(segment_id="scene_1", start_time=0.0, end_time=2.0, metadata={"shots": 2, "frames": 6})

    phase3 = Phase3TemporalEncodingUseCase(_StubEncoder())
    phase4 = Phase4ContextReasoningUseCase(_StubReasoner())
    phase5 = Phase5CoherenceUseCase()
    phase6 = Phase6MultiscaleUseCase()
    phase7 = Phase7DecisionExecutionUseCase()

    t3 = phase3.execute(state, structured)
    t4 = phase4.execute(t3.state, t3.output)
    t5 = phase5.execute(t4.state, t3.output)
    t6 = phase6.execute(t5.state, t3.output, t4.output)
    t7 = phase7.execute(t6.state, t4.output, t5.output, t6.output, ExecutionMode.CUT_ONLY)

    assert t7.output.segment_id == "scene_1"
    assert t7.state.temporal.step == 5
    assert t7.state.render_plan is not None


def test_no_implicit_reset_in_transition() -> None:
    state = SystemState()
    with pytest.raises(ValueError, match="Implicit reset detected"):
        state.transition(temporal=state.temporal.__class__(step=3, timestamp=1.0))
