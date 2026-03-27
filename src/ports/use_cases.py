from __future__ import annotations

from typing import Protocol

from src.domain.core import CoherenceMetrics, MultiscaleProfile, NarrativeState, RenderPlan, SequenceState, StructuredData


class TemporalEncodingPort(Protocol):
    def encode(self, structured: StructuredData) -> SequenceState: ...


class ContextReasoningPort(Protocol):
    def reason(self, sequence: SequenceState) -> NarrativeState: ...


class CoherencePort(Protocol):
    def evaluate(self, previous: SequenceState, current: SequenceState) -> CoherenceMetrics: ...


class MultiscalePort(Protocol):
    def profile(self, sequence: SequenceState, narrative: NarrativeState) -> MultiscaleProfile: ...


class DecisionExecutionPort(Protocol):
    def decide(self, narrative: NarrativeState, coherence: CoherenceMetrics, multiscale: MultiscaleProfile) -> RenderPlan: ...
