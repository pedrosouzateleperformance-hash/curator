from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Generic, Mapping, TypeVar

from src.domain.core import (
    CoherenceMetrics,
    MultiscaleProfile,
    NarrativeState,
    RenderPlan,
    SequenceState,
    StructuredData,
)
from src.domain.invariants import validate_system_transition

PhaseOutputT = TypeVar("PhaseOutputT")


@dataclass(frozen=True)
class ContextState:
    """Topos-level context state C_t."""

    segment_id: str = ""
    local_context: Mapping[str, float] = field(default_factory=dict)
    global_context: Mapping[str, float] = field(default_factory=dict)
    contradiction_count: int = 0
    narrative: NarrativeState | None = None


@dataclass(frozen=True)
class TemporalState:
    """Coalgebra-level temporal state T_t."""

    step: int = 0
    timestamp: float = 0.0
    segment_id: str = ""
    structured: StructuredData | None = None
    sequence: SequenceState | None = None


@dataclass(frozen=True)
class ProbabilisticState:
    """Information-geometry state P_t."""

    embedding: tuple[float, ...] = ()
    previous_embedding: tuple[float, ...] = ()
    coherence: CoherenceMetrics | None = None


@dataclass(frozen=True)
class MultiscaleState:
    """Renormalization-group multiscale state M_t."""

    frame_count: int = 0
    shot_count: int = 0
    scene_count: int = 0
    act_count: int = 0
    profile: MultiscaleProfile | None = None


@dataclass(frozen=True)
class SystemState:
    context: ContextState = field(default_factory=ContextState)
    temporal: TemporalState = field(default_factory=TemporalState)
    probabilistic: ProbabilisticState = field(default_factory=ProbabilisticState)
    multiscale: MultiscaleState = field(default_factory=MultiscaleState)
    render_plan: RenderPlan | None = None

    def transition(self, **changes: object) -> "SystemState":
        next_state = replace(self, **changes)
        validate_system_transition(self, next_state)
        return next_state


@dataclass(frozen=True)
class TransitionResult(Generic[PhaseOutputT]):
    """Coalgebra transition result, SystemState -> (PhaseOutput, SystemState)."""

    output: PhaseOutputT
    state: SystemState
