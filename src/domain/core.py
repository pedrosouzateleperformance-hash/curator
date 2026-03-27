from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ExecutionMode(str, Enum):
    CUT_ONLY = "cut_only"
    AUDIO_AUGMENTED = "audio_augmented"
    FULL_GENERATION = "full_generation"


@dataclass(frozen=True)
class StructuredData:
    segment_id: str
    start_time: float
    end_time: float
    entities: tuple[str, ...] = ()
    audio_events: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")


@dataclass(frozen=True)
class SequenceState:
    segment_id: str
    latent_state: tuple[float, ...]
    transition_score: float


@dataclass(frozen=True)
class NarrativeState:
    segment_id: str
    local_context: Mapping[str, float] = field(default_factory=dict)
    global_context: Mapping[str, float] = field(default_factory=dict)
    contradiction_count: int = 0


@dataclass(frozen=True)
class CoherenceMetrics:
    segment_id: str
    kl_divergence: float
    wasserstein_distance: float
    cosine_distance: float

    @property
    def smoothness(self) -> float:
        return -(self.kl_divergence + self.wasserstein_distance + self.cosine_distance) / 3.0


@dataclass(frozen=True)
class MultiscaleProfile:
    segment_id: str
    frame_level: Mapping[str, float]
    shot_level: Mapping[str, float]
    scene_level: Mapping[str, float]
    act_level: Mapping[str, float]
    regime_shift_score: float


@dataclass(frozen=True)
class RenderPlan:
    segment_id: str
    mode: ExecutionMode
    selected_actions: tuple[str, ...]
    scoring_breakdown: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemState:
    context: NarrativeState
    temporal: SequenceState
    coherence: CoherenceMetrics
    multiscale: MultiscaleProfile
