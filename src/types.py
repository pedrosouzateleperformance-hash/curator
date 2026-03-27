from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    CUT = "CUT"
    HOLD = "HOLD"
    TRANSITION = "TRANSITION"
    INSERT = "INSERT"
    EMPHASIZE_AUDIO = "EMPHASIZE_AUDIO"
    SHIFT_PACING = "SHIFT_PACING"


@dataclass(frozen=True)
class GraphState:
    semantic: Mapping[str, float] = field(default_factory=dict)
    temporal: Mapping[str, float] = field(default_factory=dict)
    causal: Mapping[str, float] = field(default_factory=dict)
    entity: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NarrativeState:
    progression: float = 0.0
    tension: float = 0.0
    emotional_intensity: float = 0.0
    coherence: float = 0.0


@dataclass(frozen=True)
class DecisionCandidate:
    candidate_id: str
    action_type: ActionType
    timestamp: float
    graph_context: Mapping[str, float]
    reasoning_path: list[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplanationTrace:
    trace_id: str
    steps: list[str]
    confidence: float = 0.0


@dataclass(frozen=True)
class Phase4Input:
    graph_state: GraphState
    narrative_state: NarrativeState
    decision_candidates: list[DecisionCandidate]
    explanation_trace: ExplanationTrace | None = None
