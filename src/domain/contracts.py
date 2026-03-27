from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShotNode:
    shot_id: str
    start_time: float
    end_time: float
    scene_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class SceneNode:
    scene_id: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameFeature:
    frame_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    features: list[float]
    object_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(frozen=True)
class AudioEventNode:
    event_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    event_type: str
    features: list[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class UtteranceNode:
    utterance_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    speaker_id: str
    text_embedding: list[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class EntityNode:
    entity_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    role: str
    features: list[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class AlignmentEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    timestamp: float | None = None


@dataclass
class AlignmentGraph:
    edges: list[AlignmentEdge]
    node_context: dict[str, dict[str, float]] = field(default_factory=dict)

    def neighbors(self, node_id: str) -> list[AlignmentEdge]:
        return [edge for edge in self.edges if edge.source_id == node_id or edge.target_id == node_id]

    def context_for(self, node_id: str) -> dict[str, float]:
        return self.node_context.get(node_id, {})


@dataclass(frozen=True)
class SequenceToken:
    token_id: str
    timestamp: float
    modality: str
    payload: list[float]
    source_ref: str
    entity_links: list[str]
    shot_id: str
    scene_id: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedSequenceState:
    token_id: str
    timestamp: float
    modality: str
    hidden_state: list[float]
    memory_state: dict[str, Any]
    confidence: float


@dataclass
class SegmentState:
    segment_id: str
    start_time: float
    end_time: float
    summary_state: list[float]
    dominant_entities: list[str]
    dominant_audio_events: list[str]
    continuity_features: dict[str, float]


@dataclass
class FusionTrace:
    source_tokens: list[str]
    fused_state: list[float]
    modality_weights: dict[str, float]
    agreement_score: float
    conflict_score: float


@dataclass
class MemoryTrace:
    retained_states: list[dict[str, Any]]
    summary_states: list[dict[str, Any]]
    entity_pointers: dict[str, list[str]]
    unresolved_links: list[dict[str, Any]]
