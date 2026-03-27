from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    features: List[float]
    object_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(frozen=True)
class AudioEventNode:
    event_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    event_type: str
    features: List[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class UtteranceNode:
    utterance_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    speaker_id: str
    text_embedding: List[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class EntityNode:
    entity_id: str
    timestamp: float
    shot_id: str
    scene_id: str
    role: str
    features: List[float]
    confidence: float = 1.0


@dataclass(frozen=True)
class AlignmentEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    timestamp: Optional[float] = None


@dataclass
class AlignmentGraph:
    edges: List[AlignmentEdge]
    node_context: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def neighbors(self, node_id: str) -> List[AlignmentEdge]:
        return [edge for edge in self.edges if edge.source_id == node_id or edge.target_id == node_id]

    def context_for(self, node_id: str) -> Dict[str, float]:
        return self.node_context.get(node_id, {})


@dataclass(frozen=True)
class SequenceToken:
    token_id: str
    timestamp: float
    modality: str
    payload: List[float]
    source_ref: str
    entity_links: List[str]
    shot_id: str
    scene_id: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceState:
    token_id: str
    timestamp: float
    modality: str
    hidden_state: List[float]
    memory_state: Dict[str, Any]
    confidence: float


@dataclass
class SegmentState:
    segment_id: str
    start_time: float
    end_time: float
    summary_state: List[float]
    dominant_entities: List[str]
    dominant_audio_events: List[str]
    continuity_features: Dict[str, float]


@dataclass
class FusionTrace:
    source_tokens: List[str]
    fused_state: List[float]
    modality_weights: Dict[str, float]
    agreement_score: float
    conflict_score: float


@dataclass
class MemoryTrace:
    retained_states: List[Dict[str, Any]]
    summary_states: List[Dict[str, Any]]
    entity_pointers: Dict[str, List[str]]
    unresolved_links: List[Dict[str, Any]]
