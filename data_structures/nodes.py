from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ShotNode:
    id: str
    start_time: float
    end_time: float
    duration: float
    transition_type: str
    confidence: float


@dataclass(frozen=True)
class SceneNode:
    id: str
    shot_ids: List[str]
    start_time: float
    end_time: float
    dominant_entities: List[str] = field(default_factory=list)
    semantic_label: Optional[str] = None


@dataclass(frozen=True)
class FrameFeature:
    frame_id: int
    timestamp: float
    objects: List[Dict[str, Any]]
    faces: List[Dict[str, Any]]
    motion_vectors: List[Tuple[float, float]]
    saliency_map: List[List[float]]
    composition_metrics: Dict[str, float]


@dataclass(frozen=True)
class AudioEventNode:
    id: str
    start_time: float
    end_time: float
    type: str
    confidence: float


@dataclass(frozen=True)
class UtteranceNode:
    id: str
    text: str
    start_time: float
    end_time: float
    speaker_id: str
    keywords: List[str]
    trigger_words: List[str]
    emotion_label: str


@dataclass(frozen=True)
class EntityNode:
    id: str
    type: str
    appearances: List[Dict[str, float]]
    associated_shots: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphReadyBundle:
    video_metadata: Dict[str, Any]
    shots: List[ShotNode]
    scenes: List[SceneNode]
    frame_features: List[FrameFeature]
    audio_events: List[AudioEventNode]
    utterances: List[UtteranceNode]
    entities: List[EntityNode]
    alignment_edges: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_metadata": self.video_metadata,
            "shots": [asdict(s) for s in self.shots],
            "scenes": [asdict(s) for s in self.scenes],
            "frame_features": [asdict(f) for f in self.frame_features],
            "audio_events": [asdict(a) for a in self.audio_events],
            "utterances": [asdict(u) for u in self.utterances],
            "entities": [asdict(e) for e in self.entities],
            "alignment_edges": self.alignment_edges,
        }
