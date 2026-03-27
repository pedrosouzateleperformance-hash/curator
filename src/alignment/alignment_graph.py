from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from data_structures.nodes import AudioEventNode, SceneNode, ShotNode, UtteranceNode


@dataclass(frozen=True)
class AlignmentEdge:
    source: str
    target: str
    edge_type: str
    start_time: float
    end_time: float
    confidence: float


def build_alignment_graph(
    shots: Sequence[ShotNode],
    scenes: Sequence[SceneNode],
    audio_events: Sequence[AudioEventNode],
    utterances: Sequence[UtteranceNode],
    utterance_entities: Dict[str, List[str]],
) -> List[Dict[str, object]]:
    edges: List[AlignmentEdge] = []

    for scene in scenes:
        for shot_id in scene.shot_ids:
            shot = next(s for s in shots if s.id == shot_id)
            edges.append(
                AlignmentEdge(
                    source=scene.id,
                    target=shot_id,
                    edge_type="scene_contains_shot",
                    start_time=shot.start_time,
                    end_time=shot.end_time,
                    confidence=1.0,
                )
            )

    for shot in shots:
        for audio in audio_events:
            if not (audio.end_time < shot.start_time or audio.start_time > shot.end_time):
                edges.append(
                    AlignmentEdge(
                        source=shot.id,
                        target=audio.id,
                        edge_type="shot_overlaps_audio",
                        start_time=max(shot.start_time, audio.start_time),
                        end_time=min(shot.end_time, audio.end_time),
                        confidence=min(shot.confidence, audio.confidence),
                    )
                )

    for utt in utterances:
        for ent in utterance_entities.get(utt.id, []):
            edges.append(
                AlignmentEdge(
                    source=utt.id,
                    target=ent,
                    edge_type="utterance_mentions_entity",
                    start_time=utt.start_time,
                    end_time=utt.end_time,
                    confidence=0.75,
                )
            )

    return [asdict(e) for e in edges]
