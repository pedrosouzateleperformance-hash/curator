from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from data_structures.nodes import SceneNode, ShotNode


@dataclass(frozen=True)
class SceneSegmentationConfig:
    max_gap_seconds: float = 8.0
    min_entity_overlap: int = 1
    min_visual_similarity: float = 0.7


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def segment_scenes(
    shots: Sequence[ShotNode],
    shot_entities: Dict[str, Set[str]],
    visual_similarity: Dict[str, float],
    audio_continuity: Dict[str, bool],
    config: SceneSegmentationConfig | None = None,
) -> List[SceneNode]:
    cfg = config or SceneSegmentationConfig()
    if not shots:
        return []

    scenes: List[SceneNode] = []
    current: List[ShotNode] = [shots[0]]

    def flush_scene(index: int, scene_shots: List[ShotNode]) -> None:
        entity_union: Set[str] = set()
        for shot in scene_shots:
            entity_union |= shot_entities.get(shot.id, set())
        scenes.append(
            SceneNode(
                id=f"scene_{index:03d}",
                shot_ids=[s.id for s in scene_shots],
                start_time=scene_shots[0].start_time,
                end_time=scene_shots[-1].end_time,
                dominant_entities=sorted(entity_union),
                semantic_label=None,
            )
        )

    for prev, curr in zip(shots[:-1], shots[1:]):
        prev_entities = shot_entities.get(prev.id, set())
        curr_entities = shot_entities.get(curr.id, set())
        entity_overlap = len(prev_entities & curr_entities)
        semantic_similarity = _jaccard(prev_entities, curr_entities)
        visual_key = f"{prev.id}->{curr.id}"
        visual_sim = visual_similarity.get(visual_key, semantic_similarity)
        audio_ok = audio_continuity.get(visual_key, True)
        temporal_gap = curr.start_time - prev.end_time

        same_scene = (
            temporal_gap <= cfg.max_gap_seconds
            and (entity_overlap >= cfg.min_entity_overlap or visual_sim >= cfg.min_visual_similarity)
            and audio_ok
        )

        if same_scene:
            current.append(curr)
        else:
            flush_scene(len(scenes), current)
            current = [curr]

    flush_scene(len(scenes), current)
    return scenes
