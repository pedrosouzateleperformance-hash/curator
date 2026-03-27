from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from data_structures.nodes import AudioEventNode, SceneNode, ShotNode, UtteranceNode


@dataclass(frozen=True)
class TimelineIndex:
    shots: Sequence[ShotNode]
    scenes: Sequence[SceneNode]
    audio_events: Sequence[AudioEventNode]
    utterances: Sequence[UtteranceNode]

    def audio_for_shot(self, shot_id: str) -> List[AudioEventNode]:
        shot = next(s for s in self.shots if s.id == shot_id)
        return [a for a in self.audio_events if not (a.end_time < shot.start_time or a.start_time > shot.end_time)]

    def entities_for_scene(self, scene_id: str, shot_entities: Dict[str, List[str]]) -> List[str]:
        scene = next(s for s in self.scenes if s.id == scene_id)
        out = set()
        for shot_id in scene.shot_ids:
            out.update(shot_entities.get(shot_id, []))
        return sorted(out)
