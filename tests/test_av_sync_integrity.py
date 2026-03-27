from data_structures.nodes import AudioEventNode, SceneNode, ShotNode, UtteranceNode
from src.alignment.timeline import TimelineIndex


def test_audio_overlap_query_by_shot():
    shots = [ShotNode("shot_0000", 2.0, 4.0, 2.0, "hard_cut", 0.9)]
    scenes = [SceneNode("scene_000", ["shot_0000"], 2.0, 4.0, [])]
    audio = [
        AudioEventNode("audio_0000", 0.0, 1.0, "music", 0.8),
        AudioEventNode("audio_0001", 2.5, 3.0, "speech", 0.8),
    ]
    timeline = TimelineIndex(shots, scenes, audio, [UtteranceNode("utt", "", 0, 0, "s", [], [], "neutral")])
    overlaps = timeline.audio_for_shot("shot_0000")
    assert [a.id for a in overlaps] == ["audio_0001"]
