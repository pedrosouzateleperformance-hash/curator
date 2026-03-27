from data_structures.nodes import AudioEventNode, SceneNode, ShotNode, UtteranceNode
from src.alignment.alignment_graph import build_alignment_graph


def test_alignment_edges_have_valid_intervals():
    shots = [ShotNode("shot_0000", 0.0, 2.0, 2.0, "start", 1.0)]
    scenes = [SceneNode("scene_000", ["shot_0000"], 0.0, 2.0, [])]
    audio = [AudioEventNode("audio_0000", 1.0, 3.0, "speech", 0.8)]
    utt = [UtteranceNode("utt_0000", "hello", 1.2, 1.8, "speaker_0", [], [], "neutral")]
    edges = build_alignment_graph(shots, scenes, audio, utt, {"utt_0000": ["ent_0000"]})

    assert edges
    for edge in edges:
        assert edge["start_time"] <= edge["end_time"]
        assert 0.0 <= edge["confidence"] <= 1.0
