from __future__ import annotations

from src.pipeline.contracts import (
    AlignmentEdge,
    AlignmentGraph,
    AudioEventNode,
    EntityNode,
    FrameFeature,
    SceneNode,
    ShotNode,
    UtteranceNode,
)
from src.pipeline.encode_sequences import SequenceEncoderPipeline


def build_fixture(length: int = 40):
    shots = [ShotNode(shot_id=f"sh{i}", start_time=i * 1.0, end_time=i * 1.0 + 0.9, scene_id=f"sc{i//10}") for i in range(length)]
    scenes = [SceneNode(scene_id=f"sc{i}", start_time=i * 10.0, end_time=i * 10.0 + 9.9) for i in range(max(1, length // 10))]

    frames = [
        FrameFeature(
            frame_id=f"fr{i}",
            timestamp=i * 1.0,
            shot_id=f"sh{i}",
            scene_id=f"sc{i//10}",
            features=[0.1 * (i % 5), 0.2, 0.3, 0.4],
            object_ids=[f"e{i%3}"],
            confidence=0.9,
        )
        for i in range(length)
    ]

    audio = [
        AudioEventNode(
            event_id=f"au{i}",
            timestamp=i * 1.0 + 0.1,
            shot_id=f"sh{i}",
            scene_id=f"sc{i//10}",
            event_type="music" if i % 2 == 0 else "fx",
            features=[0.2, 0.1 * (i % 3), 0.5],
            confidence=0.85,
        )
        for i in range(length)
    ]

    utterances = [
        UtteranceNode(
            utterance_id=f"ut{i}",
            timestamp=i * 1.0 + 0.15,
            shot_id=f"sh{i}",
            scene_id=f"sc{i//10}",
            speaker_id=f"e{i%2}",
            text_embedding=[0.3, 0.6, 0.1],
            confidence=0.8,
        )
        for i in range(0, length, 2)
    ]

    entities = [
        EntityNode(
            entity_id=f"e{i%3}",
            timestamp=i * 1.0 + 0.05,
            shot_id=f"sh{i}",
            scene_id=f"sc{i//10}",
            role="character",
            features=[0.2, 0.2, 0.2],
            confidence=0.88,
        )
        for i in range(length)
    ]

    edges = [AlignmentEdge(source_id=f"fr{i}", target_id=f"e{i%3}", edge_type="semantic", weight=0.5) for i in range(length)]
    edges += [AlignmentEdge(source_id=f"sh{i}", target_id=f"sh{i+1}", edge_type="causal", weight=0.7) for i in range(length - 1)]
    graph = AlignmentGraph(edges=edges, node_context={f"fr{i}": {"salience": 0.2} for i in range(length)})

    return shots, scenes, frames, audio, utterances, entities, graph


def test_long_sequence_stability_and_chunk_memory():
    pipeline = SequenceEncoderPipeline(chunk_size=16)
    fixture = build_fixture(length=96)

    out = pipeline.encode_sequences(*fixture)

    assert len(out.sequence_states) > 300
    assert len(out.segment_states) >= 9
    assert len(out.memory_trace.retained_states) > 0


def test_cross_modal_alignment_generates_fusion_trace():
    pipeline = SequenceEncoderPipeline(chunk_size=12)
    fixture = build_fixture(length=24)

    out = pipeline.encode_sequences(*fixture)

    assert len(out.fusion_traces) > 0
    trace = out.fusion_traces[0]
    assert 0.0 <= trace.agreement_score <= 1.0
    assert 0.0 <= trace.conflict_score <= 1.0
    assert abs(sum(trace.modality_weights.values()) - 1.0) < 1e-6


def test_graph_conditioned_updates_are_explicit():
    pipeline = SequenceEncoderPipeline(chunk_size=8)
    fixture = build_fixture(length=20)

    out = pipeline.encode_sequences(*fixture)

    found_graph_signal = any(state.memory_state.get("graph_signal") for state in out.sequence_states)
    assert found_graph_signal


def test_memory_retention_across_chunks_queryable_by_entity():
    pipeline = SequenceEncoderPipeline(chunk_size=5)
    fixture = build_fixture(length=30)

    pipeline.encode_sequences(*fixture)

    pointers = pipeline.memory.query_by_entity("e1")
    assert len(pointers) > 0


def test_deterministic_reset_behavior():
    pipeline = SequenceEncoderPipeline(chunk_size=10)
    fixture = build_fixture(length=25)

    out1 = pipeline.encode_sequences(*fixture)
    pipeline.reset_state()
    out2 = pipeline.encode_sequences(*fixture)

    h1 = out1.sequence_states[0].hidden_state
    h2 = out2.sequence_states[0].hidden_state
    assert h1 == h2
    assert out1.sequence_states[0].token_id == out2.sequence_states[0].token_id
