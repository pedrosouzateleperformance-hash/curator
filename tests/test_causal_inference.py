from src.pipeline.reasoning_runner import ReasoningRunner


def test_causal_inference_builds_chain():
    runner = ReasoningRunner()
    segments = [
        {
            "segment_id": "s0",
            "event_id": "e0",
            "timestamp_start": 0.0,
            "timestamp_end": 1.0,
            "semantic_embedding": [1.0, 0.0, 0.0],
            "action": "setup",
            "entities": ["hero"],
        },
        {
            "segment_id": "s1",
            "event_id": "e1",
            "timestamp_start": 1.0,
            "timestamp_end": 2.0,
            "semantic_embedding": [0.9, 0.1, 0.0],
            "action": "conflict",
            "entities": ["hero", "villain"],
        },
        {
            "segment_id": "s2",
            "event_id": "e2",
            "timestamp_start": 2.0,
            "timestamp_end": 3.0,
            "semantic_embedding": [0.9, 0.0, 0.1],
            "state_changes": ["resolved"],
            "action": "resolution",
            "entities": ["hero"],
        },
    ]

    for seg in segments:
        runner.process({}, seg, {}, {})

    result = runner.memory.retrieve_by_causal_chain("e2")
    assert "event:e2" in result.node_ids
    assert result.edge_ids
