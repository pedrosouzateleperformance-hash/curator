from src.pipeline.reasoning_runner import ReasoningRunner


def test_multi_hop_retrieval_returns_paths():
    runner = ReasoningRunner()
    for i in range(5):
        runner.process(
            {},
            {
                "segment_id": f"s{i}",
                "event_id": f"e{i}",
                "timestamp_start": float(i),
                "timestamp_end": float(i + 1),
                "entities": ["a", f"b{i}"],
                "semantic_embedding": [0.9, 0.1, 0.2],
                "action": "advance",
            },
            {},
            {},
        )

    result = runner.memory.retrieve_by_entity("a")
    assert result.node_ids
    assert result.confidence >= 0.0
