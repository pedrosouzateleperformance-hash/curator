from src.pipeline.reasoning_runner import ReasoningRunner


def _segment(idx: int):
    return {
        "segment_id": f"s{idx}",
        "event_id": f"e{idx}",
        "timestamp_start": float(idx),
        "timestamp_end": float(idx + 1),
        "entities": [{"id": "hero", "type": "character"}, {"id": f"obj{idx}", "type": "object"}],
        "semantic_embedding": [1.0, 0.0, float(idx) * 0.1],
        "themes": ["loyalty"],
        "concepts": ["quest"],
        "action": "advance",
    }


def test_graphs_build_consistently():
    runner = ReasoningRunner()
    for i in range(3):
        out = runner.process({}, _segment(i), {}, {})

    graph_state = out.graph_state
    assert len(graph_state.temporal_node_ids) == 3
    assert len(graph_state.entity_node_ids) >= 2
    assert len(graph_state.semantic_node_ids) >= 2
    assert len(graph_state.causal_node_ids) == 1
