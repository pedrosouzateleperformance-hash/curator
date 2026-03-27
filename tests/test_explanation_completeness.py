from src.pipeline.reasoning_runner import ReasoningRunner


def test_explanations_cover_required_fields():
    runner = ReasoningRunner()
    out = runner.process(
        {},
        {
            "segment_id": "s0",
            "event_id": "e0",
            "timestamp_start": 0.0,
            "timestamp_end": 2.5,
            "semantic_embedding": [0.8, 0.2, 0.1],
            "action": "conflict",
            "entities": ["hero", "mentor"],
            "themes": ["duty"],
        },
        {},
        {},
    )

    assert out.decision_candidates
    assert out.explanation_traces
    trace = out.explanation_traces[0]
    assert trace.decision_id
    assert trace.reasoning_path
    assert trace.graph_sources
    assert 0.0 <= trace.confidence <= 1.0
