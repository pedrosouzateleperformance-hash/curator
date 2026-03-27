from src.pipeline.reasoning_runner import ReasoningRunner


def test_narrative_tracking_stable_fields():
    runner = ReasoningRunner()
    last = None
    for i in range(4):
        out = runner.process(
            {},
            {
                "segment_id": f"s{i}",
                "event_id": f"e{i}",
                "timestamp_start": float(i),
                "timestamp_end": float(i + 1),
                "semantic_embedding": [1.0, 0.2, 0.1],
                "action": "conflict" if i in (1, 2) else "advance",
                "entities": ["hero", "friend"],
            },
            {},
            {},
        )
        last = out.narrative_state

    assert last is not None
    assert isinstance(last.current_entities, list)
    assert last.narrative_phase in {"setup", "complication", "confrontation", "resolution"}
    assert 0.0 <= last.tension_level <= 1.0
