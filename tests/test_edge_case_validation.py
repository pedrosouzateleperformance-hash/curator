from src.reward.engagement_model import EngagementModel
from src.types import NarrativeState


def test_engagement_empty_inputs():
    model = EngagementModel()
    total, comps = model.score([], [], [])
    assert total == 0.0
    assert comps["pacing_entropy"] == 0.0


def test_emotional_variance_singleton():
    model = EngagementModel()
    val = model.emotional_variance([NarrativeState(emotional_intensity=0.9)])
    assert val == 0.0
