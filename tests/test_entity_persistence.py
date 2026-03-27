from data_structures.nodes import FrameFeature
from src.entities.entity_tracker import track_entities


def test_entity_persists_across_overlapping_boxes():
    f1 = FrameFeature(0, 0.0, objects=[], faces=[{"bbox": [10, 10, 50, 50]}], motion_vectors=[], saliency_map=[], composition_metrics={})
    f2 = FrameFeature(1, 0.5, objects=[], faces=[{"bbox": [12, 12, 52, 52]}], motion_vectors=[], saliency_map=[], composition_metrics={})

    entities = track_entities([f1, f2], iou_threshold=0.2)
    assert len(entities) == 1
    interval = entities[0].appearances[0]
    assert interval["start_time"] == 0.0
    assert interval["end_time"] == 0.5
