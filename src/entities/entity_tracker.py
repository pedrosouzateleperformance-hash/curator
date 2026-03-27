from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from data_structures.nodes import EntityNode, FrameFeature, ShotNode


def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(1, union)


@dataclass
class _Track:
    id: str
    type: str
    last_bbox: List[int]
    start_time: float
    end_time: float


def track_entities(frame_features: Sequence[FrameFeature], iou_threshold: float = 0.35) -> List[EntityNode]:
    tracks: List[_Track] = []
    next_id = 0

    for frame in frame_features:
        detections: List[Tuple[str, List[int], dict]] = []
        detections.extend(("character", f["bbox"], f) for f in frame.faces)
        detections.extend(("object", o["bbox"], o) for o in frame.objects if "bbox" in o)

        matched = set()
        for det_type, bbox, _ in detections:
            best_idx, best_score = -1, 0.0
            for idx, tr in enumerate(tracks):
                if tr.type != det_type or idx in matched:
                    continue
                score = _iou(bbox, tr.last_bbox)
                if score > best_score:
                    best_idx, best_score = idx, score
            if best_idx >= 0 and best_score >= iou_threshold:
                tr = tracks[best_idx]
                tr.last_bbox = bbox
                tr.end_time = frame.timestamp
                matched.add(best_idx)
            else:
                tracks.append(
                    _Track(
                        id=f"ent_{next_id:04d}",
                        type=det_type,
                        last_bbox=bbox,
                        start_time=frame.timestamp,
                        end_time=frame.timestamp,
                    )
                )
                next_id += 1

    entities: List[EntityNode] = []
    for tr in tracks:
        entities.append(
            EntityNode(
                id=tr.id,
                type=tr.type,
                appearances=[{"start_time": tr.start_time, "end_time": tr.end_time}],
                associated_shots=[],
                attributes={"last_bbox": tr.last_bbox},
            )
        )
    return entities


def link_entities_to_shots(entities: Sequence[EntityNode], shots: Sequence[ShotNode]) -> List[EntityNode]:
    linked: List[EntityNode] = []
    for ent in entities:
        shot_ids: List[str] = []
        for interval in ent.appearances:
            s, e = interval["start_time"], interval["end_time"]
            for shot in shots:
                if not (e < shot.start_time or s > shot.end_time):
                    shot_ids.append(shot.id)
        linked.append(
            EntityNode(
                id=ent.id,
                type=ent.type,
                appearances=ent.appearances,
                associated_shots=sorted(set(shot_ids)),
                attributes=ent.attributes,
            )
        )
    return linked
