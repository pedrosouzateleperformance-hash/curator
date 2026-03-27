from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from data_structures.nodes import GraphReadyBundle
from src.alignment.alignment_graph import build_alignment_graph
from src.entities.entity_tracker import link_entities_to_shots, track_entities
from src.features.audio_features import classify_audio_frames, load_wav_mono, merge_audio_labels
from src.features.speech_processing import TranscriptSegment, asr_from_jsonl, build_utterances
from src.features.visual_features import extract_frame_features
from src.ingest.scene_segmenter import segment_scenes
from src.ingest.shot_detector import detect_shots, frame_difference_scores
from src.ingest.video_loader import VideoLoader


class PipelineRunner:
    """Phase-2 deterministic ingest + segmentation + graph-ready population."""

    def __init__(self, graph_schema_path: str = "data_structures/graph_schema.json"):
        self.graph_schema_path = Path(graph_schema_path)

    def run(
        self,
        video_path: str,
        audio_wav_path: str,
        transcript_records: Sequence[Dict[str, object]],
        output_json_path: str | None = None,
        frame_stride: int = 5,
    ) -> GraphReadyBundle:
        loader = VideoLoader(video_path)
        metadata = loader.load_metadata()

        frame_ids: List[int] = []
        timestamps: List[float] = []
        frames: List[np.ndarray] = []
        for frame_id, ts, frame in loader.iter_frames(stride=frame_stride):
            frame_ids.append(frame_id)
            timestamps.append(ts)
            frames.append(frame)

        diffs = frame_difference_scores(frames)
        shots = detect_shots(diffs, fps=metadata.fps / frame_stride, duration=metadata.duration)

        frame_features = []
        prev = None
        for fid, ts, frame in zip(frame_ids, timestamps, frames):
            frame_features.append(extract_frame_features(fid, ts, frame, prev))
            prev = frame

        audio, sr = load_wav_mono(audio_wav_path)
        audio_labels = classify_audio_frames(audio, sr)
        audio_events = merge_audio_labels(audio_labels, sr)

        transcript_segments: List[TranscriptSegment] = asr_from_jsonl(transcript_records)
        utterances = build_utterances(transcript_segments)

        entities = link_entities_to_shots(track_entities(frame_features), shots)
        shot_entity_map = {shot.id: [] for shot in shots}
        for ent in entities:
            for sid in ent.associated_shots:
                shot_entity_map[sid].append(ent.id)

        visual_similarity = {
            f"{a.id}->{b.id}": 0.8 for a, b in zip(shots[:-1], shots[1:])
        }
        audio_continuity = {
            f"{a.id}->{b.id}": True for a, b in zip(shots[:-1], shots[1:])
        }
        scenes = segment_scenes(shots, {k: set(v) for k, v in shot_entity_map.items()}, visual_similarity, audio_continuity)

        mention_map: Dict[str, List[str]] = {}
        ent_ids = {e.id for e in entities}
        for utt in utterances:
            mention_map[utt.id] = [eid for eid in ent_ids if eid.split("_")[-1] in utt.text]

        alignment_edges = build_alignment_graph(shots, scenes, audio_events, utterances, mention_map)

        bundle = GraphReadyBundle(
            video_metadata=asdict(metadata),
            shots=shots,
            scenes=scenes,
            frame_features=frame_features,
            audio_events=audio_events,
            utterances=utterances,
            entities=entities,
            alignment_edges=alignment_edges,
        )
        self._validate_schema(bundle.to_dict())

        if output_json_path:
            Path(output_json_path).write_text(json.dumps(bundle.to_dict(), indent=2))
        return bundle

    def _validate_schema(self, payload: Dict[str, object]) -> None:
        schema = json.loads(self.graph_schema_path.read_text())
        required = schema["required_collections"]
        for collection_name, required_fields in required.items():
            records = payload.get(collection_name, [])
            for rec in records:
                missing = [k for k in required_fields if k not in rec]
                if missing:
                    raise ValueError(f"Schema violation in {collection_name}: missing {missing}")
