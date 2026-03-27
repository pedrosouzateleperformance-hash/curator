from __future__ import annotations

from typing import Sequence

from src.domain.core import StructuredData
from src.domain.graph import GraphEdge, GraphNode
from src.ports.graph import GraphRepositoryPort
from src.ports.media_io import MediaIngestPort


class Phase2IngestUseCase:
    def __init__(self, ingest_port: MediaIngestPort, graph_repository: GraphRepositoryPort) -> None:
        self._ingest = ingest_port
        self._graph = graph_repository

    def execute(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[dict[str, object]],
        *,
        frame_stride: int = 5,
    ) -> StructuredData:
        bundle = self._ingest.ingest(input_video, audio_wav_path, transcript_records, frame_stride=frame_stride)
        first_scene = bundle.scenes[0]
        entities = tuple(entity.id for entity in bundle.entities)
        audio_events = tuple(event.type for event in bundle.audio_events)

        scene_node = GraphNode(id=first_scene.id, type="scene", layer="phase2", attributes={"start": first_scene.start_time, "end": first_scene.end_time})
        self._graph.add_node(scene_node)
        for shot in bundle.shots:
            self._graph.add_node(GraphNode(id=shot.id, type="shot", layer="phase2", attributes={"start": shot.start_time, "end": shot.end_time}))
            self._graph.add_edge(GraphEdge(id=f"contains_{first_scene.id}_{shot.id}", type="contains", layer="phase2", source=first_scene.id, target=shot.id, weight=1.0))

        return StructuredData(
            segment_id=first_scene.id,
            start_time=float(first_scene.start_time),
            end_time=float(first_scene.end_time),
            entities=entities,
            audio_events=audio_events,
            metadata={"shots": len(bundle.shots), "utterances": len(bundle.utterances)},
        )
