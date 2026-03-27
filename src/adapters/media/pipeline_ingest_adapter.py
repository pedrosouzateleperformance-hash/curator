from __future__ import annotations

from typing import Sequence

from data_structures.nodes import GraphReadyBundle
from src.pipeline.pipeline_runner import PipelineRunner
from src.ports.media_io import MediaIngestPort


class PipelineIngestAdapter(MediaIngestPort):
    def __init__(self, graph_schema_path: str = "data_structures/graph_schema.json") -> None:
        self._runner = PipelineRunner(graph_schema_path=graph_schema_path)

    def ingest(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[dict[str, object]],
        *,
        frame_stride: int,
    ) -> GraphReadyBundle:
        return self._runner.run(input_video, audio_wav_path, transcript_records, frame_stride=frame_stride)
