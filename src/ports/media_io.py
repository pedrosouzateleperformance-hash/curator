from __future__ import annotations

from typing import Protocol, Sequence

from data_structures.nodes import GraphReadyBundle


class MediaIngestPort(Protocol):
    def ingest(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[dict[str, object]],
        *,
        frame_stride: int,
    ) -> GraphReadyBundle: ...
