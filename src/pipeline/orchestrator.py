from __future__ import annotations

from typing import Dict, Sequence

from src.pipeline.full_pipeline import FullPipelineResult, run_full_pipeline


class PipelineOrchestrator:
    """Thin orchestration wrapper for the full end-to-end pipeline."""

    def run_full_pipeline(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[Dict[str, object]],
        **kwargs: object,
    ) -> FullPipelineResult:
        return run_full_pipeline(
            input_video=input_video,
            audio_wav_path=audio_wav_path,
            transcript_records=transcript_records,
            **kwargs,
        )
