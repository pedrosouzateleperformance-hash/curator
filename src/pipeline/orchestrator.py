from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.application.use_cases import (
    CostAwareRoutingUseCase,
    Phase2IngestUseCase,
    Phase3TemporalEncodingUseCase,
    Phase4ContextReasoningUseCase,
    Phase5CoherenceUseCase,
    Phase6MultiscaleUseCase,
    Phase7DecisionExecutionUseCase,
)
from src.domain.core import RenderPlan
from src.utils.execution import execute_plan


@dataclass(frozen=True)
class PipelineExecutionResult:
    render_plan: RenderPlan
    execution_output: dict[str, object]


class PipelineOrchestrator:
    """Orchestrates use-case execution and dependency wiring only."""

    def __init__(self, graph_schema_path: str = "data_structures/graph_schema.json") -> None:
        from src.adapters import (
            ContextReasoningAdapter,
            LocalAudioGenerator,
            LocalCutter,
            LocalHapticModule,
            LocalSubtitleGenerator,
            LocalVideoGenerator,
            PipelineIngestAdapter,
            RuVectorGraphRepository,
            SequenceEncoderAdapter,
        )

        graph = RuVectorGraphRepository()
        self.phase2 = Phase2IngestUseCase(PipelineIngestAdapter(graph_schema_path=graph_schema_path), graph)
        self.phase3 = Phase3TemporalEncodingUseCase(SequenceEncoderAdapter())
        self.phase4 = Phase4ContextReasoningUseCase(ContextReasoningAdapter(graph))
        self.phase5 = Phase5CoherenceUseCase()
        self.phase6 = Phase6MultiscaleUseCase()
        self.router = CostAwareRoutingUseCase()
        self.phase7 = Phase7DecisionExecutionUseCase()

        self.cutter = LocalCutter()
        self.audio = LocalAudioGenerator()
        self.video = LocalVideoGenerator()
        self.subtitles = LocalSubtitleGenerator()
        self.haptic = LocalHapticModule()

    def run(
        self,
        input_video: str,
        audio_wav_path: str,
        transcript_records: Sequence[dict[str, object]],
        *,
        budget: float = 0.5,
        frame_stride: int = 5,
    ) -> PipelineExecutionResult:
        structured = self.phase2.execute(
            input_video=input_video,
            audio_wav_path=audio_wav_path,
            transcript_records=transcript_records,
            frame_stride=frame_stride,
        )
        sequence_previous = self.phase3.execute(structured)
        sequence_current = self.phase3.execute(structured)
        narrative = self.phase4.execute(sequence_current)
        coherence = self.phase5.execute(sequence_previous, sequence_current)
        multiscale = self.phase6.execute(sequence_current, narrative)
        mode = self.router.execute(budget)
        plan = self.phase7.execute(narrative, coherence, multiscale, mode)

        execution_output = execute_plan(plan, self.cutter, self.audio, self.video, self.subtitles, self.haptic)
        return PipelineExecutionResult(render_plan=plan, execution_output=execution_output)
