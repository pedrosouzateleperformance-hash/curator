from __future__ import annotations

from src.domain.core import CoherenceMetrics, ExecutionMode, MultiscaleProfile, NarrativeState, RenderPlan


class Phase7DecisionExecutionUseCase:
    def execute(
        self,
        narrative: NarrativeState,
        coherence: CoherenceMetrics,
        multiscale: MultiscaleProfile,
        mode: ExecutionMode,
    ) -> RenderPlan:
        context_score = max(0.0, 1.0 - float(narrative.contradiction_count))
        coherence_score = max(0.0, 1.0 + coherence.smoothness)
        scale_score = max(0.0, 1.0 - multiscale.regime_shift_score)

        actions = ["cut"]
        if mode in {ExecutionMode.AUDIO_AUGMENTED, ExecutionMode.FULL_GENERATION}:
            actions.append("audio")
        if mode is ExecutionMode.FULL_GENERATION:
            actions.extend(["video", "subtitles", "haptic"])

        return RenderPlan(
            segment_id=narrative.segment_id,
            mode=mode,
            selected_actions=tuple(actions),
            scoring_breakdown={
                "context": context_score,
                "coherence": coherence_score,
                "multiscale": scale_score,
            },
        )
