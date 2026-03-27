from __future__ import annotations

from src.domain.core import CoherenceMetrics, ExecutionMode, MultiscaleProfile, NarrativeState, RenderPlan
from src.domain.system_state import SystemState, TemporalState, TransitionResult


class Phase7DecisionExecutionUseCase:
    def execute(
        self,
        state: SystemState,
        narrative: NarrativeState,
        coherence: CoherenceMetrics,
        multiscale: MultiscaleProfile,
        mode: ExecutionMode,
    ) -> TransitionResult[RenderPlan]:
        context_score = max(0.0, 1.0 - float(narrative.contradiction_count))
        coherence_score = max(0.0, 1.0 + coherence.smoothness)
        scale_score = max(0.0, 1.0 - multiscale.regime_shift_score)

        actions = ["cut"]
        if mode in {ExecutionMode.AUDIO_AUGMENTED, ExecutionMode.FULL_GENERATION}:
            actions.append("audio")
        if mode is ExecutionMode.FULL_GENERATION:
            actions.extend(["video", "subtitles", "haptic"])

        plan = RenderPlan(
            segment_id=narrative.segment_id,
            mode=mode,
            selected_actions=tuple(actions),
            scoring_breakdown={
                "context": context_score,
                "coherence": coherence_score,
                "multiscale": scale_score,
            },
        )
        next_state = state.transition(
            temporal=TemporalState(
                step=state.temporal.step + 1,
                timestamp=max(state.temporal.timestamp, state.temporal.structured.end_time if state.temporal.structured else state.temporal.timestamp),
                segment_id=narrative.segment_id,
                structured=state.temporal.structured,
                sequence=state.temporal.sequence,
            ),
            render_plan=plan,
        )
        return TransitionResult(output=plan, state=next_state)
