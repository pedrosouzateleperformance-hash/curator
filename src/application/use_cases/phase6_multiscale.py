from __future__ import annotations

from src.domain.core import MultiscaleProfile, NarrativeState, SequenceState
from src.domain.invariants import validate_multiscale_stability
from src.domain.system_state import MultiscaleState, SystemState, TemporalState, TransitionResult


class Phase6MultiscaleUseCase:
    def execute(self, state: SystemState, sequence: SequenceState, narrative: NarrativeState) -> TransitionResult[MultiscaleProfile]:
        magnitude = sum(abs(value) for value in sequence.latent_state)
        frame = {"energy": magnitude}
        shot = {"continuity": sequence.transition_score}
        scene = {"context_density": float(len(narrative.local_context))}
        act = {"global_context_density": float(len(narrative.global_context))}

        regime_shift = abs(shot["continuity"] - frame["energy"]) / max(1.0, frame["energy"])
        profile = MultiscaleProfile(
            segment_id=sequence.segment_id,
            frame_level=frame,
            shot_level=shot,
            scene_level=scene,
            act_level=act,
            regime_shift_score=regime_shift,
        )
        validate_multiscale_stability(profile)
        next_state = state.transition(
            temporal=TemporalState(
                step=state.temporal.step + 1,
                timestamp=max(state.temporal.timestamp, state.temporal.structured.end_time if state.temporal.structured else state.temporal.timestamp),
                segment_id=sequence.segment_id,
                structured=state.temporal.structured,
                sequence=sequence,
            ),
            multiscale=MultiscaleState(
                frame_count=max(1, int(state.temporal.structured.metadata.get("frames", 0)) if state.temporal.structured else 1),
                shot_count=max(1, int(state.temporal.structured.metadata.get("shots", 1)) if state.temporal.structured else 1),
                scene_count=1,
                act_count=1,
                profile=profile,
            ),
        )
        return TransitionResult(output=profile, state=next_state)
