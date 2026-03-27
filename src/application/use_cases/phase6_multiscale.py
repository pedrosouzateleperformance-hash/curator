from __future__ import annotations

from src.domain.core import MultiscaleProfile, NarrativeState, SequenceState
from src.domain.invariants import validate_multiscale_stability


class Phase6MultiscaleUseCase:
    def execute(self, sequence: SequenceState, narrative: NarrativeState) -> MultiscaleProfile:
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
        return profile
