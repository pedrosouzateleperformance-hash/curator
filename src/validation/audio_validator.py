from __future__ import annotations

from dataclasses import dataclass

from src.types import DecisionCandidate, GraphState, NarrativeState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class AudioWeights:
    w_sync: float = 0.45
    w_dialogue: float = 0.35
    w_emotion_audio: float = 0.20


class AudioValidator:
    def __init__(self, weights: AudioWeights | None = None):
        self.weights = weights or AudioWeights()

    def components(self, candidate: DecisionCandidate, graph_state: GraphState, narrative_state: NarrativeState) -> dict[str, float]:
        ctx = candidate.graph_context
        return {
            "synchresis": _clip01(ctx.get("av_sync", graph_state.temporal.get("audio_sync", 0.0))),
            "dialogue_clarity": _clip01(ctx.get("dialogue_clarity", graph_state.semantic.get("dialogue_clarity", 0.0))),
            "emotional_congruence": _clip01(0.5 * ctx.get("audio_emotion_match", 0.0) + 0.5 * narrative_state.emotional_intensity),
        }

    def score(self, candidate: DecisionCandidate, graph_state: GraphState, narrative_state: NarrativeState) -> float:
        c = self.components(candidate, graph_state, narrative_state)
        w = self.weights
        return (
            w.w_sync * c["synchresis"]
            + w.w_dialogue * c["dialogue_clarity"]
            + w.w_emotion_audio * c["emotional_congruence"]
        )
