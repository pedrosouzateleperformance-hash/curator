from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.pipeline.contracts import SequenceToken


@dataclass
class EncoderConfig:
    hidden_size: int = 16
    input_size: int = 16
    decay: float = 0.92
    gate_scale: float = 0.5


class MambaLikeEncoder:
    """A deterministic, linear-time selective state-space encoder."""

    def __init__(self, modality: str, config: Optional[EncoderConfig] = None) -> None:
        self.modality = modality
        self.config = config or EncoderConfig()
        self.reset_state()

    def reset_state(self) -> None:
        self._state = [0.0 for _ in range(self.config.hidden_size)]
        self._trace: List[Dict[str, object]] = []

    def _project_input(self, payload: List[float]) -> List[float]:
        projected = [0.0 for _ in range(self.config.hidden_size)]
        if not payload:
            return projected
        for i in range(self.config.hidden_size):
            src = payload[i % len(payload)]
            projected[i] = src * (0.3 + 0.7 * ((i + 1) / self.config.hidden_size))
        return projected

    def _selective_gate(self, token: SequenceToken, projected: List[float]) -> List[float]:
        confidence = max(0.0, min(1.0, token.confidence))
        tempo = 1.0 / (1.0 + token.timestamp)
        modality_bias = 1.0 if token.modality == self.modality else 0.8
        return [
            max(0.02, min(0.98, self.config.gate_scale * confidence * modality_bias + tempo * 0.1 + abs(v) * 0.02))
            for v in projected
        ]

    def update_state(self, token: SequenceToken, graph_signal: Optional[Dict[str, float]] = None) -> List[float]:
        graph_signal = graph_signal or {}
        projected = self._project_input(token.payload)
        gate = self._selective_gate(token, projected)
        graph_gate = graph_signal.get("retention", 1.0)
        transition_boost = graph_signal.get("transition_emphasis", 0.0)

        next_state: List[float] = []
        for i, old in enumerate(self._state):
            drive = projected[i] + transition_boost * 0.1
            kept = old * self.config.decay * graph_gate
            nxt = kept * (1.0 - gate[i]) + drive * gate[i]
            next_state.append(nxt)
        self._state = next_state
        self._trace.append({"token_id": token.token_id, "timestamp": token.timestamp, "state": next_state.copy()})
        return next_state

    def forward_sequence(
        self,
        tokens: List[SequenceToken],
        graph_signals: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[List[float]]:
        graph_signals = graph_signals or {}
        outputs = []
        for token in tokens:
            signal = graph_signals.get(token.token_id)
            outputs.append(self.update_state(token, signal))
        return outputs

    def export_latent_trace(self) -> List[Dict[str, object]]:
        return self._trace.copy()
