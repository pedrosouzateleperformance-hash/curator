from __future__ import annotations

from typing import List

from src.encoders.mamba_base import EncoderConfig, MambaLikeEncoder
from src.pipeline.contracts import SequenceToken


class TransitionMambaEncoder(MambaLikeEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(modality="transition", config=config or EncoderConfig(hidden_size=14, input_size=8, decay=0.9))

    def _selective_gate(self, token: SequenceToken, projected: List[float]) -> List[float]:
        base = super()._selective_gate(token, projected)
        # Transition stream is intentionally more reactive to cut salience.
        return [min(0.995, g * 1.1) for g in base]

    def encode_tokens(self, tokens: List[SequenceToken]) -> List[List[float]]:
        return self.forward_sequence(tokens)
