from __future__ import annotations

from typing import List

from src.encoders.mamba_base import EncoderConfig, MambaLikeEncoder
from src.pipeline.contracts import SequenceToken


class EntityMambaEncoder(MambaLikeEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(modality="entity", config=config or EncoderConfig(hidden_size=18, input_size=12, decay=0.95))

    def _project_input(self, payload: List[float]) -> List[float]:
        base = super()._project_input(payload)
        # Entity persistence: lower-frequency dimensions stay smoother.
        for i in range(0, len(base), 3):
            base[i] *= 0.9
        return base

    def encode_tokens(self, tokens: List[SequenceToken]) -> List[List[float]]:
        return self.forward_sequence(tokens)
