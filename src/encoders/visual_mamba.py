from __future__ import annotations

from typing import List

from src.encoders.mamba_base import EncoderConfig, MambaLikeEncoder
from src.pipeline.contracts import SequenceToken


class VisualMambaEncoder(MambaLikeEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(modality="visual", config=config or EncoderConfig(hidden_size=24, input_size=16, decay=0.94))

    def _project_input(self, payload: List[float]) -> List[float]:
        base = super()._project_input(payload)
        # Visual emphasis: shot boundary + motion continuity proxies in early dimensions.
        for i in range(min(4, len(base))):
            base[i] *= 1.2
        return base

    def encode_tokens(self, tokens: List[SequenceToken]) -> List[List[float]]:
        return self.forward_sequence(tokens)
