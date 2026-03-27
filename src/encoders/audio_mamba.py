from __future__ import annotations

from typing import List

from src.encoders.mamba_base import EncoderConfig, MambaLikeEncoder
from src.pipeline.contracts import SequenceToken


class AudioMambaEncoder(MambaLikeEncoder):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__(modality="audio", config=config or EncoderConfig(hidden_size=20, input_size=16, decay=0.91))

    def _project_input(self, payload: List[float]) -> List[float]:
        base = super()._project_input(payload)
        # Audio emphasis: rhythm/silence transitions represented in tail dimensions.
        for i in range(max(0, len(base) - 4), len(base)):
            base[i] *= 1.15
        return base

    def encode_tokens(self, tokens: List[SequenceToken]) -> List[List[float]]:
        return self.forward_sequence(tokens)
