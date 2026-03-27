from __future__ import annotations

from typing import Dict, List, Tuple

from src.pipeline.contracts import FusionTrace, SequenceToken


class MultimodalFusion:
    """Cross-modal gating with modality-aware weighting and confidence control."""

    def __init__(self, output_size: int = 24) -> None:
        self.output_size = output_size

    def _normalized_weights(self, tokens: List[SequenceToken], graph_bias: Dict[str, float] | None = None) -> Dict[str, float]:
        graph_bias = graph_bias or {}
        raw: Dict[str, float] = {}
        for token in tokens:
            base = token.confidence
            if token.modality not in raw:
                raw[token.modality] = 0.0
            raw[token.modality] += base
        for modality, value in list(raw.items()):
            raw[modality] = value * graph_bias.get(modality, 1.0)
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def fuse(
        self,
        tokens: List[SequenceToken],
        latent_by_token_id: Dict[str, List[float]],
        graph_bias: Dict[str, float] | None = None,
    ) -> Tuple[List[float], FusionTrace]:
        weights = self._normalized_weights(tokens, graph_bias)
        fused = [0.0 for _ in range(self.output_size)]

        for token in tokens:
            latent = latent_by_token_id[token.token_id]
            w = weights[token.modality]
            for i in range(self.output_size):
                fused[i] += w * latent[i % len(latent)]

        modality_vectors: Dict[str, List[float]] = {}
        for token in tokens:
            modality_vectors.setdefault(token.modality, [0.0] * self.output_size)
            latent = latent_by_token_id[token.token_id]
            for i in range(self.output_size):
                modality_vectors[token.modality][i] += latent[i % len(latent)]

        agreement = 0.0
        modalities = list(modality_vectors.keys())
        if len(modalities) > 1:
            pairs = 0
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    pairs += 1
                    v1 = modality_vectors[modalities[i]]
                    v2 = modality_vectors[modalities[j]]
                    dist = sum(abs(a - b) for a, b in zip(v1, v2)) / self.output_size
                    agreement += 1.0 / (1.0 + dist)
            agreement /= max(1, pairs)
        else:
            agreement = 1.0

        conflict = 1.0 - agreement

        return fused, FusionTrace(
            source_tokens=[t.token_id for t in tokens],
            fused_state=fused,
            modality_weights=weights,
            agreement_score=agreement,
            conflict_score=conflict,
        )
