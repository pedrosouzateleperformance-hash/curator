from __future__ import annotations

from typing import Dict

from src.pipeline.contracts import AlignmentGraph, SequenceToken


class GraphConditioner:
    """Inspectable graph-conditioned control signals for sequence updates."""

    def build_signal(self, token: SequenceToken, graph: AlignmentGraph) -> Dict[str, float]:
        context = graph.context_for(token.source_ref)
        neighbors = graph.neighbors(token.source_ref)

        causal_strength = sum(edge.weight for edge in neighbors if edge.edge_type == "causal")
        temporal_strength = sum(edge.weight for edge in neighbors if edge.edge_type == "temporal")
        semantic_strength = sum(edge.weight for edge in neighbors if edge.edge_type == "semantic")

        retention = 1.0 + min(0.25, temporal_strength * 0.05)
        transition_emphasis = min(1.0, causal_strength * 0.2 + context.get("cut_prior", 0.0))
        modality_weight_bias = max(0.1, min(2.0, 1.0 + semantic_strength * 0.03 + context.get("salience", 0.0)))

        return {
            "retention": retention,
            "transition_emphasis": transition_emphasis,
            "modality_weight_bias": modality_weight_bias,
            "causal_strength": causal_strength,
            "semantic_strength": semantic_strength,
        }

    def signal_map(self, tokens: list[SequenceToken], graph: AlignmentGraph) -> Dict[str, Dict[str, float]]:
        return {token.token_id: self.build_signal(token, graph) for token in tokens}
