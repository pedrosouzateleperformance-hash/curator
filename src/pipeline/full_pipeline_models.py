from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from data_structures.nodes import GraphReadyBundle
from src.mamba_graph_av import Phase7Outputs, RenderedScene
from src.pipeline.encode_sequences import EncodeOutput
from src.pipeline.reasoning_runner import ReasoningOutput


PHASE_ORDER: List[str] = [
    "phase2_ingest",
    "phase3_encode",
    "phase4_reasoning",
    "phase5_reward_policy_scoring",
    "phase6_controller_decision",
    "phase7_synthesis",
]


@dataclass(frozen=True)
class FullPipelineTrace:
    seed: int
    processed_segment_ids: List[str]
    phase_order: List[str] = field(default_factory=lambda: list(PHASE_ORDER))


@dataclass(frozen=True)
class FullPipelineResult:
    rendered_scene: RenderedScene
    phase7_outputs: Phase7Outputs
    ingest_output: GraphReadyBundle
    encode_output: EncodeOutput
    reasoning_outputs: List[ReasoningOutput]
    phase5_scored_candidates: List[Dict[str, Any]]
    controller_decision: Dict[str, Any]
    trace: FullPipelineTrace
