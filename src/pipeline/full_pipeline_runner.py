from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Mapping, Sequence

from src.mamba_graph_av import Phase7Engine
from src.pipeline.contracts import SegmentState
from src.pipeline.encode_sequences import SequenceEncoderPipeline
from src.pipeline.full_pipeline_adapters import (
    _phase2_to_phase3,
    _phase4_to_phase5,
    _phase6_to_phase7,
    to_reasoning_segment_payload,
    to_reasoning_sequence_payload,
)
from src.pipeline.full_pipeline_controller import score_phase5_and_select
from src.pipeline.full_pipeline_models import FullPipelineResult, FullPipelineTrace
from src.pipeline.reasoning_runner import ReasoningOutput, ReasoningRunner
from src.policy.policy_model import PolicyConfig, PolicyModel
from src.reward.reward_function import RewardFunction
from src.types import GraphState, NarrativeState, Phase4Input


def run_full_pipeline(
    input_video: str,
    audio_wav_path: str,
    transcript_records: Sequence[Dict[str, object]],
    *,
    seed: int = 7,
    frame_stride: int = 5,
    graph_schema_path: str = "data_structures/graph_schema.json",
    output_json_path: str | None = None,
    ingest_runner: object | None = None,
    encoder: SequenceEncoderPipeline | None = None,
    reasoning_runner: ReasoningRunner | None = None,
    reward_function: RewardFunction | None = None,
    policy_model: PolicyModel | None = None,
    phase7_engine: Phase7Engine | None = None,
) -> FullPipelineResult:
    random.seed(seed)
    phase2 = ingest_runner or _default_ingest_runner(graph_schema_path)
    phase3 = encoder or SequenceEncoderPipeline()
    phase4 = reasoning_runner or ReasoningRunner()
    phase5 = reward_function or RewardFunction()
    phase6 = policy_model or PolicyModel(PolicyConfig(seed=seed))
    phase7 = phase7_engine or Phase7Engine()

    ingest_output = phase2.run(input_video, audio_wav_path, transcript_records, output_json_path, frame_stride)
    encode_output = phase3.encode_sequences(**_phase2_to_phase3(ingest_output))

    segment_states = _ordered_segments(encode_output.segment_states)
    reasoning_outputs = _run_phase4(segment_states, encode_output.sequence_states, encode_output.fusion_traces, encode_output.memory_trace, phase4)
    phase4_inputs = _phase4_to_phase5(reasoning_outputs)
    controller = score_phase5_and_select(phase4_inputs, phase5, phase6)

    phase4_final = phase4_inputs[-1] if phase4_inputs else _empty_phase4_input()
    segment_final = segment_states[-1] if segment_states else _empty_segment()
    phase7_outputs = phase7.run(**_phase6_to_phase7(controller.selected_candidate, phase4_final, segment_final))

    trace = FullPipelineTrace(seed=seed, processed_segment_ids=[item.segment_id for item in segment_states])
    decision = _decision_payload(controller.selected_candidate.candidate_id, controller.selected_candidate.action_type.value, controller.selected_row, controller.ranking)

    return FullPipelineResult(
        rendered_scene=phase7_outputs.rendered_scene,
        phase7_outputs=phase7_outputs,
        ingest_output=ingest_output,
        encode_output=encode_output,
        reasoning_outputs=reasoning_outputs,
        phase5_scored_candidates=controller.ranking,
        controller_decision=decision,
        trace=trace,
    )


def _default_ingest_runner(graph_schema_path: str) -> object:
    from src.pipeline.pipeline_runner import PipelineRunner

    return PipelineRunner(graph_schema_path=graph_schema_path)


def _ordered_segments(segment_states: List[SegmentState]) -> List[SegmentState]:
    return sorted(segment_states, key=lambda item: (item.start_time, item.end_time, item.segment_id))


def _run_phase4(
    segment_states: List[SegmentState],
    sequence_states: Sequence[object],
    fusion_traces: Sequence[object],
    memory_trace: object,
    reasoning_runner: ReasoningRunner,
) -> List[ReasoningOutput]:
    by_segment = _group_sequences_by_segment(sequence_states)
    fusion_payload = [asdict(trace) for trace in fusion_traces]
    outputs: List[ReasoningOutput] = []
    for segment in segment_states:
        outputs.append(
            reasoning_runner.process(
                sequence_state=by_segment.get(segment.segment_id, []),
                segment_state=to_reasoning_segment_payload(segment),
                fusion_trace=fusion_payload,
                memory_trace=memory_trace,
            )
        )
    return outputs


def _group_sequences_by_segment(sequence_states: Sequence[object]) -> Dict[str, List[Mapping[str, object]]]:
    groups: Dict[str, List[Mapping[str, object]]] = {}
    ordered = sorted(sequence_states, key=lambda item: (item.timestamp, item.token_id))
    for sequence in ordered:
        segment_id = str(getattr(sequence, "scene_id", ""))
        groups.setdefault(segment_id, []).append(to_reasoning_sequence_payload(sequence))
    return groups


def _empty_phase4_input() -> Phase4Input:
    return Phase4Input(graph_state=GraphState(), narrative_state=NarrativeState(), decision_candidates=[])


def _empty_segment() -> SegmentState:
    return SegmentState(
        segment_id="segment_0",
        start_time=0.0,
        end_time=1.0,
        summary_state=[0.0],
        dominant_entities=[],
        dominant_audio_events=[],
        continuity_features={"token_density": 0.0, "modality_span": 0.0},
    )


def _decision_payload(candidate_id: str, action: str, selected_scores: Mapping[str, object], ranking: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "selected_candidate_id": candidate_id,
        "selected_action": action,
        "selected_scores": dict(selected_scores),
        "ranking": [dict(row) for row in ranking],
    }
