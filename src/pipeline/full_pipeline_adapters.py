from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

from data_structures.nodes import GraphReadyBundle
from src.mamba_graph_av import (
    BudgetProfile,
    DecisionCandidate as Phase7DecisionCandidate,
    GraphState as Phase7GraphState,
    MemoryTrace as Phase7MemoryTrace,
    NarrativeState as Phase7NarrativeState,
    PlatformProfile,
    Segment,
    SegmentState as Phase7SegmentState,
    StyleProfile,
)
from src.pipeline.contracts import (
    AlignmentEdge,
    AlignmentGraph,
    AudioEventNode,
    EntityNode,
    FrameFeature,
    SceneNode,
    SegmentState,
    ShotNode,
    UtteranceNode,
)
from src.pipeline.reasoning_runner import ReasoningOutput
from src.policy.action_space import ACTION_SPACE
from src.types import ActionType, DecisionCandidate, GraphState, NarrativeState, Phase4Input

DEFAULT_MODE_QUALITY_GAIN = 0.7
ALTERNATE_MODE_QUALITY_GAIN = 0.45
MODE_COSTS: Dict[str, float] = {"cut_only": 0.2, "audio_augmented": 0.55, "full_synthesis": 0.85}

ACTION_MAP: Dict[str, ActionType] = {
    "cut_here": ActionType.CUT,
    "extend_shot": ActionType.HOLD,
    "insert_reaction_shot": ActionType.INSERT,
    "maintain_continuity": ActionType.HOLD,
}


class Phase7InputBundle(Dict[str, Any]):
    pass


def _phase2_to_phase3(bundle: GraphReadyBundle) -> Dict[str, Any]:
    shots, shot_bounds = _convert_shots(bundle)
    scenes = _convert_scenes(bundle)
    frame_features = _convert_frame_features(bundle, shot_bounds)
    audio_events = _convert_audio_events(bundle, shot_bounds)
    utterances = _convert_utterances(bundle, shot_bounds)
    entities = _convert_entities(bundle, shot_bounds)
    alignment_graph = _convert_alignment_graph(bundle)
    return {
        "shots": shots,
        "scenes": scenes,
        "frame_features": frame_features,
        "audio_events": audio_events,
        "utterances": utterances,
        "entities": entities,
        "alignment_graph": alignment_graph,
    }


def _phase4_to_phase5(reasoning_outputs: Sequence[ReasoningOutput]) -> List[Phase4Input]:
    return [_reasoning_to_phase5_input(item) for item in reasoning_outputs]


def _phase6_to_phase7(
    selected: DecisionCandidate,
    phase4_input: Phase4Input,
    segment: SegmentState,
) -> Phase7InputBundle:
    return {
        "graph_state": _to_phase7_graph_state(phase4_input),
        "narrative_state": _to_phase7_narrative_state(phase4_input),
        "decision_candidates": _build_phase7_candidates(selected.action_type),
        "style_profile": StyleProfile("clean_minimal", grain_level=0.25, contrast_level=0.6, voice_preferred=True),
        "budget_profile": BudgetProfile(max_cost=1.0, audio_budget=0.5, video_budget=0.5, prefer_low_cost=True),
        "platform_profile": PlatformProfile(platform="web", haptic_supported=False),
        "segment_state": Phase7SegmentState(segments=[_to_phase7_segment(phase4_input, segment)]),
        "memory_trace": Phase7MemoryTrace(prior_violations=tuple()),
    }


def to_reasoning_segment_payload(segment: SegmentState) -> Dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "start": segment.start_time,
        "end": segment.end_time,
        "timestamp_start": segment.start_time,
        "timestamp_end": segment.end_time,
        "entities": [{"id": item} for item in segment.dominant_entities],
        "semantic_embedding": list(segment.summary_state),
        "causal_embedding": list(segment.summary_state),
        "themes": list(segment.dominant_audio_events),
        "action": "conflict" if segment.continuity_features.get("modality_span", 0.0) > 0.75 else "event",
    }


def to_reasoning_sequence_payload(sequence_state: Any) -> Dict[str, Any]:
    return asdict(sequence_state)


def _convert_shots(bundle: GraphReadyBundle) -> Tuple[List[ShotNode], Dict[str, Tuple[float, float, str]]]:
    scene_by_shot = _scene_id_map(bundle)
    shots = [
        ShotNode(s.id, float(s.start_time), float(s.end_time), scene_by_shot.get(s.id, "scene_unassigned"), float(s.confidence))
        for s in bundle.shots
    ]
    ordered = sorted(shots, key=lambda item: (item.start_time, item.end_time, item.shot_id))
    bounds = {item.shot_id: (item.start_time, item.end_time, item.scene_id) for item in ordered}
    return ordered, bounds


def _convert_scenes(bundle: GraphReadyBundle) -> List[SceneNode]:
    scenes = [SceneNode(s.id, float(s.start_time), float(s.end_time), confidence=1.0) for s in bundle.scenes]
    return sorted(scenes, key=lambda item: (item.start_time, item.end_time, item.scene_id))


def _convert_frame_features(
    bundle: GraphReadyBundle,
    shot_bounds: Dict[str, Tuple[float, float, str]],
) -> List[FrameFeature]:
    items: List[FrameFeature] = []
    ordered = sorted(bundle.frame_features, key=lambda item: (item.timestamp, item.frame_id))
    for frame in ordered:
        shot_id, scene_id = _resolve_shot_scene(frame.timestamp, shot_bounds)
        feature_values = _feature_vector(frame.motion_vectors, frame.composition_metrics)
        object_ids = [str(obj.get("id", obj.get("label", "obj"))) for obj in frame.objects]
        items.append(FrameFeature(str(frame.frame_id), float(frame.timestamp), shot_id, scene_id, feature_values, object_ids, 1.0))
    return items


def _convert_audio_events(
    bundle: GraphReadyBundle,
    shot_bounds: Dict[str, Tuple[float, float, str]],
) -> List[AudioEventNode]:
    events: List[AudioEventNode] = []
    ordered = sorted(bundle.audio_events, key=lambda item: (item.start_time, item.id))
    for event in ordered:
        timestamp = (float(event.start_time) + float(event.end_time)) / 2.0
        shot_id, scene_id = _resolve_shot_scene(timestamp, shot_bounds)
        events.append(AudioEventNode(event.id, timestamp, shot_id, scene_id, event.type, [float(event.confidence)], float(event.confidence)))
    return events


def _convert_utterances(
    bundle: GraphReadyBundle,
    shot_bounds: Dict[str, Tuple[float, float, str]],
) -> List[UtteranceNode]:
    items: List[UtteranceNode] = []
    ordered = sorted(bundle.utterances, key=lambda item: (item.start_time, item.id))
    for utterance in ordered:
        timestamp = (float(utterance.start_time) + float(utterance.end_time)) / 2.0
        shot_id, scene_id = _resolve_shot_scene(timestamp, shot_bounds)
        embedding = [float(len(utterance.text)), float(len(utterance.keywords)), float(len(utterance.trigger_words))]
        items.append(UtteranceNode(utterance.id, timestamp, shot_id, scene_id, utterance.speaker_id, embedding, 1.0))
    return items


def _convert_entities(
    bundle: GraphReadyBundle,
    shot_bounds: Dict[str, Tuple[float, float, str]],
) -> List[EntityNode]:
    items: List[EntityNode] = []
    for entity in sorted(bundle.entities, key=lambda item: item.id):
        appearances = sorted(entity.appearances, key=lambda item: (float(item.get("start_time", 0.0)), float(item.get("end_time", 0.0))))
        for appearance in appearances:
            timestamp = (float(appearance.get("start_time", 0.0)) + float(appearance.get("end_time", 0.0))) / 2.0
            shot_id, scene_id = _resolve_shot_scene(timestamp, shot_bounds)
            features = [float(len(entity.associated_shots)), float(len(entity.attributes))]
            items.append(EntityNode(entity.id, timestamp, shot_id, scene_id, entity.type, features, 1.0))
    return items


def _convert_alignment_graph(bundle: GraphReadyBundle) -> AlignmentGraph:
    edges = [
        AlignmentEdge(
            source_id=str(item.get("source_id", "")),
            target_id=str(item.get("target_id", "")),
            edge_type=str(item.get("edge_type", "unknown")),
            weight=float(item.get("weight", 0.0)),
            timestamp=float(item["timestamp"]) if item.get("timestamp") is not None else None,
        )
        for item in bundle.alignment_edges
    ]
    ordered_edges = sorted(edges, key=lambda item: (item.timestamp if item.timestamp is not None else -1.0, item.source_id, item.target_id, item.edge_type))
    return AlignmentGraph(edges=ordered_edges, node_context={})


def _reasoning_to_phase5_input(output: ReasoningOutput) -> Phase4Input:
    graph_state = _to_phase5_graph_state(output)
    narrative_state = _to_phase5_narrative_state(output)
    candidates = _to_phase5_candidates(output, narrative_state.emotional_intensity)
    return Phase4Input(graph_state=graph_state, narrative_state=narrative_state, decision_candidates=candidates)


def _to_phase5_graph_state(output: ReasoningOutput) -> GraphState:
    temporal = output.graph_state.temporal_graph.current_pacing_profile()
    temporal_nodes = max(1, len(output.graph_state.temporal_graph.nodes))
    semantic_density = float(len(output.graph_state.semantic_graph.nodes)) / temporal_nodes
    causal_consistency = float(len(output.graph_state.causal_graph.edges)) / max(1, len(output.graph_state.causal_graph.nodes))
    entity_persistence = float(len(output.graph_state.entity_graph.nodes)) / temporal_nodes
    rhythm_consistency = 1.0 / (1.0 + abs(2.5 - temporal.get("avg_duration", 0.0)))
    return GraphState(
        semantic={"semantic_density": semantic_density},
        temporal={"rhythm_consistency": rhythm_consistency},
        causal={"causal_consistency": causal_consistency},
        entity={"entity_persistence": entity_persistence},
    )


def _to_phase5_narrative_state(output: ReasoningOutput) -> NarrativeState:
    emotional_intensity = min(1.0, float(output.narrative_state.tension_level))
    coherence = 1.0 if output.narrative_state.pacing_state == "steady" else 0.75
    progression = min(1.0, len(output.graph_state.temporal_graph.nodes) / 10.0)
    return NarrativeState(progression=progression, tension=emotional_intensity, emotional_intensity=emotional_intensity, coherence=coherence)


def _to_phase5_candidates(output: ReasoningOutput, emotional_intensity: float) -> List[DecisionCandidate]:
    candidates: List[DecisionCandidate] = []
    ordered = sorted(output.decision_candidates, key=lambda item: (item.decision_id, item.action))
    for candidate in ordered:
        graph_context = {"story_gain": float(candidate.confidence), "emotion_match": emotional_intensity}
        mapped = ACTION_MAP.get(candidate.action, ActionType.HOLD)
        candidates.append(
            DecisionCandidate(
                candidate_id=candidate.decision_id,
                action_type=mapped,
                timestamp=0.0,
                graph_context=graph_context,
                reasoning_path=list(candidate.justification_path),
                metadata={"raw_action": candidate.action},
            )
        )
    return candidates


def _to_phase7_graph_state(phase4_input: Phase4Input) -> Phase7GraphState:
    complexity = min(1.0, phase4_input.graph_state.semantic.get("semantic_density", 0.0) + 0.1)
    semantic_density = min(1.0, phase4_input.graph_state.causal.get("causal_consistency", 0.0) + 0.1)
    return Phase7GraphState(complexity=complexity, semantic_density=semantic_density)


def _to_phase7_narrative_state(phase4_input: Phase4Input) -> Phase7NarrativeState:
    return Phase7NarrativeState(
        emotional_intensity=phase4_input.narrative_state.emotional_intensity,
        coherence_target=phase4_input.narrative_state.coherence,
    )


def _to_phase7_segment(phase4_input: Phase4Input, segment: SegmentState) -> Segment:
    return Segment(
        segment_id=segment.segment_id,
        source_clip_id=segment.segment_id,
        source_in_ms=int(segment.start_time * 1000),
        source_out_ms=int(segment.end_time * 1000),
        desired_in_ms=int(segment.start_time * 1000),
        desired_out_ms=int(segment.end_time * 1000),
        importance=min(1.0, phase4_input.narrative_state.tension + 0.2),
        motion_level=min(1.0, segment.continuity_features.get("token_density", 0.0)),
        edge_density=min(1.0, segment.continuity_features.get("modality_span", 0.0)),
        luminance=0.5,
        dialogue_present=bool(segment.dominant_audio_events),
    )


def _build_phase7_candidates(chosen_action: ActionType) -> List[Phase7DecisionCandidate]:
    mode_by_action = {
        ActionType.CUT: "cut_only",
        ActionType.HOLD: "cut_only",
        ActionType.TRANSITION: "audio_augmented",
        ActionType.INSERT: "audio_augmented",
        ActionType.EMPHASIZE_AUDIO: "audio_augmented",
        ActionType.SHIFT_PACING: "full_synthesis",
    }
    chosen_mode = mode_by_action.get(chosen_action, "cut_only")
    candidates: List[Phase7DecisionCandidate] = []
    for action in ACTION_SPACE:
        mode = mode_by_action[action]
        quality_gain = DEFAULT_MODE_QUALITY_GAIN if mode == chosen_mode else ALTERNATE_MODE_QUALITY_GAIN
        candidates.append(Phase7DecisionCandidate(mode=mode, quality_gain=quality_gain, cost_estimate=MODE_COSTS[mode]))
    return sorted(candidates, key=lambda item: (item.cost_estimate, -item.quality_gain, item.mode))


def _scene_id_map(bundle: GraphReadyBundle) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for scene in bundle.scenes:
        for shot_id in scene.shot_ids:
            mapping[shot_id] = scene.id
    return mapping


def _resolve_shot_scene(timestamp: float, shot_bounds: Dict[str, Tuple[float, float, str]]) -> Tuple[str, str]:
    for shot_id in sorted(shot_bounds):
        start, end, scene_id = shot_bounds[shot_id]
        if start <= float(timestamp) <= end:
            return shot_id, scene_id
    return "unknown_shot", "scene_unassigned"


def _feature_vector(motion_vectors: Sequence[Tuple[float, float]], composition_metrics: Dict[str, float]) -> List[float]:
    motion_features = [abs(float(x)) + abs(float(y)) for x, y in motion_vectors][:8]
    composition_features = [float(value) for _, value in sorted(composition_metrics.items())][:8]
    values = motion_features + composition_features
    return values if values else [0.0]
