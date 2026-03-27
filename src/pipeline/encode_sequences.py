from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Dict, Iterable, List, Tuple

from src.encoders.audio_mamba import AudioMambaEncoder
from src.encoders.entity_mamba import EntityMambaEncoder
from src.encoders.transition_mamba import TransitionMambaEncoder
from src.encoders.visual_mamba import VisualMambaEncoder
from src.fusion.graph_conditioning import GraphConditioner
from src.fusion.multimodal_fusion import MultimodalFusion
from src.memory.latent_memory import LatentMemory
from src.memory.memory_trace import MemoryRecord, SegmentSummary
from src.pipeline.contracts import (
    AlignmentGraph,
    AudioEventNode,
    EntityNode,
    FrameFeature,
    FusionTrace,
    SceneNode,
    SegmentState,
    SequenceState,
    SequenceToken,
    ShotNode,
    UtteranceNode,
)


@dataclass
class EncodeOutput:
    sequence_states: List[SequenceState]
    segment_states: List[SegmentState]
    fusion_traces: List[FusionTrace]
    memory_trace: object


class SequenceEncoderPipeline:
    def __init__(self, chunk_size: int = 128) -> None:
        self.chunk_size = chunk_size
        self.visual_encoder = VisualMambaEncoder()
        self.audio_encoder = AudioMambaEncoder()
        self.entity_encoder = EntityMambaEncoder()
        self.transition_encoder = TransitionMambaEncoder()
        self.graph_conditioner = GraphConditioner()
        self.fusion = MultimodalFusion(output_size=24)
        self.memory = LatentMemory()

    def reset_state(self) -> None:
        self.visual_encoder.reset_state()
        self.audio_encoder.reset_state()
        self.entity_encoder.reset_state()
        self.transition_encoder.reset_state()
        self.memory = LatentMemory()

    def _transition_tokens(self, shots: List[ShotNode], scenes: List[SceneNode]) -> List[SequenceToken]:
        tokens: List[SequenceToken] = []
        for i in range(1, len(shots)):
            prev, cur = shots[i - 1], shots[i]
            payload = [cur.start_time - prev.end_time, cur.confidence, prev.confidence]
            tokens.append(
                SequenceToken(
                    token_id=f"transition_shot_{prev.shot_id}_{cur.shot_id}",
                    timestamp=cur.start_time,
                    modality="transition",
                    payload=payload,
                    source_ref=cur.shot_id,
                    entity_links=[],
                    shot_id=cur.shot_id,
                    scene_id=cur.scene_id,
                    confidence=min(prev.confidence, cur.confidence),
                    metadata={"transition_type": "shot"},
                )
            )
        for i in range(1, len(scenes)):
            prev, cur = scenes[i - 1], scenes[i]
            payload = [cur.start_time - prev.end_time, cur.confidence]
            tokens.append(
                SequenceToken(
                    token_id=f"transition_scene_{prev.scene_id}_{cur.scene_id}",
                    timestamp=cur.start_time,
                    modality="transition",
                    payload=payload,
                    source_ref=cur.scene_id,
                    entity_links=[],
                    shot_id="",
                    scene_id=cur.scene_id,
                    confidence=min(prev.confidence, cur.confidence),
                    metadata={"transition_type": "scene"},
                )
            )
        return tokens

    def form_sequences(
        self,
        shots: List[ShotNode],
        scenes: List[SceneNode],
        frame_features: List[FrameFeature],
        audio_events: List[AudioEventNode],
        utterances: List[UtteranceNode],
        entities: List[EntityNode],
    ) -> List[SequenceToken]:
        tokens: List[SequenceToken] = []

        for frame in frame_features:
            tokens.append(
                SequenceToken(
                    token_id=f"visual_{frame.frame_id}",
                    timestamp=frame.timestamp,
                    modality="visual",
                    payload=frame.features,
                    source_ref=frame.frame_id,
                    entity_links=frame.object_ids,
                    shot_id=frame.shot_id,
                    scene_id=frame.scene_id,
                    confidence=frame.confidence,
                    metadata={"source": "frame"},
                )
            )

        for event in audio_events:
            tokens.append(
                SequenceToken(
                    token_id=f"audio_{event.event_id}",
                    timestamp=event.timestamp,
                    modality="audio",
                    payload=event.features,
                    source_ref=event.event_id,
                    entity_links=[],
                    shot_id=event.shot_id,
                    scene_id=event.scene_id,
                    confidence=event.confidence,
                    metadata={"event_type": event.event_type},
                )
            )

        for utt in utterances:
            tokens.append(
                SequenceToken(
                    token_id=f"speech_{utt.utterance_id}",
                    timestamp=utt.timestamp,
                    modality="audio",
                    payload=utt.text_embedding,
                    source_ref=utt.utterance_id,
                    entity_links=[utt.speaker_id],
                    shot_id=utt.shot_id,
                    scene_id=utt.scene_id,
                    confidence=utt.confidence,
                    metadata={"source": "utterance", "speaker": utt.speaker_id},
                )
            )

        for entity in entities:
            tokens.append(
                SequenceToken(
                    token_id=f"entity_{entity.entity_id}_{entity.timestamp}",
                    timestamp=entity.timestamp,
                    modality="entity",
                    payload=entity.features,
                    source_ref=entity.entity_id,
                    entity_links=[entity.entity_id],
                    shot_id=entity.shot_id,
                    scene_id=entity.scene_id,
                    confidence=entity.confidence,
                    metadata={"role": entity.role},
                )
            )

        tokens.extend(self._transition_tokens(shots, scenes))
        return sorted(tokens, key=lambda t: (t.timestamp, t.token_id))

    def _encode_chunk(
        self,
        chunk: List[SequenceToken],
        graph_signals: Dict[str, Dict[str, float]],
    ) -> Dict[str, List[float]]:
        by_modality: Dict[str, List[SequenceToken]] = {"visual": [], "audio": [], "entity": [], "transition": []}
        for token in chunk:
            by_modality[token.modality].append(token)

        latent: Dict[str, List[float]] = {}
        for token, state in zip(
            by_modality["visual"],
            self.visual_encoder.forward_sequence(by_modality["visual"], graph_signals),
        ):
            latent[token.token_id] = state
        for token, state in zip(
            by_modality["audio"],
            self.audio_encoder.forward_sequence(by_modality["audio"], graph_signals),
        ):
            latent[token.token_id] = state
        for token, state in zip(
            by_modality["entity"],
            self.entity_encoder.forward_sequence(by_modality["entity"], graph_signals),
        ):
            latent[token.token_id] = state
        for token, state in zip(
            by_modality["transition"],
            self.transition_encoder.forward_sequence(by_modality["transition"], graph_signals),
        ):
            latent[token.token_id] = state
        return latent

    def _segment_summary(self, segment_id: str, items: List[Tuple[SequenceToken, List[float]]]) -> SegmentState:
        start = min(t.timestamp for t, _ in items)
        end = max(t.timestamp for t, _ in items)
        summary_dim = max(len(state) for _, state in items)
        summary = [0.0] * summary_dim
        for _, state in items:
            for i in range(summary_dim):
                summary[i] += state[i % len(state)]
        summary = [s / len(items) for s in summary]

        entity_hits: Dict[str, int] = {}
        audio_events: Dict[str, int] = {}
        for token, _ in items:
            for e in token.entity_links:
                entity_hits[e] = entity_hits.get(e, 0) + 1
            if token.modality == "audio":
                kind = str(token.metadata.get("event_type", token.metadata.get("source", "unknown")))
                audio_events[kind] = audio_events.get(kind, 0) + 1

        continuity = {
            "token_density": len(items) / max(1e-6, end - start + 1e-6),
            "modality_span": len({token.modality for token, _ in items}) / 4.0,
        }

        return SegmentState(
            segment_id=segment_id,
            start_time=start,
            end_time=end,
            summary_state=summary,
            dominant_entities=sorted(entity_hits, key=entity_hits.get, reverse=True)[:5],
            dominant_audio_events=sorted(audio_events, key=audio_events.get, reverse=True)[:5],
            continuity_features=continuity,
        )

    def encode_sequences(
        self,
        shots: List[ShotNode],
        scenes: List[SceneNode],
        frame_features: List[FrameFeature],
        audio_events: List[AudioEventNode],
        utterances: List[UtteranceNode],
        entities: List[EntityNode],
        alignment_graph: AlignmentGraph,
    ) -> EncodeOutput:
        tokens = self.form_sequences(shots, scenes, frame_features, audio_events, utterances, entities)
        graph_signals = self.graph_conditioner.signal_map(tokens, alignment_graph)

        sequence_states: List[SequenceState] = []
        fusion_traces: List[FusionTrace] = []
        token_latents: Dict[str, List[float]] = {}

        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i : i + self.chunk_size]
            latent = self._encode_chunk(chunk, graph_signals)
            token_latents.update(latent)

            for token in chunk:
                state = latent[token.token_id]
                memory_state = {
                    "chunk_index": i // self.chunk_size,
                    "graph_signal": graph_signals.get(token.token_id, {}),
                }
                sequence_states.append(
                    SequenceState(
                        token_id=token.token_id,
                        timestamp=token.timestamp,
                        modality=token.modality,
                        hidden_state=state,
                        memory_state=memory_state,
                        confidence=token.confidence,
                    )
                )
                self.memory.add_state(
                    MemoryRecord(
                        token_id=token.token_id,
                        timestamp=token.timestamp,
                        modality=token.modality,
                        state=state,
                        segment_id=token.scene_id or token.shot_id,
                    ),
                    salience=graph_signals.get(token.token_id, {}).get("causal_strength", 0.0),
                    entity_links=token.entity_links,
                )

            # Local fusion per timestamp with asynchronous support (windowed neighborhood).
            window_groups: Dict[float, List[SequenceToken]] = {}
            for token in chunk:
                ts_key = round(token.timestamp, 2)
                window_groups.setdefault(ts_key, []).append(token)
            for group in window_groups.values():
                if len(group) < 2:
                    continue
                fused, trace = self.fusion.fuse(group, latent)
                fusion_traces.append(trace)
                if trace.conflict_score > 0.75:
                    self.memory.add_unresolved_link(source_token=group[0].token_id, reason="high_cross_modal_conflict")

        segment_states: List[SegmentState] = []
        for scene_id, group in groupby(sorted(tokens, key=lambda t: (t.scene_id, t.timestamp)), key=lambda t: t.scene_id):
            scene_tokens = [(token, token_latents[token.token_id]) for token in list(group)]
            if not scene_tokens:
                continue
            segment = self._segment_summary(scene_id, scene_tokens)
            segment_states.append(segment)
            self.memory.add_segment_summary(
                SegmentSummary(
                    segment_id=segment.segment_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    summary=segment.summary_state,
                    entity_ids=segment.dominant_entities,
                )
            )

        memory_trace = self.memory.export()
        return EncodeOutput(
            sequence_states=sequence_states,
            segment_states=segment_states,
            fusion_traces=fusion_traces,
            memory_trace=memory_trace,
        )
