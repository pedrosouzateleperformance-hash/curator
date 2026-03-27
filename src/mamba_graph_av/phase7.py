from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Input state models
# -----------------------------

@dataclass(frozen=True)
class GraphState:
    complexity: float  # 0..1 narrative/scene complexity
    semantic_density: float  # 0..1


@dataclass(frozen=True)
class NarrativeState:
    emotional_intensity: float  # 0..1
    coherence_target: float  # 0..1


@dataclass(frozen=True)
class DecisionCandidate:
    mode: str  # cut_only | audio_augmented | full_synthesis
    quality_gain: float  # expected quality uplift 0..1
    cost_estimate: float


@dataclass(frozen=True)
class StyleProfile:
    subtitle_style: str
    grain_level: float  # 0..1
    contrast_level: float  # 0..1
    voice_preferred: bool


@dataclass(frozen=True)
class BudgetProfile:
    max_cost: float
    audio_budget: float
    video_budget: float
    prefer_low_cost: bool = True


@dataclass(frozen=True)
class PlatformProfile:
    platform: str
    subtitle_safe_margins: Tuple[float, float] = (0.08, 0.1)  # x, y
    haptic_supported: bool = False


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source_clip_id: str
    source_in_ms: int
    source_out_ms: int
    desired_in_ms: int
    desired_out_ms: int
    importance: float  # 0..1
    motion_level: float  # 0..1
    edge_density: float  # 0..1 used for tactile inference
    luminance: float  # 0..1 used for temperature affect inference
    dialogue_present: bool


@dataclass(frozen=True)
class SegmentState:
    segments: List[Segment]


@dataclass(frozen=True)
class MemoryTrace:
    prior_violations: Tuple[str, ...] = ()


# -----------------------------
# Output models
# -----------------------------

@dataclass
class RenderPlan:
    enable_cutting: bool
    enable_audio_generation: bool
    enable_voice_generation: bool
    enable_video_generation: bool
    enable_subtitle_generation: bool
    enable_style_transfer: bool
    enable_haptic_emulation: bool
    selected_mode: str
    estimated_cost: float
    debug_trace: List[str] = field(default_factory=list)


@dataclass
class CutEvent:
    segment_id: str
    source_clip_id: str
    source_in_ms: int
    source_out_ms: int
    timeline_in_ms: int
    timeline_out_ms: int
    transition: str


@dataclass
class CutTimeline:
    events: List[CutEvent]
    duration_ms: int


@dataclass
class SegmentSoundContract:
    segment_id: str
    vococentrism_level: float
    synchresis_priority: float
    music_role: str
    sfx_role: str
    silence_role: str
    timing_tolerance_ms: int
    materiality_emphasis: float


@dataclass
class AudioPlan:
    segment_contracts: List[SegmentSoundContract]
    generate_voice: bool
    generate_music: bool
    generate_ambience: bool
    generate_sfx: bool


@dataclass
class VoiceTrack:
    enabled: bool
    authority_score: float
    timeline_notes: List[str]


@dataclass
class VideoPlan:
    enabled: bool
    style_transfer_only: bool
    synthesis_scope: str
    notes: List[str]


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str
    x: float
    y: float
    style: str


@dataclass
class SubtitleTrack:
    enabled: bool
    cues: List[SubtitleCue]


@dataclass
class MaterialityVector:
    hardness: float
    roughness: float
    elasticity: float
    temperature_affect: float
    weight: float
    texture_density: float


@dataclass
class ContractReport:
    synchresis_score: float
    vococentrism_score: float
    subtitle_legibility_score: float
    style_consistency_score: float
    continuity_score: float
    materiality_score: float
    violations: List[str]


@dataclass
class RenderedScene:
    mode: str
    timeline_duration_ms: int
    artifact_refs: Dict[str, str]


@dataclass
class Phase7Outputs:
    render_plan: RenderPlan
    cut_timeline: CutTimeline
    audio_plan: AudioPlan
    voice_track: VoiceTrack
    video_plan: VideoPlan
    subtitle_track: SubtitleTrack
    materiality_vector: MaterialityVector
    contract_report: ContractReport
    rendered_scene: RenderedScene


# -----------------------------
# Pipeline
# -----------------------------


class Phase7Engine:
    """Phase 7 only: modular audiovisual synthesis workflow.

    Implements:
    - content-aware routing
    - deterministic cutting and assembly
    - optional audio/video generation
    - sound grammar / contract enforcement
    - subtitle design
    - hapticity inference
    - low-cost style transfer preference
    """

    def run(
        self,
        graph_state: GraphState,
        narrative_state: NarrativeState,
        decision_candidates: List[DecisionCandidate],
        style_profile: StyleProfile,
        budget_profile: BudgetProfile,
        platform_profile: PlatformProfile,
        segment_state: SegmentState,
        memory_trace: MemoryTrace,
    ) -> Phase7Outputs:
        render_plan = self._analyze_and_route(
            graph_state,
            narrative_state,
            decision_candidates,
            style_profile,
            budget_profile,
            platform_profile,
            segment_state,
            memory_trace,
        )

        cut_timeline = self._build_cut_timeline(segment_state)
        audio_plan = self._plan_sound_grammar(segment_state, render_plan)
        materiality_vector = self._infer_hapticity(segment_state)

        voice_track = self._optional_audio_generation(
            narrative_state, render_plan, audio_plan, materiality_vector
        )
        video_plan = self._optional_video_generation(
            render_plan, style_profile, materiality_vector
        )
        subtitle_track = self._design_subtitles(
            segment_state, render_plan, style_profile, platform_profile
        )
        contract_report = self._validate_contract(
            render_plan,
            cut_timeline,
            audio_plan,
            subtitle_track,
            materiality_vector,
            narrative_state,
            memory_trace,
        )
        rendered_scene = self._assemble_scene(render_plan, cut_timeline)

        return Phase7Outputs(
            render_plan=render_plan,
            cut_timeline=cut_timeline,
            audio_plan=audio_plan,
            voice_track=voice_track,
            video_plan=video_plan,
            subtitle_track=subtitle_track,
            materiality_vector=materiality_vector,
            contract_report=contract_report,
            rendered_scene=rendered_scene,
        )

    # ---- step 1: analyze and route ----
    def _analyze_and_route(
        self,
        graph_state: GraphState,
        narrative_state: NarrativeState,
        decision_candidates: List[DecisionCandidate],
        style_profile: StyleProfile,
        budget_profile: BudgetProfile,
        platform_profile: PlatformProfile,
        segment_state: SegmentState,
        memory_trace: MemoryTrace,
    ) -> RenderPlan:
        debug = ["Phase7 routing start"]

        if not decision_candidates:
            decision_candidates = [
                DecisionCandidate("cut_only", quality_gain=0.0, cost_estimate=0.0)
            ]
            debug.append("No candidates supplied: defaulting to cut_only")

        # deterministic ordering by cost first (cheap), then quality gain desc, then mode name
        ranked = sorted(
            decision_candidates,
            key=lambda c: (c.cost_estimate, -c.quality_gain, c.mode),
        )

        complexity_pressure = (graph_state.complexity + graph_state.semantic_density) / 2
        emotion_pressure = narrative_state.emotional_intensity
        need_generation = complexity_pressure > 0.62 or emotion_pressure > 0.75
        debug.append(
            f"Need generation heuristic={need_generation} "
            f"(complexity_pressure={complexity_pressure:.2f}, emotion_pressure={emotion_pressure:.2f})"
        )

        selected = ranked[0]
        if need_generation:
            viable = [c for c in ranked if c.cost_estimate <= budget_profile.max_cost]
            if viable:
                # still cost-aware: choose smallest cost that still has measurable quality gain
                gainful = [c for c in viable if c.quality_gain >= 0.1]
                selected = (gainful or viable)[0]
        debug.append(
            f"Selected mode={selected.mode}, estimated_cost={selected.cost_estimate:.2f}"
        )

        enable_cutting = True
        enable_audio_generation = selected.mode in ("audio_augmented", "full_synthesis")
        enable_voice_generation = enable_audio_generation and style_profile.voice_preferred
        enable_video_generation = selected.mode == "full_synthesis"
        enable_subtitle_generation = True
        enable_style_transfer = enable_video_generation or (
            selected.mode == "audio_augmented" and style_profile.grain_level > 0.2
        )
        enable_haptic_emulation = platform_profile.haptic_supported

        # budget gates
        if enable_audio_generation and budget_profile.audio_budget <= 0:
            enable_audio_generation = False
            enable_voice_generation = False
            debug.append("Audio disabled by audio_budget<=0")
        if enable_video_generation and budget_profile.video_budget <= 0:
            enable_video_generation = False
            debug.append("Video generation disabled by video_budget<=0")

        # continuity hardening from memory violations
        if "continuity_break" in memory_trace.prior_violations:
            debug.append("MemoryTrace continuity_break found: enforcing conservative transitions")

        debug.append(f"Segments={len(segment_state.segments)}; prefer_low_cost={budget_profile.prefer_low_cost}")

        return RenderPlan(
            enable_cutting=enable_cutting,
            enable_audio_generation=enable_audio_generation,
            enable_voice_generation=enable_voice_generation,
            enable_video_generation=enable_video_generation,
            enable_subtitle_generation=enable_subtitle_generation,
            enable_style_transfer=enable_style_transfer,
            enable_haptic_emulation=enable_haptic_emulation,
            selected_mode=selected.mode,
            estimated_cost=selected.cost_estimate,
            debug_trace=debug,
        )

    # ---- step 2: cutter module ----
    def _build_cut_timeline(self, segment_state: SegmentState) -> CutTimeline:
        events: List[CutEvent] = []
        cursor = 0

        # deterministic order: by desired timeline in/out then segment_id
        ordered = sorted(
            segment_state.segments,
            key=lambda s: (s.desired_in_ms, s.desired_out_ms, s.segment_id),
        )

        for idx, seg in enumerate(ordered):
            source_duration = max(1, seg.source_out_ms - seg.source_in_ms)
            desired_duration = max(1, seg.desired_out_ms - seg.desired_in_ms)
            use_duration = min(source_duration, desired_duration)

            transition = "cut"
            if idx > 0:
                prev = ordered[idx - 1]
                transition = "match_cut" if abs(prev.motion_level - seg.motion_level) < 0.25 else "hard_cut"

            event = CutEvent(
                segment_id=seg.segment_id,
                source_clip_id=seg.source_clip_id,
                source_in_ms=seg.source_in_ms,
                source_out_ms=seg.source_in_ms + use_duration,
                timeline_in_ms=cursor,
                timeline_out_ms=cursor + use_duration,
                transition=transition,
            )
            events.append(event)
            cursor += use_duration

        return CutTimeline(events=events, duration_ms=cursor)

    # ---- step 3: sound grammar planner ----
    def _plan_sound_grammar(self, segment_state: SegmentState, render_plan: RenderPlan) -> AudioPlan:
        contracts: List[SegmentSoundContract] = []
        for seg in segment_state.segments:
            voco = 0.8 if seg.dialogue_present else 0.3
            synch = 0.9 if seg.motion_level > 0.6 else 0.65
            contracts.append(
                SegmentSoundContract(
                    segment_id=seg.segment_id,
                    vococentrism_level=voco,
                    synchresis_priority=synch,
                    music_role="pulse_support" if seg.importance > 0.6 else "bed",
                    sfx_role="material_emphasis",
                    silence_role="punctuation" if seg.importance > 0.7 else "breathing_room",
                    timing_tolerance_ms=45 if seg.motion_level > 0.7 else 80,
                    materiality_emphasis=max(0.2, seg.edge_density),
                )
            )

        return AudioPlan(
            segment_contracts=contracts,
            generate_voice=render_plan.enable_voice_generation,
            generate_music=render_plan.enable_audio_generation,
            generate_ambience=render_plan.enable_audio_generation,
            generate_sfx=render_plan.enable_audio_generation,
        )

    # ---- step 4: hapticity module ----
    def _infer_hapticity(self, segment_state: SegmentState) -> MaterialityVector:
        if not segment_state.segments:
            return MaterialityVector(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

        n = float(len(segment_state.segments))
        avg_motion = sum(s.motion_level for s in segment_state.segments) / n
        avg_edge = sum(s.edge_density for s in segment_state.segments) / n
        avg_lum = sum(s.luminance for s in segment_state.segments) / n
        avg_importance = sum(s.importance for s in segment_state.segments) / n

        hardness = min(1.0, 0.3 + 0.7 * avg_edge)
        roughness = min(1.0, 0.2 + 0.8 * avg_edge)
        elasticity = max(0.0, 1.0 - hardness * 0.7)
        temperature_affect = 1.0 - avg_lum
        weight = min(1.0, 0.2 + 0.6 * avg_importance + 0.2 * avg_motion)
        texture_density = min(1.0, (avg_edge + avg_motion) / 2)

        return MaterialityVector(
            hardness=hardness,
            roughness=roughness,
            elasticity=elasticity,
            temperature_affect=temperature_affect,
            weight=weight,
            texture_density=texture_density,
        )

    # ---- step 5: optional audio generation ----
    def _optional_audio_generation(
        self,
        narrative_state: NarrativeState,
        render_plan: RenderPlan,
        audio_plan: AudioPlan,
        materiality: MaterialityVector,
    ) -> VoiceTrack:
        if not render_plan.enable_audio_generation:
            return VoiceTrack(enabled=False, authority_score=0.0, timeline_notes=["cut-only mode"])

        authority = min(
            1.0,
            0.5 + 0.4 * narrative_state.emotional_intensity + 0.1 * (1.0 - materiality.roughness),
        )
        notes = [
            "Voice aligned to cut points",
            f"Authority target={authority:.2f}",
            f"Voice enabled={audio_plan.generate_voice}",
        ]
        return VoiceTrack(enabled=audio_plan.generate_voice, authority_score=authority, timeline_notes=notes)

    # ---- step 6: optional video generation ----
    def _optional_video_generation(
        self,
        render_plan: RenderPlan,
        style_profile: StyleProfile,
        materiality: MaterialityVector,
    ) -> VideoPlan:
        if not render_plan.enable_video_generation:
            return VideoPlan(
                enabled=False,
                style_transfer_only=render_plan.enable_style_transfer,
                synthesis_scope="none",
                notes=["Reuse existing imagery; no synthesis"],
            )

        # low-cost bias: style transfer + partial synthesis
        style_only = render_plan.enable_style_transfer
        scope = "partial" if style_only else "full"
        notes = [
            f"grain={style_profile.grain_level:.2f}, contrast={style_profile.contrast_level:.2f}",
            f"material texture_density={materiality.texture_density:.2f}",
        ]
        return VideoPlan(enabled=True, style_transfer_only=style_only, synthesis_scope=scope, notes=notes)

    # ---- step 7: subtitle design ----
    def _design_subtitles(
        self,
        segment_state: SegmentState,
        render_plan: RenderPlan,
        style_profile: StyleProfile,
        platform_profile: PlatformProfile,
    ) -> SubtitleTrack:
        if not render_plan.enable_subtitle_generation:
            return SubtitleTrack(enabled=False, cues=[])

        cues: List[SubtitleCue] = []
        cursor = 0
        margin_x, margin_y = platform_profile.subtitle_safe_margins

        for seg in sorted(segment_state.segments, key=lambda s: (s.desired_in_ms, s.segment_id)):
            dur = max(700, seg.desired_out_ms - seg.desired_in_ms)
            x = 0.5
            # eye-trace placement: higher when motion is high to avoid occlusion in lower thirds
            y = max(margin_y, min(0.88, 0.84 - 0.18 * seg.motion_level))
            text = f"[{seg.segment_id}] beat {seg.importance:.2f}"
            cues.append(
                SubtitleCue(
                    start_ms=cursor,
                    end_ms=cursor + dur,
                    text=text,
                    x=max(margin_x, min(1 - margin_x, x)),
                    y=y,
                    style=style_profile.subtitle_style,
                )
            )
            cursor += dur

        return SubtitleTrack(enabled=True, cues=cues)

    # ---- step 8: contract validation ----
    def _validate_contract(
        self,
        render_plan: RenderPlan,
        cut_timeline: CutTimeline,
        audio_plan: AudioPlan,
        subtitle_track: SubtitleTrack,
        materiality: MaterialityVector,
        narrative_state: NarrativeState,
        memory_trace: MemoryTrace,
    ) -> ContractReport:
        violations: List[str] = []

        synch = 0.9 if audio_plan.generate_sfx else 0.68
        voco_values = [c.vococentrism_level for c in audio_plan.segment_contracts]
        voco = sum(voco_values) / len(voco_values) if voco_values else 0.5
        subtitle_legibility = 0.95 if subtitle_track.enabled else 0.0
        style_consistency = 0.88 if render_plan.enable_style_transfer or not render_plan.enable_video_generation else 0.74
        continuity = 0.92 if cut_timeline.events else 0.0
        material_score = (materiality.hardness + materiality.roughness + materiality.texture_density) / 3

        if render_plan.estimated_cost < 0:
            violations.append("invalid_cost")
        if continuity < narrative_state.coherence_target:
            violations.append("continuity_below_target")
        if "continuity_break" in memory_trace.prior_violations and continuity < 0.95:
            violations.append("recurring_continuity_risk")

        return ContractReport(
            synchresis_score=synch,
            vococentrism_score=voco,
            subtitle_legibility_score=subtitle_legibility,
            style_consistency_score=style_consistency,
            continuity_score=continuity,
            materiality_score=material_score,
            violations=violations,
        )

    def _assemble_scene(self, render_plan: RenderPlan, cut_timeline: CutTimeline) -> RenderedScene:
        return RenderedScene(
            mode=render_plan.selected_mode,
            timeline_duration_ms=cut_timeline.duration_ms,
            artifact_refs={
                "timeline": "timeline://cut_timeline_v1",
                "audio": "audio://mix_v1" if render_plan.enable_audio_generation else "audio://source_passthrough",
                "video": "video://synth_v1" if render_plan.enable_video_generation else "video://source_recut",
            },
        )


def serialize_outputs(outputs: Phase7Outputs) -> Dict[str, Any]:
    """Structured dict export for debugging and contract trace emission."""
    return asdict(outputs)
