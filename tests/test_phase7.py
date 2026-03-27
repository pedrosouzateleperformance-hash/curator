import unittest

from src.mamba_graph_av.phase7 import (
    BudgetProfile,
    DecisionCandidate,
    GraphState,
    MemoryTrace,
    NarrativeState,
    Phase7Engine,
    PlatformProfile,
    Segment,
    SegmentState,
    StyleProfile,
)


class Phase7EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Phase7Engine()
        self.segments = SegmentState(
            segments=[
                Segment(
                    segment_id="s1",
                    source_clip_id="c1",
                    source_in_ms=0,
                    source_out_ms=3000,
                    desired_in_ms=0,
                    desired_out_ms=2500,
                    importance=0.9,
                    motion_level=0.8,
                    edge_density=0.7,
                    luminance=0.4,
                    dialogue_present=True,
                ),
                Segment(
                    segment_id="s2",
                    source_clip_id="c2",
                    source_in_ms=0,
                    source_out_ms=2000,
                    desired_in_ms=2500,
                    desired_out_ms=4200,
                    importance=0.5,
                    motion_level=0.3,
                    edge_density=0.4,
                    luminance=0.7,
                    dialogue_present=False,
                ),
            ]
        )

    def test_cut_only_mode_disables_generation(self):
        out = self.engine.run(
            graph_state=GraphState(complexity=0.2, semantic_density=0.1),
            narrative_state=NarrativeState(emotional_intensity=0.2, coherence_target=0.8),
            decision_candidates=[DecisionCandidate("cut_only", 0.0, 0.0)],
            style_profile=StyleProfile("clean", 0.1, 0.5, voice_preferred=True),
            budget_profile=BudgetProfile(max_cost=1.0, audio_budget=0.0, video_budget=0.0),
            platform_profile=PlatformProfile(platform="mobile", haptic_supported=True),
            segment_state=self.segments,
            memory_trace=MemoryTrace(),
        )

        self.assertTrue(out.render_plan.enable_cutting)
        self.assertFalse(out.render_plan.enable_audio_generation)
        self.assertFalse(out.render_plan.enable_video_generation)
        self.assertEqual(out.render_plan.selected_mode, "cut_only")
        self.assertEqual(out.cut_timeline.duration_ms, 4200)

    def test_audio_augmented_mode_respects_budget(self):
        out = self.engine.run(
            graph_state=GraphState(complexity=0.9, semantic_density=0.8),
            narrative_state=NarrativeState(emotional_intensity=0.8, coherence_target=0.8),
            decision_candidates=[
                DecisionCandidate("full_synthesis", 0.5, 9.0),
                DecisionCandidate("audio_augmented", 0.3, 3.0),
                DecisionCandidate("cut_only", 0.0, 0.0),
            ],
            style_profile=StyleProfile("cinematic", 0.6, 0.7, voice_preferred=True),
            budget_profile=BudgetProfile(max_cost=4.0, audio_budget=1.0, video_budget=0.0),
            platform_profile=PlatformProfile(platform="desktop", haptic_supported=False),
            segment_state=self.segments,
            memory_trace=MemoryTrace(prior_violations=("continuity_break",)),
        )

        self.assertEqual(out.render_plan.selected_mode, "audio_augmented")
        self.assertTrue(out.audio_plan.generate_music)
        self.assertFalse(out.video_plan.enabled)
        self.assertTrue(any("continuity_break" in x for x in out.render_plan.debug_trace))


if __name__ == "__main__":
    unittest.main()
