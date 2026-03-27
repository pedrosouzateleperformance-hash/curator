from __future__ import annotations

from src.domain.core import RenderPlan
from src.ports.execution import AudioGenerationPort, CutterPort, HapticPort, SubtitlePort, VideoGenerationPort


class LocalCutter(CutterPort):
    def cut(self, plan: RenderPlan) -> dict[str, object]:
        return {"module": "cutter", "segment_id": plan.segment_id, "actions": list(plan.selected_actions)}


class LocalAudioGenerator(AudioGenerationPort):
    def generate_audio(self, plan: RenderPlan) -> dict[str, object]:
        return {"module": "audio", "segment_id": plan.segment_id, "enhancement": "adaptive"}


class LocalVideoGenerator(VideoGenerationPort):
    def generate_video(self, plan: RenderPlan) -> dict[str, object]:
        return {"module": "video", "segment_id": plan.segment_id, "style": "coherence_preserving"}


class LocalSubtitleGenerator(SubtitlePort):
    def generate_subtitles(self, plan: RenderPlan) -> dict[str, object]:
        return {"module": "subtitles", "segment_id": plan.segment_id, "format": "srt"}


class LocalHapticModule(HapticPort):
    def infer_hapticity(self, plan: RenderPlan) -> dict[str, float]:
        base = max(0.0, min(1.0, plan.scoring_breakdown.get("coherence", 0.5)))
        return {
            "roughness": 1.0 - base,
            "hardness": 0.5 + 0.5 * base,
            "density": 0.4 + 0.6 * base,
            "texture": 0.3 + 0.7 * base,
        }
