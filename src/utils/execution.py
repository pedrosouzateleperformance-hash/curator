from __future__ import annotations

from src.domain.core import ExecutionMode, RenderPlan
from src.ports.execution import AudioGenerationPort, CutterPort, HapticPort, SubtitlePort, VideoGenerationPort


def execute_plan(
    plan: RenderPlan,
    cutter: CutterPort,
    audio: AudioGenerationPort | None = None,
    video: VideoGenerationPort | None = None,
    subtitles: SubtitlePort | None = None,
    haptic: HapticPort | None = None,
) -> dict[str, object]:
    output: dict[str, object] = {"mode": plan.mode.value, "steps": [cutter.cut(plan)]}

    if plan.mode in {ExecutionMode.AUDIO_AUGMENTED, ExecutionMode.FULL_GENERATION} and audio is not None:
        output["steps"].append(audio.generate_audio(plan))

    if plan.mode is ExecutionMode.FULL_GENERATION:
        if video is not None:
            output["steps"].append(video.generate_video(plan))
        if subtitles is not None:
            output["steps"].append(subtitles.generate_subtitles(plan))
        if haptic is not None:
            output["hapticity"] = haptic.infer_hapticity(plan)

    return output
