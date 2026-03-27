from __future__ import annotations

from typing import Protocol

from src.domain.core import RenderPlan


class CutterPort(Protocol):
    def cut(self, plan: RenderPlan) -> dict[str, object]: ...


class AudioGenerationPort(Protocol):
    def generate_audio(self, plan: RenderPlan) -> dict[str, object]: ...


class VideoGenerationPort(Protocol):
    def generate_video(self, plan: RenderPlan) -> dict[str, object]: ...


class SubtitlePort(Protocol):
    def generate_subtitles(self, plan: RenderPlan) -> dict[str, object]: ...


class HapticPort(Protocol):
    def infer_hapticity(self, plan: RenderPlan) -> dict[str, float]: ...
