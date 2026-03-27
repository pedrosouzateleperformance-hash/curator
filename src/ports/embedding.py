from __future__ import annotations

from typing import Protocol


class EmbeddingPort(Protocol):
    def embed(self, values: list[float]) -> tuple[float, ...]: ...
