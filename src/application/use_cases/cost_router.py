from __future__ import annotations

from src.domain.core import ExecutionMode


class CostAwareRoutingUseCase:
    def execute(self, budget: float) -> ExecutionMode:
        if budget < 0.33:
            return ExecutionMode.CUT_ONLY
        if budget < 0.66:
            return ExecutionMode.AUDIO_AUGMENTED
        return ExecutionMode.FULL_GENERATION
