from __future__ import annotations

from src.types import ActionType

ACTION_SPACE = [
    ActionType.CUT,
    ActionType.HOLD,
    ActionType.TRANSITION,
    ActionType.INSERT,
    ActionType.EMPHASIZE_AUDIO,
    ActionType.SHIFT_PACING,
]

ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_SPACE)}
