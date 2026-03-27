from __future__ import annotations

from typing import Dict, Sequence


def convergence_rate(reward_history: Sequence[float]) -> float:
    if len(reward_history) < 2:
        return 0.0
    diffs = [reward_history[i + 1] - reward_history[i] for i in range(len(reward_history) - 1)]
    positives = sum(1 for d in diffs if d > 0)
    return positives / len(diffs)


def stability_score(grad_norm_history: Sequence[float], threshold: float = 5.0) -> float:
    if not grad_norm_history:
        return 0.0
    stable = sum(1 for g in grad_norm_history if g <= threshold)
    return stable / len(grad_norm_history)


def multi_objective_balance(component_means: Dict[str, float]) -> float:
    if not component_means:
        return 0.0
    vals = list(component_means.values())
    spread = max(vals) - min(vals)
    return max(0.0, 1.0 - spread)
