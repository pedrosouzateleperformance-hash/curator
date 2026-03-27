from __future__ import annotations

from typing import Dict, Sequence

from src.evaluation.metrics import convergence_rate, multi_objective_balance, stability_score


def evaluate_training_logs(logs: Dict[str, Sequence[float]], component_means: Dict[str, float]) -> Dict[str, float]:
    return {
        "convergence_rate": convergence_rate(logs.get("mean_reward", [])),
        "stability_score": stability_score(logs.get("grad_norm", [])),
        "multi_objective_balance": multi_objective_balance(component_means),
        "final_mean_reward": logs.get("mean_reward", [0.0])[-1] if logs.get("mean_reward") else 0.0,
    }
