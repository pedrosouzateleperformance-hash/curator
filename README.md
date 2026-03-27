# Phase 6 Inference-Time Decision Engine

This repository contains a real-time, graph-grounded controller for audiovisual edit actions.

## What it does

`phase6_controller.py` selects one of:

- `CUT`
- `HOLD`
- `TRANSITION`
- `ACCELERATE`
- `DEFER`
- `EXPERIMENT`

using constrained expected utility:

- reward vector components,
- information-theoretic signals,
- scale-flow stability,
- validator agreement,
- latency/compute/risk penalties,
- hard/soft rule checks.

## Output contract

The engine returns a structured decision object with:

1. `action`
2. `reward_vector`
3. `info_metrics`
4. `scale_flow`
5. `constraints`
6. `routing`
7. `explanation_path`
8. `final_decision`

## Quick usage

```python
from phase6_controller import (
    Action,
    CandidateAction,
    Constraints,
    InfoMetrics,
    Phase6DecisionEngine,
    RewardVector,
    ScaleFlow,
    ValidatorSignals,
)

engine = Phase6DecisionEngine()

candidate = CandidateAction(
    action=Action.TRANSITION,
    reward_vector=RewardVector(
        R_emotion=0.74,
        R_story=0.71,
        R_rhythm=0.80,
        R_continuity=0.86,
        R_audio_sync=0.88,
        R_balance=0.78,
        R_engagement=0.76,
        R_efficiency=0.84,
        R_surprise=0.55,
        R_compression=0.69,
    ),
    info_metrics=InfoMetrics(
        entropy_before=0.62,
        entropy_after=0.51,
        surprise=0.44,
        entropy_excess=0.12,
        KL_shift=0.08,
    ),
    scale_flow=ScaleFlow(micro=0.81, meso=0.79, macro=0.75, supra=0.72),
    constraints=Constraints(),
    validators=ValidatorSignals(murch=0.83, katz=0.88, chion=0.86, gestalt=0.80),
    graph_path_confidence=0.84,
    latency_cost=0.18,
    compute_cost=0.22,
    uncertainty_cost=0.17,
    explanation_nodes=["emotion/escalation", "rhythm/sync_point", "continuity/entity_stable"],
    explanation_edges=[
        "emotion/escalation -> rhythm/sync_point",
        "rhythm/sync_point -> audio/beat_alignment",
        "continuity/entity_stable -> story/causal_legibility",
    ],
    resolved_conflicts=["minor composition drift resolved by stable center-of-mass"],
    dominating_validators=["katz", "chion"],
)

decision = engine.decide([candidate])
print(decision)
```

## Real-time behavior

- Rejects hard-constraint violations first.
- Uses conservative fallback behavior if all candidates fail (returns `HOLD` + `defer`).
- Supports cost-aware route selection:
  - `local-only`
  - `hybrid`
  - `cloud-assisted`
