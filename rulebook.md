# Mamba-Graph Phase 1 Rulebook

This rulebook defines deterministic validation behavior for the four graph layers:

1. **Heuristic Aesthetic Graph**
2. **Narrative Memory Graph**
3. **Visual Composition Graph**
4. **Audio Ontology Graph**

## Validation contract

Every validation run must return:
- `is_valid`: overall pass/fail.
- `violations`: list of violated rule IDs with evidence.
- `passed_rules`: list of rule IDs that passed.
- `explanations`: explainable output suitable for answering:
  - "Why is this cut valid?"
  - "Which rule was violated?"

## Rules

### R_LAYER_SEPARATION
- Node type must be declared in ontology.
- Node layer must match ontology node type layer.
- Edge type must be declared in ontology.
- Edge layer must be in edge type `layer_scope`.
- Edge source/target node types must match edge type source/target constraints.

### R_MURCH_CUT_SCORE
For each `CUT` edge:
- `murch_scores` must include:
  - `emotion`, `story`, `rhythm`, `eye_trace`, `continuity_2d`, `continuity_3d`.
- Weighted score is computed using ontology weights.
- Weighted score must be `>= constraints.murch_min_cut_score`.

### R_KATZ_CONTINUITY
For each `CUT` edge:
- `continuity.spatial`, `continuity.temporal`, `continuity.action` must all be `true`.
- `katz_logical_inference` must be `true`.

### R_CHION_SYNC
For each `AUDIO_IMAGE_ALIGNMENT` edge:
- `sync_score >= constraints.audio_sync_min`.
- `dialogue_prominence >= constraints.dialogue_prominence_min`.
- `alignment_score` should remain within `[0,1]` and is reported for explainability.

### R_NARRATIVE_COHERENCE
For each `NARRATIVE_CAUSAL` edge:
- `cause_strength >= constraints.narrative_cause_min`.
- If both events have `time_index`, cause event time must be `<=` effect event time.

For each `NARRATIVE_TEMPORAL` edge:
- `delta_t >= 0`.
- Source `time_index` must be `<=` target `time_index` when both exist.

## Explainability conventions

- Explanations include `rule_id`, graph element ID, measured values, thresholds, and decision.
- Violations are explicit and deterministic.
- Output avoids hidden assumptions and keeps all thresholds data-driven from `ontology.yaml`.
