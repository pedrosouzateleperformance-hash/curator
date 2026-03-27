from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.memory.memory_orchestrator import GraphState, MemoryOrchestrator


@dataclass
class NarrativeState:
    current_entities: List[str]
    active_conflicts: List[str]
    emotional_state: str
    narrative_phase: str
    unresolved_threads: List[str]
    pacing_state: str
    scene_goal: str = "advance_story"
    tension_level: float = 0.0
    narrative_transitions: List[str] = field(default_factory=list)


class NarrativeStateTracker:
    def __init__(self, memory: MemoryOrchestrator) -> None:
        self.memory = memory
        self._last_phase: str | None = None

    def update(self, graph_state: GraphState, timestamp: float) -> NarrativeState:
        entities = self.memory.retrieve_by_time(timestamp).node_ids
        unresolved = sorted(self.memory.unresolved_threads)

        conflict_nodes = [node_id for node_id in graph_state.causal_node_ids if "conflict" in node_id]
        tension = min(1.0, 0.15 * len(unresolved) + 0.2 * len(conflict_nodes))

        emotion = "neutral"
        if tension > 0.7:
            emotion = "high_tension"
        elif tension > 0.4:
            emotion = "rising_tension"

        pacing_state = "steady"
        if graph_state.avg_segment_duration < 1.8:
            pacing_state = "fast"
        elif graph_state.avg_segment_duration > 4.5:
            pacing_state = "slow"

        phase = self._infer_phase(graph_state)
        transitions: list[str] = []
        if self._last_phase and self._last_phase != phase:
            transitions.append(f"{self._last_phase}->{phase}")
        self._last_phase = phase

        return NarrativeState(
            current_entities=entities,
            active_conflicts=conflict_nodes,
            emotional_state=emotion,
            narrative_phase=phase,
            unresolved_threads=unresolved,
            pacing_state=pacing_state,
            scene_goal="resolve_conflict" if unresolved else "advance_story",
            tension_level=tension,
            narrative_transitions=transitions,
        )

    def _infer_phase(self, graph_state: GraphState) -> str:
        event_count = len(graph_state.causal_node_ids)
        unresolved = len(self.memory.unresolved_threads)
        if event_count < 3:
            return "setup"
        if unresolved >= 3:
            return "complication"
        if unresolved >= 1:
            return "confrontation"
        return "resolution"
