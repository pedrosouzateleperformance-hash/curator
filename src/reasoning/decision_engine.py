from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.memory.memory_orchestrator import GraphState
from src.reasoning.narrative_tracker import NarrativeState


@dataclass
class DecisionCandidate:
    decision_id: str
    action: str
    confidence: float
    justification_path: List[str]
    supporting_nodes: List[str]
    supporting_edges: List[str]


class DecisionProposalEngine:
    def propose(self, graph_state: GraphState, narrative_state: NarrativeState) -> List[DecisionCandidate]:
        candidates: List[DecisionCandidate] = []
        avg_duration = graph_state.avg_segment_duration

        if narrative_state.tension_level > 0.6 and avg_duration > 2.2:
            candidates.append(
                DecisionCandidate(
                    decision_id="decision:cut_here",
                    action="cut_here",
                    confidence=min(0.98, 0.65 + narrative_state.tension_level * 0.25),
                    justification_path=["causal:unresolved->temporal:pacing"],
                    supporting_nodes=narrative_state.unresolved_threads[:3],
                    supporting_edges=graph_state.causal_edge_ids[:3],
                )
            )

        if narrative_state.pacing_state == "fast" and narrative_state.tension_level < 0.4:
            candidates.append(
                DecisionCandidate(
                    decision_id="decision:extend_shot",
                    action="extend_shot",
                    confidence=0.72,
                    justification_path=["temporal:short_duration->rhythm_balance"],
                    supporting_nodes=sorted(graph_state.temporal_node_ids)[-2:],
                    supporting_edges=[],
                )
            )

        if narrative_state.active_conflicts and narrative_state.current_entities:
            candidates.append(
                DecisionCandidate(
                    decision_id="decision:reaction_shot",
                    action="insert_reaction_shot",
                    confidence=0.78,
                    justification_path=["entity:active_character->causal:conflict"],
                    supporting_nodes=narrative_state.current_entities[:2] + narrative_state.active_conflicts[:2],
                    supporting_edges=graph_state.entity_co_occurrence_edges[:2],
                )
            )

        if narrative_state.narrative_phase == "resolution" and narrative_state.pacing_state == "fast":
            candidates.append(
                DecisionCandidate(
                    decision_id="decision:maintain_continuity",
                    action="maintain_continuity",
                    confidence=0.7,
                    justification_path=["narrative:resolution->continuity"],
                    supporting_nodes=sorted(graph_state.temporal_node_ids)[-3:],
                    supporting_edges=[],
                )
            )

        if not candidates:
            candidates.append(
                DecisionCandidate(
                    decision_id="decision:hold",
                    action="maintain_continuity",
                    confidence=0.55,
                    justification_path=["default:insufficient_signal"],
                    supporting_nodes=sorted(graph_state.temporal_node_ids)[-1:],
                    supporting_edges=[],
                )
            )

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)
