"""Graph validator for ontology rules in Phase 1."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Graph, GraphEdge, ValidationReport, Violation


class GraphValidator:
    def __init__(self, ontology: dict[str, Any]):
        self.ontology = ontology

    def validate(self, graph: Graph) -> ValidationReport:
        violations: list[Violation] = []
        passed_rules: list[str] = []
        explanations: list[str] = []

        for rule_fn in (
            self._rule_layer_separation,
            self._rule_murch_cut_score,
            self._rule_katz_continuity,
            self._rule_chion_sync,
            self._rule_narrative_coherence,
        ):
            rule_id, rule_violations, rule_explanations = rule_fn(graph)
            explanations.extend(rule_explanations)
            if rule_violations:
                violations.extend(rule_violations)
            else:
                passed_rules.append(rule_id)

        return ValidationReport(
            is_valid=not violations,
            violations=violations,
            passed_rules=passed_rules,
            explanations=explanations,
        )

    def explain_cut(self, graph: Graph, cut_edge_id: str) -> str:
        edge = next((e for e in graph.edges if e.id == cut_edge_id and e.type == "CUT"), None)
        if not edge:
            return f"CUT edge '{cut_edge_id}' not found."
        score, breakdown = self._compute_murch_score(edge)
        threshold = self.ontology["constraints"]["murch_min_cut_score"]
        continuity = edge.attributes.get("continuity", {})
        logical = edge.attributes.get("katz_logical_inference")
        return (
            f"CUT {cut_edge_id}: murch_score={score:.3f} (threshold={threshold:.2f}, breakdown={breakdown}), "
            f"continuity={continuity}, katz_logical_inference={logical}."
        )

    def _node_index(self, graph: Graph) -> dict[str, Any]:
        return {node.id: node for node in graph.nodes}

    def _rule_layer_separation(self, graph: Graph):
        rule_id = "R_LAYER_SEPARATION"
        violations: list[Violation] = []
        explanations: list[str] = []
        node_types = self.ontology["node_types"]
        edge_types = self.ontology["edge_types"]
        nodes = self._node_index(graph)

        for node in graph.nodes:
            ndef = node_types.get(node.type)
            if not ndef:
                violations.append(Violation(rule_id, f"Unknown node type {node.type}", {"node_id": node.id}))
                continue
            if node.layer != ndef["layer"]:
                violations.append(
                    Violation(rule_id, "Node layer mismatch", {"node_id": node.id, "expected": ndef["layer"], "actual": node.layer})
                )
            for req in ndef.get("required_attributes", []):
                if req not in node.attributes:
                    violations.append(Violation(rule_id, f"Missing node attribute '{req}'", {"node_id": node.id}))
            for key, allowed in ndef.get("allowed_values", {}).items():
                if node.attributes.get(key) not in allowed:
                    violations.append(
                        Violation(rule_id, f"Node attribute '{key}' out of allowed set", {"node_id": node.id, "allowed": allowed})
                    )

        for edge in graph.edges:
            edef = edge_types.get(edge.type)
            if not edef:
                violations.append(Violation(rule_id, f"Unknown edge type {edge.type}", {"edge_id": edge.id}))
                continue
            if edge.layer not in edef["layer_scope"]:
                violations.append(
                    Violation(rule_id, "Edge layer mismatch", {"edge_id": edge.id, "allowed": edef["layer_scope"], "actual": edge.layer})
                )
            src = nodes.get(edge.source)
            tgt = nodes.get(edge.target)
            if not src or not tgt:
                violations.append(Violation(rule_id, "Edge references missing node", {"edge_id": edge.id}))
                continue
            if src.type not in edef["source_types"]:
                violations.append(Violation(rule_id, "Invalid edge source type", {"edge_id": edge.id, "source_type": src.type}))
            if tgt.type not in edef["target_types"]:
                violations.append(Violation(rule_id, "Invalid edge target type", {"edge_id": edge.id, "target_type": tgt.type}))
            for req in edef.get("required_attributes", []):
                if req not in edge.attributes:
                    violations.append(Violation(rule_id, f"Missing edge attribute '{req}'", {"edge_id": edge.id}))

        explanations.append(f"{rule_id}: checked {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return rule_id, violations, explanations

    def _compute_murch_score(self, edge: GraphEdge):
        weights = self.ontology["weights"]["murch"]
        scores = edge.attributes.get("murch_scores", {})
        breakdown = {k: round(scores.get(k, 0.0) * w, 4) for k, w in weights.items()}
        return sum(breakdown.values()), breakdown

    def _rule_murch_cut_score(self, graph: Graph):
        rule_id = "R_MURCH_CUT_SCORE"
        violations: list[Violation] = []
        explanations: list[str] = []
        threshold = self.ontology["constraints"]["murch_min_cut_score"]
        for edge in graph.edges:
            if edge.type != "CUT":
                continue
            score, breakdown = self._compute_murch_score(edge)
            explanations.append(f"{rule_id}:{edge.id} score={score:.3f} threshold={threshold:.3f} breakdown={breakdown}")
            if score < threshold:
                violations.append(
                    Violation(rule_id, "Murch weighted cut score below threshold", {"edge_id": edge.id, "score": score, "threshold": threshold})
                )
        return rule_id, violations, explanations

    def _rule_katz_continuity(self, graph: Graph):
        rule_id = "R_KATZ_CONTINUITY"
        violations: list[Violation] = []
        explanations: list[str] = []
        for edge in graph.edges:
            if edge.type != "CUT":
                continue
            continuity = edge.attributes.get("continuity", {})
            required_flags = {
                "spatial": continuity.get("spatial") is True,
                "temporal": continuity.get("temporal") is True,
                "action": continuity.get("action") is True,
                "logical": edge.attributes.get("katz_logical_inference") is True,
            }
            explanations.append(f"{rule_id}:{edge.id} flags={required_flags}")
            if not all(required_flags.values()):
                violations.append(Violation(rule_id, "Katz continuity requirements failed", {"edge_id": edge.id, "flags": required_flags}))
        return rule_id, violations, explanations

    def _rule_chion_sync(self, graph: Graph):
        rule_id = "R_CHION_SYNC"
        violations: list[Violation] = []
        explanations: list[str] = []
        sync_min = self.ontology["constraints"]["audio_sync_min"]
        dialogue_min = self.ontology["constraints"]["dialogue_prominence_min"]
        for edge in graph.edges:
            if edge.type != "AUDIO_IMAGE_ALIGNMENT":
                continue
            sync = edge.attributes.get("sync_score", 0.0)
            dialogue = edge.attributes.get("dialogue_prominence", 0.0)
            alignment = edge.attributes.get("alignment_score", 0.0)
            explanations.append(
                f"{rule_id}:{edge.id} sync={sync:.3f}/{sync_min:.3f} dialogue={dialogue:.3f}/{dialogue_min:.3f} alignment={alignment:.3f}"
            )
            if sync < sync_min or dialogue < dialogue_min or not (0.0 <= alignment <= 1.0):
                violations.append(
                    Violation(
                        rule_id,
                        "Chion sync constraints failed",
                        {
                            "edge_id": edge.id,
                            "sync_score": sync,
                            "sync_min": sync_min,
                            "dialogue_prominence": dialogue,
                            "dialogue_min": dialogue_min,
                            "alignment_score": alignment,
                        },
                    )
                )
        return rule_id, violations, explanations

    def _rule_narrative_coherence(self, graph: Graph):
        rule_id = "R_NARRATIVE_COHERENCE"
        violations: list[Violation] = []
        explanations: list[str] = []
        cause_min = self.ontology["constraints"]["narrative_cause_min"]
        nodes = self._node_index(graph)

        for edge in graph.edges:
            if edge.type == "NARRATIVE_CAUSAL":
                cause = edge.attributes.get("cause_strength", 0.0)
                src_t = nodes[edge.source].attributes.get("time_index")
                tgt_t = nodes[edge.target].attributes.get("time_index")
                temporal_ok = src_t is None or tgt_t is None or src_t <= tgt_t
                explanations.append(f"{rule_id}:{edge.id} cause={cause:.3f}/{cause_min:.3f} temporal_ok={temporal_ok}")
                if cause < cause_min or not temporal_ok:
                    violations.append(
                        Violation(
                            rule_id,
                            "Narrative causal coherence failed",
                            {"edge_id": edge.id, "cause_strength": cause, "cause_min": cause_min, "source_time": src_t, "target_time": tgt_t},
                        )
                    )
            elif edge.type == "NARRATIVE_TEMPORAL":
                delta_t = edge.attributes.get("delta_t", -1)
                src_t = nodes[edge.source].attributes.get("time_index")
                tgt_t = nodes[edge.target].attributes.get("time_index")
                temporal_ok = src_t is None or tgt_t is None or src_t <= tgt_t
                explanations.append(f"{rule_id}:{edge.id} delta_t={delta_t} temporal_ok={temporal_ok}")
                if delta_t < 0 or not temporal_ok:
                    violations.append(
                        Violation(
                            rule_id,
                            "Narrative temporal coherence failed",
                            {"edge_id": edge.id, "delta_t": delta_t, "source_time": src_t, "target_time": tgt_t},
                        )
                    )
        return rule_id, violations, explanations


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "is_valid": report.is_valid,
        "violations": [asdict(v) for v in report.violations],
        "passed_rules": report.passed_rules,
        "explanations": report.explanations,
    }
