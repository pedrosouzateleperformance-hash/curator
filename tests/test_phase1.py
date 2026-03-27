import copy
import json
import unittest
from pathlib import Path

from src import GraphValidator, build_graph, load_graph_schema, load_ontology, validate_graph_structure


ROOT = Path(__file__).resolve().parents[1]


class Phase1ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = load_ontology(ROOT / "ontology.yaml")
        cls.schema = load_graph_schema(ROOT / "graph_schema.json")
        cls.example = json.loads((ROOT / "examples" / "scene_transition.json").read_text(encoding="utf-8"))

    def _validate(self, graph_dict):
        validate_graph_structure(graph_dict, self.schema)
        graph = build_graph(graph_dict)
        validator = GraphValidator(self.ontology)
        return validator, validator.validate(graph), graph

    def test_valid_cut(self):
        validator, report, graph = self._validate(copy.deepcopy(self.example))
        self.assertTrue(report.is_valid)
        self.assertIn("R_MURCH_CUT_SCORE", report.passed_rules)
        explanation = validator.explain_cut(graph, "cut_1")
        self.assertIn("murch_score", explanation)

    def test_continuity_violation(self):
        graph = copy.deepcopy(self.example)
        cut = next(edge for edge in graph["edges"] if edge["id"] == "cut_1")
        cut["attributes"]["continuity"]["action"] = False

        _, report, _ = self._validate(graph)
        self.assertFalse(report.is_valid)
        violated = {v.rule_id for v in report.violations}
        self.assertIn("R_KATZ_CONTINUITY", violated)

    def test_audio_sync_violation(self):
        graph = copy.deepcopy(self.example)
        sync = next(edge for edge in graph["edges"] if edge["id"] == "sync_1")
        sync["attributes"]["sync_score"] = 0.2

        _, report, _ = self._validate(graph)
        self.assertFalse(report.is_valid)
        violated = {v.rule_id for v in report.violations}
        self.assertIn("R_CHION_SYNC", violated)

    def test_narrative_coherence_violation(self):
        graph = copy.deepcopy(self.example)
        cause = next(edge for edge in graph["edges"] if edge["id"] == "cause_1")
        cause["attributes"]["cause_strength"] = 0.1

        _, report, _ = self._validate(graph)
        self.assertFalse(report.is_valid)
        violated = {v.rule_id for v in report.violations}
        self.assertIn("R_NARRATIVE_COHERENCE", violated)


if __name__ == "__main__":
    unittest.main()
