"""Schema loading and deterministic structural validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Graph, GraphEdge, GraphNode


class SchemaError(ValueError):
    """Raised when a graph violates structural schema."""


def load_json_like(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ontology(path: str | Path) -> dict[str, Any]:
    # ontology.yaml is intentionally JSON-compatible YAML for zero-dependency parsing.
    return load_json_like(path)


def load_graph_schema(path: str | Path) -> dict[str, Any]:
    return load_json_like(path)


def _expect_type(value: Any, expected: str, where: str) -> None:
    mapping = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "boolean": bool,
    }
    if expected not in mapping:
        return
    if not isinstance(value, mapping[expected]):
        raise SchemaError(f"{where}: expected {expected}, got {type(value).__name__}")


def _validate_object(obj: dict[str, Any], schema: dict[str, Any], where: str) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in obj:
            raise SchemaError(f"{where}: missing required key '{key}'")

    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extras = set(obj.keys()) - allowed
        if extras:
            raise SchemaError(f"{where}: unexpected keys {sorted(extras)}")

    for key, subschema in schema.get("properties", {}).items():
        if key in obj:
            _validate_json(obj[key], subschema, f"{where}.{key}")


def _validate_array(arr: list[Any], schema: dict[str, Any], where: str) -> None:
    if "minItems" in schema and len(arr) < schema["minItems"]:
        raise SchemaError(f"{where}: expected at least {schema['minItems']} items")
    item_schema = schema.get("items")
    if item_schema:
        for idx, item in enumerate(arr):
            _validate_json(item, item_schema, f"{where}[{idx}]")


def _validate_json(value: Any, schema: dict[str, Any], where: str) -> None:
    if "type" in schema:
        _expect_type(value, schema["type"], where)
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{where}: expected one of {schema['enum']}, got {value}")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise SchemaError(f"{where}: expected minLength {schema['minLength']}")
    if isinstance(value, (int, float)):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaError(f"{where}: expected >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaError(f"{where}: expected <= {schema['maximum']}, got {value}")

    if isinstance(value, dict):
        _validate_object(value, schema, where)
    elif isinstance(value, list):
        _validate_array(value, schema, where)


def validate_graph_structure(graph_dict: dict[str, Any], schema_dict: dict[str, Any]) -> None:
    _validate_json(graph_dict, schema_dict, "graph")


def build_graph(graph_dict: dict[str, Any]) -> Graph:
    nodes = [GraphNode(**node) for node in graph_dict["nodes"]]
    edges = [GraphEdge(**edge) for edge in graph_dict["edges"]]
    return Graph(
        graph_id=graph_dict["graph_id"],
        metadata=graph_dict.get("metadata", {}),
        nodes=nodes,
        edges=edges,
    )
