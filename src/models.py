"""Typed interfaces for graph objects used in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    layer: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    id: str
    type: str
    layer: str
    source: str
    target: str
    weight: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Graph:
    graph_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Violation:
    rule_id: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    violations: list[Violation]
    passed_rules: list[str]
    explanations: list[str]
