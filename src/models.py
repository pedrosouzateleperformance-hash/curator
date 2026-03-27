"""Compatibility re-exports for graph domain entities."""

from src.domain.graph import Graph, GraphEdge, GraphNode, ValidationReport, Violation

__all__ = ["GraphNode", "GraphEdge", "Graph", "Violation", "ValidationReport"]
