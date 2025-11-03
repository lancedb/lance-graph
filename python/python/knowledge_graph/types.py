"""Internal TypedDicts for knowledge graph CLI orchestration."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class SeedEntity(TypedDict, total=False):
    entity_id: str
    name: NotRequired[str | None]
    entity_type: NotRequired[str | None]
    similarity: NotRequired[float]
    context: NotRequired[str | None]


class SeedNeighbor(TypedDict, total=False):
    seed_entity_id: str
    seed_name: NotRequired[str | None]
    seed_entity_type: NotRequired[str | None]
    neighbor_entity_id: str
    neighbor_name: NotRequired[str | None]
    neighbor_entity_type: NotRequired[str | None]
    relationship_type: NotRequired[str | None]
    relationship_description: NotRequired[str | None]
    direction: NotRequired[str]


class PlanStep(TypedDict, total=False):
    cypher: str
    description: NotRequired[str]


class QueryResult(TypedDict, total=False):
    cypher: str
    description: NotRequired[str]
    rows: NotRequired[list[dict[str, Any]]]
    truncated: NotRequired[bool]
    error: NotRequired[str]
