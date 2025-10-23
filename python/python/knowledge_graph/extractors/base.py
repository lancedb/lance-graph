"""Shared dataclasses and helpers for extraction implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str
    context: str
    confidence: float


@dataclass(frozen=True)
class ExtractedRelationship:
    source: str
    target: str
    relationship_type: str
    description: str
    confidence: float


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
