"""Semantic helpers: seed entity search and neighbor collection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..embeddings import EmbeddingGenerator, cosine_similarity

if TYPE_CHECKING:
    from ..service import LanceKnowledgeGraph
    from ..types import SeedEntity, SeedNeighbor

LOGGER = logging.getLogger(__name__)


def find_seed_entities(
    question: str,
    service: "LanceKnowledgeGraph",
    embedding_generator: EmbeddingGenerator | None,
    *,
    limit: int,
) -> list["SeedEntity"]:
    if not embedding_generator:
        return []
    prepared_question = question.strip()
    if not prepared_question:
        return []
    if limit <= 0:
        return []
    if not service.has_dataset("Entity"):
        return []
    try:
        question_vector = embedding_generator.embed_one(prepared_question)
    except Exception as exc:  # pragma: no cover - defensive logging path
        LOGGER.warning("Failed to embed question for semantic search: %s", exc)
        return []
    if not question_vector:
        return []
    try:
        question_vector = [float(value) for value in question_vector]
    except (TypeError, ValueError):
        LOGGER.warning("Question embedding returned non-numeric values.")
        return []
    try:
        entity_table = service.load_table("Entity")
    except Exception as exc:
        LOGGER.warning("Unable to load Entity dataset for semantic search: %s", exc)
        return []
    seeds: list[SeedEntity] = []
    for row in entity_table.to_pylist():
        embedding = row.get("embedding")
        if not isinstance(embedding, (list, tuple)):
            continue
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError):
            continue
        try:
            similarity = float(cosine_similarity(question_vector, vector))
        except Exception:
            similarity = 0.0
        entity_id = row.get("entity_id")
        if not entity_id:
            continue
        seeds.append(
            {
                "entity_id": entity_id,
                "name": row.get("name"),
                "entity_type": row.get("entity_type"),
                "similarity": similarity,
                "context": row.get("context"),
            }
        )
    seeds.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    if limit and len(seeds) > limit:
        seeds = seeds[:limit]
    return seeds


def collect_seed_neighbors(
    service: "LanceKnowledgeGraph",
    seed_entities: Sequence["SeedEntity"],
    *,
    limit: int,
) -> list["SeedNeighbor"]:
    if not seed_entities:
        return []
    if not (service.has_dataset("Entity") and service.has_dataset("RELATIONSHIP")):
        return []
    try:
        entity_rows = service.load_table("Entity").to_pylist()
        relationship_rows = service.load_table("RELATIONSHIP").to_pylist()
    except Exception as exc:
        LOGGER.warning("Unable to load datasets for neighbor expansion: %s", exc)
        return []

    id_to_entity: dict[str, Mapping[str, Any]] = {}
    for entity in entity_rows:
        entity_id = entity.get("entity_id")
        if entity_id:
            id_to_entity[str(entity_id)] = entity

    seed_ids = {
        str(seed.get("entity_id")) for seed in seed_entities if seed.get("entity_id")
    }
    if not seed_ids:
        return []

    neighbors: list[SeedNeighbor] = []
    for relation in relationship_rows:
        source_id = relation.get("source_entity_id")
        target_id = relation.get("target_entity_id")
        if source_id in seed_ids or target_id in seed_ids:
            if source_id in seed_ids:
                direction = "outgoing"
                seed_id = str(source_id)
                neighbor_id = str(target_id) if target_id else ""
            else:
                direction = "incoming"
                seed_id = str(target_id)
                neighbor_id = str(source_id) if source_id else ""
            if not neighbor_id:
                continue
            seed_entity = id_to_entity.get(seed_id, {})
            neighbor_entity = id_to_entity.get(neighbor_id, {})
            neighbors.append(
                {
                    "seed_entity_id": seed_id,
                    "seed_name": seed_entity.get("name"),
                    "seed_entity_type": seed_entity.get("entity_type"),
                    "neighbor_entity_id": neighbor_id,
                    "neighbor_name": neighbor_entity.get("name"),
                    "neighbor_entity_type": neighbor_entity.get("entity_type"),
                    "relationship_type": relation.get("relationship_type"),
                    "relationship_description": relation.get("description"),
                    "direction": direction,
                }
            )
    if not neighbors:
        return []
    neighbors.sort(
        key=lambda item: (
            str(item.get("seed_name") or ""),
            str(item.get("neighbor_name") or ""),
            str(item.get("relationship_type") or ""),
        )
    )
    if limit and len(neighbors) > limit:
        return neighbors[:limit]
    return neighbors
