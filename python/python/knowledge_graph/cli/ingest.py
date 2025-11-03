"""Extraction preview and ingest helpers for the knowledge graph CLI."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .. import extraction as kg_extraction

if TYPE_CHECKING:
    from ..embeddings import EmbeddingGenerator
    from ..service import LanceKnowledgeGraph

LOGGER = logging.getLogger(__name__)


def preview_extraction(source: str, extractor: kg_extraction.BaseExtractor) -> None:
    """Preview extracted knowledge from a text source or inline text."""
    text = _resolve_text_input(source)
    result = kg_extraction.preview_extraction(text, extractor=extractor)
    print(json.dumps(_result_to_dict(result), indent=2))


def extract_and_add(
    source: str,
    service: LanceKnowledgeGraph,
    extractor: kg_extraction.BaseExtractor,
    *,
    embedding_generator: EmbeddingGenerator | None = None,
) -> None:
    """Extract knowledge and append it to the backing graph."""
    import pyarrow as pa

    text = _resolve_text_input(source)
    result = kg_extraction.preview_extraction(text, extractor=extractor)
    entity_rows, name_to_id = _prepare_entity_rows(
        result.entities, embedding_generator=embedding_generator
    )
    relationships = result.relationships

    if not entity_rows and not relationships:
        print("No candidate entities or relationships detected.")
        return

    if entity_rows:
        entity_table = pa.Table.from_pylist(entity_rows)
        service.upsert_table("Entity", entity_table, merge=True)
        message = f"Upserted {entity_table.num_rows} entity rows into dataset 'Entity'."
        print(message)

    relationship_rows = _prepare_relationship_rows(
        relationships,
        name_to_id,
        embedding_generator=embedding_generator,
    )
    if relationship_rows:
        rel_table = pa.Table.from_pylist(relationship_rows)
        service.upsert_table("RELATIONSHIP", rel_table, merge=True)
        message = (
            "Upserted "
            f"{rel_table.num_rows} relationship rows into dataset "
            "'RELATIONSHIP'."
        )
        print(message)


def _resolve_text_input(raw: str) -> str:
    """Load text from a file if it exists, otherwise treat the string as content."""
    candidate = Path(raw)
    if candidate.exists():
        if candidate.is_dir():
            raise IsADirectoryError(f"Expected text file, got directory: {candidate}")
        return candidate.read_text(encoding="utf-8")
    return raw


def _ensure_dict(item: object) -> dict:
    if is_dataclass(item):
        return asdict(item)  # type: ignore[arg-type]
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported extraction item type: {type(item)!r}")


def _result_to_dict(result: "kg_extraction.ExtractionResult") -> dict[str, list[dict]]:
    return {
        "entities": [asdict(entity) for entity in result.entities],
        "relationships": [asdict(rel) for rel in result.relationships],
    }


def _prepare_entity_rows(
    entities: list[Any],
    *,
    embedding_generator: EmbeddingGenerator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    name_to_id: dict[str, str] = {}
    for entity in entities:
        payload = _ensure_dict(entity)
        name = str(payload.get("name", "")).strip()
        entity_type = str(
            payload.get("entity_type") or payload.get("type") or ""
        ).strip()
        if not name:
            continue
        base = f"{name}|{entity_type}".encode("utf-8")
        entity_id = hashlib.md5(base).hexdigest()
        payload["entity_id"] = entity_id
        payload["entity_type"] = entity_type or "UNKNOWN"
        payload["name_lower"] = name.lower()
        rows.append(payload)
        name_to_id.setdefault(name.lower(), entity_id)
    if embedding_generator and rows:
        _assign_embeddings(
            rows,
            embedding_generator,
            _format_entity_embedding_input,
        )
    return rows, name_to_id


def _prepare_relationship_rows(
    relationships: list[Any],
    name_to_id: dict[str, str],
    *,
    embedding_generator: EmbeddingGenerator | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in relationships:
        payload = _ensure_dict(relation)
        source_name = str(
            payload.get("source_entity_name") or payload.get("source") or ""
        ).strip()
        target_name = str(
            payload.get("target_entity_name") or payload.get("target") or ""
        ).strip()
        source_id = name_to_id.get(source_name.lower())
        target_id = name_to_id.get(target_name.lower())
        if not (source_id and target_id):
            continue
        payload["source_entity_id"] = source_id
        payload["target_entity_id"] = target_id
        payload["relationship_type"] = (
            payload.get("relationship_type") or payload.get("type") or "RELATED_TO"
        )
        payload.setdefault("source_entity_name", source_name)
        payload.setdefault("target_entity_name", target_name)
        rows.append(payload)
    if embedding_generator and rows:
        _assign_embeddings(
            rows,
            embedding_generator,
            _format_relationship_embedding_input,
        )
    return rows


def _assign_embeddings(
    rows: list[dict[str, Any]],
    embedding_generator: EmbeddingGenerator,
    formatter: Callable[[Mapping[str, Any]], str],
) -> None:
    texts: list[str] = []
    indices: list[int] = []
    for idx, row in enumerate(rows):
        text = formatter(row)
        if text:
            texts.append(text)
            indices.append(idx)
    if not texts:
        return
    try:
        vectors = embedding_generator.embed(texts)
    except Exception as exc:  # pragma: no cover - defensive logging path
        LOGGER.warning("Failed to generate embeddings: %s", exc)
        return
    if len(vectors) != len(indices):
        LOGGER.warning(
            "Mismatch between embedding count and row count: expected %s, got %s",
            len(indices),
            len(vectors),
        )
        return
    for idx, vector in zip(indices, vectors):
        rows[idx]["embedding"] = vector


def _format_entity_embedding_input(row: Mapping[str, Any]) -> str:
    name = str(row.get("name", "")).strip()
    entity_type = str(row.get("entity_type", "")).strip()
    context = str(row.get("context", "")).strip()
    pieces = []
    if name:
        pieces.append(name)
    if entity_type:
        pieces.append(f"Type: {entity_type}")
    if context:
        pieces.append(f"Context: {context}")
    return " | ".join(pieces)


def _format_relationship_embedding_input(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_entity_name") or row.get("source") or "").strip()
    target = str(row.get("target_entity_name") or row.get("target") or "").strip()
    relationship_type = str(row.get("relationship_type", "")).strip()
    description = str(row.get("description", "")).strip()
    core: list[str] = []
    if source or target:
        if relationship_type:
            core.append(f"{source} -[{relationship_type}]-> {target}".strip())
        else:
            core.append(f"{source} -> {target}".strip())
    if description:
        core.append(f"Description: {description}")
    return " | ".join(part for part in core if part)
