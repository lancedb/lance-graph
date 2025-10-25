"""Command line interface for the Lance-backed knowledge graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import yaml

from . import extraction as kg_extraction
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingGenerator,
    cosine_similarity,
)
from .config import KnowledgeGraphConfig
from .service import LanceKnowledgeGraph
from .store import LanceGraphStore

if TYPE_CHECKING:
    import pyarrow as pa

    from .extractors import ExtractionResult


LOGGER = logging.getLogger(__name__)

DEFAULT_SEED_COUNT = 5
DEFAULT_SEED_NEIGHBOR_LIMIT = 50


def init_graph(config: KnowledgeGraphConfig) -> None:
    """Initialize the on-disk storage and scaffold the schema file."""
    config.ensure_directories()
    LanceGraphStore(config).ensure_layout()
    schema_path = config.resolved_schema_path()
    if schema_path.exists():
        print(f"Schema already present at {schema_path}")
        return

    schema_stub = """\
# Lance knowledge graph schema
#
# Define node labels and relationship mappings. Example:
# nodes:
#   Person:
#     id_field: person_id
# relationships:
#   WORKS_FOR:
#     source: person_id
#     target: company_id
# entity_types:
#   - PERSON
#   - ORGANIZATION
# relationship_types:
#   - WORKS_FOR
#   - PART_OF
nodes: {}
relationships: {}
entity_types: []
relationship_types: []
"""
    schema_path.write_text(schema_stub, encoding="utf-8")
    print(f"Created schema template at {schema_path}")


def list_datasets(config: KnowledgeGraphConfig) -> None:
    """List the Lance datasets available under the configured root."""
    store = LanceGraphStore(config)
    store.ensure_layout()
    datasets = store.list_datasets()
    if not datasets:
        print("No Lance datasets found. Load data or run extraction first.")
        return
    print("Available Lance datasets:")
    for name, path in sorted(datasets.items()):
        print(f"  - {name}: {path}")


def run_interactive(service: LanceKnowledgeGraph) -> None:
    """Enter an interactive shell for issuing Cypher queries."""
    print("Lance Knowledge Graph interactive shell")
    print("Type ':help' for commands, or 'quit' to exit.")

    while True:
        try:
            text = input("kg> ").strip()
        except EOFError:
            print()
            break

        if not text:
            continue
        lowered = text.lower()
        if lowered in {"quit", "exit", "q"}:
            break
        if text.startswith(":"):
            _handle_command(text, service)
            continue

        _execute_query(service, text)


def _handle_command(command: str, service: LanceKnowledgeGraph) -> None:
    """Handle meta-commands in the interactive shell."""
    cmd = command.strip()
    if cmd in {":help", ":h"}:
        print("Commands:")
        print("  :help           Show this message")
        print("  :datasets       List persisted Lance datasets")
        print("  :config         Show the configured node/relationship mappings")
        print("  quit/exit/q     Leave the shell")
        return
    if cmd in {":datasets", ":ls"}:
        list_datasets(service.store.config)
        return
    if cmd in {":config", ":schema"}:
        _print_config_summary(service)
        return
    print(f"Unknown command: {command}")


def _print_config_summary(service: LanceKnowledgeGraph) -> None:
    """Print a brief summary of the graph configuration."""
    config = service.config
    # GraphConfig does not currently expose direct iterators; rely on repr.
    print("Graph configuration:")
    print(f"  {config!r}")


def _execute_query(service: LanceKnowledgeGraph, statement: str) -> None:
    """Execute a single Cypher statement and print results."""
    try:
        result = service.run(statement)
    except Exception as exc:  # pragma: no cover - CLI feedback path
        print(f"Query failed: {exc}", file=sys.stderr)
        return

    _print_table(result)


def _print_table(table: "pa.Table") -> None:
    """Render a PyArrow table in a simple textual format."""
    if table.num_rows == 0:
        print("(no rows)")
        return

    column_names = table.column_names
    columns = [table.column(i).to_pylist() for i in range(len(column_names))]
    widths = []
    for name, values in zip(column_names, columns):
        str_values = ["" if value is None else str(value) for value in values]
        if str_values:
            width = max(len(name), *(len(value) for value in str_values))
        else:
            width = len(name)
        widths.append(width)

    header = " | ".join(name.ljust(width) for name, width in zip(column_names, widths))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)
    for row_values in zip(*columns):
        str_row = ["" if value is None else str(value) for value in row_values]
        line = " | ".join(value.ljust(width) for value, width in zip(str_row, widths))
        print(line)


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


def ask_question(
    question: str,
    service: LanceKnowledgeGraph,
    args: argparse.Namespace,
) -> None:
    """Answer a natural-language question using the graph via LLM-assisted Cypher."""
    client_options = _load_llm_options(args.llm_config)
    llm_client = _create_llm_client(args, options=client_options)
    embedding_generator = _resolve_embedding_generator(args, options=client_options)
    seed_limit = getattr(args, "seed_count", DEFAULT_SEED_COUNT)
    try:
        seed_limit = int(seed_limit)
    except (TypeError, ValueError):
        seed_limit = DEFAULT_SEED_COUNT
    if seed_limit < 0:
        seed_limit = 0

    seed_entities = _find_seed_entities(
        question,
        service,
        embedding_generator,
        limit=seed_limit,
    )
    seed_neighbors = _collect_seed_neighbors(
        service,
        seed_entities,
        limit=DEFAULT_SEED_NEIGHBOR_LIMIT,
    )

    schema_summary = _summarize_schema(service)
    type_hints = service.store.config.type_hints()
    type_hint_lines = _build_type_hint_lines(type_hints)
    query_prompt = _build_query_prompt(
        question,
        schema_summary,
        type_hint_lines,
        type_hints,
        seed_entities,
        seed_neighbors,
    )

    raw_plan = llm_client.complete(query_prompt)
    plan_payload = kg_extraction.parse_llm_json(raw_plan)
    query_plan = _extract_query_plan(plan_payload)

    if not query_plan and not seed_entities:
        print("Unable to generate Cypher queries for the question.")
        return

    execution_results: list[dict[str, Any]] = []
    if seed_entities:
        execution_results.append(
            {
                "cypher": "(semantic search)",
                "description": (
                    "Top seed entities retrieved via embedding similarity search."
                ),
                "rows": seed_entities,
                "truncated": False,
            }
        )
    if seed_neighbors:
        execution_results.append(
            {
                "cypher": "(seed expansion)",
                "description": ("Neighboring entities connected to the seed entities."),
                "rows": seed_neighbors,
                "truncated": bool(
                    DEFAULT_SEED_NEIGHBOR_LIMIT
                    and len(seed_neighbors) >= DEFAULT_SEED_NEIGHBOR_LIMIT
                ),
            }
        )
    if query_plan:
        execution_results.extend(_execute_queries(service, query_plan))

    if not execution_results:
        print("Unable to gather context for the question.")
        return

    answer_prompt = _build_answer_prompt(question, schema_summary, execution_results)
    raw_answer = llm_client.complete(answer_prompt)

    print(raw_answer.strip())


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


def _result_to_dict(result: "ExtractionResult") -> dict[str, list[dict]]:
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


def _find_seed_entities(
    question: str,
    service: LanceKnowledgeGraph,
    embedding_generator: EmbeddingGenerator | None,
    *,
    limit: int = DEFAULT_SEED_COUNT,
) -> list[dict[str, Any]]:
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
    seeds: list[dict[str, Any]] = []
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


def _collect_seed_neighbors(
    service: LanceKnowledgeGraph,
    seed_entities: Sequence[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_SEED_NEIGHBOR_LIMIT,
) -> list[dict[str, Any]]:
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

    neighbors: list[dict[str, Any]] = []
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


def _summarize_schema(
    service: LanceKnowledgeGraph,
    max_columns: int = 20,
    max_value_samples: int = 10,
) -> str:
    type_hints = service.store.config.type_hints()
    lines = []
    for name in service.dataset_names():
        try:
            table = service.load_table(name)
        except Exception:
            continue
        if hasattr(table, "schema"):
            columns = list(getattr(table.schema, "names", []))
        else:
            columns = []
        if len(columns) > max_columns:
            columns = columns[:max_columns] + ["..."]
        column_summary = f"- {name}: {', '.join(columns)}"
        lines.append(column_summary)

        # Provide helpful value samples for well-known columns.
        extras: list[str] = []
        upper_name = name.upper()
        if upper_name == "ENTITY" and type_hints.get("entity_types"):
            allowed = ", ".join(type_hints["entity_types"])
            extras.append(f"allowed entity_type values: {allowed}")
        try:
            if (
                upper_name == "RELATIONSHIP"
                and "relationship_type" in table.column_names
            ):
                values = table.column("relationship_type").to_pylist()
                distinct = sorted({str(value) for value in values if value is not None})
                if distinct:
                    if len(distinct) > max_value_samples:
                        display = ", ".join(distinct[:max_value_samples]) + ", ..."
                    else:
                        display = ", ".join(distinct)
                    extras.append(f"relationship_type values: {display}")
            if upper_name == "RELATIONSHIP" and type_hints.get("relationship_types"):
                allowed = ", ".join(type_hints["relationship_types"])
                extras.append(f"allowed relationship_type values: {allowed}")
        except Exception:
            pass

        lines.extend(f"  {extra}" for extra in extras)
    if not lines:
        return "(no datasets available)"
    return "\n".join(lines)


def _build_type_hint_lines(type_hints: Mapping[str, tuple[str, ...]]) -> list[str]:
    hints: list[str] = []
    entity_types = type_hints.get("entity_types") or ()
    relationship_types = type_hints.get("relationship_types") or ()
    if entity_types:
        hints.append(f"entity_type values: {', '.join(entity_types)}")
    if relationship_types:
        hints.append(f"relationship_type values: {', '.join(relationship_types)}")
    return hints


def _select_example_relationship_type(
    type_hints: Mapping[str, tuple[str, ...]],
) -> str:
    relationship_types = type_hints.get("relationship_types") or ()
    if relationship_types:
        return relationship_types[0]
    return "RELATIONSHIP_TYPE"


def _build_query_prompt(
    question: str,
    schema_summary: str,
    type_hint_lines: list[str],
    type_hints: Mapping[str, tuple[str, ...]],
    seed_entities: Sequence[Mapping[str, Any]] | None = None,
    seed_neighbors: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    example_rel_type = _select_example_relationship_type(type_hints)
    instruction_lines = [
        "You translate questions into Cypher for Lance graph datasets.",
        ("Use the schema summary to craft queries that directly answer the question."),
        (
            "  • Use the schema summary and allowed relationship_type values to "
            "identify candidate relationship directions and types."
        ),
        (
            "  • When the schema lists relationship_type values and the question "
            "does not narrow them down, treat the list as exhaustive and include "
            "every value in your filter using OR clauses or "
            "WHERE rel.relationship_type IN [...]."
        ),
        (
            "Always specify node labels and relationship types in MATCH patterns "
            "that introduce aliases."
        ),
        "Supported constructs include:",
        ("  • MATCH (e:Entity) to scan entity rows (name, name_lower, entity_id)."),
        (
            "  • MATCH (src:Entity)-[rel:RELATIONSHIP]->(dst:Entity) to traverse "
            "relationships (relationship_type column); `src` aligns with "
            "`source_entity_id` and `dst` with `target_entity_id`."
        ),
        (
            "  • Decide which node should be `src` versus `dst` based on the "
            "relationship meaning in the question and schema hints."
        ),
        (
            "  • Map natural language roles (team, person, product, etc.) to the "
            "`entity_type` column so queries filter to the expected entities."
        ),
        "  • Use WHERE e.column = 'value' for node-level filters.",
        (
            "  • Filter relationships with WHERE rel.relationship_type = 'VALUE' "
            "or by comparing rel.source_entity_id / rel.target_entity_id; when the "
            "question does not name a specific relationship type, include every "
            "relevant value from the schema summary using OR clauses or "
            "WHERE rel.relationship_type IN [...], explicitly note which values "
            "you considered, and avoid emitting only a single guessed type."
        ),
        (
            "  • Select columns using the aliases you define, such as e.name or "
            "rel.relationship_type."
        ),
        (
            "  • Avoid inventing relationship datasets; match RELATIONSHIP and "
            "filter rel.relationship_type instead of [:TYPE]."
        ),
        (
            "Example: MATCH (part:Entity)-[rel:RELATIONSHIP]->(whole:Entity) "
            f"WHERE rel.relationship_type = '{example_rel_type}' "
            "RETURN part.name, whole.name."
        ),
        (
            "Example: MATCH (a:Entity)-[rel:RELATIONSHIP]->(b:Entity) WHERE "
            "rel.relationship_type = 'TYPE_A' OR rel.relationship_type = 'TYPE_B' "
            "RETURN a.name, b.name."
        ),
        (
            "Example: MATCH (src:Entity)-[rel:RELATIONSHIP]->(dst:Entity) WHERE "
            "rel.relationship_type IN ['TYPE_A', 'TYPE_B', 'TYPE_C'] "
            "RETURN src.name, dst.name."
        ),
        (
            "Example: MATCH (dst:Entity) WHERE dst.name_lower = 'acme corp' "
            "RETURN dst.name, dst.entity_id."
        ),
        (
            f"Do not use relationship patterns like [:{example_rel_type}]; rely on "
            "rel.relationship_type filters instead."
        ),
        (
            "Always emit at least one query when relevant data exists; only "
            "return [] when it is impossible to answer."
        ),
        "Return ONLY a JSON array where each item has `cypher` and `description`.",
    ]
    if seed_entities:
        instruction_lines.append(
            "Prefer queries that start from the provided seed entities by referencing "
            "their entity_id values before exploring related nodes."
        )
    if seed_neighbors:
        instruction_lines.extend(
            [
                (
                    "Use the provided seed neighbor relationships to decide "
                    "relationship direction."
                ),
                (
                    "  • Each neighbor entry includes a `direction` field: 'outgoing' "
                    "means the seed entity is the relationship source; 'incoming' "
                    "means the seed entity is the target."
                ),
                (
                    "  • Build MATCH patterns accordingly, e.g., outgoing -> "
                    "(seed)-[rel:RELATIONSHIP]->(neighbor); incoming -> "
                    "(neighbor)-[rel:RELATIONSHIP]->(seed)."
                ),
            ]
        )
    instructions = "\n".join(instruction_lines)

    if type_hint_lines:
        hint_block = "\n".join(f"  • {line}" for line in type_hint_lines)
        instructions = "\n".join(
            [
                instructions,
                "Allowed labels and type values:",
                hint_block,
            ]
        )

    prompt_parts = [
        instructions,
        f"Schema summary:\n{schema_summary}",
    ]
    if seed_entities:
        seed_lines = []
        for item in seed_entities:
            similarity = item.get("similarity")
            if isinstance(similarity, (int, float)):
                score = f"{similarity:.3f}"
            else:
                score = "n/a"
            display_name = str(item.get("name") or "(unknown)")
            seed_lines.append(
                (
                    f"- {display_name} "
                    f"(entity_id={item.get('entity_id')}, "
                    f"entity_type={item.get('entity_type')}, similarity={score})"
                )
            )
        prompt_parts.append(
            "Seed entities discovered via embedding similarity:\n"
            + "\n".join(seed_lines)
        )
    if seed_neighbors:
        neighbor_lines: list[str] = []
        for entry in seed_neighbors:
            direction = str(entry.get("direction") or "outgoing")
            seed_name = str(
                entry.get("seed_name") or entry.get("seed_entity_id") or "(seed)"
            )
            neighbor_name = str(
                entry.get("neighbor_name")
                or entry.get("neighbor_entity_id")
                or "(neighbor)"
            )
            rel_type = entry.get("relationship_type") or "RELATIONSHIP"
            description = entry.get("relationship_description") or ""
            seed_id = entry.get("seed_entity_id")
            neighbor_id = entry.get("neighbor_entity_id")
            if direction.lower() == "incoming":
                arrow = f"{neighbor_name} -[{rel_type}]-> {seed_name}"
            else:
                arrow = f"{seed_name} -[{rel_type}]-> {neighbor_name}"
            line = (
                f"- {arrow} (seed_entity_id={seed_id}, "
                f"neighbor_entity_id={neighbor_id}, direction={direction}"
            )
            if description:
                line += f", description={description}"
            line += ")"
            neighbor_lines.append(line)
        prompt_parts.append(
            "Seed neighbor relationships (match patterns to respect direction):\n"
            + "\n".join(neighbor_lines)
        )
    prompt_parts.extend(
        [
            f"Question:\n{question}",
            "JSON:",
        ]
    )
    return "\n\n".join(prompt_parts)


def _extract_query_plan(payload: Any) -> list[dict[str, str]]:
    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("queries") or payload.get("plan") or []
    else:
        return []

    plan: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        cypher = entry.get("cypher") or entry.get("query")
        if not cypher:
            continue
        plan.append(
            {
                "cypher": cypher,
                "description": entry.get("description", ""),
            }
        )
    return plan


def _execute_queries(
    service: LanceKnowledgeGraph, plan: list[dict[str, str]], max_rows: int = 20
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for step in plan:
        cypher = step["cypher"]
        description = step.get("description", "")
        entry: dict[str, Any] = {"cypher": cypher, "description": description}
        try:
            table = service.run(cypher)
            rows = table.to_pylist() if hasattr(table, "to_pylist") else []
            truncated = False
            if isinstance(rows, list) and len(rows) > max_rows:
                truncated = True
                rows = rows[:max_rows]
            entry["rows"] = rows
            entry["truncated"] = truncated
            preview = json.dumps(rows, ensure_ascii=False, indent=2)
            if truncated:
                logging.debug(
                    "Cypher result (truncated to %s rows): %s",
                    max_rows,
                    preview,
                )
            else:
                logging.debug("Cypher result rows: %s", preview)
            logging.debug(
                "Cypher execution",
                extra={"lance_graph": {"cypher": cypher, "rows": rows}},
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            entry["error"] = str(exc)
            data_preview = {}
            for name, table in service.load_tables(service.dataset_names()).items():
                if hasattr(table, "schema"):
                    schema_names = list(table.schema.names)
                else:
                    schema_names = []
                try:
                    row_limit = min(max_rows, getattr(table, "num_rows", 0))
                    sample_rows = table.slice(0, row_limit).to_pylist()
                except Exception:
                    sample_rows = []
                data_preview[name] = {
                    "schema": schema_names,
                    "rows_preview": sample_rows,
                }
            dataset_summary = json.dumps(data_preview, ensure_ascii=False, indent=2)
            logging.debug(
                "Cypher execution error\nCypher: %s\nError: %s\nDatasets: %s",
                cypher,
                str(exc),
                dataset_summary,
            )
        results.append(entry)
    return results


def _build_answer_prompt(
    question: str,
    schema_summary: str,
    query_results: list[dict[str, Any]],
) -> str:
    sections = [
        "You are a graph analysis assistant.",
        "Provide a concise answer using the query results.",
        "If the data is insufficient, state that clearly.",
        "Schema summary:",
        schema_summary,
        "Query results:",
    ]
    for idx, item in enumerate(query_results, 1):
        sections.append(f"Query {idx}: {item['cypher']}")
        if item.get("description"):
            sections.append(f"Description: {item['description']}")
        if "error" in item:
            sections.append(f"Error: {item['error']}")
        else:
            rows_json = json.dumps(item.get("rows", []), ensure_ascii=False, indent=2)
            sections.append(f"Rows: {rows_json}")
            if item.get("truncated"):
                sections.append("(results truncated)")
    sections.append(f"Question: {question}")
    sections.append("Answer:")
    return "\n".join(sections)


def _load_config(args: argparse.Namespace) -> KnowledgeGraphConfig:
    """Derive the configuration object from CLI arguments."""
    if args.root:
        config = KnowledgeGraphConfig.from_root(Path(args.root))
    else:
        config = KnowledgeGraphConfig.default()
    if args.schema:
        config = config.with_schema(Path(args.schema))
    return config


def _load_service(config: KnowledgeGraphConfig) -> LanceKnowledgeGraph:
    """Instantiate the knowledge graph service."""
    graph_config = config.load_graph_config()
    storage = LanceGraphStore(config)
    service = LanceKnowledgeGraph(graph_config, storage=storage)
    service.ensure_initialized()
    return service


def _resolve_extractor(args: argparse.Namespace) -> kg_extraction.BaseExtractor:
    options = _load_llm_options(args.llm_config)
    return kg_extraction.get_extractor(
        args.extractor,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_options=options,
    )


def _resolve_embedding_generator(
    args: argparse.Namespace,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> EmbeddingGenerator | None:
    model = getattr(args, "embedding_model", None)
    model_name = (model or "").strip()
    if not model_name or model_name.lower() == "none":
        return None
    client_options = dict(options or _load_llm_options(args.llm_config))
    try:
        return EmbeddingGenerator(model=model_name, client_options=client_options)
    except RuntimeError as exc:
        LOGGER.warning("Embeddings disabled: %s", exc)
        return None


def _load_llm_options(path: Optional[Path]) -> dict:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("LLM config must be a mapping")
    # Normalize nested http headers key to match OpenAI naming (default_headers)
    if "default_headers" not in data and "http_headers" in data:
        headers = data.pop("http_headers")
        if isinstance(headers, dict):
            data["default_headers"] = headers
    return data


def _create_llm_client(
    args: argparse.Namespace,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> kg_extraction.LLMClient:
    resolved_options = dict(options or _load_llm_options(args.llm_config))
    return kg_extraction.get_llm_client(
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_options=resolved_options,
    )


def _configure_logging(level: str) -> None:
    normalized = level.upper()
    numeric = getattr(logging, normalized, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_graph",
        description="Operate the Lance-backed knowledge graph.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Root directory for Lance datasets (default: ./knowledge_graph_data).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to a YAML file describing node and relationship mappings.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List stored Lance datasets and exit.",
    )
    parser.add_argument(
        "--extractor",
        choices=["heuristic", "llm"],
        default=kg_extraction.DEFAULT_STRATEGY,
        help="Extraction strategy to use (default: llm).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model identifier when using --extractor llm.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for --extractor llm (default: 0.2).",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help=(
            "Optional YAML file with OpenAI client options (api_key, base_url, "
            "headers, etc)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=("OpenAI embedding model for semantic search (set to 'none' to disable)."),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=DEFAULT_SEED_COUNT,
        help=(
            "Maximum number of seed entities to surface from similarity search "
            f"(default: {DEFAULT_SEED_COUNT})."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--init",
        action="store_true",
        help="Initialize the knowledge graph storage.",
    )
    group.add_argument(
        "--extract-preview",
        metavar="INPUT",
        help="Preview extracted knowledge from a file path or raw text.",
    )
    group.add_argument(
        "--extract-and-add",
        metavar="INPUT",
        help="Extract knowledge from a file path or raw text and insert it.",
    )
    group.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask a natural-language question over the knowledge graph.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Execute a single Cypher query against the Lance datasets.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args)
    _configure_logging(args.log_level)

    exclusive_args = any(
        [
            args.init,
            args.extract_preview is not None,
            args.extract_and_add is not None,
            args.ask is not None,
            args.list_datasets,
        ]
    )
    if args.query and exclusive_args:
        parser.error(
            "Query argument cannot be combined with --init/--ask/--extract-* flags."
        )

    if args.init:
        init_graph(config)
        return 0
    if args.list_datasets:
        list_datasets(config)
        return 0
    if args.extract_preview:
        extractor = _resolve_extractor(args)
        preview_extraction(args.extract_preview, extractor)
        return 0
    try:
        service = _load_service(config)
    except FileNotFoundError as exc:
        message = (
            f"{exc}. Run `knowledge_graph --init` or provide a schema with --schema."
        )
        print(message, file=sys.stderr)
        return 1

    if args.extract_and_add:
        extractor = _resolve_extractor(args)
        embedding_generator = _resolve_embedding_generator(args)
        extract_and_add(
            args.extract_and_add,
            service,
            extractor,
            embedding_generator=embedding_generator,
        )
        return 0
    if args.ask:
        ask_question(args.ask, service, args)
        return 0

    if args.query:
        _execute_query(service, args.query)
        return 0

    run_interactive(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
