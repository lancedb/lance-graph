"""QA orchestration: planning, execution, and answer synthesis."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .. import extraction as kg_extraction
from .llm_utils import create_llm_client, load_llm_options, resolve_embedding_generator
from .prompts import (
    build_answer_prompt,
    build_query_prompt,
    build_type_hint_lines,
    summarize_schema,
)
from .semantic import collect_seed_neighbors, find_seed_entities

if TYPE_CHECKING:
    from ..service import LanceKnowledgeGraph
    from ..types import PlanStep, QueryResult, SeedEntity, SeedNeighbor

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED_COUNT = 5
DEFAULT_SEED_NEIGHBOR_LIMIT = 50


def ask_question(
    question: str,
    service: LanceKnowledgeGraph,
    *,
    llm_model: str,
    llm_temperature: float,
    llm_config_path,
    embedding_model: str | None,
) -> str:
    client_options = load_llm_options(llm_config_path)
    llm_client = create_llm_client(
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_options=client_options,
    )
    embedding_generator = resolve_embedding_generator(
        model_name=embedding_model, options=client_options
    )

    seed_limit = DEFAULT_SEED_COUNT

    seed_entities: list[SeedEntity] = find_seed_entities(
        question,
        service,
        embedding_generator,
        limit=seed_limit,
    )
    seed_neighbors: list[SeedNeighbor] = collect_seed_neighbors(
        service,
        seed_entities,
        limit=DEFAULT_SEED_NEIGHBOR_LIMIT,
    )

    schema_summary = summarize_schema(service)
    type_hints = service.store.config.type_hints()
    allowed_relationship_types = tuple(
        str(t) for t in (type_hints.get("relationship_types") or ())
    )
    if not allowed_relationship_types:
        discovered = _discover_relationship_types(service)
        allowed_relationship_types = tuple(discovered)
        if discovered:
            LOGGER.debug(
                "Discovered relationship_type values from dataset: %s",
                ", ".join(discovered),
            )
    type_hint_lines = build_type_hint_lines(type_hints)
    query_prompt = build_query_prompt(
        question,
        schema_summary,
        type_hint_lines,
        type_hints,
        seed_entities,
        seed_neighbors,
    )

    raw_plan = llm_client.complete(query_prompt)
    plan_payload = kg_extraction.parse_llm_json(raw_plan)
    query_plan: list[PlanStep] = extract_query_plan(plan_payload)
    if allowed_relationship_types and query_plan:
        query_plan = _normalize_relationship_types(query_plan, allowed_relationship_types)
        # Log normalized cypher for debugging
        for step in query_plan:
            LOGGER.debug("Normalized Cypher: %s", step.get("cypher"))

    if not query_plan and not seed_entities:
        return "Unable to generate Cypher queries for the question."

    execution_results: list[QueryResult] = []
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
        execution_results.extend(execute_queries(service, query_plan))

    if not execution_results:
        return "Unable to gather context for the question."

    answer_prompt = build_answer_prompt(question, schema_summary, execution_results)
    raw_answer = llm_client.complete(answer_prompt)
    return raw_answer.strip()


def extract_query_plan(payload: Any) -> list[PlanStep]:
    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("queries") or payload.get("plan") or []
    else:
        return []

    plan: list[PlanStep] = []
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


def execute_queries(
    service: LanceKnowledgeGraph, plan: list[PlanStep], max_rows: int = 20
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for step in plan:
        cypher = step["cypher"]
        description = step.get("description", "")
        entry: QueryResult = {"cypher": cypher, "description": description}
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


def _normalize_relationship_types(
    plan: list[PlanStep], allowed_types: tuple[str, ...]
) -> list[PlanStep]:
    """Normalize or correct relationship_type filters to allowed canonical values.

    - Case-insensitive normalization for known values
    - If unknown literal is used, replace predicate with IN [allowed_types]
    - Supports patterns with and without the `rel.` alias
    - Cleans IN lists to allowed values; if none remain, falls back to full allowed list
    """
    if not allowed_types:
        return plan
    lowered = {t.lower(): t for t in allowed_types}

    def normalize_literal(value: str) -> tuple[str, bool]:
        k = value.strip().lower()
        if k in lowered:
            return lowered[k], True
        return value, False

    # equality patterns
    eq_pats = [
        re.compile(r"(rel\.relationship_type\s*=\s*)'([^']+)'", flags=re.IGNORECASE),
        re.compile(r"(relationship_type\s*=\s*)'([^']+)'", flags=re.IGNORECASE),
    ]

    def replace_eq(match: re.Match[str]) -> str:
        prefix, literal = match.group(1), match.group(2)
        normalized, known = normalize_literal(literal)
        if known:
            return f"{prefix}'{normalized}'"
        # Unknown literal: broaden to IN of all allowed types
        allowed_list = ", ".join(f"'{t}'" for t in allowed_types)
        # build a neutral predicate using the same LHS
        lhs = prefix[:-1].strip()
        return f"{lhs} IN [{allowed_list}]"

    # IN list pattern, with and without alias
    in_pats = [
        re.compile(
            r"(rel\.relationship_type\s+IN\s*\[)([^\]]+)(\])",
            flags=re.IGNORECASE,
        ),
        re.compile(r"(relationship_type\s+IN\s*\[)([^\]]+)(\])", flags=re.IGNORECASE),
    ]

    def replace_in(match: re.Match[str]) -> str:
        start, inner, end = match.group(1), match.group(2), match.group(3)
        items = re.findall(r"'([^']*)'", inner)
        normalized_items: list[str] = []
        for it in items:
            norm, known = normalize_literal(it)
            if known:
                normalized_items.append(f"'{norm}'")
        if not normalized_items:
            normalized_items = [f"'{t}'" for t in allowed_types]
        return f"{start}{', '.join(normalized_items)}{end}"

    normalized_plan: list[PlanStep] = []
    for step in plan:
        cypher = step.get("cypher", "")
        for pat in eq_pats:
            cypher = pat.sub(replace_eq, cypher)
        for pat in in_pats:
            cypher = pat.sub(replace_in, cypher)
        normalized_step: PlanStep = {
            "cypher": cypher,
            "description": step.get("description", ""),
        }
        normalized_plan.append(normalized_step)
    return normalized_plan


def _discover_relationship_types(service: LanceKnowledgeGraph) -> list[str]:
    """Discover distinct relationship_type values from the dataset as a fallback."""
    try:
        table = service.load_table("RELATIONSHIP")
        if "relationship_type" in table.column_names:
            values = table.column("relationship_type").to_pylist()
            distinct = sorted({str(v) for v in values if v is not None and str(v)})
            return distinct
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Unable to discover relationship types: %s", exc)
    return []
