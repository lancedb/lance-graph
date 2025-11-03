"""Prompt and summary builders for LLM-assisted QA over the graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from ..service import LanceKnowledgeGraph


def summarize_schema(
    service: "LanceKnowledgeGraph",
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


def build_type_hint_lines(type_hints: Mapping[str, tuple[str, ...]]) -> list[str]:
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


def build_query_prompt(
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


def build_answer_prompt(
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
            import json as _json

            rows_json = _json.dumps(item.get("rows", []), ensure_ascii=False, indent=2)
            sections.append(f"Rows: {rows_json}")
            if item.get("truncated"):
                sections.append("(results truncated)")
    sections.append(f"Question: {question}")
    sections.append("Answer:")
    return "\n".join(sections)
