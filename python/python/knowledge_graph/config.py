"""Configuration helpers for Lance-backed knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from lance_graph import GraphConfig


@dataclass(slots=True)
class KnowledgeGraphConfig:
    """Root configuration for the Lance-backed knowledge graph."""

    storage_path: Path
    schema_path: Optional[Path] = None
    default_dataset: Optional[str] = None
    entity_types: tuple[str, ...] = field(default_factory=tuple)
    relationship_types: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        default_dataset: Optional[str] = None,
    ) -> KnowledgeGraphConfig:
        """Create a configuration anchored at ``root``."""
        schema_path = root / "graph.yaml"
        return cls(
            storage_path=root,
            schema_path=schema_path,
            default_dataset=default_dataset,
        )

    @classmethod
    def default(cls) -> KnowledgeGraphConfig:
        """Use a storage folder relative to the current working directory."""
        return cls.from_root(Path.cwd() / "knowledge_graph_data")

    def resolved_schema_path(self) -> Path:
        """Return the expected path to the graph schema definition."""
        return self.schema_path or (self.storage_path / "graph.yaml")

    def ensure_directories(self) -> None:
        """Create required directories for persistent storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.resolved_schema_path().parent.mkdir(parents=True, exist_ok=True)

    def with_schema(self, schema_path: Path) -> KnowledgeGraphConfig:
        """Return a copy of the config with an explicit schema path."""
        return KnowledgeGraphConfig(
            storage_path=self.storage_path,
            schema_path=schema_path,
            default_dataset=self.default_dataset,
            entity_types=self.entity_types,
            relationship_types=self.relationship_types,
        )

    def load_graph_config(self) -> GraphConfig:
        """Load the Lance ``GraphConfig`` from the schema document."""
        payload = self._load_schema_payload()
        entity_types = payload.get("entity_types") or []
        relationship_types = payload.get("relationship_types") or []
        self.entity_types = _normalize_type_list(entity_types)
        self.relationship_types = _normalize_type_list(relationship_types)
        if not payload.get("nodes") and not payload.get("relationships"):
            return build_default_graph_config()
        return build_graph_config_from_mapping(payload)

    def type_hints(self) -> dict[str, tuple[str, ...]]:
        """Return configured entity and relationship type hints."""
        # Ensure the latest schema values are loaded.
        try:
            self.load_graph_config()
        except FileNotFoundError:
            pass
        return {
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
        }

    def _load_schema_payload(self) -> Mapping[str, Any]:
        schema_path = self.resolved_schema_path()
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Graph schema configuration not found at {schema_path}"
            )
        with schema_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Graph schema configuration must be a mapping")
        return payload  # type: ignore[return-value]


def build_graph_config_from_mapping(data: Mapping[str, Any]) -> GraphConfig:
    """Create a ``GraphConfig`` instance from a schema mapping."""
    builder = GraphConfig.builder()

    for label, node_spec in data.get("nodes", {}).items():
        if isinstance(node_spec, str):
            builder = builder.with_node_label(label, node_spec)
        else:
            id_field = node_spec.get("id_field")
            if not id_field:
                raise ValueError(f"Node '{label}' is missing 'id_field'")
            builder = builder.with_node_label(label, id_field)

    for rel_type, rel_spec in data.get("relationships", {}).items():
        if not isinstance(rel_spec, Mapping):
            raise ValueError(
                (
                    f"Relationship '{rel_type}' must be a mapping with 'source' and "
                    "'target'"
                )
            )
        source_field = rel_spec.get("source")
        target_field = rel_spec.get("target")
        if not (source_field and target_field):
            raise ValueError(
                f"Relationship '{rel_type}' must define 'source' and 'target' fields"
            )
        builder = builder.with_relationship(rel_type, source_field, target_field)

    return builder.build()


def build_default_graph_config() -> GraphConfig:
    builder = GraphConfig.builder()
    builder = builder.with_node_label("Entity", "entity_id")
    builder = builder.with_relationship(
        "RELATIONSHIP",
        "source_entity_id",
        "target_entity_id",
    )
    return builder.build()


def _normalize_type_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return tuple()
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        trimmed = item.strip()
        if trimmed:
            cleaned.append(trimmed)
    return tuple(cleaned)
