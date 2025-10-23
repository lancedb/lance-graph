"""High-level helpers for working with Lance-backed knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import pyarrow as pa
from lance_graph import CypherQuery, GraphConfig

try:  # Prefer to import for typing without raising at runtime.
    from lance_graph import GraphConfigBuilder
except ImportError:  # pragma: no cover - builder is available in normal installs.
    GraphConfigBuilder = object  # type: ignore[assignment]

from .component import KnowledgeGraphComponent
from .config import KnowledgeGraphConfig, build_graph_config_from_mapping
from .extraction import (
    DEFAULT_STRATEGY,
    BaseExtractor,
    get_extractor,
    preview_extraction,
)
from .extractors import HeuristicExtractor, LLMExtractor
from .service import LanceKnowledgeGraph, create_default_service
from .store import LanceGraphStore
from .webservice import create_app

TableMapping = Mapping[str, pa.Table]


def _ensure_table(name: str, table: pa.Table) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise TypeError(
            f"Dataset '{name}' must be a pyarrow.Table (got {type(table)!r})"
        )
    return table


@dataclass(frozen=True)
class KnowledgeGraph:
    """Wraps a ``GraphConfig`` alongside the Arrow tables backing it."""

    config: GraphConfig
    _tables: Dict[str, pa.Table]

    def __init__(self, config: GraphConfig, datasets: TableMapping) -> None:
        object.__setattr__(self, "config", config)
        normalized = {
            name: _ensure_table(name, table) for name, table in datasets.items()
        }
        object.__setattr__(self, "_tables", normalized)

    def run(
        self,
        statement: str,
        *,
        datasets: Optional[TableMapping] = None,
    ):
        """Execute a Cypher statement, overriding tables when provided."""
        query = CypherQuery(statement).with_config(self.config)
        sources: Dict[str, pa.Table] = dict(self._tables)
        if datasets:
            sources.update(
                {name: _ensure_table(name, table) for name, table in datasets.items()}
            )
        return query.execute(sources)

    def tables(self) -> Dict[str, pa.Table]:
        """Return a shallow copy of the registered datasets."""
        return dict(self._tables)


class KnowledgeGraphBuilder:
    """Collects nodes, relationships, and datasets before building a graph."""

    def __init__(self) -> None:
        builder = GraphConfig.builder()
        self._builder: GraphConfigBuilder = builder  # type: ignore[annotation-unchecked]
        self._datasets: Dict[str, pa.Table] = {}

    def with_node(
        self,
        label: str,
        primary_key: str,
        table: pa.Table,
    ) -> KnowledgeGraphBuilder:
        """Register a node label and Arrow table."""
        self._builder = self._builder.with_node_label(label, primary_key)
        self._datasets[label] = _ensure_table(label, table)
        return self

    def with_relationship(
        self,
        name: str,
        source_key: str,
        target_key: str,
        table: pa.Table,
    ) -> KnowledgeGraphBuilder:
        """Register a relationship and its underlying table."""
        self._builder = self._builder.with_relationship(name, source_key, target_key)
        self._datasets[name] = _ensure_table(name, table)
        return self

    def with_dataset(self, name: str, table: pa.Table) -> KnowledgeGraphBuilder:
        """Attach arbitrary supporting datasets (e.g., reference tables)."""
        self._datasets[name] = _ensure_table(name, table)
        return self

    def build(self) -> KnowledgeGraph:
        """Materialize the ``KnowledgeGraph`` instance."""
        config = self._builder.build()
        return KnowledgeGraph(config, self._datasets)


__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphConfig",
    "build_graph_config_from_mapping",
    "LanceGraphStore",
    "LanceKnowledgeGraph",
    "create_default_service",
    "KnowledgeGraphComponent",
    "create_app",
    "DEFAULT_STRATEGY",
    "BaseExtractor",
    "get_extractor",
    "preview_extraction",
    "HeuristicExtractor",
    "LLMExtractor",
]
