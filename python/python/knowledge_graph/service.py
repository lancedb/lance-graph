"""High-level Lance knowledge graph orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping, MutableMapping, Optional

from lance_graph import CypherQuery, GraphConfig

from .config import KnowledgeGraphConfig, build_default_graph_config
from .store import LanceGraphStore

if TYPE_CHECKING:
    import pyarrow as pa

    from . import KnowledgeGraph


class LanceKnowledgeGraph:
    """Coordinate Lance datasets, configs, and query execution."""

    def __init__(
        self,
        config: GraphConfig,
        *,
        storage: LanceGraphStore,
    ):
        self._config = config
        self._store = storage

    @property
    def config(self) -> GraphConfig:
        """Expose the active graph configuration."""
        return self._config

    @property
    def store(self) -> LanceGraphStore:
        """Access the underlying Lance dataset store."""
        return self._store

    def dataset_names(self) -> Iterable[str]:
        """Return the known dataset identifiers."""
        return self._store.list_datasets().keys()

    def ensure_initialized(self) -> None:
        """Create any required on-disk structure."""
        self._store.ensure_layout()

    def materialize(self) -> KnowledgeGraph:
        """Materialize an in-memory ``KnowledgeGraph`` view of the datasets."""
        from . import KnowledgeGraph  # Avoid circular import at module load time

        tables = self._store.load_tables()
        return KnowledgeGraph(self._config, tables)

    def has_dataset(self, name: str) -> bool:
        """Return ``True`` when a dataset has been persisted under ``name``."""
        path = self._store.list_datasets().get(name)
        return path is not None and path.exists()

    def load_tables(
        self,
        names: Optional[Iterable[str]] = None,
    ) -> Mapping[str, "pa.Table"]:
        """Load persisted datasets as PyArrow tables."""
        return self._store.load_tables(names)

    def load_table(self, name: str) -> "pa.Table":
        """Load a single dataset by name."""
        tables = self._store.load_tables([name])
        return tables[name]

    def write_tables(
        self,
        tables: Mapping[str, "pa.Table"],
    ) -> None:
        """Persist a batch of tables."""
        self._store.write_tables(tables)

    def upsert_table(
        self,
        name: str,
        table: "pa.Table",
        *,
        merge: bool = True,
    ) -> None:
        """Insert or replace a dataset, merging with the existing table if requested."""
        import pyarrow as pa

        self.ensure_initialized()
        new_rows = table.to_pylist()
        if merge and name in self._store.list_datasets():
            existing = self.load_table(name)
            existing_rows = existing.to_pylist()
            combined_rows = _normalize_rows(name, existing_rows + new_rows)
        else:
            combined_rows = _normalize_rows(name, new_rows)
        if combined_rows:
            table = pa.Table.from_pylist(combined_rows)
        else:
            table = pa.Table.from_arrays([], schema=table.schema)
        self._store.write_tables({name: table})

    def upsert_tables(
        self,
        tables: Mapping[str, "pa.Table"],
        *,
        merge: bool = True,
    ) -> None:
        """Insert multiple datasets, merging each with existing data when requested."""
        for name, table in tables.items():
            self.upsert_table(name, table, merge=merge)

    def run(
        self,
        statement: str,
        *,
        datasets: Optional[Mapping[str, pa.Table]] = None,
    ) -> pa.Table:
        """Execute a Cypher statement against Lance datasets."""
        query = CypherQuery(statement).with_config(self._config)
        base_tables: MutableMapping[str, "pa.Table"] = dict(self._store.load_tables())
        if datasets:
            base_tables.update(datasets)
        return query.execute(base_tables)

    def query(
        self,
        statement: str,
        *,
        datasets: Optional[Mapping[str, "pa.Table"]] = None,
    ) -> "pa.Table":
        """Alias for :meth:`run` to match the semantic service naming."""
        return self.run(statement, datasets=datasets)


def create_default_service(
    config: Optional[KnowledgeGraphConfig] = None,
    *,
    graph_config: Optional[GraphConfig] = None,
) -> LanceKnowledgeGraph:
    """Construct a knowledge graph service with default settings."""
    config = config or KnowledgeGraphConfig.default()
    storage = LanceGraphStore(config)
    if graph_config is None:
        graph_config = build_default_graph_config()
    return LanceKnowledgeGraph(graph_config, storage=storage)


def _normalize_rows(name: str, rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    upper = name.upper()
    if upper == "ENTITY":
        dedupe = {}
        for row in rows:
            entity_id = row.get("entity_id")
            if not entity_id:
                continue
            row["name_lower"] = str(row.get("name", "")).lower()
            row["entity_type"] = row.get("entity_type") or row.get("type") or "UNKNOWN"
            dedupe[entity_id] = row
        normalized = list(dedupe.values())
    elif upper == "RELATIONSHIP":
        dedupe = {}
        for row in rows:
            source = row.get("source_entity_id")
            target = row.get("target_entity_id")
            if not source or not target:
                continue
            relationship_type = (
                row.get("relationship_type") or row.get("type") or "RELATED_TO"
            )
            key = (
                source,
                target,
                relationship_type,
                row.get("description"),
            )
            row["relationship_type"] = relationship_type
            dedupe[key] = row
        normalized = list(dedupe.values())
    else:
        normalized = rows
    return normalized
