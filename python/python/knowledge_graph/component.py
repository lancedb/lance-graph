"""Reusable FastAPI component for the Lance knowledge graph service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pyarrow as pa
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import KnowledgeGraphConfig
from .service import LanceKnowledgeGraph
from .store import LanceGraphStore


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rows: List[Dict[str, Any]]
    row_count: int


class DatasetUpsertRequest(BaseModel):
    records: List[Dict[str, Any]]
    merge: bool = True


class KnowledgeGraphComponent:
    """Bundle FastAPI routes that expose the Lance knowledge graph."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self._config = config or KnowledgeGraphConfig.default()
        self._service: Optional[LanceKnowledgeGraph] = None
        self.router = APIRouter(tags=["knowledge-graph"])
        self._setup_routes()

    def _get_service(self) -> LanceKnowledgeGraph:
        if self._service is None:
            try:
                self._service = _create_service(self._config)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return self._service

    def _setup_routes(self) -> None:
        @self.router.get("/health")
        async def health() -> Dict[str, str]:
            return {"status": "healthy", "service": "lance-knowledge-graph"}

        @self.router.get("/datasets")
        async def list_datasets() -> Dict[str, List[str]]:
            service = self._get_service()
            names = list(service.dataset_names())
            return {"datasets": names}

        @self.router.get("/datasets/{name}")
        async def get_dataset(name: str, limit: int = 100) -> Dict[str, Any]:
            service = self._get_service()
            if not service.has_dataset(name):
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{name}' not found"
                )

            table = service.load_table(name)
            rows = table.to_pylist()
            if limit is not None:
                rows = rows[:limit]
            return {"name": name, "row_count": len(rows), "rows": rows}

        @self.router.post("/datasets/{name}")
        async def upsert_dataset(
            name: str, request: DatasetUpsertRequest
        ) -> Dict[str, Any]:
            if not request.records:
                raise HTTPException(status_code=400, detail="records cannot be empty")

            table = pa.Table.from_pylist(request.records)
            service = self._get_service()
            service.upsert_table(name, table, merge=request.merge)
            return {"status": "ok", "dataset": name, "row_count": table.num_rows}

        @self.router.post("/query", response_model=QueryResponse)
        async def execute_query(request: QueryRequest) -> QueryResponse:
            service = self._get_service()
            result = service.query(request.query)
            rows = result.to_pylist()
            return QueryResponse(rows=rows, row_count=len(rows))

        @self.router.get("/schema")
        async def get_schema() -> Dict[str, Any]:
            schema_path = self._config.resolved_schema_path()
            if not schema_path.exists():
                raise HTTPException(status_code=404, detail="Schema file not found")
            with schema_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            return {"path": str(schema_path), "schema": payload}

    def close(self) -> None:
        """Release retained resources."""
        self._service = None


def _create_service(config: KnowledgeGraphConfig) -> LanceKnowledgeGraph:
    graph_config = config.load_graph_config()
    storage = LanceGraphStore(config)
    service = LanceKnowledgeGraph(graph_config, storage=storage)
    service.ensure_initialized()
    return service
