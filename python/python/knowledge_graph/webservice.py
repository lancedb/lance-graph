"""Standalone FastAPI application for the Lance knowledge graph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

import uvicorn
from fastapi import FastAPI

from .component import KnowledgeGraphComponent

if TYPE_CHECKING:
    from .config import KnowledgeGraphConfig


def create_app(config: Optional["KnowledgeGraphConfig"] = None) -> FastAPI:
    component = KnowledgeGraphComponent(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            component.close()

    app = FastAPI(
        title="Lance Knowledge Graph API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(component.router, prefix="/graph")
    return app


app = create_app()


def main() -> None:
    """Run the web service using uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
