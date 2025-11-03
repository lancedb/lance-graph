from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from ..config import KnowledgeGraphConfig
from ..service import LanceKnowledgeGraph
from ..store import LanceGraphStore


def load_config(args) -> KnowledgeGraphConfig:
    if args.root:
        config = KnowledgeGraphConfig.from_root(Path(args.root))
    else:
        config = KnowledgeGraphConfig.default()
    if args.schema:
        config = config.with_schema(Path(args.schema))
    return config


def load_service(config: KnowledgeGraphConfig) -> LanceKnowledgeGraph:
    graph_config = config.load_graph_config()
    storage = LanceGraphStore(config)
    service = LanceKnowledgeGraph(graph_config, storage=storage)
    service.ensure_initialized()
    return service


def configure_logging(level: str) -> None:
    normalized = level.upper()
    numeric = getattr(logging, normalized, None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric)


def load_llm_options(path: Optional[Path]) -> dict:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("LLM config must be a mapping")
    if "default_headers" not in data and "http_headers" in data:
        headers = data.pop("http_headers")
        if isinstance(headers, dict):
            data["default_headers"] = headers
    return data
