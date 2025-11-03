"""LLM and embedding helpers for the knowledge graph CLI and QA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional

import yaml

from .. import extraction as kg_extraction
from ..embeddings import EmbeddingGenerator

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def load_llm_options(path: Optional["Path"]) -> dict:
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


def create_llm_client(
    *,
    llm_model: str,
    llm_temperature: float,
    llm_options: Optional[Mapping[str, Any]] = None,
) -> kg_extraction.LLMClient:
    resolved_options = dict(llm_options or {})
    return kg_extraction.get_llm_client(
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_options=resolved_options,
    )


def resolve_embedding_generator(
    *,
    model_name: str | None,
    options: Optional[Mapping[str, Any]] = None,
) -> EmbeddingGenerator | None:
    model = (model_name or "").strip()
    if not model or model.lower() == "none":
        return None
    client_options = dict(options or {})
    try:
        return EmbeddingGenerator(model=model, client_options=client_options)
    except RuntimeError as exc:
        LOGGER.warning("Embeddings disabled: %s", exc)
        return None
