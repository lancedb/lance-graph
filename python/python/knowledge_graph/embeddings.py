"""Embedding utilities backed by the OpenAI client."""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, MutableMapping, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._client_options: MutableMapping[str, Any] = dict(client_options or {})
        self._client = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for the provided texts."""
        sanitized = [text for text in texts if text]
        if not sanitized:
            return []

        client = self._ensure_client()
        response = client.embeddings.create(model=self._model, input=sanitized)
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("OpenAI embedding response missing 'data' entries.")

        embeddings: list[list[float]] = []
        for item in data:
            vector = getattr(item, "embedding", None)
            if vector is None and isinstance(item, Mapping):
                vector = item.get("embedding")
            if vector is None:
                raise RuntimeError("OpenAI embedding response missing 'embedding'.")
            try:
                embeddings.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Embedding vector contains non-numeric values."
                ) from exc
        return embeddings

    def embed_one(self, text: str) -> list[float] | None:
        """Return a single embedding for convenience."""
        vectors = self.embed([text])
        return vectors[0] if vectors else None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for embeddings. "
                    "Install it or supply a custom client."
                ) from exc

            sanitized_opts = _sanitize_options(self._client_options)
            LOGGER.debug(
                "Initializing OpenAI embeddings client",
                extra={
                    "lance_graph": {
                        "openai_model": self._model,
                        "openai_options": sanitized_opts,
                    }
                },
            )
            self._client = OpenAI(**self._client_options)
        return self._client


def cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    """Return cosine similarity between two vectors."""
    if len(lhs) != len(rhs):
        LOGGER.debug(
            "Unable to compute cosine similarity due to mismatched lengths: %s vs %s",
            len(lhs),
            len(rhs),
        )
        return 0.0
    dot = sum(x * y for x, y in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(x * x for x in lhs))
    rhs_norm = math.sqrt(sum(y * y for y in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _sanitize_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Strip sensitive values for logging."""
    sanitized: dict[str, Any] = {}
    for key, value in options.items():
        if key.lower() in {"api_key", "api-key", "authorization"}:
            sanitized[key] = "***"
        else:
            sanitized[key] = value
    return sanitized
