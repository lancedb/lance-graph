"""High-level extraction utilities combining heuristic and LLM strategies."""

from __future__ import annotations

from typing import Any, Optional

from .extractors import HeuristicExtractor, LLMExtractor
from .extractors.base import ExtractedEntity, ExtractedRelationship, ExtractionResult
from .extractors.llm import load_json_payload

DEFAULT_STRATEGY = "llm"


def get_extractor(
    strategy: str = DEFAULT_STRATEGY,
    *,
    llm_callable=None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_options: Optional[dict] = None,
) -> "BaseExtractor":
    strategy = strategy.lower()
    if strategy == "heuristic":
        return HeuristicExtractor()
    if strategy == "llm":
        return LLMExtractor(
            send_callable=llm_callable,
            model=llm_model,
            temperature=llm_temperature,
            client_options=llm_options or {},
        )
    raise ValueError(f"Unknown extractor strategy '{strategy}'")


def preview_extraction(
    text: str,
    *,
    extractor: Optional["BaseExtractor"] = None,
    strategy: str = DEFAULT_STRATEGY,
) -> ExtractionResult:
    extractor = extractor or get_extractor(strategy)
    entities = extractor.extract_entities(text)
    relationships = extractor.extract_relationships(text, entities)
    return ExtractionResult(entities=entities, relationships=relationships)


class BaseExtractor:
    """Protocol-like base class for IDE support."""

    def extract_entities(self, text: str) -> list[ExtractedEntity]:  # pragma: no cover
        raise NotImplementedError

    def extract_relationships(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelationship]:  # pragma: no cover
        raise NotImplementedError


class LLMClient:
    def __init__(self, extractor: LLMExtractor):
        self._extractor = extractor

    def complete(self, prompt: str) -> str:
        return self._extractor.generate(prompt)


def get_llm_client(
    *,
    llm_callable=None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    llm_options: Optional[dict] = None,
) -> LLMClient:
    extractor = LLMExtractor(
        send_callable=llm_callable,
        model=llm_model,
        temperature=llm_temperature,
        client_options=llm_options or {},
    )
    return LLMClient(extractor)


def parse_llm_json(raw: str) -> Any:
    return load_json_payload(raw)
