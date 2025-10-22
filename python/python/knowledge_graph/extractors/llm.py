"""LLM-backed extraction using a configurable text-generation callable."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Sequence

from .base import ExtractedEntity, ExtractedRelationship

LOGGER = logging.getLogger(__name__)


class LLMExtractor:
    """Use a large language model to extract entities and relationships."""

    def __init__(
        self,
        *,
        send_callable: Callable[[str], str] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        client_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._send = send_callable or self._default_send
        self._client_options = dict(client_options or {})

    def generate(self, prompt: str) -> str:
        """Return the raw text output from the LLM for a given prompt."""
        return self._send(prompt)

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        prompt = _entity_prompt(text)
        response = self._send(prompt)
        payload = load_json_payload(response)
        records = _select_records(payload, preferred_key="entities")
        entities: list[ExtractedEntity] = []
        for item in records:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(item.get("type", "TERM"))
            context = str(item.get("description", ""))
            confidence = float(item.get("confidence", 0.5) or 0.5)
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    context=context,
                    confidence=confidence,
                )
            )
        return entities

    def extract_relationships(
        self,
        text: str,
        entities: Sequence[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        if not entities:
            return []
        prompt = _relationship_prompt(text, entities)
        response = self._send(prompt)
        payload = load_json_payload(response)
        records = _select_records(payload, preferred_key="relationships")
        relationships: list[ExtractedRelationship] = []
        for item in records:
            source = str(item.get("source_entity_name", "")).strip()
            target = str(item.get("target_entity_name", "")).strip()
            if not source or not target:
                continue
            rel_type = str(item.get("type", "RELATED_TO"))
            description = str(item.get("description", ""))
            confidence = float(item.get("confidence", 0.5) or 0.5)
            relationships.append(
                ExtractedRelationship(
                    source=source,
                    target=target,
                    relationship_type=rel_type,
                    description=description,
                    confidence=confidence,
                )
            )
        return relationships

    # Default send ---------------------------------------------------------
    def _default_send(self, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for LLM extraction. "
                "Install it or provide `send_callable`."
            ) from exc

        sanitized = _sanitize_options(self._client_options)
        LOGGER.debug(
            "Initializing OpenAI client with options",
            extra={"lance_graph": {"openai_options": sanitized}},
        )

        client = OpenAI(**self._client_options)
        LOGGER.debug(
            "Sending extraction prompt",
            extra={
                "lance_graph": {
                    "openai_model": self._model,
                    "openai_temperature": self._temperature,
                    "openai_base_url": getattr(client, "base_url", None),
                    "openai_headers": sanitized.get("default_headers"),
                }
            },
        )
        response = client.responses.create(
            model=self._model,
            input=prompt,
            temperature=self._temperature,
        )
        if hasattr(response, "output_text"):
            return response.output_text
        if getattr(response, "output", None):
            try:
                chunks = []
                for item in response.output:  # type: ignore[attr-defined]
                    for content in getattr(item, "content", []):
                        text = getattr(content, "text", None)
                        if text and hasattr(text, "value"):
                            chunks.append(text.value)
                if chunks:
                    return "\n".join(chunks)
            except Exception:  # pragma: no cover - defensive fallback
                pass
        if getattr(response, "choices", None):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message and hasattr(message, "content"):
                return message.content  # type: ignore[no-any-return]
        raise RuntimeError("Unexpected response format from OpenAI responses API.")


# Prompt helpers --------------------------------------------------------------
ENTITY_PROMPT_TEMPLATE = (
    "You are an information extraction assistant. Read the text and list entities "
    "that are explicitly mentioned (do not infer unstated companies or people). "
    "Return ONLY a valid JSON array. Each item must include keys `name`, `type`, "
    "`description`, and `confidence` (0.0-1.0). Use types such as PERSON, "
    "ORGANIZATION, TEAM, PROJECT, LOCATION, TECHNOLOGY when appropriate.\n\n"
    "Text:\n"
    "{body}\n\n"
    "JSON:\n"
)


def _entity_prompt(text: str) -> str:
    return ENTITY_PROMPT_TEMPLATE.format(body=text.strip())


RELATIONSHIP_PROMPT_TEMPLATE = (
    "Given the text and the explicitly extracted entities, list relationships that "
    "are directly stated (do not infer unstated connections). Return ONLY a valid "
    "JSON array where each item contains `source_entity_name`, `target_entity_name`, "
    "`type`, `description`, and `confidence` (0.0-1.0).\n\n"
    "Text:\n"
    "{body}\n\n"
    "Entities:\n"
    "{entities}\n\n"
    "JSON:\n"
)


def _relationship_prompt(
    text: str,
    entities: Sequence[ExtractedEntity],
) -> str:
    entity_lines = "\n".join(
        f"- {entity.name} ({entity.entity_type})" for entity in entities
    )
    return RELATIONSHIP_PROMPT_TEMPLATE.format(
        body=text.strip(),
        entities=entity_lines,
    )


def load_json_payload(raw: str) -> Any:
    LOGGER.debug("LLM raw response: %s", raw)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        if len(parts) >= 2:
            cleaned = parts[1]
    if cleaned.startswith("json"):
        cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        LOGGER.debug("Failed to decode LLM response as JSON")
        return []


def _select_records(payload: Any, preferred_key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if preferred_key in payload and isinstance(payload[preferred_key], list):
            return _select_records(payload[preferred_key], preferred_key)
        for value in payload.values():
            if isinstance(value, list):
                records = [item for item in value if isinstance(item, dict)]
                if records:
                    return records
    return []


def _sanitize_options(options: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in options.items():
        if key.lower() in {"api_key", "organization"}:
            sanitized[key] = "***"
        elif key == "default_headers" and isinstance(value, Mapping):
            sanitized[key] = {
                header_key: "***"
                if "authorization" in header_key.lower()
                else header_value
                for header_key, header_value in value.items()
            }
        else:
            sanitized[key] = value
    return sanitized
