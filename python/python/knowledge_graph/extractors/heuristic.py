"""Rule-based extraction that identifies simple entity candidates."""

from __future__ import annotations

import re

from .base import ExtractedEntity, ExtractedRelationship


class HeuristicExtractor:
    """Simplified extractor that tags title-cased words as entity candidates."""

    _word_pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_]+\b")

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        seen: set[str] = set()
        entities: list[ExtractedEntity] = []

        for match in self._word_pattern.finditer(text):
            word = match.group()
            if len(word) <= 2:
                continue
            if word.upper() == word:
                continue  # Skip acronyms such as SQL.
            if word in seen:
                continue
            seen.add(word)
            entities.append(
                ExtractedEntity(
                    name=word,
                    entity_type="TERM",
                    context=_surrounding_context(text, match.start(), match.end()),
                    confidence=0.4,
                )
            )
        return entities

    def extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        del text, entities
        return []


def _surrounding_context(text: str, start: int, end: int, radius: int = 50) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].strip()
    return snippet.replace("\n", " ")
