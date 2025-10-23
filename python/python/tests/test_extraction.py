import json

import pytest
from knowledge_graph.extraction import preview_extraction
from knowledge_graph.extractors import HeuristicExtractor, LLMExtractor


@pytest.mark.parametrize(
    "text",
    [
        "Alice collaborates with Bob on GraphAI",
        "Carol joined DataCorp alongside David",
    ],
)
def test_heuristic_extractor_identifies_title_case_entities(text):
    extractor = HeuristicExtractor()
    entities = extractor.extract_entities(text)
    names = {entity.name for entity in entities}
    assert any(name in names for name in {"Alice", "Bob", "Carol", "David"})


def test_preview_extraction_returns_entities_and_relationships():
    result = preview_extraction("Alice works with Bob", strategy="heuristic")
    assert result.entities
    assert isinstance(result.entities[0].name, str)
    assert isinstance(result.relationships, list)


def test_llm_extractor_can_be_injected_with_callable():
    entities_payload = json.dumps(
        [
            {
                "name": "Alice",
                "type": "PERSON",
                "description": "Engineer",
                "confidence": 0.9,
            }
        ]
    )
    relationships_payload = json.dumps(
        [
            {
                "source_entity_name": "Alice",
                "target_entity_name": "Bob",
                "type": "WORKS_WITH",
                "description": "Alice works with Bob",
                "confidence": 0.8,
            }
        ]
    )

    def fake_send(prompt: str) -> str:
        if "source_entity_name" in prompt:
            return relationships_payload
        return entities_payload

    extractor = LLMExtractor(send_callable=fake_send)
    result = preview_extraction(
        "Alice works with Bob",
        extractor=extractor,
    )
    assert len(result.entities) == 1
    assert result.entities[0].name == "Alice"
    assert result.relationships[0].relationship_type == "WORKS_WITH"
