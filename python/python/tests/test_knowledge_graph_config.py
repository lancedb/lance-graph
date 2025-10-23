import pyarrow as pa
import pytest
from knowledge_graph.config import build_graph_config_from_mapping
from lance_graph import CypherQuery


def test_build_graph_config_from_mapping_supports_simple_nodes():
    mapping = {"nodes": {"Person": "person_id"}}
    config = build_graph_config_from_mapping(mapping)

    table = pa.table(
        {
            "person_id": [1, 2],
            "name": ["Alice", "Bob"],
        }
    )

    data = (
        CypherQuery("MATCH (p:Person) RETURN p.person_id AS id")
        .with_config(config)
        .execute({"Person": table})
        .to_pydict()
    )
    assert data["person_id"] == [1, 2]


def test_build_graph_config_from_mapping_requires_id_field():
    with pytest.raises(ValueError):
        build_graph_config_from_mapping({"nodes": {"Person": {}}})
