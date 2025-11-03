"""Tests for explain_datafusion API."""

import pyarrow as pa
import pytest
from lance_graph import CypherQuery, GraphConfig


@pytest.fixture
def person_data():
    """Create simple Person dataset for testing."""
    people_table = pa.table(
        {
            "person_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "David"],
            "age": [28, 34, 29, 42],
        }
    )

    config = GraphConfig.builder().with_node_label("Person", "person_id").build()

    return config, people_table


def test_explain_simple_query(person_data):
    """Test explain output contains all expected sections."""
    config, people = person_data
    query = CypherQuery("MATCH (p:Person) RETURN p.name, p.age").with_config(config)
    plan = query.explain_datafusion({"Person": people})

    # Verify the plan is a non-empty string
    assert isinstance(plan, str)
    assert len(plan) > 0

    # Verify it contains expected sections
    assert "Cypher Query:" in plan
    assert "MATCH (p:Person) RETURN p.name, p.age" in plan
    assert "graph_logical_plan" in plan
    assert "logical_plan" in plan
    assert "physical_plan" in plan

    # Verify table format
    assert "+" in plan and "|" in plan


def test_explain_with_clauses(person_data):
    """Test explain output includes query clauses (WHERE, ORDER BY, LIMIT)."""
    config, people = person_data
    query = CypherQuery(
        "MATCH (p:Person) WHERE p.age > 30 RETURN p.name ORDER BY p.age LIMIT 2"
    ).with_config(config)
    plan = query.explain_datafusion({"Person": people})

    assert isinstance(plan, str)
    assert "WHERE p.age > 30" in plan
    assert "ORDER BY" in plan
    assert "LIMIT" in plan


def test_explain_error_handling(person_data):
    """Test explain error handling for missing config and datasets."""
    config, people = person_data

    # Missing config
    query_no_config = CypherQuery("MATCH (p:Person) RETURN p.name")
    with pytest.raises(ValueError, match="Graph configuration is required"):
        query_no_config.explain_datafusion({"Person": people})

    # Missing datasets
    query_with_config = CypherQuery("MATCH (p:Person) RETURN p.name").with_config(
        config
    )
    with pytest.raises(ValueError, match="No input datasets provided"):
        query_with_config.explain_datafusion({})
