"""
Basic Cypher graph query using the lance-graph Python bindings.

Requirements:
- Build/install the Python extension first (from repo root):
  maturin develop -m python/Cargo.toml
- Python deps: pyarrow

Run:
  python examples/basic_cypher.py
"""

from __future__ import annotations

import pyarrow as pa

from lance_graph import GraphConfigBuilder, CypherQuery


def make_people_batch() -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            pa.array(["Alice", "Bob", "Carol", "David", "Eve"], type=pa.string()),
            pa.array([28, 34, 29, 42, 31], type=pa.int32()),
        ],
        names=["person_id", "name", "age"],
    )


def main() -> None:
    config = (
        GraphConfigBuilder()
        .with_node_label("Person", "person_id")
        .build()
    )

    query = (
        CypherQuery("MATCH (n:Person) WHERE n.age > 30 RETURN n.name")
        .with_config(config)
    )

    datasets = {"Person": make_people_batch()}
    result = query.execute(datasets)

    # Convert to a simple Python dict for display
    print(result.to_pydict())


if __name__ == "__main__":
    main()
