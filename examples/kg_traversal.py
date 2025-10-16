"""
Simple knowledge-graph multi-hop traversal with lance-graph Python bindings.

Requirements:
- Build/install the Python extension first (from repo root):
  maturin develop -m python/Cargo.toml
- Python deps: pyarrow

Run:
  python examples/kg_traversal.py
"""

from __future__ import annotations

import pyarrow as pa

from lance_graph import GraphConfigBuilder, CypherQuery


def make_people_batch(n: int = 6) -> pa.RecordBatch:
    return pa.record_batch(
        [
            pa.array(list(range(1, n + 1)), type=pa.int32()),
            pa.array([f"P{i}" for i in range(1, n + 1)], type=pa.string()),
        ],
        names=["person_id", "name"],
    )


def make_friendship_batch(n: int = 6) -> pa.RecordBatch:
    # Create a simple ring: 1->2->3->...->n->1
    src = list(range(1, n + 1))
    dst = [i + 1 if i < n else 1 for i in src]
    return pa.record_batch(
        [pa.array(src, type=pa.int32()), pa.array(dst, type=pa.int32())],
        names=["person1_id", "person2_id"],
    )


def main() -> None:
    config = (
        GraphConfigBuilder()
        .with_node_label("Person", "person_id")
        .with_relationship("FRIEND_OF", "person1_id", "person2_id")
        .build()
    )

    # Two-hop traversal from a person to friend-of-a-friend
    query = (
        CypherQuery(
            "MATCH (a:Person)-[:FRIEND_OF]->(b:Person)-[:FRIEND_OF]->(c:Person) RETURN a.name, c.name"
        )
        .with_config(config)
    )

    datasets = {
        "Person": make_people_batch(),
        "FRIEND_OF": make_friendship_batch(),
    }
    result = query.execute(datasets)
    print(result.to_pydict())


if __name__ == "__main__":
    main()
