# Python Examples for lance-graph

## Setup

From repo root, build/install the Python bindings for development:

```bash
maturin develop -m python/Cargo.toml
```

Ensure Python has `pyarrow` available (e.g., `pip install pyarrow`).

## Examples

- `basic_cypher.py`: simple node filter and projection.

```bash
python python/examples/basic_cypher.py
```

- `kg_traversal.py`: multi-hop traversal on a small synthetic knowledge graph.

```bash
python python/examples/kg_traversal.py
```


