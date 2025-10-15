# Examples for lance-graph

## Setup

From repo root, build/install the Python bindings for development:

```bash
maturin develop -m python/Cargo.toml
```

Ensure Python has `pyarrow` available (e.g., `pip install pyarrow`).

### Run with uv (optional)

If you prefer `uv` for isolation without global installs:

```bash
# Build/install the Python bindings (editable) using uvx + maturin
uvx --from maturin maturin develop -m python/Cargo.toml

# Run examples with uv; point PYTHONPATH to the source package
PYTHONPATH=python/python uv run --with pyarrow python examples/basic_cypher.py
PYTHONPATH=python/python uv run --with pyarrow python examples/kg_traversal.py
```

## Examples

- `basic_cypher.py`: simple node filter and projection.

```bash
python examples/basic_cypher.py
```

- `kg_traversal.py`: multi-hop traversal on a small synthetic knowledge graph.

```bash
python examples/kg_traversal.py
```


