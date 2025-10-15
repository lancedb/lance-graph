# Examples for lance-graph

## How to Run Examples

Pick one of the two approaches:

### 1) Local build (editable) with maturin

```bash
# From the repo root; installs the Python extension in editable mode
maturin develop -m python/Cargo.toml

# Run examples
python examples/basic_cypher.py
python examples/kg_traversal.py
```

With `uv`:

```bash
# Build/install using uv
uvx --from maturin maturin develop -m python/Cargo.toml

# Run examples (uv provides pyarrow; no PYTHONPATH needed after develop)
uv run --with pyarrow python examples/basic_cypher.py
uv run --with pyarrow python examples/kg_traversal.py
```

### 2) Install the package, then run

- From PyPI (when published):

```bash
pip install lance-graph pyarrow
python examples/basic_cypher.py
python examples/kg_traversal.py
```

- From git (editable):

```bash
pip install -e python
python examples/basic_cypher.py
python examples/kg_traversal.py
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
