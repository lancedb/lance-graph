# Examples for lance-graph

## How to Run Examples

Pick one of the two approaches:

### 1) Local build (editable) with maturin

```bash
# From the repo root; installs the Python extension in editable mode
pip install pyarrow  # or: uv pip install pyarrow
maturin develop -m python/Cargo.toml

# Run examples
python examples/basic_cypher.py
python examples/kg_traversal.py
```

With `uv` (persistent venv recommended):

```bash
# Create and activate a reusable environment
uv venv .venv && . .venv/bin/activate

# Install dependencies and build editable extension
uv pip install pyarrow
uvx --from maturin maturin develop -m python/Cargo.toml

# Run examples using the venv's Python
python examples/basic_cypher.py
python examples/kg_traversal.py
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

### 3) Single-shot with uv (no install, run from source)

uv creates a fresh isolated environment per run. If you prefer a one-liner without installing the package, point `PYTHONPATH` to the source package and let uv provide `pyarrow`:

```bash
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
