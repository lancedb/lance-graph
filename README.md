# Lance Graph

Lance Graph is a Cypher-capable graph query engine built in Rust with Python bindings for building high-performance, scalable, and serverless multimodal knowledge graphs.

This repository contains:

- `rust/lance-graph` – the Cypher-capable query engine implemented in Rust
- `python/` – PyO3 bindings and a thin `lance_graph` Python package

## Prerequisites

- Rust toolchain (1.82 or newer recommended)
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) available on your `PATH`

## Rust crate quick start

```bash
cd rust/lance-graph
cargo check
cargo test
```

## Python package quick start

```bash
cd python
uv venv --python 3.11 .venv      # create the local virtualenv
source .venv/bin/activate         # activate the virtual environment
uv pip install maturin[patchelf] # install build tool
uv pip install -e '.[tests]'     # editable install with test extras
maturin develop                   # build and install the Rust extension
pytest python/tests/ -v          # run the test suite
```

> If another virtual environment is already active, run `deactivate` (or
> `unset VIRTUAL_ENV`) before the `uv run` command so uv binds to `.venv`.

## Python example: Cypher query

```python
import pyarrow as pa
from lance_graph import CypherQuery, GraphConfig

people = pa.table({
    "person_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Carol", "David"],
    "age": [28, 34, 29, 42],
})

config = (
    GraphConfig.builder()
    .with_node_label("Person", "person_id")
    .build()
)

query = (
    CypherQuery("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name, p.age AS age")
    .with_config(config)
)
result = query.execute({"Person": people})
print(result.to_pydict())  # {'name': ['Bob', 'David'], 'age': [34, 42]}
```

### Development workflow

For linting and type checks:

```bash
# Install dev dependencies and run linters
uv pip install -e '.[dev]'
ruff format python/              # format code
ruff check python/               # lint code
pyright                          # type check

# Or run individual tests
pytest python/tests/test_graph.py::test_basic_node_selection -v
```

The Python README (`python/README.md`) contains additional details if you are
working solely on the bindings.
