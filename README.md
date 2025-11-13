# Lance Graph

Lance Graph is a Cypher-capable graph query engine built in Rust with Python bindings for building high-performance, scalable, and serverless multimodal knowledge graphs.

This repository contains:

- `rust/lance-graph` – the Cypher-capable query engine implemented in Rust
- `python/` – PyO3 bindings and Python packages:
  - `lance_graph` – thin wrapper around the Rust query engine
  - `knowledge_graph` – Lance-backed knowledge graph CLI, API, and utilities

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

## Knowledge Graph CLI & API

The `knowledge_graph` package layers a simple Lance-backed knowledge graph
service on top of the `lance_graph` engine. It provides:

- A CLI (`knowledge_graph.main`) for initializing storage, running Cypher
  queries, and bootstrapping data via heuristic text extraction.
- A reusable FastAPI component, plus a standalone web service
  (`knowledge_graph.webservice`) that exposes query and dataset endpoints.
- Storage helpers that persist node and relationship tables as Lance datasets.

### CLI usage

```bash
uv run knowledge_graph --init                    # initialize storage and schema stub
uv run knowledge_graph --list-datasets           # list Lance datasets on disk
uv run knowledge_graph --extract-preview notes.txt
uv run knowledge_graph --extract-preview "Alice joined the graph team"
uv run knowledge_graph --extract-and-add notes.txt
uv run knowledge_graph "MATCH (n) RETURN n LIMIT 5"
uv run knowledge_graph --log-level DEBUG --extract-preview "Inline text"
uv run knowledge_graph --ask "Who is working on the Presto project?"


# Configure LLM extraction (default)
uv sync --extra llm  # install optional LLM dependencies
uv sync --extra lance-storage  # install Lance dataset support
export OPENAI_API_KEY=sk-...
uv run knowledge_graph --llm-model gpt-4o-mini --extract-preview notes.txt

# Supply additional OpenAI client options via YAML (base_url, headers, etc.)
uv run knowledge_graph --llm-config llm_config.yaml --extract-and-add notes.txt

# Fall back to the heuristic extractor when LLM access is unavailable
uv run knowledge_graph --extractor heuristic --extract-preview notes.txt

```

The default extractor uses OpenAI. Configure credentials via environment
variables supported by the SDK (for example `OPENAI_API_BASE` or
`OPENAI_API_KEY`), or place them in a YAML file passed through `--llm-config`.
Override the model and temperature with `--llm-model` and `--llm-temperature`.
```

By default the CLI writes datasets under `./knowledge_graph_data`. Provide
`--root` and `--schema` to point at alternate storage locations and schema YAML.

### FastAPI service

Run the web service after installing the `knowledge_graph` package (and
dependencies such as FastAPI):

```bash
uv run --package knowledge_graph knowledge_graph-webservice
```

The service exposes endpoints under `/graph`, including `/graph/health`,
`/graph/query`, `/graph/datasets`, and `/graph/schema`.

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

## Benchmarks

- Requirements:
  - protoc: install `protobuf-compiler` (Debian/Ubuntu: `sudo apt-get install -y protobuf-compiler`).
  - Optional: gnuplot for Criterion's gnuplot backend; otherwise the plotters backend is used.

- Run (from `rust/lance-graph`):

```bash
cargo bench --bench graph_execution

# Quicker local run (shorter warm-up/measurement):
cargo bench --bench graph_execution -- --warm-up-time 1 --measurement-time 2 --sample-size 10
```

- Reports:
  - Global index: `rust/lance-graph/target/criterion/report/index.html`
  - Group index: `rust/lance-graph/target/criterion/cypher_execution/report/index.html`

- Typical results (x86_64, quick run: warm-up 1s, measurement 2s, sample size 10):

| Benchmark                | Size      | Median time | Approx. throughput |
|--------------------------|-----------|-------------|--------------------|
| basic_node_filter        | 100       | ~680 µs     | ~147 Kelem/s       |
| basic_node_filter        | 10,000    | ~715 µs     | ~13.98 Melem/s     |
| basic_node_filter        | 1,000,000 | ~743 µs     | ~1.35 Gelem/s      |
| single_hop_expand        | 100       | ~2.79 ms    | ~35.9 Kelem/s      |
| single_hop_expand        | 10,000    | ~3.77 ms    | ~2.65 Melem/s      |
| single_hop_expand        | 1,000,000 | ~3.70 ms    | ~270 Melem/s       |
| two_hop_expand           | 100       | ~4.52 ms    | ~22.1 Kelem/s      |
| two_hop_expand           | 10,000    | ~6.41 ms    | ~1.56 Melem/s      |
| two_hop_expand           | 1,000,000 | ~6.16 ms    | ~162 Melem/s       |

Numbers are illustrative; your hardware, compiler, and runtime load will affect results.

## External Wiki

For additional documentation, architecture, and examples, see the DeepWiki page: [DeepWiki — lance-graph](https://deepwiki.com/lancedb/lance-graph)
