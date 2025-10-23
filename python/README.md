# Lance Graph Python Package

This package exposes the Cypher graph query interface that wraps the
`lance-graph` Rust crate. Development uses [uv](https://docs.astral.sh/uv/)
to manage dependencies inside a project-local `.venv`.

## Quick start

```bash
cd python
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install maturin[patchelf]
uv pip install -e '.[tests]'
maturin develop
pytest python/tests/ -v
```

## Development workflow

For linting and type checks:

```bash
# Install dev dependencies
uv pip install -e '.[dev]'

# Run linters and type checker
ruff format python/              # format code
ruff check python/               # lint code
pyright                          # type check

# Run specific tests
pytest python/tests/test_graph.py::test_basic_node_selection -v

# Rebuild extension after Rust changes
maturin develop
```

> If another virtual environment is already active, run `deactivate` (or
> `unset VIRTUAL_ENV`) before invoking `uv run` so uv binds to `.venv`.

## Repository layout

- `python/src/` – PyO3 bridge that exposes graph APIs to Python
- `python/python/lance_graph/` – pure-Python wrapper and `__init__`
- `python/python/knowledge_graph/` – CLI, FastAPI, and extractor utilities built on Lance
- `python/python/tests/` – graph-centric functional tests

Refer to the repository root `README.md` for information about the Rust crate.

> Run CLI commands through `uv run knowledge_graph ...`. The default uses an
> LLM-backed extractor; install the LLM extra with `uv sync --extra llm` (or
> `uv pip install -e '.[llm]'`) and configure `OPENAI_API_KEY`. Install
> `uv sync --extra lance-storage` to enable Lance dataset persistence. Supply
> extra options (e.g., `base_url`, HTTP headers) via `--llm-config`. Use
> `--extractor heuristic` to avoid LLM calls during testing or offline work.
