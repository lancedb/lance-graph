# Lance Graph Python Package

This package exposes the Cypher graph query interface that wraps the
`lance-graph` Rust crate. Development uses [uv](https://docs.astral.sh/uv/)
to manage dependencies inside a project-local `.venv`.

## Quick start

```bash
cd python
uv venv --python 3.11 .venv
uv pip install -e '.[tests]'
uv run --extra tests pytest python/python/tests/test_graph.py
```

For linting and type checks install the optional `dev` extra:

```bash
uv run --extra dev ruff check python
uv run --extra dev pyright
```

> If another virtual environment is already active, run `deactivate` (or
> `unset VIRTUAL_ENV`) before invoking `uv run` so uv binds to `.venv`.

## Repository layout

- `python/src/` – PyO3 bridge that exposes graph APIs to Python
- `python/python/lance_graph/` – pure-Python wrapper and `__init__`
- `python/python/tests/` – graph-centric functional tests

Refer to the repository root `README.md` for information about the Rust crate.
