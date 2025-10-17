# Repository Guidelines

## Project Structure & Module Organization
- `rust/lance-graph/` hosts the Rust Cypher engine; keep new modules under `src/` and co-locate helpers inside `query/` or feature-specific submodules.
- `python/src/` contains the PyO3 bridge; `python/python/lance_graph/` holds the pure-Python facade and packaging metadata.
- `python/python/tests/` stores functional tests; mirror new features with targeted cases here and in the corresponding Rust module.
- `examples/` demonstrates Cypher usage; update or add examples when introducing new public APIs.

## Build, Test, and Development Commands
- `cargo check` / `cargo test --all` (run inside `rust/lance-graph`) validate Rust code paths.
- `cargo bench --bench graph_execution` measures performance-critical changes; include shortened runs with `--warm-up-time 1`.
- `uv venv --python 3.11 .venv` and `uv pip install -e '.[tests]'` bootstrap the Python workspace.
- `maturin develop` rebuilds the extension after Rust edits; `pytest python/python/tests/ -v` exercises Python bindings.
- `make lint` (in `python/`) runs `ruff`, formatting checks, and `pyright`.

## Coding Style & Naming Conventions
- Format Rust with `cargo fmt --all`; keep modules and functions snake_case, types PascalCase, and reuse `snafu` error patterns.
- Run `cargo clippy --all-targets --all-features` to catch lint regressions.
- Use 4-space indentation in Python; maintain snake_case modules, CamelCase classes, and type-annotated public APIs.
- Apply `ruff format python/` before committing; `ruff check` and `pyright` enforce import hygiene and typing.

## Testing Guidelines
- Add Rust unit tests alongside implementations via `#[cfg(test)]`; prefer focused scenarios over broad integration.
- Python tests belong in `python/python/tests/`; name files `test_*.py` and use markers (`gpu`, `cuda`, `integration`, `slow`) consistently.
- When touching performance-sensitive code, capture representative `cargo bench` or large-table pytest timing notes in the PR.

## Commit & Pull Request Guidelines
- Follow the existing history style (`feat(graph):`, `docs:`, `refactor(query):`), using imperative, ≤72-character subjects.
- Reference issues or discussions when relevant and include brief context in the body.
- PRs should describe scope, list test commands run, mention benchmark deltas when applicable, and highlight impacts on bindings or examples.
