"""Interactive shell and CLI display helpers for the knowledge graph."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..store import LanceGraphStore

if TYPE_CHECKING:
    from ..config import KnowledgeGraphConfig
    from ..service import LanceKnowledgeGraph

if TYPE_CHECKING:
    import pyarrow as pa


def list_datasets(config: "KnowledgeGraphConfig") -> None:
    """List the Lance datasets available under the configured root."""
    store = LanceGraphStore(config)
    store.ensure_layout()
    datasets = store.list_datasets()
    if not datasets:
        print("No Lance datasets found. Load data or run extraction first.")
        return
    print("Available Lance datasets:")
    for name, path in sorted(datasets.items()):
        print(f"  - {name}: {path}")


def run_interactive(service: "LanceKnowledgeGraph") -> None:
    """Enter an interactive shell for issuing Cypher queries."""
    print("Lance Knowledge Graph interactive shell")
    print("Type ':help' for commands, or 'quit' to exit.")

    while True:
        try:
            text = input("kg> ").strip()
        except EOFError:
            print()
            break

        if not text:
            continue
        lowered = text.lower()
        if lowered in {"quit", "exit", "q"}:
            break
        if text.startswith(":"):
            _handle_command(text, service)
            continue

        _execute_query(service, text)


def _handle_command(command: str, service: "LanceKnowledgeGraph") -> None:
    """Handle meta-commands in the interactive shell."""
    cmd = command.strip()
    if cmd in {":help", ":h"}:
        print("Commands:")
        print("  :help           Show this message")
        print("  :datasets       List persisted Lance datasets")
        print("  :config         Show the configured node/relationship mappings")
        print("  quit/exit/q     Leave the shell")
        return
    if cmd in {":datasets", ":ls"}:
        list_datasets(service.store.config)
        return
    if cmd in {":config", ":schema"}:
        _print_config_summary(service)
        return
    print(f"Unknown command: {command}")


def _print_config_summary(service: "LanceKnowledgeGraph") -> None:
    """Print a brief summary of the graph configuration."""
    config = service.config
    # GraphConfig does not currently expose direct iterators; rely on repr.
    print("Graph configuration:")
    print(f"  {config!r}")


def _execute_query(service: "LanceKnowledgeGraph", statement: str) -> None:
    """Execute a single Cypher statement and print results."""
    try:
        result = service.run(statement)
    except Exception as exc:  # pragma: no cover - CLI feedback path
        print(f"Query failed: {exc}", file=sys.stderr)
        return

    _print_table(result)


def _print_table(table: "pa.Table") -> None:
    """Render a PyArrow table in a simple textual format."""
    if table.num_rows == 0:
        print("(no rows)")
        return

    column_names = table.column_names
    columns = [table.column(i).to_pylist() for i in range(len(column_names))]
    widths = []
    for name, values in zip(column_names, columns):
        str_values = ["" if value is None else str(value) for value in values]
        if str_values:
            width = max(len(name), *(len(value) for value in str_values))
        else:
            width = len(name)
        widths.append(width)

    header = " | ".join(name.ljust(width) for name, width in zip(column_names, widths))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)
    for row_values in zip(*columns):
        str_row = ["" if value is None else str(value) for value in row_values]
        line = " | ".join(value.ljust(width) for value, width in zip(str_row, widths))
        print(line)
