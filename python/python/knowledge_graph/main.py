"""Command line interface for the knowledge_graph helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def init_graph() -> None:
    """Initialize storage for the knowledge graph."""
    pass


def run_interactive() -> None:
    """Enter an interactive shell for issuing commands."""
    pass


def execute_query(text: str) -> None:
    """Execute a single knowledge graph query."""
    del text  # placeholder until implementation


def preview_extraction(path: Path) -> None:
    """Preview extracted knowledge from a text source."""
    del path  # placeholder until implementation


def extract_and_add(path: Path) -> None:
    """Extract knowledge and append it to the backing graph."""
    del path  # placeholder until implementation


def ask_question(question: str) -> None:
    """Answer a natural-language question using the graph."""
    del question  # placeholder until implementation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_graph",
        description="Operate the Lance-backed knowledge graph.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--init",
        action="store_true",
        help="Initialize the knowledge graph storage.",
    )
    group.add_argument(
        "--extract-preview",
        metavar="PATH",
        help="Preview extracted entities and relations from a text file.",
    )
    group.add_argument(
        "--extract-and-add",
        metavar="PATH",
        help="Extract and insert knowledge from a text file.",
    )
    group.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask a natural-language question over the knowledge graph.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Execute a single Cypher or semantic query.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    exclusive_args = any(
        [
            args.init,
            args.extract_preview is not None,
            args.extract_and_add is not None,
            args.ask is not None,
        ]
    )
    if args.query and exclusive_args:
        parser.error("Query argument cannot be combined with flags.")

    if args.init:
        init_graph()
        return 0
    if args.extract_preview:
        preview_extraction(Path(args.extract_preview))
        return 0
    if args.extract_and_add:
        extract_and_add(Path(args.extract_and_add))
        return 0
    if args.ask:
        ask_question(args.ask)
        return 0
    if args.query:
        execute_query(args.query)
        return 0

    run_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
