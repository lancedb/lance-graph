from __future__ import annotations

import argparse
from pathlib import Path

from .. import extraction as kg_extraction
from ..embeddings import DEFAULT_EMBEDDING_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_graph",
        description="Operate the Lance-backed knowledge graph.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Root directory for Lance datasets (default: ./knowledge_graph_data).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Path to a YAML file describing node and relationship mappings.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List stored Lance datasets and exit.",
    )
    parser.add_argument(
        "--extractor",
        choices=["heuristic", "llm"],
        default=kg_extraction.DEFAULT_STRATEGY,
        help="Extraction strategy to use (default: llm).",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model identifier when using --extractor llm.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for --extractor llm (default: 0.2).",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help=(
            "Optional YAML file with OpenAI client options (api_key, base_url, "
            "headers, etc)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=("OpenAI embedding model for semantic search (set to 'none' to disable)."),
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=5,
        help=(
            "Maximum number of seed entities to surface from similarity search "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--init",
        action="store_true",
        help="Initialize the knowledge graph storage.",
    )
    group.add_argument(
        "--extract-preview",
        metavar="INPUT",
        help="Preview extracted knowledge from a file path or raw text.",
    )
    group.add_argument(
        "--extract-and-add",
        metavar="INPUT",
        help="Extract knowledge from a file path or raw text and insert it.",
    )
    group.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask a natural-language question over the knowledge graph.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Execute a single Cypher query against the Lance datasets.",
    )
    return parser
