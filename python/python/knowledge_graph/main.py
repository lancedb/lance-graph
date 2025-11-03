"""Command line interface for the Lance-backed knowledge graph."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional, Sequence

from . import extraction as kg_extraction
from .cli.ingest import extract_and_add, preview_extraction
from .cli.interactive import _execute_query, list_datasets, run_interactive
from .cli.parser import build_parser
from .cli.runtime import configure_logging as cli_configure_logging
from .cli.runtime import load_config as cli_load_config
from .cli.runtime import load_service as cli_load_service
from .store import LanceGraphStore

if TYPE_CHECKING:
    import argparse

    from .config import KnowledgeGraphConfig
    from .service import LanceKnowledgeGraph


LOGGER = logging.getLogger(__name__)


def init_graph(config: "KnowledgeGraphConfig") -> None:
    """Initialize the on-disk storage and scaffold the schema file."""
    config.ensure_directories()
    LanceGraphStore(config).ensure_layout()
    schema_path = config.resolved_schema_path()
    if schema_path.exists():
        print(f"Schema already present at {schema_path}")
        return

    schema_stub = """\
# Lance knowledge graph schema
#
# Define node labels and relationship mappings. Example:
# nodes:
#   Person:
#     id_field: person_id
# relationships:
#   WORKS_FOR:
#     source: person_id
#     target: company_id
# entity_types:
#   - PERSON
#   - ORGANIZATION
# relationship_types:
#   - WORKS_FOR
#   - PART_OF
nodes: {}
relationships: {}
entity_types: []
relationship_types: []
"""
    schema_path.write_text(schema_stub, encoding="utf-8")
    print(f"Created schema template at {schema_path}")


def ask_question(
    question: str,
    service: "LanceKnowledgeGraph",
    args: "argparse.Namespace",
) -> None:
    """Answer a natural-language question using the graph via LLM-assisted Cypher."""
    from .llm.qa import ask_question as qa_ask

    answer = qa_ask(
        question,
        service,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_config_path=args.llm_config,
        embedding_model=getattr(args, "embedding_model", None),
    )
    print(answer)


def _load_config(args: "argparse.Namespace") -> "KnowledgeGraphConfig":
    return cli_load_config(args)


def _load_service(config: "KnowledgeGraphConfig"):
    return cli_load_service(config)


def _resolve_extractor(args: "argparse.Namespace") -> kg_extraction.BaseExtractor:
    from .llm.llm_utils import load_llm_options

    options = load_llm_options(args.llm_config)
    return kg_extraction.get_extractor(
        args.extractor,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_options=options,
    )


def _resolve_embedding_generator(
    args: "argparse.Namespace",
    *,
    options: Optional[dict] = None,
):
    from .llm.llm_utils import resolve_embedding_generator

    model = getattr(args, "embedding_model", None)
    return resolve_embedding_generator(model_name=model, options=options)


def _configure_logging(level: str) -> None:
    cli_configure_logging(level)


def _build_parser() -> "argparse.ArgumentParser":
    return build_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args)
    _configure_logging(args.log_level)

    exclusive_args = any(
        [
            args.init,
            args.extract_preview is not None,
            args.extract_and_add is not None,
            args.ask is not None,
            args.list_datasets,
        ]
    )
    if args.query and exclusive_args:
        parser.error(
            "Query argument cannot be combined with --init/--ask/--extract-* flags."
        )

    if args.init:
        init_graph(config)
        return 0
    if args.list_datasets:
        list_datasets(config)
        return 0
    if args.extract_preview:
        extractor = _resolve_extractor(args)
        preview_extraction(args.extract_preview, extractor)
        return 0
    try:
        service = _load_service(config)
    except FileNotFoundError as exc:
        message = (
            f"{exc}. Run `knowledge_graph --init` or provide a schema with --schema."
        )
        print(message, file=sys.stderr)
        return 1

    if args.extract_and_add:
        extractor = _resolve_extractor(args)
        embedding_generator = _resolve_embedding_generator(args)
        extract_and_add(
            args.extract_and_add,
            service,
            extractor,
            embedding_generator=embedding_generator,
        )
        return 0
    if args.ask:
        ask_question(args.ask, service, args)
        return 0

    if args.query:
        _execute_query(service, args.query)
        return 0

    run_interactive(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
