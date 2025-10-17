"""Executable module wrapper for `python -m knowledge_graph`."""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
