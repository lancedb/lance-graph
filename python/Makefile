UV ?= uv
PYTHON ?= 3.11
PYTEST_ARGS ?= -vv -s

export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: venv install test lint clean

venv:
	$(UV) venv --python $(PYTHON) .venv

install:
	$(UV) pip install --project . --editable . --extra tests

test:
	$(UV) run --project . --with tests pytest $(PYTEST_ARGS) python/python/tests/test_graph.py

lint:
	$(UV) run --project . --with dev ruff check python
	$(UV) run --project . --with dev ruff format --check python
	$(UV) run --project . --with dev pyright

clean:
	rm -rf .venv
	rm -rf target
