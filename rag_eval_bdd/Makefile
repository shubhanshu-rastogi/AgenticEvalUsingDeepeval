PYTHON ?= python3
# Prefer shared repo-level venv if it exists; fallback to local rag_eval_bdd/.venv
VENV ?= $(if $(wildcard ../.venv/bin/python),../.venv,.venv)
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: venv install test smoke live live-notebook-parity live-non-blocking ci test-layer1 test-layer2 report run-layer1 run-layer2 run-tags synthesize

venv:
	@if [ ! -x "$(VENV)/bin/python" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -c pytest.ini steps

smoke:
	$(PYTEST) -c pytest.ini tests -m "smoke" -q

live:
	$(PYTEST) -c pytest.ini steps -m "live"

live-notebook-parity:
	RAG_EVAL_NOTEBOOK_PARITY_MODE=1 $(PYTEST) -c pytest.ini steps -m "live"

live-non-blocking:
	-$(PYTEST) -c pytest.ini steps -m "live"

ci:
	$(MAKE) smoke
	$(MAKE) live-non-blocking

test-layer1:
	$(PYTEST) -c pytest.ini steps -m "layer1"

test-layer2:
	$(PYTEST) -c pytest.ini steps -m "layer2"

report:
	PYTHONPATH=src $(VENV)/bin/python -m rag_eval_bdd report

run-layer1:
	PYTHONPATH=src $(VENV)/bin/python -m rag_eval_bdd run --tags layer1

run-layer2:
	PYTHONPATH=src $(VENV)/bin/python -m rag_eval_bdd run --tags layer2

run-tags:
	@if [ -z "$(TAGS)" ]; then \
		echo "Usage: make run-tags TAGS='@sanity and @smoke'"; \
		exit 2; \
	fi
	PYTHONPATH=src $(VENV)/bin/python -m rag_eval_bdd run --tags "$(TAGS)"

synthesize:
	PYTHONPATH=src $(VENV)/bin/python -m rag_eval_bdd synthesize --input data/source --output data/generated/questions.json --num-questions 50
