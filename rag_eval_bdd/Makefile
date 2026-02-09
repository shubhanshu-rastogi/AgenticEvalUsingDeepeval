PYTHON ?= python3
# Prefer shared repo-level venv if it exists; fallback to local rag_eval_bdd/.venv
VENV ?= $(if $(wildcard ../.venv/bin/python),../.venv,.venv)
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: venv install test smoke live live-notebook-parity live-non-blocking ci test-layer1 test-layer2 report run-layer1 run-layer2 synthesize

venv:
	@if [ ! -x "$(VENV)/bin/python" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -c pytest.ini steps --alluredir=allure-results

smoke:
	$(PYTEST) -c pytest.ini tests -m "smoke" -q --alluredir=allure-results

live:
	$(PYTEST) -c pytest.ini steps -m "live" --alluredir=allure-results

live-notebook-parity:
	RAG_EVAL_NOTEBOOK_PARITY_MODE=1 $(PYTEST) -c pytest.ini steps -m "live" --alluredir=allure-results

live-non-blocking:
	-$(PYTEST) -c pytest.ini steps -m "live" --alluredir=allure-results

ci:
	$(MAKE) smoke
	$(MAKE) live-non-blocking

test-layer1:
	$(PYTEST) -c pytest.ini steps -m "layer1" --alluredir=allure-results

test-layer2:
	$(PYTEST) -c pytest.ini steps -m "layer2" --alluredir=allure-results

report:
	allure generate allure-results --clean -o allure-report

run-layer1:
	$(VENV)/bin/python -m rag_eval_bdd run --tags layer1

run-layer2:
	$(VENV)/bin/python -m rag_eval_bdd run --tags layer2

synthesize:
	$(VENV)/bin/python -m rag_eval_bdd synthesize --input data/source --output data/generated/questions.json --num-questions 50
