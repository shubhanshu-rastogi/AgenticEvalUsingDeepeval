# RAG Eval BDD - Execution Runbook

Operational runbook for testers and demo users.

## 1) Setup

```bash
cd /Users/shubhanshurastogi_1/Learning/rag-session-qa-eval/rag_eval_bdd
source ../.venv/bin/activate
python -m pip install -r requirements.txt

export BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="<your_key>"
```

## 2) Primary execution commands

### A) Smoke unit checks (fast, deterministic)

```bash
make smoke
```

### B) Live evaluation scenarios

```bash
make live
```

### C) Live with notebook-parity mode

```bash
make live-notebook-parity
```

## 3) Tag-based execution (recommended)

Use tag filters to run only required scenarios.

```bash
make run-tags TAGS='@smoke'
make run-tags TAGS='@sanity and @layer1'
make run-tags TAGS='@regression and @layer2'
make run-tags TAGS='@contextual_precision'
```

You can combine tags using `and`, `or`, `not`.

## 4) Tag meaning

- `@smoke`: minimal fast BDD checks
- `@sanity`: confidence checks
- `@regression`: broader checks
- `@layer1`: retrieval/context metrics
- `@layer2`: answer metrics
- Metric tags:
  - `@contextual_precision`
  - `@contextual_recall`
  - `@contextual_relevancy`
  - `@answer_relevancy`
  - `@faithfulness`
  - `@completeness`

## 5) Reporting behavior

### Executive report (current run only)

`results/reports/index.html`

- Auto-generated from the current pytest session.
- Shows business-friendly table and filters.
- Includes a collapsible "Complete Technical Logs" section.

### Trend report (historical)

`results/trends/last5.html`

- Historical dashboard using recent run history.
- Run retention count comes from `config/config.yaml` (`reporting.keep_last_n_runs`).

## 6) How to validate results quickly

1. Open `results/reports/index.html`.
2. Verify only current run rows are displayed.
3. Check Metric Health table and status values.
4. Open `results/trends/last5.html` to review historical graph.
5. Expand technical logs section for full JSON traces if needed.

## 7) Common issues

### Error: Step definition not found

Usually caused by typo/mismatch in Gherkin step text vs `steps/test_eval_steps.py`.

### "collected 0 items"

Use tag filters that match existing scenarios and run via `make run-tags` or `pytest ... steps`.

### Report seems stale

- `index.html` is regenerated for current session.
- `last5.html` is historical and intentionally includes previous runs.

## 8) Key artifact paths

- `results/reports/index.html`
- `results/reports/technical_logs.json`
- `results/trends/last5.html`
- `results/trends/last5.json`
- `results/runs/<run_id>/results.json`
- `results/index.json`
- `results/current_index.json`
