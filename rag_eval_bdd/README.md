# RAG Eval BDD Framework

Business-friendly and technical RAG evaluation framework built with `pytest-bdd` + DeepEval.

## What this framework does

- Runs BDD scenarios for Layer 1 (retrieval/context quality) and Layer 2 (answer quality).
- Supports tag-based execution (`@smoke`, `@sanity`, `@regression`, metric tags, layer tags).
- Stores every run as JSON under `results/runs/<run_id>/results.json`.
- Generates two HTML reports:
  - `results/reports/index.html`: current execution session only
  - `results/trends/last5.html`: trend dashboard across recent runs

## Reporting model

### 1) Executive report (current run only)

`results/reports/index.html`

- Shows only scenarios executed in the current pytest session.
- Includes business table columns:
  - Metric
  - RunID
  - TimeStamp
  - Type (Inline/External)
  - Threshold
  - Question
  - Expected Output
  - Actual Output
  - Score
  - Result
  - Reason For Score
  - Technical Logs
- "Complete Technical Logs" section is collapsible and links per-run JSON logs.

### 2) Trend report (historical)

`results/trends/last5.html`

- Shows last N runs trend (N from `config/config.yaml` -> `reporting.keep_last_n_runs`).
- Used for historical health comparison.

## Quick start

```bash
cd /Users/shubhanshurastogi_1/Learning/rag-session-qa-eval/rag_eval_bdd
source ../.venv/bin/activate
python -m pip install -r requirements.txt

export BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="<your_key>"
```

## Core commands

### Deterministic smoke checks (unit/internals)

```bash
make smoke
```

### Live BDD evaluation

```bash
make live
```

### Live BDD evaluation with notebook-parity behavior

```bash
make live-notebook-parity
```

## Tag-based execution

Use this to run exactly what you want.

```bash
make run-tags TAGS='@smoke'
make run-tags TAGS='@sanity and @layer1'
make run-tags TAGS='@regression and @layer2'
make run-tags TAGS='@contextual_precision'
make run-tags TAGS='@unseen and @layer1'
make run-tags TAGS='@unseen and @layer2'
make run-tags TAGS='@unseen'
```

Notes:
- `@tag` expressions are accepted directly.
- Comma-separated tags are treated as OR.
- Layer and metric tags can be combined.
- Runtime unseen mode auto-picks the latest document uploaded in the application UI (`http://localhost:5173`).
- Runtime unseen generation uses `2` questions per layer by default. Override with `RAG_EVAL_UNSEEN_QUESTIONS_PER_LAYER=<N>`.
- Optional fallback (file-path mode): set `RAG_EVAL_UNSEEN_DOCUMENT` and use the env-based upload step.

## Tag conventions

- `@smoke`: minimal fast BDD scenarios
- `@sanity`: fast confidence scenarios
- `@regression`: broader coverage scenarios
- `@layer1`: contextual precision/recall/relevancy scope
- `@layer2`: answer relevancy/faithfulness/completeness scope
- Metric tags (examples):
  - `@contextual_precision`
  - `@contextual_recall`
  - `@contextual_relevancy`
  - `@answer_relevancy`
  - `@faithfulness`
  - `@completeness`
- `@unseen`: runtime-generated dataset from the latest document uploaded in application UI

## Current smoke design

There is one dedicated smoke scenario in each layer feature:

- Layer 1 smoke: one inline row, one metric (`contextual_precision`)
- Layer 2 smoke: one inline row, one metric (`answer_relevancy`)

This keeps smoke runs fast and predictable.

## Data and features

- Features:
  - `features/layer1_context_metrics.feature`
  - `features/layer2_answer_metrics.feature`
- Dataset files:
  - `data/datasets/layer1_questions.json`
  - `data/datasets/layer2_questions.json`
  - Runtime-generated unseen files:
    - `data/generated/layer1_unseen_questions.json`
    - `data/generated/layer2_unseen_questions.json`

## Configuration

Main config file:

- `config/config.yaml`

Important sections:

- `thresholds`: metric pass criteria
- `reporting`:
  - `keep_last_n_runs`
  - `trend_status_pass_rate_rule`
  - `trend_status_min_pass_rate`
- `evaluation`: performance/cost behavior

## Result storage

- `results/runs/<run_id>/results.json`: full run artifact
- `results/index.json`: recent historical run index (for trends)
- `results/current_index.json`: current session run index (for executive report)
- `results/reports/index.html`: current execution session report
- `results/reports/technical_logs.json`: executive report logs payload
- `results/trends/last5.json`: trend data
- `results/trends/last5.html`: trend dashboard

## Troubleshooting

### Why does trend include older runs?

Because `last5.html` is historical by design (reads recent runs index).

### Why does `index.html` show only my latest session?

By design, current session index is reset at pytest session start.

### No tests collected

Make sure you run pytest on `steps/` for BDD scenarios, not directly on `.feature` files.

## CI notes

Recommended gate:

1. `make smoke`
2. `make live` or non-blocking live stage depending on environment

Archive `results/` so both business report and trend artifacts are preserved.
