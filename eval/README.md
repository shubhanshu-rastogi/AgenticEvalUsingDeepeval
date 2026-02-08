# Evaluation (DeepEval)

This folder contains retrieval and output quality evaluation utilities and notebooks.

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r eval/requirements.txt
```

Set environment variables:

```bash
export OPENAI_API_KEY=your_key
# Optional: publish results to Confident AI
export DEEPEVAL_CONFIDENT_API_KEY=your_confident_key
```

## Scripts

- `eval/tests/layer1_retrieval_eval.py`
- `eval/tests/layer2_output_eval.py`

Both export helper functions:

```python
from eval.tests.layer1_retrieval_eval import evaluate_layer1
from eval.tests.layer2_output_eval import evaluate_layer2
```

## Notebooks

- `eval/notebooks/01_end_to_end_demo.ipynb`
- `eval/notebooks/02_batch_golden_eval.ipynb`
- `eval/notebooks/03_full_metrics_eval.ipynb`

These notebooks call the FastAPI backend to upload and query documents, then run DeepEval metrics.

By default they point to `eval/sample_docs/Match_Summary.pdf`. Set `SAMPLE_FILE` to override.

## Optional publishing

If `DEEPEVAL_CONFIDENT_API_KEY` is set, the functions will attempt to log in and publish results when `publish=True`.
