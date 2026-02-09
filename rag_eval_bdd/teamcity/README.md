# TeamCity Setup Notes

Use command-line build steps (non-interactive):

## Step 1: Create virtual environment

```bash
cd rag_eval_bdd
python3 -m venv .venv-ci
. .venv-ci/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Run deterministic smoke gate

```bash
cd rag_eval_bdd
. .venv-ci/bin/activate
pytest -c pytest.ini tests -m "smoke" -q
```

## Step 3: Run live evaluation (non-blocking recommended)

```bash
cd rag_eval_bdd
. .venv-ci/bin/activate
pytest -c pytest.ini steps -m "live and (layer1 or layer2)" --alluredir=allure-results || true
```

## Step 4: Publish artifacts

Configure TeamCity artifact paths:

- `rag_eval_bdd/allure-results => allure-results`
- `rag_eval_bdd/allure-report => allure-report`
- `rag_eval_bdd/results => eval-results`

If you use an Allure plugin in TeamCity, point it to `rag_eval_bdd/allure-results`.
