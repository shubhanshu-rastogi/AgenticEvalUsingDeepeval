# Jenkins E2E Pipeline Runbook

This pipeline is designed for demo-app commit validation with a smoke gate and business-friendly HTML reporting.

## Outcome

- New commit triggers Jenkins automatically.
- Smoke gate runs.
- Build fails if smoke fails.
- Reports are published in Jenkins UI:
  - Executive report (current run)
  - Trend dashboard (last 5)
- Reports are emailed (if recipients are configured).

## Jenkinsfile

- Path: `rag_eval_bdd/jenkins/Jenkinsfile`

## Required Jenkins plugins

- Pipeline
- Git + GitHub integration
- Credentials Binding
- HTML Publisher
- Email Extension

## Required credentials

- Secret text credential:
  - ID: `openai-api-key`
  - Value: OpenAI API key used by DeepEval metric execution

## Job setup (recommended)

Use a **Multibranch Pipeline** pointed to the demo app repository.

### Trigger from GitHub commits

1. In Jenkins multibranch job:
   - Enable GitHub webhook trigger (or equivalent branch source scanning).
2. In GitHub repo webhook:
   - Payload URL: `https://<jenkins-url>/github-webhook/`
   - Content type: `application/json`
   - Events: `Push` and `Pull request`

## Merge gate behavior (important)

To enforce \"merge only after smoke pass\", configure branch protection in GitHub:

1. Repository settings -> Branches -> Branch protection rule.
2. Require status checks to pass before merging.
3. Select this Jenkins job status check.
4. Optionally require PR branch up-to-date before merge.

This is the correct way to guarantee merge gating; Jenkins build result becomes the gate signal.

## Pipeline parameters

- `SMOKE_TAGS`: marker expression for BDD smoke gate, default `@smoke`
- `BASE_URL`: demo app backend URL used by evaluation
- `EMAIL_RECIPIENTS`: comma-separated email list (blank skips email)
- `RUN_UNIT_SMOKE`: whether to run deterministic unit smoke first

## Reports produced

- `rag_eval_bdd/results/reports/index.html` (current run only)
- `rag_eval_bdd/results/trends/last5.html` (historical last-5 trend)
- `rag_eval_bdd/results/reports/technical_logs.json`

The pipeline also archives all `rag_eval_bdd/results/**` artifacts.

## Email output

When `EMAIL_RECIPIENTS` is set:

- Email includes build status summary.
- Attaches:
  - `index.html`
  - `last5.html`
  - `technical_logs.json`
  - compressed bundle `report_bundle_<build>.tar.gz`

## Visuals in Jenkins

You get clear stage visualization (good for demos) with these stages:

1. Checkout
2. Setup Python
3. Smoke Gate (Deterministic Unit)
4. Smoke Gate (BDD @smoke)
5. Publish HTML In Jenkins
6. Prepare Email Bundle
7. Email Reports

For best UI experience, open the job with Blue Ocean or Pipeline Stage View.
