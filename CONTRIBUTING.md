# Contributing to DocsQuery

## Running Python scripts

The `scripts/` directory is a Python package because it contains
`scripts/__init__.py`.

Run package scripts using Python's module syntax:

```bash
python -m scripts.evaluate_retrieval
python -m scripts.analyze_retrieval
python -m scripts.benchmark_summary

## Continuous Integration

DocsQuery uses GitHub Actions to validate changes automatically.

### Pull Request CI

Every pull request targeting `main` runs:

```bash
ruff check .
ruff format --check .
pytest tests/unit -v
```

The pull request should not be merged when any of these checks fail.

### Full RAG Evaluation

The full retrieval and Gemini evaluation is intentionally separate from normal pull-request CI because it depends on:

* Qdrant
* downloaded ML models
* Gemini API credentials
* the evaluation corpus

It can be started manually from the GitHub Actions interface using:

```text
DocsQuery Evaluation
```

The evaluation workflow generates:

```text
data/evaluation/evaluation_run.json
```

The artifact contains the retrieval and answer evaluation results together with evaluation metadata.

### Local CI-equivalent check

Before pushing a change, run:

```bash
make check
```

and verify:

```bash
pytest tests/unit -v
ruff check .
ruff format --check .
```

A successful local check should closely match the deterministic portion of GitHub CI.
