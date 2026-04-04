# Contributing to tqk

## Local development

```bash
git clone https://github.com/RemizovDenis/tqk-llm.git
cd tqk-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,transformers]"
```

## Required checks before PR

```bash
ruff check tqk tests
ruff format --check tqk tests
mypy tqk --strict --ignore-missing-imports --show-error-context --show-error-codes --python-version 3.11
pytest tests/ -v --tb=short
python3 verify_quality.py
```

## Pull request rules

1. Branch from `main`.
2. Keep changes focused.
3. Use conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
4. Ensure CI is green.
