# Contributing to tqk

We love your help! tqk is a technical project, and we value precision and quality.

## Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RemizovDenis/tqk-llm.git
   cd tqk
   ```

2. **Setup virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,transformers]"
   ```

3. **Verify installation**:
   ```bash
   python3 verify_quality.py
   ```

## Development Workflow

- **Linting**: We use `ruff`. Run `ruff check .` and `ruff format .` before committing.
- **Testing**: Run `pytest tests/` to ensure no regressions.
- **Type Checking**: We enforce strict typing with `mypy`. Run `mypy tqk --strict`.

## Pull Request Process

1. Create a new branch for your feature or fix.
2. Ensure all tests pass and linting is clean.
3. We use **Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`, `chore:`).
4. Submit your PR against the `main` branch. 
5. All PRs require CI passing and a maintainer review.

## Code of Conduct

Be respectful and focus on technical excellence.

---

*Thank you for making LLM memory portable!*
