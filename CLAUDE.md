# Claude Code Instructions

## Python Environment

Use `uv run python` to run Python commands (not `python` or `python3`). This uses the project's virtual environment with all dependencies installed.

Examples:
```bash
uv run python -m specparser.amt data/amt.yml --expand 2024 2024
uv run python -m specparser.expander --selftest
uv run pytest
```
