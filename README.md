# claude-code-uv-template

A Python project template for developing with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and [uv](https://docs.astral.sh/uv/).

## What's Included

- **uv** for fast Python environment and dependency management
- **pytest** with `unit` and `integration` markers pre-configured
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest-cov** for coverage reporting (80% threshold)
- **pytest-mock** and **pytest-xdist** for mocking and parallel test runs
- **python-dotenv** for environment variable management
- **GitHub Actions CI** workflow out of the box
- **Makefile** with common dev commands
- **Claude Code hooks** that auto-lint `.py` files after edits
- **CLAUDE.md** with development guidelines for Claude Code agents
- **src layout** for clean package structure

## Getting Started

```bash
# Install dependencies
uv sync

# Run tests
make test

# Lint and format
make lint
make lint-fix
make format
```

## Project Structure

```
├── src/                          # Application source code
│   └── claude_code_uv_template/  # Main package (rename to your project)
├── tests/
│   ├── conftest.py               # Shared test fixtures
│   ├── unit/                     # Unit tests (@pytest.mark.unit)
│   └── integration/              # Integration tests (@pytest.mark.integration)
├── .claude/
│   └── settings.json             # Claude Code hooks config
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI
├── .githooks/
│   └── pre-commit                # Pre-commit hook (runs make check)
├── .env.example                  # Environment variable template
├── CLAUDE.md                     # Development guidelines for Claude Code
├── Makefile                      # Dev commands (lint, format, test)
└── pyproject.toml                # Project config and tool settings
```

## Testing

Tests use pytest with two markers:

```bash
make test              # Run all tests
make test-unit         # Run only @pytest.mark.unit tests
make test-integration  # Run only @pytest.mark.integration tests
make coverage          # Run tests with coverage report
make check             # Run lint + typecheck + unit tests (also runs as pre-commit hook)
```

## CI

GitHub Actions runs `make check` and `make coverage` on every push to `main` and on PRs. See `.github/workflows/ci.yml`.

## Customizing

1. Rename the package directory under `src/` to match your project name
2. Update `name` and `description` in `pyproject.toml`
3. Update `tool.hatch.build.targets.wheel.packages` in `pyproject.toml`
4. Update `CLAUDE.md` with project-specific guidelines
