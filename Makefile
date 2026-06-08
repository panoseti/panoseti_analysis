# Developer convenience targets. All wrap `uv run` so they use the project venv.
# Run `make help` to list targets.

.DEFAULT_GOAL := help
.PHONY: help sync lint format typecheck test check hooks-install hooks-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync: ## Create/update the .venv from pyproject (uv sync)
	uv sync

lint: ## Lint (and auto-fix) with ruff
	uv run ruff check --fix src tests

format: ## Format with ruff
	uv run ruff format src tests

typecheck: ## Type-check the package with mypy --strict
	uv run mypy src/panoseti_analysis

test: ## Run the fast unit/integration tests
	uv run pytest tests -q -m "not slow and not ral_only"

check: lint format typecheck test ## Run lint + format + typecheck + test

hooks-install: ## Install the git pre-commit + pre-push hooks
	uv run pre-commit install

hooks-run: ## Run all pre-commit hooks across the whole repo
	uv run pre-commit run --all-files
