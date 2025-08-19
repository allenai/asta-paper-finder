.PHONY: all install install-dev lock rm graph sync sync-dev uninstall venv run type-check lint lint-check format format-check style fix test test-cov test-update help

# Extract all arguments after the make target (e.g. for 'make install foo --bar' returns 'foo --bar').
ARGS = $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

# Default target
all: help

# Add a package
install:
	uv add $(ARGS)

# Add development dependency
install-dev:
	uv add --dev $(ARGS)

# Generate dependency graph
graph:
	uv tree $(ARGS)

# Sync virtual environment with uv.lock
sync:
	uv sync $(ARGS)

# Sync virtual environment with uv.lock and install development dependencies
sync-dev:
	uv sync --dev $(ARGS)

# Remove a package
uninstall:
	uv remove $(ARGS)

# Remove a development package
uninstall-dev:
	uv remove --group dev $(ARGS)

# Create virtual environment
venv:
	uv venv -p 3.12.8

# Run a command in virtual environment
run:
	uv run $(ARGS)

# Type checking
type-check:
	uv run pyright .

# Linting
lint:
	uv run ruff check --fix .

# Linting checking
lint-check:
	uv run ruff check . && uv run flake8 --select SST .

# Formatting
format:
	uv run ruff format .

# Format checking
format-check:
	uv run ruff format --check .

deps-check:
	uv run deptry .

# Style check
style: deps-check format-check lint-check type-check

# Style check with auto-fix
fix: lint format type-check

# Testing
test:
	uv run pytest -m 'not regression' . $(ARGS)

test-cov:
	uv run pytest -m 'not regression' . --cov-config=.coveragerc --cov --cov-report=lcov:lcov.info --cov-report=term $(ARGS)

test-update:
	uv run pytest -m 'not regression' --snapshot-update . $(ARGS)

%:
	@:

help::
	@echo "Common commands:"
	@echo "  make install      	- Install all dependencies"
	@echo "  make install-dev  	- Install all, including development, dependencies"
	@echo "  make lock        	- Lock dependencies"
	@echo "  make clean       	- Remove virtual environment"
	@echo "  make graph       	- Generate dependency graph"
	@echo "  make sync        	- Sync virtual environment with uv.lock"
	@echo "  make sync-dev    	- Sync virtual environment with uv.lock, including development dependencies"
	@echo "  make rm        	- Remove a specific package"
	@echo "  make venv        	- Create virtual environment"
	@echo "  make run         	- Run a command in virtual environment"
	@echo ""
	@echo "Development commands:"
	@echo "  make style     	- Run format check, lint and type-check"
	@echo "  make type-check     	- Run pyright type checker"
	@echo "  make lint          	- Run Ruff linter"
	@echo "  make format        	- Format code"
	@echo "  make format-check  	- Check formatting with Ruff"
	@echo "  make test         	- Run tests (excluding regression)"
	@echo "  make test-cov     	- Run tests with coverage report"
	@echo "  make test-update  	- Update test snapshots"
	@echo ""
	@echo "NOTE: You can pass additional arguments to most commands using ``--``, e.g. 'make install -- black --dev'"
