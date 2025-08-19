.PHONY: all install install-dev lock rm graph sync sync-dev uninstall venv run type-check lint lint-check format format-check style fix help
SHELL := bash

# Extract all arguments after the make target (e.g. for 'make install foo --bar' returns 'foo --bar').
ARGS = $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
SUBPROJECTS = "./agents/mabool/api" "./libs/common" "./libs/di" "./libs/dcollection" "./libs/chain" "./libs/config"

# Default target
all: help

# Lock dependencies
lock:
	uv lock $(ARGS)

# Remove virtual environment
rm:
	rm -rf .venv

# Generate dependency graph
graph:
	uv tree $(ARGS)

# Sync virtual environment with uv.lock
sync:
	uv sync --all-packages $(ARGS)

# Sync virtual environment with uv.lock and install development dependencies
sync-dev:
	uv sync  --all-packages --dev $(ARGS)

# Create virtual environment
venv:
	uv venv -p 3.12.8

MODE ?= par

# Type checking
type-check: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) type-check

# Linting
lint: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) lint

# Linting checking
lint-check: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) lint-check

# Formatting
format: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) format

# Format checking
format-check: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) format-check

# Style check
style: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) style

# Style check with auto-fix
fix: sync-dev
	@source ./dev/make/make_foreach.sh && make_foreach $(MODE) $(SUBPROJECTS) fix

%:
	@:

help::
	@echo "Common commands:"
	@echo "  make lock     - Lock dependencies"
	@echo "  make clean    - Remove virtual environment"
	@echo "  make graph    - Generate dependency graph"
	@echo "  make sync     - Sync virtual environment with uv.lock"
	@echo "  make sync-dev - Sync virtual environment with uv.lock, including development dependencies"
	@echo "  make venv     - Create virtual environment"
	@echo ""
	@echo "Development commands:"
	@echo "  make style        - Run format check, lint and type-check"
	@echo "  make type-check   - Run pyright type checker"
	@echo "  make lint         - Run Ruff linter"
	@echo "  make format       - Format code"
	@echo "  make format-check - Check formatting with Ruff"
	@echo ""
	@echo "NOTE: by default all Development commands will run in parallel, capturing all stdout/stderr and only"
	@echo "      saying which (if any) sub projects have failed. You can choose to run the subtasks sequentially"
	@echo "      and see the stdout/stderr of the subprojects by setting 'MODE=seq'"
	@echo "      For Example:"
	@echo "        make format-check MODE=seq"
