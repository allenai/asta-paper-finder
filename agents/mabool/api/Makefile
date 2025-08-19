include ../../../dev/make/common.mk
include ../../../dev/make/functions.mk

.PHONY: type-stubgen type-cfg-gen type-ufs-gen start-dev

fix: type-ufs-gen lint format type-check

# Type config generation
type-cfg-gen:
	uv run python -m ai2i.config.gen_config_types --output-dir mabool/data_model
	$(MAKE) fix

# Type user facing strings generation source to target mapping
./mabool/data_model/ufs.py: ./conf/user_facing/*.toml
	uv run python -m ai2i.config.gen_config_types --user-facing --output-dir mabool/data_model

# Type user facing strings generation
type-ufs-gen: ./mabool/data_model/ufs.py

# Development and production startup
start-dev:
	uv run sh dev.sh

help::
	@echo ""
	@echo "Paper-finder development commands:"
	@echo "  make type-cfg-gen		- Generate type settings"
	@echo "  make type-ufs-gen  	- Generate user facing strings"
	@echo "Server commands:"
	@echo "  make start-dev     		- Start development server"
	@echo ""
