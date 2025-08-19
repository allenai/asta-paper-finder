# 🔧 ai2i.config

A configuration management library for Python applications.

## Overview

`ai2i.config` provides a robust way to manage application configuration and user-facing strings with features like:

- Hierarchical configuration with inheritance and merging
- Environment-specific settings
- Type-safe configuration and user-facing strings access
- Dependency injection via config placeholders
- Context-based configuration management
- Automatic type generation from configuration files

## Basic Usage

### Loading Configuration

```python
from pathlib import Path
from ai2i.config import load_conf, application_config_ctx

# Load app config and set up the configuration context
with application_config_ctx(load_conf(Path("./conf"))):
    # Your application code here
    pass
```

### Accessing Configuration Values

```python
from ai2i.config import config_value
from your_project.config import cfg_schema  # Auto-generated from your config files

# Access configuration values
api_key = config_value(cfg_schema.s2_api.api_key)
timeout = config_value(cfg_schema.s2_api.timeout, default=30)
```

### Accessing User-Facing Strings

```python
from ai2i.config import ufv
from your_project.ufs import uf  # Auto-generated from your user-facing strings files

# Get a user-facing string
message = ufv(uf.perfect_matches.plural)

# With string formatting
welcome = ufv(uf.welcome_message, username="User")
```

## Advanced Features

### Configurable Function Parameters

Use the `configurable` decorator to inject configuration values into function parameters:

```python
from ai2i.config import configurable, ConfigValue
from your_project.config import cfg_schema

@configurable
def fetch_data(timeout=ConfigValue(cfg_schema.s2_api.timeout, default=100)):
    # The actual timeout value will be resolved from config
    pass
```

## Configuration File Structure

The library expects configuration files organized as follows:

```
conf/
├── config.toml            # Base configuration (required)
├── config.extra.*.toml    # Additional configuration files
├── .env.secret            # Secret values (not in version control)
└── user_facing/           # User-facing string files
    └── *.toml             # User-facing string definitions
```

### Configuration Format

Configuration files use TOML format and support hierarchical structure:

```toml
# Example config.toml
[default]
app_name = "My Application"
debug = false

[default.logging]
level = "info"
format = "json"

[production]
debug = false

[production.logging]
level = "warning"
```

### Environment-Based Configuration

The library merges configuration based on the current environment:

1. Base settings from `default` section
2. Environment-specific overrides (controlled by `APP_CONFIG_ENV`)
3. Domain-specific settings (controlled by `DATA_DOMAIN`)
4. Secret values from `.env.secret`
5. Environment variables

## Type Generation

The library can generate type definitions from your configuration files, making it easier to work with configurations in a type-safe manner:

```shell script
# Generate application configuration types
uv run python -m ai2i.config.gen_config_types --conf-dir conf --output-dir your_project

# Generate user-facing string types
uv run python -m ai2i.config.gen_config_types --conf-dir conf --output-dir your_project --user-facing
```

This creates:
- `your_project-output-dir/config.py` - Types for application configuration
- `your_project-output-dir/ufs.py` - Types for user-facing strings

## Environment Variables

The library recognizes these environment variables:

- `APP_CONFIG_ENV`: Specifies which configuration environment to use (default: uses the `default` section)
- `DATA_DOMAIN`: Specifies the domain for domain-specific settings (default: "cs")

## Configuration Context

The library uses context variables to manage configuration state. This allows for configuration to be different between async tasks or threads.

```python
from ai2i.config import application_config_ctx, load_conf

# Load a different configuration
test_config = load_conf(Path("./test_conf"))

# Temporarily change configuration
with application_config_ctx(test_config):
    # Code in this block uses the test configuration
    pass
```

You can also temporarily override specific settings (useful mostly for testing):

```python
from ai2i.config import with_config_overrides

@with_config_overrides(api={"timeout": 60})
def test_function():
    # This function will see the overridden timeout
    pass
```

## Development

### Setup Development Environment

```shell script
make sync-dev
```

### Testing

Before running tests, you might need to generate the types:
```shell script
make type-cfg-gen -- --conf-dir ai2i/config/tests/conf --output-dir ai2i/config/tests
```

```shell script
make test
# -or-
make test-cov  # With coverage
```

### Code Quality

```shell script
make style  # Run format check, lint and type-check
make fix    # Automatically fix style issues
```
