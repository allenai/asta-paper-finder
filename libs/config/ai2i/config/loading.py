from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from deepmerge import always_merger
from dotenv import dotenv_values

from ai2i.config.config_models import AppConfig, ConfigDict, ConfigSettings, UserFacing

logger = logging.getLogger(__name__)


def load_config_files(dir: Path) -> dict[str, Any]:
    basic_settings: dict[str, Any]
    with (dir / "config.toml").open(mode="rb") as f:
        basic_settings = tomllib.load(f)

    for file in dir.glob("config.extra.*.toml"):
        with file.open(mode="rb") as f:
            extra_settings = tomllib.load(f)
            basic_settings = always_merger.merge(basic_settings, extra_settings)

    return basic_settings


def load_user_facing_files(dir: Path) -> dict[str, Any]:
    merged_user_facing_strings: dict[str, Any] = {}
    for file in dir.glob("*.toml"):
        with file.open(mode="rb") as f:
            user_facing_strings = tomllib.load(f)
            merged_user_facing_strings = always_merger.merge(merged_user_facing_strings, user_facing_strings)

    return merged_user_facing_strings


def load_secrets_file(dir: Path) -> dict[str, Any]:
    try:
        secrets = dotenv_values(dir / ".env.secret")
        return {"default": {k.lower(): v for k, v in secrets.items()}}
    except Exception as e:
        logger.warning(f"Failed to load secrets file: {e}")
        return {}


def update_environment_with_secrets(settings_dict: dict[str, Any]) -> None:
    # NOTE: some of our internal tools expect these values to be in the environment
    #       for example to access vespa a secret is first retrieved from AWS secrets manager
    #       and the access to that secrets manager expects the AWS_* environment varables to
    #       be set.
    for key in settings_dict.keys():
        if (
            key.lower() == "openai_api_key"
            or key.lower() == "s2_api_key"
            or key.lower() == "cohere_api_key"
            or key.lower() in ["aws_access_key_id", "aws_secret_access_key"]
            or key.lower().startswith("langchain")
        ):
            os.environ[key.upper()] = str(settings_dict[key])


def load_conf(conf_dir: Path) -> AppConfig:
    return AppConfig(config=load_config_settings(conf_dir), user_facing=load_user_facing(conf_dir))


def load_user_facing(conf_dir: Path) -> UserFacing:
    user_facing_strings_dir = conf_dir / "user_facing"

    user_facing_strings = load_user_facing_files(user_facing_strings_dir)
    return ConfigDict(user_facing_strings)


def load_config_settings(conf_dir: Path) -> ConfigSettings:
    basic_settings = load_config_files(conf_dir)
    secrets_settings = load_secrets_file(conf_dir)
    settings_from_files: dict[str, Any] = always_merger.merge(basic_settings, secrets_settings)

    # Pick the settings namespace based on specified config env
    settings_env = os.getenv("APP_CONFIG_ENV", None)

    if settings_env is None:
        settings_from_files = settings_from_files.get("default", {})
    else:
        settings_from_files = always_merger.merge(
            settings_from_files.get("default", {}), settings_from_files.get(settings_env, {})
        )

    # Add environment variables to the config
    environ: dict[str, Any] = {k: v for k, v in os.environ.items()}
    final_settings_dict = always_merger.merge(_deep_lowercase_keys(settings_from_files), _deep_lowercase_keys(environ))

    update_environment_with_secrets(final_settings_dict)

    return ConfigDict(final_settings_dict)


def _deep_lowercase_keys(d: dict[str, Any]) -> dict[str, Any]:
    new_d = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = _deep_lowercase_keys(v)
        new_d[k] = v
        new_d[k.lower()] = v
    return new_d
