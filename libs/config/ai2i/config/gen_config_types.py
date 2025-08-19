import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Sequence, TypeVar

import toml

T = TypeVar("T")
SettingsType = Any | dict[str, Any]
SyncOrAsyncFunc = Callable[..., T | Awaitable[T]]

logger = logging.getLogger(__name__)

VALUE_CONTAINER = "ConfigValuePlaceholder"
DefaultEnv = Literal["default", "all"]
SchemaClassName = Literal["AppConfigSchema", "UserFacingSchema"]
SchemaVariableName = Literal["cfg_schema", "uf"]


def generate_settings_types(
    settings_files: list[Path],
    output_file: Path,
    default_env: DefaultEnv,
    schema_class_name: SchemaClassName,
    schema_variable_name: SchemaVariableName,
) -> None:
    def camel_case(s: str) -> str:
        words = s.replace("_", " ").split()
        return "".join(word.capitalize() for word in words)

    def get_type_hint(
        key: str, value: Any, all_values: dict[str, set], parent_key_path: Sequence[str] | None = None
    ) -> str:
        if parent_key_path is None:
            parent_key_path = []
        full_key_path = [*parent_key_path, key]
        full_key = ".".join(full_key_path)

        type_value: str
        if isinstance(value, bool):
            type_value = "bool"
        elif isinstance(value, int):
            type_value = "int"
        elif isinstance(value, float):
            type_value = "float"
        elif isinstance(value, str):
            literals = all_values.get(full_key, {value})
            if len(literals) == 1:
                type_value = f"Literal[{repr(next(iter(literals)))}]"
            else:
                type_value = f"Literal[{', '.join(sorted(repr(v) for v in literals))}]"
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                type_value = f"list[{camel_case(key.removesuffix('s'))}]"
            elif value:
                inner_type = get_type_hint(key, value[0], all_values, full_key)
                type_value = f"list[{inner_type}]"
            else:
                type_value = "list[Any]"
        elif isinstance(value, dict):
            type_value = camel_case(key)
        else:
            type_value = "Any"

        full_key_path_str: str
        if len(full_key_path) > 1:
            full_key_path_str = ", ".join([f'"{v}"' for v in full_key_path])
        else:
            full_key_path_str = f'"{full_key_path[0]}"'

        return f"{VALUE_CONTAINER}[{type_value}] = {VALUE_CONTAINER}([{full_key_path_str}])"

    def generate_class(
        name: str, data: dict[str, Any], all_values: dict[str, set], parent_key_path: Sequence[str] | None = None
    ) -> list[str]:
        if parent_key_path is None:
            parent_key_path = []
        class_name = name
        lines = ["@dataclass(frozen=True)", f"class {class_name}:"]
        lines_with_value = []
        lines_without_value = []
        for key, value in data.items():
            if isinstance(value, dict):
                inner_class_name = camel_case(key)
                lines_without_value.append(f"    {key.lower()}: {inner_class_name} = {inner_class_name}()")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                value_type = f"list[{camel_case(key.removesuffix('s'))}]"

                key_path = [*parent_key_path, key.lower()]
                full_key_path_str: str
                if len(key_path) > 1:
                    full_key_path_str = ", ".join([f'"{v}"' for v in key_path])
                else:
                    full_key_path_str = f'"{key_path[0]}"'
                type_hint = f"{VALUE_CONTAINER}[{value_type}] = {VALUE_CONTAINER}([{full_key_path_str}])"

                lines_with_value.append(f"    {key.lower()}: {type_hint}")
            else:
                type_hint = get_type_hint(key, value, all_values, parent_key_path)
                lines_with_value.append(f"    {key.lower()}: {type_hint}")
        return lines + lines_without_value + lines_with_value

    def generate_named_tuple(name: str, data: list[dict[str, Any]], all_values: dict[str, set]) -> list[str]:
        lines = ["@dataclass(frozen=True)", f"class {name}:"]

        if not data:
            return lines

        # Collect all possible values for each key
        field_values = defaultdict(set)
        for item in data:
            for key, value in item.items():
                field_values[key].add(value)

        # Generate type hints for each field
        for key in field_values.keys():
            values = field_values[key]
            if len(values) == 1:
                type_hint = f"Literal[{repr(next(iter(values)))}]"
            else:
                type_hint = f"Literal[{', '.join(sorted(repr(v) for v in values))}]"
            lines.append(f"    {key}: {type_hint}")

        return lines

    def deep_merge(dict1: dict, dict2: dict) -> dict:
        """Recursively merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if isinstance(value, dict):
                result[key] = deep_merge(result.get(key, {}), value)
            elif isinstance(value, list) and result.get(key) is not None:
                result[key] = result[key] + value
            else:
                result[key] = value
        return result

    def collect_values(data: Any, all_values: dict[str, set], prefix: str = "") -> None:
        """Recursively collect all string values from nested dictionaries."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                collect_values(value, all_values, new_prefix)
        elif isinstance(data, list):
            for item in data:
                collect_values(item, all_values, prefix)
        elif isinstance(data, str):
            all_values[prefix].add(data)

    # Load settings from all TOML files
    settings: dict = defaultdict(dict)
    all_values: dict = defaultdict(set)

    for settings_file in settings_files:
        with open(settings_file, "r") as f:
            file_settings = toml.load(f)

        # Merge settings, keeping all values
        for env, env_settings in file_settings.items():
            settings[env] = deep_merge(settings[env], env_settings)

        # Collect all values for each key across environments and files
        for env in file_settings:
            collect_values(file_settings[env], all_values)

    output_lines = [
        "from __future__ import annotations",
        "",
        "from typing import Literal",
        "from dataclasses import dataclass",
        "",
        f"from ai2i.config.config_models import {VALUE_CONTAINER}",
        "",
        "",
    ]

    settings_to_generate = settings[default_env] if default_env != "all" else {k: v for k, v in settings.items()}
    classes_to_generate: list[tuple[str, dict[str, Any], list[str]]] = [(schema_class_name, settings_to_generate, [])]
    generated_classes = set()
    named_tuples_to_generate = []
    output_chunks = []

    while classes_to_generate:
        class_name, class_data, key_path = classes_to_generate.pop(0)
        if class_name not in generated_classes:
            class_lines = generate_class(class_name, class_data, all_values, key_path)
            if class_name == schema_class_name:
                class_lines += [
                    "",
                    f"    def __getattr__(self, name: str) -> {VALUE_CONTAINER}[str]:",
                    f"        return {VALUE_CONTAINER}([name])",
                ]
            output_chunks.append(class_lines)
            generated_classes.add(class_name)

            for key, value in class_data.items():
                if isinstance(value, dict):
                    classes_to_generate.append((camel_case(key), value, [*key_path, key]))
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    named_tuples_to_generate.append((camel_case(key.removesuffix("s")), value))

    for name, data in named_tuples_to_generate:
        named_tuple_lines = generate_named_tuple(name, data, all_values)
        output_chunks.append(named_tuple_lines)

    output_chunks.reverse()
    output_lines.extend(["\n".join(chunk) + "\n" for chunk in output_chunks])

    output_lines.extend([f"\n\n{schema_variable_name} = {schema_class_name}()"])

    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Settings types have been generated in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate settings type definitions")
    parser.add_argument("--conf-dir", type=str, help="Path to the project root directory", default="conf")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated files")
    parser.add_argument(
        "--user-facing", action="store_true", help="Generate user-facing settings types instead of app settings"
    )

    args = parser.parse_args()
    conf_dir = Path(args.conf_dir)
    output_dir = Path(args.output_dir)
    if args.user_facing:
        settings_dir = conf_dir / "user_facing"
        settings_files = list(settings_dir.glob("*.toml"))
        output_file = output_dir / "ufs.py"
        default_env = "all"
        schema_class_name = "UserFacingSchema"
        schema_variable_name = "uf"
    else:
        settings_dir = conf_dir
        settings_files = list(settings_dir.glob("config*.toml"))
        output_file = output_dir / "config.py"
        default_env = "default"
        schema_class_name = "AppConfigSchema"
        schema_variable_name = "cfg_schema"
    generate_settings_types(settings_files, output_file, default_env, schema_class_name, schema_variable_name)
