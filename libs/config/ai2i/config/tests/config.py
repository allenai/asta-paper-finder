from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai2i.config.config_models import ConfigValuePlaceholder


@dataclass(frozen=True)
class DenseAgent:
    initial_top_k_per_query: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "initial_top_k_per_query"]
    )
    reformulate_prompt_example_docs: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "reformulate_prompt_example_docs"]
    )


@dataclass(frozen=True)
class AppConfigSchema:
    dense_agent: DenseAgent = DenseAgent()
    env: ConfigValuePlaceholder[Literal["test"]] = ConfigValuePlaceholder(["env"])

    def __getattr__(self, name: str) -> ConfigValuePlaceholder[str]:
        return ConfigValuePlaceholder([name])


cfg_schema = AppConfigSchema()
