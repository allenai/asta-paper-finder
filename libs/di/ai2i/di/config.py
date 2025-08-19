from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai2i.config.config_models import ConfigValuePlaceholder


@dataclass(frozen=True)
class Di:
    round_scope_timeout: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["di", "round_scope_timeout"])


@dataclass(frozen=True)
class AppConfigSchema:
    di: Di = Di()
    domain: ConfigValuePlaceholder[Literal["cs"]] = ConfigValuePlaceholder(["domain"])
    log_max_length: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["log_max_length"])

    def __getattr__(self, name: str) -> ConfigValuePlaceholder[str]:
        return ConfigValuePlaceholder([name])


cfg_schema = AppConfigSchema()
