from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, Sequence, TypeVar

from deepmerge.merger import Merger

from ai2i.common.utils.value import ValueNotSet

dicts_only_merger = Merger([(dict, ["merge"]), (list, ["override"]), (set, ["override"])], ["override"], ["override"])


class Config:
    def _wrap_value(self, value: Any) -> Config | Any:
        if isinstance(value, dict):
            return ConfigDict(value)
        if isinstance(value, list):
            return ConfigList(value)
        else:
            return value


class ConfigList(Config):
    _values: list[Any]

    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def __iter__(self) -> Iterator[ConfigDict | Any]:
        return map(self._wrap_value, iter(self._values))

    def __getitem__(self, key: int) -> Any:
        v = self._values[key]
        return self._wrap_value(v)


class ConfigDict(Config):
    @staticmethod
    def empty() -> ConfigDict:
        return ConfigDict({})

    _values: dict[str, Any]

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str, /) -> Any:
        try:
            value = self._values[name]
        except KeyError:
            return None
        return self._wrap_value(value)

    def merge_dict(self, d: dict[str, Any]) -> ConfigDict:
        return ConfigDict(dicts_only_merger.merge(self._values, d))


ConfigSettings = ConfigDict
UserFacing = ConfigDict


@dataclass(frozen=True)
class AppConfig:
    config: ConfigDict
    user_facing: ConfigDict


A = TypeVar("A", covariant=True)
B = TypeVar("B")


@dataclass(frozen=True)
class ConfigValuePlaceholder(Generic[A]):
    keys_path: Sequence[str]
    default: A | ValueNotSet = ValueNotSet.instance()
    default_factory: Callable[[], A] | ValueNotSet = ValueNotSet.instance()

    def read_from_dict(self, d: dict[str, Any]) -> A:
        if len(self.keys_path) == 0:
            raise ValueError("Empty keys_path for read_from_dict")

        key = self.keys_path[0]
        result, d = self._follow_key(d, key)
        for key in self.keys_path[1:]:
            result, d = self._follow_key(d, key)
        return result

    def with_default(self, v: B) -> ConfigValuePlaceholder[A | B]:
        return ConfigValuePlaceholder(self.keys_path, default=v)

    def with_default_factory(self, f: Callable[[], B]) -> ConfigValuePlaceholder[B]:
        return ConfigValuePlaceholder(self.keys_path, default_factory=f)

    def _follow_key(self, d: dict[str, Any], key: str) -> tuple[A, dict[str, Any]]:
        try:
            result = d[key]
            d = d[key]
            return result, d
        except KeyError as e:
            if not isinstance(self.default_factory, ValueNotSet):
                return self.default_factory(), d
            if not isinstance(self.default, ValueNotSet):
                return self.default, d
            else:
                raise KeyError(f"{'.'.join(self.keys_path)} not found in config") from e
