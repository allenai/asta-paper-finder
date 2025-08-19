from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec, TypeVar, cast

from deepmerge.merger import Merger

from ai2i.common.utils.value import ValueNotSet
from ai2i.config.common import Substitution, SubstitutionInfo
from ai2i.config.config_models import (
    AppConfig,
    ConfigDict,
    ConfigList,
    ConfigSettings,
    ConfigValuePlaceholder,
    UserFacing,
)

logger = logging.getLogger(__name__)

dicts_only_merger = Merger([(dict, ["merge"]), (list, ["override"]), (set, ["override"])], ["override"], ["override"])


def is_test() -> bool:
    return "PYTEST_VERSION" in os.environ


_app_config_context: ContextVar[ConfigSettings | None] = ContextVar("app_config_context", default=None)
_user_facing_context: ContextVar[ConfigSettings | None] = ContextVar("user_facing_context", default=None)


def _get_config() -> ConfigSettings | None:
    return _app_config_context.get()


def _get_user_facing() -> UserFacing | None:
    return _user_facing_context.get()


def get_user_facing_or_throw() -> UserFacing:
    user_facing = _get_user_facing()
    if user_facing is None:
        raise Exception("user facing strings not loaded in context")
    return user_facing


def get_config_or_throw() -> ConfigSettings:
    config = _get_config()
    if config is None:
        raise Exception("application config not loaded in context")
    return config


@contextmanager
def application_config_ctx(app_conf: AppConfig) -> Iterator[None]:
    config_reset_token = _app_config_context.set(app_conf.config)
    user_facing_reset_token = _user_facing_context.set(app_conf.user_facing)
    yield
    _user_facing_context.reset(user_facing_reset_token)
    _app_config_context.reset(config_reset_token)


SettingsType = Any | dict[str, Any]
P = ParamSpec("P")
R = TypeVar("R")
WrappedFunc = Callable[P, R]


def with_config_overrides(**kwargs: SettingsType) -> Callable[[WrappedFunc], WrappedFunc]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **func_kwargs: Any) -> R:
                current_config: ConfigSettings = _get_config() or ConfigDict.empty()
                user_facing: UserFacing = _get_user_facing() or ConfigDict.empty()

                with application_config_ctx(
                    AppConfig(config=current_config.merge_dict(kwargs), user_facing=user_facing)
                ):
                    return await func(*args, **func_kwargs)

            return cast(Callable[P, R], async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **func_kwargs: Any) -> R:
                current_config: ConfigSettings = _get_config() or ConfigDict.empty()
                user_facing: UserFacing = _get_user_facing() or ConfigDict.empty()

                with application_config_ctx(
                    AppConfig(config=current_config.merge_dict(kwargs), user_facing=user_facing)
                ):
                    return func(*args, **func_kwargs)

            return sync_wrapper

    return decorator


def _deep_lowercase_keys(d: dict[str, Any]) -> dict[str, Any]:
    new_d = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = _deep_lowercase_keys(v)
        new_d[k] = v
        new_d[k.lower()] = v
    return new_d


A = TypeVar("A", covariant=True)
P = ParamSpec("P")
B = TypeVar("B")
C = TypeVar("C")


def resolve_config_placeholder(placeholder: ConfigValuePlaceholder[A]) -> A:
    config_values = get_config_or_throw()._values
    return placeholder.read_from_dict(config_values)


_configurable_substitution = [Substitution(ConfigValuePlaceholder, resolve_config_placeholder)]


def configurable(f: Callable[P, B]) -> Callable[P, B]:
    f_info = SubstitutionInfo.from_function(f)

    @wraps(f)
    def _decorated(*args: P.args, **kwargs: P.kwargs) -> B:
        resolved_args, resolved_kwargs = f_info.resolve(_configurable_substitution, *args, **kwargs)
        return f(*resolved_args, **resolved_kwargs)  # type: ignore

    return _decorated


def ConfigValue(  # noqa: N802
    reader: ConfigValuePlaceholder[A],
    /,
    default: B | ValueNotSet = ValueNotSet.instance(),
    default_factory: Callable[[], B] | ValueNotSet = ValueNotSet.instance(),
) -> A | B:
    resolved_reader: ConfigValuePlaceholder[A | B] = reader
    if not isinstance(default, ValueNotSet):
        resolved_reader = reader.with_default(default)
    if not isinstance(default_factory, ValueNotSet):
        resolved_reader = reader.with_default_factory(default_factory)
    return cast(A | B, resolved_reader)


def config_value(
    reader: ConfigValuePlaceholder[A],
    /,
    default: B | ValueNotSet = ValueNotSet.instance(),
    default_factory: Callable[[], B] | ValueNotSet = ValueNotSet.instance(),
) -> A | B:
    config = get_config_or_throw()
    resolved_reader: ConfigValuePlaceholder[A | B] = reader
    if not isinstance(default, ValueNotSet):
        resolved_reader = reader.with_default(default)
    if not isinstance(default_factory, ValueNotSet):
        resolved_reader = reader.with_default_factory(default_factory)

    config_value = resolved_reader.read_from_dict(config._values)
    if isinstance(config_value, dict):
        return cast(A | B, ConfigDict(config_value))
    if isinstance(config_value, list):
        return cast(A | B, ConfigList(config_value))
    else:
        return config_value


def ufv(reader: ConfigValuePlaceholder[str], /, **format_args: Any) -> str:
    config = get_user_facing_or_throw()
    resolved_reader: ConfigValuePlaceholder[str] = reader
    config_value = resolved_reader.read_from_dict(config._values)

    if isinstance(config_value, dict):
        return cast(str, ConfigDict(config_value))
    if isinstance(config_value, list):
        return cast(str, ConfigList(config_value))

    if format_args and isinstance(config_value, str):
        return config_value.format(**format_args)

    return config_value
