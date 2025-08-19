# ruff: noqa: B008

import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Never

import pytest

from ai2i.config import ConfigValue, with_config_overrides
from ai2i.di.config import cfg_schema
from ai2i.di.interface.errors import (
    DependencyDefinitionError,
    OutOfScopeDependencyError,
    ProviderBuildError,
    ProviderReleaseError,
)
from ai2i.di.interface.gateway import DI
from ai2i.di.managed_env import ManagedEnv


@pytest.fixture
def some_prefix() -> str:
    return f"prefix_{uuid.uuid4()}"


@pytest.fixture
def some_value() -> str:
    return f"value_{uuid.uuid4()}"


@pytest.fixture
def some_value2() -> str:
    return f"value2_{uuid.uuid4()}"


@pytest.fixture(scope="function")
def env() -> ManagedEnv:
    return ManagedEnv.simple()


async def test_basic_injection(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> MyValue:
        return MyValue(some_value)

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value)) -> None:
        nonlocal effect
        effect = prefix + value.val

    async with env.managed_scope():
        await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:{some_value}"


async def test_basic_injection_provider_extra_args(
    env: ManagedEnv, some_prefix: str, some_value: str, some_value2: str
) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value(other_value: str = some_value2) -> MyValue:
        return MyValue(some_value + ";" + other_value)

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value)) -> None:
        nonlocal effect
        effect = prefix + value.val

    async with env.managed_scope():
        await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:{some_value};{some_value2}"


async def test_basic_injection_default_value(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> MyValue:
        return MyValue("irrelevant")

    default_value = MyValue(some_value)

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value, default=default_value)) -> None:
        nonlocal effect
        effect = prefix + value.val

    # outside of scope, will not be able to inject, should get default value
    await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:{some_value}"


async def test_basic_injection_default_factory(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> MyValue:
        return MyValue("irrelevant")

    @env.managed
    async def store_value(
        prefix: str, value: MyValue = DI.requires(my_value, default_factory=lambda: MyValue(some_value))
    ) -> None:
        nonlocal effect
        effect = prefix + value.val

    # outside of scope, will not be able to inject, should get default value
    await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:{some_value}"


async def test_basic_injection_missing_default_value(env: ManagedEnv, some_prefix: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> MyValue:
        return MyValue("missing")

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value)) -> None:
        nonlocal effect
        effect = prefix + value.val

    # outside of scope, will not be able to inject, should get default value
    with pytest.raises(OutOfScopeDependencyError):
        await store_value(f"{some_prefix}:")


async def test_basic_provider_extra_args_error(env: ManagedEnv, some_value: str) -> None:
    @dataclass(frozen=True)
    class MyValue:
        val: str

    # regular arg that is missing a dependency definition
    with pytest.raises(DependencyDefinitionError):

        @env.provides()
        async def my_value(a: int) -> MyValue:
            return MyValue(some_value)

    # keyword arg that is missing a dependency definition
    with pytest.raises(DependencyDefinitionError):

        @env.provides()
        async def my_value_kwargs(*, a: int) -> MyValue:
            return MyValue(some_value)


async def test_basic_transitive_injection(env: ManagedEnv, some_prefix: str, some_value: str, some_value2: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides(name="base-value")
    async def my_value() -> MyValue:
        return MyValue(some_value)

    @env.provides(name="dependant-value")
    async def my_transitive_value(value: MyValue = DI.requires(my_value)) -> str:
        return value.val

    assert my_value.unique_name in my_transitive_value.depends_on

    @env.managed
    async def store_value(prefix: str, value: str = DI.requires(my_transitive_value)) -> None:
        nonlocal effect
        effect = prefix + value

    async with env.managed_scope():
        await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:{some_value}"

    await store_value(f"{some_prefix}:", some_value2)
    assert effect == f"{some_prefix}:{some_value2}"


@with_config_overrides(log_max_length=1717)
async def test_provider_with_config(env: ManagedEnv, some_prefix: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value(v: int = ConfigValue(cfg_schema.log_max_length)) -> MyValue:
        return MyValue(str(v))

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value)) -> None:
        nonlocal effect
        effect = prefix + str(value.val)

    async with env.managed_scope():
        await store_value(f"{some_prefix}:")

    assert effect == f"{some_prefix}:1717"


@with_config_overrides(log_max_length=1717)
async def test_managed_with_config(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> MyValue:
        return MyValue(some_value)

    @env.managed
    async def store_value(
        prefix: str, v: int = ConfigValue(cfg_schema.log_max_length), value: MyValue = DI.requires(my_value)
    ) -> None:
        nonlocal effect
        effect = ":".join((prefix, value.val, str(v)))

    async with env.managed_scope():
        await store_value(f"{some_prefix}")

    assert effect == f"{some_prefix}:{some_value}:1717"


async def test_basic_provider_with_cleanup(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""
    cleanup_done: bool = False

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> AsyncIterator[MyValue]:
        yield MyValue(some_value)
        nonlocal cleanup_done
        cleanup_done = True

    @env.managed
    async def store_value(prefix: str, value: MyValue = DI.requires(my_value)) -> None:
        nonlocal effect
        effect = prefix + value.val

    async with env.managed_scope():
        await store_value(f"{some_prefix}:")
        assert not cleanup_done

    # out of scope cleaning called
    assert cleanup_done
    assert effect == f"{some_prefix}:{some_value}"


@with_config_overrides(log_max_length=1717)
async def test_complex_provider_with_cleanup(env: ManagedEnv, some_prefix: str, some_value: str) -> None:
    effect: str = ""
    cleanup1_done: bool = False
    cleanup2_done: bool = False

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @env.provides()
    async def my_value() -> AsyncIterator[MyValue]:
        yield MyValue(some_value)
        nonlocal cleanup1_done
        cleanup1_done = True

    @env.provides()
    async def my_transitive_value(value: MyValue = DI.requires(my_value)) -> AsyncIterator[str]:
        yield value.val
        nonlocal cleanup2_done
        cleanup2_done = True

    @env.managed
    async def store_value(
        prefix: str,
        v: int = ConfigValue(cfg_schema.log_max_length),
        value: str = DI.requires(my_transitive_value),
    ) -> None:
        nonlocal effect
        effect = ":".join((prefix, value, str(v)))

    async with env.managed_scope():
        await store_value(f"{some_prefix}")
        assert not cleanup1_done
        assert not cleanup2_done

    # out of scope cleaning called
    assert cleanup1_done
    assert cleanup2_done
    assert effect == f"{some_prefix}:{some_value}:1717"


async def test_env_scope_build_failure_factory(env: ManagedEnv, some_value: str) -> None:
    @env.provides(name=some_value)
    async def my_value() -> Never:
        raise Exception("Fail on purpose")

    try:
        async with env.managed_scope():
            raise AssertionError("should never reach here")
    except ProviderBuildError as e:
        assert e.provider_name == some_value


async def test_env_scope_build_failure_context(env: ManagedEnv, some_value: str) -> None:
    @env.provides(name=some_value)
    async def my_value() -> AsyncIterator[Never]:
        raise Exception("Fail on purpose")
        yield

    try:
        async with env.managed_scope():
            raise AssertionError("should never reach here")
    except ProviderBuildError as e:
        assert e.provider_name == some_value


async def test_env_provider_errors_missing_yield(env: ManagedEnv, some_value: str) -> None:
    @env.provides(name=some_value)
    async def my_value() -> AsyncIterator[str]:
        if 0 > 1:
            yield some_value

    with pytest.raises(ProviderBuildError):
        await env.managed_scope().__aenter__()


async def test_env_provider_errors_extra_yield(env: ManagedEnv, some_value: str) -> None:
    @env.provides(name=some_value)
    async def my_value() -> AsyncIterator[str]:
        yield some_value
        yield some_value

    scope = env.managed_scope()
    await scope.__aenter__()
    with pytest.raises(ProviderReleaseError):
        await scope.__aexit__(None, None, None)


async def test_formulation_of_mananged(env: ManagedEnv, some_value: str) -> None:
    @env.provides(name=some_value)
    async def my_value() -> AsyncIterator[str]:
        yield some_value

    @env.managed
    async def async_func(
        value: str = DI.requires(my_value),
    ) -> str:
        return value

    @env.managed
    async def async_gen(
        value: str = DI.requires(my_value),
    ) -> AsyncIterator[str]:
        yield value

    @env.managed
    def gen(
        value: str = DI.requires(my_value),
    ) -> Iterator[str]:
        yield value

    @env.managed
    def func(
        value: str = DI.requires(my_value),
    ) -> str:
        return value

    async with env.managed_scope():
        assert func() == some_value
        assert next(gen()) == some_value
        assert await async_func() == some_value
        assert await async_gen().__anext__() == some_value
