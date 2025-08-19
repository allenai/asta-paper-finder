# ruff: noqa: B008

import asyncio
import uuid
from asyncio import Task
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from fastapi.requests import Request

from ai2i.config import with_config_overrides
from ai2i.di.app_context import ApplicationContext, Module
from ai2i.di.factory.app_context import create_app_context
from ai2i.di.factory.modules import create_module
from ai2i.di.interface import builtin_deps
from ai2i.di.interface.errors import OutOfScopeDependencyError, RoundStorageError
from ai2i.di.interface.gateway import DI
from ai2i.di.interface.models import RequestAndBody, RoundId, TurnId


@pytest.fixture
def some_value() -> str:
    return f"value_{uuid.uuid4()}"


@pytest.fixture
def some_name() -> str:
    return f"name_{uuid.uuid4()}"


@pytest.fixture
def some_value2() -> str:
    return f"value2_{uuid.uuid4()}"


@pytest.fixture
def some_name2() -> str:
    return f"name2_{uuid.uuid4()}"


@pytest.fixture(scope="function")
async def app_ctx_and_module() -> tuple[ApplicationContext, Module]:
    module = create_module(name="test")
    return create_app_context(module), module


@pytest.fixture(scope="function")
async def app_ctx(app_ctx_and_module: tuple[ApplicationContext, Module]) -> ApplicationContext:
    return app_ctx_and_module[0]


@pytest.fixture(scope="function")
async def module(app_ctx_and_module: tuple[ApplicationContext, Module]) -> Module:
    return app_ctx_and_module[1]


async def test_simple_singleton_scoping(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str
) -> None:
    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    await app_ctx.scopes.singleton.open_scope()
    try:
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)
    finally:
        await app_ctx.scopes.singleton.close_scope()

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)


async def test_simple_request_scoping(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_request_and_body: RequestAndBody = Mock()

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="request", name=some_name2)
    async def request_value(request: RequestAndBody = DI.requires(builtin_deps.request)) -> MyValue:
        assert request == mock_request_and_body
        return MyValue(some_value2)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)

        await app_ctx.scopes.request.open_scope(mock_request_and_body)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(request_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.scopes.request.close_scope()

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)


async def test_request_scope_with_sub_task(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_request_and_body: RequestAndBody = Mock()

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="request", name=some_name2)
    async def request_value(request: RequestAndBody = DI.requires(builtin_deps.request)) -> MyValue:
        assert request == mock_request_and_body
        return MyValue(some_value2)

    task: Task[None] | None = None

    async with app_ctx.scopes.singleton.managed_scope():
        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)

        await app_ctx.scopes.request.open_scope(mock_request_and_body)

        try:
            assert DI.get_dependency(request_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)

            async def _sub_task() -> None:
                await asyncio.sleep(1)

            task = await app_ctx.create_task(_sub_task())
        finally:
            await app_ctx.scopes.request.close_scope()

        # should still be in scope as long as the sub task is running
        assert DI.get_dependency(request_value) == MyValue(some_value2)
        assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)

        await asyncio.wait([task])

        # sub task finished so the scope should be destroyed
        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)


async def test_simple_turn_scoping(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_turn_id: TurnId = Mock()

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="turn", name=some_name2)
    async def turn_value(turn_id: TurnId = DI.requires(builtin_deps.turn_id)) -> MyValue:
        assert turn_id == mock_turn_id
        return MyValue(some_value2)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(turn_value)

        await app_ctx.scopes.turn.open_scope(mock_turn_id)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(turn_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.scopes.turn.close_scope()

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(turn_value)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)


async def test_turn_scope_with_sub_task(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_turn_id: Request = Mock()

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="turn", name=some_name2)
    async def request_value(request: TurnId = DI.requires(builtin_deps.turn_id)) -> MyValue:
        assert request == mock_turn_id
        return MyValue(some_value2)

    task: Task[None] | None = None

    async with app_ctx.scopes.singleton.managed_scope():
        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)

        await app_ctx.scopes.turn.open_scope(mock_turn_id)

        try:
            assert DI.get_dependency(request_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)

            async def _sub_task() -> None:
                await asyncio.sleep(1)

            task = await app_ctx.create_task(_sub_task())
        finally:
            await app_ctx.scopes.turn.close_scope()

        # should still be in scope as long as the sub task is running
        assert DI.get_dependency(request_value) == MyValue(some_value2)
        assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)

        await asyncio.wait([task])

        # sub task finished so the scope should be destroyed
        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(request_value)


async def test_simple_round_scoping(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_round_id: RoundId = Mock()

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="round", name=some_name2)
    async def round_value(round_id: RoundId = DI.requires(builtin_deps.round_id)) -> MyValue:
        assert round_id == mock_round_id
        return MyValue(some_value2)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

        await app_ctx.open_round_scope(mock_round_id)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(round_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.close_round_scope()
            await app_ctx.destroy_round_scope(mock_round_id)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)


async def test_round_scoping_hibernate(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_round_id: RoundId = Mock()
    num_factory_calls = 0

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="round", name=some_name2)
    async def round_value(round_id: RoundId = DI.requires(builtin_deps.round_id)) -> MyValue:
        nonlocal num_factory_calls

        assert round_id == mock_round_id
        num_factory_calls += 1
        return MyValue(some_value2)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

        await app_ctx.open_round_scope(mock_round_id)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(round_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.close_round_scope()

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

        await asyncio.sleep(1)

        await app_ctx.reopen_round_scope(mock_round_id)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(round_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.close_round_scope()
            await app_ctx.destroy_round_scope(mock_round_id)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    assert num_factory_calls == 1


@with_config_overrides(di={"round_scope_timeout": 0})
async def test_round_scoping_timeout(
    app_ctx: ApplicationContext, module: Module, some_value: str, some_name: str, some_value2: str, some_name2: str
) -> None:
    mock_round_id: RoundId = Mock()
    num_factory_calls = 0

    @dataclass(frozen=True)
    class MyValue:
        val: str

    @module.provides(scope="singleton", name=some_name)
    async def singleton_value() -> MyValue:
        return MyValue(some_value)

    @module.provides(scope="round", name=some_name2)
    async def round_value(round_id: RoundId = DI.requires(builtin_deps.round_id)) -> MyValue:
        nonlocal num_factory_calls

        assert round_id == mock_round_id
        num_factory_calls += 1
        return MyValue(some_value2)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(singleton_value)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(singleton_value) == MyValue(some_value)
        assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

        await app_ctx.open_round_scope(mock_round_id)
        try:
            assert DI.get_dependency(singleton_value) == MyValue(some_value)
            assert DI.get_dependency_by_name(some_name) == MyValue(some_value)

            assert DI.get_dependency(round_value) == MyValue(some_value2)
            assert DI.get_dependency_by_name(some_name2) == MyValue(some_value2)
        finally:
            await app_ctx.close_round_scope()

        with pytest.raises(OutOfScopeDependencyError):
            DI.get_dependency(round_value)

        # let the round timeout
        await asyncio.sleep(1)

        with pytest.raises(RoundStorageError):
            await app_ctx.reopen_round_scope(mock_round_id)

    assert num_factory_calls == 1


async def test_scopes_injection_from_parent_singleton_request(
    app_ctx: ApplicationContext,
    module: Module,
    some_value: str,
    some_value2: str,
) -> None:
    mock_request: Request = Mock()

    @module.provides(scope="singleton")
    async def singleton_value() -> str:
        return some_value

    @module.provides(scope="request")
    async def request_value(sv: str = DI.requires(singleton_value)) -> tuple[str, str]:
        return (sv, some_value2)

    async with app_ctx.scopes.singleton.managed_scope():
        await app_ctx.scopes.request.open_scope(mock_request)

        assert DI.get_dependency(request_value) == (some_value, some_value2)

        await app_ctx.scopes.request.close_scope()


async def test_scopes_injection_from_parent_round_request(
    app_ctx: ApplicationContext,
    module: Module,
    some_value: str,
) -> None:
    mock_request_and_body: Request = Mock()
    mock_round_id: RoundId = Mock()

    @module.provides(scope="request")
    async def request_value(r: RequestAndBody = DI.requires(builtin_deps.request)) -> RequestAndBody:
        assert r == mock_request_and_body
        return r

    @module.provides(scope="round")
    async def round_value(r: RequestAndBody = DI.requires(request_value)) -> tuple[RequestAndBody, str]:
        return (r, some_value)

    async with app_ctx.scopes.singleton.managed_scope():
        await app_ctx.scopes.request.open_scope(mock_request_and_body)
        await app_ctx.open_round_scope(mock_round_id)
        assert DI.get_dependency(round_value) == (mock_request_and_body, some_value)
        await app_ctx.close_round_scope()
        await app_ctx.scopes.request.close_scope()


async def test_get_application_context_as_dep(
    app_ctx: ApplicationContext,
    module: Module,
) -> None:
    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(builtin_deps.application_context)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(builtin_deps.application_context) == app_ctx


async def test_require_application_context_as_transitive_dep(
    app_ctx: ApplicationContext,
    module: Module,
    some_value: str,
) -> None:
    @module.provides(scope="singleton")
    async def some_value_with_context(
        app_ctx: ApplicationContext = DI.requires(builtin_deps.application_context),
    ) -> tuple[ApplicationContext, str]:
        return (app_ctx, some_value)

    with pytest.raises(OutOfScopeDependencyError):
        DI.get_dependency(builtin_deps.application_context)

    async with app_ctx.scopes.singleton.managed_scope():
        assert DI.get_dependency(some_value_with_context) == (app_ctx, some_value)
