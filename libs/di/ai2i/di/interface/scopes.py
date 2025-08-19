from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, AsyncContextManager, Literal, Sequence

from ai2i.di.interface.models import (
    CachedValues,
    CleanUpFunc,
    DependencyDefinition,
    ProvidesDecorator,
    RequestAndBody,
    RoundId,
    Scope,
    TurnId,
)
from ai2i.di.managed_env import ManagedEnv
from ai2i.di.providers import Providers

ScopeOrCustom = Literal["custom"] | Scope


class ScopeContext(ABC):
    @abstractmethod
    def replace_env(self, env: ManagedEnv, cleanup: CleanUpFunc | None = None) -> None: ...

    @property
    @abstractmethod
    def is_active(self) -> bool: ...

    @property
    @abstractmethod
    def env(self) -> ManagedEnv | None: ...

    @property
    @abstractmethod
    def cleanup(self) -> CleanUpFunc | None: ...

    @property
    @abstractmethod
    def scope(self) -> ScopeOrCustom: ...


class SimpleScope(ScopeContext, ABC):
    @abstractmethod
    def managed_scope(self, values: CachedValues | None = None) -> AsyncContextManager[ManagedEnv]: ...

    @abstractmethod
    async def open_scope(self, values: CachedValues | None = None) -> ManagedEnv: ...

    @abstractmethod
    async def close_scope(self) -> None: ...


class NestingScope[A](ScopeContext, ABC):
    @abstractmethod
    async def open_scope(self, id: A) -> None: ...

    @abstractmethod
    async def continue_scope(self) -> None: ...

    @abstractmethod
    async def close_scope(self) -> None: ...


class HibernatingScope[A](ScopeContext, ABC):
    @abstractmethod
    async def open_scope(self, id: A) -> None: ...

    @abstractmethod
    async def continue_scope(self) -> None: ...

    @abstractmethod
    async def close_scope(self) -> tuple[ManagedEnv | None, CleanUpFunc | None]: ...

    @abstractmethod
    async def reopen_scope(self, env: ManagedEnv, cleanup: CleanUpFunc | None) -> None: ...


# class ApplicationScopes(ABC):
class ApplicationScopes(metaclass=ABCMeta):
    @property
    @abstractmethod
    def singleton(self) -> SimpleScope: ...

    @singleton.setter
    @abstractmethod
    def singleton(self, v: SimpleScope) -> None: ...

    @property
    @abstractmethod
    def custom(self) -> SimpleScope: ...

    @custom.setter
    @abstractmethod
    def custom(self, v: SimpleScope) -> None: ...

    @property
    @abstractmethod
    def request(self) -> NestingScope[RequestAndBody]: ...

    @request.setter
    @abstractmethod
    def request(self, v: NestingScope[RequestAndBody]) -> None: ...

    @property
    @abstractmethod
    def turn(self) -> NestingScope[TurnId]: ...

    @turn.setter
    @abstractmethod
    def turn(self, v: NestingScope[TurnId]) -> None: ...

    @property
    @abstractmethod
    def round(self) -> HibernatingScope[RoundId]: ...

    @round.setter
    @abstractmethod
    def round(self, v: HibernatingScope[RoundId]) -> None: ...

    @property
    @abstractmethod
    def envs(self) -> Sequence[ScopeContext]: ...

    @abstractmethod
    def resolve[A](self, definition: DependencyDefinition[A]) -> A: ...


class ProvidersPerScope(ABC):
    @property
    @abstractmethod
    def singleton(self) -> Providers: ...

    @property
    @abstractmethod
    def request(self) -> Providers: ...

    @property
    @abstractmethod
    def turn(self) -> Providers: ...

    @property
    @abstractmethod
    def round(self) -> Providers: ...

    @abstractmethod
    def provides(self, *, scope: Scope, name: str | None = None) -> ProvidesDecorator: ...

    @abstractmethod
    def chain_with(self, other: ProvidersPerScope) -> ProvidersPerScope: ...

    @abstractmethod
    def create_scopes(self, *, patched_instances: dict[str, Any]) -> ApplicationScopes: ...
