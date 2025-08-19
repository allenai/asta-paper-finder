from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import CoroutineType
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    ChainMap,
    Iterator,
    Literal,
    Mapping,
    NewType,
    Protocol,
    Self,
    Sequence,
)

from fastapi import Request

from ai2i.common.utils.value import ValueNotSet

type Scope = Literal["singleton", "request", "round", "turn"]

# a simple factory as an async function

type Func[**P, A] = Callable[P, A]
type AsyncFunc[**P, A] = Callable[P, CoroutineType[Any, Any, A]]
type AsyncFactory[A] = Callable[[], CoroutineType[Any, Any, A]]

# an async context manager styled factory that can do cleanup after yield

type ContextFunc[**P, A] = Callable[P, Iterator[A]]
type AsyncContextFunc[**P, A] = Callable[P, AsyncIterator[A]]
type AsyncContextFactory[A] = Callable[[], AsyncIterator[A]]


type DependencyName = str
type MutableCachedValues = dict[DependencyName, Any]
type CachedValues = Mapping[DependencyName, Any]
type CleanUpFunc = Callable[[], CoroutineType[Any, Any, None]]


class DependencyResolver(Protocol):
    def __call__[A](self, definition: DependencyDefinition[A]) -> A: ...


@dataclass(frozen=True)
class RequestAndBody:
    request: Request
    json_body: Mapping[str, Any]


@dataclass(frozen=True)
class DependencyDefinition[A]:
    unique_name: DependencyName
    create_context_manager: Callable[[DependencyResolver], AsyncContextManager[A]]
    depends_on: Sequence[DependencyName]

    @staticmethod
    def empty(name: DependencyName) -> DependencyDefinition:
        return DependencyDefinition(
            unique_name=name,
            create_context_manager=lambda _: nullcontext(),
            depends_on=[],
        )

    def without_dep(self, name: DependencyName) -> Self:
        return replace(self, depends_on=[d for d in self.depends_on if d != name])


@dataclass(frozen=True)
class ScopedDependencyDefinition[A](DependencyDefinition[A]):
    scope: Scope

    @staticmethod
    def predefined[R](t: type[R], name: str, scope: Scope) -> ScopedDependencyDefinition[R]:
        return ScopedDependencyDefinition(
            unique_name=name,
            create_context_manager=lambda _: nullcontext(),  # type:  ignore
            depends_on=[],
            scope=scope,
        )


type NamedDependenciesDict = dict[str, DependencyDefinition[Any]] | ChainMap[str, DependencyDefinition[Any]]


type DefaultFactory[A] = Callable[[], A]


class ProvidesDecorator(Protocol):
    def __call__[A](self, f: AsyncFunc[..., A] | AsyncContextFunc[..., A]) -> DependencyDefinition[A]: ...


@dataclass(frozen=True)
class DependencyPlaceholder[A]:
    definition: DependencyDefinition[A]
    default: A | ValueNotSet = ValueNotSet.instance()
    default_factory: DefaultFactory[A] | ValueNotSet = ValueNotSet.instance()

    def with_default[B](self, v: B) -> DependencyPlaceholder[A | B]:
        return DependencyPlaceholder(self.definition, default=v)

    def with_default_factory(self, factory: DefaultFactory) -> DependencyPlaceholder[A]:
        return DependencyPlaceholder(self.definition, default_factory=factory)


RoundId = NewType("RoundId", str)
TurnId = NewType("TurnId", str)
