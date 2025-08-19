from __future__ import annotations

import inspect
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
    cast,
    overload,
)

from ai2i.common.utils.value import ValueNotSet
from ai2i.config import (
    ConfigValuePlaceholder,
    Substitution,
    SubstitutionInfo,
    resolve_config_placeholder,
)
from ai2i.di.interface.errors import (
    OutOfScopeDependencyError,
)
from ai2i.di.interface.models import (
    AsyncContextFunc,
    AsyncFunc,
    CachedValues,
    ContextFunc,
    DefaultFactory,
    DependencyDefinition,
    DependencyPlaceholder,
    DependencyResolver,
    Func,
)


class ResolverFromCache:
    """
    A simple resolver of DependencyDefinitions that reads the values from the provided
    cache dictionary.

    A resolver is simply a function: DependencyDefinition[A] -> A
    """

    _cache: CachedValues

    def __init__(self, cache: CachedValues) -> None:
        self._cache = cache

    def __call__[A](self, definition: DependencyDefinition[A]) -> A:
        try:
            return cast(A, self._cache[definition.unique_name])
        except KeyError as e:
            raise OutOfScopeDependencyError(definition.unique_name) from e


class GetDependencyOperations:
    """
    A utility class to inherit from that exposes a number of convenience methods around the
    resolver.

    The operations it provides belong to two grous:
        - Direct retrieval using `get_dependency` and `get_dependency_by_name`

        Example:
            ```
            v = env.get_dependency(other)
            ```
    """

    _resolver: DependencyResolver

    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

    def get_dependency_by_name(
        self,
        name: str,
        /,
        default: Any | ValueNotSet = ValueNotSet.instance(),
        default_factory: DefaultFactory[Any] | ValueNotSet = ValueNotSet.instance(),
    ) -> Any:
        definition = DependencyDefinition.empty(name)
        return self.get_dependency(definition, default=default, default_factory=default_factory)

    def get_dependency[A, B](
        self,
        definition: DependencyDefinition[A],
        /,
        default: B | ValueNotSet = ValueNotSet.instance(),
        default_factory: DefaultFactory[A] | ValueNotSet = ValueNotSet.instance(),
    ) -> A | B:
        try:
            return self._resolver(definition)
        except OutOfScopeDependencyError as e:
            if not isinstance(default_factory, ValueNotSet):
                return default_factory()

            if not isinstance(default, ValueNotSet):
                return default

            raise e


class ManagedOperations:
    """
    A utility class to inherit from that exposes a number of convenience methods around the
    resolver.

        - Injection into an async function through the `managed` decorator

        Example:
            ```
            @env.managed
            async def add_other(v: int, other_value: OtherValue = env.require(other)) -> int:
                return await other_value.val + v
            ```
    """

    _substitutions: Sequence[Substitution]
    _resolver: DependencyResolver

    def __init__(self, resolver: DependencyResolver) -> None:
        self._resolver = resolver

        def _resolve_with_defaults(placeholder: DependencyPlaceholder[Any]) -> Any:
            try:
                return self._resolver(placeholder.definition)
            except OutOfScopeDependencyError as e:
                if not isinstance(placeholder.default_factory, ValueNotSet):
                    return placeholder.default_factory()

                if not isinstance(placeholder.default, ValueNotSet):
                    return placeholder.default

                raise e

        self._substitutions = [
            Substitution(DependencyPlaceholder, _resolve_with_defaults),
            Substitution(ConfigValuePlaceholder, resolve_config_placeholder),
        ]

    @overload
    def managed[**P, A](self, f: AsyncFunc[P, A]) -> AsyncFunc[P, A]:
        pass

    @overload
    def managed[**P, A](self, f: AsyncContextFunc[P, A]) -> AsyncContextFunc[P, A]:
        pass

    @overload
    def managed[**P, A](self, f: ContextFunc[P, A]) -> ContextFunc[P, A]:
        pass

    @overload
    def managed[**P, A](self, f: Func[P, A]) -> Func[P, A]:
        pass

    def managed(
        self, f: Func[..., Any] | AsyncFunc[..., Any] | ContextFunc[..., Any] | AsyncContextFunc[..., Any]
    ) -> Func[..., Any] | AsyncFunc[..., Any] | ContextFunc[..., Any] | AsyncContextFunc[..., Any]:
        if inspect.iscoroutinefunction(f):
            return self._managed_async_func(f)
        elif inspect.isasyncgenfunction(f):
            return self._managed_async_generator(f)
        elif inspect.isgeneratorfunction(f):
            return self._managed_generator(f)
        else:
            return self._managed_func(f)

    def _managed_async_func[**P, A](self, f: AsyncFunc[P, A]) -> AsyncFunc[P, A]:
        f_info = SubstitutionInfo.from_function(f)

        @wraps(f)
        async def _decorated(*args: P.args, **kwargs: P.kwargs) -> A:
            resolved_args, resolved_kwargs = f_info.resolve(self._substitutions, *args, **kwargs)
            return await f(*resolved_args, **resolved_kwargs)  # type: ignore

        return _decorated

    def _managed_func[**P, A](self, f: Func[P, A]) -> Func[P, A]:
        f_info = SubstitutionInfo.from_function(f)

        @wraps(f)
        def _decorated(*args: P.args, **kwargs: P.kwargs) -> A:
            resolved_args, resolved_kwargs = f_info.resolve(self._substitutions, *args, **kwargs)
            return f(*resolved_args, **resolved_kwargs)  # type: ignore

        return _decorated

    def _managed_async_generator[**P, A](self, f: AsyncContextFunc[P, A]) -> Callable[P, AsyncIterator[A]]:
        f_info = SubstitutionInfo.from_function(f)

        @wraps(f)
        async def _decorated(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[A]:
            resolved_args, resolved_kwargs = f_info.resolve(self._substitutions, *args, **kwargs)
            async for v in f(*resolved_args, **resolved_kwargs):  # type: ignore
                yield v

        return _decorated

    def _managed_generator[**P, A](self, f: ContextFunc[P, A]) -> ContextFunc[P, A]:
        f_info = SubstitutionInfo.from_function(f)

        @wraps(f)
        def _decorated(*args: P.args, **kwargs: P.kwargs) -> Iterator[A]:
            resolved_args, resolved_kwargs = f_info.resolve(self._substitutions, *args, **kwargs)
            for v in f(*resolved_args, **resolved_kwargs):  # type: ignore
                yield v

        return _decorated
