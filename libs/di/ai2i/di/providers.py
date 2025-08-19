import inspect
from contextlib import asynccontextmanager, suppress
from functools import partial
from itertools import chain, zip_longest
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)

from ai2i.di.interface.errors import (
    DependencyDefinitionError,
    ProviderBuildError,
    ProviderReleaseError,
    UnreachableCodeBlockError,
)
from ai2i.di.interface.models import (
    AsyncContextFunc,
    AsyncFactory,
    AsyncFunc,
    CleanUpFunc,
    DependencyDefinition,
    DependencyName,
    DependencyPlaceholder,
    DependencyResolver,
    NamedDependenciesDict,
    ProvidesDecorator,
)
from ai2i.di.resolver import ManagedOperations


class Providers:
    """
    A collection of factories. The most general form of supported factory is an AsyncContextManager, meaning
    we support factories that are async in their creation and cleanup.

    This class contains the main utility decorator `provides` that allows for several forms of factories to
    be defined and registered into this collection.

    The factories are allowed to define dependecies on other values or config values in their parameter lists

    Allowed forms:
      1. Just an async factory (no cleanup needed)
        ```
            @env.provides()
            async def my_value(other_value: OtherValue = env.require(other)) -> MyValue:
                return await MyValue.build(other_value.val)

        ```
      2. Async factory with cleanup logic
        ```
            @env.provides()
            async def my_value(other_value: OtherValue = env.require(other)) -> AsyncIterator[MyValue]:
                value = await MyValue.build(other_value.val)
                yield value
                await value.close()
        ```
    """

    factories: NamedDependenciesDict

    def __init__(self, factories: NamedDependenciesDict | None = None) -> None:
        self.factories = {} if factories is None else factories

    async def build_from_name(self, name: DependencyName, resolver: DependencyResolver) -> tuple[Any, CleanUpFunc]:
        d = self.factories[name]
        ctx_mgr = d.create_context_manager(resolver)

        a = await ctx_mgr.__aenter__()

        async def clean(mgr: AsyncContextManager) -> None:
            await mgr.__aexit__(None, None, None)

        # We use partial apply the ctx_mgr instead of using a closure, the closure version will
        # capture the 'ctx_manager' mutable var which will cause all cleanups to clean only
        # the last item that was in the loop
        return a, partial(clean, ctx_mgr)

    def provides(self, *, name: str | None = None) -> ProvidesDecorator:
        def _internal[A](f: AsyncFunc[..., A] | AsyncContextFunc[..., A]) -> DependencyDefinition[A]:
            if not inspect.iscoroutinefunction(f) and not inspect.isasyncgenfunction(f):
                raise DependencyDefinitionError(
                    f"Providers must be 'async' functions or generators, unlike: {f.__qualname__}"
                )

            f_spec = inspect.getfullargspec(f)
            arg_names = f_spec.args

            arg_defaults = f_spec.defaults

            if arg_defaults is None:
                arg_defaults = ()

            args_and_defaults = {
                t[0]: t[1] for t in list(zip_longest(reversed(arg_names), reversed(list(arg_defaults))))
            }

            kwargs_defaults = f_spec.kwonlydefaults

            if kwargs_defaults is None:
                kwargs_defaults = {}

            kwargs_and_defaults = {**{k: None for k in f_spec.kwonlyargs}, **kwargs_defaults}

            all_defaults = chain(args_and_defaults.items(), kwargs_and_defaults.items())

            # validate all args are other Dependecies
            dependent_unique_names: list[DependencyName] = []
            for arg, default in all_defaults:
                if default is None:
                    raise DependencyDefinitionError(f"DI Provider '{f.__qualname__}', has a non bound argument '{arg}'")

                if isinstance(default, DependencyPlaceholder):
                    dependent_unique_names.append(default.definition.unique_name)

            # the factory itself is managed, since we want the ability to inject dependencies/config into the factory

            if inspect.iscoroutinefunction(f):
                return self.register_async_func(f, name=name, deps=dependent_unique_names)
            elif inspect.isasyncgenfunction(f):
                return self.register_async_ctx_manager(f, name=name, deps=dependent_unique_names)
            else:
                raise UnreachableCodeBlockError()

        return _internal

    def register_async_func[A](
        self,
        factory: AsyncFunc[..., A],
        *,
        name: str | None = None,
        deps: Sequence[DependencyName] | None = None,
    ) -> DependencyDefinition[A]:
        dependencies: Sequence[DependencyName] = deps or []

        provider_name = name or _get_qualified_name(factory)
        dep = DependencyDefinition(provider_name, _async_factory_adapter(factory, provider_name), dependencies)
        self._register_dependency(dep)
        return dep

    def register_async_ctx_manager[A](
        self,
        factory: AsyncContextFunc[..., A],
        *,
        name: str | None = None,
        deps: Sequence[DependencyName] | None = None,
    ) -> DependencyDefinition[A]:
        dependencies: Sequence[DependencyName] = deps or []

        provider_name = name or _get_qualified_name(factory)
        dep = DependencyDefinition(
            provider_name,
            _async_context_adapter(factory, provider_name),
            dependencies,
        )

        self._register_dependency(dep)
        return dep

    def _register_dependency[A](self, definition: DependencyDefinition[A]) -> None:
        if definition.unique_name in self.factories:
            raise DependencyDefinitionError(f"multiple factories with the name: '{definition.unique_name}' found")

        self.factories[definition.unique_name] = definition

    def __iter__(self) -> Iterator[DependencyDefinition[Any]]:
        for x in self.factories.values():
            yield x


def _async_factory_adapter[A](
    factory: AsyncFactory[A], provider_name: str
) -> Callable[[DependencyResolver], AsyncContextManager]:
    @asynccontextmanager
    async def _factory_wrapper(resolver: DependencyResolver) -> AsyncIterator[A]:
        managed_factory = ManagedOperations(resolver).managed(factory)
        try:
            value = await managed_factory()
        except Exception as e:
            raise ProviderBuildError(str(e), provider_name) from e
        yield value

    return _factory_wrapper


def _async_context_adapter[A](
    context_factory: AsyncContextFunc[..., A], provider_name: str
) -> Callable[[DependencyResolver], AsyncContextManager]:
    @asynccontextmanager
    async def _factory_wrapper(resolver: DependencyResolver) -> AsyncIterator[A]:
        managed_factory = ManagedOperations(resolver).managed(context_factory)
        values_iter = managed_factory()
        try:
            value = await anext(values_iter)  # create value
        except StopAsyncIteration:
            raise ProviderBuildError(
                "async iterator provider did not yield a value, maybe you're missing a `yield None`?", provider_name
            )
        except Exception as e:
            raise ProviderBuildError(str(e), provider_name) from e
        yield value
        with suppress(StopAsyncIteration):
            await anext(values_iter)  # destroy value
            raise ProviderReleaseError(
                "async iterator provider did not return after first yield, maybe you have an extra `yield`?",
                provider_name,
            )

    return _factory_wrapper


def _get_qualified_name[A](f: AsyncFunc[..., A] | AsyncContextFunc[..., A]) -> str:
    return f"{f.__module__}.{f.__qualname__}"
