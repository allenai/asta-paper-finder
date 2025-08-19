from __future__ import annotations

from collections import ChainMap
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ai2i.di.interface.errors import ProviderBuildError
from ai2i.di.interface.models import (
    CachedValues,
    CleanUpFunc,
    MutableCachedValues,
    ProvidesDecorator,
)
from ai2i.di.providers import Providers
from ai2i.di.resolver import GetDependencyOperations, ManagedOperations, ResolverFromCache
from ai2i.di.utils import dependency_topological_sort


class ManagedEnv(GetDependencyOperations, ManagedOperations):
    """
    A ManagedEnv unifies Providers under a single unbrella with a cache of objects
    constructed by these providers. It also manages a resolver that can take the provider definitions
    and use them in other coputations throught `ResolverOperations`.

    The ManagedEnv also provides a managed scope (AsyncContextManager) to define the code block
    for which the env will construct all provider (and later clean) defined values and will manage the injection
    of those values to whom ever required them.
    """

    _providers: Providers
    _cache: CachedValues
    _internal_cache: MutableCachedValues
    _already_constructed_keys: list[str]

    @staticmethod
    def simple() -> ManagedEnv:
        return ManagedEnv(Providers(), {})

    def __init__(self, providers: Providers, existing_cache: CachedValues) -> None:
        self._internal_cache = {}
        self._cache = ChainMap(self._internal_cache, {**existing_cache})

        resolver = ResolverFromCache(self._cache)
        GetDependencyOperations.__init__(self, resolver)
        ManagedOperations.__init__(self, resolver)

        self._providers = providers
        self._already_constructed_keys = list(existing_cache.keys())

    def clone_with(self, providers: Providers, pre_populated_cache: CachedValues | None = None) -> ManagedEnv:
        if pre_populated_cache is None:
            pre_populated_cache = {}

        return ManagedEnv(providers, {**pre_populated_cache, **self._cache})

    @property
    def cached_values(self) -> CachedValues:
        return self._cache

    @asynccontextmanager
    async def managed_scope(self) -> AsyncIterator[None]:
        clean = await self._build_all()
        try:
            yield
        finally:
            await clean()

    async def _build_all(self) -> CleanUpFunc:
        cleanups: list[CleanUpFunc] = []
        # validate all dependencies depend on providers defined in this env
        available_def_names = set(d.unique_name for d in self._providers) | set(self._already_constructed_keys)
        for d in self._providers:
            if any(dd not in available_def_names for dd in d.depends_on):
                unavailable_dep = next(filter(lambda dd: dd not in available_def_names, d.depends_on))
                raise ProviderBuildError(
                    "Attempting to build a provider with a non existant depenedncy, maybe a scope mismatch?"
                    + f" Missing dep: {unavailable_dep}",
                    d.unique_name,
                )

        # create dependencies in topological sort, so that we build things in correct order
        ordered_definitions = dependency_topological_sort(self._providers, self._already_constructed_keys)

        async def clean_all() -> None:
            # clean instances in the reversed order of their creation
            for clean in reversed(cleanups):
                await clean()

        try:
            for definition in ordered_definitions:
                if definition.unique_name not in self._already_constructed_keys:
                    value, clean = await self._providers.build_from_name(definition.unique_name, self._resolver)
                    self._internal_cache[definition.unique_name] = value

                    cleanups.append(clean)
        except Exception as e:
            # if the env fails to build a provider, make sure to cleanup all the providers that we already
            # created
            await clean_all()
            raise e

        return clean_all

    def provides(self, *, name: str | None = None) -> ProvidesDecorator:
        return self._providers.provides(name=name)
