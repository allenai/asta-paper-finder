from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    ChainMap,
    Sequence,
    final,
)

from ai2i.di.context import ResolverFromContextVar
from ai2i.di.interface import builtin_deps
from ai2i.di.interface.errors import ManagedScopeError, OutOfScopeDependencyError, ProviderBuildError, ScopeAdapterError
from ai2i.di.interface.models import (
    CachedValues,
    CleanUpFunc,
    DependencyDefinition,
    DependencyResolver,
    ProvidesDecorator,
    RequestAndBody,
    RoundId,
    Scope,
    TurnId,
)
from ai2i.di.interface.scopes import (
    ApplicationScopes,
    HibernatingScope,
    NestingScope,
    ProvidersPerScope,
    ScopeContext,
    ScopeOrCustom,
    SimpleScope,
)
from ai2i.di.managed_env import ManagedEnv
from ai2i.di.providers import Providers


class ScopeContextImpl(ScopeContext):
    _scope: ScopeOrCustom
    _providers: Providers
    _env: ManagedEnv | None
    _cleanup: CleanUpFunc | None

    def __init__(self, scope: ScopeOrCustom, providers: Providers) -> None:
        self._scope = scope
        self._providers = providers
        self._env = None
        self._cleanup = None

    def replace_env(self, env: ManagedEnv, cleanup: CleanUpFunc | None = None) -> None:
        self._env = env
        self.cleaup = cleanup

    @property
    def is_active(self) -> bool:
        return self._env is not None

    @property
    def env(self) -> ManagedEnv | None:
        return self._env

    @property
    def cleanup(self) -> CleanUpFunc | None:
        return self._cleanup

    @property
    def scope(self) -> ScopeOrCustom:
        return self._scope


@final
class SimpleScopeImpl(ScopeContextImpl, SimpleScope):
    def __init__(
        self, scope: ScopeOrCustom, providers: Providers, patched_instances: dict[str, Any] | None = None
    ) -> None:
        super().__init__(scope, providers)

        if patched_instances is None:
            patched_instances = {}
        self._patched_instances = patched_instances

    @asynccontextmanager
    async def managed_scope(self, values: CachedValues | None = None) -> AsyncIterator[ManagedEnv]:
        env = await self.open_scope(values)
        try:
            yield env
        finally:
            await self.close_scope()

    async def open_scope(self, values: CachedValues | None = None) -> ManagedEnv:
        if self._env is not None:
            raise ManagedScopeError(f"Attempted to open a scope that is already active: '{self._scope}'")

        if values is None:
            values = {}

        env = ManagedEnv(self._providers, {**self._patched_instances, **values})
        mgr = env.managed_scope()
        await _aenter_with_error_handling(mgr, self)

        async def _cleanup() -> None:
            await mgr.__aexit__(None, None, None)

        self._env = env
        self._cleanup = _cleanup
        return env

    async def close_scope(self) -> None:
        if self._env is None:
            raise ManagedScopeError(f"Attempted to close scope that isn't active: '{self._scope}'")

        if self._cleanup is not None:
            await self._cleanup()

        self._cleanup = None
        self._env = None


@final
class NestingScopeImpl[A](ScopeContextImpl, NestingScope):
    _parent_scope: ScopeContext | None
    _active_scopes: int
    _scope_id_def: DependencyDefinition[A]

    def __init__(
        self,
        scope: ScopeOrCustom,
        providers: Providers,
        scope_id_def: DependencyDefinition[A],
        patched_instances: dict[str, Any] | None = None,
        /,
        parent_scope: ScopeContext | None = None,
    ) -> None:
        super().__init__(scope, providers)

        self._active_scopes = 0
        self._scope_id_def = scope_id_def
        self._parent_scope = parent_scope

        if patched_instances is None:
            patched_instances = {}
        self._patched_instances = patched_instances

    async def open_scope(self, id: A) -> None:
        if self._env is not None:
            raise ManagedScopeError(f"Attempted to open a scope that is already active: '{self._scope}'")

        scope_id_value = {self._scope_id_def.unique_name: id}
        parent_scope_values: CachedValues
        if self._parent_scope is None:
            parent_scope_values = {}
        else:
            env = self._parent_scope.env
            parent_scope_values = {} if env is None else env.cached_values

        env = ManagedEnv(self._providers, {**self._patched_instances, **scope_id_value, **parent_scope_values})
        mgr = env.managed_scope()

        await _aenter_with_error_handling(mgr, self)

        async def _cleanup() -> None:
            await mgr.__aexit__(None, None, None)

        self._env = env
        self._cleanup = _cleanup
        self._active_scopes = 1

    async def continue_scope(self) -> None:
        if self._env is None:
            raise ManagedScopeError(f"Attempted to continue a scope that isn't active: '{self._scope}'")

        self._active_scopes += 1

    async def close_scope(self) -> None:
        if self._env is None:
            raise ManagedScopeError(f"Attempted to close scope that isn't active: '{self._scope}'")

        self._active_scopes -= 1

        if self._active_scopes <= 0:
            self._env = None
            if self._cleanup is not None:
                await self._cleanup()
                self._cleanup = None


@final
class HibernatingScopeImpl[A](ScopeContextImpl, HibernatingScope):
    _parent_scope: ScopeContext | None
    _active_scopes: int
    _scope_id_def: DependencyDefinition[A]

    def __init__(
        self,
        scope: ScopeOrCustom,
        providers: Providers,
        scope_id_def: DependencyDefinition[A],
        patched_instances: dict[str, Any] | None = None,
        /,
        parent_scope: ScopeContext | None = None,
    ) -> None:
        super().__init__(scope, providers)

        self._active_scopes = 0
        self._scope_id_def = scope_id_def
        self._parent_scope = parent_scope

        if patched_instances is None:
            patched_instances = {}
        self._patched_instances = patched_instances

    async def open_scope(self, id: A) -> None:
        if self._env is not None:
            raise ManagedScopeError(f"Attempted to open a scope that is already active: '{self._scope}'")

        scope_id_value = {self._scope_id_def.unique_name: id}
        parent_scope_values: CachedValues
        if self._parent_scope is None:
            parent_scope_values = {}
        else:
            env = self._parent_scope.env
            parent_scope_values = {} if env is None else env.cached_values

        env = ManagedEnv(self._providers, {**self._patched_instances, **scope_id_value, **parent_scope_values})
        mgr = env.managed_scope()

        await _aenter_with_error_handling(mgr, self)

        async def _cleanup() -> None:
            await mgr.__aexit__(None, None, None)

        self._env = env
        self._cleanup = _cleanup
        self._active_scopes = 1

    async def continue_scope(self) -> None:
        if self._env is None:
            raise ManagedScopeError(f"Attempted to continue a scope that isn't active: '{self._scope}'")

        self._active_scopes += 1

    async def close_scope(self) -> tuple[ManagedEnv | None, CleanUpFunc | None]:
        """
        if this closes the final scope, return the env and its clean function
        instead of closing it (like in other scope implementations s)
        """
        if self._env is None:
            raise ManagedScopeError(f"Attempted to close scope that isn't active: '{self._scope}'")

        self._active_scopes -= 1

        if self._active_scopes <= 0:
            result = (self._env, self._cleanup)
            self._env = None
            self._cleanup = None
            return result
        return None, None

    async def reopen_scope(self, env: ManagedEnv, cleanup: CleanUpFunc | None) -> None:
        """
        open a new scope from the provided (pre-initialized) env with its clean function
        """
        if self._env is not None:
            raise ManagedScopeError(f"Attempted to re-open a scope that is already active: '{self._scope}'")

        self._env = env
        self._cleanup = cleanup
        self._active_scopes = 1


async def _aenter_with_error_handling(mgr: AsyncContextManager[None], scope: ScopeContext) -> None:
    try:
        await mgr.__aenter__()
    except ProviderBuildError as e:
        raise ManagedScopeError(
            f"Failed opening scope: '{scope.scope}', error in building the provider: {e.provider_name}, cause: {e}"
        ) from e


@final
class ApplicationScopesImpl(ApplicationScopes):
    _singleton: SimpleScope
    _custom: SimpleScope
    _request: NestingScope[RequestAndBody]
    _turn: NestingScope[TurnId]
    _round: HibernatingScope[RoundId]

    def __init__(
        self,
        singleton: SimpleScope,
        custom: SimpleScope,
        request: NestingScope[RequestAndBody],
        turn: NestingScope[TurnId],
        round: HibernatingScope[RoundId],
    ) -> None:
        self._singleton = singleton
        self._custom = custom
        self._request = request
        self._turn = turn
        self._round = round

    @property
    def singleton(self) -> SimpleScope:
        return self._singleton

    @singleton.setter
    def singleton(self, v: SimpleScope) -> None:
        self._singleton = v

    @property
    def custom(self) -> SimpleScope:
        return self._custom

    @custom.setter
    def custom(self, v: SimpleScope) -> None:
        self._custom = v

    @property
    def request(self) -> NestingScope[RequestAndBody]:
        return self._request

    @request.setter
    def request(self, v: NestingScope[RequestAndBody]) -> None:
        self._request = v

    @property
    def turn(self) -> NestingScope[TurnId]:
        return self._turn

    @turn.setter
    def turn(self, v: NestingScope[TurnId]) -> None:
        self._turn = v

    @property
    def round(self) -> HibernatingScope[RoundId]:
        return self._round

    @round.setter
    def round(self, v: HibernatingScope[RoundId]) -> None:
        self._round = v

    @property
    def envs(self) -> Sequence[ScopeContext]:
        return [
            self._custom,
            self._turn,
            self._request,
            self._round,
            self._singleton,
        ]

    def resolve[A](self, definition: DependencyDefinition[A]) -> A:
        # in order of lookup
        active_envs = [e.env for e in self.envs if e.env is not None]

        for active_env in active_envs:
            with suppress(OutOfScopeDependencyError):
                return active_env._resolver(definition)

        raise OutOfScopeDependencyError(definition.unique_name)


class ProvidersPerScopeImpl(ProvidersPerScope):
    _singleton: Providers
    _request: Providers
    _round: Providers
    _turn: Providers

    def __init__(
        self,
        *,
        singleton: Providers | None = None,
        request: Providers | None = None,
        round: Providers | None = None,
        turn: Providers | None = None,
    ) -> None:
        self._singleton = Providers() if singleton is None else singleton
        self._request = Providers() if request is None else request
        self._round = Providers() if round is None else round
        self._turn = Providers() if turn is None else turn

    @property
    def singleton(self) -> Providers:
        return self._singleton

    @property
    def request(self) -> Providers:
        return self._request

    @property
    def turn(self) -> Providers:
        return self._turn

    @property
    def round(self) -> Providers:
        return self._round

    def provides(self, *, scope: Scope, name: str | None = None) -> ProvidesDecorator:
        match scope:
            case "singleton":
                return self.singleton.provides(name=name)
            case "request":
                return self.request.provides(name=name)
            case "round":
                return self.round.provides(name=name)
            case "turn":
                return self.turn.provides(name=name)

    def chain_with(self, other: ProvidersPerScope) -> ProvidersPerScope:
        return ProvidersPerScopeImpl(
            singleton=Providers(ChainMap(self.singleton.factories, other.singleton.factories)),
            request=Providers(ChainMap(self.request.factories, other.request.factories)),
            round=Providers(ChainMap(self.round.factories, other.round.factories)),
            turn=Providers(ChainMap(self.turn.factories, other.turn.factories)),
        )

    def create_scopes(self, *, patched_instances: dict[str, Any]) -> ApplicationScopes:
        singleton = SimpleScopeImpl("singleton", self.singleton, patched_instances)
        custom = SimpleScopeImpl("custom", Providers(), patched_instances)
        request = NestingScopeImpl(
            "request", self.request, builtin_deps.request, patched_instances, parent_scope=singleton
        )
        round = HibernatingScopeImpl(
            "round", self.round, builtin_deps.round_id, patched_instances, parent_scope=request
        )

        turn = NestingScopeImpl("turn", self.turn, builtin_deps.turn_id, patched_instances, parent_scope=round)

        return ApplicationScopesImpl(
            singleton,
            custom,
            request,
            turn,
            round,
        )


class ScopeAdaptingDynamicProxy[A]:
    _dep: DependencyDefinition[A]
    _resolver: DependencyResolver

    def __init__(self, dep: DependencyDefinition[A]) -> None:
        self._dep = dep
        self._resolver = ResolverFromContextVar()

    def __getattr__(self, name: str) -> Any:
        try:
            real_dep = self._resolver(self._dep)
        except Exception as e:
            raise ScopeAdapterError(self._dep.unique_name) from e
        return getattr(real_dep, name)
