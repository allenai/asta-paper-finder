from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Coroutine, final

from ai2i.di.context import ctx_active_scopes
from ai2i.di.factory.modules import create_module
from ai2i.di.interface import builtin_deps
from ai2i.di.interface.app_context import ApplicationContext, RoundsManager
from ai2i.di.interface.errors import OutOfScopeDependencyError
from ai2i.di.interface.models import (
    DependencyDefinition,
    RoundId,
)
from ai2i.di.interface.modules import Module
from ai2i.di.interface.scopes import ApplicationScopes, NestingScope
from ai2i.di.rounds_storage import RoundsStorageImpl

builtin_module = create_module("DI-Built-In")


@final
class ApplicationContextImpl(ApplicationContext):
    """
    ApplicationContext is an extension of the ManagedEnv concept.
    In addition to the existing functionality on ManagedEnv, the Application context also
    adds the concept of `scopes`, allowing for groups of providers that are created and
    destroyed at different times
    """

    _module: Module
    _rounds_manager_def: DependencyDefinition[RoundsManager]

    def __init__(self, module: Module) -> None:
        self._module = create_module("__intenal_module", extends=[module])

        # we'll manage the data needed for round scopes in the singleton scope
        # this way there is a natural point at which the round caches will be destroyed
        @self._module.providers.singleton.provides()
        async def _round_manager() -> AsyncIterator[RoundsManager]:
            rounds_storage = RoundsStorageImpl()
            yield RoundsManager(rounds_storage, set())
            await rounds_storage.destroy_all()

        self._rounds_manager_def = _round_manager

        if ctx_active_scopes.get() is None:
            self.create_fresh_scopes_context(patched_instances={})

    def create_fresh_scopes_context(self, *, patched_instances: dict[str, Any]) -> None:
        app_scopes = self._module.providers.create_scopes(
            patched_instances={builtin_deps.application_context.unique_name: self, **patched_instances}
        )
        ctx_active_scopes.set(app_scopes)

    @property
    def scopes(self) -> ApplicationScopes:
        active_scopes = ctx_active_scopes.get()
        if active_scopes is None:
            raise OutOfScopeDependencyError("No application context found to get scopes from")
        return active_scopes

    @scopes.setter
    def scopes(self, scopes: ApplicationScopes) -> None:
        ctx_active_scopes.set(scopes)

    async def create_task[A](self, coro: Coroutine[Any, Any, A]) -> asyncio.Task[A]:
        # continue all open scopes (we do it outside the task to avoid the passibility
        # of the current scope finishing without there being a yield to the inner scope
        # causing the release of the outer scope pre-maturely)

        active_nested_scopes = [e for e in self.scopes.envs if isinstance(e, NestingScope) and e.is_active]
        for scope in active_nested_scopes:
            await scope.continue_scope()

        # special handling for round
        is_rounds_env_active = self.scopes.round.is_active

        if is_rounds_env_active:
            await self.continue_round_scope()

        async def _run_with_finalized_scope() -> A:
            try:
                a = await coro
            finally:
                for scope in active_nested_scopes:
                    await scope.close_scope()

                # special handling for round
                if is_rounds_env_active:
                    await self.close_round_scope()
            return a

        return asyncio.create_task(_run_with_finalized_scope())

    # ~ Round Scope Management ~ #

    async def open_round_scope(self, round_id: RoundId) -> None:
        await self.scopes.round.open_scope(round_id)

    async def reopen_round_scope(self, round_id: RoundId) -> None:
        singleton_env = self.scopes.singleton.env
        if singleton_env is None:
            raise OutOfScopeDependencyError("Trying to re-open a round outside a singleton scope")

        round_manger = singleton_env.get_dependency(self._rounds_manager_def)
        env, cleanup = await round_manger.rounds_storage.pop(round_id)
        await self.scopes.round.reopen_scope(env, cleanup)

    async def continue_round_scope(self) -> None:
        await self.scopes.round.continue_scope()

    async def close_round_scope(self) -> None:
        singleton_env = self.scopes.singleton.env
        if singleton_env is None:
            raise OutOfScopeDependencyError("Trying to close round outside a singleton scope")

        rounds_manager = singleton_env.get_dependency(self._rounds_manager_def)
        env, cleanup = await self.scopes.round.close_scope()
        # this is the final closing scope, and we need to store it
        if env is not None:
            round_id = env.get_dependency(builtin_deps.round_id)
            await rounds_manager.rounds_storage.push(round_id, env, cleanup)

            # if round was marked for destruction, destroy it
            if round_id in rounds_manager.rounds_marked_for_destruction:
                await rounds_manager.rounds_storage.destroy(round_id)
                rounds_manager.rounds_marked_for_destruction.remove(round_id)

    async def destroy_round_scope(self, round_id: RoundId) -> None:
        singleton_env = self.scopes.singleton.env
        if singleton_env is None:
            raise OutOfScopeDependencyError("Trying to destroy round outside a singleton scope")

        rounds_manager = singleton_env.get_dependency(self._rounds_manager_def)
        if self.scopes.round.is_active:
            # the round is still active mark for destruction when it closes
            rounds_manager.rounds_marked_for_destruction.add(round_id)
        else:
            await rounds_manager.rounds_storage.destroy(round_id)
