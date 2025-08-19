from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ai2i.di.interface.models import RoundId
from ai2i.di.interface.scopes import ApplicationScopes
from ai2i.di.interface.tasks import TaskRunner
from ai2i.di.rounds_storage import RoundsStorage


@dataclass(frozen=True)
class RoundsManager:
    rounds_storage: RoundsStorage
    rounds_marked_for_destruction: set[RoundId]


class ApplicationContext(TaskRunner):
    """
    ApplicationContext is an extension of the ManagedEnv concept.
    In addition to the existing functionality on ManagedEnv, the Application context also
    adds the concept of `scopes`, allowing for groups of providers that are created and
    destroyed at different times
    """

    @abstractmethod
    def create_fresh_scopes_context(self, *, patched_instances: dict[str, Any]) -> None: ...

    @property
    @abstractmethod
    def scopes(self) -> ApplicationScopes: ...

    @scopes.setter
    @abstractmethod
    def scopes(self, scopes: ApplicationScopes) -> None: ...

    # ~ Round Scope Management ~ #

    @abstractmethod
    async def open_round_scope(self, round_id: RoundId) -> None: ...

    @abstractmethod
    async def reopen_round_scope(self, round_id: RoundId) -> None: ...

    @abstractmethod
    async def continue_round_scope(self) -> None: ...

    @abstractmethod
    async def close_round_scope(self) -> None: ...

    @abstractmethod
    async def destroy_round_scope(self, round_id: RoundId) -> None: ...
