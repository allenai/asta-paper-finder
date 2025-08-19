import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai2i.di.interface.models import CleanUpFunc, RoundId
from ai2i.di.managed_env import ManagedEnv


@dataclass(frozen=True)
class StorageEntry:
    env: ManagedEnv
    cleanup: CleanUpFunc | None
    timeout_task: asyncio.Task[None]


class RoundsStorage(ABC):
    @abstractmethod
    async def push(self, round_id: RoundId, env: ManagedEnv, cleanup: CleanUpFunc | None) -> None: ...

    @abstractmethod
    async def pop(self, round_id: RoundId) -> tuple[ManagedEnv, CleanUpFunc | None]: ...

    @abstractmethod
    async def destroy(self, round_id: RoundId) -> None: ...

    @abstractmethod
    async def destroy_all(self) -> None: ...
