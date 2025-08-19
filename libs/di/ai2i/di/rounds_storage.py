import asyncio
from dataclasses import dataclass
from typing import final

from ai2i.config import config_value
from ai2i.di.config import cfg_schema
from ai2i.di.interface.errors import RoundStorageError
from ai2i.di.interface.models import CleanUpFunc, RoundId
from ai2i.di.interface.round_storage import RoundsStorage
from ai2i.di.managed_env import ManagedEnv


@dataclass(frozen=True)
class StorageEntry:
    env: ManagedEnv
    cleanup: CleanUpFunc | None
    timeout_task: asyncio.Task[None]


TWO_HOURS_IN_SECONDS = 60 * 60 * 2


@final
class RoundsStorageImpl(RoundsStorage):
    _rounds_cache: dict[RoundId, StorageEntry]

    def __init__(self) -> None:
        self._rounds_cache = {}

    async def push(self, round_id: RoundId, env: ManagedEnv, cleanup: CleanUpFunc | None) -> None:
        if round_id in self._rounds_cache:
            raise RoundStorageError(f"Can't add entry for {round_id=}, round already exists in storage")

        async def _timeout_destroy(timeout_seconds: int) -> None:
            await asyncio.sleep(timeout_seconds)
            await self.destroy(round_id)

        timeout = config_value(cfg_schema.di.round_scope_timeout, default=TWO_HOURS_IN_SECONDS)
        task = asyncio.create_task(_timeout_destroy(timeout))
        self._rounds_cache[round_id] = StorageEntry(env, cleanup, task)

    async def pop(self, round_id: RoundId) -> tuple[ManagedEnv, CleanUpFunc | None]:
        if round_id not in self._rounds_cache:
            raise RoundStorageError(f"Can't get round, entry for {round_id=} not found in storage ")
        entry = self._rounds_cache[round_id]
        del self._rounds_cache[round_id]
        entry.timeout_task.cancel()
        return (entry.env, entry.cleanup)

    async def destroy(self, round_id: RoundId) -> None:
        if round_id not in self._rounds_cache:
            raise RoundStorageError(f"Can't destroy round, entry for {round_id=} not found in storage ")

        entry = self._rounds_cache[round_id]
        del self._rounds_cache[round_id]
        entry.timeout_task.cancel()
        if entry.cleanup is not None:
            await entry.cleanup()

    async def destroy_all(self) -> None:
        for entry in self._rounds_cache.values():
            entry.timeout_task.cancel()
            if entry.cleanup is not None:
                await entry.cleanup()

        self._rounds_cache = {}
