import asyncio
from abc import ABC, abstractmethod
from typing import Any, Coroutine


class TaskRunner(ABC):
    @abstractmethod
    async def create_task[A](self, coro: Coroutine[Any, Any, A]) -> asyncio.Task[A]: ...
