from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
)

T = TypeVar("T")
P = ParamSpec("P")

Priority = Literal[0, 1, 2]

HIGH_PRIORITY: Priority = 0
DEFAULT_PRIORITY: Priority = 1
LOW_PRIORITY: Priority = 2

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_ticket: asyncio.Future = field(compare=False)


class PrioritySemaphore:
    def __init__(self, concurrency: int = 1) -> None:
        self._available_concurrency: int = concurrency
        self._queue: asyncio.PriorityQueue[PrioritizedTask] = asyncio.PriorityQueue()

    async def acquire(self, priority: int = DEFAULT_PRIORITY) -> None:
        if self._available_concurrency > 0:
            self._available_concurrency -= 1
            return

        if priority in [HIGH_PRIORITY, DEFAULT_PRIORITY] and self._queue.qsize() > 30:
            queue_full_msg = "Too many tasks are waiting to be executed"
            logger.error(queue_full_msg)
            raise asyncio.QueueFull(queue_full_msg)

        task_ticket = asyncio.get_running_loop().create_future()
        logger.info(
            f"Task with priority {priority} is waiting to be executed. "
            f"Available concurrency: {self._available_concurrency}. "
            f"Queue size: {self._queue.qsize()}"
        )
        await self._queue.put(PrioritizedTask(priority, task_ticket))
        await task_ticket

    async def release(self) -> None:
        self._available_concurrency += 1

        if not self._queue.empty():
            prioritized_task = await self._queue.get()
            if not prioritized_task.task_ticket.done():
                prioritized_task.task_ticket.set_result(None)
                self._available_concurrency -= 1
                logger.info(f"Task with priority {prioritized_task.priority} is being executed")

    async def __aenter__(self) -> PrioritySemaphore:
        await self.acquire()
        return self

    async def __aexit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        await self.release()

    def priority_context(self, priority: int = DEFAULT_PRIORITY) -> AsyncContextManager[PrioritySemaphore]:
        @dataclass
        class PrioritySemaphoreContext:
            semaphore: PrioritySemaphore
            priority: int

            async def __aenter__(self) -> PrioritySemaphore:
                await self.semaphore.acquire(self.priority)
                return self.semaphore

            async def __aexit__(
                self,
                exc_type: Optional[type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[Any],
            ) -> None:
                await self.semaphore.release()

        return PrioritySemaphoreContext(semaphore=self, priority=priority)


def with_priority(
    func: Callable[P, Awaitable[T]], semaphore: PrioritySemaphore, priority: int = DEFAULT_PRIORITY
) -> Callable[P, Awaitable[T]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with semaphore.priority_context(priority):
            return await func(*args, **kwargs)

    return wrapper


def prioritized(
    semaphore: PrioritySemaphore, priority: int = DEFAULT_PRIORITY
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with semaphore.priority_context(priority):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
