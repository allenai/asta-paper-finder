import asyncio
from typing import List, ParamSpec, TypeVar

import pytest

from mabool.services.prioritized_task import (
    PrioritySemaphore,
    prioritized,
    with_priority,
)

T = TypeVar("T")
P = ParamSpec("P")


@pytest.mark.asyncio
async def test_priority_semaphore_basic() -> None:
    semaphore = PrioritySemaphore(2)

    async def task(order: List[int], index: int) -> None:
        await semaphore.acquire()
        try:
            order.append(index)
            await asyncio.sleep(0.1)
        finally:
            await semaphore.release()

    order: List[int] = []
    await asyncio.gather(task(order, 1), task(order, 2), task(order, 3), task(order, 4))

    assert order == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_priority_semaphore_priorities() -> None:
    semaphore = PrioritySemaphore(1)

    async def dummy_task() -> None:
        await semaphore.acquire()
        try:
            await asyncio.sleep(0.2)
        finally:
            await semaphore.release()

    async def task(order: List[int], index: int, priority: int) -> None:
        await semaphore.acquire(priority)
        try:
            order.append(index)
            await asyncio.sleep(0.1)
        finally:
            await semaphore.release()

    order: List[int] = []
    await asyncio.gather(
        dummy_task(),
        task(order, 1, 2),  # Medium priority
        task(order, 2, 1),  # High priority
        task(order, 3, 0),  # Highest priority
        task(order, 4, 3),  # Lowest priority
    )

    assert order == [3, 2, 1, 4]


@pytest.mark.asyncio
async def test_with_priority_decorator() -> None:
    semaphore = PrioritySemaphore(1)
    order: List[int] = []

    async def dummy_task() -> None:
        await semaphore.acquire()
        try:
            await asyncio.sleep(0.2)
        finally:
            await semaphore.release()

    @prioritized(semaphore, 2)
    async def task1() -> None:
        order.append(1)
        await asyncio.sleep(0.1)

    @prioritized(semaphore, 1)
    async def task2() -> None:
        order.append(2)
        await asyncio.sleep(0.1)

    @prioritized(semaphore, 0)
    async def task3() -> None:
        order.append(3)
        await asyncio.sleep(0.1)

    await asyncio.gather(dummy_task(), task1(), task2(), task3())

    assert order == [3, 2, 1]


@pytest.mark.asyncio
async def test_priority_semaphore_exception_handling() -> None:
    semaphore = PrioritySemaphore(1)

    async def task() -> None:
        await semaphore.acquire()
        try:
            raise ValueError("Test exception")
        finally:
            await semaphore.release()

    with pytest.raises(ValueError, match="Test exception"):
        await task()

    # Check that the semaphore was properly released
    assert semaphore._available_concurrency == 1


@pytest.mark.asyncio
async def test_priority_semaphore_concurrency() -> None:
    semaphore = PrioritySemaphore(2)
    running_tasks = 0
    max_running_tasks = 0

    async def task() -> None:
        nonlocal running_tasks, max_running_tasks
        await semaphore.acquire()
        try:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)
            await asyncio.sleep(0.1)
        finally:
            running_tasks -= 1
            await semaphore.release()

    await asyncio.gather(*(task() for _ in range(5)))

    assert max_running_tasks == 2


@pytest.mark.asyncio
async def test_with_priority_decorator_with_arguments() -> None:
    semaphore = PrioritySemaphore(1)
    result: List[str] = []

    async def dummy_task() -> None:
        await semaphore.acquire()
        try:
            await asyncio.sleep(0.2)
        finally:
            await semaphore.release()

    @prioritized(semaphore, 2)
    async def greet(name: str) -> None:
        result.append(f"Hello, {name}!")
        await asyncio.sleep(0.1)

    await asyncio.gather(dummy_task(), greet("Alice"), greet("Bob"), greet("Charlie"))

    assert result == ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]


@pytest.mark.asyncio
async def test_with_priority_decorator_different_order() -> None:
    semaphore = PrioritySemaphore(1)
    order: List[int] = []

    async def dummy_task() -> None:
        await semaphore.acquire()
        try:
            await asyncio.sleep(0.2)
        finally:
            await semaphore.release()

    @prioritized(semaphore, 2)
    async def task1() -> None:
        order.append(1)
        await asyncio.sleep(0.1)

    @prioritized(semaphore, 1)
    async def task2() -> None:
        order.append(2)
        await asyncio.sleep(0.1)

    @prioritized(semaphore, 0)
    async def task3() -> None:
        order.append(3)
        await asyncio.sleep(0.1)

    await asyncio.gather(dummy_task(), task3(), task2(), task1())

    assert order == [3, 2, 1]


@pytest.mark.asyncio
async def test_to_priority_non_decorator() -> None:
    semaphore = PrioritySemaphore(1)
    order: List[int] = []

    async def dummy_task() -> None:
        await semaphore.acquire()
        try:
            await asyncio.sleep(0.2)
        finally:
            await semaphore.release()

    async def task(index: int) -> None:
        order.append(index)
        await asyncio.sleep(0.1)

    prioritized_task1 = with_priority(task, semaphore, 2)
    prioritized_task2 = with_priority(task, semaphore, 1)
    prioritized_task3 = with_priority(task, semaphore, 0)

    await asyncio.gather(dummy_task(), prioritized_task1(1), prioritized_task2(2), prioritized_task3(3))

    assert order == [3, 2, 1]
