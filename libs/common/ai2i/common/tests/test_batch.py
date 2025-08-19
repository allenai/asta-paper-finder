import asyncio
from typing import AsyncIterator, Sequence

import pytest

from ai2i.common.utils.batch import batch_process, with_batch


@pytest.mark.asyncio
async def test_batch_process_basic() -> None:
    async def process_batch(batch: Sequence[int]) -> Sequence[int]:
        result = []
        for item in batch:
            await asyncio.sleep(0.01)
            result.append(item * 2)
        return result

    items = range(100)
    batch_size = 10

    results = await batch_process(items, batch_size, process_batch)

    assert len(results) == 100
    assert results == [i * 2 for i in range(100)]


@pytest.mark.asyncio
async def test_batch_process_uneven_batches() -> None:
    async def process_batch(batch: Sequence[int]) -> Sequence[int]:
        return [item * 2 for item in batch]

    items = range(25)
    batch_size = 10

    results = await batch_process(items, batch_size, process_batch)

    assert len(results) == 25
    assert results == [i * 2 for i in range(25)]


@pytest.mark.asyncio
async def test_batch_process_empty_input() -> None:
    async def process_batch(batch: Sequence[int]) -> Sequence[int]:
        return [item * 2 for item in batch]

    batch_size = 10

    results = await batch_process([], batch_size, process_batch)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_batch_process_exception_handling() -> None:
    async def process_batch(batch: Sequence[int]) -> Sequence[int]:
        result = []
        for item in batch:
            if item == 5:
                raise ValueError("Error processing item 5")
            result.append(item * 2)
        return result

    items = range(10)
    batch_size = 2

    with pytest.raises(ValueError, match="Error processing item 5"):
        await batch_process(items, batch_size, process_batch)


@pytest.mark.asyncio
async def test_batch_process_async_iterator() -> None:
    async def process_batch(batch: Sequence[int]) -> AsyncIterator[int]:
        for item in batch:
            await asyncio.sleep(0.01)
            yield item * 2

    items = range(50)
    batch_size = 5

    results = await batch_process(items, batch_size, process_batch)

    assert len(results) == 50
    assert results == [i * 2 for i in range(50)]


@pytest.mark.asyncio
async def test_batch_decorator() -> None:
    @with_batch(batch_size=3, max_concurrency=2)
    async def process_items(batch: Sequence[int]) -> AsyncIterator[int]:
        for item in batch:
            await asyncio.sleep(0.01)
            yield item * 2

    items = list(range(10))
    results = await process_items(items)

    assert len(results) == 10
    assert results == [i * 2 for i in range(10)]
