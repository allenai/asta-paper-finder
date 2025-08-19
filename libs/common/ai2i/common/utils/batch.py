import asyncio
from functools import wraps
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Sequence,
    TypeVar,
    overload,
)

from ai2i.common.utils.asyncio import custom_gather

A = TypeVar("A")
B = TypeVar("B")
P = ParamSpec("P")


ProcessFunc = Callable[[Sequence[A]], Awaitable[Sequence[B]] | AsyncIterator[B]]


class BatchProcessor(Generic[A, B]):
    def __init__(
        self,
        items: Sequence[A],
        batch_size: int,
        process_func: ProcessFunc[A, B],
        max_concurrency: int,
        force_deterministic: bool,
    ):
        self.items = items
        self.batch_size = batch_size
        self.process_func = process_func
        self.max_concurrency = max_concurrency
        self.force_deterministic = force_deterministic

    async def process(self) -> Sequence[B]:
        batches = list(self._create_batches())
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_batch(batch: Sequence[A]) -> Sequence[B]:
            async with semaphore:
                results = self.process_func(batch)
                if isinstance(results, AsyncIterator):
                    return [result async for result in results]
                elif isinstance(results, Awaitable):
                    return await results
                elif isinstance(results, Sequence):
                    return results
                else:
                    raise TypeError("process_func must return Iterable, Awaitable[Iterable], or AsyncIterator")

        batch_results = await custom_gather(
            *(process_batch(batch) for batch in batches),
            force_deterministic=self.force_deterministic,
        )

        return [item for batch in batch_results for item in batch]

    def _create_batches(self) -> Iterable[Sequence[A]]:
        batch: list[A] = []
        for item in self.items:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


@overload
async def batch_process(
    items: Sequence[A],
    batch_size: int,
    process_func: Callable[[Sequence[A]], Awaitable[Sequence[B]]],
    max_concurrency: int = 1000,
    force_deterministic: bool = False,
) -> Sequence[B]:
    pass


@overload
async def batch_process(
    items: Sequence[A],
    batch_size: int,
    process_func: Callable[[Sequence[A]], AsyncIterator[B]],
    max_concurrency: int = 1000,
    force_deterministic: bool = False,
) -> Sequence[B]:
    pass


async def batch_process(
    items: Sequence[A],
    batch_size: int,
    process_func: ProcessFunc[A, B],
    max_concurrency: int = 1000,
    force_deterministic: bool = False,
) -> Sequence[B]:
    processor = BatchProcessor(items, batch_size, process_func, max_concurrency, force_deterministic)
    return await processor.process()


Seq2SeqFunc = Callable[Concatenate[Sequence[A], P], Awaitable[Sequence[B]]]
Seq2IterFunc = Callable[Concatenate[Sequence[A], P], AsyncIterator[B]]


def with_batch(
    *,
    batch_size: int,
    max_concurrency: int,
    force_deterministic: bool = False,
) -> Callable[
    [Seq2SeqFunc[A, P, B] | Seq2IterFunc[A, P, B]],
    Seq2SeqFunc[A, P, B],
]:
    def decorator(
        func: Seq2SeqFunc[A, P, B] | Seq2IterFunc[A, P, B],
    ) -> Seq2SeqFunc[A, P, B]:
        @wraps(func)
        async def wrapper(items: Sequence[A], *args: P.args, **kwargs: P.kwargs) -> Sequence[B]:
            async def process_func(batch: Sequence[A]) -> Sequence[B]:
                result = func(batch, *args, **kwargs)
                if isinstance(result, AsyncIterator):
                    return [item async for item in result]
                elif isinstance(result, Awaitable):
                    return await result
                return result

            return await batch_process(items, batch_size, process_func, max_concurrency, force_deterministic)

        return wrapper

    return decorator
