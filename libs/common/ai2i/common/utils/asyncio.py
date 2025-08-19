import asyncio
import logging
from asyncio.futures import Future
from typing import Any, Awaitable, Literal, overload

logger = logging.getLogger(__name__)

type _Promise[A] = Future[A] | Awaitable[A]


@overload
async def custom_gather[T1, T2](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    /,
    *,
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, T2]: ...


@overload
async def custom_gather[T1, T2](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    /,
    *,
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[T1 | BaseException, T2 | BaseException]: ...


@overload
async def custom_gather[T1, T2, T3](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    /,
    *,
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, T2, T3]: ...


@overload
async def custom_gather[T1, T2, T3](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    /,
    *,
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[T1 | BaseException, T2 | BaseException, T3 | BaseException]: ...


@overload
async def custom_gather[T1, T2, T3, T4](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    /,
    *,
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, T2, T3, T4]: ...


@overload
async def custom_gather[T1, T2, T3, T4](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    /,
    *,
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[T1 | BaseException, T2 | BaseException, T3 | BaseException, T4 | BaseException]: ...


@overload
async def custom_gather[T1, T2, T3, T4, T5](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    pt5: _Promise[T5],
    /,
    *,
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, T2, T3, T4, T5]: ...


@overload
async def custom_gather[T1, T2, T3, T4, T5](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    pt5: _Promise[T5],
    /,
    *,
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[
    T1 | BaseException,
    T2 | BaseException,
    T3 | BaseException,
    T4 | BaseException,
    T5 | BaseException,
]: ...


@overload
async def custom_gather[T1, T2, T3, T4, T5, T6](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    pt5: _Promise[T5],
    pt6: _Promise[T6],
    /,
    *,
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, T2, T3, T4, T5, T6]: ...


@overload
async def custom_gather[T1, T2, T3, T4, T5, T6](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    pt4: _Promise[T4],
    pt5: _Promise[T5],
    pt6: _Promise[T6],
    /,
    *,
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[
    T1 | BaseException,
    T2 | BaseException,
    T3 | BaseException,
    T4 | BaseException,
    T5 | BaseException,
    T6 | BaseException,
]: ...


@overload
async def custom_gather[T1](
    *tasks: _Promise[T1],
    return_exceptions: Literal[False] = False,
    force_deterministic: bool,
) -> tuple[T1, ...]: ...


@overload
async def custom_gather[T1](
    *tasks: _Promise[T1],
    return_exceptions: bool,
    force_deterministic: bool,
) -> tuple[T1 | BaseException, ...]: ...


async def custom_gather(
    *tasks: _Promise[Any], return_exceptions: bool = False, force_deterministic: bool
) -> tuple[Any, ...]:
    if force_deterministic:
        results = []
        for task in tasks:
            try:
                results.append(await task)  # Run tasks serially
            except Exception as e:
                logger.exception(f"Failed to process task result for task: {task}, error: {e}")
                if return_exceptions:
                    results.append(e)
                else:
                    raise e
        return tuple(results)
    else:
        return tuple(await asyncio.gather(*tasks, return_exceptions=return_exceptions))  # Run tasks in parallel
