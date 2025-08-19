import logging
from asyncio.futures import Future
from typing import Any, Awaitable, Literal, overload

from ai2i.common.utils.asyncio import custom_gather as asyncio_custom_gather
from ai2i.config import config_value

from mabool.data_model.config import cfg_schema

logger = logging.getLogger(__name__)

type _Promise[A] = Future[A] | Awaitable[A]


@overload
async def custom_gather[T1, T2](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    /,
    *,
    return_exceptions: Literal[False] = False,
) -> tuple[T1, T2]: ...


@overload
async def custom_gather[T1, T2](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    /,
    *,
    return_exceptions: bool,
) -> tuple[T1 | BaseException, T2 | BaseException]: ...


@overload
async def custom_gather[T1, T2, T3](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    /,
    *,
    return_exceptions: Literal[False] = False,
) -> tuple[T1, T2, T3]: ...


@overload
async def custom_gather[T1, T2, T3](
    pt1: _Promise[T1],
    pt2: _Promise[T2],
    pt3: _Promise[T3],
    /,
    *,
    return_exceptions: bool,
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
) -> tuple[T1 | BaseException, T2 | BaseException, T3 | BaseException, T4 | BaseException, T5 | BaseException]: ...


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
) -> tuple[T1, ...]: ...


@overload
async def custom_gather[T1](
    *tasks: _Promise[T1],
    return_exceptions: bool,
) -> tuple[T1 | BaseException, ...]: ...


async def custom_gather(
    *tasks: _Promise[Any],
    return_exceptions: bool = False,
) -> tuple[Any, ...]:
    return await asyncio_custom_gather(
        *tasks, return_exceptions=return_exceptions, force_deterministic=config_value(cfg_schema.force_deterministic)
    )
