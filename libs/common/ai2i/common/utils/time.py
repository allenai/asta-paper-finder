import logging
from datetime import UTC, datetime
from functools import wraps
from time import time
from typing import Awaitable, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)


def get_utc_time() -> datetime:
    # returns utc time without timezone info
    return datetime.now(UTC).replace(tzinfo=None)


T = TypeVar("T")
P = ParamSpec("P")


def timing(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
        ts = time()

        result = f(*args, **kwargs)
        te = time()
        logger.info("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def atiming(f: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @wraps(f)
    async def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
        ts = time()

        result = await f(*args, **kwargs)
        te = time()
        logger.info("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap
