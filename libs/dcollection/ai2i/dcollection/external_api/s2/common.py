from functools import wraps
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

from httpcore import TimeoutException as HTTPCoreTimeoutException
from httpx import RemoteProtocolError as HTTPXRemoteProtocolError
from httpx import TimeoutException as HTTPXTimeoutException
from semanticscholar.PaginatedResults import PaginatedResults
from semanticscholar.SemanticScholarException import (
    GatewayTimeoutException,
    InternalServerErrorException,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def s2_retry(
    retry_attempts: int = 3, max_sec_to_wait: int = 10
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            func_with_retry = retry(
                retry=retry_if_exception_type(
                    (
                        ConnectionRefusedError,
                        InternalServerErrorException,
                        GatewayTimeoutException,
                        HTTPCoreTimeoutException,
                        HTTPXTimeoutException,
                        HTTPXRemoteProtocolError,
                    )
                ),
                wait=wait_exponential(min=0.1, max=retry_attempts),
                stop=stop_after_attempt(max_sec_to_wait),
                reraise=True,
            )(func)
            return await func_with_retry(*args, **kwargs)

        return wrapped

    return decorator


class AsyncPaginatedResults(Generic[T], PaginatedResults):
    def __init__(self, results: PaginatedResults, max_results: int = 10000) -> None:
        super().__init__(
            requester=results._requester,
            data_type=results._data_type,
            url=results._url,
            query=results._query,
            fields=results._fields,
            limit=results._limit,
            headers=results._headers,
            max_results=max_results,
        )
        self._update_params(
            {
                "data": results.raw_data,
                "next": results.next,
                "offset": results.offset,
                "total": results.total,
                "continuation_token": results._continuation_token,
            }
        )
        self._results_yielded = 0
        self.max_results = max_results

    async def __aiter__(self) -> AsyncIterator[T]:
        for item in self.items:
            if self._results_yielded >= self.max_results:
                return
            yield item
            self._results_yielded += 1

        while self._has_next_page() and self._results_yielded < self.max_results:
            new_items = cast(list[T], await self._async_get_next_page())
            if not new_items:
                return
            for item in new_items:
                if self._results_yielded >= self.max_results:
                    return
                yield item
                self._results_yielded += 1

    def _has_next_page(self) -> bool:
        return super()._has_next_page() and self._results_yielded < self.max_results
