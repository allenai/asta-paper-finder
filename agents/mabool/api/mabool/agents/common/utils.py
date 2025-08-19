from functools import wraps
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def log_args(log_function: Callable) -> Callable[..., Any]:
    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        @wraps(f)
        def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
            log_function(f"{f.__qualname__} >> {args}, {kwargs}")
            result = f(*args, **kwargs)
            log_function(f"{f.__qualname__} << {result}")
            return result

        return wrap

    return decorator


def alog_args(log_function: Callable) -> Callable[..., Any]:
    def decorator(f: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(f)
        async def wrap(*args: P.args, **kwargs: P.kwargs) -> T:
            log_function(f"{f.__qualname__} >> {args}, {kwargs}")
            result = await f(*args, **kwargs)
            log_function(f"{f.__qualname__} << {result}")
            return result

        return wrap

    return decorator
