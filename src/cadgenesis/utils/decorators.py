"""cadgenesis.utils.decorators
===========================
Reusable decorators for CADGenesis-LM v6.0: retry, memoize, timed, singleton,
deprecated, and synchronized.

Every decorator preserves the wrapped function's ``__name__``, ``__doc__`` and
signature metadata via ``functools.wraps``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import random
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R")

log = logging.getLogger(__name__)


def timed(enabled: bool = True) -> Callable[[F], F]:
    """Log the wall-clock duration of each call at INFO level.

    Args:
        enabled: When False the decorator is a no-op pass-through.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        if not enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log.info("%s took %.4fs", getattr(func, "__qualname__", func.__name__), elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator


def retry(
    attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[F], F]:
    """Retry a callable with exponential backoff.

    Args:
        attempts: Maximum number of attempts including the first.
        delay: Initial delay between attempts (seconds).
        backoff: Multiplier applied to the delay after each failure.
        exceptions: Exception type(s) that trigger a retry.
        on_retry: Optional callback ``(attempt_number, exception, next_delay)``.

    Raises:
        The last exception if all attempts are exhausted.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    if attempt >= attempts:
                        raise
                    next_delay = current_delay + random.uniform(0, current_delay * 0.1)
                    if on_retry is not None:
                        on_retry(attempt, exc, next_delay)
                    log.warning(
                        "Retrying %s (attempt %d/%d) after %s: %s",
                        getattr(func, "__qualname__", func.__name__),
                        attempt,
                        attempts,
                        exc,
                        next_delay,
                    )
                    time.sleep(next_delay)
                    current_delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


def memoize(func: F) -> F:
    """Cache return values keyed on positional + keyword arguments.

    Results are stored in an unbounded dict on the wrapper.  For bounded caches
    use ``functools.lru_cache`` instead.
    """
    cache: dict[Any, Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = _make_key(args, kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # type: ignore[attr-defined]
    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _make_key(args: tuple, kwargs: dict[str, Any]) -> tuple:
    """Build a hashable cache key from args/kwargs, falling back to repr."""
    try:
        return (args, tuple(sorted(kwargs.items())))
    except TypeError:
        return (
            tuple(repr(a) for a in args),
            tuple(sorted((k, repr(v)) for k, v in kwargs.items())),
        )


def singleton(cls: type[R]) -> type[R]:
    """Make a class a thread-safe singleton.

    The first ``cls()`` instantiation is kept and returned on every subsequent
    call; constructor arguments on later calls are ignored (a warning is logged
    when they differ from the initial ones).
    """
    _instances: dict[Any, R] = {}
    _lock = threading.Lock()

    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> R:
        with _lock:
            if cls not in _instances:
                _instances[cls] = cls(*args, **kwargs)
            elif args or kwargs:
                log.warning(
                    "Singleton %s already initialised; ignoring new arguments",
                    cls.__name__,
                )
            return _instances[cls]

    return get_instance  # type: ignore[return-value]


def deprecated(reason: str = "", version: str = "") -> Callable[[F], F]:
    """Mark a function or class as deprecated, emitting a ``DeprecationWarning``."""

    def decorator(obj: F) -> F:
        message = f"{getattr(obj, '__qualname__', obj.__name__)} is deprecated"
        if version:
            message += f" since {version}"
        if reason:
            message += f": {reason}"

        if inspect.isclass(obj):

            @functools.wraps(obj)
            def new_class(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return new_class  # type: ignore[return-value]

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def synchronized(func: F) -> F:
    """Serialize concurrent calls to a function with a per-function lock."""
    _lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _lock:
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def log_calls(level: int = logging.DEBUG) -> Callable[[F], F]:
    """Log entry/exit of the wrapped function at the given logging level."""
    logger = logging.getLogger("cadgenesis.utils.calls")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = getattr(func, "__qualname__", func.__name__)
            logger.log(level, "call %s(args=%r, kwargs=%r)", name, args, kwargs)
            result = func(*args, **kwargs)
            logger.log(level, "return %s -> %r", name, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def classmethod_dispatch(func: F) -> F:
    """Dispatch on the first argument's type using a ``_dispatch`` dict."""
    implementations: dict[type, Callable[..., Any]] = {}

    @functools.wraps(func)
    def wrapper(obj: Any, *args: Any, **kwargs: Any) -> Any:
        for cls in type(obj).__mro__:
            if cls in implementations:
                return implementations[cls](obj, *args, **kwargs)
        return func(obj, *args, **kwargs)

    wrapper.register = lambda cls, impl: implementations.__setitem__(cls, impl)  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
