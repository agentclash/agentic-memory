from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_with_exponential_backoff(
    operation: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    initial_delay_seconds: float = 0.25,
    backoff_multiplier: float = 2.0,
    max_delay_seconds: float = 2.0,
    jitter_ratio: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    # This helper is sync because the Gemini SDK call sites are sync today.
    # If embedding moves onto the async request path directly, wrap the full
    # operation in a threadpool rather than swapping this helper to async.
    delay = initial_delay_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not should_retry(exc):
                raise

            jitter = delay * jitter_ratio * random.random()
            sleep(min(delay + jitter, max_delay_seconds))
            delay = min(delay * backoff_multiplier, max_delay_seconds)

    raise RuntimeError("unreachable")
