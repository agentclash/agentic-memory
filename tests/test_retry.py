"""Unit tests for the shared retry helper."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.retry import retry_with_exponential_backoff


def test_retry_with_exponential_backoff_retries_until_success():
    attempts = []
    sleeps = []

    def operation():
        attempts.append("try")
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    result = retry_with_exponential_backoff(
        operation,
        should_retry=lambda exc: isinstance(exc, RuntimeError),
        initial_delay_seconds=0.5,
        jitter_ratio=0.0,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]


def test_retry_with_exponential_backoff_stops_when_should_retry_is_false():
    attempts = []
    sleeps = []

    def operation():
        attempts.append("try")
        raise ValueError("fatal")

    try:
        retry_with_exponential_backoff(
            operation,
            should_retry=lambda exc: False,
            sleep=sleeps.append,
        )
        raise AssertionError("Expected retry_with_exponential_backoff to raise")
    except ValueError as exc:
        assert str(exc) == "fatal"

    assert len(attempts) == 1
    assert sleeps == []


if __name__ == "__main__":
    test_retry_with_exponential_backoff_retries_until_success()
    test_retry_with_exponential_backoff_stops_when_should_retry_is_false()
    print("All retry tests passed.")
