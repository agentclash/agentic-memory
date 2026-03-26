"""Verify ConsoleLogger output formatting and stream routing."""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import ConsoleLogger, EventBus


def test_console_logger_formats_events():
    bus = EventBus()
    stream = io.StringIO()
    logger = ConsoleLogger(stream=stream)
    logger.register(bus)

    bus.emit(
        "memory.stored",
        {
            "record_id": "abc-123",
            "memory_type": "semantic",
            "content": "Python was created by Guido van Rossum",
        },
    )
    bus.emit(
        "memory.retrieved",
        {
            "query": "Who created Python?",
            "candidate_count": 2,
            "top_similarity": 0.95,
        },
    )
    bus.emit(
        "memory.ranked",
        {
            "query": "Who created Python?",
            "results": [
                {
                    "final_score": 0.85,
                }
            ],
        },
    )
    bus.emit(
        "memory.accessed",
        {
            "record_id": "abc-123",
            "access_count": 1,
        },
    )

    output = stream.getvalue().splitlines()
    assert output[0] == (
        "[EVENT] memory.stored | record_id=abc-123 | type=semantic | "
        "content=Python was created by Guido van Rossum"
    )
    assert output[1] == (
        "[EVENT] memory.retrieved | query='Who created Python?' | candidates=2 | top_sim=0.9500"
    )
    assert output[2] == (
        "[EVENT] memory.ranked | query='Who created Python?' | top_score=0.8500 | results=1"
    )
    assert output[3] == "[EVENT] memory.accessed | record_id=abc-123 | count=1"
    print("  PASS  ConsoleLogger formats event lines consistently")


if __name__ == "__main__":
    print("Logger tests:\n")
    test_console_logger_formats_events()
    print("\nAll tests passed.")
