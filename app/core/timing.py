"""
DocsQuery - Timing Utilities

Provides a small utility for measuring elapsed time.

We use perf_counter() because it is intended for measuring
elapsed durations.
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def measure_time() -> Iterator[dict[str, float]]:
    """
    Measure the execution time of a code block.

    Example:

        with measure_time() as timing:
            do_work()

        print(timing["duration_ms"])
    """

    start = perf_counter()

    data = {
        "duration_ms": 0.0,
    }

    try:
        yield data

    finally:
        data["duration_ms"] = (perf_counter() - start) * 1000
