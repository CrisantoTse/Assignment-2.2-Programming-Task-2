
import time

from statistics import median

from .utils import _validate_integer

import platform
import numpy as np


def benchmark(func, *args, warmup=1, repeats=5, **kwargs):

    if not callable(func):
        raise TypeError("func must be callable.")
    """
    the times of warm up could be 0 but the formal test at least 1 times
    """
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        "times_seconds": times,
        "median_seconds": median(times),
        "min_seconds": min(times),
    }

def environment_info():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }