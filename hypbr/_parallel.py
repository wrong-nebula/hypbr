"""Row-chunked thread parallelism for per-pixel NumPy kernels.

NumPy ufuncs release the GIL, so running independent row bands of a purely per-pixel kernel on a
small thread pool gives a multi-core speedup with bit-identical output. Thread count defaults to
min(4, cpu_count) and can be set with `HYPBR_THREADS` (1 disables threading).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

MIN_CHUNK_PX = 131072  # below ~2x this many pixels a single call is faster than splitting

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_in_worker = threading.local()


def n_threads() -> int:
    try:
        return max(1, int(os.environ.get("HYPBR_THREADS", min(4, os.cpu_count() or 1))))
    except ValueError:
        return 1


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=n_threads(), thread_name_prefix="hypbr-rows")
        return _pool


def row_slices(n_rows: int, row_px: int) -> list[slice]:
    """Split `n_rows` rows into contiguous bands of roughly MIN_CHUNK_PX pixels (>= 2 bands per thread)."""
    total = n_rows * max(row_px, 1)
    if n_threads() == 1 or n_rows < 2 or total < 2 * MIN_CHUNK_PX:
        return [slice(0, n_rows)]
    n_chunks = min(n_rows, max(2 * n_threads(), total // MIN_CHUNK_PX))
    step = -(-n_rows // n_chunks)
    return [slice(i, min(i + step, n_rows)) for i in range(0, n_rows, step)]


def map_rows(fn: Callable[[slice], T], slices: Sequence[slice]) -> list[T]:
    """Run `fn(slice)` for each band, in order. Serial when there is one band or we are already in a worker."""
    if len(slices) <= 1 or getattr(_in_worker, "active", False):
        return [fn(s) for s in slices]

    def run(s: slice) -> T:
        _in_worker.active = True
        try:
            return fn(s)
        finally:
            _in_worker.active = False

    return list(_get_pool().map(run, slices))
