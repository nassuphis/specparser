"""Numba-accelerated kernels for straddle schedule date expansion.

This module provides high-performance date expansion functions using Numba JIT compilation.
The main function `expand_months_to_date32` takes arrays of (year, month, month_count) and
produces arrays of (date32, parent_idx) suitable for Arrow table construction.

Performance characteristics:
- Sequential version: ~10-100x faster than Python loops
- Parallel version: Additional 2-4x on multi-core for large outputs
- Inner loop is bandwidth-bound (sequential memory writes)
- Uses Howard Hinnant's days_from_civil algorithm for O(1) date conversion
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# -----------------------------
# Calendar helpers (Numba-safe)
# -----------------------------

_DAYS_PER_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@njit(cache=True)
def is_leap_year(year: int) -> bool:
    """Fast leap-year test using bit operations.

    A year is a leap year if:
    - divisible by 4
    - not divisible by 100 unless divisible by 400
    """
    if (year & 3) != 0:  # year % 4 != 0
        return False
    if (year % 100) != 0:
        return True
    return (year % 400) == 0


@njit(cache=True)
def last_day_of_month(year: int, month: int) -> int:
    """Return number of days in given month (1-12)."""
    d = _DAYS_PER_MONTH[month - 1]
    if month == 2 and is_leap_year(year):
        return 29
    return d


@njit(cache=True)
def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    """Add k months to (year, month), handling year rollover.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        k: Number of months to add

    Returns:
        (new_year, new_month) tuple
    """
    t = year * 12 + (month - 1) + k
    yy = t // 12
    mm = (t - yy * 12) + 1
    return yy, mm


@njit(cache=True)
def ymd_to_date32(year: int, month: int, day: int) -> int:
    """Convert civil date to days since 1970-01-01 (proleptic Gregorian).

    Uses Howard Hinnant's days_from_civil algorithm - O(1), no loops.
    Returns an integer suitable for Arrow date32.

    Reference: https://howardhinnant.github.io/date_algorithms.html
    """
    y = year - (1 if month <= 2 else 0)
    if y >= 0:
        era = y // 400
    else:
        era = (y - 399) // 400
    yoe = y - era * 400
    mp = month + (12 if month <= 2 else 0) - 3  # Mar=0..Feb=11
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468  # 719468 aligns to 1970-01-01


# -----------------------------
# Sequential version
# -----------------------------


@njit(cache=True)
def expand_months_to_date32(
    year: np.ndarray, month: np.ndarray, month_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand (year, month, month_count) rows into date32 and parent_idx.

    Args:
        year: int32 array of entry years
        month: int32 array of entry months (1-12)
        month_count: int32 array of month spans (e.g., 2 for N, 3 for F)

    Returns:
        date32: int32 array of days since 1970-01-01 (Arrow date32 compatible)
        parent_idx: int32 array giving input row index for each output date

    Output is grouped into contiguous blocks per input row. Within each block,
    dates are in chronological order (ascending by month, then day).

    Example:
        year = [2025], month = [1], month_count = [2]
        -> 59 output rows (31 for Jan + 28 for Feb)
        -> parent_idx = [0, 0, ..., 0] (all 59 are from input row 0)
        -> date32 = [days for 2025-01-01 through 2025-02-28]
    """
    n = len(year)

    # 1) Compute total output size
    total = 0
    for i in range(n):
        span = month_count[i]
        if span <= 0:
            continue
        y0 = year[i]
        m0 = month[i]
        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            total += last_day_of_month(yy, mm)

    # 2) Allocate outputs
    date32 = np.empty(total, dtype=np.int32)
    parent_idx = np.empty(total, dtype=np.int32)

    # 3) Fill outputs - compute base date once per month, then base + j
    pos = 0
    for i in range(n):
        span = month_count[i]
        if span <= 0:
            continue
        y0 = year[i]
        m0 = month[i]

        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            last = last_day_of_month(yy, mm)

            # Base date for first day of month
            base = ymd_to_date32(yy, mm, 1)

            # Inner loop: sequential writes, just base + offset
            for j in range(last):
                date32[pos] = base + j
                parent_idx[pos] = i
                pos += 1

    return date32, parent_idx


# -----------------------------
# Parallel version (for millions of rows)
# -----------------------------


@njit(cache=True)
def _compute_starts(
    year: np.ndarray, month: np.ndarray, month_count: np.ndarray
) -> np.ndarray:
    """Compute start offset for each input row (prefix sum of output counts).

    Returns an array of length n+1 where starts[i] is the output position
    where input row i begins, and starts[n] is the total output size.
    """
    n = len(year)
    starts = np.empty(n + 1, dtype=np.int64)
    starts[0] = 0
    running = 0
    for i in range(n):
        span = month_count[i]
        cnt = 0
        if span > 0:
            y0 = year[i]
            m0 = month[i]
            for k in range(span):
                yy, mm = add_months(y0, m0, k)
                cnt += last_day_of_month(yy, mm)
        running += cnt
        starts[i + 1] = running
    return starts


@njit(cache=True, parallel=True)
def expand_months_to_date32_parallel(
    year: np.ndarray, month: np.ndarray, month_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Same as expand_months_to_date32, but parallel fill.

    Best when output is very large (millions+ rows).
    Pre-computes start offsets, then fills in parallel with prange.

    Args:
        year: int32 array of entry years
        month: int32 array of entry months (1-12)
        month_count: int32 array of month spans

    Returns:
        date32: int32 array of days since 1970-01-01
        parent_idx: int32 array giving input row index for each output date
    """
    n = len(year)
    starts = _compute_starts(year, month, month_count)
    total = starts[n]

    date32 = np.empty(total, dtype=np.int32)
    parent_idx = np.empty(total, dtype=np.int32)

    for i in prange(n):
        pos = starts[i]
        span = month_count[i]
        if span <= 0:
            continue

        y0 = year[i]
        m0 = month[i]

        for k in range(span):
            yy, mm = add_months(y0, m0, k)
            last = last_day_of_month(yy, mm)
            base = ymd_to_date32(yy, mm, 1)

            # Tight inner loop: sequential writes
            for j in range(last):
                date32[pos + j] = base + j
                parent_idx[pos + j] = i

            pos += last

    return date32, parent_idx
