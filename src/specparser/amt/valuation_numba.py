"""Numba-accelerated kernels for straddle valuation and backtesting.

This module provides high-performance functions for price lookup, model computation,
and PnL calculation in the backtest pipeline.

Functions are organized into:
- Math helpers: Normal CDF approximation for Black-Scholes
- Model: European Straddle pricing model
- Batch operations: Roll-forward, PnL computation, boundary detection
- Price lookup: Vectorized price matrix lookups
- Unified kernel: Full backtest computation in a single parallel kernel
- High-level API: get_straddle_backtests_numba() for batch backtest
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
from numba import njit, prange

# Import shared calendar helper from schedules_numba
from .schedules_numba import ymd_to_date32

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from . import prices as prices_module

# Re-export for backward compatibility
__all__ = [
    "ymd_to_date32",
    "model_ES_vectorized",
    "roll_forward_by_straddle",
    "compute_pnl_batch",
    "compute_straddle_boundaries",
    "compute_days_to_expiry",
    "batch_price_lookup",
    "build_col_indices",
    "full_backtest_kernel",
    # Sorted array lookup (for PricesNumba)
    "lookup_price_sorted",
    "batch_lookup_sorted",
    "batch_lookup_vol_hedge_sorted",
    "full_backtest_kernel_sorted",
    # High-level API
    "get_straddle_backtests_numba",
    "BacktestArraysSorted",
]


# -----------------------------
# Math helpers
# -----------------------------


@njit(cache=True)
def _norm_cdf_approx(x: float) -> float:
    """Fast approximation of standard normal CDF.

    Uses the error function approximation. Accurate to ~6 decimal places.
    This is equivalent to scipy.special.ndtr but works in Numba.
    """
    # erf approximation using Horner's method - good accuracy for |x| < 4
    # For |x| >= 4, CDF is essentially 0 or 1
    if x > 4.0:
        return 1.0
    if x < -4.0:
        return 0.0

    # Constants for erf approximation (Abramowitz and Stegun)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0 else -1.0
    x_abs = abs(x) / np.sqrt(2.0)

    t = 1.0 / (1.0 + p * x_abs)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * np.exp(-x_abs * x_abs)
    erf_val = sign * y

    return 0.5 * (1.0 + erf_val)


# -----------------------------
# Model
# -----------------------------


@njit(cache=True, parallel=True)
def model_ES_vectorized(
    hedge: np.ndarray,
    strike: np.ndarray,
    vol: np.ndarray,
    days_to_expiry: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized European Straddle model.

    Computes mark-to-market value and delta for all rows in parallel.

    Args:
        hedge: float64[N] - underlying price (S)
        strike: float64[N] - strike price (X)
        vol: float64[N] - implied volatility in percent (e.g., 25.0 for 25%)
        days_to_expiry: int32[N] - days until expiry
        valid_mask: bool[N] - True for rows that should be computed

    Returns:
        mv: float64[N] - mark-to-market value (normalized by strike), NaN for invalid
        delta: float64[N] - delta, NaN for invalid

    The formula is the standard Black-Scholes for a straddle:
        tv = (vol/100) * sqrt(t/365)
        d1 = ln(S/X) / tv + 0.5 * tv
        d2 = d1 - tv
        mv = S * (2*N(d1) - 1) - X * (2*N(d2) - 1)
        delta = 2*N(d1) - 1

    At expiry (t=0), intrinsic value is used:
        mv = |S - X| / X
        delta = +1 if S >= X else -1
    """
    n = len(hedge)
    mv = np.full(n, np.nan, dtype=np.float64)
    delta = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n):
        if not valid_mask[i]:
            continue

        S = hedge[i]
        X = strike[i]
        v = vol[i]
        t = days_to_expiry[i]

        # Validate inputs
        if np.isnan(S) or np.isnan(X) or np.isnan(v):
            continue
        if S <= 0.0 or X <= 0.0 or v <= 0.0 or t < 0:
            continue

        # At expiry: intrinsic value
        if t == 0:
            mv[i] = abs(S - X) / X
            delta[i] = 1.0 if S >= X else -1.0
            continue

        # Total volatility
        tv = (v / 100.0) * np.sqrt(float(t) / 365.0)

        if tv < 1e-10:  # Near-zero vol
            mv[i] = abs(S - X) / X
            delta[i] = 1.0 if S >= X else -1.0
            continue

        # d1, d2
        d1 = np.log(S / X) / tv + 0.5 * tv
        d2 = d1 - tv

        # N(d1), N(d2) for straddle (2x because call + put)
        N_d1 = 2.0 * _norm_cdf_approx(d1)  # 2*N(d1) for straddle
        N_d2 = 2.0 * _norm_cdf_approx(d2)  # 2*N(d2) for straddle

        # MV = S * N_d1 - X * N_d2 + X - S (straddle formula)
        mv_val = S * N_d1 - X * N_d2 + X - S
        mv[i] = mv_val / X  # Normalize by strike
        delta[i] = N_d1 - 1.0

    return mv, delta


# -----------------------------
# Batch operations
# -----------------------------


@njit(cache=True)
def roll_forward_by_straddle(
    values: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    ntry_offsets: np.ndarray,
    xpry_offsets: np.ndarray,
) -> np.ndarray:
    """Fill NaN with last valid value within each straddle's active range.

    Roll-forward logic: if a value is NaN on a given day, use the last
    non-NaN value from a previous day within the same straddle. Only
    applies within the active range [ntry_offset, xpry_offset].

    Args:
        values: float64[N] - input values with NaN for missing
        straddle_starts: int32[S] - start index in values array for each straddle
        straddle_lengths: int32[S] - number of days for each straddle
        ntry_offsets: int32[S] - entry offset within straddle (relative to start)
        xpry_offsets: int32[S] - expiry offset within straddle (relative to start)

    Returns:
        float64[N] - copy of values with NaN filled by roll-forward
    """
    result = values.copy()
    n_straddles = len(straddle_starts)

    for s in range(n_straddles):
        start = straddle_starts[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]

        if ntry < 0 or xpry < 0:
            continue

        ntry_abs = start + ntry
        xpry_abs = start + xpry

        # Get initial value at entry
        last_valid = result[ntry_abs]

        # Roll forward within active range
        for i in range(ntry_abs, xpry_abs + 1):
            if np.isnan(result[i]):
                result[i] = last_valid
            else:
                last_valid = result[i]

    return result


@njit(cache=True, parallel=True)
def compute_pnl_batch(
    mv: np.ndarray,
    delta: np.ndarray,
    hedge: np.ndarray,
    strike: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    ntry_offsets: np.ndarray,
    xpry_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PnL for all straddles in parallel.

    PnL formulas:
        opnl[i] = mv[i] - mv[i-1]  (option PnL)
        hpnl[i] = -delta[i-1] * (hedge[i] - hedge[i-1]) / strike  (hedge PnL)
        pnl[i] = opnl[i] + hpnl[i]  (total PnL)

    Entry day (ntry) has all PnL = 0.

    Args:
        mv: float64[N] - mark-to-market values
        delta: float64[N] - deltas
        hedge: float64[N] - hedge prices
        strike: float64[N] - strike prices (captured at entry)
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - days per straddle
        ntry_offsets: int32[S] - entry offset (relative to start)
        xpry_offsets: int32[S] - expiry offset (relative to start)

    Returns:
        opnl: float64[N] - option PnL
        hpnl: float64[N] - hedge PnL
        pnl: float64[N] - total PnL
    """
    n_days = len(mv)
    n_straddles = len(straddle_starts)

    opnl = np.full(n_days, np.nan, dtype=np.float64)
    hpnl = np.full(n_days, np.nan, dtype=np.float64)
    pnl = np.full(n_days, np.nan, dtype=np.float64)

    for s in prange(n_straddles):
        start = straddle_starts[s]
        ntry = ntry_offsets[s]
        xpry = xpry_offsets[s]

        if ntry < 0 or xpry < 0:
            continue

        ntry_abs = start + ntry
        xpry_abs = start + xpry
        strike_price = strike[ntry_abs]

        # Entry day: all PnL = 0
        opnl[ntry_abs] = 0.0
        hpnl[ntry_abs] = 0.0
        pnl[ntry_abs] = 0.0

        prev_mv = mv[ntry_abs]
        prev_delta = delta[ntry_abs]
        prev_hedge = hedge[ntry_abs]

        # Sequential within straddle (depends on previous day)
        for i in range(ntry_abs + 1, xpry_abs + 1):
            curr_mv = mv[i]
            curr_hedge = hedge[i]

            # Option PnL
            if not np.isnan(curr_mv) and not np.isnan(prev_mv):
                opnl[i] = curr_mv - prev_mv
            else:
                opnl[i] = np.nan

            # Hedge PnL
            if (not np.isnan(prev_delta) and
                not np.isnan(curr_hedge) and
                not np.isnan(prev_hedge) and
                strike_price > 0):
                hpnl[i] = -prev_delta * (curr_hedge - prev_hedge) / strike_price
            else:
                hpnl[i] = np.nan

            # Total PnL
            if not np.isnan(opnl[i]) and not np.isnan(hpnl[i]):
                pnl[i] = opnl[i] + hpnl[i]
            else:
                pnl[i] = np.nan

            # Update for next iteration
            if not np.isnan(curr_mv):
                prev_mv = curr_mv
            if not np.isnan(delta[i]):
                prev_delta = delta[i]
            if not np.isnan(curr_hedge):
                prev_hedge = curr_hedge

    return opnl, hpnl, pnl


@njit(cache=True)
def compute_straddle_boundaries(
    parent_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute start index and length for each straddle from parent_idx.

    The parent_idx array maps each day to its straddle index. This function
    finds where each straddle starts and how many days it contains.

    Args:
        parent_idx: int32[N] - straddle index for each day (must be sorted/grouped)

    Returns:
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days for each straddle
    """
    if len(parent_idx) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Count unique straddles
    n_straddles = parent_idx[-1] + 1

    straddle_starts = np.zeros(n_straddles, dtype=np.int32)
    straddle_lengths = np.zeros(n_straddles, dtype=np.int32)

    # Count lengths
    for i in range(len(parent_idx)):
        straddle_lengths[parent_idx[i]] += 1

    # Compute starts (prefix sum)
    running = 0
    for s in range(n_straddles):
        straddle_starts[s] = running
        running += straddle_lengths[s]

    return straddle_starts, straddle_lengths


@njit(cache=True)
def compute_days_to_expiry(
    dates: np.ndarray,
    straddle_starts: np.ndarray,
    straddle_lengths: np.ndarray,
    xpry_offsets: np.ndarray,
) -> np.ndarray:
    """Compute days to expiry for each row.

    Args:
        dates: int32[N] - date32 values for each day
        straddle_starts: int32[S] - start index for each straddle
        straddle_lengths: int32[S] - number of days for each straddle
        xpry_offsets: int32[S] - expiry offset within each straddle

    Returns:
        int32[N] - days until expiry for each row
    """
    n_days = len(dates)
    days_to_expiry = np.zeros(n_days, dtype=np.int32)
    n_straddles = len(straddle_starts)

    for s in range(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        xpry = xpry_offsets[s]

        if xpry < 0:
            continue

        xpry_date = dates[start + xpry]

        for i in range(length):
            idx = start + i
            days_to_expiry[idx] = xpry_date - dates[idx]

    return days_to_expiry


# -----------------------------
# Price matrix lookup kernels
# -----------------------------


@njit(cache=True, parallel=True)
def batch_price_lookup(
    price_matrix: np.ndarray,
    vol_row_idx: np.ndarray,
    hedge_row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch price lookup using pre-computed indices.

    This is the core optimization kernel - replaces Python dict lookups
    with direct array indexing in a parallel Numba kernel.

    Args:
        price_matrix: float64[n_ticker_fields, n_dates] - the price data
        vol_row_idx: int32[N] - row index for vol lookup (-1 if invalid)
        hedge_row_idx: int32[N] - row index for hedge lookup (-1 if invalid)
        col_idx: int32[N] - column index for date (-1 if invalid)

    Returns:
        vol: float64[N] - vol prices (NaN for invalid lookups)
        hedge: float64[N] - hedge prices (NaN for invalid lookups)

    Performance: O(N) array accesses in parallel, vs O(N) Python dict lookups.
    Expected speedup: 10-50x for large N.
    """
    n = len(col_idx)
    vol = np.full(n, np.nan, dtype=np.float64)
    hedge = np.full(n, np.nan, dtype=np.float64)

    n_rows = price_matrix.shape[0]
    n_cols = price_matrix.shape[1]

    for i in prange(n):
        c = col_idx[i]
        if c < 0 or c >= n_cols:
            continue

        # Vol lookup
        vr = vol_row_idx[i]
        if vr >= 0 and vr < n_rows:
            vol[i] = price_matrix[vr, c]

        # Hedge lookup
        hr = hedge_row_idx[i]
        if hr >= 0 and hr < n_rows:
            hedge[i] = price_matrix[hr, c]

    return vol, hedge


@njit(cache=True)
def build_col_indices(
    dates: np.ndarray,
    date32_to_col: np.ndarray,
    min_date32: int,
) -> np.ndarray:
    """Convert date32 values to column indices.

    Args:
        dates: int32[N] - date32 values
        date32_to_col: int32[M] - mapping from date32 offset to column index
        min_date32: int - minimum date32 value (for offset calculation)

    Returns:
        int32[N] - column indices (-1 for invalid dates)
    """
    n = len(dates)
    col_idx = np.full(n, -1, dtype=np.int32)
    n_mapping = len(date32_to_col)

    for i in range(n):
        offset = dates[i] - min_date32
        if offset >= 0 and offset < n_mapping:
            col_idx[i] = date32_to_col[offset]

    return col_idx


# -----------------------------
# Sorted array lookup (for PricesNumba)
# -----------------------------


@njit(cache=True)
def lookup_price_sorted(
    sorted_keys: np.ndarray,
    sorted_values: np.ndarray,
    ticker_idx: int,
    field_idx: int,
    date_offset: int,
    n_fields: int,
    n_dates: int,
) -> float:
    """Binary search lookup for a single price.

    Uses composite key: (ticker_idx * n_fields + field_idx) * n_dates + date_offset

    Args:
        sorted_keys: int64[N] - sorted composite keys
        sorted_values: float64[N] - values corresponding to sorted_keys
        ticker_idx: int - ticker index from PricesNumba.ticker_to_idx
        field_idx: int - field index from PricesNumba.field_to_idx
        date_offset: int - date offset (date32 - min_date32)
        n_fields: int - number of unique fields
        n_dates: int - date range span

    Returns:
        float - price value, or NaN if not found
    """
    if ticker_idx < 0 or field_idx < 0 or date_offset < 0 or date_offset >= n_dates:
        return np.nan

    key = (ticker_idx * n_fields + field_idx) * n_dates + date_offset

    # Binary search
    lo, hi = 0, len(sorted_keys)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid

    if lo < len(sorted_keys) and sorted_keys[lo] == key:
        return sorted_values[lo]
    return np.nan


@njit(cache=True, parallel=True)
def batch_lookup_sorted(
    sorted_keys: np.ndarray,
    sorted_values: np.ndarray,
    ticker_indices: np.ndarray,
    field_indices: np.ndarray,
    date_offsets: np.ndarray,
    n_fields: int,
    n_dates: int,
) -> np.ndarray:
    """Batch binary search lookup for multiple prices.

    Args:
        sorted_keys: int64[N] - sorted composite keys
        sorted_values: float64[N] - values corresponding to sorted_keys
        ticker_indices: int32[M] - ticker indices for each lookup
        field_indices: int32[M] - field indices for each lookup
        date_offsets: int32[M] - date offsets for each lookup
        n_fields: int - number of unique fields
        n_dates: int - date range span

    Returns:
        float64[M] - price values (NaN for not found)
    """
    n = len(ticker_indices)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n):
        result[i] = lookup_price_sorted(
            sorted_keys, sorted_values,
            ticker_indices[i], field_indices[i], date_offsets[i],
            n_fields, n_dates
        )

    return result


@njit(cache=True, parallel=True)
def batch_lookup_vol_hedge_sorted(
    sorted_keys: np.ndarray,
    sorted_values: np.ndarray,
    vol_ticker_indices: np.ndarray,
    vol_field_indices: np.ndarray,
    hedge_ticker_indices: np.ndarray,
    hedge_field_indices: np.ndarray,
    date_offsets: np.ndarray,
    n_fields: int,
    n_dates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch lookup for vol and hedge prices.

    Args:
        sorted_keys: int64[N] - sorted composite keys
        sorted_values: float64[N] - values corresponding to sorted_keys
        vol_ticker_indices: int32[M] - ticker indices for vol lookup
        vol_field_indices: int32[M] - field indices for vol lookup
        hedge_ticker_indices: int32[M] - ticker indices for hedge lookup
        hedge_field_indices: int32[M] - field indices for hedge lookup
        date_offsets: int32[M] - date offsets for each lookup
        n_fields: int - number of unique fields
        n_dates: int - date range span

    Returns:
        (vol, hedge) - float64[M] arrays of price values
    """
    n = len(date_offsets)
    vol = np.full(n, np.nan, dtype=np.float64)
    hedge = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n):
        date_off = date_offsets[i]
        if date_off < 0 or date_off >= n_dates:
            continue

        # Vol lookup
        vt = vol_ticker_indices[i]
        vf = vol_field_indices[i]
        if vt >= 0 and vf >= 0:
            vol[i] = lookup_price_sorted(
                sorted_keys, sorted_values, vt, vf, date_off, n_fields, n_dates
            )

        # Hedge lookup
        ht = hedge_ticker_indices[i]
        hf = hedge_field_indices[i]
        if ht >= 0 and hf >= 0:
            hedge[i] = lookup_price_sorted(
                sorted_keys, sorted_values, ht, hf, date_off, n_fields, n_dates
            )

    return vol, hedge


# -----------------------------
# Full backtest kernel (unified)
# -----------------------------


@njit(cache=True, parallel=True)
def full_backtest_kernel(
    # Price matrix
    price_matrix: np.ndarray,      # float64[R, C]
    date32_to_col: np.ndarray,     # int32[D]
    min_date32: int,

    # Per-straddle inputs (length S)
    vol_row_idx: np.ndarray,       # int32[S]
    hedge_row_idx: np.ndarray,     # int32[S]
    hedge1_row_idx: np.ndarray,    # int32[S] - price matrix row for hedge1 (-1 if not used)
    hedge2_row_idx: np.ndarray,    # int32[S] - price matrix row for hedge2 (-1 if not used)
    hedge3_row_idx: np.ndarray,    # int32[S] - price matrix row for hedge3 (-1 if not used)
    n_hedges: np.ndarray,          # int8[S] - number of hedge columns required (1-4)
    straddle_starts: np.ndarray,   # int32[S]
    straddle_lengths: np.ndarray,  # int32[S]
    ntry_anchor_date32: np.ndarray,  # int32[S] - entry anchor date (from override/code)
    xpry_anchor_date32: np.ndarray,  # int32[S] - expiry anchor date (from override/code)
    ntrv_offsets: np.ndarray,        # int32[S] - calendar days to add to entry anchor
    ntry_month_end: np.ndarray,      # int32[S] - last day of entry month (for fallback)
    xpry_month_end: np.ndarray,      # int32[S] - last day of expiry month (for fallback)

    # Per-day inputs (length N)
    dates: np.ndarray,             # int32[N]
):
    """Unified Numba kernel for full backtest computation.

    Combines all phases into a single parallel kernel:
    1. Price lookup (vol, hedge, hedge1, hedge2, hedge3)
    2. Find entry/expiry targets (with anchor + offset logic)
    3. Find first valid day at/after targets (with month-end fallback)
    4. Roll-forward and compute strike/strike1/strike2/strike3
    5. Model computation (Black-Scholes straddle)
    6. PnL computation

    Processing is parallel across straddles. Each straddle is processed
    independently, which is safe because straddles don't overlap in the
    output arrays.

    Args:
        price_matrix: float64[R, C] - the price data matrix
        date32_to_col: int32[D] - mapping from date32 offset to column index
        min_date32: int - minimum date32 value for offset calculation
        vol_row_idx: int32[S] - price matrix row index for vol (per straddle)
        hedge_row_idx: int32[S] - price matrix row index for hedge (per straddle)
        hedge1_row_idx: int32[S] - price matrix row for hedge1 (-1 if not used)
        hedge2_row_idx: int32[S] - price matrix row for hedge2 (-1 if not used)
        hedge3_row_idx: int32[S] - price matrix row for hedge3 (-1 if not used)
        n_hedges: int8[S] - number of hedge columns required (1-4)
        straddle_starts: int32[S] - start index in output arrays for each straddle
        straddle_lengths: int32[S] - number of days for each straddle
        ntry_anchor_date32: int32[S] - entry anchor date (from override/F/R/W/BD)
        xpry_anchor_date32: int32[S] - expiry anchor date (from override/F/R/W/BD)
        ntrv_offsets: int32[S] - calendar days to add to entry anchor
        ntry_month_end: int32[S] - last day of entry month (for fallback)
        xpry_month_end: int32[S] - last day of expiry month (for fallback)
        dates: int32[N] - date32 for each day

    Returns:
        Tuple of 18 arrays:
        - vol: float64[N] - raw vol prices
        - hedge: float64[N] - raw hedge prices
        - hedge1: float64[N] - raw hedge1 prices (NaN if not used)
        - hedge2: float64[N] - raw hedge2 prices (NaN if not used)
        - hedge3: float64[N] - raw hedge3 prices (NaN if not used)
        - ntry_offsets: int32[S] - entry offset per straddle (-1 if not found)
        - xpry_offsets: int32[S] - expiry offset per straddle (-1 if not found)
        - strike: float64[N] - strike price (hedge at entry)
        - strike1: float64[N] - strike1 price (hedge1 at entry)
        - strike2: float64[N] - strike2 price (hedge2 at entry)
        - strike3: float64[N] - strike3 price (hedge3 at entry)
        - days_to_expiry: int32[N] - days until expiry
        - mv: float64[N] - mark-to-market value (normalized by strike)
        - delta: float64[N] - option delta
        - opnl: float64[N] - option PnL
        - hpnl: float64[N] - hedge PnL
        - pnl: float64[N] - total PnL
        - action: int8[N] - action code (0=none, 1=ntry, 2=xpry)
    """
    n_days = len(dates)
    n_straddles = len(straddle_starts)
    n_rows = price_matrix.shape[0]
    n_cols = price_matrix.shape[1]
    n_mapping = len(date32_to_col)

    # Output arrays
    vol = np.full(n_days, np.nan, dtype=np.float64)
    hedge = np.full(n_days, np.nan, dtype=np.float64)
    hedge1 = np.full(n_days, np.nan, dtype=np.float64)
    hedge2 = np.full(n_days, np.nan, dtype=np.float64)
    hedge3 = np.full(n_days, np.nan, dtype=np.float64)
    ntry_offsets = np.full(n_straddles, -1, dtype=np.int32)
    xpry_offsets = np.full(n_straddles, -1, dtype=np.int32)
    strike = np.full(n_days, np.nan, dtype=np.float64)
    strike1 = np.full(n_days, np.nan, dtype=np.float64)
    strike2 = np.full(n_days, np.nan, dtype=np.float64)
    strike3 = np.full(n_days, np.nan, dtype=np.float64)
    days_to_expiry = np.zeros(n_days, dtype=np.int32)
    mv = np.full(n_days, np.nan, dtype=np.float64)
    delta = np.full(n_days, np.nan, dtype=np.float64)
    opnl = np.full(n_days, np.nan, dtype=np.float64)
    hpnl = np.full(n_days, np.nan, dtype=np.float64)
    pnl = np.full(n_days, np.nan, dtype=np.float64)
    action = np.zeros(n_days, dtype=np.int8)

    # Process each straddle in parallel
    for s in prange(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        vr = vol_row_idx[s]
        hr = hedge_row_idx[s]
        h1r = hedge1_row_idx[s]
        h2r = hedge2_row_idx[s]
        h3r = hedge3_row_idx[s]
        num_hedges = n_hedges[s]
        ntry_anchor = ntry_anchor_date32[s]
        xpry_anchor = xpry_anchor_date32[s]
        ntrv_off = ntrv_offsets[s]
        ntry_end = ntry_month_end[s]
        xpry_end = xpry_month_end[s]

        # --- PHASE 1: Price lookup ---
        for i in range(length):
            idx = start + i
            d = dates[idx]
            offset = d - min_date32
            if offset >= 0 and offset < n_mapping:
                c = date32_to_col[offset]
                if c >= 0 and c < n_cols:
                    if vr >= 0 and vr < n_rows:
                        vol[idx] = price_matrix[vr, c]
                    if hr >= 0 and hr < n_rows:
                        hedge[idx] = price_matrix[hr, c]
                    # Additional hedges (for CDS and calc assets)
                    if h1r >= 0 and h1r < n_rows:
                        hedge1[idx] = price_matrix[h1r, c]
                    if h2r >= 0 and h2r < n_rows:
                        hedge2[idx] = price_matrix[h2r, c]
                    if h3r >= 0 and h3r < n_rows:
                        hedge3[idx] = price_matrix[h3r, c]

        # --- PHASE 2: Find entry/expiry target dates ---
        # INT32_MAX sentinel means anchor lookup failed - skip this straddle
        INVALID_ANCHOR = 2147483647  # INT32_MAX
        if ntry_anchor == INVALID_ANCHOR or xpry_anchor == INVALID_ANCHOR:
            ntry_offsets[s] = -1
            xpry_offsets[s] = -1
            continue

        # Entry: anchor + ntrv_offset, clamped to month end
        ntry_target_date = ntry_anchor + ntrv_off
        if ntry_target_date > ntry_end:
            ntry_target_date = ntry_end

        # Expiry: anchor directly (no offset)
        xpry_target_date = xpry_anchor

        # Find first day index at or after target dates
        ntry_target = -1
        xpry_target = -1
        for i in range(length):
            d = dates[start + i]
            if ntry_target < 0 and d >= ntry_target_date:
                ntry_target = i
            if xpry_target < 0 and d >= xpry_target_date:
                xpry_target = i

        # --- PHASE 3: Find first valid day at/after targets ---
        # Validity check: all required params must be non-NaN
        # Skip check for params where row_idx == -1 (not required)
        # For entry: find first valid day, fallback to last good day in month
        ntry_off = -1
        if ntry_target >= 0:
            # Try to find first good day at or after target
            for i in range(ntry_target, length):
                idx = start + i
                d = dates[idx]
                # Don't look beyond entry month
                if d > ntry_end:
                    break
                # Check vol if required
                if vr >= 0 and np.isnan(vol[idx]):
                    continue
                # Check primary hedge if required
                if hr >= 0 and np.isnan(hedge[idx]):
                    continue
                # Check additional hedges based on num_hedges
                if num_hedges >= 2 and h1r >= 0 and np.isnan(hedge1[idx]):
                    continue
                if num_hedges >= 3 and h2r >= 0 and np.isnan(hedge2[idx]):
                    continue
                if num_hedges >= 4 and h3r >= 0 and np.isnan(hedge3[idx]):
                    continue
                ntry_off = i
                break

            # If not found, fallback to last good day in entry month
            if ntry_off < 0:
                for i in range(length - 1, -1, -1):
                    idx = start + i
                    d = dates[idx]
                    if d > ntry_end:
                        continue  # Skip days after month end
                    # Check vol if required
                    if vr >= 0 and np.isnan(vol[idx]):
                        continue
                    # Check primary hedge if required
                    if hr >= 0 and np.isnan(hedge[idx]):
                        continue
                    # Check additional hedges based on num_hedges
                    if num_hedges >= 2 and h1r >= 0 and np.isnan(hedge1[idx]):
                        continue
                    if num_hedges >= 3 and h2r >= 0 and np.isnan(hedge2[idx]):
                        continue
                    if num_hedges >= 4 and h3r >= 0 and np.isnan(hedge3[idx]):
                        continue
                    ntry_off = i
                    break

        # For expiry: find first valid day at or after target (within month)
        xpry_off = -1
        if xpry_target >= 0:
            for i in range(xpry_target, length):
                idx = start + i
                d = dates[idx]
                # Don't look beyond expiry month
                if d > xpry_end:
                    break
                # Check vol if required
                if vr >= 0 and np.isnan(vol[idx]):
                    continue
                # Check primary hedge if required
                if hr >= 0 and np.isnan(hedge[idx]):
                    continue
                # Check additional hedges based on num_hedges
                if num_hedges >= 2 and h1r >= 0 and np.isnan(hedge1[idx]):
                    continue
                if num_hedges >= 3 and h2r >= 0 and np.isnan(hedge2[idx]):
                    continue
                if num_hedges >= 4 and h3r >= 0 and np.isnan(hedge3[idx]):
                    continue
                xpry_off = i
                break

        ntry_offsets[s] = ntry_off
        xpry_offsets[s] = xpry_off

        if ntry_off < 0 or xpry_off < 0:
            continue  # Skip this straddle

        ntry_idx = start + ntry_off
        xpry_idx = start + xpry_off
        xpry_date = dates[xpry_idx]

        # Mark actions
        action[ntry_idx] = 1  # ntry
        action[xpry_idx] = 2  # xpry

        # --- PHASE 4: Roll-forward and compute strike ---
        strike_val = hedge[ntry_idx] if hr >= 0 else np.nan
        strike1_val = hedge1[ntry_idx] if h1r >= 0 else np.nan
        strike2_val = hedge2[ntry_idx] if h2r >= 0 else np.nan
        strike3_val = hedge3[ntry_idx] if h3r >= 0 else np.nan

        last_vol = vol[ntry_idx]
        last_hedge = hedge[ntry_idx]
        last_hedge1 = hedge1[ntry_idx]
        last_hedge2 = hedge2[ntry_idx]
        last_hedge3 = hedge3[ntry_idx]

        for i in range(ntry_off, xpry_off + 1):
            idx = start + i

            # Roll-forward vol
            if np.isnan(vol[idx]):
                vol[idx] = last_vol
            else:
                last_vol = vol[idx]

            # Roll-forward hedge
            if np.isnan(hedge[idx]):
                hedge[idx] = last_hedge
            else:
                last_hedge = hedge[idx]

            # Roll-forward hedge1 (if used)
            if h1r >= 0:
                if np.isnan(hedge1[idx]):
                    hedge1[idx] = last_hedge1
                else:
                    last_hedge1 = hedge1[idx]

            # Roll-forward hedge2 (if used)
            if h2r >= 0:
                if np.isnan(hedge2[idx]):
                    hedge2[idx] = last_hedge2
                else:
                    last_hedge2 = hedge2[idx]

            # Roll-forward hedge3 (if used)
            if h3r >= 0:
                if np.isnan(hedge3[idx]):
                    hedge3[idx] = last_hedge3
                else:
                    last_hedge3 = hedge3[idx]

            # Set strikes
            strike[idx] = strike_val
            if h1r >= 0:
                strike1[idx] = strike1_val
            if h2r >= 0:
                strike2[idx] = strike2_val
            if h3r >= 0:
                strike3[idx] = strike3_val

            # Days to expiry
            days_to_expiry[idx] = xpry_date - dates[idx]

        # --- PHASE 5: Model computation ---
        for i in range(ntry_off, xpry_off + 1):
            idx = start + i
            S = hedge[idx]
            X = strike[idx]
            v = vol[idx]
            t = days_to_expiry[idx]

            if S <= 0 or X <= 0 or v <= 0:
                continue

            if t == 0:
                mv[idx] = abs(S - X) / X
                delta[idx] = 1.0 if S >= X else -1.0
            else:
                tv = (v / 100.0) * np.sqrt(float(t) / 365.0)
                if tv < 1e-10:
                    mv[idx] = abs(S - X) / X
                    delta[idx] = 1.0 if S >= X else -1.0
                else:
                    d1 = np.log(S / X) / tv + 0.5 * tv
                    d2 = d1 - tv
                    N_d1 = 2.0 * _norm_cdf_approx(d1)
                    N_d2 = 2.0 * _norm_cdf_approx(d2)
                    mv_val = S * N_d1 - X * N_d2 + X - S
                    mv[idx] = mv_val / X
                    delta[idx] = N_d1 - 1.0

        # --- PHASE 6: PnL computation ---
        opnl[ntry_idx] = 0.0
        hpnl[ntry_idx] = 0.0
        pnl[ntry_idx] = 0.0

        prev_mv = mv[ntry_idx]
        prev_delta = delta[ntry_idx]
        prev_hedge = hedge[ntry_idx]

        for i in range(ntry_off + 1, xpry_off + 1):
            idx = start + i
            curr_mv = mv[idx]
            curr_hedge = hedge[idx]

            if not np.isnan(curr_mv) and not np.isnan(prev_mv):
                opnl[idx] = curr_mv - prev_mv

            if not np.isnan(prev_delta) and not np.isnan(curr_hedge) and not np.isnan(prev_hedge):
                hpnl[idx] = -prev_delta * (curr_hedge - prev_hedge) / strike_val

            if not np.isnan(opnl[idx]) and not np.isnan(hpnl[idx]):
                pnl[idx] = opnl[idx] + hpnl[idx]

            if not np.isnan(curr_mv):
                prev_mv = curr_mv
            if not np.isnan(delta[idx]):
                prev_delta = delta[idx]
            if not np.isnan(curr_hedge):
                prev_hedge = curr_hedge

    return (vol, hedge, hedge1, hedge2, hedge3,
            ntry_offsets, xpry_offsets,
            strike, strike1, strike2, strike3,
            days_to_expiry, mv, delta, opnl, hpnl, pnl, action)


# -----------------------------
# Unified kernel for sorted array price format
# -----------------------------


@njit(cache=True, parallel=True)
def full_backtest_kernel_sorted(
    # Sorted price arrays (from PricesNumba)
    sorted_keys: np.ndarray,       # int64[P] - composite keys
    sorted_values: np.ndarray,     # float64[P] - price values
    n_fields: int,
    n_dates: int,
    min_date32: int,

    # Per-straddle ticker/field indices (length S)
    vol_ticker_idx: np.ndarray,    # int32[S]
    vol_field_idx: np.ndarray,     # int32[S]
    hedge_ticker_idx: np.ndarray,  # int32[S]
    hedge_field_idx: np.ndarray,   # int32[S]
    hedge1_ticker_idx: np.ndarray, # int32[S] - hedge1 ticker index (-1 if not used)
    hedge1_field_idx: np.ndarray,  # int32[S]
    hedge2_ticker_idx: np.ndarray, # int32[S] - hedge2 ticker index (-1 if not used)
    hedge2_field_idx: np.ndarray,  # int32[S]
    hedge3_ticker_idx: np.ndarray, # int32[S] - hedge3 ticker index (-1 if not used)
    hedge3_field_idx: np.ndarray,  # int32[S]
    n_hedges: np.ndarray,          # int8[S] - number of hedge columns required (1-4)

    # Per-straddle inputs (length S)
    straddle_starts: np.ndarray,   # int32[S]
    straddle_lengths: np.ndarray,  # int32[S]
    ntry_anchor_date32: np.ndarray,  # int32[S] - entry anchor date (from override/code)
    xpry_anchor_date32: np.ndarray,  # int32[S] - expiry anchor date (from override/code)
    ntrv_offsets: np.ndarray,        # int32[S] - calendar days to add to entry anchor
    ntry_month_end: np.ndarray,      # int32[S] - last day of entry month (for fallback)
    xpry_month_end: np.ndarray,      # int32[S] - last day of expiry month (for fallback)

    # Per-day inputs (length N)
    dates: np.ndarray,             # int32[N]
):
    """Unified Numba kernel for backtest with sorted array price lookup.

    Same as full_backtest_kernel but uses binary search on sorted arrays
    instead of matrix indexing. Optimized for PyArrow-loaded price data.

    Phases:
    1. Price lookup (vol, hedge, hedge1, hedge2, hedge3) using binary search
    2. Find entry/expiry targets (anchor + offset logic)
    3. Find first valid day at/after targets (with month-end fallback)
    4. Roll-forward and compute strike/strike1/strike2/strike3
    5. Model computation (Black-Scholes straddle)
    6. PnL computation

    Returns:
        Tuple of 18 arrays (matching full_backtest_kernel):
        - vol: float64[N] - raw vol prices
        - hedge: float64[N] - raw hedge prices
        - hedge1: float64[N] - raw hedge1 prices (NaN if not used)
        - hedge2: float64[N] - raw hedge2 prices (NaN if not used)
        - hedge3: float64[N] - raw hedge3 prices (NaN if not used)
        - ntry_offsets: int32[S] - entry offset per straddle (-1 if not found)
        - xpry_offsets: int32[S] - expiry offset per straddle (-1 if not found)
        - strike: float64[N] - strike price (hedge at entry)
        - strike1: float64[N] - strike1 price (hedge1 at entry)
        - strike2: float64[N] - strike2 price (hedge2 at entry)
        - strike3: float64[N] - strike3 price (hedge3 at entry)
        - days_to_expiry: int32[N] - days until expiry
        - mv: float64[N] - mark-to-market value (normalized by strike)
        - delta: float64[N] - option delta
        - opnl: float64[N] - option PnL
        - hpnl: float64[N] - hedge PnL
        - pnl: float64[N] - total PnL
        - action: int8[N] - action code (0=none, 1=ntry, 2=xpry)
    """
    n_days = len(dates)
    n_straddles = len(straddle_starts)

    # Output arrays
    vol = np.full(n_days, np.nan, dtype=np.float64)
    hedge = np.full(n_days, np.nan, dtype=np.float64)
    hedge1 = np.full(n_days, np.nan, dtype=np.float64)
    hedge2 = np.full(n_days, np.nan, dtype=np.float64)
    hedge3 = np.full(n_days, np.nan, dtype=np.float64)
    ntry_offsets_out = np.full(n_straddles, -1, dtype=np.int32)
    xpry_offsets_out = np.full(n_straddles, -1, dtype=np.int32)
    strike = np.full(n_days, np.nan, dtype=np.float64)
    strike1 = np.full(n_days, np.nan, dtype=np.float64)
    strike2 = np.full(n_days, np.nan, dtype=np.float64)
    strike3 = np.full(n_days, np.nan, dtype=np.float64)
    days_to_expiry = np.zeros(n_days, dtype=np.int32)
    mv = np.full(n_days, np.nan, dtype=np.float64)
    delta = np.full(n_days, np.nan, dtype=np.float64)
    opnl = np.full(n_days, np.nan, dtype=np.float64)
    hpnl = np.full(n_days, np.nan, dtype=np.float64)
    pnl = np.full(n_days, np.nan, dtype=np.float64)
    action = np.zeros(n_days, dtype=np.int8)

    # Process each straddle in parallel
    for s in prange(n_straddles):
        start = straddle_starts[s]
        length = straddle_lengths[s]
        vt = vol_ticker_idx[s]
        vf = vol_field_idx[s]
        ht = hedge_ticker_idx[s]
        hf = hedge_field_idx[s]
        h1t = hedge1_ticker_idx[s]
        h1f = hedge1_field_idx[s]
        h2t = hedge2_ticker_idx[s]
        h2f = hedge2_field_idx[s]
        h3t = hedge3_ticker_idx[s]
        h3f = hedge3_field_idx[s]
        num_hedges = n_hedges[s]
        ntry_anchor = ntry_anchor_date32[s]
        xpry_anchor = xpry_anchor_date32[s]
        ntrv_off = ntrv_offsets[s]
        ntry_end = ntry_month_end[s]
        xpry_end = xpry_month_end[s]

        # --- PHASE 1: Price lookup using binary search ---
        for i in range(length):
            idx = start + i
            d = dates[idx]
            date_offset = d - min_date32

            if date_offset >= 0 and date_offset < n_dates:
                # Vol lookup
                if vt >= 0 and vf >= 0:
                    vol[idx] = lookup_price_sorted(
                        sorted_keys, sorted_values, vt, vf, date_offset, n_fields, n_dates
                    )

                # Hedge lookup
                if ht >= 0 and hf >= 0:
                    hedge[idx] = lookup_price_sorted(
                        sorted_keys, sorted_values, ht, hf, date_offset, n_fields, n_dates
                    )

                # Hedge1 lookup
                if h1t >= 0 and h1f >= 0:
                    hedge1[idx] = lookup_price_sorted(
                        sorted_keys, sorted_values, h1t, h1f, date_offset, n_fields, n_dates
                    )

                # Hedge2 lookup
                if h2t >= 0 and h2f >= 0:
                    hedge2[idx] = lookup_price_sorted(
                        sorted_keys, sorted_values, h2t, h2f, date_offset, n_fields, n_dates
                    )

                # Hedge3 lookup
                if h3t >= 0 and h3f >= 0:
                    hedge3[idx] = lookup_price_sorted(
                        sorted_keys, sorted_values, h3t, h3f, date_offset, n_fields, n_dates
                    )

        # --- PHASE 2: Find entry/expiry target dates ---
        # INT32_MAX sentinel means anchor lookup failed - skip this straddle
        INVALID_ANCHOR = 2147483647  # INT32_MAX
        if ntry_anchor == INVALID_ANCHOR or xpry_anchor == INVALID_ANCHOR:
            ntry_offsets_out[s] = -1
            xpry_offsets_out[s] = -1
            continue

        # Entry: anchor + ntrv_offset, clamped to month end
        ntry_target_date = ntry_anchor + ntrv_off
        if ntry_target_date > ntry_end:
            ntry_target_date = ntry_end

        # Expiry: anchor directly (no offset)
        xpry_target_date = xpry_anchor

        # Find first day index at or after target dates
        ntry_target = -1
        xpry_target = -1
        for i in range(length):
            d = dates[start + i]
            if ntry_target < 0 and d >= ntry_target_date:
                ntry_target = i
            if xpry_target < 0 and d >= xpry_target_date:
                xpry_target = i

        # --- PHASE 3: Find first valid day at/after targets ---
        # For entry: find first valid day, fallback to last good day in month
        # Ticker index meaning: -1 = not required, -2 = required but missing, >= 0 = exists
        ntry_off = -1
        if ntry_target >= 0:
            # Try to find first good day at or after target
            for i in range(ntry_target, length):
                idx = start + i
                d = dates[idx]
                # Don't look beyond entry month
                if d > ntry_end:
                    break
                # Check vol if required (-2 = required but ticker missing, always fails)
                if vt != -1 and (vt < 0 or np.isnan(vol[idx])):
                    continue
                # Check primary hedge if required
                if ht != -1 and (ht < 0 or np.isnan(hedge[idx])):
                    continue
                # Check additional hedges based on num_hedges
                if num_hedges >= 2 and h1t != -1 and (h1t < 0 or np.isnan(hedge1[idx])):
                    continue
                if num_hedges >= 3 and h2t != -1 and (h2t < 0 or np.isnan(hedge2[idx])):
                    continue
                if num_hedges >= 4 and h3t != -1 and (h3t < 0 or np.isnan(hedge3[idx])):
                    continue
                ntry_off = i
                break

            # If not found, fallback to last good day in entry month
            if ntry_off < 0:
                for i in range(length - 1, -1, -1):
                    idx = start + i
                    d = dates[idx]
                    if d > ntry_end:
                        continue  # Skip days after month end
                    # Check vol if required (-2 = required but ticker missing, always fails)
                    if vt != -1 and (vt < 0 or np.isnan(vol[idx])):
                        continue
                    # Check primary hedge if required
                    if ht != -1 and (ht < 0 or np.isnan(hedge[idx])):
                        continue
                    # Check additional hedges based on num_hedges
                    if num_hedges >= 2 and h1t != -1 and (h1t < 0 or np.isnan(hedge1[idx])):
                        continue
                    if num_hedges >= 3 and h2t != -1 and (h2t < 0 or np.isnan(hedge2[idx])):
                        continue
                    if num_hedges >= 4 and h3t != -1 and (h3t < 0 or np.isnan(hedge3[idx])):
                        continue
                    ntry_off = i
                    break

        # For expiry: find first valid day at or after target (within month)
        xpry_off = -1
        if xpry_target >= 0:
            for i in range(xpry_target, length):
                idx = start + i
                d = dates[idx]
                # Don't look beyond expiry month
                if d > xpry_end:
                    break
                # Check vol if required (-2 = required but ticker missing, always fails)
                if vt != -1 and (vt < 0 or np.isnan(vol[idx])):
                    continue
                # Check primary hedge if required
                if ht != -1 and (ht < 0 or np.isnan(hedge[idx])):
                    continue
                # Check additional hedges based on num_hedges
                if num_hedges >= 2 and h1t != -1 and (h1t < 0 or np.isnan(hedge1[idx])):
                    continue
                if num_hedges >= 3 and h2t != -1 and (h2t < 0 or np.isnan(hedge2[idx])):
                    continue
                if num_hedges >= 4 and h3t != -1 and (h3t < 0 or np.isnan(hedge3[idx])):
                    continue
                xpry_off = i
                break

        ntry_offsets_out[s] = ntry_off
        xpry_offsets_out[s] = xpry_off

        if ntry_off < 0 or xpry_off < 0:
            continue  # Skip this straddle

        ntry_idx = start + ntry_off
        xpry_idx = start + xpry_off
        xpry_date = dates[xpry_idx]

        # Mark actions
        action[ntry_idx] = 1  # ntry
        action[xpry_idx] = 2  # xpry

        # --- PHASE 4: Roll-forward and compute strike ---
        strike_val = hedge[ntry_idx] if ht >= 0 else np.nan
        strike1_val = hedge1[ntry_idx] if h1t >= 0 else np.nan
        strike2_val = hedge2[ntry_idx] if h2t >= 0 else np.nan
        strike3_val = hedge3[ntry_idx] if h3t >= 0 else np.nan

        last_vol = vol[ntry_idx]
        last_hedge = hedge[ntry_idx]
        last_hedge1 = hedge1[ntry_idx]
        last_hedge2 = hedge2[ntry_idx]
        last_hedge3 = hedge3[ntry_idx]

        for i in range(ntry_off, xpry_off + 1):
            idx = start + i

            # Roll-forward vol
            if np.isnan(vol[idx]):
                vol[idx] = last_vol
            else:
                last_vol = vol[idx]

            # Roll-forward hedge
            if np.isnan(hedge[idx]):
                hedge[idx] = last_hedge
            else:
                last_hedge = hedge[idx]

            # Roll-forward hedge1 (if used)
            if h1t >= 0:
                if np.isnan(hedge1[idx]):
                    hedge1[idx] = last_hedge1
                else:
                    last_hedge1 = hedge1[idx]

            # Roll-forward hedge2 (if used)
            if h2t >= 0:
                if np.isnan(hedge2[idx]):
                    hedge2[idx] = last_hedge2
                else:
                    last_hedge2 = hedge2[idx]

            # Roll-forward hedge3 (if used)
            if h3t >= 0:
                if np.isnan(hedge3[idx]):
                    hedge3[idx] = last_hedge3
                else:
                    last_hedge3 = hedge3[idx]

            # Set strikes
            strike[idx] = strike_val
            if h1t >= 0:
                strike1[idx] = strike1_val
            if h2t >= 0:
                strike2[idx] = strike2_val
            if h3t >= 0:
                strike3[idx] = strike3_val

            # Days to expiry
            days_to_expiry[idx] = xpry_date - dates[idx]

        # --- PHASE 5: Model computation ---
        for i in range(ntry_off, xpry_off + 1):
            idx = start + i
            S = hedge[idx]
            X = strike[idx]
            v = vol[idx]
            t = days_to_expiry[idx]

            if S <= 0 or X <= 0 or v <= 0:
                continue

            if t == 0:
                mv[idx] = abs(S - X) / X
                delta[idx] = 1.0 if S >= X else -1.0
            else:
                tv = (v / 100.0) * np.sqrt(float(t) / 365.0)
                if tv < 1e-10:
                    mv[idx] = abs(S - X) / X
                    delta[idx] = 1.0 if S >= X else -1.0
                else:
                    d1 = np.log(S / X) / tv + 0.5 * tv
                    d2 = d1 - tv
                    N_d1 = 2.0 * _norm_cdf_approx(d1)
                    N_d2 = 2.0 * _norm_cdf_approx(d2)
                    mv_val = S * N_d1 - X * N_d2 + X - S
                    mv[idx] = mv_val / X
                    delta[idx] = N_d1 - 1.0

        # --- PHASE 6: PnL computation ---
        opnl[ntry_idx] = 0.0
        hpnl[ntry_idx] = 0.0
        pnl[ntry_idx] = 0.0

        prev_mv = mv[ntry_idx]
        prev_delta = delta[ntry_idx]
        prev_hedge = hedge[ntry_idx]

        for i in range(ntry_off + 1, xpry_off + 1):
            idx = start + i
            curr_mv = mv[idx]
            curr_hedge = hedge[idx]

            if not np.isnan(curr_mv) and not np.isnan(prev_mv):
                opnl[idx] = curr_mv - prev_mv

            if not np.isnan(prev_delta) and not np.isnan(curr_hedge) and not np.isnan(prev_hedge):
                hpnl[idx] = -prev_delta * (curr_hedge - prev_hedge) / strike_val

            if not np.isnan(opnl[idx]) and not np.isnan(hpnl[idx]):
                pnl[idx] = opnl[idx] + hpnl[idx]

            if not np.isnan(curr_mv):
                prev_mv = curr_mv
            if not np.isnan(delta[idx]):
                prev_delta = delta[idx]
            if not np.isnan(curr_hedge):
                prev_hedge = curr_hedge

    return (vol, hedge, hedge1, hedge2, hedge3,
            ntry_offsets_out, xpry_offsets_out,
            strike, strike1, strike2, strike3,
            days_to_expiry, mv, delta, opnl, hpnl, pnl, action)


# -----------------------------
# High-level API for batch backtest
# -----------------------------


@dataclass
class BacktestArraysSorted:
    """Container for pre-computed arrays needed by full_backtest_kernel_sorted.

    Similar to BacktestArrays but uses ticker/field indices for sorted array lookup
    instead of price matrix row indices.
    """
    # Per-straddle ticker/field indices (length S)
    vol_ticker_idx: np.ndarray        # int32[S] - ticker index from PricesNumba.ticker_to_idx
    vol_field_idx: np.ndarray         # int32[S] - field index from PricesNumba.field_to_idx
    hedge_ticker_idx: np.ndarray      # int32[S] - ticker index from PricesNumba.ticker_to_idx
    hedge_field_idx: np.ndarray       # int32[S] - field index from PricesNumba.field_to_idx

    # Additional hedge ticker/field indices (for CDS and calc assets)
    hedge1_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge1 (-1 if not used)
    hedge1_field_idx: np.ndarray      # int32[S] - field index for hedge1
    hedge2_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge2 (-1 if not used)
    hedge2_field_idx: np.ndarray      # int32[S] - field index for hedge2
    hedge3_ticker_idx: np.ndarray     # int32[S] - ticker index for hedge3 (-1 if not used)
    hedge3_field_idx: np.ndarray      # int32[S] - field index for hedge3
    n_hedges: np.ndarray              # int8[S] - number of hedge columns required (1-4)

    # Entry/expiry anchor dates (length S)
    ntry_anchor_date32: np.ndarray    # int32[S] - date32 of entry anchor (from override/code)
    xpry_anchor_date32: np.ndarray    # int32[S] - date32 of expiry anchor (from override/code)
    ntrv_offsets: np.ndarray          # int32[S] - calendar days to add to entry anchor
    ntry_month_end: np.ndarray        # int32[S] - date32 of entry month end (for fallback)
    xpry_month_end: np.ndarray        # int32[S] - date32 of expiry month end (for fallback)

    # For output formatting
    asset_idx: np.ndarray             # int32[S] - index into unique_assets
    straddle_idx: np.ndarray          # int32[S] - index into unique_straddles
    model_idx: np.ndarray             # int32[S] - index into unique_models

    # For Arrow dictionary encoding
    unique_assets: list[str]
    unique_straddles: list[str]
    unique_models: list[str]


def _compute_starts_lengths_from_parent_idx(parent_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute straddle_starts and straddle_lengths from parent_idx array.

    Args:
        parent_idx: int32 array mapping each day to its source straddle index

    Returns:
        straddle_starts: int32 array of start positions for each straddle
        straddle_lengths: int32 array of lengths for each straddle
    """
    if len(parent_idx) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Find where parent_idx changes (boundaries between straddles)
    # Note: parent_idx is contiguous - all days for straddle 0 come first, then straddle 1, etc.
    n_straddles = int(parent_idx[-1]) + 1
    straddle_starts = np.zeros(n_straddles, dtype=np.int32)
    straddle_lengths = np.zeros(n_straddles, dtype=np.int32)

    # Count occurrences of each straddle index
    counts = np.bincount(parent_idx, minlength=n_straddles)
    straddle_lengths[:] = counts[:n_straddles].astype(np.int32)

    # Compute starts as cumsum of lengths (shifted by 1)
    straddle_starts[1:] = np.cumsum(straddle_lengths[:-1])

    return straddle_starts, straddle_lengths


def _prepare_backtest_arrays_sorted(
    straddle_list: list[tuple[str, str]],
    ticker_map: dict[str, dict[str, tuple[str, str]]],
    prices_numba: "prices_module.PricesNumba",
    stryms: list[str],
    ntrcs: list[str],
    amt_path: str,
    overrides_path: str | None = None,
) -> BacktestArraysSorted:
    """Prepare arrays for full_backtest_kernel_sorted.

    Similar to _prepare_backtest_arrays but builds ticker/field indices
    for sorted array lookup instead of price matrix row indices.

    Args:
        straddle_list: List of (asset, straddle) tuples
        ticker_map: Dict from _batch_resolve_tickers()
        prices_numba: PricesNumba object with sorted arrays and index mappings
        stryms: List of strym strings per straddle
        ntrcs: List of ntrc codes per straddle
        amt_path: Path to AMT YAML file
        overrides_path: Path to overrides CSV (for OVERRIDE code)

    Returns:
        BacktestArraysSorted containing all pre-computed arrays
    """
    # Lazy imports to avoid circular dependencies
    from . import asset_straddle_tickers
    from . import loader
    from . import schedules
    from .valuation import _anchor_day

    n_straddles = len(straddle_list)

    # === OPTIMIZATION: Pre-compute asset properties for unique assets ===
    # Instead of calling loader.get_asset() 231K times, call it once per unique asset (189 assets)
    unique_asset_names = set(asset for asset, _ in straddle_list)
    asset_to_data: dict[str, dict | None] = {}
    for asset_name in unique_asset_names:
        asset_to_data[asset_name] = loader.get_asset(amt_path, asset_name)

    # Per-straddle ticker/field index arrays
    vol_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    vol_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge_field_idx = np.full(n_straddles, -1, dtype=np.int32)

    # Additional hedge ticker/field indices (for CDS and calc assets)
    hedge1_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge1_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge2_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge2_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge3_ticker_idx = np.full(n_straddles, -1, dtype=np.int32)
    hedge3_field_idx = np.full(n_straddles, -1, dtype=np.int32)
    n_hedges = np.ones(n_straddles, dtype=np.int8)  # Default to 1 (just primary hedge)

    # Entry/expiry anchor dates
    ntry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    xpry_anchor_date32 = np.zeros(n_straddles, dtype=np.int32)
    ntrv_offsets = np.zeros(n_straddles, dtype=np.int32)
    ntry_month_end = np.zeros(n_straddles, dtype=np.int32)
    xpry_month_end = np.zeros(n_straddles, dtype=np.int32)

    # For Arrow dictionary encoding (string -> index)
    unique_assets: dict[str, int] = {}
    unique_straddles: dict[str, int] = {}
    unique_models: dict[str, int] = {}
    asset_idx = np.empty(n_straddles, dtype=np.int32)
    straddle_idx = np.empty(n_straddles, dtype=np.int32)
    model_idx = np.empty(n_straddles, dtype=np.int32)

    for s, (asset, straddle) in enumerate(straddle_list):
        strym = stryms[s]
        ntrc = ntrcs[s]

        # Build unique asset/straddle lists for Arrow output
        if asset not in unique_assets:
            unique_assets[asset] = len(unique_assets)
        if straddle not in unique_straddles:
            unique_straddles[straddle] = len(unique_straddles)
        asset_idx[s] = unique_assets[asset]
        straddle_idx[s] = unique_straddles[straddle]

        # Get asset data from pre-computed cache (O(1) dict lookup)
        asset_data = asset_to_data[asset]
        if asset_data is None:
            if "" not in unique_models:
                unique_models[""] = len(unique_models)
            model_idx[s] = unique_models[""]
            continue

        # Get model name
        valuation = asset_data.get("Valuation", {})
        model_name = valuation.get("Model", "") if isinstance(valuation, dict) else ""
        if model_name not in unique_models:
            unique_models[model_name] = len(unique_models)
        model_idx[s] = unique_models[model_name]

        vol_cfg = asset_data.get("Vol")
        hedge_cfg = asset_data.get("Hedge")
        if vol_cfg is None or hedge_cfg is None:
            continue

        # Compute cache key
        cache_key = asset_straddle_tickers.asset_straddle_ticker_key(
            asset, strym, ntrc, vol_cfg, hedge_cfg
        )

        if cache_key in ticker_map:
            param_map = ticker_map[cache_key]

            # Get vol ticker/field indices
            # Use -2 for "required but ticker missing from prices" vs -1 for "not required"
            if "vol" in param_map:
                vol_ticker, vol_field = param_map["vol"]
                # Look up indices in PricesNumba
                if vol_ticker in prices_numba.ticker_to_idx:
                    vol_ticker_idx[s] = prices_numba.ticker_to_idx[vol_ticker]
                else:
                    vol_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if vol_field in prices_numba.field_to_idx:
                    vol_field_idx[s] = prices_numba.field_to_idx[vol_field]

            # Get hedge ticker/field indices
            if "hedge" in param_map:
                hedge_ticker, hedge_field = param_map["hedge"]
                # Look up indices in PricesNumba
                if hedge_ticker in prices_numba.ticker_to_idx:
                    hedge_ticker_idx[s] = prices_numba.ticker_to_idx[hedge_ticker]
                else:
                    hedge_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if hedge_field in prices_numba.field_to_idx:
                    hedge_field_idx[s] = prices_numba.field_to_idx[hedge_field]

            # Get hedge1 ticker/field indices (for CDS assets)
            if "hedge1" in param_map:
                h1_ticker, h1_field = param_map["hedge1"]
                if h1_ticker in prices_numba.ticker_to_idx:
                    hedge1_ticker_idx[s] = prices_numba.ticker_to_idx[h1_ticker]
                else:
                    hedge1_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h1_field in prices_numba.field_to_idx:
                    hedge1_field_idx[s] = prices_numba.field_to_idx[h1_field]
                n_hedges[s] = max(n_hedges[s], 2)

            # Get hedge2 ticker/field indices (for calc assets)
            if "hedge2" in param_map:
                h2_ticker, h2_field = param_map["hedge2"]
                if h2_ticker in prices_numba.ticker_to_idx:
                    hedge2_ticker_idx[s] = prices_numba.ticker_to_idx[h2_ticker]
                else:
                    hedge2_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h2_field in prices_numba.field_to_idx:
                    hedge2_field_idx[s] = prices_numba.field_to_idx[h2_field]
                n_hedges[s] = max(n_hedges[s], 3)

            # Get hedge3 ticker/field indices (for calc assets)
            if "hedge3" in param_map:
                h3_ticker, h3_field = param_map["hedge3"]
                if h3_ticker in prices_numba.ticker_to_idx:
                    hedge3_ticker_idx[s] = prices_numba.ticker_to_idx[h3_ticker]
                else:
                    hedge3_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h3_field in prices_numba.field_to_idx:
                    hedge3_field_idx[s] = prices_numba.field_to_idx[h3_field]
                n_hedges[s] = max(n_hedges[s], 4)

            # Handle hedge4 -> hedge3 mapping for calc assets
            if "hedge4" in param_map and hedge3_ticker_idx[s] == -1:
                h4_ticker, h4_field = param_map["hedge4"]
                if h4_ticker in prices_numba.ticker_to_idx:
                    hedge3_ticker_idx[s] = prices_numba.ticker_to_idx[h4_ticker]
                else:
                    hedge3_ticker_idx[s] = -2  # Required but ticker not in loaded prices
                if h4_field in prices_numba.field_to_idx:
                    hedge3_field_idx[s] = prices_numba.field_to_idx[h4_field]
                n_hedges[s] = max(n_hedges[s], 4)

        # Parse straddle string for entry/expiry dates
        ntry_y = schedules.ntry(straddle)
        ntry_m = schedules.ntrm(straddle)
        xpry_y = schedules.xpry(straddle)
        xpry_m = schedules.xprm(straddle)

        # Get codes from straddle
        # Note: u8m format produces padded strings, so strip to get clean values
        xprc = schedules.xprc(straddle).strip()
        xprv = schedules.xprv(straddle).strip()
        ntrv_str = schedules.ntrv(straddle).strip()

        # Compute entry month end date
        _, ntry_num_days = calendar.monthrange(ntry_y, ntry_m)
        ntry_month_end[s] = ymd_to_date32(ntry_y, ntry_m, ntry_num_days)

        # Compute expiry month end date
        _, xpry_num_days = calendar.monthrange(xpry_y, xpry_m)
        xpry_month_end[s] = ymd_to_date32(xpry_y, xpry_m, xpry_num_days)

        # Parse ntrv as calendar day offset
        try:
            ntrv_offsets[s] = int(ntrv_str) if ntrv_str else 0
        except (ValueError, TypeError):
            ntrv_offsets[s] = 0

        # Compute entry anchor using _anchor_day
        entry_anchor = _anchor_day(xprc, xprv, ntry_y, ntry_m, asset, overrides_path)
        if entry_anchor is not None:
            y, m, d = map(int, entry_anchor.split("-"))
            ntry_anchor_date32[s] = ymd_to_date32(y, m, d)
        else:
            ntry_anchor_date32[s] = np.iinfo(np.int32).max

        # Compute expiry anchor using _anchor_day
        expiry_anchor = _anchor_day(xprc, xprv, xpry_y, xpry_m, asset, overrides_path)
        if expiry_anchor is not None:
            y, m, d = map(int, expiry_anchor.split("-"))
            xpry_anchor_date32[s] = ymd_to_date32(y, m, d)
        else:
            xpry_anchor_date32[s] = np.iinfo(np.int32).max

    return BacktestArraysSorted(
        vol_ticker_idx=vol_ticker_idx,
        vol_field_idx=vol_field_idx,
        hedge_ticker_idx=hedge_ticker_idx,
        hedge_field_idx=hedge_field_idx,
        hedge1_ticker_idx=hedge1_ticker_idx,
        hedge1_field_idx=hedge1_field_idx,
        hedge2_ticker_idx=hedge2_ticker_idx,
        hedge2_field_idx=hedge2_field_idx,
        hedge3_ticker_idx=hedge3_ticker_idx,
        hedge3_field_idx=hedge3_field_idx,
        n_hedges=n_hedges,
        ntry_anchor_date32=ntry_anchor_date32,
        xpry_anchor_date32=xpry_anchor_date32,
        ntrv_offsets=ntrv_offsets,
        ntry_month_end=ntry_month_end,
        xpry_month_end=xpry_month_end,
        asset_idx=asset_idx,
        straddle_idx=straddle_idx,
        model_idx=model_idx,
        unique_assets=list(unique_assets.keys()),
        unique_straddles=list(unique_straddles.keys()),
        unique_models=list(unique_models.keys()),
    )


def _build_arrow_output_sorted(
    # Numeric arrays from kernel
    dates: np.ndarray,                 # int32[N] - date32 values
    vol: np.ndarray,                   # float64[N] - raw vol prices
    hedge: np.ndarray,                 # float64[N] - raw hedge prices
    hedge1: np.ndarray,                # float64[N] - raw hedge1 prices
    hedge2: np.ndarray,                # float64[N] - raw hedge2 prices
    hedge3: np.ndarray,                # float64[N] - raw hedge3 prices
    strike: np.ndarray,                # float64[N] - strike price
    strike1: np.ndarray,               # float64[N] - strike1 price
    strike2: np.ndarray,               # float64[N] - strike2 price
    strike3: np.ndarray,               # float64[N] - strike3 price
    mv: np.ndarray,                    # float64[N] - mark-to-market
    delta: np.ndarray,                 # float64[N] - delta
    opnl: np.ndarray,                  # float64[N] - option PnL
    hpnl: np.ndarray,                  # float64[N] - hedge PnL
    pnl: np.ndarray,                   # float64[N] - total PnL
    action: np.ndarray,                # int8[N] - action code

    # For string columns
    parent_idx: np.ndarray,            # int32[N] - which straddle each day belongs to
    backtest_arrays: BacktestArraysSorted,  # Pre-computed array mappings

    # Straddle structure
    straddle_starts: np.ndarray,       # int32[S] - start index per straddle
    straddle_lengths: np.ndarray,      # int32[S] - length per straddle
    ntry_offsets: np.ndarray,          # int32[S] - entry offset per straddle
    xpry_offsets: np.ndarray,          # int32[S] - expiry offset per straddle

    # Options
    valid_only: bool = False,
) -> pa.Table:
    """Build Arrow table from numeric arrays for sorted kernel output.

    Full version matching _build_arrow_output with hedge1/2/3 and strike1/2/3.
    """
    n_straddles = len(straddle_starts)

    # Pre-compute strike_vol per straddle
    strike_vol = np.full(n_straddles, np.nan, dtype=np.float64)
    for s in range(n_straddles):
        ntry = ntry_offsets[s]
        if ntry >= 0:
            start = straddle_starts[s]
            strike_vol[s] = vol[start + ntry]

    # Pre-compute expiry date32 per straddle
    expiry_date32 = np.full(n_straddles, -1, dtype=np.int32)
    for s in range(n_straddles):
        xpry = xpry_offsets[s]
        if xpry >= 0:
            start = straddle_starts[s]
            expiry_date32[s] = dates[start + xpry]

    # Build valid_mask for filtering (rows with valid mv)
    if valid_only:
        filter_mask = ~np.isnan(mv)
        indices = np.where(filter_mask)[0]

        # Filter arrays
        dates = dates[indices]
        vol = vol[indices]
        hedge = hedge[indices]
        hedge1 = hedge1[indices]
        hedge2 = hedge2[indices]
        hedge3 = hedge3[indices]
        strike = strike[indices]
        strike1 = strike1[indices]
        strike2 = strike2[indices]
        strike3 = strike3[indices]
        mv = mv[indices]
        delta = delta[indices]
        opnl = opnl[indices]
        hpnl = hpnl[indices]
        pnl = pnl[indices]
        action = action[indices]
        parent_idx = parent_idx[indices]

    # Build per-row arrays from per-straddle data
    asset_indices = backtest_arrays.asset_idx[parent_idx]
    straddle_indices = backtest_arrays.straddle_idx[parent_idx]
    model_indices = backtest_arrays.model_idx[parent_idx]
    strike_vol_values = strike_vol[parent_idx]
    expiry_values = expiry_date32[parent_idx]

    # Build dictionary-encoded string columns
    asset_dict = pa.DictionaryArray.from_arrays(
        pa.array(asset_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_assets, type=pa.string())
    )

    straddle_dict = pa.DictionaryArray.from_arrays(
        pa.array(straddle_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_straddles, type=pa.string())
    )

    model_dict = pa.DictionaryArray.from_arrays(
        pa.array(model_indices, type=pa.int32()),
        pa.array(backtest_arrays.unique_models, type=pa.string())
    )

    action_strs = ["", "ntry", "xpry"]
    action_dict = pa.DictionaryArray.from_arrays(
        pa.array(action, type=pa.int8()),
        pa.array(action_strs, type=pa.string())
    )

    date_array = pa.array(dates, type=pa.date32())
    expiry_array = pa.array(expiry_values, type=pa.date32())

    # Build table with all columns (matching _build_arrow_output)
    return pa.table({
        'asset': asset_dict,
        'straddle': straddle_dict,
        'date': date_array,
        'vol': pa.array(vol),
        'hedge': pa.array(hedge),
        'hedge1': pa.array(hedge1),
        'hedge2': pa.array(hedge2),
        'hedge3': pa.array(hedge3),
        'action': action_dict,
        'model': model_dict,
        'strike_vol': pa.array(strike_vol_values),
        'strike': pa.array(strike),
        'strike1': pa.array(strike1),
        'strike2': pa.array(strike2),
        'strike3': pa.array(strike3),
        'expiry': expiry_array,
        'mv': pa.array(mv),
        'delta': pa.array(delta),
        'opnl': pa.array(opnl),
        'hpnl': pa.array(hpnl),
        'pnl': pa.array(pnl),
    })


def get_straddle_backtests_numba(
    pattern: str,
    start_year: int,
    end_year: int,
    amt_path: str,
    prices_parquet: str | None = None,
    valid_only: bool = False,
    overrides_path: str | None = None,
) -> pa.Table:
    """Fast Numba-based backtest for all straddles matching pattern.

    This is the optimized entry point that uses:
    - find_straddle_days_u8m() for straddle expansion
    - Numba kernels for price lookup and model computation
    - Arrow output format (fastest for downstream processing)

    Args:
        pattern: Regex pattern to match assets (e.g., "^LA Comdty", ".")
        start_year: Start year for straddles
        end_year: End year for straddles
        amt_path: Path to AMT YAML file
        prices_parquet: Path to prices parquet file (required)
        valid_only: If True, only return rows with valid mv values
        overrides_path: Path to overrides CSV file for OVERRIDE code anchor dates.
            Defaults to "data/overrides.csv" if not specified.

    Returns:
        PyArrow Table with columns: asset, straddle, date, vol, hedge, hedge1, hedge2, hedge3,
        action, model, strike_vol, strike, strike1, strike2, strike3, expiry, mv, delta,
        opnl, hpnl, pnl

    Example:
        >>> from specparser.amt.valuation_numba import get_straddle_backtests_numba
        >>> result = get_straddle_backtests_numba(
        ...     "^LA Comdty", 2022, 2024, "data/amt.yml",
        ...     prices_parquet="data/prices_sorted.parquet"
        ... )
        >>> print(f"{result.num_rows:,} rows")
    """
    # Lazy imports to avoid circular dependencies
    from . import prices as prices_module
    from . import schedules
    from . import strings as strings_module
    from .valuation import _batch_resolve_tickers

    if prices_parquet is None:
        raise ValueError("prices_parquet is required for get_straddle_backtests_numba()")

    # Phase 1+2: Use fast u8m-based date expansion (Numba kernel)
    straddle_days_table = schedules.find_straddle_days_u8m(
        amt_path, start_year, end_year, pattern, live_only=True, parallel=True,
    )

    # Extract numpy arrays (orientation: "numpy")
    asset_u8m = straddle_days_table["rows"][0]  # (n_days, width) uint8
    straddle_u8m = straddle_days_table["rows"][1]  # (n_days, width) uint8
    dates = straddle_days_table["rows"][2]  # (n_days,) int32
    straddle_id = straddle_days_table["rows"][3]  # (n_days,) int32 - maps each day to source straddle

    n_days = len(dates)

    if n_days == 0:
        # Return empty Arrow table with correct schema
        return pa.table({
            'asset': pa.array([], type=pa.dictionary(pa.int32(), pa.string())),
            'straddle': pa.array([], type=pa.dictionary(pa.int32(), pa.string())),
            'date': pa.array([], type=pa.date32()),
            'vol': pa.array([], type=pa.float64()),
            'hedge': pa.array([], type=pa.float64()),
            'hedge1': pa.array([], type=pa.float64()),
            'hedge2': pa.array([], type=pa.float64()),
            'hedge3': pa.array([], type=pa.float64()),
            'action': pa.array([], type=pa.dictionary(pa.int8(), pa.string())),
            'model': pa.array([], type=pa.dictionary(pa.int32(), pa.string())),
            'strike_vol': pa.array([], type=pa.float64()),
            'strike': pa.array([], type=pa.float64()),
            'strike1': pa.array([], type=pa.float64()),
            'strike2': pa.array([], type=pa.float64()),
            'strike3': pa.array([], type=pa.float64()),
            'expiry': pa.array([], type=pa.date32()),
            'mv': pa.array([], type=pa.float64()),
            'delta': pa.array([], type=pa.float64()),
            'opnl': pa.array([], type=pa.float64()),
            'hpnl': pa.array([], type=pa.float64()),
            'pnl': pa.array([], type=pa.float64()),
        })

    # Compute straddle_starts and straddle_lengths from straddle_id
    straddle_starts, straddle_lengths = _compute_starts_lengths_from_parent_idx(straddle_id)
    n_straddles_actual = len(straddle_starts)

    # Build straddle_list from unique u8m rows (one per straddle, not per day)
    unique_indices = straddle_starts  # First day of each straddle
    unique_asset_u8m = asset_u8m[unique_indices]
    unique_straddle_u8m = straddle_u8m[unique_indices]

    # Convert u8m to strings for downstream compatibility
    unique_assets = strings_module.u8m2s(unique_asset_u8m)
    unique_straddles = strings_module.u8m2s(unique_straddle_u8m)

    # Strip padding from assets (required for loader.get_asset() lookups)
    straddle_list = [
        (asset.strip(), straddle)
        for asset, straddle in zip(unique_assets.tolist(), unique_straddles.tolist())
    ]

    # Phase 3: Resolve tickers (batched)
    stryms = []
    ntrcs = []
    assets_for_tickers = []
    for asset, straddle in straddle_list:
        xpry = schedules.xpry(straddle)
        xprm = schedules.xprm(straddle)
        ntrc = schedules.ntrc(straddle)
        stryms.append(f"{xpry}-{xprm:02d}")
        ntrcs.append(ntrc)
        assets_for_tickers.append(asset)

    ticker_map = _batch_resolve_tickers(assets_for_tickers, stryms, ntrcs, amt_path)

    # Get or load the PricesNumba structure
    prices_numba = prices_module.get_prices_numba()
    if prices_numba is None:
        prices_numba = prices_module.load_prices_numba(prices_parquet)

    # Prepare all arrays (one-time string lookups)
    effective_overrides = overrides_path if overrides_path is not None else "data/overrides.csv"
    backtest_arrays = _prepare_backtest_arrays_sorted(
        straddle_list,
        ticker_map,
        prices_numba,
        stryms,
        ntrcs,
        amt_path,
        effective_overrides,
    )

    # Run the unified Numba kernel with sorted array lookup
    (vol_array, hedge_array, hedge1_array, hedge2_array, hedge3_array,
     ntry_offsets, xpry_offsets,
     strike_array, strike1_array, strike2_array, strike3_array,
     days_to_expiry, mv, delta, opnl, hpnl, pnl, action) = \
        full_backtest_kernel_sorted(
            prices_numba.sorted_keys,
            prices_numba.sorted_values,
            prices_numba.n_fields,
            prices_numba.n_dates,
            prices_numba.min_date32,
            backtest_arrays.vol_ticker_idx,
            backtest_arrays.vol_field_idx,
            backtest_arrays.hedge_ticker_idx,
            backtest_arrays.hedge_field_idx,
            backtest_arrays.hedge1_ticker_idx,
            backtest_arrays.hedge1_field_idx,
            backtest_arrays.hedge2_ticker_idx,
            backtest_arrays.hedge2_field_idx,
            backtest_arrays.hedge3_ticker_idx,
            backtest_arrays.hedge3_field_idx,
            backtest_arrays.n_hedges,
            straddle_starts,
            straddle_lengths,
            backtest_arrays.ntry_anchor_date32,
            backtest_arrays.xpry_anchor_date32,
            backtest_arrays.ntrv_offsets,
            backtest_arrays.ntry_month_end,
            backtest_arrays.xpry_month_end,
            dates,
        )

    # Build parent_idx mapping each day back to its straddle
    # Note: straddle_id is already this mapping from find_straddle_days_u8m
    parent_idx = straddle_id

    # Build Arrow output
    return _build_arrow_output_sorted(
        dates=dates,
        vol=vol_array,
        hedge=hedge_array,
        hedge1=hedge1_array,
        hedge2=hedge2_array,
        hedge3=hedge3_array,
        strike=strike_array,
        strike1=strike1_array,
        strike2=strike2_array,
        strike3=strike3_array,
        mv=mv,
        delta=delta,
        opnl=opnl,
        hpnl=hpnl,
        pnl=pnl,
        action=action,
        parent_idx=parent_idx,
        backtest_arrays=backtest_arrays,
        straddle_starts=straddle_starts,
        straddle_lengths=straddle_lengths,
        ntry_offsets=ntry_offsets,
        xpry_offsets=xpry_offsets,
        valid_only=valid_only,
    )
