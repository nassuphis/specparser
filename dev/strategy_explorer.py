"""Strategy Explorer - Interactive strategy construction and visualization."""
# Configure Numba BEFORE importing numba to avoid race conditions in Streamlit
import os
os.environ['NUMBA_NUM_THREADS'] = '1'  # Single-threaded to avoid concurrent access issues

import re
import time
import threading
from dataclasses import dataclass

import bottleneck as bn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.feather as pf
import streamlit as st
import yaml
from numba import njit

# Lock for serializing access to numba functions (Streamlit uses multiple threads)
_numba_lock = threading.Lock()

st.set_page_config(page_title="Strategy Explorer", layout="wide")

# Global CSS
st.markdown("""
<style>
/* Black table headers */
.stDataFrame thead th {
    background-color: #1a1a2e !important;
    color: white !important;
    font-weight: 600 !important;
}
/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class PortfolioStats:
    sharpe: float
    long_sharpe: float
    short_sharpe: float
    ls_correl: float
    max_drawdown: float


@dataclass
class WorkflowInfo:
    """Captures dimensions and timings at each pipeline step.

    Two-Hedge Pipeline:
    1. signal_weights: ranked/scaled signal
    2. risk_weighted_signal = combine(signal, asset_rp)  # normabs rule
    3. rp_hedge_pnl = inception_pnl(risk_weighted_signal)  # for RP hedge β
    4. rp_hedge_weights = hedge(rp_hedge_pnl)  # β matrix (1 for shorts, β for longs)
    5. asset_weights = risk_weighted_signal * rp_hedge  # simple multiply
    6. class_rp_pnl = inception_pnl(asset_weights)  # for class RP vol estimation
    7. class_rp_weights = 1/vol(class_rp_pnl by class)
    8. asset_class_weights = combine(asset_weights, class_rp)  # normabs rule
    9. final_hedge_pnl = inception_pnl(asset_class_weights)  # for final hedge β
    10. final_hedge_weights = hedge(final_hedge_pnl)  # β matrix
    11. final_weights = asset_class_weights * final_hedge  # simple multiply
    12. wpnl = inception_pnl(final_weights)
    """
    n_days: int
    n_assets: int
    n_active_assets: int  # Number of assets after filtering (excludes masked assets)
    n_groups: int
    # Timings (ms) - Two-hedge pipeline stages
    t_file_load: float  # Arrow file loading
    t_preprocess: float  # parse_wgt → aggregate → PnL/Vol → npnl/rnpnl/wpnl
    t_signal: float  # Signal computation (rank → scale → lag)
    t_asset_rp: float  # Asset RP: 1/vol → combine with signal → risk_weighted_signal
    t_rp_hedge_pnl: float  # PnL from risk_weighted_signal for RP hedge β
    t_rp_hedge: float  # RP hedge β computation
    t_asset_weights: float  # risk_weighted_signal * rp_hedge (simple multiply)
    t_class_rp_pnl: float  # PnL from asset_weights for class RP vol
    t_class_rp: float  # Class RP: 1/vol(class) → expand to assets
    t_asset_class_weights: float  # combine(asset_weights, class_rp)
    t_final_hedge_pnl: float  # PnL from asset_class_weights for final hedge β
    t_final_hedge: float  # Final hedge β computation
    t_final_weights: float  # asset_class_weights * final_hedge (simple multiply)
    t_pnl: float  # Final weighted PnL + long/short split
    t_total: float
    # Dimensions at each step (all weight matrices are n_days × n_assets)
    input_shape: tuple[int, int]
    signal_weights_shape: tuple[int, int]
    asset_rp_weights_shape: tuple[int, int]
    risk_weighted_signal_shape: tuple[int, int]
    rp_hedge_pnl_shape: tuple[int, int]  # Long/short matrices for RP hedge
    rp_hedge_weights_shape: tuple[int, int]
    asset_weights_shape: tuple[int, int]  # After rp_hedge applied
    class_rp_pnl_shape: tuple[int, int]  # PnL for class RP vol
    class_rp_weights_shape: tuple[int, int]
    asset_class_weights_shape: tuple[int, int]  # After class RP combined
    final_hedge_pnl_shape: tuple[int, int]  # Long/short matrices for final hedge
    final_hedge_weights_shape: tuple[int, int]
    final_weights_shape: tuple[int, int]  # After final_hedge applied
    wpnl_shape: tuple[int, int]


@dataclass
class BacktestContext:
    """Holds all data needed for backtesting."""
    raw_pnl_matrix: np.ndarray  # Raw pnl matrix (no normalization)
    npnl: np.ndarray
    rnpnl: np.ndarray
    wpnl: np.ndarray  # Winsorized npnl (rolling 1%/99% clipping)
    vol_matrix: np.ndarray
    group_ids: np.ndarray
    n_groups: int
    unique_groups: list[str]  # List of group names (ordered by group_id)
    out0s: np.ndarray
    lens: np.ndarray
    starts: np.ndarray
    asset_ids: np.ndarray
    weights: np.ndarray
    asset_starts: np.ndarray
    asset_counts: np.ndarray
    d0: int
    grid_size: int
    n_assets: int
    pnl: np.ndarray
    dte: np.ndarray
    have_dte: bool


# ============================================================================
# Numba Kernels
# ============================================================================
@njit(cache=True)
def _aggregate_weighted_daily_by_asset_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets,
    asset_starts, asset_counts
):
    """Parallel aggregation of weighted daily pnl per asset into 2D grid."""
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)

    for a in range(n_assets):
        start_idx = asset_starts[a]
        count = asset_counts[a]

        for i in range(count):
            k = start_idx + i
            o = out0s[k]
            L = lens[k]
            start = starts_epoch[k]
            w = weights[k]

            for j in range(L):
                idx = o + j
                day_idx = start + j - d0

                if day_idx < 0 or day_idx >= grid_size:
                    continue
                if have_dte and dte[idx] < 0:
                    continue

                p = pnl[idx]
                if not np.isnan(p):
                    pnl_sum[day_idx, a] += p * w

    return pnl_sum


@njit(cache=True)
def _aggregate_inception_weighted_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets,
    signal_matrix, asset_starts, asset_counts
):
    """Aggregation with inception-locked signal weights."""
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)

    for a in range(n_assets):
        start_idx = asset_starts[a]
        count = asset_counts[a]

        for i in range(count):
            k = start_idx + i
            o = out0s[k]
            L = lens[k]
            start = starts_epoch[k]
            w = weights[k]

            inception_day = start - d0
            if inception_day < 0 or inception_day >= grid_size:
                continue

            sig = signal_matrix[inception_day, a]
            if np.isnan(sig):
                continue

            for j in range(L):
                idx = o + j
                day_idx = start + j - d0

                if day_idx < 0 or day_idx >= grid_size:
                    continue
                if have_dte and dte[idx] < 0:
                    continue

                p = pnl[idx]
                if not np.isnan(p):
                    pnl_sum[day_idx, a] += p * w * sig

    return pnl_sum


@njit(cache=True)
def _aggregate_vol_by_asset_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, vol, dte, have_dte, grid_size, n_assets,
    asset_starts, asset_counts
):
    """Parallel aggregation of weighted vol."""
    vol_sum = np.zeros((grid_size, n_assets), np.float64)

    for a in range(n_assets):
        start_idx = asset_starts[a]
        count = asset_counts[a]

        for i in range(count):
            k = start_idx + i
            o = out0s[k]
            L = lens[k]
            start = starts_epoch[k]
            w = weights[k]

            for j in range(L):
                idx = o + j
                day_idx = start + j - d0

                if day_idx < 0 or day_idx >= grid_size:
                    continue
                if have_dte and dte[idx] < 0:
                    continue

                v = vol[idx]
                if not np.isnan(v):
                    vol_sum[day_idx, a] += v * w

    return vol_sum


@njit(cache=True)
def _cross_sectional_rank_parallel(matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional ranking normalized to [-1, +1]."""
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for d in range(n_days):
        row = matrix[d, :]
        valid_mask = ~np.isnan(row)
        n_valid = np.sum(valid_mask)

        if n_valid < 2:
            continue

        valid_indices = np.empty(n_valid, dtype=np.int64)
        valid_values = np.empty(n_valid, dtype=np.float64)
        k = 0
        for i in range(n_assets):
            if valid_mask[i]:
                valid_indices[k] = i
                valid_values[k] = row[i]
                k += 1

        sort_order = np.argsort(valid_values)

        for rank, sorted_pos in enumerate(sort_order):
            original_idx = valid_indices[sorted_pos]
            normalized_rank = 2.0 * rank / (n_valid - 1) - 1.0
            result[d, original_idx] = normalized_rank

    return result


def precompute_group_indices(group_ids: np.ndarray, n_groups: int):
    """Precompute CSR-like group membership for fast iteration.

    Returns:
        grp_ptr: array of length n_groups+1 with start/end pointers
        grp_idx: array of asset indices sorted by group
    """
    # Count members per group
    counts = np.bincount(group_ids, minlength=n_groups)
    # Compute start pointers
    grp_ptr = np.zeros(n_groups + 1, dtype=np.int64)
    grp_ptr[1:] = np.cumsum(counts)
    # Build sorted index array (assets sorted by their group)
    grp_idx = np.argsort(group_ids).astype(np.int64)
    return grp_ptr, grp_idx


@njit(cache=True)
def _cross_sectional_rank_by_group_fast(
    matrix: np.ndarray, grp_ptr: np.ndarray, grp_idx: np.ndarray, n_groups: int
) -> np.ndarray:
    """Cross-sectional ranking within groups using precomputed CSR indices.

    Much faster than rebuilding group masks every day.
    """
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for d in range(n_days):
        row = matrix[d, :]

        for g in range(n_groups):
            start, end = grp_ptr[g], grp_ptr[g + 1]
            group_size = end - start

            if group_size < 2:
                continue

            # Collect valid values for this group only (iterate group members, not all assets)
            valid_indices = []
            valid_values = []
            for k in range(start, end):
                a = grp_idx[k]
                val = row[a]
                if not np.isnan(val):
                    valid_indices.append(a)
                    valid_values.append(val)

            n_valid = len(valid_values)
            if n_valid < 2:
                continue

            # Convert to arrays for sorting
            valid_indices_arr = np.array(valid_indices, dtype=np.int64)
            valid_values_arr = np.array(valid_values, dtype=np.float64)
            sort_order = np.argsort(valid_values_arr)

            for rank, sorted_pos in enumerate(sort_order):
                original_idx = valid_indices_arr[sorted_pos]
                normalized_rank = 2.0 * rank / (n_valid - 1) - 1.0
                result[d, original_idx] = normalized_rank

    return result


@njit(cache=True)
def _cross_sectional_rank_by_group_parallel(
    matrix: np.ndarray, group_ids: np.ndarray, n_groups: int
) -> np.ndarray:
    """Cross-sectional ranking within groups, normalized to [-1, +1].

    Legacy interface - calls precompute internally. Use _cross_sectional_rank_by_group_fast
    with precomputed indices for better performance when calling multiple times.
    """
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for d in range(n_days):
        row = matrix[d, :]

        for g in range(n_groups):
            # Still uses mask approach for backward compatibility
            valid_indices = []
            valid_values = []
            for i in range(n_assets):
                if group_ids[i] == g and not np.isnan(row[i]):
                    valid_indices.append(i)
                    valid_values.append(row[i])

            n_valid = len(valid_values)
            if n_valid < 2:
                continue

            valid_indices_arr = np.array(valid_indices, dtype=np.int64)
            valid_values_arr = np.array(valid_values, dtype=np.float64)
            sort_order = np.argsort(valid_values_arr)

            for rank, sorted_pos in enumerate(sort_order):
                original_idx = valid_indices_arr[sorted_pos]
                normalized_rank = 2.0 * rank / (n_valid - 1) - 1.0
                result[d, original_idx] = normalized_rank

    return result


@njit(cache=True)
def _ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """EMA for 1D array."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[0] = arr[0] if not np.isnan(arr[0]) else 0.0

    for i in range(1, n):
        val = arr[i]
        if np.isnan(val):
            result[i] = result[i-1]
        else:
            result[i] = alpha * val + (1 - alpha) * result[i-1]

    return result


@njit(cache=True)
def _ema_2d_columnwise(matrix: np.ndarray, alpha: float) -> np.ndarray:
    """EMA applied column-wise."""
    n_rows, n_cols = matrix.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)

    for c in range(n_cols):
        result[:, c] = _ema_1d(matrix[:, c], alpha)

    return result


@njit(cache=True)
def _compute_rolling_beta_ema(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Compute rolling beta using EMA."""
    n = len(x)
    result = np.empty(n, dtype=np.float64)

    ema_xy = 0.0
    ema_xx = 0.0

    for i in range(n):
        xi, yi = x[i], y[i]
        if np.isnan(xi) or np.isnan(yi):
            result[i] = result[i-1] if i > 0 else 0.0
            continue

        ema_xy = alpha * (xi * yi) + (1 - alpha) * ema_xy
        ema_xx = alpha * (xi * xi) + (1 - alpha) * ema_xx

        if ema_xx > 1e-12:
            result[i] = -ema_xy / ema_xx
        else:
            result[i] = 0.0

    return result


@njit(cache=True)
def _compute_rolling_beta_by_group_ema(
    long_matrix: np.ndarray, short_matrix: np.ndarray,
    group_ids: np.ndarray, n_groups: int, alpha: float
) -> np.ndarray:
    """Compute rolling beta for each group."""
    n_days = long_matrix.shape[0]
    n_assets = long_matrix.shape[1]
    group_betas = np.zeros((n_days, n_groups), dtype=np.float64)

    for g in range(n_groups):
        long_g = np.zeros(n_days, dtype=np.float64)
        short_g = np.zeros(n_days, dtype=np.float64)

        for a in range(n_assets):
            if group_ids[a] == g:
                for d in range(n_days):
                    val = long_matrix[d, a]
                    if not np.isnan(val):
                        long_g[d] += val
                    val = short_matrix[d, a]
                    if not np.isnan(val):
                        short_g[d] += val

        group_betas[:, g] = _compute_rolling_beta_ema(long_g, short_g, alpha)

    return group_betas


@njit(cache=True)
def _rolling_winsorize_parallel(
    matrix: np.ndarray, window: int = 365, lo_pct: float = 0.01, hi_pct: float = 0.99
) -> np.ndarray:
    """Rolling winsorization: clip at rolling quantiles (no look-ahead).

    Optimized: single pass to count and extract valid values.
    """
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for a in range(n_assets):
        for d in range(n_days):
            val = matrix[d, a]
            if np.isnan(val):
                continue

            start = max(0, d - window + 1)

            # Single pass: extract valid values and count
            valid_count = 0
            for i in range(start, d + 1):
                v = matrix[i, a]
                if not np.isnan(v):
                    valid_count += 1

            if valid_count < 20:
                result[d, a] = val
                continue

            # Allocate and fill array for sorting
            valid_vals = np.empty(valid_count, dtype=np.float64)
            k = 0
            for i in range(start, d + 1):
                v = matrix[i, a]
                if not np.isnan(v):
                    valid_vals[k] = v
                    k += 1

            # Use numpy's O(n log n) sort
            valid_vals.sort()

            lo_idx = int(lo_pct * (valid_count - 1))
            hi_idx = int(hi_pct * (valid_count - 1))
            lo_val = valid_vals[lo_idx]
            hi_val = valid_vals[hi_idx]

            if val < lo_val:
                result[d, a] = lo_val
            elif val > hi_val:
                result[d, a] = hi_val
            else:
                result[d, a] = val

    return result


# ============================================================================
# Signal Functions
# ============================================================================
def signal_mac(input_matrix: np.ndarray, slow: int, fast: int) -> np.ndarray:
    """Moving average crossover signal."""
    alpha_slow = 2 / (slow + 1)
    alpha_fast = 2 / (fast + 1)
    cumsum = np.nancumsum(input_matrix, axis=0)
    ema_slow = _ema_2d_columnwise(cumsum, alpha_slow)
    ema_fast = _ema_2d_columnwise(cumsum, alpha_fast)
    return ema_fast - ema_slow


def signal_sharpe(input_matrix: np.ndarray, period: int) -> np.ndarray:
    """Sharpe-like ratio signal."""
    alpha = 2 / (period + 1)
    ema_return = _ema_2d_columnwise(input_matrix, alpha)
    ema_abs = _ema_2d_columnwise(np.abs(input_matrix), alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = ema_return / ema_abs
        result[~np.isfinite(result)] = np.nan
    return result


def signal_sign(input_matrix: np.ndarray, period: int) -> np.ndarray:
    """Sign persistence signal."""
    alpha = 2 / (period + 1)
    signs = np.sign(input_matrix)
    return _ema_2d_columnwise(signs, alpha)


def signal_clm(input_matrix: np.ndarray) -> np.ndarray:
    """Calmar-like signal: cumsum / max_drawdown."""
    cumsum = np.nancumsum(input_matrix, axis=0)
    running_max = np.maximum.accumulate(cumsum, axis=0)
    drawdown = running_max - cumsum
    max_drawdown = np.maximum.accumulate(drawdown, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = cumsum / max_drawdown
        result[~np.isfinite(result)] = np.nan
    return result


def signal_viso(vol_matrix: np.ndarray) -> np.ndarray:
    """Vol isolation signal: EMA(vol) / vol."""
    alpha = 2 / 366
    with np.errstate(divide='ignore', invalid='ignore'):
        ema_vol = _ema_2d_columnwise(vol_matrix, alpha)
        result = ema_vol / vol_matrix
        result[~np.isfinite(result)] = np.nan
    return result


def signal_rng(input_matrix: np.ndarray, window: int = 365) -> np.ndarray:
    """Range signal using MAD proxy: rolling_mean / range_proxy.

    Original: px / (q90 - q10) which requires O(n × window × log(window)) sorting.

    Optimization: Use MAD (Median Absolute Deviation) as proxy for range.
    For normal distribution: (q90 - q10) ≈ 2.563 * σ ≈ 2.563 * 1.4826 * MAD

    This is O(n) per asset using bottleneck.move_median.
    """
    n_days, n_assets = input_matrix.shape
    result = np.full((n_days, n_assets), np.nan, dtype=np.float64)

    # MAD to range conversion factor: 2.563 * 1.4826 ≈ 3.8
    MAD_TO_RANGE = 2.563 * 1.4826

    for a in range(n_assets):
        col = input_matrix[:, a]

        # O(n) rolling mean
        px = bn.move_mean(col, window=window, min_count=1)

        # O(n) rolling median of px
        rolling_median = bn.move_median(px, window=window, min_count=1)

        # O(n) absolute deviation from rolling median
        abs_dev = np.abs(px - rolling_median)

        # O(n) MAD = median of absolute deviations
        mad = bn.move_median(abs_dev, window=window, min_count=1)

        # Range proxy from MAD
        range_proxy = MAD_TO_RANGE * mad

        # Compute result: px / range_proxy
        with np.errstate(divide='ignore', invalid='ignore'):
            result[:, a] = px / range_proxy
            result[~np.isfinite(result[:, a]), a] = np.nan

    return result


@njit(cache=True)
def _compute_streak_cummax(input_matrix: np.ndarray) -> tuple:
    """Compute cummax indices for positive and negative values (numba accelerated)."""
    n_days, n_assets = input_matrix.shape
    updif = np.zeros((n_days, n_assets), dtype=np.float64)
    dndif = np.zeros((n_days, n_assets), dtype=np.float64)

    for a in range(n_assets):
        max_up_idx = 0.0
        max_dn_idx = 0.0
        for d in range(n_days):
            row_idx = d + 1  # 1-indexed
            val = input_matrix[d, a]
            if not np.isnan(val):
                if val > 0:
                    max_up_idx = row_idx
                elif val < 0:
                    max_dn_idx = row_idx
            updif[d, a] = row_idx - max_up_idx  # days since last positive
            dndif[d, a] = row_idx - max_dn_idx  # days since last negative

    return updif, dndif


def signal_streak(input_matrix: np.ndarray, window: int = 365) -> np.ndarray:
    """Streak signal: rolling mean of (days_since_negative - days_since_positive).

    Positive signal when asset has had more recent/frequent positive values.
    Negative signal when asset has had more recent/frequent negative values.

    Uses numba for cummax computation and bottleneck for O(n) rolling means.
    """
    n_days, n_assets = input_matrix.shape

    # Compute days since last positive/negative using numba
    updif, dndif = _compute_streak_cummax(input_matrix)

    # Use bottleneck for O(n) rolling means (column-wise)
    result = np.full((n_days, n_assets), np.nan, dtype=np.float64)
    for a in range(n_assets):
        up_mean = bn.move_mean(updif[:, a], window=window, min_count=window)
        dn_mean = bn.move_mean(dndif[:, a], window=window, min_count=window)
        result[:, a] = dn_mean - up_mean

    return result


# ============================================================================
# Signal Registry
# ============================================================================
SIGNAL_SPECS = {
    "mac.180.30": (signal_mac, {"slow": 180, "fast": 30}),
    "mac.180.10": (signal_mac, {"slow": 180, "fast": 10}),
    "mac.90.10": (signal_mac, {"slow": 90, "fast": 10}),
    "mac.90.30": (signal_mac, {"slow": 90, "fast": 30}),
    "mac.30.10": (signal_mac, {"slow": 30, "fast": 10}),
    "srp.365": (signal_sharpe, {"period": 365}),
    "sign.365": (signal_sign, {"period": 365}),
    "clm.365": (signal_clm, {}),
    "rng.365": (signal_rng, {"window": 365}),
    "streak.365": (signal_streak, {"window": 365}),
}

COMBINED_SIGNALS = {
    "mac.combo": ["mac.180.30", "mac.180.10", "mac.90.10", "mac.90.30", "mac.30.10"],
}

HEDGE_LABELS = {
    "none": "None",
    "h": "Global (h)",
    "hg": "By Group (hg)",
    "gah": "Group→All (gah)",
}


# ============================================================================
# Signal Transforms
# ============================================================================
def rank_all(signal: np.ndarray) -> np.ndarray:
    return _cross_sectional_rank_parallel(signal)


def rank_by_group(signal: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    return _cross_sectional_rank_by_group_parallel(signal, group_ids, n_groups)


def sumabs_norm(signal: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum_abs = np.nansum(np.abs(signal), axis=1, keepdims=True)
        result = signal / row_sum_abs
        result[~np.isfinite(result)] = np.nan
    return result


def lag(signal: np.ndarray, days: int = 1) -> np.ndarray:
    result = np.empty_like(signal)
    result[:days, :] = np.nan
    result[days:, :] = signal[:-days, :]
    return result


# ============================================================================
# Signal Cache
# ============================================================================
class SignalCache:
    """Cache for pre-computed ranked/normalized/lagged signals."""

    def __init__(self, input_matrix: np.ndarray, vol_matrix: np.ndarray,
                 group_ids: np.ndarray, n_groups: int):
        self.input_matrix = input_matrix
        self.vol_matrix = vol_matrix
        self.group_ids = group_ids
        self.n_groups = n_groups
        self._cache_all = {}
        self._cache_grp = {}
        # Precompute CSR-like group indices for fast group ranking
        self._grp_ptr, self._grp_idx = precompute_group_indices(group_ids, n_groups)

    def _compute_raw(self, name: str) -> np.ndarray:
        if name == "viso":
            return signal_viso(self.vol_matrix)
        elif name in SIGNAL_SPECS:
            fn, params = SIGNAL_SPECS[name]
            return fn(self.input_matrix, **params)
        else:
            raise ValueError(f"Unknown signal: {name}")

    def _transform(self, raw: np.ndarray, rank_mode: str, asset_mask: np.ndarray | None) -> np.ndarray:
        # Apply asset mask before ranking - excluded assets become NaN
        if asset_mask is not None:
            raw = raw.copy()
            raw[:, ~asset_mask] = np.nan
        if rank_mode == "all":
            ranked = rank_all(raw)
        else:
            # Use fast group ranking with precomputed CSR indices
            ranked = _cross_sectional_rank_by_group_fast(
                raw, self._grp_ptr, self._grp_idx, self.n_groups
            )
        return lag(sumabs_norm(ranked), days=1)

    def get(self, name: str, rank_mode: str, asset_mask: np.ndarray | None = None) -> np.ndarray:
        # When asset_mask is provided, we can't use cached values (different masks = different results)
        # Compute fresh with mask applied
        if asset_mask is not None:
            if name == "mac.combo":
                combo = None
                for mac_name in COMBINED_SIGNALS["mac.combo"]:
                    sig = self.get(mac_name, rank_mode, asset_mask)
                    if combo is None:
                        combo = np.copy(sig)
                    else:
                        combo = combo + np.nan_to_num(sig, nan=0.0)
                return sumabs_norm(combo)
            else:
                raw = self._compute_raw(name)
                return self._transform(raw, rank_mode, asset_mask)

        # No mask - use cache
        cache = self._cache_all if rank_mode == "all" else self._cache_grp

        if name not in cache:
            if name == "mac.combo":
                combo = None
                for mac_name in COMBINED_SIGNALS["mac.combo"]:
                    sig = self.get(mac_name, rank_mode)
                    if combo is None:
                        combo = np.copy(sig)
                    else:
                        combo = combo + np.nan_to_num(sig, nan=0.0)
                cache[name] = sumabs_norm(combo)
            else:
                raw = self._compute_raw(name)
                cache[name] = self._transform(raw, rank_mode, None)

        return cache[name]


def get_combined_signal(signal_names: list[str], cache: SignalCache, rank_mode: str,
                        asset_mask: np.ndarray | None = None) -> np.ndarray:
    if len(signal_names) == 1:
        return cache.get(signal_names[0], rank_mode, asset_mask)

    combined = None
    for name in signal_names:
        sig = cache.get(name, rank_mode, asset_mask)
        if combined is None:
            combined = np.copy(sig)
        else:
            combined = combined + np.nan_to_num(sig, nan=0.0)
    return sumabs_norm(combined)


# ============================================================================
# Asset-Level Risk Parity Weights
# ============================================================================
def compute_asset_rp_weights(vol_matrix: np.ndarray) -> np.ndarray:
    """Compute asset-level risk parity weights (1/vol).

    Returns weight matrix (n_days × n_assets) with 1/vol weights,
    normalized by sumabs and lagged by 1 day to avoid look-ahead.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_vol = 1.0 / vol_matrix
        inv_vol[~np.isfinite(inv_vol)] = 0.0

    # Normalize by row sumabs
    result = sumabs_norm(inv_vol)

    # Lag by 1 day to avoid look-ahead
    lagged = np.zeros_like(result)
    lagged[1:, :] = result[:-1, :]

    return lagged


def compute_asset_rp_weights_rolling_abs(pnl_matrix: np.ndarray, alpha: float = 2/366) -> np.ndarray:
    """Compute asset-level risk parity weights using rolling abs (EMA of |pnl|).

    Returns weight matrix (n_days × n_assets) with 1/EMA(|pnl|) weights,
    normalized by sumabs and lagged by 1 day to avoid look-ahead.
    """
    # Compute EMA of |pnl| per asset
    abs_pnl = np.abs(pnl_matrix)
    rolling_abs = _ema_2d_columnwise(abs_pnl, alpha)

    # Compute 1/rolling_abs weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_rolling_abs = 1.0 / rolling_abs
        inv_rolling_abs[~np.isfinite(inv_rolling_abs)] = 0.0

    # Normalize by row sumabs
    result = sumabs_norm(inv_rolling_abs)

    # Lag by 1 day to avoid look-ahead
    lagged = np.zeros_like(result)
    lagged[1:, :] = result[:-1, :]

    return lagged


def compute_asset_rp_weights_rolling_stdev(pnl_matrix: np.ndarray, alpha: float = 2/366) -> np.ndarray:
    """Compute asset-level risk parity weights using rolling stdev.

    Rolling stdev = sqrt(EMA(pnl^2) - EMA(pnl)^2)

    Returns weight matrix (n_days × n_assets) with 1/rolling_stdev weights,
    normalized by sumabs and lagged by 1 day to avoid look-ahead.
    """
    # Compute EMA of pnl and EMA of pnl^2 per asset
    ema_pnl = _ema_2d_columnwise(pnl_matrix, alpha)
    ema_pnl_sq = _ema_2d_columnwise(pnl_matrix ** 2, alpha)

    # Rolling variance = EMA(pnl^2) - EMA(pnl)^2
    rolling_var = ema_pnl_sq - ema_pnl ** 2
    rolling_var = np.maximum(rolling_var, 0.0)  # Ensure non-negative

    # Rolling stdev
    rolling_stdev = np.sqrt(rolling_var)

    # Compute 1/rolling_stdev weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_rolling_stdev = 1.0 / rolling_stdev
        inv_rolling_stdev[~np.isfinite(inv_rolling_stdev)] = 0.0

    # Normalize by row sumabs
    result = sumabs_norm(inv_rolling_stdev)

    # Lag by 1 day to avoid look-ahead
    lagged = np.zeros_like(result)
    lagged[1:, :] = result[:-1, :]

    return lagged


def compute_asset_rp_weights_rolling_mad(pnl_matrix: np.ndarray, window: int = 365) -> np.ndarray:
    """Compute asset-level risk parity weights using rolling MAD (Median Absolute Deviation).

    MAD = median(|x - median(x)|) over rolling window.
    Uses bottleneck.move_median for O(n) complexity instead of O(n*window).

    Returns weight matrix (n_days × n_assets) with 1/rolling_mad weights,
    normalized by sumabs and lagged by 1 day to avoid look-ahead.
    """
    n_assets = pnl_matrix.shape[1]
    rolling_mad = np.empty_like(pnl_matrix)

    # bottleneck.move_median works on 1D, so loop over columns (fast since heavy work is in C)
    for j in range(n_assets):
        col = pnl_matrix[:, j].astype(np.float64)
        # Rolling median of the column
        rolling_median = bn.move_median(col, window=window, min_count=1)
        # Absolute deviation from rolling median
        abs_dev = np.abs(col - rolling_median)
        # Rolling median of absolute deviations = MAD
        rolling_mad[:, j] = bn.move_median(abs_dev, window=window, min_count=1)

    # Compute 1/rolling_mad weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_rolling_mad = 1.0 / rolling_mad
        inv_rolling_mad[~np.isfinite(inv_rolling_mad)] = 0.0

    # Normalize by row sumabs
    result = sumabs_norm(inv_rolling_mad)

    # Lag by 1 day to avoid look-ahead
    lagged = np.zeros_like(result)
    lagged[1:, :] = result[:-1, :]

    return lagged


def apply_asset_rp(signal_weights: np.ndarray, asset_rp_weights: np.ndarray) -> np.ndarray:
    """Apply asset-level risk parity to signal weights.

    risk_weighted_signal = normabs(normabs(signals) * normabs(asset_rp_weights))

    Since both inputs are already normalized, we just multiply and re-normalize.
    """
    # Both should already be normalized, but be explicit
    sig_norm = sumabs_norm(signal_weights)
    rp_norm = sumabs_norm(asset_rp_weights)

    # Element-wise multiply and re-normalize
    combined = sig_norm * rp_norm
    return sumabs_norm(combined)


# ============================================================================
# Class Risk Parity Weights
# ============================================================================
def compute_class_rp_weights(
    pnl_matrix: np.ndarray, group_ids: np.ndarray, n_groups: int, alpha: float = 2/366
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk parity weights at the asset class level.

    1. Aggregate PnL by asset class (row-sum per class)
    2. Compute rolling vol (EMA of abs returns) per class
    3. Compute 1/vol weights, normalized by sumabs
    4. Expand to per-asset weights
    5. Lag by 1 day to avoid look-ahead

    Returns:
        asset_weights: weight matrix (n_days × n_assets) where assets in the same class get the same weight
        class_vol: rolling volatility per class (n_days × n_groups) for diagnostics
    """
    n_days, n_assets = pnl_matrix.shape

    # Aggregate PnL by class
    class_pnl = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_pnl[:, g] = np.nansum(pnl_matrix[:, mask], axis=1)

    # Compute rolling vol (EMA of abs returns)
    abs_class_pnl = np.abs(class_pnl)
    class_vol = _ema_2d_columnwise(abs_class_pnl, alpha)

    # Compute 1/vol weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_vol = 1.0 / class_vol
        inv_vol[~np.isfinite(inv_vol)] = 0.0

    # Normalize by sumabs per row (so class weights sum to 1)
    row_sum = np.nansum(np.abs(inv_vol), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_weights = inv_vol / row_sum
        class_weights[~np.isfinite(class_weights)] = 0.0

    # Lag by 1 day to avoid look-ahead
    class_weights_lagged = np.zeros_like(class_weights)
    class_weights_lagged[1:, :] = class_weights[:-1, :]

    # Expand to per-asset weights
    asset_weights = np.zeros((n_days, n_assets), dtype=np.float64)
    for a in range(n_assets):
        g = group_ids[a]
        asset_weights[:, a] = class_weights_lagged[:, g]

    return asset_weights, class_vol


def compute_class_rp_weights_rolling_mad(
    pnl_matrix: np.ndarray, group_ids: np.ndarray, n_groups: int, window: int = 365
) -> tuple[np.ndarray, np.ndarray]:
    """Compute class risk parity weights using rolling MAD (Median Absolute Deviation).

    1. Aggregate PnL by asset class (row-sum per class)
    2. Compute rolling MAD per class using bottleneck.move_median
    3. Compute 1/MAD weights, normalized by sumabs
    4. Expand to per-asset weights
    5. Lag by 1 day to avoid look-ahead

    Returns:
        asset_weights: weight matrix (n_days × n_assets) where assets in the same class get the same weight
        class_mad: rolling MAD per class (n_days × n_groups) for diagnostics
    """
    n_days, n_assets = pnl_matrix.shape

    # Aggregate PnL by class
    class_pnl = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_pnl[:, g] = np.nansum(pnl_matrix[:, mask], axis=1)

    # Compute rolling MAD per class using bottleneck
    class_mad = np.empty((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        col = class_pnl[:, g]
        rolling_median = bn.move_median(col, window=window, min_count=1)
        abs_dev = np.abs(col - rolling_median)
        class_mad[:, g] = bn.move_median(abs_dev, window=window, min_count=1)

    # Compute 1/MAD weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_mad = 1.0 / class_mad
        inv_mad[~np.isfinite(inv_mad)] = 0.0

    # Normalize by sumabs per row (so class weights sum to 1)
    row_sum = np.nansum(np.abs(inv_mad), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_weights = inv_mad / row_sum
        class_weights[~np.isfinite(class_weights)] = 0.0

    # Lag by 1 day to avoid look-ahead
    class_weights_lagged = np.zeros_like(class_weights)
    class_weights_lagged[1:, :] = class_weights[:-1, :]

    # Expand to per-asset weights
    asset_weights = np.zeros((n_days, n_assets), dtype=np.float64)
    for a in range(n_assets):
        g = group_ids[a]
        asset_weights[:, a] = class_weights_lagged[:, g]

    return asset_weights, class_mad


def compute_class_rp_weights_rolling_stdev(
    pnl_matrix: np.ndarray, group_ids: np.ndarray, n_groups: int, alpha: float = 2/366
) -> tuple[np.ndarray, np.ndarray]:
    """Compute class risk parity weights using rolling stdev.

    Rolling stdev = sqrt(EMA(pnl²) - EMA(pnl)²)

    1. Aggregate PnL by asset class (row-sum per class)
    2. Compute rolling stdev per class
    3. Compute 1/stdev weights, normalized by sumabs
    4. Expand to per-asset weights
    5. Lag by 1 day to avoid look-ahead

    Returns:
        asset_weights: weight matrix (n_days × n_assets) where assets in the same class get the same weight
        class_stdev: rolling stdev per class (n_days × n_groups) for diagnostics
    """
    n_days, n_assets = pnl_matrix.shape

    # Aggregate PnL by class
    class_pnl = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_pnl[:, g] = np.nansum(pnl_matrix[:, mask], axis=1)

    # Compute rolling stdev: sqrt(EMA(pnl²) - EMA(pnl)²)
    ema_pnl = _ema_2d_columnwise(class_pnl, alpha)
    ema_pnl_sq = _ema_2d_columnwise(class_pnl ** 2, alpha)
    rolling_var = ema_pnl_sq - ema_pnl ** 2
    rolling_var = np.maximum(rolling_var, 0.0)  # Numerical stability
    class_stdev = np.sqrt(rolling_var)

    # Compute 1/stdev weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_stdev = 1.0 / class_stdev
        inv_stdev[~np.isfinite(inv_stdev)] = 0.0

    # Normalize by sumabs per row (so class weights sum to 1)
    row_sum = np.nansum(np.abs(inv_stdev), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_weights = inv_stdev / row_sum
        class_weights[~np.isfinite(class_weights)] = 0.0

    # Lag by 1 day to avoid look-ahead
    class_weights_lagged = np.zeros_like(class_weights)
    class_weights_lagged[1:, :] = class_weights[:-1, :]

    # Expand to per-asset weights
    asset_weights = np.zeros((n_days, n_assets), dtype=np.float64)
    for a in range(n_assets):
        g = group_ids[a]
        asset_weights[:, a] = class_weights_lagged[:, g]

    return asset_weights, class_stdev


def compute_class_rp_weights_ivol(
    input_weights: np.ndarray, vol_matrix: np.ndarray, group_ids: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute class risk parity weights using weighted average Ivol per class.

    Risk measure per class = sum(ivol * |input_weights|) for assets in class.
    This gives the ivol-weighted exposure of the portfolio to each class.

    1. Compute weighted sum of ivol by class: sum(ivol * |weights|)
    2. Compute 1/class_ivol weights, normalized by sumabs
    3. Expand to per-asset weights
    4. Lag by 1 day to avoid look-ahead

    Returns:
        asset_weights: weight matrix (n_days × n_assets) where assets in the same class get the same weight
        class_ivol: weighted ivol per class (n_days × n_groups) for diagnostics
    """
    n_days, n_assets = input_weights.shape

    # Compute weighted sum of ivol by class
    abs_weights = np.abs(input_weights)
    weighted_ivol = abs_weights * vol_matrix  # element-wise: |weight| * ivol

    class_ivol = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_ivol[:, g] = np.nansum(weighted_ivol[:, mask], axis=1)

    # Compute 1/class_ivol weights
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_ivol = 1.0 / class_ivol
        inv_ivol[~np.isfinite(inv_ivol)] = 0.0

    # Normalize by sumabs per row (so class weights sum to 1)
    row_sum = np.nansum(np.abs(inv_ivol), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_weights = inv_ivol / row_sum
        class_weights[~np.isfinite(class_weights)] = 0.0

    # Lag by 1 day to avoid look-ahead
    class_weights_lagged = np.zeros_like(class_weights)
    class_weights_lagged[1:, :] = class_weights[:-1, :]

    # Expand to per-asset weights
    asset_weights = np.zeros((n_days, n_assets), dtype=np.float64)
    for a in range(n_assets):
        g = group_ids[a]
        asset_weights[:, a] = class_weights_lagged[:, g]

    return asset_weights, class_ivol


# ============================================================================
# Weight Combination
# ============================================================================
def combine_weights(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Combine two weight matrices using the sumabs normalization rule.

    combined = (w1 * w2) normalized so that sumabs(combined) is preserved.

    The rule ensures that the combined weights have consistent scale:
    combined = (w1 * w2) / scale
    where scale = sumabs(w1) * sumabs(w2) / sumabs(combined)

    In practice, we just element-wise multiply and re-normalize by sumabs.
    """
    combined = w1 * w2

    # Normalize each row by its sum of absolute values
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum_abs = np.nansum(np.abs(combined), axis=1, keepdims=True)
        result = combined / row_sum_abs
        result[~np.isfinite(result)] = 0.0

    return result


# ============================================================================
# PnL Computation
# ============================================================================
def compute_weighted_pnl_matrix(weight_matrix: np.ndarray, ctx: BacktestContext) -> np.ndarray:
    """Compute the weighted PnL matrix using inception-locked weights.

    This is the final step: multiply the final weight matrix by straddle PnLs,
    applying inception-locking (weight is determined at straddle start and
    held constant throughout its life).

    Returns: wpnl_matrix (n_days × n_assets)
    """
    return _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        weight_matrix, ctx.asset_starts, ctx.asset_counts
    )


def compute_preliminary_pnl_matrices(signal_weights: np.ndarray, ctx: BacktestContext):
    """Compute preliminary PnL matrices from signal weights for hedge computation.

    These are needed to compute rolling betas for hedging. We split by sign
    of signal weights to get long and short PnL matrices.

    Returns: (long_pnl_matrix, short_pnl_matrix, long_pnl_agg, short_pnl_agg)
    """
    long_weights = np.maximum(signal_weights, 0.0)
    long_pnl_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        long_weights, ctx.asset_starts, ctx.asset_counts
    )
    long_pnl_agg = np.nansum(long_pnl_matrix, axis=1)

    short_weights = np.minimum(signal_weights, 0.0)
    short_pnl_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        short_weights, ctx.asset_starts, ctx.asset_counts
    )
    short_pnl_agg = np.nansum(short_pnl_matrix, axis=1)

    return long_pnl_matrix, short_pnl_matrix, long_pnl_agg, short_pnl_agg


# ============================================================================
# Hedge Weight Computation (produces weight matrices, not PnL)
# ============================================================================
@njit(cache=True)
def _build_hedge_weights_global(
    signal_weights: np.ndarray, rolling_beta_lagged: np.ndarray
) -> np.ndarray:
    """Build hedge weight matrix for global hedging (numba-accelerated)."""
    n_days, n_assets = signal_weights.shape
    hedge_weights = np.ones((n_days, n_assets), dtype=np.float64)

    for t in range(n_days):
        beta = rolling_beta_lagged[t]
        for a in range(n_assets):
            sig = signal_weights[t, a]
            if np.isnan(sig) or sig == 0:
                hedge_weights[t, a] = 0.0
            elif sig > 0:  # Long position
                hedge_weights[t, a] = beta
            # else: short position, stays 1.0

    return hedge_weights


@njit(cache=True)
def _build_hedge_weights_by_group(
    signal_weights: np.ndarray, group_betas_lagged: np.ndarray, group_ids: np.ndarray
) -> np.ndarray:
    """Build hedge weight matrix for group hedging (numba-accelerated)."""
    n_days, n_assets = signal_weights.shape
    hedge_weights = np.ones((n_days, n_assets), dtype=np.float64)

    for t in range(n_days):
        for a in range(n_assets):
            sig = signal_weights[t, a]
            g = group_ids[a]
            if np.isnan(sig) or sig == 0:
                hedge_weights[t, a] = 0.0
            elif sig > 0:  # Long position
                hedge_weights[t, a] = group_betas_lagged[t, g]
            # else: short position, stays 1.0

    return hedge_weights


@njit(cache=True)
def _build_hedge_weights_gah(
    signal_weights: np.ndarray, group_betas_lagged: np.ndarray,
    global_beta_lagged: np.ndarray, group_ids: np.ndarray
) -> np.ndarray:
    """Build hedge weight matrix for GAH hedging (numba-accelerated)."""
    n_days, n_assets = signal_weights.shape
    hedge_weights = np.ones((n_days, n_assets), dtype=np.float64)

    for t in range(n_days):
        for a in range(n_assets):
            sig = signal_weights[t, a]
            g = group_ids[a]
            if np.isnan(sig) or sig == 0:
                hedge_weights[t, a] = 0.0
            elif sig > 0:  # Long position
                hedge_weights[t, a] = group_betas_lagged[t, g] * global_beta_lagged[t]
            # else: short position, stays 1.0

    return hedge_weights


def compute_hedge_weights_none(
    signal_weights: np.ndarray,
    **kwargs
) -> np.ndarray:
    """No hedging - return all ones (neutral weights)."""
    return np.ones_like(signal_weights)


def compute_hedge_weights_global(
    signal_weights: np.ndarray,
    long_pnl_agg: np.ndarray,
    short_pnl_agg: np.ndarray,
    alpha: float = 1/365,
    **kwargs
) -> np.ndarray:
    """Global hedge weights: shorts get 1.0, longs get beta."""
    # Compute rolling beta from aggregated long/short PnL
    rolling_beta = _compute_rolling_beta_ema(long_pnl_agg, short_pnl_agg, alpha)

    # Lag by 1 day to avoid look-ahead
    rolling_beta_lagged = np.zeros_like(rolling_beta)
    rolling_beta_lagged[1:] = rolling_beta[:-1]

    return _build_hedge_weights_global(signal_weights, rolling_beta_lagged)


def compute_hedge_weights_by_group(
    signal_weights: np.ndarray,
    long_pnl_matrix: np.ndarray,
    short_pnl_matrix: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    alpha: float = 1/365,
    **kwargs
) -> np.ndarray:
    """Hedge by group: each group gets its own beta."""
    # Compute rolling beta per group
    group_betas = _compute_rolling_beta_by_group_ema(
        long_pnl_matrix, short_pnl_matrix, group_ids, n_groups, alpha
    )

    # Lag by 1 day to avoid look-ahead
    group_betas_lagged = np.zeros_like(group_betas)
    group_betas_lagged[1:, :] = group_betas[:-1, :]

    return _build_hedge_weights_by_group(signal_weights, group_betas_lagged, group_ids)


def compute_hedge_weights_gah(
    signal_weights: np.ndarray,
    long_pnl_matrix: np.ndarray,
    short_pnl_matrix: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    alpha: float = 1/365,
    **kwargs
) -> np.ndarray:
    """Group-then-All hedge: first hedge within groups, then global."""
    n_days = signal_weights.shape[0]

    # Stage 1: Compute group betas
    group_betas = _compute_rolling_beta_by_group_ema(
        long_pnl_matrix, short_pnl_matrix, group_ids, n_groups, alpha
    )

    # Compute group-hedged aggregates for stage 2 (vectorized)
    hedged_long_total = np.zeros(n_days, dtype=np.float64)
    hedged_short_total = np.zeros(n_days, dtype=np.float64)

    for g in range(n_groups):
        mask = group_ids == g
        long_g = np.nansum(long_pnl_matrix[:, mask], axis=1)
        short_g = np.nansum(short_pnl_matrix[:, mask], axis=1)
        hedged_long_total += group_betas[:, g] * long_g
        hedged_short_total += short_g

    # Stage 2: Global beta on hedged aggregates
    global_beta = _compute_rolling_beta_ema(hedged_long_total, hedged_short_total, alpha)

    # Lag both by 1 day
    group_betas_lagged = np.zeros_like(group_betas)
    group_betas_lagged[1:, :] = group_betas[:-1, :]
    global_beta_lagged = np.zeros_like(global_beta)
    global_beta_lagged[1:] = global_beta[:-1]

    return _build_hedge_weights_gah(signal_weights, group_betas_lagged, global_beta_lagged, group_ids)


HEDGE_WEIGHT_FUNCTIONS = {
    "none": compute_hedge_weights_none,
    "h": compute_hedge_weights_global,
    "hg": compute_hedge_weights_by_group,
    "gah": compute_hedge_weights_gah,
}


# ============================================================================
# Stats
# ============================================================================
def compute_sharpe(pnl: np.ndarray) -> float:
    mean = np.nanmean(pnl)
    std = np.nanstd(pnl)
    if std == 0:
        return np.nan
    return (mean * 252) / (std * np.sqrt(252))


def compute_max_drawdown(pnl: np.ndarray) -> float:
    cum_pnl = np.nancumsum(pnl)
    running_max = np.maximum.accumulate(np.nan_to_num(cum_pnl, nan=0.0))
    drawdown = running_max - cum_pnl
    return np.nanmax(drawdown)


def compute_ls_correl(long_pnl: np.ndarray, short_pnl: np.ndarray) -> float:
    valid = ~(np.isnan(long_pnl) | np.isnan(short_pnl))
    if np.sum(valid) < 2:
        return np.nan
    long_valid = long_pnl[valid]
    short_valid = short_pnl[valid]
    if np.std(long_valid) == 0 or np.std(short_valid) == 0:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(long_valid, short_valid)[0, 1]
    return corr if np.isfinite(corr) else np.nan


def compute_stats(pnl: np.ndarray, long_pnl: np.ndarray, short_pnl: np.ndarray) -> PortfolioStats:
    return PortfolioStats(
        sharpe=compute_sharpe(pnl),
        long_sharpe=compute_sharpe(long_pnl),
        short_sharpe=compute_sharpe(short_pnl),
        ls_correl=compute_ls_correl(long_pnl, short_pnl),
        max_drawdown=compute_max_drawdown(pnl),
    )


# ============================================================================
# Data Loading
# ============================================================================
def load_arrow_as_dict(arrow_path: str) -> dict:
    table = pf.read_table(arrow_path)
    return {col: table[col].to_numpy(zero_copy_only=False) for col in table.column_names}


def parse_weights(wgt_arr: np.ndarray) -> np.ndarray:
    weights = np.ones(len(wgt_arr), dtype=np.float64)
    for i, w in enumerate(wgt_arr):
        try:
            s = str(w).strip()
            if s:
                weights[i] = float(s) / 100.0
        except (ValueError, TypeError):
            pass
    return weights


def prepare_asset_groups(asset_ids: np.ndarray, n_assets: int):
    sort_idx = np.argsort(asset_ids)
    asset_starts = np.zeros(n_assets, dtype=np.int32)
    asset_counts = np.zeros(n_assets, dtype=np.int32)

    sorted_assets = asset_ids[sort_idx]
    for a in range(n_assets):
        mask = sorted_assets == a
        if np.any(mask):
            indices = np.where(mask)[0]
            asset_starts[a] = indices[0]
            asset_counts[a] = len(indices)

    return sort_idx, asset_starts, asset_counts


def load_group_table(amt_path: str) -> list[tuple[str, re.Pattern, str]]:
    with open(amt_path) as f:
        data = yaml.safe_load(f)

    table = data.get("group_table", {})
    columns = table.get("Columns", [])
    rows = table.get("Rows", [])

    field_idx = columns.index("field")
    rgx_idx = columns.index("rgx")
    value_idx = columns.index("value")

    rules = []
    for row in rows:
        field = row[field_idx]
        pattern = re.compile(row[rgx_idx])
        value = row[value_idx]
        rules.append((field, pattern, value))
    return rules


def load_asset_class_map(amt_path: str) -> dict[str, str]:
    with open(amt_path) as f:
        data = yaml.safe_load(f)

    amt = data.get("amt", {})
    class_map = {}
    for asset_data in amt.values():
        if isinstance(asset_data, dict):
            underlying = asset_data.get("Underlying")
            cls = asset_data.get("Class", "")
            if underlying:
                class_map[underlying] = cls
    return class_map


def assign_group(underlying: str, cls: str, rules: list) -> str:
    field_values = {"Underlying": underlying, "Class": cls}
    for field, pattern, value in rules:
        field_val = field_values.get(field, "")
        if pattern.match(field_val):
            return value
    return "error"


def build_asset_group_table(asset_names: list[str], amt_path: str) -> dict[str, str]:
    rules = load_group_table(amt_path)
    class_map = load_asset_class_map(amt_path)

    group_map = {}
    for underlying in asset_names:
        cls = class_map.get(underlying, "")
        group = assign_group(underlying, cls, rules)
        group_map[underlying] = group
    return group_map


@st.cache_resource
def load_all_data():
    """Load and prepare all data for backtesting."""
    t_start = time.perf_counter()

    straddles = load_arrow_as_dict("data/straddles.arrow")
    valuations = load_arrow_as_dict("data/valuations.arrow")

    t_file_load = time.perf_counter()

    weights = parse_weights(straddles["wgt"])
    out0s = straddles["out0"].astype(np.int32)
    lens = straddles["length"].astype(np.int32)
    starts = straddles["month_start_epoch"].astype(np.int32)

    asset_str = np.asarray([str(x) for x in straddles["asset"]], dtype=object)
    asset_codes, asset_names = pd.factorize(asset_str, sort=True)
    asset_ids = asset_codes.astype(np.int32)
    n_assets = len(asset_names)

    # Build groups
    group_map = build_asset_group_table(list(asset_names), "data/amt.yml")
    asset_groups = np.array([group_map.get(str(name), "error") for name in asset_names])
    unique_groups = np.unique(asset_groups)
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_id[g] for g in asset_groups], dtype=np.int32)
    n_groups = len(unique_groups)

    # Grid parameters
    d0 = int(starts.min())
    d1 = int((starts + lens - 1).max())
    grid_size = d1 - d0 + 1

    pnl = valuations["pnl"]
    vol = valuations["vol"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    # Prepare sorted arrays
    sort_idx, asset_starts, asset_counts = prepare_asset_groups(asset_ids, n_assets)
    out0s_sorted = out0s[sort_idx]
    lens_sorted = lens[sort_idx]
    starts_sorted = starts[sort_idx]
    weights_sorted = weights[sort_idx]

    # Build matrices
    pnl_matrix = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, pnl, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )

    vol_matrix = _aggregate_vol_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, vol, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )

    # Compute npnl and rnpnl
    with np.errstate(divide='ignore', invalid='ignore'):
        npnl = pnl_matrix / vol_matrix
        npnl[~np.isfinite(npnl)] = np.nan

    market = np.nansum(npnl, axis=1, keepdims=True)
    rnpnl = npnl - market

    # Compute winsorized npnl (rolling 1%/99% quantile clipping)
    wpnl = _rolling_winsorize_parallel(npnl, window=365, lo_pct=0.01, hi_pct=0.99)

    # Build context
    ctx = BacktestContext(
        raw_pnl_matrix=pnl_matrix,
        npnl=npnl,
        rnpnl=rnpnl,
        wpnl=wpnl,
        vol_matrix=vol_matrix,
        group_ids=group_ids,
        n_groups=n_groups,
        unique_groups=list(unique_groups),
        out0s=out0s_sorted,
        lens=lens_sorted,
        starts=starts_sorted,
        asset_ids=asset_ids[sort_idx],
        weights=weights_sorted,
        asset_starts=asset_starts,
        asset_counts=asset_counts,
        d0=d0,
        grid_size=grid_size,
        n_assets=n_assets,
        pnl=pnl,
        dte=dte,
        have_dte=have_dte,
    )

    t_end = time.perf_counter()

    # Return timings in ms
    timings = {
        "t_file_load": (t_file_load - t_start) * 1000,
        "t_preprocess": (t_end - t_file_load) * 1000,
    }

    return ctx, d0, timings


# ============================================================================
# Strategy Execution (Weight-Centric Pipeline)
# ============================================================================
def run_strategy(input_type: str, signal_names: list[str], rank_mode: str,
                 rp_hedge: str, final_hedge: str,
                 ctx: BacktestContext, caches: dict[str, SignalCache],
                 asset_filter: list[str] | None = None,
                 asset_rp: str = "none",
                 class_rp: str = "none",
                 t_file_load_ms: float = 0.0,
                 t_preprocess_ms: float = 0.0):
    """Execute a single strategy using two-hedge weight-centric pipeline.

    Pipeline:
    1. signal_weights: ranked/scaled signal from input data
    2. risk_weighted_signal = combine(signal, asset_rp)  # normabs rule
    3. rp_hedge_pnl = inception_pnl(risk_weighted_signal)  # for RP hedge β
    4. rp_hedge_weights = hedge(rp_hedge_pnl)  # β matrix (1 for shorts, β for longs)
    5. asset_weights = risk_weighted_signal * rp_hedge  # simple multiply
    6. class_rp_pnl = inception_pnl(asset_weights)  # for class RP vol
    7. class_rp_weights = 1/vol(class_rp_pnl by class)
    8. asset_class_weights = combine(asset_weights, class_rp)  # normabs rule
    9. final_hedge_pnl = inception_pnl(asset_class_weights)  # for final hedge β
    10. final_hedge_weights = hedge(final_hedge_pnl)  # β matrix
    11. final_weights = asset_class_weights * final_hedge  # simple multiply
    12. wpnl = inception_pnl(final_weights)

    Hedges are applied via simple multiplication (scaling longs by β).
    Weight combinations (signal+asset_rp, asset+class_rp) use normabs rule.
    """
    t0 = time.perf_counter()

    # Compute asset mask early - excluded assets won't participate in any calculations
    if asset_filter:
        group_ids_selected = [ctx.unique_groups.index(g) for g in asset_filter]
        asset_mask = np.isin(ctx.group_ids, group_ids_selected)
    else:
        asset_mask = None

    # Stage 1: Signal Weights (with asset mask applied before ranking)
    cache = caches[input_type]
    signal_weights = get_combined_signal(signal_names, cache, rank_mode, asset_mask)

    t_signal = time.perf_counter()

    # Stage 2: Asset-Level Risk Parity weights
    # All options produce sumabs_norm'd weights:
    # - "none": sumabs_norm(ones) = 1/N per live asset
    # - "inverse_ivol": sumabs_norm(1/vol)
    # - "rolling_abs": sumabs_norm(1/EMA(|pnl|))
    # - "rolling_stdev": sumabs_norm(1/rolling_stdev)
    if asset_rp == "inverse_ivol":
        asset_rp_weights = compute_asset_rp_weights(ctx.vol_matrix)
    elif asset_rp == "rolling_abs":
        asset_rp_weights = compute_asset_rp_weights_rolling_abs(ctx.npnl)
    elif asset_rp == "rolling_stdev":
        asset_rp_weights = compute_asset_rp_weights_rolling_stdev(ctx.npnl)
    elif asset_rp == "rolling_mad":
        asset_rp_weights = compute_asset_rp_weights_rolling_mad(ctx.npnl)
    else:
        asset_rp_weights = sumabs_norm(np.ones_like(signal_weights))

    # Apply to signal weights using normabs combination rule
    if asset_rp != "none":
        risk_weighted_signal = apply_asset_rp(signal_weights, asset_rp_weights)
    else:
        risk_weighted_signal = signal_weights
    t_asset_rp = time.perf_counter()

    # Stage 2b: Straddle diversification PnL matrices (no signals, just weights)
    # - equal_weighted_pnl: sumabs_norm(ones) = 1/N per live asset
    # - asset_rp_pnl: asset_rp_weights (sumabs_norm'd, either ones or 1/vol)
    equal_weighted_pnl = compute_weighted_pnl_matrix(sumabs_norm(np.ones_like(signal_weights)), ctx)
    asset_rp_pnl_matrix = compute_weighted_pnl_matrix(asset_rp_weights, ctx)

    # Stage 3: RP Hedge PnL - compute preliminary PnL for RP hedge β
    rp_long_pnl, rp_short_pnl, rp_long_agg, rp_short_agg = \
        compute_preliminary_pnl_matrices(risk_weighted_signal, ctx)
    t_rp_hedge_pnl = time.perf_counter()

    # Stage 4: RP Hedge Weights - β matrix (1 for shorts, β for longs)
    rp_hedge_fn = HEDGE_WEIGHT_FUNCTIONS[rp_hedge]
    if rp_hedge == "none":
        rp_hedge_weights = rp_hedge_fn(signal_weights=risk_weighted_signal)
    elif rp_hedge == "h":
        rp_hedge_weights = rp_hedge_fn(
            signal_weights=risk_weighted_signal,
            long_pnl_agg=rp_long_agg,
            short_pnl_agg=rp_short_agg
        )
    else:  # hg, gah
        rp_hedge_weights = rp_hedge_fn(
            signal_weights=risk_weighted_signal,
            long_pnl_matrix=rp_long_pnl,
            short_pnl_matrix=rp_short_pnl,
            group_ids=ctx.group_ids,
            n_groups=ctx.n_groups
        )
    t_rp_hedge = time.perf_counter()

    # Stage 5: Asset Weights - simple multiply (hedge scales longs by β)
    asset_weights = risk_weighted_signal * rp_hedge_weights
    t_asset_weights = time.perf_counter()

    # Stage 6: Class RP PnL - compute PnL from asset_weights for class RP vol
    class_long_pnl, class_short_pnl, _, _ = \
        compute_preliminary_pnl_matrices(asset_weights, ctx)
    class_rp_pnl_matrix = class_long_pnl + class_short_pnl
    t_class_rp_pnl = time.perf_counter()

    # Stage 7: Class Risk Parity Weights - 1/vol(class) expanded to assets
    if class_rp == "ivol":
        class_rp_weights, class_vol = compute_class_rp_weights_ivol(
            asset_weights, ctx.vol_matrix, ctx.group_ids, ctx.n_groups
        )
    elif class_rp == "rolling_abs":
        class_rp_weights, class_vol = compute_class_rp_weights(
            class_rp_pnl_matrix, ctx.group_ids, ctx.n_groups
        )
    elif class_rp == "rolling_stdev":
        class_rp_weights, class_vol = compute_class_rp_weights_rolling_stdev(
            class_rp_pnl_matrix, ctx.group_ids, ctx.n_groups
        )
    elif class_rp == "rolling_mad":
        class_rp_weights, class_vol = compute_class_rp_weights_rolling_mad(
            class_rp_pnl_matrix, ctx.group_ids, ctx.n_groups
        )
    else:  # "none"
        class_rp_weights = np.ones_like(signal_weights)
        class_vol = None  # Not computed for "none"
    t_class_rp = time.perf_counter()

    # Stage 8: Asset Class Weights - combine(asset_weights, class_rp) using normabs rule
    asset_class_weights = combine_weights(asset_weights, class_rp_weights)
    t_asset_class_weights = time.perf_counter()

    # Stage 9: Final Hedge PnL - compute PnL for final hedge β
    final_long_pnl, final_short_pnl, final_long_agg, final_short_agg = \
        compute_preliminary_pnl_matrices(asset_class_weights, ctx)
    t_final_hedge_pnl = time.perf_counter()

    # Stage 10: Final Hedge Weights - β matrix
    final_hedge_fn = HEDGE_WEIGHT_FUNCTIONS[final_hedge]
    if final_hedge == "none":
        final_hedge_weights = final_hedge_fn(signal_weights=asset_class_weights)
    elif final_hedge == "h":
        final_hedge_weights = final_hedge_fn(
            signal_weights=asset_class_weights,
            long_pnl_agg=final_long_agg,
            short_pnl_agg=final_short_agg
        )
    else:  # hg, gah
        final_hedge_weights = final_hedge_fn(
            signal_weights=asset_class_weights,
            long_pnl_matrix=final_long_pnl,
            short_pnl_matrix=final_short_pnl,
            group_ids=ctx.group_ids,
            n_groups=ctx.n_groups
        )
    t_final_hedge = time.perf_counter()

    # Stage 11: Final Weights - simple multiply (hedge scales longs by β)
    final_weights = asset_class_weights * final_hedge_weights
    t_final_weights = time.perf_counter()

    # Stage 12: Final PnL Computation
    wpnl_matrix = compute_weighted_pnl_matrix(final_weights, ctx)
    total_pnl = np.nansum(wpnl_matrix, axis=1)

    # Compute long/short breakdown from final weights
    long_final = np.maximum(final_weights, 0.0)
    short_final = np.minimum(final_weights, 0.0)
    long_matrix = compute_weighted_pnl_matrix(long_final, ctx)
    short_matrix = compute_weighted_pnl_matrix(short_final, ctx)
    long_pnl = np.nansum(long_matrix, axis=1)
    short_pnl = np.nansum(short_matrix, axis=1)

    t_pnl = time.perf_counter()

    stats = compute_stats(total_pnl, long_pnl, short_pnl)

    # Build workflow info
    n_active = int(np.sum(asset_mask)) if asset_mask is not None else ctx.n_assets
    workflow = WorkflowInfo(
        n_days=ctx.grid_size,
        n_assets=ctx.n_assets,
        n_active_assets=n_active,
        n_groups=ctx.n_groups,
        t_file_load=t_file_load_ms,
        t_preprocess=t_preprocess_ms,
        t_signal=(t_signal - t0) * 1000,
        t_asset_rp=(t_asset_rp - t_signal) * 1000,
        t_rp_hedge_pnl=(t_rp_hedge_pnl - t_asset_rp) * 1000,
        t_rp_hedge=(t_rp_hedge - t_rp_hedge_pnl) * 1000,
        t_asset_weights=(t_asset_weights - t_rp_hedge) * 1000,
        t_class_rp_pnl=(t_class_rp_pnl - t_asset_weights) * 1000,
        t_class_rp=(t_class_rp - t_class_rp_pnl) * 1000,
        t_asset_class_weights=(t_asset_class_weights - t_class_rp) * 1000,
        t_final_hedge_pnl=(t_final_hedge_pnl - t_asset_class_weights) * 1000,
        t_final_hedge=(t_final_hedge - t_final_hedge_pnl) * 1000,
        t_final_weights=(t_final_weights - t_final_hedge) * 1000,
        t_pnl=(t_pnl - t_final_weights) * 1000,
        t_total=(t_pnl - t0) * 1000,
        input_shape=(ctx.grid_size, ctx.n_assets),
        signal_weights_shape=signal_weights.shape,
        asset_rp_weights_shape=asset_rp_weights.shape,
        risk_weighted_signal_shape=risk_weighted_signal.shape,
        rp_hedge_pnl_shape=rp_long_pnl.shape,
        rp_hedge_weights_shape=rp_hedge_weights.shape,
        asset_weights_shape=asset_weights.shape,
        class_rp_pnl_shape=class_rp_pnl_matrix.shape,
        class_rp_weights_shape=class_rp_weights.shape,
        asset_class_weights_shape=asset_class_weights.shape,
        final_hedge_pnl_shape=final_long_pnl.shape,
        final_hedge_weights_shape=final_hedge_weights.shape,
        final_weights_shape=final_weights.shape,
        wpnl_shape=wpnl_matrix.shape,
    )

    # PnL matrices for class vol visualization
    # class_rp_pnl_matrix = before class RP (computed earlier at Stage 6)
    # final_pnl_matrix = after class RP (from Stage 9)
    final_pnl_matrix = final_long_pnl + final_short_pnl

    return total_pnl, long_pnl, short_pnl, stats, workflow, wpnl_matrix, long_matrix, short_matrix, class_rp_pnl_matrix, final_pnl_matrix, rp_hedge_weights, signal_weights, equal_weighted_pnl, asset_rp_pnl_matrix


# ============================================================================
# Plotting
# ============================================================================
def plot_cumulative_pnl(pnl: np.ndarray, long_pnl: np.ndarray, short_pnl: np.ndarray, d0: int):
    """Create cumulative PnL chart with Plotly."""
    cum_pnl = np.nancumsum(pnl)
    cum_long = np.nancumsum(long_pnl)
    cum_short = np.nancumsum(short_pnl)

    # Create date index
    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=len(pnl), freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=cum_pnl,
        mode='lines',
        name='Total',
        line=dict(color='#2196F3', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=cum_long,
        mode='lines',
        name='Long',
        line=dict(color='#4CAF50', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=cum_short,
        mode='lines',
        name='Short',
        line=dict(color='#F44336', width=1.5)
    ))

    fig.update_layout(
        title="Cumulative PnL",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_cumulative_pnl_by_class(
    total_matrix: np.ndarray, group_ids: np.ndarray, unique_groups: list[str], d0: int
):
    """Create cumulative PnL chart by asset class with Plotly."""
    n_days = total_matrix.shape[0]
    n_groups = len(unique_groups)

    # Create date index
    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    # Color palette for classes
    colors = [
        '#2196F3',  # blue
        '#4CAF50',  # green
        '#F44336',  # red
        '#FF9800',  # orange
        '#9C27B0',  # purple
        '#00BCD4',  # cyan
        '#E91E63',  # pink
        '#8BC34A',  # light green
        '#3F51B5',  # indigo
        '#FFEB3B',  # yellow
    ]

    fig = go.Figure()

    for g in range(n_groups):
        # Sum PnL for assets in this group
        mask = group_ids == g
        class_pnl = np.nansum(total_matrix[:, mask], axis=1)
        cum_class_pnl = np.nancumsum(class_pnl)

        fig.add_trace(go.Scatter(
            x=dates, y=cum_class_pnl,
            mode='lines',
            name=unique_groups[g].capitalize(),
            line=dict(color=colors[g % len(colors)], width=2)
        ))

    fig.update_layout(
        title="Cumulative PnL by Asset Class",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def compute_rolling_correlation(long_pnl: np.ndarray, short_pnl: np.ndarray, alpha: float = 1/365) -> np.ndarray:
    """Compute rolling EMA correlation between long and short legs."""
    n = len(long_pnl)
    corr = np.full(n, np.nan)

    # EMA of products and squares for correlation
    ema_long = 0.0
    ema_short = 0.0
    ema_long_sq = 0.0
    ema_short_sq = 0.0
    ema_long_short = 0.0
    initialized = False

    for i in range(n):
        l = long_pnl[i] if not np.isnan(long_pnl[i]) else 0.0
        s = short_pnl[i] if not np.isnan(short_pnl[i]) else 0.0

        if not initialized:
            ema_long = l
            ema_short = s
            ema_long_sq = l * l
            ema_short_sq = s * s
            ema_long_short = l * s
            initialized = True
        else:
            ema_long = alpha * l + (1 - alpha) * ema_long
            ema_short = alpha * s + (1 - alpha) * ema_short
            ema_long_sq = alpha * (l * l) + (1 - alpha) * ema_long_sq
            ema_short_sq = alpha * (s * s) + (1 - alpha) * ema_short_sq
            ema_long_short = alpha * (l * s) + (1 - alpha) * ema_long_short

        # Compute correlation: cov(L,S) / (std(L) * std(S))
        var_long = ema_long_sq - ema_long * ema_long
        var_short = ema_short_sq - ema_short * ema_short
        cov = ema_long_short - ema_long * ema_short

        if var_long > 1e-10 and var_short > 1e-10:
            corr[i] = cov / np.sqrt(var_long * var_short)

    return corr


def plot_rolling_vol(total_pnl: np.ndarray, d0: int, alpha: float = 2/365):
    """Plot rolling strategy volatility (EMA of |total_pnl|)."""
    # Compute EMA of absolute PnL
    abs_pnl = np.abs(total_pnl)
    n = len(abs_pnl)
    rolling_vol = np.zeros(n, dtype=np.float64)
    rolling_vol[0] = abs_pnl[0] if not np.isnan(abs_pnl[0]) else 0.0

    for i in range(1, n):
        val = abs_pnl[i]
        if np.isnan(val):
            rolling_vol[i] = rolling_vol[i-1]
        else:
            rolling_vol[i] = alpha * val + (1 - alpha) * rolling_vol[i-1]

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n, freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=rolling_vol,
        mode='lines',
        name='Rolling Vol',
        line=dict(color='#FF5722', width=2)
    ))

    fig.update_layout(
        title="Strategy Rolling Volatility (365-day EMA of |PnL|)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_rolling_correlation(long_pnl: np.ndarray, short_pnl: np.ndarray, d0: int):
    """Create rolling correlation chart with Plotly."""
    corr = compute_rolling_correlation(long_pnl, short_pnl, alpha=1/365)

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=len(long_pnl), freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=corr,
        mode='lines',
        name='L/S Correlation',
        line=dict(color='#9C27B0', width=2)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Rolling L/S Correlation (365-day EMA)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_cross_sectional_range(total_matrix: np.ndarray, d0: int):
    """Plot cross-sectional range (max - min) of per-asset PnLs per day."""
    # Compute daily range (ptp = peak-to-peak = max - min)
    daily_max = np.nanmax(total_matrix, axis=1)
    daily_min = np.nanmin(total_matrix, axis=1)
    daily_range = daily_max - daily_min

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=len(daily_range), freq='D')

    fig = go.Figure()

    # Show range as filled area between min and max
    fig.add_trace(go.Scatter(
        x=dates, y=daily_max,
        mode='lines',
        name='Max',
        line=dict(color='#4CAF50', width=1),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=daily_min,
        mode='lines',
        name='Min',
        line=dict(color='#F44336', width=1),
        fill='tonexty',
        fillcolor='rgba(156, 39, 176, 0.2)'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Cross-Sectional PnL Range (per-asset min/max)",
        xaxis_title="Date",
        yaxis_title="PnL",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_cross_sectional_hit_ratio(total_matrix: np.ndarray, d0: int):
    """Plot cross-sectional hit ratio (% of assets with positive returns per day)."""
    # Count positive returns and valid (non-nan) entries per day
    positive_count = np.nansum(total_matrix > 0, axis=1)
    valid_count = np.sum(~np.isnan(total_matrix), axis=1)

    # Compute hit ratio (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_ratio = positive_count / valid_count
        hit_ratio[valid_count == 0] = np.nan

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=len(hit_ratio), freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=hit_ratio,
        mode='lines',
        name='Hit Ratio',
        line=dict(color='#FF9800', width=2)
    ))

    # Add 50% reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Cross-Sectional Hit Ratio (% positive returns)",
        xaxis_title="Date",
        yaxis_title="Hit Ratio",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_cross_sectional_winloss_ratio(total_matrix: np.ndarray, d0: int):
    """Plot cross-sectional win/loss ratio (sum of gains / sum of losses per day)."""
    # Sum of positive returns per day
    gains = np.where(total_matrix > 0, total_matrix, 0)
    total_gains = np.nansum(gains, axis=1)

    # Sum of negative returns per day (absolute value)
    losses = np.where(total_matrix < 0, total_matrix, 0)
    total_losses = np.abs(np.nansum(losses, axis=1))

    # Compute win/loss ratio (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        winloss_ratio = total_gains / total_losses
        winloss_ratio[total_losses == 0] = np.nan  # No losses = undefined

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=len(winloss_ratio), freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=winloss_ratio,
        mode='lines',
        name='Win/Loss Ratio',
        line=dict(color='#E91E63', width=2)
    ))

    # Add 1.0 reference line (breakeven)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Cross-Sectional Win/Loss Ratio (gains / losses)",
        xaxis_title="Date",
        yaxis_title="Win/Loss Ratio",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_class_realized_vol(
    pnl_matrix: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int, alpha: float = 2/366,
    title: str = "Realized Class Volatility (after Class RP weights)"
):
    """Plot realized rolling volatility per class.

    Used to compare before/after class RP - if RP is working,
    all classes should have similar volatility after RP.
    All measures are row-normalized (sumabs=1) for comparability.
    """
    n_groups = len(unique_groups)
    n_days = pnl_matrix.shape[0]

    # Aggregate PnL by class
    class_pnl = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_pnl[:, g] = np.nansum(pnl_matrix[:, mask], axis=1)

    # Compute rolling vol (EMA of |PnL|) - same method as class RP calculation
    class_vol = _ema_2d_columnwise(np.abs(class_pnl), alpha)

    # Normalize by sumabs per row for comparability
    row_sum = np.nansum(np.abs(class_vol), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_vol_norm = class_vol / row_sum
        class_vol_norm[~np.isfinite(class_vol_norm)] = 0.0

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    fig = go.Figure()

    # Color palette for classes
    colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800',
              '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#FFC107']

    for g, group_name in enumerate(unique_groups):
        fig.add_trace(go.Scatter(
            x=dates, y=class_vol_norm[:, g],
            mode='lines',
            name=group_name,
            line=dict(color=colors[g % len(colors)], width=1.5)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility (sumabs normalized)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_class_diversification(
    pnl_matrix: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int, alpha: float = 2/365,
    title: str = "Class Diversification Ratio"
):
    """Plot rolling diversification ratio across asset classes.

    Diversification ratio = EMA(sum(|class_pnl|)) / EMA(|sum(class_pnl)|)

    - Numerator: sum of absolute class PnLs (treats each class independently)
    - Denominator: absolute value of total PnL (allows cancellation)

    Ratio is floored at 1. Higher values = more diversification benefit.
    If all classes are perfectly correlated, ratio = 1.
    """
    n_groups = len(unique_groups)
    n_days = pnl_matrix.shape[0]

    # Aggregate PnL by class
    class_pnl = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_pnl[:, g] = np.nansum(pnl_matrix[:, mask], axis=1)

    # Numerator: sum of |class_pnl| per day
    sum_abs_class = np.sum(np.abs(class_pnl), axis=1)

    # Denominator: |sum of class_pnl| per day
    abs_sum_class = np.abs(np.sum(class_pnl, axis=1))

    # Apply EMA to both
    ema_sum_abs = np.zeros(n_days, dtype=np.float64)
    ema_abs_sum = np.zeros(n_days, dtype=np.float64)
    ema_sum_abs[0] = sum_abs_class[0] if not np.isnan(sum_abs_class[0]) else 0.0
    ema_abs_sum[0] = abs_sum_class[0] if not np.isnan(abs_sum_class[0]) else 0.0

    for i in range(1, n_days):
        val1 = sum_abs_class[i]
        val2 = abs_sum_class[i]
        if np.isnan(val1):
            ema_sum_abs[i] = ema_sum_abs[i-1]
        else:
            ema_sum_abs[i] = alpha * val1 + (1 - alpha) * ema_sum_abs[i-1]
        if np.isnan(val2):
            ema_abs_sum[i] = ema_abs_sum[i-1]
        else:
            ema_abs_sum[i] = alpha * val2 + (1 - alpha) * ema_abs_sum[i-1]

    # Compute ratio, floor at 1
    with np.errstate(divide='ignore', invalid='ignore'):
        div_ratio = ema_sum_abs / ema_abs_sum
        div_ratio[~np.isfinite(div_ratio)] = 1.0
        div_ratio = np.maximum(div_ratio, 1.0)

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=div_ratio,
        mode='lines',
        name='Diversification Ratio',
        line=dict(color='#673AB7', width=2)
    ))

    # Add floor line at 1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Diversification Ratio (≥1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_rp_beta_by_class(
    rp_hedge_weights: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int,
    title: str = "RP Hedge Beta by Asset Class"
):
    """Plot RP hedge beta values (positive weights) averaged by asset class.

    The rp_hedge_weights matrix contains:
    - 1.0 for short positions
    - beta for long positions (computed from rolling regression)
    - 0.0 for inactive positions

    This plot shows the average beta for long positions within each asset class.
    """
    n_groups = len(unique_groups)
    n_days = rp_hedge_weights.shape[0]

    # Compute average positive weight (beta) per class per day
    class_beta = np.zeros((n_days, n_groups), dtype=np.float64)
    for g in range(n_groups):
        mask = group_ids == g
        class_weights = rp_hedge_weights[:, mask]
        # Only consider positive weights (betas for longs), excluding 1.0 (shorts) and 0.0 (inactive)
        # Beta values are typically not exactly 1.0, so we look for values > 0 and != 1
        for d in range(n_days):
            row = class_weights[d, :]
            # Get positive values that aren't exactly 1.0 (those are shorts)
            beta_vals = row[(row > 0) & (row != 1.0)]
            if len(beta_vals) > 0:
                class_beta[d, g] = np.mean(beta_vals)
            else:
                # If no beta values, check if there are any positive weights
                pos_vals = row[row > 0]
                if len(pos_vals) > 0:
                    class_beta[d, g] = np.mean(pos_vals)

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    fig = go.Figure()

    # Color palette for classes
    colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800',
              '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#FFC107']

    for g, group_name in enumerate(unique_groups):
        fig.add_trace(go.Scatter(
            x=dates, y=class_beta[:, g],
            mode='lines',
            name=group_name,
            line=dict(color=colors[g % len(colors)], width=1.5)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Beta (hedge ratio for longs)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_histogram(data: np.ndarray, title: str, xlabel: str, nbins: int = 100):
    """Create a histogram using Plotly."""
    # Flatten and remove NaN values
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=valid_data,
        nbinsx=nbins,
        marker_color='#2196F3',
        opacity=0.75
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Count",
        margin=dict(l=50, r=20, t=60, b=50),
        bargap=0.05
    )

    return fig


def plot_asset_rolling_vol(
    pnl_with_rp: np.ndarray, pnl_without_rp: np.ndarray,
    d0: int, alpha: float = 2/365,
    title: str = "Asset Rolling Volatility"
):
    """Plot rolling volatility (EMA of |total PnL|) with and without asset RP.

    Shows two curves comparing the effect of asset-level risk parity on volatility.
    """
    n_days = pnl_with_rp.shape[0]
    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    def compute_rolling_vol(pnl_matrix: np.ndarray) -> np.ndarray:
        # Row sums = total PnL per day
        total_pnl = np.nansum(pnl_matrix, axis=1)
        abs_pnl = np.abs(total_pnl)

        # EMA of |total_pnl|
        rolling_vol = np.zeros(n_days, dtype=np.float64)
        rolling_vol[0] = abs_pnl[0] if not np.isnan(abs_pnl[0]) else 0.0

        for i in range(1, n_days):
            val = abs_pnl[i]
            if np.isnan(val):
                rolling_vol[i] = rolling_vol[i-1]
            else:
                rolling_vol[i] = alpha * val + (1 - alpha) * rolling_vol[i-1]

        return rolling_vol

    vol_with_rp = compute_rolling_vol(pnl_with_rp)
    vol_without_rp = compute_rolling_vol(pnl_without_rp)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=vol_with_rp,
        mode='lines',
        name='With Asset RP',
        line=dict(color='#2196F3', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=vol_without_rp,
        mode='lines',
        name='Without Asset RP',
        line=dict(color='#FF9800', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Volatility (EMA of |PnL|)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_asset_diversification(
    pnl_with_rp: np.ndarray, pnl_without_rp: np.ndarray,
    d0: int, alpha: float = 2/365,
    title: str = "Asset Diversification Ratio"
):
    """Plot rolling diversification ratio at asset level, with and without asset RP.

    Diversification ratio = EMA(sum(|asset_pnl|)) / EMA(|sum(asset_pnl)|)

    - Numerator: sum of absolute asset PnLs (treats each asset independently)
    - Denominator: absolute value of total PnL (allows cancellation)

    Ratio is floored at 1. Higher values = more diversification benefit.
    """
    n_days = pnl_with_rp.shape[0]
    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    def compute_div_ratio(pnl_matrix: np.ndarray) -> np.ndarray:
        # Numerator: sum of |asset_pnl| per day
        sum_abs_assets = np.nansum(np.abs(pnl_matrix), axis=1)

        # Denominator: |sum of asset_pnl| per day
        abs_sum_assets = np.abs(np.nansum(pnl_matrix, axis=1))

        # Apply EMA to numerator and denominator separately
        ema_sum_abs = np.zeros(n_days, dtype=np.float64)
        ema_abs_sum = np.zeros(n_days, dtype=np.float64)
        ema_sum_abs[0] = sum_abs_assets[0] if not np.isnan(sum_abs_assets[0]) else 0.0
        ema_abs_sum[0] = abs_sum_assets[0] if not np.isnan(abs_sum_assets[0]) else 0.0

        for i in range(1, n_days):
            val1 = sum_abs_assets[i]
            val2 = abs_sum_assets[i]
            if np.isnan(val1):
                ema_sum_abs[i] = ema_sum_abs[i-1]
            else:
                ema_sum_abs[i] = alpha * val1 + (1 - alpha) * ema_sum_abs[i-1]
            if np.isnan(val2):
                ema_abs_sum[i] = ema_abs_sum[i-1]
            else:
                ema_abs_sum[i] = alpha * val2 + (1 - alpha) * ema_abs_sum[i-1]

        # Ratio of EMAs, floor at 1
        with np.errstate(divide='ignore', invalid='ignore'):
            div_ratio = ema_sum_abs / ema_abs_sum
            div_ratio[~np.isfinite(div_ratio)] = 1.0
            div_ratio = np.maximum(div_ratio, 1.0)

        return div_ratio

    div_with_rp = compute_div_ratio(pnl_with_rp)
    div_without_rp = compute_div_ratio(pnl_without_rp)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=div_with_rp,
        mode='lines',
        name='With Asset RP',
        line=dict(color='#2196F3', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=div_without_rp,
        mode='lines',
        name='Without Asset RP',
        line=dict(color='#FF9800', width=2)
    ))

    # Add floor line at 1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Diversification Ratio (≥1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


def plot_pnl_histogram(data: np.ndarray, nbins: int = 100):
    """Create a PnL histogram with winsorization and signed log1p transform.

    1. Winsorize to 1%/99% percentiles
    2. Transform: sign(x) * log(|x| + 1)
    3. Filters out zeros and reports percentage of zeros in title.
    """
    # Flatten and remove NaN values
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]

    # Count zeros
    n_total = len(valid_data)
    n_zeros = np.sum(valid_data == 0)
    pct_zeros = 100.0 * n_zeros / n_total if n_total > 0 else 0.0

    # Filter out zeros
    nonzero_data = valid_data[valid_data != 0]

    # Winsorize to 1%/99% percentiles
    p1, p99 = np.percentile(nonzero_data, [1, 99])
    winsorized = np.clip(nonzero_data, p1, p99)

    # Apply signed log1p transform: sign(x) * log(|x| + 1)
    transformed = np.sign(winsorized) * np.log1p(np.abs(winsorized))

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=transformed,
        nbinsx=nbins,
        marker_color='#2196F3',
        opacity=0.75
    ))

    fig.update_layout(
        title=f"Straddle P&L Histogram - winsorized 1%/99%, signed log1p ({pct_zeros:.1f}% zeros excluded)",
        xaxis_title="sign(x) * log(|x| + 1)",
        yaxis_title="Count",
        margin=dict(l=50, r=20, t=60, b=50),
        bargap=0.05
    )

    return fig


def plot_live_asset_counts(
    data_matrix: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int,
    title: str = "Live Asset Counts by Group"
):
    """Plot the count of non-NaN (live) assets per group over time."""
    n_days, n_assets = data_matrix.shape
    n_groups = len(unique_groups)

    # Count live assets per group per day
    live_counts = np.zeros((n_days, n_groups), dtype=np.int32)
    for g in range(n_groups):
        mask = group_ids == g
        group_data = data_matrix[:, mask]
        # Count non-NaN values per day
        live_counts[:, g] = np.sum(~np.isnan(group_data), axis=1)

    dates = pd.date_range(start=pd.Timestamp('1970-01-01') + pd.Timedelta(days=d0),
                          periods=n_days, freq='D')

    fig = go.Figure()

    # Color palette for classes
    colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800',
              '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#FFC107']

    for g, group_name in enumerate(unique_groups):
        fig.add_trace(go.Scatter(
            x=dates, y=live_counts[:, g],
            mode='lines',
            name=group_name,
            line=dict(color=colors[g % len(colors)], width=1.5)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Live Asset Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode='x unified'
    )

    return fig


# ============================================================================
# Main App
# ============================================================================
def main():
    st.title("Strategy Explorer")

    # Load data (track time for workflow display)
    with st.spinner("Loading data..."):
        ctx, d0, timings = load_all_data()
        # Store load timings in session state (only on first load / cache miss)
        if "t_file_load_ms" not in st.session_state:
            st.session_state.t_file_load_ms = timings["t_file_load"]
            st.session_state.t_preprocess_ms = timings["t_preprocess"]
        t_file_load_ms = st.session_state.t_file_load_ms
        t_preprocess_ms = st.session_state.t_preprocess_ms

    # Build caches (session state to persist across reruns)
    if "pnl_cache" not in st.session_state:
        st.session_state.pnl_cache = SignalCache(ctx.raw_pnl_matrix, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)
    if "npnl_cache" not in st.session_state:
        st.session_state.npnl_cache = SignalCache(ctx.npnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)
    if "rnpnl_cache" not in st.session_state:
        st.session_state.rnpnl_cache = SignalCache(ctx.rnpnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)
    if "wpnl_cache" not in st.session_state:
        st.session_state.wpnl_cache = SignalCache(ctx.wpnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)

    caches = {
        "pnl": st.session_state.pnl_cache,
        "npnl": st.session_state.npnl_cache,
        "rnpnl": st.session_state.rnpnl_cache,
        "wpnl": st.session_state.wpnl_cache,
    }

    # Layout: sidebar for controls, main area for chart and stats
    with st.sidebar:
        st.header("Strategy Construction")

        # Asset class filter (empty = all)
        asset_filter = st.multiselect(
            "Asset Classes",
            options=ctx.unique_groups,
            default=[],
            format_func=lambda x: x.capitalize(),
            help="Filter to selected groups (empty = all)"
        )

        # Input type
        INPUT_LABELS = {
            "pnl": "PNL (raw)",
            "npnl": "NPNL (vol-normalized)",
            "rnpnl": "RNPNL (market-residual)",
            "wpnl": "WPNL (winsorized 1%-99%)",
        }
        input_type = st.selectbox(
            "Signal Input",
            options=["pnl", "npnl", "rnpnl", "wpnl"],
            format_func=lambda x: INPUT_LABELS[x],
            help="PNL = raw, NPNL = PnL / Vol, RNPNL = NPNL - Market, WPNL = rolling winsorized NPNL"
        )

        # Signals (multi-select)
        all_signals = list(SIGNAL_SPECS.keys()) + ["mac.combo"]
        selected_signals = st.multiselect(
            "Signals",
            options=all_signals,
            default=["clm.365"],
            help="Select one or more signals to combine"
        )

        if not selected_signals:
            selected_signals = ["clm.365"]

        # Ranking mode
        rank_mode = st.selectbox(
            "Signal Combination",
            options=["all", "grp"],
            format_func=lambda x: "rank(all) → scale → sum → rank → scale" if x == "all" else "rank(grp) → scale → sum → rank → scale",
            help="Cross-sectional ranking across all assets or within each group, then sum signals, re-rank, and scale"
        )

        # Asset Risk Parity (1/vol per asset)
        ASSET_RP_LABELS = {
            "none": "Equal (1/N)",
            "inverse_ivol": "Inverse IVol (1/vol)",
            "rolling_abs": "RollingAbs (365d EMA)",
            "rolling_stdev": "Rolling StDev (365d EMA)",
            "rolling_mad": "Rolling MAD (365d)",
        }
        asset_rp = st.selectbox(
            "Asset Risk Parity",
            options=["none", "inverse_ivol", "rolling_abs", "rolling_stdev", "rolling_mad"],
            format_func=lambda x: ASSET_RP_LABELS[x],
            help="Apply inverse-vol weighting per asset: risk_weighted = normabs(signal × (1/vol))"
        )

        # Risk Parity Hedge (first hedge - after asset RP)
        rp_hedge = st.selectbox(
            "Risk Parity Hedge",
            options=list(HEDGE_WEIGHT_FUNCTIONS.keys()),
            format_func=lambda x: HEDGE_LABELS[x],
            index=3,  # Default to 'gah'
            help="First hedge: applied to risk_weighted_signal before class RP"
        )

        # Class Risk Parity
        CLASS_RP_LABELS = {
            "none": "Equal (1/N)",
            "ivol": "Inverse IVol (weighted avg)",
            "rolling_abs": "RollingAbs (365d EMA)",
            "rolling_stdev": "Rolling StDev (365d EMA)",
            "rolling_mad": "Rolling MAD (365d)",
        }
        class_rp = st.selectbox(
            "Class Risk Parity",
            options=["none", "ivol", "rolling_abs", "rolling_stdev", "rolling_mad"],
            format_func=lambda x: CLASS_RP_LABELS[x],
            help="Apply inverse-vol weighting by asset class"
        )

        # Final Hedge (second hedge - after class RP)
        final_hedge = st.selectbox(
            "Final Hedge",
            options=list(HEDGE_WEIGHT_FUNCTIONS.keys()),
            format_func=lambda x: HEDGE_LABELS[x],
            index=3,  # Default to 'gah'
            help="Second hedge: applied to asset_class_weights after class RP"
        )

        st.divider()

    # Strategy name display (under main title)
    signal_str = "+".join(selected_signals)
    asset_suffix = f"@{'+'.join(asset_filter)}" if asset_filter else ""
    asset_rp_suffix = f"-arp" if asset_rp != "none" else ""
    class_rp_suffix = f"-crp" if class_rp != "none" else ""
    hedge_str = f"{rp_hedge}-{final_hedge}"
    strategy_name = f"{input_type}-{signal_str}-{rank_mode}-{hedge_str}{asset_rp_suffix}{class_rp_suffix}{asset_suffix}"
    st.caption(f"**Strategy:** `{strategy_name}`")

    # Main content area - tabs
    tab_stats, tab_workflow, tab_calc = st.tabs(["Statistics", "Workflow", "Calculation"])

    # Run strategy with filter (use lock to prevent Numba threading issues in Streamlit)
    with _numba_lock:
        pnl, long_pnl, short_pnl, stats, workflow, total_matrix, long_matrix, short_matrix, class_rp_pnl_matrix, final_pnl_matrix, rp_hedge_weights, signal_weights, equal_weighted_pnl, asset_rp_pnl_matrix = run_strategy(
            input_type, selected_signals, rank_mode, rp_hedge, final_hedge,
            ctx, caches, asset_filter=asset_filter, asset_rp=asset_rp, class_rp=class_rp,
            t_file_load_ms=t_file_load_ms, t_preprocess_ms=t_preprocess_ms
        )

    # Statistics tab
    with tab_stats:
        col_stats, col_chart = st.columns([1, 3])

        with col_stats:
            # Display filter: select which asset classes to show results for
            # Default mirrors the Asset Classes filter from sidebar
            display_filter = st.multiselect(
                "Show Results For",
                options=ctx.unique_groups,
                default=asset_filter if asset_filter else [],
                format_func=lambda x: x.capitalize(),
                help="Filter displayed results (empty = all)"
            )

            # Apply display filter to matrices
            # Note: In the weight-centric pipeline, hedge/RP weights are already baked in.
            # Display filter just shows a subset of assets from the final wpnl matrix.
            if display_filter:
                orig_group_ids = [ctx.unique_groups.index(g) for g in display_filter]
                display_mask = np.isin(ctx.group_ids, orig_group_ids)
                display_total = total_matrix[:, display_mask]
                display_long = long_matrix[:, display_mask]
                display_short = short_matrix[:, display_mask]
                display_equal_weighted_pnl = equal_weighted_pnl[:, display_mask]
                display_asset_rp_pnl = asset_rp_pnl_matrix[:, display_mask]
                display_signal_weights = signal_weights[:, display_mask]
                display_rp_hedge_weights = rp_hedge_weights[:, display_mask]
                display_class_rp_pnl = class_rp_pnl_matrix[:, display_mask]
                display_final_pnl = final_pnl_matrix[:, display_mask]

                # Create remapped group_ids for filtered assets (0-indexed within filtered set)
                # e.g., if filtering to ["equity", "fx"] (orig ids 2, 3), remap to [0, 1]
                orig_to_new = {orig: new for new, orig in enumerate(orig_group_ids)}
                filtered_orig_ids = ctx.group_ids[display_mask]
                display_asset_group_ids = np.array([orig_to_new[g] for g in filtered_orig_ids], dtype=np.int32)
                display_unique_groups = display_filter

                # Aggregate filtered matrices
                display_pnl = np.nansum(display_total, axis=1)
                display_long_pnl = np.nansum(display_long, axis=1)
                display_short_pnl = np.nansum(display_short, axis=1)
            else:
                # No display filter - use the already-computed values
                display_pnl = pnl
                display_long_pnl = long_pnl
                display_short_pnl = short_pnl
                display_total = total_matrix
                display_equal_weighted_pnl = equal_weighted_pnl
                display_asset_rp_pnl = asset_rp_pnl_matrix
                display_signal_weights = signal_weights
                display_rp_hedge_weights = rp_hedge_weights
                display_class_rp_pnl = class_rp_pnl_matrix
                display_final_pnl = final_pnl_matrix
                display_asset_group_ids = ctx.group_ids
                display_unique_groups = ctx.unique_groups

            # Compute stats for displayed subset
            display_stats = compute_stats(display_pnl, display_long_pnl, display_short_pnl)

            st.divider()
            st.metric("Sharpe Ratio", f"{display_stats.sharpe:.4f}")
            st.metric("Max Drawdown", f"{display_stats.max_drawdown:.4f}")
            st.metric("Long Sharpe", f"{display_stats.long_sharpe:.4f}")
            st.metric("Short Sharpe", f"{display_stats.short_sharpe:.4f}")
            st.metric("L/S Correlation", f"{display_stats.ls_correl:.4f}")

            st.divider()
            st.caption(f"Computed in {workflow.t_total:.1f} ms")

        with col_chart:
            # Use radio instead of tabs - radio persists selection via key
            chart_view = st.radio(
                "Chart",
                options=["PnL Hist", "Vol Hist", "Signal Hist", "Live Counts", "AssetRollVol", "AssetRollDiversify", "RPBeta", "ClassVolsBefore", "ClassVolsAfter", "ClassRollDiversify", "StratRollVol", "StratRollCorrel", "Cross-Sectional Range", "Cross-Sectional Hit Ratio", "Cross-Sectional Win/Loss", "Cumulative PnL", "CumClass P&L"],
                horizontal=True,
                key="chart_view",
                label_visibility="collapsed"
            )

            if chart_view == "PnL Hist":
                fig_pnl_hist = plot_pnl_histogram(ctx.npnl)
                st.plotly_chart(fig_pnl_hist, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: Raw straddle PnL from parquet file
# ═══════════════════════════════════════════════════════════════════════════════
# ctx.npnl is loaded directly from the parquet file during preprocessing:
#   ctx.npnl = data['npnl'].to_numpy()  # normalized PnL per straddle-day
#
# This is the RAW straddle PnL before any weighting or aggregation.
# Shape: (n_straddle_days,) - one value per straddle-day observation

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_pnl_histogram(ctx.npnl)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pnl_histogram(data: np.ndarray, nbins: int = 100):
    """Create a PnL histogram with winsorization and signed log1p transform."""

    # Step 1: Flatten and remove NaN values
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]

    # Step 2: Count and report zeros (shown in title)
    n_total = len(valid_data)
    n_zeros = np.sum(valid_data == 0)
    pct_zeros = 100.0 * n_zeros / n_total

    # Step 3: Filter out zeros for histogram
    nonzero_data = valid_data[valid_data != 0]

    # Step 4: Winsorize to 1%/99% percentiles (reduce outlier impact)
    p1, p99 = np.percentile(nonzero_data, [1, 99])
    winsorized = np.clip(nonzero_data, p1, p99)

    # Step 5: Apply signed log1p transform for better visualization
    # sign(x) * log(|x| + 1) - preserves sign, compresses magnitude
    transformed = np.sign(winsorized) * np.log1p(np.abs(winsorized))

    # Step 6: Plot histogram of transformed values
    fig.add_trace(go.Histogram(x=transformed, nbinsx=nbins))

# The histogram shows the DISTRIBUTION of individual straddle-day PnLs,
# revealing the typical range and shape of returns before any aggregation.
''', language='python')
            elif chart_view == "Vol Hist":
                # Apply log transform to vol (filter out zeros/negatives)
                vol_flat = ctx.vol_matrix.flatten()
                vol_valid = vol_flat[(~np.isnan(vol_flat)) & (vol_flat > 0)]
                log_vol = np.log(vol_valid)
                vol_min, vol_max = np.min(vol_valid), np.max(vol_valid)
                fig_vol_hist = plot_histogram(log_vol.reshape(-1, 1), f"Straddle Vol Histogram (log scale, min={vol_min:.4f}, max={vol_max:.4f})", "log(Volatility)")
                st.plotly_chart(fig_vol_hist, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: ctx.vol_matrix - implied volatility per asset per day
# ═══════════════════════════════════════════════════════════════════════════════
# ctx.vol_matrix is computed during preprocessing from raw straddle data.
# It aggregates straddle-level volatility to a [n_days x n_assets] grid:
#
#   vol_matrix = np.full((grid_size, n_assets), np.nan)
#   # For each straddle, its vol is placed at the corresponding (day, asset) cell
#
# Shape: (n_days, n_assets) - one ivol value per asset per day

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING: Log transform for visualization
# ═══════════════════════════════════════════════════════════════════════════════
vol_flat = ctx.vol_matrix.flatten()                          # Flatten to 1D
vol_valid = vol_flat[(~np.isnan(vol_flat)) & (vol_flat > 0)] # Remove NaN/non-positive
log_vol = np.log(vol_valid)                                  # Log transform

# Log transform is used because:
#   - Volatility is strictly positive and right-skewed
#   - Log scale reveals the full distribution better than linear
#   - Title shows min/max in ORIGINAL (non-log) scale for context

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_histogram(log_vol, ...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_histogram(data: np.ndarray, title: str, xlabel: str, nbins: int = 100):
    """Create a simple histogram using Plotly."""
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]
    fig.add_trace(go.Histogram(x=valid_data, nbinsx=nbins))

# The histogram shows the DISTRIBUTION of implied volatilities across all
# asset-days, revealing typical ivol levels and any regime differences.
''', language='python')
            elif chart_view == "Signal Hist":
                # Filter out zeros and NaNs
                sig_flat = signal_weights.flatten()
                sig_valid = sig_flat[(~np.isnan(sig_flat)) & (sig_flat != 0)]
                # Compute min/max BEFORE winsorization
                raw_min, raw_max = np.min(sig_valid), np.max(sig_valid)
                # Winsorize to 1%/99% percentiles
                p1, p99 = np.percentile(sig_valid, [1, 99])
                sig_winsorized = np.clip(sig_valid, p1, p99)
                fig_sig_hist = plot_histogram(sig_winsorized.reshape(-1, 1), f"Signal Histogram - winsorized 1%/99% (raw min={raw_min:.4f}, raw max={raw_max:.4f})", "Signal Value")
                st.plotly_chart(fig_sig_hist, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: signal_weights - the raw signal before any weighting stages
# ═══════════════════════════════════════════════════════════════════════════════
# signal_weights comes from Stage 1 of the pipeline:
signal_weights = get_combined_signal(signal_names, cache, rank_mode, asset_mask)

# get_combined_signal does:
#   1. Load each selected signal from cache (precomputed [n_days x n_assets])
#   2. Combine signals: mean or product depending on mode
#   3. Apply ranking if rank_mode != "none":
#      - "cross_sectional": rank across assets per day -> uniform(-1, 1)
#      - "time_series": rank each asset over time -> uniform(-1, 1)
#   4. Apply asset_mask: set filtered-out assets to NaN
#
# Shape: (n_days, n_assets) - one signal value per asset per day

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING: Filter and winsorize for visualization
# ═══════════════════════════════════════════════════════════════════════════════
sig_flat = signal_weights.flatten()                          # Flatten to 1D
sig_valid = sig_flat[(~np.isnan(sig_flat)) & (sig_flat != 0)] # Remove NaN/zeros

# Compute raw min/max BEFORE winsorization (shown in title for context)
raw_min, raw_max = np.min(sig_valid), np.max(sig_valid)

# Winsorize to 1%/99% percentiles (clips extreme outliers)
p1, p99 = np.percentile(sig_valid, [1, 99])
sig_winsorized = np.clip(sig_valid, p1, p99)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
fig_sig_hist = plot_histogram(sig_winsorized, title, "Signal Value")

# The histogram shows the DISTRIBUTION of signal weights across all asset-days.
# For ranked signals, expect roughly uniform distribution.
# For raw signals, distribution shape reveals signal characteristics.
''', language='python')
            elif chart_view == "Live Counts":
                fig_live = plot_live_asset_counts(display_signal_weights, display_asset_group_ids, display_unique_groups, d0)
                st.plotly_chart(fig_live, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: display_signal_weights - filtered signal matrix
# ═══════════════════════════════════════════════════════════════════════════════
# display_signal_weights is signal_weights filtered by "Show Results for":
#   - If display_filter is set: display_signal_weights = signal_weights[:, display_mask]
#   - Otherwise: display_signal_weights = signal_weights (full matrix)
#
# The signal_weights matrix has NaN for days where an asset has no live straddles.
# We count non-NaN values to get "live" counts per asset class.

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_live_asset_counts(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_live_asset_counts(
    data_matrix: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int
):
    """Plot the count of non-NaN (live) assets per group over time."""
    n_days, n_assets = data_matrix.shape
    n_groups = len(unique_groups)

    # Count live assets per group per day
    live_counts = np.zeros((n_days, n_groups), dtype=np.int32)
    for g in range(n_groups):
        mask = group_ids == g              # Assets in this class
        group_data = data_matrix[:, mask]  # Slice columns for this class
        # Count non-NaN values per day (non-NaN = has live straddles)
        live_counts[:, g] = np.sum(~np.isnan(group_data), axis=1)

    # Plot one line per asset class
    for g, group_name in enumerate(unique_groups):
        fig.add_trace(go.Scatter(x=dates, y=live_counts[:, g], name=group_name))

# The plot shows how many assets are "live" (have straddles) in each class
# over time. Useful for understanding data coverage and survivorship.
''', language='python')
            elif chart_view == "AssetRollVol":
                fig_asset_vol = plot_asset_rolling_vol(
                    display_asset_rp_pnl, display_equal_weighted_pnl, d0,
                    title=f"Asset Rolling Volatility (Asset RP: {asset_rp})"
                )
                st.plotly_chart(fig_asset_vol, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: Two PnL matrices for comparison
# ═══════════════════════════════════════════════════════════════════════════════
# display_equal_weighted_pnl: PnL with equal weights (1/N per live asset)
#   equal_weighted_pnl = compute_weighted_pnl_matrix(sumabs_norm(ones), ctx)
#   This is the "baseline" - no risk parity, just equal allocation
#
# display_asset_rp_pnl: PnL with asset RP weights applied
#   asset_rp_pnl_matrix = compute_weighted_pnl_matrix(asset_rp_weights, ctx)
#   Where asset_rp_weights = 1/vol for inverse_ivol, or rolling measures
#
# Both use INCEPTION WEIGHTING: weight locked at straddle start

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_asset_rolling_vol(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_asset_rolling_vol(
    pnl_with_rp: np.ndarray, pnl_without_rp: np.ndarray,
    d0: int, alpha: float = 2/365
):
    """Plot rolling volatility with and without asset RP."""

    def compute_rolling_vol(pnl_matrix: np.ndarray) -> np.ndarray:
        # Row sums = total PnL per day across all assets
        total_pnl = np.nansum(pnl_matrix, axis=1)
        abs_pnl = np.abs(total_pnl)

        # EMA of |total_pnl| = rolling volatility estimate
        rolling_vol = np.zeros(n_days, dtype=np.float64)
        rolling_vol[0] = abs_pnl[0]
        for i in range(1, n_days):
            rolling_vol[i] = alpha * abs_pnl[i] + (1 - alpha) * rolling_vol[i-1]
        return rolling_vol

    vol_with_rp = compute_rolling_vol(pnl_with_rp)      # "With Asset RP"
    vol_without_rp = compute_rolling_vol(pnl_without_rp) # "Equal Weight (1/N)"

    # Plot both lines for comparison
    fig.add_trace(go.Scatter(y=vol_with_rp, name="With Asset RP"))
    fig.add_trace(go.Scatter(y=vol_without_rp, name="Equal Weight (1/N)"))

# If asset RP is effective, the "With Asset RP" line should be LOWER,
# indicating reduced portfolio volatility due to inverse-vol weighting.
''', language='python')
            elif chart_view == "AssetRollDiversify":
                fig_asset_div = plot_asset_diversification(
                    display_asset_rp_pnl, display_equal_weighted_pnl, d0,
                    title=f"Asset Diversification Ratio (Asset RP: {asset_rp})"
                )
                st.plotly_chart(fig_asset_div, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: Same two PnL matrices as AssetRollVol
# ═══════════════════════════════════════════════════════════════════════════════
# display_equal_weighted_pnl: PnL with equal weights (1/N per live asset)
# display_asset_rp_pnl: PnL with asset RP weights applied
# Both use INCEPTION WEIGHTING: weight locked at straddle start

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_asset_diversification(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_asset_diversification(
    pnl_with_rp: np.ndarray, pnl_without_rp: np.ndarray,
    d0: int, alpha: float = 2/365
):
    """Plot rolling diversification ratio at asset level."""

    # Diversification ratio = EMA(sum(|asset_pnl|)) / EMA(|sum(asset_pnl)|)
    #
    # Numerator: sum of absolute asset PnLs (treats each asset independently)
    #   - No cancellation: +5 and -5 = 10
    # Denominator: absolute value of total PnL (allows cancellation)
    #   - With cancellation: +5 and -5 = 0
    #
    # Ratio is floored at 1. Higher values = more diversification benefit.

    def compute_div_ratio(pnl_matrix: np.ndarray) -> np.ndarray:
        # Numerator: sum of |asset_pnl| per day
        sum_abs_assets = np.nansum(np.abs(pnl_matrix), axis=1)

        # Denominator: |sum of asset_pnl| per day
        abs_sum_assets = np.abs(np.nansum(pnl_matrix, axis=1))

        # Apply EMA to numerator and denominator separately
        ema_sum_abs = ema(sum_abs_assets, alpha)
        ema_abs_sum = ema(abs_sum_assets, alpha)

        # Diversification ratio = EMA(numerator) / EMA(denominator)
        div_ratio = ema_sum_abs / ema_abs_sum
        return np.maximum(div_ratio, 1.0)  # Floor at 1

    div_with_rp = compute_div_ratio(pnl_with_rp)
    div_without_rp = compute_div_ratio(pnl_without_rp)

    # Plot both lines
    fig.add_trace(go.Scatter(y=div_with_rp, name="With Asset RP"))
    fig.add_trace(go.Scatter(y=div_without_rp, name="Equal Weight (1/N)"))

# A ratio of 2.0 means: the sum of individual |PnLs| is 2x the |total PnL|,
# indicating 50% of moves cancel out. Higher = more diversification.
''', language='python')
            elif chart_view == "Cumulative PnL":
                fig = plot_cumulative_pnl(display_pnl, display_long_pnl, display_short_pnl, d0)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: Final strategy PnL (after ALL weighting stages)
# ═══════════════════════════════════════════════════════════════════════════════
# display_pnl, display_long_pnl, display_short_pnl are the FINAL outputs:
#
# The full pipeline is:
#   Stage 1: signal_weights = get_combined_signal(...)
#   Stage 2: asset_rp_weights = compute_asset_rp_weights(...)
#   Stage 3: rp_hedge_weights = HEDGE_FN[rp_hedge](long_pnl, short_pnl, ...)
#   Stage 4: asset_weights = combine_weights(signal, asset_rp, rp_hedge)
#   Stage 5-7: class_rp_weights = compute_class_rp_weights(...)
#   Stage 8: asset_class_weights = combine_weights(asset_weights, class_rp)
#   Stage 9-11: final_hedge_weights = HEDGE_FN[final_hedge](...)
#   Stage 12: final_weights = asset_class_weights * final_hedge_weights
#   Stage 13: wpnl_matrix = compute_weighted_pnl_matrix(final_weights, ctx)
#             ⭐ Uses INCEPTION WEIGHTING throughout
#
# Finally:
#   long_matrix = wpnl_matrix where final_weights > 0
#   short_matrix = wpnl_matrix where final_weights < 0
#   pnl = sum(long_pnl) + sum(short_pnl) per day

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_cumulative_pnl(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cumulative_pnl(pnl, long_pnl, short_pnl, d0):
    """Create cumulative PnL chart with Plotly."""

    # Simple cumulative sum of daily PnL
    cum_pnl = np.nancumsum(pnl)        # Total (blue)
    cum_long = np.nancumsum(long_pnl)  # Longs only (green)
    cum_short = np.nancumsum(short_pnl) # Shorts only (red)

    # Plot all three lines
    fig.add_trace(go.Scatter(y=cum_pnl, name='Total'))
    fig.add_trace(go.Scatter(y=cum_long, name='Long'))
    fig.add_trace(go.Scatter(y=cum_short, name='Short'))

# Shows the cumulative return of the full strategy, plus breakdown
# into long and short components. Useful for understanding where
# returns come from and whether long/short are complementary.
''', language='python')
            elif chart_view == "RPBeta":
                fig_beta = plot_rp_beta_by_class(display_rp_hedge_weights, display_asset_group_ids, display_unique_groups, d0)
                st.plotly_chart(fig_beta, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: rp_hedge_weights - hedge betas from Stage 3
# ═══════════════════════════════════════════════════════════════════════════════
# rp_hedge_weights comes from the RP hedge computation:
rp_hedge_weights = HEDGE_FN[rp_hedge](
    signal_weights=signal_weights,
    long_pnl_matrix=rp_long_pnl,
    short_pnl_matrix=rp_short_pnl,
    group_ids=group_ids, n_groups=n_groups
)

# For "hg" (hedge) or "gah" (group-aware hedge):
#   beta = -EMA(long_pnl * short_pnl) / EMA(long_pnl * long_pnl)
#   This is the regression coefficient of long vs short PnL
#
# The rp_hedge_weights matrix contains:
#   - 1.0 for SHORT positions (shorts are unhedged)
#   - beta for LONG positions (longs scaled to hedge shorts)
#   - 0.0 for INACTIVE positions

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_rp_beta_by_class(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_rp_beta_by_class(
    rp_hedge_weights: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int
):
    """Plot RP hedge beta values averaged by asset class."""

    # For each class, for each day, compute average beta:
    class_beta = np.zeros((n_days, n_groups))
    for g in range(n_groups):
        mask = group_ids == g
        class_weights = rp_hedge_weights[:, mask]

        for d in range(n_days):
            row = class_weights[d, :]
            # Get beta values: positive weights that aren't 1.0 (shorts)
            beta_vals = row[(row > 0) & (row != 1.0)]
            if len(beta_vals) > 0:
                class_beta[d, g] = np.mean(beta_vals)

    # Plot one line per asset class
    for g, group_name in enumerate(unique_groups):
        fig.add_trace(go.Scatter(y=class_beta[:, g], name=group_name))

# Shows how hedge ratios vary across asset classes over time.
# Beta < 1 means longs are scaled down relative to shorts.
# Higher beta = more long exposure relative to short exposure.
''', language='python')
            elif chart_view == "CumClass P&L":
                fig_class = plot_cumulative_pnl_by_class(display_total, display_asset_group_ids, display_unique_groups, d0)
                st.plotly_chart(fig_class, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE DATA: display_total - final PnL matrix per asset
# ═══════════════════════════════════════════════════════════════════════════════
# display_total = total_matrix[:, display_mask]  (filtered by "Show Results for")
#
# total_matrix is the FINAL weighted PnL matrix:
#   wpnl_matrix = compute_weighted_pnl_matrix(final_weights, ctx)
#   total_matrix = long_matrix + short_matrix
#
# Shape: (n_days, n_filtered_assets) - one PnL value per asset per day
# Uses INCEPTION WEIGHTING: weight locked at straddle start, held constant

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATION: plot_cumulative_pnl_by_class(...)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cumulative_pnl_by_class(
    total_matrix: np.ndarray, group_ids: np.ndarray,
    unique_groups: list[str], d0: int
):
    """Create cumulative PnL chart by asset class."""
    n_groups = len(unique_groups)

    for g in range(n_groups):
        # Sum PnL for assets in this group (columns where group_ids == g)
        mask = group_ids == g
        class_pnl = np.nansum(total_matrix[:, mask], axis=1)  # Sum across assets

        # Cumulative sum over time
        cum_class_pnl = np.nancumsum(class_pnl)

        # Plot this class's cumulative PnL
        fig.add_trace(go.Scatter(
            x=dates, y=cum_class_pnl,
            name=unique_groups[g]
        ))

# Shows cumulative returns BROKEN DOWN by asset class.
# Useful for identifying which classes contribute most to returns,
# and whether certain classes are consistently profitable or not.
# Note: display_asset_group_ids is remapped to 0-indexed within filtered set.
''', language='python')
            elif chart_view == "ClassVolsBefore":
                title = "Class Volatility (before Class RP weights)"
                fig_crp = plot_class_realized_vol(display_class_rp_pnl, display_asset_group_ids, display_unique_groups, d0, title=title)
                st.plotly_chart(fig_crp, use_container_width=True, config={"displaylogo": False})
            elif chart_view == "ClassVolsAfter":
                title = f"Class Volatility (after Class RP: {class_rp})"
                fig_cvol = plot_class_realized_vol(display_final_pnl, display_asset_group_ids, display_unique_groups, d0, title=title)
                st.plotly_chart(fig_cvol, use_container_width=True, config={"displaylogo": False})

                # Code path documentation
                with st.expander("📝 Code Path: How This Plot Is Generated", expanded=False):
                    st.code('''
# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Compute Class RP weights (e.g., rolling_abs, rolling_stdev, ivol, etc.)
# ═══════════════════════════════════════════════════════════════════════════════
# For "rolling_abs" method:
class_rp_weights, class_vol = compute_class_rp_weights_rolling_abs(
    class_rp_pnl_matrix,  # PnL matrix BEFORE class RP (from asset RP stage)
    group_ids,            # Maps each asset to its class index (0-based)
    n_groups,             # Number of asset classes
    alpha=2/366           # EMA decay for ~365 day half-life
)

# Inside compute_class_rp_weights_rolling_abs:
#   1. Aggregate PnL by class: class_pnl[t, g] = sum(pnl[t, assets_in_class_g])
#   2. Compute rolling abs: class_vol[t, g] = EMA(|class_pnl|, alpha)
#   3. Risk weights = 1 / class_vol (inverse volatility)
#   4. Normalize: class_weights = sumabs_norm(risk_weights)  # sum(|w|) = 1 per row
#   5. Expand to assets: class_rp_weights[t, a] = class_weights[t, group_ids[a]]
#   6. LAG BY 1 DAY: class_rp_weights[t] uses vol estimated at t-1 (no lookahead)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Combine asset weights with class RP weights
# ═══════════════════════════════════════════════════════════════════════════════
asset_class_weights = combine_weights(asset_weights, class_rp_weights)

# combine_weights does: sumabs_norm(sumabs_norm(a) * sumabs_norm(b))
# This ensures the combined weights sum to 1 in absolute value

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Compute final PnL using INCEPTION WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════════
final_long_pnl, final_short_pnl, _, _ = compute_preliminary_pnl_matrices(
    asset_class_weights,  # Weights with class RP baked in
    ctx                   # Contains straddle data: pnl, starts, weights, etc.
)
final_pnl_matrix = final_long_pnl + final_short_pnl

# Inside compute_preliminary_pnl_matrices:
#   long_weights = np.maximum(asset_class_weights, 0.0)
#   long_pnl = _aggregate_inception_weighted_parallel(..., long_weights, ...)
#   short_weights = np.minimum(asset_class_weights, 0.0)
#   short_pnl = _aggregate_inception_weighted_parallel(..., short_weights, ...)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: The INCEPTION WEIGHTING kernel (this is the key!)
# ═══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _aggregate_inception_weighted_parallel(..., signal_matrix, ...):
    """Weight is LOCKED at straddle inception, held constant throughout life."""
    for a in range(n_assets):           # Parallel over assets
        for straddle in asset_straddles:
            inception_day = straddle.start - d0

            # ⭐ KEY: Weight is read ONCE at inception_day
            sig = signal_matrix[inception_day, a]

            for j in range(straddle.length):
                day_idx = straddle.start + j - d0
                # ⭐ Same 'sig' used for ALL days of this straddle
                pnl_sum[day_idx, a] += straddle.pnl[j] * straddle.weight * sig

# This means: if class RP weight for Equity on Jan 1 is 0.3, and a straddle
# starts on Jan 1, that straddle uses weight 0.3 for its ENTIRE life,
# even if class RP weight changes to 0.5 on Jan 15.

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: Plot realized volatility per class
# ═══════════════════════════════════════════════════════════════════════════════
plot_class_realized_vol(final_pnl_matrix, group_ids, unique_groups, d0)

# Inside plot_class_realized_vol:
#   1. Aggregate by class: class_pnl[t, g] = sum(final_pnl[t, assets_in_class_g])
#   2. Compute rolling vol: class_vol = EMA(|class_pnl|, alpha=2/366)
#   3. Normalize for display: class_vol_norm = class_vol / sum(|class_vol|)
#   4. Plot one line per class

# If class RP is working correctly, all lines should converge toward
# similar heights (equal vol contribution per class).
''', language='python')
            elif chart_view == "ClassRollDiversify":
                fig_div = plot_class_diversification(display_final_pnl, display_asset_group_ids, display_unique_groups, d0)
                st.plotly_chart(fig_div, use_container_width=True, config={"displaylogo": False})
            elif chart_view == "StratRollVol":
                fig_vol = plot_rolling_vol(display_pnl, d0)
                st.plotly_chart(fig_vol, use_container_width=True, config={"displaylogo": False})
            elif chart_view == "StratRollCorrel":
                fig_corr = plot_rolling_correlation(display_long_pnl, display_short_pnl, d0)
                st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})
            elif chart_view == "Cross-Sectional Range":
                fig_range = plot_cross_sectional_range(display_total, d0)
                st.plotly_chart(fig_range, use_container_width=True, config={"displaylogo": False})
            elif chart_view == "Cross-Sectional Hit Ratio":
                fig_hit = plot_cross_sectional_hit_ratio(display_total, d0)
                st.plotly_chart(fig_hit, use_container_width=True, config={"displaylogo": False})
            else:  # Cross-Sectional Win/Loss
                fig_wl = plot_cross_sectional_winloss_ratio(display_total, d0)
                st.plotly_chart(fig_wl, use_container_width=True, config={"displaylogo": False})

    # Workflow tab
    with tab_workflow:
        # Summary info at top
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Days", f"{workflow.n_days:,}")
        with col_info2:
            st.metric("Assets", f"{workflow.n_assets}")
        with col_info3:
            st.metric("Groups", f"{workflow.n_groups}")
        with col_info4:
            st.metric("Total Time", f"{workflow.t_total:.1f} ms")

        st.divider()

        # Build descriptions for mermaid
        d, a = workflow.n_days, workflow.n_assets
        g = workflow.n_groups

        n_signals = len(selected_signals)
        if n_signals == 1:
            sig_desc = f"{selected_signals[0]}->rank({rank_mode})->scale->lag"
        else:
            sig_desc = f"{n_signals} signals->rank->scale->sum->rank->scale"
        # Render pipeline flowchart using graphviz
        graphviz_code = f"""
        digraph pipeline {{
            rankdir=TB
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10]
            edge [fontname="Helvetica", fontsize=9]

            subgraph cluster_0_data {{
                label="Stage 0: Data Loading"
                style=filled
                color="#1a365d"
                fontcolor=white
                A [label="straddles.arrow, valuations.arrow\\n↓\\nctx.pnl_matrix [{d:,} x {a}]\\nctx.returns_matrix [{d:,} x {a}]\\nctx.vol_matrix [{d:,} x {a}]" shape=cylinder fillcolor="#2c5282" fontcolor=white]
            }}

            subgraph cluster_1_signal {{
                label="Stage 1: Signal Generation"
                style=filled
                color="#065f46"
                fontcolor=white
                B [label="signal_weights = get_combined_signal(...)\\n{sig_desc}\\n[{workflow.signal_weights_shape[0]:,} x {workflow.signal_weights_shape[1]}]" fillcolor="#047857" fontcolor=white]
            }}

            subgraph cluster_2_assetrp {{
                label="Stage 2: Asset Risk Parity ({asset_rp})"
                style=filled
                color="#065f46"
                fontcolor=white
                C [label="asset_rp_weights = 1 / ctx.vol_matrix\\nrisk_weighted_signal = normabs(normabs(signal) * normabs(asset_rp))\\n[{workflow.risk_weighted_signal_shape[0]:,} x {workflow.risk_weighted_signal_shape[1]}]" fillcolor="#047857" fontcolor=white]
            }}

            subgraph cluster_3_rphedge {{
                label="Stage 3-5: RP Hedge ({rp_hedge})"
                style=filled
                color="#7c2d12"
                fontcolor=white
                D [label="rp_long_pnl, rp_short_pnl = compute_preliminary_pnl_matrices(risk_weighted_signal)\\nsplit weights into +/- then inception_weighted_pnl\\n[2 x {workflow.rp_hedge_pnl_shape[0]:,} x {workflow.rp_hedge_pnl_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
                E [label="rp_hedge_weights = HEDGE_FN[{rp_hedge}](rp_long_pnl, rp_short_pnl)\\nbeta = -EMA(long*short) / EMA(long*long)  (regression slope)\\nweights: 1.0 for shorts, beta for longs\\n[{workflow.rp_hedge_weights_shape[0]:,} x {workflow.rp_hedge_weights_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
                F [label="asset_weights = risk_weighted_signal * rp_hedge_weights\\nsimple element-wise multiply (hedge scales longs by beta)\\n[{workflow.asset_weights_shape[0]:,} x {workflow.asset_weights_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
            }}

            subgraph cluster_6_classrp {{
                label="Stage 6-8: Class Risk Parity ({class_rp})"
                style=filled
                color="#4c1d95"
                fontcolor=white
                G [label="class_rp_pnl = compute_preliminary_pnl_matrices(asset_weights)\\nlong + short combined for vol estimation\\n[{workflow.class_rp_pnl_shape[0]:,} x {workflow.class_rp_pnl_shape[1]}]" fillcolor="#5b21b6" fontcolor=white]
                H [label="class_rp_weights = compute_class_rp_weights(class_rp_pnl)\\nEMA vol per class -> 1/vol -> expand to assets\\n[{workflow.class_rp_weights_shape[0]:,} x {workflow.class_rp_weights_shape[1]}] ({g} groups)" fillcolor="#5b21b6" fontcolor=white]
                I [label="asset_class_weights = combine_weights(asset_weights, class_rp_weights)\\nnormabs(normabs(asset) * normabs(class_rp))\\n[{workflow.asset_class_weights_shape[0]:,} x {workflow.asset_class_weights_shape[1]}]" fillcolor="#5b21b6" fontcolor=white]
            }}

            subgraph cluster_9_finalhedge {{
                label="Stage 9-11: Final Hedge ({final_hedge})"
                style=filled
                color="#7c2d12"
                fontcolor=white
                J [label="final_long_pnl, final_short_pnl = compute_preliminary_pnl_matrices(asset_class_weights)\\nsplit weights into +/- then inception_weighted_pnl\\n[2 x {workflow.final_hedge_pnl_shape[0]:,} x {workflow.final_hedge_pnl_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
                K [label="final_hedge_weights = HEDGE_FN[{final_hedge}](final_long_pnl, final_short_pnl)\\nbeta = -EMA(long*short) / EMA(long*long)  (regression slope)\\nweights: 1.0 for shorts, beta for longs\\n[{workflow.final_hedge_weights_shape[0]:,} x {workflow.final_hedge_weights_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
                L [label="final_weights = asset_class_weights * final_hedge_weights\\nsimple element-wise multiply (hedge scales longs by beta)\\n[{workflow.final_weights_shape[0]:,} x {workflow.final_weights_shape[1]}]" fillcolor="#9a3412" fontcolor=white]
            }}

            subgraph cluster_12_output {{
                label="Stage 12: Output"
                style=filled
                color="#1e3a5f"
                fontcolor=white
                M [label="wpnl_matrix = compute_weighted_pnl_matrix(final_weights)\\nlong_matrix = compute_weighted_pnl_matrix(max(final_weights, 0))\\nshort_matrix = compute_weighted_pnl_matrix(min(final_weights, 0))\\n[3 x {workflow.wpnl_shape[0]:,} x {workflow.wpnl_shape[1]}]" fillcolor="#2563eb" fontcolor=white]
                N [label="total_pnl = nansum(wpnl_matrix, axis=1)\\nlong_pnl = nansum(long_matrix, axis=1)\\nshort_pnl = nansum(short_matrix, axis=1)\\n[3 x {workflow.wpnl_shape[0]:,}]" fillcolor="#2563eb" fontcolor=white]
                O [label="stats = compute_stats(total, long, short)\\nSharpe, MaxDD, Volatility, L/S Correlation" shape=box3d fillcolor="#2563eb" fontcolor=white]
            }}

            A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K -> L -> M -> N -> O
        }}
        """
        st.graphviz_chart(graphviz_code, use_container_width=True)

        # Timing breakdown table
        st.subheader("⏱️ Timing Breakdown")
        timing_data = {
            "Stage": [
                "File Load", "Preprocessing", "Signal Weights", "Asset RP",
                "RP Hedge PnL", "RP Hedge β", "Asset Weights",
                "Class RP PnL", "Class RP", "Asset Class Weights",
                "Final Hedge PnL", "Final Hedge β", "Final Weights", "Weighted PnL"
            ],
            "Time (ms)": [
                workflow.t_file_load, workflow.t_preprocess, workflow.t_signal, workflow.t_asset_rp,
                workflow.t_rp_hedge_pnl, workflow.t_rp_hedge, workflow.t_asset_weights,
                workflow.t_class_rp_pnl, workflow.t_class_rp, workflow.t_asset_class_weights,
                workflow.t_final_hedge_pnl, workflow.t_final_hedge, workflow.t_final_weights, workflow.t_pnl
            ]
        }
        timing_df = pd.DataFrame(timing_data)
        timing_df["Time (ms)"] = timing_df["Time (ms)"].apply(lambda x: f"{x:.2f}")
        st.dataframe(timing_df, use_container_width=True, hide_index=True)

    # Calculation tab - matrix variables table
    with tab_calc:
        st.subheader("Pipeline Matrix Variables")

        d, a, g = workflow.n_days, workflow.n_assets, workflow.n_groups
        active = workflow.n_active_assets

        # Show filter status
        if active < a:
            st.info(f"🔍 Asset filter active: {active} of {a} assets participating")

        calc_data = {
            "Variable": [
                "ctx.pnl_matrix",
                "ctx.returns_matrix",
                "ctx.vol_matrix",
                "signal_weights",
                "asset_rp_weights",
                "risk_weighted_signal",
                "rp_long_pnl",
                "rp_short_pnl",
                "rp_hedge_weights",
                "asset_weights",
                "class_rp_pnl_matrix",
                "class_rp_weights",
                "asset_class_weights",
                "final_long_pnl",
                "final_short_pnl",
                "final_hedge_weights",
                "final_weights",
                "wpnl_matrix",
                "long_matrix",
                "short_matrix",
                "total_pnl",
                "long_pnl",
                "short_pnl",
            ],
            "Shape": [
                f"{d:,} x {a}",  # ctx.pnl_matrix - full matrix
                f"{d:,} x {a}",  # ctx.returns_matrix - full matrix
                f"{d:,} x {a}",  # ctx.vol_matrix - full matrix
                f"{d:,} x {active}",  # signal_weights - filtered
                f"{d:,} x {active}",  # asset_rp_weights - filtered
                f"{d:,} x {active}",  # risk_weighted_signal - filtered
                f"{d:,} x {active}",  # rp_long_pnl - filtered
                f"{d:,} x {active}",  # rp_short_pnl - filtered
                f"{d:,} x {active}",  # rp_hedge_weights - filtered
                f"{d:,} x {active}",  # asset_weights - filtered
                f"{d:,} x {active}",  # class_rp_pnl_matrix - filtered
                f"{d:,} x {active}",  # class_rp_weights - filtered
                f"{d:,} x {active}",  # asset_class_weights - filtered
                f"{d:,} x {active}",  # final_long_pnl - filtered
                f"{d:,} x {active}",  # final_short_pnl - filtered
                f"{d:,} x {active}",  # final_hedge_weights - filtered
                f"{d:,} x {active}",  # final_weights - filtered
                f"{d:,} x {active}",  # wpnl_matrix - filtered
                f"{d:,} x {active}",  # long_matrix - filtered
                f"{d:,} x {active}",  # short_matrix - filtered
                f"{d:,}",  # total_pnl - 1D
                f"{d:,}",  # long_pnl - 1D
                f"{d:,}",  # short_pnl - 1D
            ],
            "Description": [
                "Raw daily PnL per asset from straddles.arrow",
                "Daily returns per asset from valuations.arrow",
                "Rolling volatility per asset (EMA, alpha=1/365)",
                "get_combined_signal(): rank -> scale -> sum -> rank -> scale",
                "1 / vol_matrix (inverse volatility weighting)",
                "normabs(normabs(signal) * normabs(asset_rp))",
                "inception_weighted_pnl(max(risk_weighted_signal, 0) * pnl)",
                "inception_weighted_pnl(min(risk_weighted_signal, 0) * pnl)",
                "beta for longs, 1.0 for shorts; beta = -EMA(L*S)/EMA(L*L)",
                "risk_weighted_signal * rp_hedge_weights",
                "rp_long_pnl + rp_short_pnl (combined for vol estimation)",
                "1/vol per class (EMA vol, expanded to assets)",
                "normabs(normabs(asset_weights) * normabs(class_rp_weights))",
                "inception_weighted_pnl(max(asset_class_weights, 0) * pnl)",
                "inception_weighted_pnl(min(asset_class_weights, 0) * pnl)",
                "beta for longs, 1.0 for shorts; beta = -EMA(L*S)/EMA(L*L)",
                "asset_class_weights * final_hedge_weights",
                "inception_weighted_pnl(final_weights * pnl)",
                "inception_weighted_pnl(max(final_weights, 0) * pnl)",
                "inception_weighted_pnl(min(final_weights, 0) * pnl)",
                "nansum(wpnl_matrix, axis=1) - total daily PnL",
                "nansum(long_matrix, axis=1) - long-only daily PnL",
                "nansum(short_matrix, axis=1) - short-only daily PnL",
            ],
        }

        calc_df = pd.DataFrame(calc_data)
        st.dataframe(calc_df, use_container_width=True, hide_index=True, height=850)


if __name__ == "__main__":
    main()
