"""Strategy: Modular Backtesting Framework.

Pipeline: Input → Signal → Transform → Hedge → PnL → Stats
"""
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import yaml
from numba import njit, prange


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class PortfolioStats:
    sharpe: float
    long_sharpe: float
    short_sharpe: float
    ls_correl: float


@dataclass
class BacktestContext:
    """Holds all data needed for backtesting."""
    # Preprocessed inputs
    npnl: np.ndarray
    rnpnl: np.ndarray
    wpnl: np.ndarray  # Winsorized npnl (rolling 1%/99% clipping)
    vol_matrix: np.ndarray

    # Group info
    group_ids: np.ndarray
    n_groups: int

    # Straddle data (sorted for parallel processing)
    out0s: np.ndarray
    lens: np.ndarray
    starts: np.ndarray
    asset_ids: np.ndarray
    weights: np.ndarray
    asset_starts: np.ndarray
    asset_counts: np.ndarray

    # Grid parameters
    d0: int
    grid_size: int
    n_assets: int

    # Valuations
    pnl: np.ndarray
    dte: np.ndarray
    have_dte: bool


# ============================================================================
# Numba Kernels (performance critical - keep as-is)
# ============================================================================
@njit(cache=True, parallel=True)
def _aggregate_weighted_daily_by_asset_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets,
    asset_starts, asset_counts
):
    """Parallel aggregation of weighted daily pnl per asset into 2D grid."""
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)

    for a in prange(n_assets):
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


@njit(cache=True, parallel=True)
def _aggregate_inception_weighted_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets,
    ptf_matrix, asset_starts, asset_counts
):
    """Aggregate daily PnL weighted by portfolio signal at straddle inception."""
    result = np.zeros((grid_size, n_assets), np.float64)

    for a in prange(n_assets):
        start_idx = asset_starts[a]
        count = asset_counts[a]

        for i in range(count):
            k = start_idx + i
            o = out0s[k]
            L = lens[k]
            start = starts_epoch[k]
            w = weights[k]

            inception_idx = start - d0

            if inception_idx < 0 or inception_idx >= grid_size:
                ptf_w = 0.0
            else:
                ptf_w = ptf_matrix[inception_idx, a]
                if np.isnan(ptf_w):
                    ptf_w = 0.0

            combined_w = w * ptf_w

            if combined_w == 0.0:
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
                    result[day_idx, a] += p * combined_w

    return result


@njit(cache=True, parallel=True)
def _ema_columns_parallel(data: np.ndarray, alpha: float) -> np.ndarray:
    """Compute EMA along axis 0 (time) for each column in parallel."""
    n_days, n_assets = data.shape
    ema = np.zeros_like(data)
    decay = 1.0 - alpha

    for a in prange(n_assets):
        first_valid = np.nan
        for t in range(n_days):
            if not np.isnan(data[t, a]):
                first_valid = data[t, a]
                break

        prev = first_valid
        for t in range(n_days):
            val = data[t, a]
            if np.isnan(val):
                ema[t, a] = prev
            else:
                if np.isnan(prev):
                    ema[t, a] = val
                    prev = val
                else:
                    ema[t, a] = alpha * val + decay * prev
                    prev = ema[t, a]

    return ema


@njit(cache=True, parallel=True)
def _rolling_max_drawdown_parallel(cumsum: np.ndarray) -> np.ndarray:
    """Compute rolling max drawdown for each column in parallel."""
    n_days, n_assets = cumsum.shape
    max_dd = np.zeros_like(cumsum)

    for a in prange(n_assets):
        running_max = -np.inf
        running_max_dd = 0.0

        for t in range(n_days):
            val = cumsum[t, a]
            if np.isnan(val):
                max_dd[t, a] = running_max_dd if running_max_dd > 0 else np.nan
                continue

            if val > running_max:
                running_max = val

            current_dd = running_max - val
            if current_dd > running_max_dd:
                running_max_dd = current_dd

            max_dd[t, a] = running_max_dd if running_max_dd > 0 else np.nan

    return max_dd


@njit(cache=True, parallel=True)
def _cross_sectional_rank_parallel(data: np.ndarray) -> np.ndarray:
    """Compute cross-sectional rank for each row, normalized to [-1, +1]."""
    n_days, n_assets = data.shape
    ranks = np.full_like(data, np.nan)

    for t in prange(n_days):
        valid_count = 0
        for a in range(n_assets):
            if not np.isnan(data[t, a]):
                valid_count += 1

        if valid_count < 2:
            continue

        for a in range(n_assets):
            if np.isnan(data[t, a]):
                continue

            rank = 0
            val = data[t, a]
            for b in range(n_assets):
                if not np.isnan(data[t, b]) and data[t, b] < val:
                    rank += 1

            ranks[t, a] = 2.0 * rank / (valid_count - 1) - 1.0

    return ranks


@njit(cache=True, parallel=True)
def _cross_sectional_rank_by_group_parallel(
    data: np.ndarray, group_ids: np.ndarray, n_groups: int
) -> np.ndarray:
    """Compute cross-sectional rank within each group, normalized to [-1, +1]."""
    n_days, n_assets = data.shape
    ranks = np.full_like(data, np.nan)

    for t in prange(n_days):
        for g in range(n_groups):
            valid_count = 0
            for a in range(n_assets):
                if group_ids[a] == g and not np.isnan(data[t, a]):
                    valid_count += 1

            if valid_count < 2:
                continue

            for a in range(n_assets):
                if group_ids[a] != g or np.isnan(data[t, a]):
                    continue

                rank = 0
                val = data[t, a]
                for b in range(n_assets):
                    if group_ids[b] == g and not np.isnan(data[t, b]) and data[t, b] < val:
                        rank += 1

                ranks[t, a] = 2.0 * rank / (valid_count - 1) - 1.0

    return ranks


@njit(cache=True)
def _compute_rolling_beta_ema(long_pnl: np.ndarray, short_pnl: np.ndarray, alpha: float) -> np.ndarray:
    """Compute rolling beta using EMA (lagged by 1 day)."""
    n = len(long_pnl)
    beta = np.zeros(n, dtype=np.float64)

    ema_long = 0.0
    ema_short = 0.0
    ema_long_sq = 0.0
    ema_long_short = 0.0
    initialized = False

    for t in range(n):
        if initialized:
            var_long = ema_long_sq - ema_long * ema_long
            cov_ls = ema_long_short - ema_long * ema_short
            if var_long > 1e-12:
                beta[t] = -cov_ls / var_long
            else:
                beta[t] = 0.0
        else:
            beta[t] = 0.0

        l = long_pnl[t]
        s = short_pnl[t]

        if np.isnan(l) or np.isnan(s):
            continue

        if not initialized:
            ema_long = l
            ema_short = s
            ema_long_sq = l * l
            ema_long_short = l * s
            initialized = True
        else:
            ema_long = alpha * l + (1 - alpha) * ema_long
            ema_short = alpha * s + (1 - alpha) * ema_short
            ema_long_sq = alpha * (l * l) + (1 - alpha) * ema_long_sq
            ema_long_short = alpha * (l * s) + (1 - alpha) * ema_long_short

    return beta


@njit(cache=True)
def _compute_rolling_beta_by_group_ema(
    long_matrix: np.ndarray, short_matrix: np.ndarray,
    group_ids: np.ndarray, n_groups: int, alpha: float
) -> np.ndarray:
    """Compute rolling beta separately for each asset group."""
    n_days, n_assets = long_matrix.shape
    betas = np.zeros((n_days, n_groups), dtype=np.float64)

    for g in range(n_groups):
        long_g = np.zeros(n_days, dtype=np.float64)
        short_g = np.zeros(n_days, dtype=np.float64)

        for a in range(n_assets):
            if group_ids[a] == g:
                for t in range(n_days):
                    l = long_matrix[t, a]
                    s = short_matrix[t, a]
                    if not np.isnan(l):
                        long_g[t] += l
                    if not np.isnan(s):
                        short_g[t] += s

        ema_long = 0.0
        ema_short = 0.0
        ema_long_sq = 0.0
        ema_long_short = 0.0
        initialized = False

        for t in range(n_days):
            if initialized:
                var_long = ema_long_sq - ema_long * ema_long
                cov_ls = ema_long_short - ema_long * ema_short
                if var_long > 1e-12:
                    betas[t, g] = -cov_ls / var_long
                else:
                    betas[t, g] = 0.0
            else:
                betas[t, g] = 0.0

            l = long_g[t]
            s = short_g[t]

            if l == 0.0 and s == 0.0:
                continue

            if not initialized:
                ema_long = l
                ema_short = s
                ema_long_sq = l * l
                ema_long_short = l * s
                initialized = True
            else:
                ema_long = alpha * l + (1 - alpha) * ema_long
                ema_short = alpha * s + (1 - alpha) * ema_short
                ema_long_sq = alpha * (l * l) + (1 - alpha) * ema_long_sq
                ema_long_short = alpha * (l * s) + (1 - alpha) * ema_long_short

    return betas


@njit(cache=True, parallel=True)
def _rolling_winsorize_parallel(
    matrix: np.ndarray, window: int = 365, lo_pct: float = 0.01, hi_pct: float = 0.99
) -> np.ndarray:
    """Rolling winsorization: clip at rolling quantiles (no look-ahead).

    For each day d, compute 1% and 99% quantiles from [d-window+1, d] window,
    then clip the value at d. This is O(n_days * n_assets * window * log(window))
    but parallelized across assets.
    """
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for a in prange(n_assets):
        for d in range(n_days):
            val = matrix[d, a]
            if np.isnan(val):
                continue

            # Collect rolling window (up to current day, no look-ahead)
            start = max(0, d - window + 1)
            window_size = d - start + 1

            # Count valid values and collect them
            valid_count = 0
            for i in range(start, d + 1):
                if not np.isnan(matrix[i, a]):
                    valid_count += 1

            if valid_count < 20:  # Need enough data for meaningful quantiles
                result[d, a] = val
                continue

            # Collect valid values into array for sorting
            valid_vals = np.empty(valid_count, dtype=np.float64)
            k = 0
            for i in range(start, d + 1):
                v = matrix[i, a]
                if not np.isnan(v):
                    valid_vals[k] = v
                    k += 1

            # Sort and get quantiles
            valid_vals.sort()
            lo_idx = int(lo_pct * (valid_count - 1))
            hi_idx = int(hi_pct * (valid_count - 1))
            lo_val = valid_vals[lo_idx]
            hi_val = valid_vals[hi_idx]

            # Clip
            if val < lo_val:
                result[d, a] = lo_val
            elif val > hi_val:
                result[d, a] = hi_val
            else:
                result[d, a] = val

    return result


# ============================================================================
# Input Preprocessors
# ============================================================================
def preprocess_npnl(pnl_matrix: np.ndarray, vol_matrix: np.ndarray) -> np.ndarray:
    """Compute normalized PnL: npnl = pnl / vol."""
    with np.errstate(divide='ignore', invalid='ignore'):
        npnl = pnl_matrix / vol_matrix
        npnl[~np.isfinite(npnl)] = np.nan
    return npnl


def preprocess_rnpnl(npnl: np.ndarray) -> np.ndarray:
    """Compute residual normalized PnL: rnpnl = npnl - market."""
    market = np.nansum(npnl, axis=1, keepdims=True)
    return npnl - market


def preprocess_wpnl(npnl: np.ndarray, window: int = 365) -> np.ndarray:
    """Compute winsorized npnl: rolling 1%/99% quantile clipping (no look-ahead)."""
    return _rolling_winsorize_parallel(npnl, window, 0.01, 0.99)


# ============================================================================
# Signal Functions (pure functions returning raw signal values)
# ============================================================================
def signal_mac(input_matrix: np.ndarray, slow: int, fast: int) -> np.ndarray:
    """Moving average crossover: EMA(fast) - EMA(slow) of cumsum."""
    cinput = np.nancumsum(input_matrix, axis=0)
    ema_slow = _ema_columns_parallel(cinput, 1.0 / slow)
    ema_fast = _ema_columns_parallel(cinput, 1.0 / fast)
    return ema_fast - ema_slow


def signal_sharpe(input_matrix: np.ndarray, period: int) -> np.ndarray:
    """Sharpe-like ratio: EMA(input) / EMA(|input|)."""
    ema_input = _ema_columns_parallel(input_matrix, 1.0 / period)
    abs_input = np.abs(input_matrix)
    abs_input[~np.isfinite(abs_input)] = np.nan
    ema_abs = _ema_columns_parallel(abs_input, 1.0 / period)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = ema_input / ema_abs
        ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def signal_sign(input_matrix: np.ndarray, period: int) -> np.ndarray:
    """Sign persistence: EMA of sign(input)."""
    sign_input = np.sign(input_matrix)
    sign_input[~np.isfinite(input_matrix)] = np.nan
    return _ema_columns_parallel(sign_input, 1.0 / period)


def signal_clm(input_matrix: np.ndarray) -> np.ndarray:
    """Calmar-like ratio: cumsum / max_drawdown."""
    cinput = np.nancumsum(input_matrix, axis=0)
    max_dd = _rolling_max_drawdown_parallel(cinput)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = cinput / max_dd
        ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def signal_viso(vol_matrix: np.ndarray) -> np.ndarray:
    """Vol-inverse short-only: -1/vol."""
    with np.errstate(divide='ignore', invalid='ignore'):
        viso = -1.0 / vol_matrix
        viso[~np.isfinite(viso)] = np.nan
    return viso


# ============================================================================
# Signal Transforms
# ============================================================================
def rank_all(signal: np.ndarray) -> np.ndarray:
    """Cross-sectional rank across all assets, normalized to [-1, +1]."""
    return _cross_sectional_rank_parallel(signal)


def rank_by_group(signal: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Cross-sectional rank within each group, normalized to [-1, +1]."""
    return _cross_sectional_rank_by_group_parallel(signal, group_ids, n_groups)


def sumabs_norm(signal: np.ndarray) -> np.ndarray:
    """Row-wise normalization: signal / sum(|signal|)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum_abs = np.nansum(np.abs(signal), axis=1, keepdims=True)
        result = signal / row_sum_abs
        result[~np.isfinite(result)] = np.nan
    return result


def lag(signal: np.ndarray, days: int = 1) -> np.ndarray:
    """Lag signal by N days (avoid look-ahead)."""
    result = np.empty_like(signal)
    result[:days, :] = np.nan
    result[days:, :] = signal[:-days, :]
    return result


def combine(*signals: np.ndarray) -> np.ndarray:
    """Combine signals by summing and applying sum-abs normalization."""
    combined = np.zeros_like(signals[0])
    for sig in signals:
        combined = combined + np.nan_to_num(sig, nan=0.0)
    return sumabs_norm(combined)


# ============================================================================
# PnL Computation
# ============================================================================
def compute_pnl_components(weight_matrix: np.ndarray, ctx: BacktestContext):
    """Compute PnL for long leg, short leg, and total using inception weighting."""
    # Total PnL
    total_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        weight_matrix, ctx.asset_starts, ctx.asset_counts
    )
    total_pnl = np.nansum(total_matrix, axis=1)

    # Long PnL
    long_weights = np.maximum(weight_matrix, 0.0)
    long_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        long_weights, ctx.asset_starts, ctx.asset_counts
    )
    long_pnl = np.nansum(long_matrix, axis=1)

    # Short PnL
    short_weights = np.minimum(weight_matrix, 0.0)
    short_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        short_weights, ctx.asset_starts, ctx.asset_counts
    )
    short_pnl = np.nansum(short_matrix, axis=1)

    return total_pnl, long_pnl, short_pnl, long_matrix, short_matrix


# ============================================================================
# Hedge Strategies
# ============================================================================
def hedge_none(long_pnl: np.ndarray, short_pnl: np.ndarray, **kwargs) -> np.ndarray:
    """No hedge: return total (long + short)."""
    return long_pnl + short_pnl


def hedge_global(long_pnl: np.ndarray, short_pnl: np.ndarray, alpha: float = 1/365, **kwargs) -> np.ndarray:
    """Global beta hedge: short + beta * long."""
    rolling_beta = _compute_rolling_beta_ema(long_pnl, short_pnl, alpha)
    return short_pnl + rolling_beta * long_pnl


def hedge_by_group(
    long_matrix: np.ndarray, short_matrix: np.ndarray,
    group_ids: np.ndarray, n_groups: int, alpha: float = 1/365, **kwargs
) -> np.ndarray:
    """Hedge each asset group separately."""
    n_days = long_matrix.shape[0]
    n_assets = long_matrix.shape[1]

    group_betas = _compute_rolling_beta_by_group_ema(
        long_matrix, short_matrix, group_ids, n_groups, alpha
    )

    hedged_pnl = np.zeros(n_days, dtype=np.float64)
    for g in range(n_groups):
        long_g = np.zeros(n_days, dtype=np.float64)
        short_g = np.zeros(n_days, dtype=np.float64)
        for a in range(n_assets):
            if group_ids[a] == g:
                long_g += np.nan_to_num(long_matrix[:, a], nan=0.0)
                short_g += np.nan_to_num(short_matrix[:, a], nan=0.0)
        hedged_pnl += short_g + group_betas[:, g] * long_g

    return hedged_pnl


def hedge_gah(
    long_matrix: np.ndarray, short_matrix: np.ndarray,
    group_ids: np.ndarray, n_groups: int, alpha: float = 1/365, **kwargs
) -> np.ndarray:
    """Two-level hedge: groups first, then global."""
    n_days = long_matrix.shape[0]
    n_assets = long_matrix.shape[1]

    group_betas = _compute_rolling_beta_by_group_ema(
        long_matrix, short_matrix, group_ids, n_groups, alpha
    )

    hedged_long_total = np.zeros(n_days, dtype=np.float64)
    hedged_short_total = np.zeros(n_days, dtype=np.float64)

    for g in range(n_groups):
        long_g = np.zeros(n_days, dtype=np.float64)
        short_g = np.zeros(n_days, dtype=np.float64)
        for a in range(n_assets):
            if group_ids[a] == g:
                long_g += np.nan_to_num(long_matrix[:, a], nan=0.0)
                short_g += np.nan_to_num(short_matrix[:, a], nan=0.0)
        hedged_long_total += group_betas[:, g] * long_g
        hedged_short_total += short_g

    global_beta = _compute_rolling_beta_ema(hedged_long_total, hedged_short_total, alpha)
    return hedged_short_total + global_beta * hedged_long_total


HEDGE_STRATEGIES = {
    "none": hedge_none,
    "h": hedge_global,
    "hg": hedge_by_group,
    "gah": hedge_gah,
}


# ============================================================================
# Stats Computation
# ============================================================================
def compute_sharpe(pnl: np.ndarray) -> float:
    """Compute annualized Sharpe ratio."""
    mean = np.nanmean(pnl)
    std = np.nanstd(pnl)
    if std == 0:
        return np.nan
    return (mean * 252) / (std * np.sqrt(252))


def compute_ls_correl(long_pnl: np.ndarray, short_pnl: np.ndarray) -> float:
    """Compute correlation between long and short legs."""
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
    """Compute all portfolio statistics."""
    return PortfolioStats(
        sharpe=compute_sharpe(pnl),
        long_sharpe=compute_sharpe(long_pnl),
        short_sharpe=compute_sharpe(short_pnl),
        ls_correl=compute_ls_correl(long_pnl, short_pnl),
    )


# ============================================================================
# Strategy Definitions
# ============================================================================
# Signal registry: name -> (function, params)
SIGNAL_SPECS = {
    # MAC signals
    "mac.180.30": (signal_mac, {"slow": 180, "fast": 30}),
    "mac.180.10": (signal_mac, {"slow": 180, "fast": 10}),
    "mac.90.10": (signal_mac, {"slow": 90, "fast": 10}),
    "mac.90.30": (signal_mac, {"slow": 90, "fast": 30}),
    "mac.30.10": (signal_mac, {"slow": 30, "fast": 10}),
    # Other signals
    "srp.365": (signal_sharpe, {"period": 365}),
    "sign.365": (signal_sign, {"period": 365}),
    "clm.365": (signal_clm, {}),
}

# Combined signals: name -> list of base signals to combine
COMBINED_SIGNALS = {
    "mac.combo": ["mac.180.30", "mac.180.10", "mac.90.10", "mac.30.10"],
}

# Strategy templates: (signals, rank_mode, hedges)
# Each generates strategies for all input types (npnl, rnpnl)
STRATEGY_TEMPLATES = [
    # Baseline: vol-inverse short-only
    (["viso"], "all", ["none"]),

    # Single MAC signals
    (["mac.180.30"], "all", ["none", "h", "hg", "gah"]),
    (["mac.180.10"], "all", ["none", "h", "hg", "gah"]),
    (["mac.90.10"], "all", ["none", "h", "hg", "gah"]),
    (["mac.30.10"], "all", ["none", "h", "hg", "gah"]),
    (["mac.90.10"], "grp", ["none", "h"]),
    (["mac.90.30"], "grp", ["none", "h"]),

    # Sharpe
    (["srp.365"], "all", ["none", "h", "hg", "gah"]),
    (["srp.365"], "grp", ["none", "h"]),

    # Sign
    (["sign.365"], "all", ["none", "h", "hg", "gah"]),
    (["sign.365"], "grp", ["none", "h", "gah"]),

    # CLM
    (["clm.365"], "all", ["none", "h", "hg", "gah"]),
    (["clm.365"], "grp", ["none", "h", "gah"]),

    # MAC + Sharpe
    (["mac.180.10", "srp.365"], "all", ["none", "h", "hg", "gah"]),
    (["mac.90.10", "srp.365"], "grp", ["none", "h"]),

    # MAC combo + Sharpe
    (["mac.combo", "srp.365"], "all", ["none", "h", "hg", "gah"]),
    (["mac.combo", "srp.365"], "grp", ["none", "h"]),

    # Sharpe + Sign + MAC
    (["srp.365", "sign.365", "mac.90.10"], "all", ["none", "h", "hg", "gah"]),
    (["srp.365", "sign.365", "mac.90.10"], "grp", ["none", "h"]),

    # CLM + Sharpe
    (["clm.365", "srp.365"], "all", ["none", "h", "gah"]),
    (["clm.365", "srp.365"], "grp", ["none", "h", "gah"]),

    # CLM + MAC + Sharpe
    (["clm.365", "mac.180.10", "srp.365"], "all", ["none", "h", "gah"]),
    (["clm.365", "mac.90.10", "srp.365"], "grp", ["none", "h", "gah"]),

    # CLM + MAC combo + Sharpe
    (["clm.365", "mac.combo", "srp.365"], "all", ["none", "h", "gah"]),
    (["clm.365", "mac.combo", "srp.365"], "grp", ["none", "h", "gah"]),

    # CLM + Sharpe + Sign + MAC
    (["clm.365", "srp.365", "sign.365", "mac.90.10"], "all", ["none", "h", "gah"]),
    (["clm.365", "srp.365", "sign.365", "mac.90.10"], "grp", ["none", "h", "gah"]),

    # Sign + MAC + Sharpe
    (["sign.365", "mac.180.10", "srp.365"], "all", ["none", "h", "gah"]),
    (["sign.365", "mac.90.10", "srp.365"], "grp", ["none", "h", "gah"]),

    # Sign + MAC combo + Sharpe
    (["sign.365", "mac.combo", "srp.365"], "all", ["none", "h", "gah"]),
    (["sign.365", "mac.combo", "srp.365"], "grp", ["none", "h", "gah"]),

    # Sign + CLM + Sharpe
    (["sign.365", "clm.365", "srp.365"], "all", ["none", "h", "gah"]),
    (["sign.365", "clm.365", "srp.365"], "grp", ["none", "h", "gah"]),

    # Sign + CLM + MAC + Sharpe
    (["sign.365", "clm.365", "mac.180.10", "srp.365"], "all", ["none", "h", "gah"]),
    (["sign.365", "clm.365", "mac.90.10", "srp.365"], "grp", ["none", "h", "gah"]),

    # Sign + CLM + MAC combo + Sharpe
    (["sign.365", "clm.365", "mac.combo", "srp.365"], "all", ["none", "h", "gah"]),
    (["sign.365", "clm.365", "mac.combo", "srp.365"], "grp", ["none", "h", "gah"]),
]


# ============================================================================
# Signal Cache (pre-computed ranked signals)
# ============================================================================
class SignalCache:
    """Cache for pre-computed ranked/normalized/lagged signals."""

    def __init__(self, input_matrix: np.ndarray, vol_matrix: np.ndarray,
                 group_ids: np.ndarray, n_groups: int):
        self.input_matrix = input_matrix
        self.vol_matrix = vol_matrix
        self.group_ids = group_ids
        self.n_groups = n_groups
        self._cache_all = {}  # signal_name -> transformed signal (rank=all)
        self._cache_grp = {}  # signal_name -> transformed signal (rank=grp)

    def _compute_raw(self, name: str) -> np.ndarray:
        """Compute raw signal value."""
        if name == "viso":
            return signal_viso(self.vol_matrix)
        elif name in SIGNAL_SPECS:
            fn, params = SIGNAL_SPECS[name]
            return fn(self.input_matrix, **params)
        else:
            raise ValueError(f"Unknown signal: {name}")

    def _transform(self, raw: np.ndarray, rank_mode: str) -> np.ndarray:
        """Apply ranking, normalization, and lag to raw signal."""
        if rank_mode == "all":
            ranked = rank_all(raw)
        else:
            ranked = rank_by_group(raw, self.group_ids, self.n_groups)
        return lag(sumabs_norm(ranked), days=1)

    def get(self, name: str, rank_mode: str) -> np.ndarray:
        """Get a transformed signal (cached)."""
        cache = self._cache_all if rank_mode == "all" else self._cache_grp

        if name not in cache:
            if name == "mac.combo":
                # mac.combo is a sum of individual MAC signals (each already ranked)
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
                cache[name] = self._transform(raw, rank_mode)

        return cache[name]


def get_combined_signal(signal_names: list[str], cache: SignalCache, rank_mode: str) -> np.ndarray:
    """Get combined signal from cache (each base signal already ranked)."""
    if len(signal_names) == 1:
        return cache.get(signal_names[0], rank_mode)

    # Sum already-transformed signals, then re-normalize
    combined = None
    for name in signal_names:
        sig = cache.get(name, rank_mode)
        if combined is None:
            combined = np.copy(sig)
        else:
            combined = combined + np.nan_to_num(sig, nan=0.0)
    return sumabs_norm(combined)


# ============================================================================
# Strategy Execution
# ============================================================================
def make_strategy_name(input_type: str, signal_names: list[str], rank_mode: str, hedge: str) -> str:
    """Generate strategy name: {input}-{signals}-{rank}-{hedge}."""
    signal_part = "+".join(signal_names)
    rank_part = "all" if rank_mode == "all" else "grp"
    return f"{input_type}-{signal_part}-{rank_part}-{hedge}"


def run_strategy(input_type: str, signal_names: list[str], rank_mode: str, hedge: str,
                 ctx: BacktestContext, cache: SignalCache) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute a single strategy and return (name, pnl, long_pnl, short_pnl, time_ms)."""
    t0 = time.perf_counter()

    # Get combined signal weights from cache
    weights = get_combined_signal(signal_names, cache, rank_mode)

    # Compute PnL components
    total_pnl, long_pnl, short_pnl, long_matrix, short_matrix = compute_pnl_components(weights, ctx)

    # Apply hedge
    hedge_fn = HEDGE_STRATEGIES[hedge]
    if hedge in ("hg", "gah"):
        hedged_pnl = hedge_fn(
            long_matrix=long_matrix, short_matrix=short_matrix,
            group_ids=ctx.group_ids, n_groups=ctx.n_groups,
            long_pnl=long_pnl, short_pnl=short_pnl
        )
    else:
        hedged_pnl = hedge_fn(long_pnl=long_pnl, short_pnl=short_pnl)

    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000

    name = make_strategy_name(input_type, signal_names, rank_mode, hedge)
    return name, hedged_pnl, long_pnl, short_pnl, time_ms


def generate_all_strategies(ctx: BacktestContext) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Generate all strategies from templates."""
    # Build signal caches for each input type
    npnl_cache = SignalCache(ctx.npnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)
    rnpnl_cache = SignalCache(ctx.rnpnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)
    wpnl_cache = SignalCache(ctx.wpnl, ctx.vol_matrix, ctx.group_ids, ctx.n_groups)

    caches = {
        "npnl": npnl_cache,
        "rnpnl": rnpnl_cache,
        "wpnl": wpnl_cache,
    }

    results = []
    for signal_names, rank_mode, hedges in STRATEGY_TEMPLATES:
        for hedge in hedges:
            for input_type in ["npnl", "rnpnl", "wpnl"]:
                cache = caches[input_type]
                result = run_strategy(input_type, signal_names, rank_mode, hedge, ctx, cache)
                results.append(result)
    return results


# ============================================================================
# Data Loading
# ============================================================================
def load_arrow_as_dict(arrow_path: str) -> dict:
    """Load Arrow/Feather file and return as dict of numpy arrays."""
    table = pf.read_table(arrow_path)
    return {col: table[col].to_numpy(zero_copy_only=False) for col in table.column_names}


def parse_weights(wgt_arr: np.ndarray) -> np.ndarray:
    """Parse weight strings to floats (as fractions)."""
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
    """Sort straddles by asset for parallel processing."""
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
    """Load and compile group_table rules from AMT YAML file."""
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
    """Load mapping from Underlying -> Class from AMT file."""
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
    """Assign group to an asset based on rules."""
    field_values = {"Underlying": underlying, "Class": cls}
    for field, pattern, value in rules:
        field_val = field_values.get(field, "")
        if pattern.match(field_val):
            return value
    return "error"


def build_asset_group_table(asset_names: list[str], amt_path: str) -> dict[str, str]:
    """Build mapping from asset name to group."""
    rules = load_group_table(amt_path)
    class_map = load_asset_class_map(amt_path)

    group_map = {}
    for underlying in asset_names:
        cls = class_map.get(underlying, "")
        group = assign_group(underlying, cls, rules)
        group_map[underlying] = group
    return group_map


# ============================================================================
# Main
# ============================================================================
def main():
    # Load data
    print("Loading data...", end="", flush=True)
    straddles = load_arrow_as_dict("data/straddles.arrow")
    valuations = load_arrow_as_dict("data/valuations.arrow")
    print(" done")

    # Prepare arrays
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
    print("Building matrices...", end="", flush=True)
    pnl_matrix = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, pnl, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )
    vol_matrix = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, vol, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )
    print(" done")

    # Preprocess inputs
    print("Computing signals...", end="", flush=True)
    npnl = preprocess_npnl(pnl_matrix, vol_matrix)
    rnpnl = preprocess_rnpnl(npnl)
    print(" done")
    print("Computing wpnl (rolling winsorization)...", end="", flush=True)
    wpnl = preprocess_wpnl(npnl)

    # Build context
    ctx = BacktestContext(
        npnl=npnl,
        rnpnl=rnpnl,
        wpnl=wpnl,
        vol_matrix=vol_matrix,
        group_ids=group_ids,
        n_groups=n_groups,
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
    print(" done")

    # Generate all strategies
    print("Computing portfolios...", end="", flush=True)
    portfolios = generate_all_strategies(ctx)
    print(f" done ({len(portfolios)} strategies)")

    # Output table
    total_time = sum(p[4] for p in portfolios)
    avg_time = total_time / len(portfolios)

    print()
    print("=" * 115)
    print("PORTFOLIO COMPARISON (PNL)")
    print("=" * 115)
    print(f"{'Strategy':<55} {'Sharpe':>10} {'Long':>10} {'Short':>10} {'L/S Corr':>10} {'ms':>8}")
    print("-" * 115)

    # Sort by Sharpe
    portfolios_sorted = sorted(portfolios, key=lambda x: compute_sharpe(x[1]), reverse=True)

    for name, pnl, long_pnl, short_pnl, time_ms in portfolios_sorted:
        stats = compute_stats(pnl, long_pnl, short_pnl)
        print(f"{name:<55} {stats.sharpe:>10.4f} {stats.long_sharpe:>10.4f} {stats.short_sharpe:>10.4f} {stats.ls_correl:>10.4f} {time_ms:>8.1f}")

    print("=" * 115)
    print(f"Total: {total_time:.0f}ms | Avg: {avg_time:.1f}ms/strategy")


if __name__ == "__main__":
    main()
