"""Strategy Explorer - Interactive strategy construction and visualization."""
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.feather as pf
import streamlit as st
import yaml
from numba import njit, prange

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
    signal_matrix, asset_starts, asset_counts
):
    """Aggregation with inception-locked signal weights."""
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


@njit(cache=True, parallel=True)
def _aggregate_vol_by_asset_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, vol, dte, have_dte, grid_size, n_assets,
    asset_starts, asset_counts
):
    """Parallel aggregation of weighted vol."""
    vol_sum = np.zeros((grid_size, n_assets), np.float64)

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

                v = vol[idx]
                if not np.isnan(v):
                    vol_sum[day_idx, a] += v * w

    return vol_sum


@njit(cache=True, parallel=True)
def _cross_sectional_rank_parallel(matrix: np.ndarray) -> np.ndarray:
    """Cross-sectional ranking normalized to [-1, +1]."""
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for d in prange(n_days):
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


@njit(cache=True, parallel=True)
def _cross_sectional_rank_by_group_parallel(
    matrix: np.ndarray, group_ids: np.ndarray, n_groups: int
) -> np.ndarray:
    """Cross-sectional ranking within groups, normalized to [-1, +1]."""
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for d in prange(n_days):
        row = matrix[d, :]

        for g in range(n_groups):
            group_mask = (group_ids == g)
            valid_mask = group_mask & ~np.isnan(row)
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


@njit(cache=True, parallel=True)
def _rolling_winsorize_parallel(
    matrix: np.ndarray, window: int = 365, lo_pct: float = 0.01, hi_pct: float = 0.99
) -> np.ndarray:
    """Rolling winsorization: clip at rolling quantiles (no look-ahead)."""
    n_days, n_assets = matrix.shape
    result = np.empty((n_days, n_assets), dtype=np.float64)
    result[:] = np.nan

    for a in prange(n_assets):
        for d in range(n_days):
            val = matrix[d, a]
            if np.isnan(val):
                continue

            start = max(0, d - window + 1)
            valid_count = 0
            for i in range(start, d + 1):
                if not np.isnan(matrix[i, a]):
                    valid_count += 1

            if valid_count < 20:
                result[d, a] = val
                continue

            valid_vals = np.empty(valid_count, dtype=np.float64)
            k = 0
            for i in range(start, d + 1):
                v = matrix[i, a]
                if not np.isnan(v):
                    valid_vals[k] = v
                    k += 1

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

    def _compute_raw(self, name: str) -> np.ndarray:
        if name == "viso":
            return signal_viso(self.vol_matrix)
        elif name in SIGNAL_SPECS:
            fn, params = SIGNAL_SPECS[name]
            return fn(self.input_matrix, **params)
        else:
            raise ValueError(f"Unknown signal: {name}")

    def _transform(self, raw: np.ndarray, rank_mode: str) -> np.ndarray:
        if rank_mode == "all":
            ranked = rank_all(raw)
        else:
            ranked = rank_by_group(raw, self.group_ids, self.n_groups)
        return lag(sumabs_norm(ranked), days=1)

    def get(self, name: str, rank_mode: str) -> np.ndarray:
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
                cache[name] = self._transform(raw, rank_mode)

        return cache[name]


def get_combined_signal(signal_names: list[str], cache: SignalCache, rank_mode: str) -> np.ndarray:
    if len(signal_names) == 1:
        return cache.get(signal_names[0], rank_mode)

    combined = None
    for name in signal_names:
        sig = cache.get(name, rank_mode)
        if combined is None:
            combined = np.copy(sig)
        else:
            combined = combined + np.nan_to_num(sig, nan=0.0)
    return sumabs_norm(combined)


# ============================================================================
# PnL Computation
# ============================================================================
def compute_pnl_components(weight_matrix: np.ndarray, ctx: BacktestContext):
    total_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        weight_matrix, ctx.asset_starts, ctx.asset_counts
    )
    total_pnl = np.nansum(total_matrix, axis=1)

    long_weights = np.maximum(weight_matrix, 0.0)
    long_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        long_weights, ctx.asset_starts, ctx.asset_counts
    )
    long_pnl = np.nansum(long_matrix, axis=1)

    short_weights = np.minimum(weight_matrix, 0.0)
    short_matrix = _aggregate_inception_weighted_parallel(
        ctx.out0s, ctx.lens, ctx.starts, ctx.asset_ids, ctx.weights,
        ctx.d0, ctx.pnl, ctx.dte, ctx.have_dte, ctx.grid_size, ctx.n_assets,
        short_weights, ctx.asset_starts, ctx.asset_counts
    )
    short_pnl = np.nansum(short_matrix, axis=1)

    return total_pnl, long_pnl, short_pnl, long_matrix, short_matrix, total_matrix


# ============================================================================
# Hedge Strategies
# ============================================================================
def hedge_none(long_pnl: np.ndarray, short_pnl: np.ndarray, **kwargs) -> np.ndarray:
    return long_pnl + short_pnl


def hedge_global(long_pnl: np.ndarray, short_pnl: np.ndarray, alpha: float = 1/365, **kwargs) -> np.ndarray:
    rolling_beta = _compute_rolling_beta_ema(long_pnl, short_pnl, alpha)
    return short_pnl + rolling_beta * long_pnl


def hedge_by_group(
    long_matrix: np.ndarray, short_matrix: np.ndarray,
    group_ids: np.ndarray, n_groups: int, alpha: float = 1/365, **kwargs
) -> np.ndarray:
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


@st.cache_data
def load_all_data():
    """Load and prepare all data for backtesting."""
    straddles = load_arrow_as_dict("data/straddles.arrow")
    valuations = load_arrow_as_dict("data/valuations.arrow")

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

    return ctx, d0


# ============================================================================
# Strategy Execution
# ============================================================================
def run_strategy(input_type: str, signal_names: list[str], rank_mode: str, hedge: str,
                 ctx: BacktestContext, caches: dict[str, SignalCache],
                 asset_filter: list[str] | None = None):
    """Execute a single strategy."""
    t0 = time.perf_counter()

    cache = caches[input_type]
    weights = get_combined_signal(signal_names, cache, rank_mode)

    # Apply asset filter: zero out weights for assets not in selected groups
    if asset_filter:
        group_ids_selected = [ctx.unique_groups.index(g) for g in asset_filter]
        asset_mask = np.isin(ctx.group_ids, group_ids_selected)
        weights = weights.copy()  # Don't modify cached weights
        weights[:, ~asset_mask] = 0.0

    total_pnl, long_pnl, short_pnl, long_matrix, short_matrix, total_matrix = compute_pnl_components(weights, ctx)

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

    stats = compute_stats(hedged_pnl, long_pnl, short_pnl)

    return hedged_pnl, long_pnl, short_pnl, stats, time_ms, total_matrix, long_matrix, short_matrix


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


# ============================================================================
# Main App
# ============================================================================
def main():
    st.title("Strategy Explorer")

    # Load data
    with st.spinner("Loading data..."):
        ctx, d0 = load_all_data()

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
            "Input",
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
            "Signal Ranking",
            options=["all", "grp"],
            format_func=lambda x: "Rank across all assets" if x == "all" else "Rank in groups, add, rank sum",
            help="Cross-sectional ranking across all assets or within each group"
        )

        # Hedge strategy
        hedge = st.selectbox(
            "Hedge",
            options=list(HEDGE_STRATEGIES.keys()),
            format_func=lambda x: HEDGE_LABELS[x],
            index=3,  # Default to 'gah'
            help="Hedge strategy to apply"
        )

        st.divider()

        # Strategy name display
        signal_str = "+".join(selected_signals)
        asset_suffix = f"@{'+'.join(asset_filter)}" if asset_filter else ""
        strategy_name = f"{input_type}-{signal_str}-{rank_mode}-{hedge}{asset_suffix}"
        st.caption(f"**Strategy:** `{strategy_name}`")

    # Main content area
    col_stats, col_chart = st.columns([1, 3])

    # Run strategy with filter
    pnl, long_pnl, short_pnl, stats, time_ms, total_matrix, long_matrix, short_matrix = run_strategy(
        input_type, selected_signals, rank_mode, hedge,
        ctx, caches, asset_filter=asset_filter
    )

    # Display stats in col_stats
    with col_stats:
        st.subheader("Statistics")

        # Display filter: select which asset classes to show results for
        display_filter = st.multiselect(
            "Show Results For",
            options=ctx.unique_groups,
            default=[],
            format_func=lambda x: x.capitalize(),
            help="Filter displayed results (empty = all)"
        )

        # Apply display filter to matrices and re-apply hedge
        if display_filter:
            display_group_ids = [ctx.unique_groups.index(g) for g in display_filter]
            display_mask = np.isin(ctx.group_ids, display_group_ids)
            display_total = total_matrix[:, display_mask]
            display_long = long_matrix[:, display_mask]
            display_short = short_matrix[:, display_mask]

            # Sum filtered long/short
            display_long_pnl = np.nansum(display_long, axis=1)
            display_short_pnl = np.nansum(display_short, axis=1)

            # Re-apply hedge to filtered data
            hedge_fn = HEDGE_STRATEGIES[hedge]
            if hedge in ("hg", "gah"):
                # For group-based hedges, we need filtered group_ids
                filtered_group_ids = ctx.group_ids[display_mask]
                # Remap group_ids to be contiguous
                unique_filtered = np.unique(filtered_group_ids)
                remap = {old: new for new, old in enumerate(unique_filtered)}
                remapped_group_ids = np.array([remap[g] for g in filtered_group_ids], dtype=np.int32)
                n_filtered_groups = len(unique_filtered)

                display_pnl = hedge_fn(
                    long_matrix=display_long, short_matrix=display_short,
                    group_ids=remapped_group_ids, n_groups=n_filtered_groups,
                    long_pnl=display_long_pnl, short_pnl=display_short_pnl
                )
            else:
                display_pnl = hedge_fn(long_pnl=display_long_pnl, short_pnl=display_short_pnl)
        else:
            # No display filter - use the already-computed hedged values
            display_pnl = pnl
            display_long_pnl = long_pnl
            display_short_pnl = short_pnl
            display_total = total_matrix

        # Compute stats for displayed subset
        display_stats = compute_stats(display_pnl, display_long_pnl, display_short_pnl)

        st.metric("Sharpe Ratio", f"{display_stats.sharpe:.4f}")
        st.metric("Max Drawdown", f"{display_stats.max_drawdown:.4f}")
        st.metric("Long Sharpe", f"{display_stats.long_sharpe:.4f}")
        st.metric("Short Sharpe", f"{display_stats.short_sharpe:.4f}")
        st.metric("L/S Correlation", f"{display_stats.ls_correl:.4f}")

        st.divider()
        st.caption(f"Computed in {time_ms:.1f} ms")

    with col_chart:
        # Use radio instead of tabs - radio persists selection via key
        chart_view = st.radio(
            "Chart",
            options=["Cumulative PnL", "Rolling Correlation", "Cross-Sectional Range", "Cross-Sectional Hit Ratio", "Cross-Sectional Win/Loss"],
            horizontal=True,
            key="chart_view",
            label_visibility="collapsed"
        )

        if chart_view == "Cumulative PnL":
            fig = plot_cumulative_pnl(display_pnl, display_long_pnl, display_short_pnl, d0)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        elif chart_view == "Rolling Correlation":
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


if __name__ == "__main__":
    main()
