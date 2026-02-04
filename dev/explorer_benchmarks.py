"""Benchmark: Weighted PnL Matrix Construction.

Loads same data as backtest_explorer.py and builds a [dates × assets] matrix
with weighted sum of PnLs. Reports timings for each phase.

Compares NPZ vs Arrow loading performance.
"""
import os
import time
import numpy as np
import pandas as pd
import pyarrow.feather as pf
from numba import njit, prange


def _fmt_time(dt: float) -> str:
    """Format duration as ms or seconds."""
    ms = dt * 1000.0
    return f"{ms:.1f} ms" if ms < 1000.0 else f"{dt:.2f} s"


def compute_portfolio_stats(daily_pnl: np.ndarray) -> dict:
    """Compute standard portfolio statistics from daily PnL series.

    Args:
        daily_pnl: 1D array of daily PnL values

    Returns:
        Dictionary with: days, total_pnl, mean_daily, daily_std,
        realized_vol, sharpe, max_drawdown
    """
    days = len(daily_pnl)
    total_pnl = np.nansum(daily_pnl)
    mean_daily = np.nanmean(daily_pnl)
    daily_std = np.nanstd(daily_pnl)
    realized_vol = daily_std * np.sqrt(252)
    sharpe = (mean_daily * 252) / (daily_std * np.sqrt(252)) if daily_std > 0 else np.nan

    cum_pnl = np.nancumsum(daily_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = np.min(drawdown)

    return {
        'days': days,
        'total_pnl': total_pnl,
        'mean_daily': mean_daily,
        'daily_std': daily_std,
        'realized_vol': realized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }


def print_portfolio_table(title: str, ptf_pnl: np.ndarray, long_pnl: np.ndarray, short_pnl: np.ndarray):
    """Print a 3-column table comparing portfolio, long leg, and short leg stats.

    Args:
        title: Section title
        ptf_pnl: Daily PnL for full portfolio
        long_pnl: Daily PnL for long leg only
        short_pnl: Daily PnL for short leg only
    """
    ptf = compute_portfolio_stats(ptf_pnl)
    lng = compute_portfolio_stats(long_pnl)
    sht = compute_portfolio_stats(short_pnl)

    print()
    print(f"{title}")
    print("-" * 70)
    print(f"{'Metric':<20} {'Portfolio':>15} {'Long Leg':>15} {'Short Leg':>15}")
    print("-" * 70)
    print(f"{'Days':<20} {ptf['days']:>15,} {lng['days']:>15,} {sht['days']:>15,}")
    print(f"{'Total PnL':<20} {ptf['total_pnl']:>15.6f} {lng['total_pnl']:>15.6f} {sht['total_pnl']:>15.6f}")
    print(f"{'Mean Daily':<20} {ptf['mean_daily']:>15.6f} {lng['mean_daily']:>15.6f} {sht['mean_daily']:>15.6f}")
    print(f"{'Daily Std':<20} {ptf['daily_std']:>15.6f} {lng['daily_std']:>15.6f} {sht['daily_std']:>15.6f}")
    print(f"{'Realized Vol (ann)':<20} {ptf['realized_vol']:>14.2%} {lng['realized_vol']:>14.2%} {sht['realized_vol']:>14.2%}")
    print(f"{'Sharpe (ann)':<20} {ptf['sharpe']:>15.4f} {lng['sharpe']:>15.4f} {sht['sharpe']:>15.4f}")
    print(f"{'Max Drawdown':<20} {ptf['max_drawdown']:>15.6f} {lng['max_drawdown']:>15.6f} {sht['max_drawdown']:>15.6f}")
    print("-" * 70)


# ============================================================================
# Signal transformation helpers
# ============================================================================
def sumabs_norm(signal: np.ndarray) -> np.ndarray:
    """Row-wise sum-abs normalization: divide each row by sum of absolute values.

    After normalization, sum(abs(row)) = 1 for each row.
    NaN values are ignored in the sum and preserved in output.

    Args:
        signal: Input matrix [n_days, n_assets]

    Returns:
        Normalized matrix with same shape
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sum_abs = np.nansum(np.abs(signal), axis=1, keepdims=True)
        result = signal / row_sum_abs
        result[~np.isfinite(result)] = np.nan
    return result


def combine_signals(*signals: np.ndarray) -> np.ndarray:
    """Combine multiple signals by summing and applying sum-abs normalization.

    Args:
        *signals: Variable number of signal matrices [n_days, n_assets]

    Returns:
        Combined signal: sumabs_norm(sum of signals)
    """
    combined = np.zeros_like(signals[0])
    for sig in signals:
        combined = combined + np.nan_to_num(sig, nan=0.0)
    return sumabs_norm(combined)


def lag_signal(signal: np.ndarray, days: int = 1) -> np.ndarray:
    """Lag a signal matrix by N days (shift down, fill top with NaN).

    Args:
        signal: Input matrix [n_days, n_assets]
        days: Number of days to lag (default 1)

    Returns:
        Lagged signal matrix
    """
    result = np.empty_like(signal)
    result[:days, :] = np.nan
    result[days:, :] = signal[:-days, :]
    return result


def long_short_corr(long_pnl: np.ndarray, short_pnl: np.ndarray) -> float:
    """Compute correlation between long and short leg PnL series.

    Args:
        long_pnl: Daily PnL for long leg
        short_pnl: Daily PnL for short leg

    Returns:
        Correlation coefficient, or NaN if insufficient data or zero variance
    """
    valid = ~(np.isnan(long_pnl) | np.isnan(short_pnl))
    if np.sum(valid) < 2:
        return np.nan
    long_valid = long_pnl[valid]
    short_valid = short_pnl[valid]
    # Check for zero variance (all zeros or constant)
    if np.std(long_valid) == 0 or np.std(short_valid) == 0:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(long_valid, short_valid)[0, 1]
    return corr if np.isfinite(corr) else np.nan


# ============================================================================
# Numba kernel for weighted aggregation
# ============================================================================
@njit(cache=True)
def _aggregate_weighted_daily_by_asset(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets
):
    """Aggregate weighted daily pnl per asset into 2D grid [days, assets].

    Args:
        out0s: Starting index into valuations for each straddle (int32[S])
        lens: Number of days per straddle (int32[S])
        starts_epoch: Start date as days since epoch (int32[S])
        asset_ids: Asset index for each straddle (int32[S])
        weights: Per-straddle weight (float64[S])
        d0: Minimum epoch day (grid offset)
        pnl: Daily PnL values (float64[N])
        dte: Days to expiry (int32[N])
        have_dte: Whether dte array is valid
        grid_size: Number of days in output grid
        n_assets: Number of unique assets

    Returns:
        pnl_sum: Weighted PnL sum [grid_size, n_assets]
    """
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)

    n = out0s.shape[0]
    for k in range(n):
        o = out0s[k]
        L = lens[k]
        start = starts_epoch[k]
        a = asset_ids[k]
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
def _aggregate_weighted_daily_by_asset_parallel(
    out0s, lens, starts_epoch, asset_ids, weights,
    d0, pnl, dte, have_dte, grid_size, n_assets,
    asset_starts, asset_counts
):
    """Parallel version: each thread processes one asset's straddles.

    Args:
        ... (same as serial version)
        asset_starts: Start index in sorted arrays for each asset (int32[n_assets])
        asset_counts: Number of straddles per asset (int32[n_assets])

    Returns:
        pnl_sum: Weighted PnL sum [grid_size, n_assets]
    """
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)

    # Parallel loop over assets - no memory contention since each thread
    # writes to its own column
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
    """Aggregate daily PnL weighted by portfolio signal at straddle inception.

    For each straddle, the portfolio weight is looked up at inception (start date)
    and locked in for all subsequent daily PnLs. This is realistic because you
    can only know the signal at the time of trade entry.

    Args:
        out0s: Starting index into valuations for each straddle (int32[S])
        lens: Number of days per straddle (int32[S])
        starts_epoch: Start date as days since epoch (int32[S])
        asset_ids: Asset index for each straddle (int32[S])
        weights: Per-straddle weight (float64[S])
        d0: Minimum epoch day (grid offset)
        pnl: Daily PnL values (float64[N])
        dte: Days to expiry (int32[N])
        have_dte: Whether dte array is valid
        grid_size: Number of days in output grid
        n_assets: Number of unique assets
        ptf_matrix: Portfolio weights [grid_size, n_assets]
        asset_starts: Start index in sorted arrays for each asset (int32[n_assets])
        asset_counts: Number of straddles per asset (int32[n_assets])

    Returns:
        result: Inception-weighted PnL sum [grid_size, n_assets]
    """
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

            # Get inception day index
            inception_idx = start - d0

            # Look up portfolio weight at inception (bounds check)
            if inception_idx < 0 or inception_idx >= grid_size:
                ptf_w = 0.0
            else:
                ptf_w = ptf_matrix[inception_idx, a]
                if np.isnan(ptf_w):
                    ptf_w = 0.0

            # Combined weight: original straddle weight × portfolio weight at inception
            combined_w = w * ptf_w

            # Skip if no position
            if combined_w == 0.0:
                continue

            # Apply to all days of this straddle
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


# ============================================================================
# Signal computation kernels
# ============================================================================
@njit(cache=True, parallel=True)
def _ema_columns_parallel(data: np.ndarray, alpha: float) -> np.ndarray:
    """Compute EMA along axis 0 (time) for each column in parallel.

    EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    Args:
        data: Input matrix [n_days, n_assets]
        alpha: Smoothing factor (e.g., 1/365 for 1-year EMA)

    Returns:
        ema: EMA matrix [n_days, n_assets]
    """
    n_days, n_assets = data.shape
    ema = np.zeros_like(data)
    decay = 1.0 - alpha

    for a in prange(n_assets):
        # Initialize with first non-NaN value
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
def _rolling_minmax_columns_parallel(data: np.ndarray, window: int) -> tuple:
    """Compute rolling min and max along axis 0 for each column in parallel.

    Args:
        data: Input matrix [n_days, n_assets]
        window: Rolling window size in days

    Returns:
        (rolling_min, rolling_max): Two matrices [n_days, n_assets]
    """
    n_days, n_assets = data.shape
    rolling_min = np.full_like(data, np.nan)
    rolling_max = np.full_like(data, np.nan)

    for a in prange(n_assets):
        for t in range(n_days):
            # Window start (inclusive)
            start = max(0, t - window + 1)

            min_val = np.inf
            max_val = -np.inf
            valid_count = 0

            for i in range(start, t + 1):
                val = data[i, a]
                if not np.isnan(val):
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
                    valid_count += 1

            if valid_count > 0:
                rolling_min[t, a] = min_val
                rolling_max[t, a] = max_val

    return rolling_min, rolling_max


@njit(cache=True, parallel=True)
def _cross_sectional_rank_parallel(data: np.ndarray) -> np.ndarray:
    """Compute cross-sectional rank for each row, normalized to [-1, +1].

    For each day, ranks assets from -1 (lowest) to +1 (highest).
    NaN values remain NaN.

    Args:
        data: Input matrix [n_days, n_assets]

    Returns:
        ranks: Rank matrix [n_days, n_assets] with values in [-1, +1]
    """
    n_days, n_assets = data.shape
    ranks = np.full_like(data, np.nan)

    for t in prange(n_days):
        # Count valid values
        valid_count = 0
        for a in range(n_assets):
            if not np.isnan(data[t, a]):
                valid_count += 1

        if valid_count < 2:
            continue

        # Simple ranking: count how many values are less than each value
        for a in range(n_assets):
            if np.isnan(data[t, a]):
                continue

            rank = 0
            val = data[t, a]
            for b in range(n_assets):
                if not np.isnan(data[t, b]) and data[t, b] < val:
                    rank += 1

            # Normalize to [-1, +1]
            # rank goes from 0 to valid_count-1
            # normalized = 2 * rank / (valid_count - 1) - 1
            ranks[t, a] = 2.0 * rank / (valid_count - 1) - 1.0

    return ranks


def prepare_asset_groups(asset_ids: np.ndarray, n_assets: int):
    """Sort straddles by asset and compute start/count arrays for parallel processing."""
    # Sort indices by asset_id
    sort_idx = np.argsort(asset_ids)

    # Compute start index and count for each asset
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


def parse_weights(wgt_arr: np.ndarray) -> np.ndarray:
    """Parse weight strings to floats (as fractions, e.g., '33.3' -> 0.333).

    Handles edge cases:
    - Empty string -> 1.0
    - Invalid/non-numeric -> 1.0
    """
    weights = np.ones(len(wgt_arr), dtype=np.float64)

    for i, w in enumerate(wgt_arr):
        try:
            s = str(w).strip()
            if s:
                weights[i] = float(s) / 100.0
        except (ValueError, TypeError):
            pass  # Keep default 1.0

    return weights


# ============================================================================
# Arrow loading
# ============================================================================
def load_arrow_as_dict(arrow_path: str) -> dict:
    """Load Arrow/Feather file and return as dict of numpy arrays."""
    table = pf.read_table(arrow_path)
    return {col: table[col].to_numpy(zero_copy_only=False) for col in table.column_names}


def load_data():
    """Load data from Arrow files."""
    print("Loading data (Arrow)...", end="", flush=True)
    t0 = time.perf_counter()
    straddles = load_arrow_as_dict("data/straddles.arrow")
    valuations = load_arrow_as_dict("data/valuations.arrow")
    dt = time.perf_counter() - t0
    print(f" {_fmt_time(dt)}")
    print(f"  straddles: {len(straddles['asset']):,}")
    print(f"  valuations: {len(valuations['pnl']):,}")
    return straddles, valuations


def main():
    # ========================================================================
    # Load data
    # ========================================================================
    straddles, valuations = load_data()

    total_start = time.perf_counter()

    # ========================================================================
    # Phase 2: Parse weights
    # ========================================================================
    print("Parsing weights......", end="", flush=True)
    t0 = time.perf_counter()

    weights = parse_weights(straddles["wgt"])

    dt_weights = time.perf_counter() - t0
    print(f": {_fmt_time(dt_weights)}")
    print(f"  unique weights: {len(np.unique(weights))}")
    print(f"  weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # ========================================================================
    # Phase 3: Prepare arrays
    # ========================================================================
    print("Preparing arrays.....", end="", flush=True)
    t0 = time.perf_counter()

    # Straddle arrays
    out0s = straddles["out0"].astype(np.int32)
    lens = straddles["length"].astype(np.int32)
    starts = straddles["month_start_epoch"].astype(np.int32)

    # Factorize assets
    asset_str = np.asarray([str(x) for x in straddles["asset"]], dtype=object)
    asset_codes, asset_names = pd.factorize(asset_str, sort=True)
    asset_ids = asset_codes.astype(np.int32)
    n_assets = len(asset_names)

    # Grid bounds
    d0 = int(starts.min())
    d1 = int((starts + lens - 1).max())
    grid_size = d1 - d0 + 1

    # Valuations
    pnl = valuations["pnl"]
    vol = valuations["vol"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    dt_prep = time.perf_counter() - t0
    print(f": {_fmt_time(dt_prep)}")
    print(f"  assets: {n_assets}")
    print(f"  grid: {grid_size:,} days")

    # ========================================================================
    # Phase 4: Build PnL matrix
    # ========================================================================
    # Prepare sorted arrays for parallel kernel
    print("Preparing asset groups...", end="", flush=True)
    t0 = time.perf_counter()
    sort_idx, asset_starts, asset_counts = prepare_asset_groups(asset_ids, n_assets)
    out0s_sorted = out0s[sort_idx]
    lens_sorted = lens[sort_idx]
    starts_sorted = starts[sort_idx]
    weights_sorted = weights[sort_idx]
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Warm up kernel (first call includes JIT compilation)
    print("Warming up kernels...", end="", flush=True)
    t0 = time.perf_counter()
    _ = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, pnl, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Build PnL matrix
    print("Building PnL matrix...", end="", flush=True)
    t0 = time.perf_counter()
    pnl_matrix = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, pnl, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )
    dt_pnl_kernel = time.perf_counter() - t0
    print(f" {_fmt_time(dt_pnl_kernel)}")

    # ========================================================================
    # Phase 5: Build Vol matrix
    # ========================================================================
    print("Building Vol matrix...", end="", flush=True)
    t0 = time.perf_counter()
    vol_matrix = _aggregate_weighted_daily_by_asset_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, vol, dte, have_dte, grid_size, n_assets,
        asset_starts, asset_counts
    )
    dt_vol_kernel = time.perf_counter() - t0
    print(f" {_fmt_time(dt_vol_kernel)}")

    # ========================================================================
    # Phase 6: Signal computation pipeline
    # ========================================================================
    print()
    print("=" * 50)
    print("SIGNAL COMPUTATION PIPELINE")
    print("=" * 50)

    # --- npnl = pnl / vol ---
    print("Computing npnl (pnl/vol)...", end="", flush=True)
    t0 = time.perf_counter()
    with np.errstate(divide='ignore', invalid='ignore'):
        npnl = pnl_matrix / vol_matrix
        npnl[~np.isfinite(npnl)] = np.nan
    dt_npnl = time.perf_counter() - t0
    print(f" {_fmt_time(dt_npnl)}")

    # --- cnpnl = cumsum of npnl ---
    print("Computing cnpnl (cumsum)...", end="", flush=True)
    t0 = time.perf_counter()
    cnpnl = np.nancumsum(npnl, axis=0)
    dt_cnpnl = time.perf_counter() - t0
    print(f" {_fmt_time(dt_cnpnl)}")

    # --- EMA warmup ---
    print("Warming up EMA kernels...", end="", flush=True)
    t0 = time.perf_counter()
    _ = _ema_columns_parallel(cnpnl, 1.0 / 365.0)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # --- ema_cnpnl_365 = EMA with alpha=1/365 ---
    print("Computing ema_cnpnl_365 (3 runs):")
    ema365_times = []
    for i in range(3):
        t0 = time.perf_counter()
        ema_cnpnl_365 = _ema_columns_parallel(cnpnl, 1.0 / 365.0)
        dt = time.perf_counter() - t0
        ema365_times.append(dt)
        print(f"  Run {i+1}: {_fmt_time(dt)}")
    dt_ema365 = sum(ema365_times) / len(ema365_times)

    # --- ema_cnpnl_90 = EMA with alpha=1/90 ---
    print("Computing ema_cnpnl_90 (3 runs):")
    ema90_times = []
    for i in range(3):
        t0 = time.perf_counter()
        ema_cnpnl_90 = _ema_columns_parallel(cnpnl, 1.0 / 90.0)
        dt = time.perf_counter() - t0
        ema90_times.append(dt)
        print(f"  Run {i+1}: {_fmt_time(dt)}")
    dt_ema90 = sum(ema90_times) / len(ema90_times)

    # --- ema_cnpnl_180 = EMA with alpha=1/180 ---
    print("Computing ema_cnpnl_180...", end="", flush=True)
    t0 = time.perf_counter()
    ema_cnpnl_180 = _ema_columns_parallel(cnpnl, 1.0 / 180.0)
    dt_ema180 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_ema180)}")

    # --- ema_cnpnl_30 = EMA with alpha=1/30 ---
    print("Computing ema_cnpnl_30...", end="", flush=True)
    t0 = time.perf_counter()
    ema_cnpnl_30 = _ema_columns_parallel(cnpnl, 1.0 / 30.0)
    dt_ema30 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_ema30)}")

    # --- ema_cnpnl_10 = EMA with alpha=1/10 ---
    print("Computing ema_cnpnl_10...", end="", flush=True)
    t0 = time.perf_counter()
    ema_cnpnl_10 = _ema_columns_parallel(cnpnl, 1.0 / 10.0)
    dt_ema10 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_ema10)}")

    # --- Cross-sectional rank warmup ---
    print("Warming up rank kernel...", end="", flush=True)
    t0 = time.perf_counter()
    _ = _cross_sectional_rank_parallel(ema_cnpnl_365)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # --- signal_365 = cross-sectional rank of ema_cnpnl_365 ---
    print("Computing signal_365 (3 runs):")
    sig365_times = []
    for i in range(3):
        t0 = time.perf_counter()
        signal_365 = _cross_sectional_rank_parallel(ema_cnpnl_365)
        dt = time.perf_counter() - t0
        sig365_times.append(dt)
        print(f"  Run {i+1}: {_fmt_time(dt)}")
    dt_sig365 = sum(sig365_times) / len(sig365_times)

    # --- signal_90 = cross-sectional rank of ema_cnpnl_90 ---
    print("Computing signal_90 (3 runs):")
    sig90_times = []
    for i in range(3):
        t0 = time.perf_counter()
        signal_90 = _cross_sectional_rank_parallel(ema_cnpnl_90)
        dt = time.perf_counter() - t0
        sig90_times.append(dt)
        print(f"  Run {i+1}: {_fmt_time(dt)}")
    dt_sig90 = sum(sig90_times) / len(sig90_times)

    # --- signal_365_90 = cross-sectional rank of (ema_cnpnl_90 - ema_cnpnl_365) ---
    print("Computing signal_365_90 (3 runs):")
    sig365_90_times = []
    ema_diff = ema_cnpnl_90 - ema_cnpnl_365
    for i in range(3):
        t0 = time.perf_counter()
        signal_365_90 = _cross_sectional_rank_parallel(ema_diff)
        dt = time.perf_counter() - t0
        sig365_90_times.append(dt)
        print(f"  Run {i+1}: {_fmt_time(dt)}")
    dt_sig365_90 = sum(sig365_90_times) / len(sig365_90_times)

    # --- signal_180_30 = cross-sectional rank of (ema_cnpnl_30 - ema_cnpnl_180) ---
    print("Computing signal_180_30...", end="", flush=True)
    t0 = time.perf_counter()
    ema_diff_180_30 = ema_cnpnl_30 - ema_cnpnl_180
    signal_180_30 = _cross_sectional_rank_parallel(ema_diff_180_30)
    dt_sig180_30 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_sig180_30)}")

    # --- signal_180_10 = cross-sectional rank of (ema_cnpnl_10 - ema_cnpnl_180) ---
    print("Computing signal_180_10...", end="", flush=True)
    t0 = time.perf_counter()
    ema_diff_180_10 = ema_cnpnl_10 - ema_cnpnl_180
    signal_180_10 = _cross_sectional_rank_parallel(ema_diff_180_10)
    dt_sig180_10 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_sig180_10)}")

    # Signal summary
    print()
    print("-" * 50)
    print("Signal Pipeline Timings:")
    print(f"  npnl (pnl/vol):        {_fmt_time(dt_npnl)}")
    print(f"  cnpnl (cumsum):        {_fmt_time(dt_cnpnl)}")
    print(f"  ema_cnpnl_365:         {_fmt_time(dt_ema365)}")
    print(f"  ema_cnpnl_90:          {_fmt_time(dt_ema90)}")
    print(f"  signal_365 (rank):     {_fmt_time(dt_sig365)}")
    print(f"  signal_90 (rank):      {_fmt_time(dt_sig90)}")
    print(f"  signal_365_90 (rank):  {_fmt_time(dt_sig365_90)}")
    dt_signal_total = dt_npnl + dt_cnpnl + dt_ema365 + dt_ema90 + dt_sig365 + dt_sig90 + dt_sig365_90
    print(f"  Total:                 {_fmt_time(dt_signal_total)}")

    # Signal statistics
    print()
    print("Signal Statistics:")
    valid_365 = np.sum(~np.isnan(signal_365))
    valid_90 = np.sum(~np.isnan(signal_90))
    valid_365_90 = np.sum(~np.isnan(signal_365_90))
    print(f"  signal_365 valid:    {valid_365:,} ({100*valid_365/signal_365.size:.1f}%)")
    print(f"  signal_90 valid:     {valid_90:,} ({100*valid_90/signal_90.size:.1f}%)")
    print(f"  signal_365_90 valid: {valid_365_90:,} ({100*valid_365_90/signal_365_90.size:.1f}%)")
    print(f"  signal_365 range:    [{np.nanmin(signal_365):.2f}, {np.nanmax(signal_365):.2f}]")
    print(f"  signal_90 range:     [{np.nanmin(signal_90):.2f}, {np.nanmax(signal_90):.2f}]")
    print(f"  signal_365_90 range: [{np.nanmin(signal_365_90):.2f}, {np.nanmax(signal_365_90):.2f}]")

    # --- mac_365_90 = row-wise sum-abs normalized signal_365_90, lagged 1 day ---
    print("Computing mac_365_90 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    mac_365_90 = lag_signal(sumabs_norm(signal_365_90), days=1)
    dt_mac = time.perf_counter() - t0
    print(f" {_fmt_time(dt_mac)}")

    # --- mac_180_30 = row-wise sum-abs normalized signal_180_30, lagged 1 day ---
    print("Computing mac_180_30 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    mac_180_30 = lag_signal(sumabs_norm(signal_180_30), days=1)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # --- mac_180_10 = row-wise sum-abs normalized signal_180_10, lagged 1 day ---
    print("Computing mac_180_10 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    mac_180_10 = lag_signal(sumabs_norm(signal_180_10), days=1)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # ========================================================================
    # sharpe_365 signal: rolling Sharpe ratio based signal
    # ========================================================================
    # EMA of npnl (grid-level)
    print("Computing ema_npnl_365...", end="", flush=True)
    t0 = time.perf_counter()
    ema_npnl_365 = _ema_columns_parallel(npnl, 1.0 / 365.0)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # EMA of abs(npnl)
    print("Computing ema_abs_npnl_365...", end="", flush=True)
    t0 = time.perf_counter()
    abs_npnl = np.abs(npnl)
    abs_npnl[~np.isfinite(abs_npnl)] = np.nan
    ema_abs_npnl_365 = _ema_columns_parallel(abs_npnl, 1.0 / 365.0)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Ratio: rolling Sharpe-like metric
    print("Computing sharpe_ratio_365...", end="", flush=True)
    t0 = time.perf_counter()
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe_ratio_365 = ema_npnl_365 / ema_abs_npnl_365
        sharpe_ratio_365[~np.isfinite(sharpe_ratio_365)] = np.nan
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Cross-sectional rank
    print("Computing sharpe_365 rank...", end="", flush=True)
    t0 = time.perf_counter()
    sharpe_365_ranked = _cross_sectional_rank_parallel(sharpe_ratio_365)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Sum-abs normalize and lag
    print("Computing sharpe_365 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    sharpe_365 = lag_signal(sumabs_norm(sharpe_365_ranked), days=1)
    dt_sharpe = time.perf_counter() - t0
    print(f" {_fmt_time(dt_sharpe)}")

    # ========================================================================
    # sign_365 signal: EMA of sign of returns
    # ========================================================================
    print("Computing sign of npnl...", end="", flush=True)
    t0 = time.perf_counter()
    sign_npnl = np.sign(npnl)
    sign_npnl[~np.isfinite(npnl)] = np.nan
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print("Computing ema_sign_365...", end="", flush=True)
    t0 = time.perf_counter()
    ema_sign_365 = _ema_columns_parallel(sign_npnl, 1.0 / 365.0)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print("Computing sign_365 rank...", end="", flush=True)
    t0 = time.perf_counter()
    sign_365_ranked = _cross_sectional_rank_parallel(ema_sign_365)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print("Computing sign_365 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    sign_365 = lag_signal(sumabs_norm(sign_365_ranked), days=1)
    dt_sign = time.perf_counter() - t0
    print(f" {_fmt_time(dt_sign)}")

    # ========================================================================
    # range_365 signal: rolling peak-to-trough position
    # ========================================================================
    # cnpnl is already computed (cumsum of npnl)
    print("Computing rolling min/max of cnpnl (365-day window)...", end="", flush=True)
    t0 = time.perf_counter()
    rolling_min, rolling_max = _rolling_minmax_columns_parallel(cnpnl, 365)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Position within range: cnpnl / (rolling_max - rolling_min)
    print("Computing range position (cnpnl / range)...", end="", flush=True)
    t0 = time.perf_counter()
    with np.errstate(divide='ignore', invalid='ignore'):
        range_width = rolling_max - rolling_min
        range_position = cnpnl / range_width
        range_position[~np.isfinite(range_position)] = np.nan
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Cross-sectional rank
    print("Computing range_365 rank...", end="", flush=True)
    t0 = time.perf_counter()
    range_365_ranked = _cross_sectional_rank_parallel(range_position)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Sum-abs normalize and lag
    print("Computing range_365 (sum-abs normalized, lagged)...", end="", flush=True)
    t0 = time.perf_counter()
    range_365 = lag_signal(sumabs_norm(range_365_ranked), days=1)
    dt_range = time.perf_counter() - t0
    print(f" {_fmt_time(dt_range)}")

    # ========================================================================
    # Combined signal: mac_365_90 + sharpe_365 + sign_365 + range_365
    # ========================================================================
    print("Computing combined signal (mac_365_90 + sharpe_365 + sign_365 + range_365)...", end="", flush=True)
    t0 = time.perf_counter()
    combined_signal = combine_signals(mac_365_90, sharpe_365, sign_365, range_365)
    dt_combined = time.perf_counter() - t0
    print(f" {_fmt_time(dt_combined)}")

    # ========================================================================
    # Combined2 signal: sum ranks -> scale to [-1,+1] -> sumabs_norm -> lag
    # ========================================================================
    print("Computing combined2 signal (sum ranks -> rank -> sumabs -> lag)...", end="", flush=True)
    t0 = time.perf_counter()
    # Sum the ranked signals (before their sumabs normalization)
    # signal_365_90 is the ranked MAC signal, others have _ranked suffix
    combined2_sum = (
        np.nan_to_num(signal_365_90, nan=0.0) +
        np.nan_to_num(sharpe_365_ranked, nan=0.0) +
        np.nan_to_num(sign_365_ranked, nan=0.0) +
        np.nan_to_num(range_365_ranked, nan=0.0)
    )
    # Scale to [-1, +1] via cross-sectional rank
    combined2_ranked = _cross_sectional_rank_parallel(combined2_sum)
    # Sum-abs normalize and lag
    combined2_signal = lag_signal(sumabs_norm(combined2_ranked), days=1)
    dt_combined2 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_combined2)}")

    # ========================================================================
    # Combined3 signal: multiplicative combination of normalized signals
    # ========================================================================
    print("Computing combined3 signal (multiply normalized signals -> sumabs -> lag)...", end="", flush=True)
    t0 = time.perf_counter()
    # Multiply the already-normalized signals (before lag)
    # mac_365_90, sharpe_365, sign_365, range_365 are lagged, so we use unlagged versions
    mac_unlagged = sumabs_norm(signal_365_90)
    sharpe_unlagged = sumabs_norm(sharpe_365_ranked)
    sign_unlagged = sumabs_norm(sign_365_ranked)
    range_unlagged = sumabs_norm(range_365_ranked)
    # Multiplicative combination: product of normalized signals, then normalize again
    combined3_product = mac_unlagged * sharpe_unlagged * sign_unlagged * range_unlagged
    combined3_signal = lag_signal(sumabs_norm(combined3_product), days=1)
    dt_combined3 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_combined3)}")

    # ========================================================================
    # Combined4 signal: MAC 180-30 + MAC 180-10 + SHARPE 365
    # ========================================================================
    print("Computing combined4 signal (mac_180_30 + mac_180_10 + sharpe_365)...", end="", flush=True)
    t0 = time.perf_counter()
    combined4_signal = combine_signals(mac_180_30, mac_180_10, sharpe_365)
    dt_combined4 = time.perf_counter() - t0
    print(f" {_fmt_time(dt_combined4)}")

    # ========================================================================
    # Compute valuations-level npnl (needed for all portfolios)
    # ========================================================================
    print("Computing valuations-level npnl...", end="", flush=True)
    t0 = time.perf_counter()
    with np.errstate(divide='ignore', invalid='ignore'):
        npnl_vals = pnl / vol
        npnl_vals[~np.isfinite(npnl_vals)] = np.nan
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # ========================================================================
    # Helper: build inception-weighted portfolio with long/short legs
    # ========================================================================
    def build_portfolio_pnls(weight_matrix, name):
        """Build inception-weighted PnL for portfolio and its long/short legs."""
        # Full portfolio
        ptf_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, pnl, dte, have_dte, grid_size, n_assets,
            weight_matrix, asset_starts, asset_counts
        )
        ptf_pnl = np.nansum(ptf_matrix, axis=1)

        # Long leg (positive weights only)
        long_weights = np.maximum(weight_matrix, 0.0)
        long_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, pnl, dte, have_dte, grid_size, n_assets,
            long_weights, asset_starts, asset_counts
        )
        long_pnl = np.nansum(long_matrix, axis=1)

        # Short leg (negative weights only)
        short_weights = np.minimum(weight_matrix, 0.0)
        short_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, pnl, dte, have_dte, grid_size, n_assets,
            short_weights, asset_starts, asset_counts
        )
        short_pnl = np.nansum(short_matrix, axis=1)

        return ptf_pnl, long_pnl, short_pnl

    def build_portfolio_npnls(weight_matrix, name):
        """Build inception-weighted npnl for portfolio and its long/short legs."""
        # Full portfolio
        ptf_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, npnl_vals, dte, have_dte, grid_size, n_assets,
            weight_matrix, asset_starts, asset_counts
        )
        ptf_npnl = np.nansum(ptf_matrix, axis=1)

        # Long leg (positive weights only)
        long_weights = np.maximum(weight_matrix, 0.0)
        long_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, npnl_vals, dte, have_dte, grid_size, n_assets,
            long_weights, asset_starts, asset_counts
        )
        long_npnl = np.nansum(long_matrix, axis=1)

        # Short leg (negative weights only)
        short_weights = np.minimum(weight_matrix, 0.0)
        short_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, npnl_vals, dte, have_dte, grid_size, n_assets,
            short_weights, asset_starts, asset_counts
        )
        short_npnl = np.nansum(short_matrix, axis=1)

        return ptf_npnl, long_npnl, short_npnl

    def compute_hedge_beta(long_pnl: np.ndarray, short_pnl: np.ndarray) -> float:
        """Compute beta for hedging: beta = cor(long, -short) * vol(short) / vol(long).

        Args:
            long_pnl: Daily PnL for long leg
            short_pnl: Daily PnL for short leg

        Returns:
            Beta hedge ratio (scalar)
        """
        valid = ~(np.isnan(long_pnl) | np.isnan(short_pnl))
        if np.sum(valid) < 2:
            return 0.0
        long_valid = long_pnl[valid]
        short_valid = short_pnl[valid]
        neg_short = -short_valid

        vol_long = np.std(long_valid)
        vol_short = np.std(short_valid)

        if vol_long == 0 or vol_short == 0:
            return 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(long_valid, neg_short)[0, 1]
        if not np.isfinite(corr):
            return 0.0

        beta = corr * vol_short / vol_long
        return beta

    def build_hedged_portfolio_pnls(weight_matrix, name):
        """Build beta-hedged PnL portfolio using inception-weighted aggregation.

        Beta = cor(long, -short) * vol(short) / vol(long)
        Hedged = short + beta * long

        Returns:
            (hedged_pnl, long_pnl, short_pnl, beta)
        """
        # Long leg (positive weights only)
        long_weights = np.maximum(weight_matrix, 0.0)
        long_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, pnl, dte, have_dte, grid_size, n_assets,
            long_weights, asset_starts, asset_counts
        )
        long_pnl = np.nansum(long_matrix, axis=1)

        # Short leg (negative weights only)
        short_weights = np.minimum(weight_matrix, 0.0)
        short_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, pnl, dte, have_dte, grid_size, n_assets,
            short_weights, asset_starts, asset_counts
        )
        short_pnl = np.nansum(short_matrix, axis=1)

        # Compute beta and hedged portfolio
        beta = compute_hedge_beta(long_pnl, short_pnl)
        hedged_pnl = short_pnl + beta * long_pnl

        return hedged_pnl, long_pnl, short_pnl, beta

    def build_hedged_portfolio_npnls(weight_matrix, name):
        """Build beta-hedged npnl portfolio using inception-weighted aggregation.

        Beta = cor(long, -short) * vol(short) / vol(long)
        Hedged = short + beta * long

        Returns:
            (hedged_npnl, long_npnl, short_npnl, beta)
        """
        # Long leg (positive weights only)
        long_weights = np.maximum(weight_matrix, 0.0)
        long_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, npnl_vals, dte, have_dte, grid_size, n_assets,
            long_weights, asset_starts, asset_counts
        )
        long_npnl = np.nansum(long_matrix, axis=1)

        # Short leg (negative weights only)
        short_weights = np.minimum(weight_matrix, 0.0)
        short_matrix = _aggregate_inception_weighted_parallel(
            out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
            d0, npnl_vals, dte, have_dte, grid_size, n_assets,
            short_weights, asset_starts, asset_counts
        )
        short_npnl = np.nansum(short_matrix, axis=1)

        # Compute beta and hedged portfolio
        beta = compute_hedge_beta(long_npnl, short_npnl)
        hedged_npnl = short_npnl + beta * long_npnl

        return hedged_npnl, long_npnl, short_npnl, beta

    # ========================================================================
    # Portfolio 1: Short-only (-1/n_assets everywhere)
    # ========================================================================
    print()
    print("=" * 70)
    print("SHORT-ONLY PORTFOLIO (-1/n_assets weights)")
    print("=" * 70)

    # Create short-only weight matrix: -1/n_assets everywhere
    print("Building short-only weight matrix...", end="", flush=True)
    t0 = time.perf_counter()
    short_only_weights = np.full((grid_size, n_assets), -1.0 / n_assets, dtype=np.float64)
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Build PnL
    print("Building short-only PnL...", end="", flush=True)
    t0 = time.perf_counter()
    short_only_pnl, short_only_long_pnl, short_only_short_pnl = build_portfolio_pnls(short_only_weights, "short-only")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Build npnl
    print("Building short-only npnl...", end="", flush=True)
    t0 = time.perf_counter()
    short_only_npnl, short_only_long_npnl, short_only_short_npnl = build_portfolio_npnls(short_only_weights, "short-only")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # Display tables
    print_portfolio_table("SHORT-ONLY PnL", short_only_pnl, short_only_long_pnl, short_only_short_pnl)
    print_portfolio_table("SHORT-ONLY npnl (vol-normalized)", short_only_npnl, short_only_long_npnl, short_only_short_npnl)

    # ========================================================================
    # Verification: Short-only using grid-level pnl with -1/vol weights
    # To verify: npnl * (-1/n) ≈ pnl * (-1/vol) / sumabs(-1/vol)
    # Uses grid-level matrices directly, not straddle aggregation
    # Small difference expected: sum(pnl/vol) ≠ sum(pnl)/sum(vol)
    # ========================================================================
    print()
    print("Building verification: grid-level pnl with -1/vol weights...", end="", flush=True)
    t0 = time.perf_counter()
    # Create per-day vol weights: -1/vol_matrix, then sumabs normalize per row
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_weights_grid = -1.0 / vol_matrix
        vol_weights_grid[~np.isfinite(vol_weights_grid)] = np.nan
    vol_weights_norm = sumabs_norm(vol_weights_grid)
    # Apply to pnl_matrix directly at grid level
    verify_weighted = pnl_matrix * vol_weights_norm
    verify_pnl = np.nansum(verify_weighted, axis=1)
    # Short leg (negative weights only)
    short_mask = vol_weights_norm < 0
    verify_short_weighted = np.where(short_mask, pnl_matrix * vol_weights_norm, 0.0)
    verify_short_pnl = np.nansum(verify_short_weighted, axis=1)
    verify_long_pnl = verify_pnl - verify_short_pnl
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    # ========================================================================
    # Portfolio 2: MAC 365-90 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("MAC 365-90 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building mac_365_90 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac_pnl, mac_long_pnl, mac_short_pnl = build_portfolio_pnls(mac_365_90, "mac_365_90")
    mac_npnl, mac_long_npnl, mac_short_npnl = build_portfolio_npnls(mac_365_90, "mac_365_90")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("MAC 365-90 PnL (unhedged)", mac_pnl, mac_long_pnl, mac_short_pnl)
    print_portfolio_table("MAC 365-90 npnl (unhedged)", mac_npnl, mac_long_npnl, mac_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building mac_365_90 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac_h_pnl, mac_h_long_pnl, mac_h_short_pnl, mac_beta_pnl = build_hedged_portfolio_pnls(mac_365_90, "mac_hedged")
    mac_h_npnl, mac_h_long_npnl, mac_h_short_npnl, mac_beta_npnl = build_hedged_portfolio_npnls(mac_365_90, "mac_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {mac_beta_pnl:.4f}, Beta (npnl): {mac_beta_npnl:.4f}")

    print_portfolio_table("MAC 365-90 PnL (hedged)", mac_h_pnl, mac_h_long_pnl, mac_h_short_pnl)
    print_portfolio_table("MAC 365-90 npnl (hedged)", mac_h_npnl, mac_h_long_npnl, mac_h_short_npnl)

    # ========================================================================
    # Portfolio 2b: MAC 180-30 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("MAC 180-30 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building mac_180_30 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac180_30_pnl, mac180_30_long_pnl, mac180_30_short_pnl = build_portfolio_pnls(mac_180_30, "mac_180_30")
    mac180_30_npnl, mac180_30_long_npnl, mac180_30_short_npnl = build_portfolio_npnls(mac_180_30, "mac_180_30")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("MAC 180-30 PnL (unhedged)", mac180_30_pnl, mac180_30_long_pnl, mac180_30_short_pnl)
    print_portfolio_table("MAC 180-30 npnl (unhedged)", mac180_30_npnl, mac180_30_long_npnl, mac180_30_short_npnl)

    # Hedged
    print("Building mac_180_30 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac180_30_h_pnl, mac180_30_h_long_pnl, mac180_30_h_short_pnl, mac180_30_beta_pnl = build_hedged_portfolio_pnls(mac_180_30, "mac_180_30_hedged")
    mac180_30_h_npnl, mac180_30_h_long_npnl, mac180_30_h_short_npnl, mac180_30_beta_npnl = build_hedged_portfolio_npnls(mac_180_30, "mac_180_30_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {mac180_30_beta_pnl:.4f}, Beta (npnl): {mac180_30_beta_npnl:.4f}")

    print_portfolio_table("MAC 180-30 PnL (hedged)", mac180_30_h_pnl, mac180_30_h_long_pnl, mac180_30_h_short_pnl)
    print_portfolio_table("MAC 180-30 npnl (hedged)", mac180_30_h_npnl, mac180_30_h_long_npnl, mac180_30_h_short_npnl)

    # ========================================================================
    # Portfolio 2c: MAC 180-10 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("MAC 180-10 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building mac_180_10 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac180_10_pnl, mac180_10_long_pnl, mac180_10_short_pnl = build_portfolio_pnls(mac_180_10, "mac_180_10")
    mac180_10_npnl, mac180_10_long_npnl, mac180_10_short_npnl = build_portfolio_npnls(mac_180_10, "mac_180_10")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("MAC 180-10 PnL (unhedged)", mac180_10_pnl, mac180_10_long_pnl, mac180_10_short_pnl)
    print_portfolio_table("MAC 180-10 npnl (unhedged)", mac180_10_npnl, mac180_10_long_npnl, mac180_10_short_npnl)

    # Hedged
    print("Building mac_180_10 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    mac180_10_h_pnl, mac180_10_h_long_pnl, mac180_10_h_short_pnl, mac180_10_beta_pnl = build_hedged_portfolio_pnls(mac_180_10, "mac_180_10_hedged")
    mac180_10_h_npnl, mac180_10_h_long_npnl, mac180_10_h_short_npnl, mac180_10_beta_npnl = build_hedged_portfolio_npnls(mac_180_10, "mac_180_10_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {mac180_10_beta_pnl:.4f}, Beta (npnl): {mac180_10_beta_npnl:.4f}")

    print_portfolio_table("MAC 180-10 PnL (hedged)", mac180_10_h_pnl, mac180_10_h_long_pnl, mac180_10_h_short_pnl)
    print_portfolio_table("MAC 180-10 npnl (hedged)", mac180_10_h_npnl, mac180_10_h_long_npnl, mac180_10_h_short_npnl)

    # ========================================================================
    # Portfolio 3: SHARPE 365 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("SHARPE 365 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building sharpe_365 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    sharpe_pnl, sharpe_long_pnl, sharpe_short_pnl = build_portfolio_pnls(sharpe_365, "sharpe_365")
    sharpe_npnl, sharpe_long_npnl, sharpe_short_npnl = build_portfolio_npnls(sharpe_365, "sharpe_365")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("SHARPE 365 PnL (unhedged)", sharpe_pnl, sharpe_long_pnl, sharpe_short_pnl)
    print_portfolio_table("SHARPE 365 npnl (unhedged)", sharpe_npnl, sharpe_long_npnl, sharpe_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building sharpe_365 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    sharpe_h_pnl, sharpe_h_long_pnl, sharpe_h_short_pnl, sharpe_beta_pnl = build_hedged_portfolio_pnls(sharpe_365, "sharpe_hedged")
    sharpe_h_npnl, sharpe_h_long_npnl, sharpe_h_short_npnl, sharpe_beta_npnl = build_hedged_portfolio_npnls(sharpe_365, "sharpe_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {sharpe_beta_pnl:.4f}, Beta (npnl): {sharpe_beta_npnl:.4f}")

    print_portfolio_table("SHARPE 365 PnL (hedged)", sharpe_h_pnl, sharpe_h_long_pnl, sharpe_h_short_pnl)
    print_portfolio_table("SHARPE 365 npnl (hedged)", sharpe_h_npnl, sharpe_h_long_npnl, sharpe_h_short_npnl)

    # ========================================================================
    # Portfolio 4: SIGN 365 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("SIGN 365 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building sign_365 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    sign_pnl, sign_long_pnl, sign_short_pnl = build_portfolio_pnls(sign_365, "sign_365")
    sign_npnl, sign_long_npnl, sign_short_npnl = build_portfolio_npnls(sign_365, "sign_365")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("SIGN 365 PnL (unhedged)", sign_pnl, sign_long_pnl, sign_short_pnl)
    print_portfolio_table("SIGN 365 npnl (unhedged)", sign_npnl, sign_long_npnl, sign_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building sign_365 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    sign_h_pnl, sign_h_long_pnl, sign_h_short_pnl, sign_beta_pnl = build_hedged_portfolio_pnls(sign_365, "sign_hedged")
    sign_h_npnl, sign_h_long_npnl, sign_h_short_npnl, sign_beta_npnl = build_hedged_portfolio_npnls(sign_365, "sign_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {sign_beta_pnl:.4f}, Beta (npnl): {sign_beta_npnl:.4f}")

    print_portfolio_table("SIGN 365 PnL (hedged)", sign_h_pnl, sign_h_long_pnl, sign_h_short_pnl)
    print_portfolio_table("SIGN 365 npnl (hedged)", sign_h_npnl, sign_h_long_npnl, sign_h_short_npnl)

    # ========================================================================
    # Portfolio 5: RANGE 365 (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("RANGE 365 PORTFOLIO")
    print("=" * 70)

    # Unhedged
    print("Building range_365 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    range_pnl, range_long_pnl, range_short_pnl = build_portfolio_pnls(range_365, "range_365")
    range_npnl, range_long_npnl, range_short_npnl = build_portfolio_npnls(range_365, "range_365")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("RANGE 365 PnL (unhedged)", range_pnl, range_long_pnl, range_short_pnl)
    print_portfolio_table("RANGE 365 npnl (unhedged)", range_npnl, range_long_npnl, range_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building range_365 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    range_h_pnl, range_h_long_pnl, range_h_short_pnl, range_beta_pnl = build_hedged_portfolio_pnls(range_365, "range_hedged")
    range_h_npnl, range_h_long_npnl, range_h_short_npnl, range_beta_npnl = build_hedged_portfolio_npnls(range_365, "range_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {range_beta_pnl:.4f}, Beta (npnl): {range_beta_npnl:.4f}")

    print_portfolio_table("RANGE 365 PnL (hedged)", range_h_pnl, range_h_long_pnl, range_h_short_pnl)
    print_portfolio_table("RANGE 365 npnl (hedged)", range_h_npnl, range_h_long_npnl, range_h_short_npnl)

    # ========================================================================
    # Portfolio 6: COMBINED (mac_365_90 + sharpe_365 + sign_365 + range_365) (unhedged and hedged)
    # ========================================================================
    print()
    print("=" * 70)
    print("COMBINED PORTFOLIO (mac_365_90 + sharpe_365 + sign_365 + range_365)")
    print("=" * 70)

    # Unhedged
    print("Building combined unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb_pnl, comb_long_pnl, comb_short_pnl = build_portfolio_pnls(combined_signal, "combined")
    comb_npnl, comb_long_npnl, comb_short_npnl = build_portfolio_npnls(combined_signal, "combined")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("COMBINED PnL (unhedged)", comb_pnl, comb_long_pnl, comb_short_pnl)
    print_portfolio_table("COMBINED npnl (unhedged)", comb_npnl, comb_long_npnl, comb_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building combined beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb_h_pnl, comb_h_long_pnl, comb_h_short_pnl, comb_beta_pnl = build_hedged_portfolio_pnls(combined_signal, "combined_hedged")
    comb_h_npnl, comb_h_long_npnl, comb_h_short_npnl, comb_beta_npnl = build_hedged_portfolio_npnls(combined_signal, "combined_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {comb_beta_pnl:.4f}, Beta (npnl): {comb_beta_npnl:.4f}")

    print_portfolio_table("COMBINED PnL (hedged)", comb_h_pnl, comb_h_long_pnl, comb_h_short_pnl)
    print_portfolio_table("COMBINED npnl (hedged)", comb_h_npnl, comb_h_long_npnl, comb_h_short_npnl)

    # ========================================================================
    # Portfolio 7: COMBINED2 (sum ranks -> rank -> sumabs -> lag)
    # ========================================================================
    print()
    print("=" * 70)
    print("COMBINED2 PORTFOLIO (sum ranks -> rank -> sumabs -> lag)")
    print("=" * 70)

    # Unhedged
    print("Building combined2 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb2_pnl, comb2_long_pnl, comb2_short_pnl = build_portfolio_pnls(combined2_signal, "combined2")
    comb2_npnl, comb2_long_npnl, comb2_short_npnl = build_portfolio_npnls(combined2_signal, "combined2")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("COMBINED2 PnL (unhedged)", comb2_pnl, comb2_long_pnl, comb2_short_pnl)
    print_portfolio_table("COMBINED2 npnl (unhedged)", comb2_npnl, comb2_long_npnl, comb2_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building combined2 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb2_h_pnl, comb2_h_long_pnl, comb2_h_short_pnl, comb2_beta_pnl = build_hedged_portfolio_pnls(combined2_signal, "combined2_hedged")
    comb2_h_npnl, comb2_h_long_npnl, comb2_h_short_npnl, comb2_beta_npnl = build_hedged_portfolio_npnls(combined2_signal, "combined2_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {comb2_beta_pnl:.4f}, Beta (npnl): {comb2_beta_npnl:.4f}")

    print_portfolio_table("COMBINED2 PnL (hedged)", comb2_h_pnl, comb2_h_long_pnl, comb2_h_short_pnl)
    print_portfolio_table("COMBINED2 npnl (hedged)", comb2_h_npnl, comb2_h_long_npnl, comb2_h_short_npnl)

    # ========================================================================
    # Portfolio 8: COMBINED3 (multiplicative combination)
    # ========================================================================
    print()
    print("=" * 70)
    print("COMBINED3 PORTFOLIO (multiplicative: norm(s1)*norm(s2)*norm(s3)*norm(s4))")
    print("=" * 70)

    # Unhedged
    print("Building combined3 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb3_pnl, comb3_long_pnl, comb3_short_pnl = build_portfolio_pnls(combined3_signal, "combined3")
    comb3_npnl, comb3_long_npnl, comb3_short_npnl = build_portfolio_npnls(combined3_signal, "combined3")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("COMBINED3 PnL (unhedged)", comb3_pnl, comb3_long_pnl, comb3_short_pnl)
    print_portfolio_table("COMBINED3 npnl (unhedged)", comb3_npnl, comb3_long_npnl, comb3_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building combined3 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb3_h_pnl, comb3_h_long_pnl, comb3_h_short_pnl, comb3_beta_pnl = build_hedged_portfolio_pnls(combined3_signal, "combined3_hedged")
    comb3_h_npnl, comb3_h_long_npnl, comb3_h_short_npnl, comb3_beta_npnl = build_hedged_portfolio_npnls(combined3_signal, "combined3_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {comb3_beta_pnl:.4f}, Beta (npnl): {comb3_beta_npnl:.4f}")

    print_portfolio_table("COMBINED3 PnL (hedged)", comb3_h_pnl, comb3_h_long_pnl, comb3_h_short_pnl)
    print_portfolio_table("COMBINED3 npnl (hedged)", comb3_h_npnl, comb3_h_long_npnl, comb3_h_short_npnl)

    # ========================================================================
    # Portfolio 9: COMBINED4 (MAC 180-30 + MAC 180-10 + SHARPE 365)
    # ========================================================================
    print()
    print("=" * 70)
    print("COMBINED4 PORTFOLIO (mac_180_30 + mac_180_10 + sharpe_365)")
    print("=" * 70)

    # Unhedged
    print("Building combined4 unhedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb4_pnl, comb4_long_pnl, comb4_short_pnl = build_portfolio_pnls(combined4_signal, "combined4")
    comb4_npnl, comb4_long_npnl, comb4_short_npnl = build_portfolio_npnls(combined4_signal, "combined4")
    print(f" {_fmt_time(time.perf_counter() - t0)}")

    print_portfolio_table("COMBINED4 PnL (unhedged)", comb4_pnl, comb4_long_pnl, comb4_short_pnl)
    print_portfolio_table("COMBINED4 npnl (unhedged)", comb4_npnl, comb4_long_npnl, comb4_short_npnl)

    # Hedged (beta = cor(long, -short) * vol(short) / vol(long))
    print("Building combined4 beta-hedged PnL/npnl...", end="", flush=True)
    t0 = time.perf_counter()
    comb4_h_pnl, comb4_h_long_pnl, comb4_h_short_pnl, comb4_beta_pnl = build_hedged_portfolio_pnls(combined4_signal, "combined4_hedged")
    comb4_h_npnl, comb4_h_long_npnl, comb4_h_short_npnl, comb4_beta_npnl = build_hedged_portfolio_npnls(combined4_signal, "combined4_hedged")
    print(f" {_fmt_time(time.perf_counter() - t0)}")
    print(f"  Beta (PnL): {comb4_beta_pnl:.4f}, Beta (npnl): {comb4_beta_npnl:.4f}")

    print_portfolio_table("COMBINED4 PnL (hedged)", comb4_h_pnl, comb4_h_long_pnl, comb4_h_short_pnl)
    print_portfolio_table("COMBINED4 npnl (hedged)", comb4_h_npnl, comb4_h_long_npnl, comb4_h_short_npnl)

    # ========================================================================
    # Final Summary
    # ========================================================================
    pnl_non_zero = np.sum(pnl_matrix != 0.0)
    vol_non_zero = np.sum(vol_matrix != 0.0)
    total_cells = pnl_matrix.size

    print()
    print("=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Matrix shape: {pnl_matrix.shape[0]:,} days × {pnl_matrix.shape[1]} assets")
    print(f"PnL non-zero: {pnl_non_zero:,} / {total_cells:,} ({100*pnl_non_zero/total_cells:.1f}%)")
    print(f"Vol non-zero: {vol_non_zero:,} / {total_cells:,} ({100*vol_non_zero/total_cells:.1f}%)")
    print(f"Total PnL sum: {np.nansum(pnl_matrix):.6f}")
    print(f"Total Vol sum: {np.nansum(vol_matrix):.6f}")

    print()
    print("All Computation Timings (parallel):")
    print(f"  Parse weights:   {_fmt_time(dt_weights)}")
    print(f"  Prepare arrays:  {_fmt_time(dt_prep)}")
    print(f"  PnL matrix:      {_fmt_time(dt_pnl_kernel)}")
    print(f"  Vol matrix:      {_fmt_time(dt_vol_kernel)}")
    print(f"  Signal pipeline: {_fmt_time(dt_signal_total)}")
    print("-" * 50)
    print(f"  Total:           {_fmt_time(time.perf_counter() - total_start)}")

    # ========================================================================
    # Portfolio Comparison Table (last)
    # ========================================================================
    print()
    print("=" * 110)
    print("PORTFOLIO COMPARISON")
    print("=" * 110)
    print(f"{'Signal':<25} {'Sharpe(npnl)':>13} {'Long(npnl)':>11} {'Short(npnl)':>12} {'L/S Corr':>9}")
    print("-" * 110)

    portfolios = [
        ("Short-only", short_only_pnl, short_only_npnl, short_only_long_pnl, short_only_short_pnl, short_only_long_npnl, short_only_short_npnl),
        ("Short-only (vol-wgt)", None, verify_pnl, None, None, verify_long_pnl, verify_short_pnl),
        ("MAC 365-90 (unhedged)", mac_pnl, mac_npnl, mac_long_pnl, mac_short_pnl, mac_long_npnl, mac_short_npnl),
        ("MAC 365-90 (hedged)", mac_h_pnl, mac_h_npnl, mac_h_long_pnl, mac_h_short_pnl, mac_h_long_npnl, mac_h_short_npnl),
        ("MAC 180-30 (unhedged)", mac180_30_pnl, mac180_30_npnl, mac180_30_long_pnl, mac180_30_short_pnl, mac180_30_long_npnl, mac180_30_short_npnl),
        ("MAC 180-30 (hedged)", mac180_30_h_pnl, mac180_30_h_npnl, mac180_30_h_long_pnl, mac180_30_h_short_pnl, mac180_30_h_long_npnl, mac180_30_h_short_npnl),
        ("MAC 180-10 (unhedged)", mac180_10_pnl, mac180_10_npnl, mac180_10_long_pnl, mac180_10_short_pnl, mac180_10_long_npnl, mac180_10_short_npnl),
        ("MAC 180-10 (hedged)", mac180_10_h_pnl, mac180_10_h_npnl, mac180_10_h_long_pnl, mac180_10_h_short_pnl, mac180_10_h_long_npnl, mac180_10_h_short_npnl),
        ("SHARPE 365 (unhedged)", sharpe_pnl, sharpe_npnl, sharpe_long_pnl, sharpe_short_pnl, sharpe_long_npnl, sharpe_short_npnl),
        ("SHARPE 365 (hedged)", sharpe_h_pnl, sharpe_h_npnl, sharpe_h_long_pnl, sharpe_h_short_pnl, sharpe_h_long_npnl, sharpe_h_short_npnl),
        ("SIGN 365 (unhedged)", sign_pnl, sign_npnl, sign_long_pnl, sign_short_pnl, sign_long_npnl, sign_short_npnl),
        ("SIGN 365 (hedged)", sign_h_pnl, sign_h_npnl, sign_h_long_pnl, sign_h_short_pnl, sign_h_long_npnl, sign_h_short_npnl),
        ("RANGE 365 (unhedged)", range_pnl, range_npnl, range_long_pnl, range_short_pnl, range_long_npnl, range_short_npnl),
        ("RANGE 365 (hedged)", range_h_pnl, range_h_npnl, range_h_long_pnl, range_h_short_pnl, range_h_long_npnl, range_h_short_npnl),
        ("COMBINED (unhedged)", comb_pnl, comb_npnl, comb_long_pnl, comb_short_pnl, comb_long_npnl, comb_short_npnl),
        ("COMBINED (hedged)", comb_h_pnl, comb_h_npnl, comb_h_long_pnl, comb_h_short_pnl, comb_h_long_npnl, comb_h_short_npnl),
        ("COMBINED2 (unhedged)", comb2_pnl, comb2_npnl, comb2_long_pnl, comb2_short_pnl, comb2_long_npnl, comb2_short_npnl),
        ("COMBINED2 (hedged)", comb2_h_pnl, comb2_h_npnl, comb2_h_long_pnl, comb2_h_short_pnl, comb2_h_long_npnl, comb2_h_short_npnl),
        ("COMBINED3 (unhedged)", comb3_pnl, comb3_npnl, comb3_long_pnl, comb3_short_pnl, comb3_long_npnl, comb3_short_npnl),
        ("COMBINED3 (hedged)", comb3_h_pnl, comb3_h_npnl, comb3_h_long_pnl, comb3_h_short_pnl, comb3_h_long_npnl, comb3_h_short_npnl),
        ("COMBINED4 (unhedged)", comb4_pnl, comb4_npnl, comb4_long_pnl, comb4_short_pnl, comb4_long_npnl, comb4_short_npnl),
        ("COMBINED4 (hedged)", comb4_h_pnl, comb4_h_npnl, comb4_h_long_pnl, comb4_h_short_pnl, comb4_h_long_npnl, comb4_h_short_npnl),
    ]

    for name, pnl_arr, npnl_arr, long_pnl_arr, short_pnl_arr, long_npnl_arr, short_npnl_arr in portfolios:
        npnl_stats = compute_portfolio_stats(npnl_arr)
        long_npnl_stats = compute_portfolio_stats(long_npnl_arr)
        short_npnl_stats = compute_portfolio_stats(short_npnl_arr)
        corr_npnl = long_short_corr(long_npnl_arr, short_npnl_arr)
        print(f"{name:<25} {npnl_stats['sharpe']:>13.4f} {long_npnl_stats['sharpe']:>11.4f} {short_npnl_stats['sharpe']:>12.4f} {corr_npnl:>9.4f}")

    print("=" * 110)

    # ========================================================================
    # Portfolio Comparison Table (PNL) - using raw pnl with inception-weighted aggregation
    # ========================================================================
    print()
    print("=" * 110)
    print("PORTFOLIO COMPARISON (PNL)")
    print("=" * 110)
    print(f"{'Signal':<25} {'Sharpe(pnl)':>12} {'Long(pnl)':>10} {'Short(pnl)':>11} {'L/S Corr':>9}")
    print("-" * 110)

    # Short-only with vol-inverse weights: sumabs_norm(-1/vol_matrix)
    # Uses inception-weighted aggregation with raw pnl vector
    # vol_weights_norm is already computed above as sumabs_norm(-1/vol_matrix)
    pnl_volinv_matrix = _aggregate_inception_weighted_parallel(
        out0s_sorted, lens_sorted, starts_sorted, asset_ids[sort_idx], weights_sorted,
        d0, pnl, dte, have_dte, grid_size, n_assets,
        vol_weights_norm, asset_starts, asset_counts
    )
    pnl_short_volinv = np.nansum(pnl_volinv_matrix, axis=1)
    # All weights in vol_weights_norm are negative (short), so short = total, long = 0
    pnl_short_volinv_short = pnl_short_volinv
    pnl_short_volinv_long = np.zeros_like(pnl_short_volinv)

    # Combined5: MAC 180-10 + SHARPE 365
    combined5_signal = combine_signals(mac_180_10, sharpe_365)
    comb5_pnl, comb5_long_pnl, comb5_short_pnl = build_portfolio_pnls(combined5_signal, "combined5")
    comb5_h_pnl, comb5_h_long_pnl, comb5_h_short_pnl, comb5_beta = build_hedged_portfolio_pnls(combined5_signal, "combined5_hedged")

    pnl_portfolios = [
        ("Short-only (vol-inv)", pnl_short_volinv, pnl_short_volinv_long, pnl_short_volinv_short),
        ("MAC 180-10", mac180_10_pnl, mac180_10_long_pnl, mac180_10_short_pnl),
        ("MAC 180-10 (hedged)", mac180_10_h_pnl, mac180_10_h_long_pnl, mac180_10_h_short_pnl),
        ("SHARPE 365", sharpe_pnl, sharpe_long_pnl, sharpe_short_pnl),
        ("SHARPE 365 (hedged)", sharpe_h_pnl, sharpe_h_long_pnl, sharpe_h_short_pnl),
        ("COMB5 (MAC+SHRP)", comb5_pnl, comb5_long_pnl, comb5_short_pnl),
        ("COMB5 (hedged)", comb5_h_pnl, comb5_h_long_pnl, comb5_h_short_pnl),
    ]

    for name, ptf_pnl, long_pnl, short_pnl in pnl_portfolios:
        ptf_stats = compute_portfolio_stats(ptf_pnl)
        long_stats = compute_portfolio_stats(long_pnl)
        short_stats = compute_portfolio_stats(short_pnl)
        corr = long_short_corr(long_pnl, short_pnl)
        print(f"{name:<25} {ptf_stats['sharpe']:>12.4f} {long_stats['sharpe']:>10.4f} {short_stats['sharpe']:>11.4f} {corr:>9.4f}")

    print("=" * 110)


if __name__ == "__main__":
    main()
