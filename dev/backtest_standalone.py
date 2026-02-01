# ============================================================================
# Imports
# ============================================================================
import numpy as np
from numba import njit, prange
import time
from pathlib import Path
from typing import Any
import yaml

# ============================================================================
# Configuration
# ============================================================================
USE_PARALLEL_MERGE = False  # Toggle: True=parallel, False=serial (test OpenMP overhead)
DEBUG = False  # Enable sanity assertions

# String dtype constants
TICKER_U = "U100"
FIELD_U = "U100"
FUT_MONTH_MAP_LEN = 12

print(f"USE_PARALLEL_MERGE {USE_PARALLEL_MERGE}")

# ============================================================================
# Numba kernels
# ============================================================================
@njit(parallel=True, cache=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Key-grouped merge: for each key, merge straddles with price series (parallel).

    3.55x faster than expand+binary search by avoiding:
    - 15.4M array allocations
    - Binary search per day
    - Random memory access patterns
    """
    n_groups = len(g_keys)
    n_blocks = len(px_block_of)

    for gi in prange(n_groups):
        key = g_keys[gi]
        if key < 0 or key >= n_blocks:
            continue

        b = px_block_of[key]
        if b < 0:
            continue

        ps = px_starts[b]
        pe = px_ends[b]

        # forward-only lower bound into price dates for this key
        pi0 = ps

        for si in range(g_starts[gi], g_ends[gi]):
            start = m_start_s[si]
            length = m_len_s[si]
            if length <= 0:
                continue
            out0 = m_out0_s[si]
            end = start + length

            # advance lower bound to first price >= start (never goes backward)
            while pi0 < pe and px_date[pi0] < start:
                pi0 += 1

            # scan from that point forward until end
            pi = pi0
            while pi < pe:
                d = px_date[pi]
                if d >= end:
                    break
                out[out0 + (d - start)] = px_value[pi]
                pi += 1

    return out

@njit(cache=True, parallel=True)
def compute_actions(
    # Per-straddle arrays (length S)
    month_out0: np.ndarray,        # int32[S] - output start index
    month_len: np.ndarray,         # int32[S] - days per straddle
    month_start_epoch: np.ndarray, # int32[S] - epoch of first day
    ntry_anchor: np.ndarray,       # int32[S] - entry anchor date
    xpry_anchor: np.ndarray,       # int32[S] - expiry anchor date
    ntrv_offsets: np.ndarray,      # int32[S] - entry offset (calendar days)
    ntry_month_end: np.ndarray,    # int32[S] - entry month end epoch
    xpry_month_end: np.ndarray,    # int32[S] - expiry month end epoch
    # Required flags (True if key exists)
    req_vol: np.ndarray,           # bool[S]
    req_hedge1: np.ndarray,        # bool[S]
    req_hedge2: np.ndarray,        # bool[S]
    req_hedge3: np.ndarray,        # bool[S]
    req_hedge4: np.ndarray,        # bool[S]
    # Daily price arrays (length N)
    vol: np.ndarray,               # float64[N]
    hedge1: np.ndarray,            # float64[N]
    hedge2: np.ndarray,            # float64[N]
    hedge3: np.ndarray,            # float64[N]
    hedge4: np.ndarray,            # float64[N]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find entry/expiry days and compute action array.

    Returns:
        action: int8[N] - action codes (0=none, 1=ntry, 2=xpry)
        ntry_offsets: int32[S] - offset within straddle for entry (-1 if not found)
        xpry_offsets: int32[S] - offset within straddle for expiry (-1 if not found)
    """
    n_straddles = len(month_out0)
    total_days = len(vol)  # derive from array length
    INVALID = 2147483647

    action = np.zeros(total_days, dtype=np.int8)
    ntry_offsets = np.full(n_straddles, -1, dtype=np.int32)
    xpry_offsets = np.full(n_straddles, -1, dtype=np.int32)

    for s in prange(n_straddles):
        out0 = month_out0[s]
        length = month_len[s]
        start_epoch = month_start_epoch[s]

        ntry_anc = ntry_anchor[s]
        xpry_anc = xpry_anchor[s]
        ntrv_off = ntrv_offsets[s]
        ntry_end = ntry_month_end[s]
        xpry_end = xpry_month_end[s]

        # Skip invalid anchors (use == for safety against overflow)
        if ntry_anc == INVALID or xpry_anc == INVALID:
            continue

        # Compute entry target (anchor + offset, clamped)
        ntry_target = ntry_anc + ntrv_off
        if ntry_target > ntry_end:
            ntry_target = ntry_end

        # Expiry target = anchor directly
        xpry_target = xpry_anc

        # Required flags for this straddle
        r_vol = req_vol[s]
        r_h1 = req_hedge1[s]
        r_h2 = req_hedge2[s]
        r_h3 = req_hedge3[s]
        r_h4 = req_hedge4[s]

        # --- Find entry day ---
        ntry_off = -1
        for i in range(length):
            idx = out0 + i
            d = start_epoch + i  # inline date computation

            if d > ntry_end:
                break
            if d < ntry_target:
                continue

            # Check validity: all required legs must be non-NaN
            if r_vol and np.isnan(vol[idx]):
                continue
            if r_h1 and np.isnan(hedge1[idx]):
                continue
            if r_h2 and np.isnan(hedge2[idx]):
                continue
            if r_h3 and np.isnan(hedge3[idx]):
                continue
            if r_h4 and np.isnan(hedge4[idx]):
                continue

            ntry_off = i
            break

        # Fallback: last good day in entry month
        if ntry_off < 0:
            for i in range(length - 1, -1, -1):
                idx = out0 + i
                d = start_epoch + i

                if d > ntry_end:
                    continue

                if r_vol and np.isnan(vol[idx]):
                    continue
                if r_h1 and np.isnan(hedge1[idx]):
                    continue
                if r_h2 and np.isnan(hedge2[idx]):
                    continue
                if r_h3 and np.isnan(hedge3[idx]):
                    continue
                if r_h4 and np.isnan(hedge4[idx]):
                    continue

                ntry_off = i
                break

        # --- Find expiry day (must be >= ntry_off to enforce xpry >= ntry) ---
        xpry_off = -1
        start_i = ntry_off if ntry_off >= 0 else 0
        for i in range(start_i, length):
            idx = out0 + i
            d = start_epoch + i

            if d > xpry_end:
                break
            if d < xpry_target:
                continue

            if r_vol and np.isnan(vol[idx]):
                continue
            if r_h1 and np.isnan(hedge1[idx]):
                continue
            if r_h2 and np.isnan(hedge2[idx]):
                continue
            if r_h3 and np.isnan(hedge3[idx]):
                continue
            if r_h4 and np.isnan(hedge4[idx]):
                continue

            xpry_off = i
            break

        # Skip if either not found
        if ntry_off < 0 or xpry_off < 0:
            continue

        ntry_offsets[s] = ntry_off
        xpry_offsets[s] = xpry_off
        action[out0 + ntry_off] = 1  # ntry
        action[out0 + xpry_off] = 2  # xpry

    return action, ntry_offsets, xpry_offsets


# ============================================================================
# Phase 4-6 kernels: Roll-forward, Model, PnL
# ============================================================================

@njit(cache=True)
def _norm_cdf_approx(x: float) -> float:
    """Fast approximation of standard normal CDF.

    Uses the error function approximation. Accurate to ~6 decimal places.
    This is equivalent to scipy.special.ndtr but works in Numba.
    """
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


@njit(cache=True, parallel=True)
def roll_forward_inplace(
    values: np.ndarray,        # float64[N] - modified in place
    month_out0: np.ndarray,    # int32[S]
    ntry_offsets: np.ndarray,  # int32[S]
    xpry_offsets: np.ndarray,  # int32[S]
):
    """Fill NaN with last valid value within [ntry..xpry] range (in-place)."""
    for s in prange(len(month_out0)):
        ntry_off = ntry_offsets[s]
        xpry_off = xpry_offsets[s]
        if ntry_off < 0 or xpry_off < 0:
            continue

        out0 = month_out0[s]
        last_valid = values[out0 + ntry_off]

        for i in range(ntry_off, xpry_off + 1):
            idx = out0 + i
            if np.isnan(values[idx]):
                values[idx] = last_valid
            else:
                last_valid = values[idx]


@njit(cache=True, parallel=True)
def fill_strikes_and_dte(
    month_out0: np.ndarray,
    month_start_epoch: np.ndarray,
    ntry_offsets: np.ndarray,
    xpry_offsets: np.ndarray,
    hedge: np.ndarray,           # float64[N] - primary hedge (for strike)
    # Outputs
    strike: np.ndarray,          # float64[N] - strike price
    days_to_expiry: np.ndarray,  # int32[N]
):
    """Capture strike at entry and compute days-to-expiry inline."""
    for s in prange(len(month_out0)):
        ntry_off = ntry_offsets[s]
        xpry_off = xpry_offsets[s]
        if ntry_off < 0 or xpry_off < 0:
            continue

        out0 = month_out0[s]
        start_epoch = month_start_epoch[s]

        # Strike = hedge at entry
        strike_val = hedge[out0 + ntry_off]

        # Expiry date (inline computation)
        expiry_date = start_epoch + xpry_off

        for i in range(ntry_off, xpry_off + 1):
            idx = out0 + i
            strike[idx] = strike_val
            days_to_expiry[idx] = expiry_date - (start_epoch + i)


@njit(cache=True, parallel=True)
def build_active_mask(
    month_out0: np.ndarray,
    ntry_offsets: np.ndarray,
    xpry_offsets: np.ndarray,
    total_days: int,
) -> np.ndarray:
    """Build mask: True for days in [ntry_off, xpry_off] range."""
    mask = np.zeros(total_days, dtype=np.bool_)
    for s in prange(len(month_out0)):
        ntry_off = ntry_offsets[s]
        xpry_off = xpry_offsets[s]
        if ntry_off < 0 or xpry_off < 0:
            continue
        out0 = month_out0[s]
        for i in range(ntry_off, xpry_off + 1):
            mask[out0 + i] = True
    return mask


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
        N_d1 = 2.0 * _norm_cdf_approx(d1)
        N_d2 = 2.0 * _norm_cdf_approx(d2)

        # MV = S * N_d1 - X * N_d2 + X - S (straddle formula)
        mv_val = S * N_d1 - X * N_d2 + X - S
        mv[i] = mv_val / X  # Normalize by strike
        delta[i] = N_d1 - 1.0

    return mv, delta


@njit(cache=True, parallel=True)
def compute_pnl_batch(
    mv: np.ndarray,
    delta: np.ndarray,
    hedge: np.ndarray,
    strike: np.ndarray,
    straddle_starts: np.ndarray,
    ntry_offsets: np.ndarray,
    xpry_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PnL for all straddles in parallel.

    PnL formulas:
        opnl[i] = mv[i] - mv[i-1]  (option PnL)
        hpnl[i] = -delta[i-1] * (hedge[i] - hedge[i-1]) / strike  (hedge PnL)
        pnl[i] = opnl[i] + hpnl[i]  (total PnL)
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
def _merge_per_key_serial(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                          px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Key-grouped merge (serial version for comparison with parallel)."""
    n_groups = len(g_keys)
    n_blocks = len(px_block_of)

    for gi in range(n_groups):  # range instead of prange
        key = g_keys[gi]
        if key < 0 or key >= n_blocks:
            continue

        b = px_block_of[key]
        if b < 0:
            continue

        ps = px_starts[b]
        pe = px_ends[b]
        pi0 = ps

        for si in range(g_starts[gi], g_ends[gi]):
            start = m_start_s[si]
            length = m_len_s[si]
            if length <= 0:
                continue
            out0 = m_out0_s[si]
            end = start + length

            while pi0 < pe and px_date[pi0] < start:
                pi0 += 1

            pi = pi0
            while pi < pe:
                d = px_date[pi]
                if d >= end:
                    break
                out[out0 + (d - start)] = px_value[pi]
                pi += 1

    return out

# ============================================================================
# Helpers (pure Python / NumPy)
# ============================================================================
def map_to_id_searchsorted(query_u, sorted_u, order_arr):
    """Vectorized string->ID mapping using searchsorted.

    Args:
        query_u: Query array, must already be 'U' dtype (caller converts once)
        sorted_u: Sorted reference array
        order_arr: Argsort permutation mapping sorted position -> original ID
    Returns:
        int32 array of IDs, -1 for missing
    """
    pos = np.searchsorted(sorted_u, query_u)
    pos_clipped = np.minimum(pos, len(sorted_u) - 1)
    valid = (pos < len(sorted_u)) & (sorted_u[pos_clipped] == query_u)
    return np.where(valid, order_arr[pos_clipped], -1).astype(np.int32)


def run_sweep_merge(use_parallel, g_keys, g_starts, g_ends,
                    m_start, m_len, m_out0,
                    px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Dispatch to parallel or serial merge kernel."""
    if use_parallel:
        return _merge_per_key(g_keys, g_starts, g_ends,
                              m_start, m_len, m_out0,
                              px_block_of, px_starts, px_ends,
                              px_date, px_value, out)
    else:
        return _merge_per_key_serial(g_keys, g_starts, g_ends,
                                     m_start, m_len, m_out0,
                                     px_block_of, px_starts, px_ends,
                                     px_date, px_value, out)


def sweep_leg_sparse(label: str, key_i32: np.ndarray,
                     month_start_epoch: np.ndarray,
                     month_len: np.ndarray,
                     month_out0: np.ndarray,
                     px_block_of, px_starts, px_ends, px_date, px_value,
                     total_days: int) -> np.ndarray:
    """Filter-first sweep: filter valid keys BEFORE sorting (crucial for sparse legs)."""
    print(label.ljust(20, "."), end="")
    t0 = time.perf_counter()

    max_valid_key = len(px_block_of) - 1
    # Filter-first: key in range AND has price data (px_block_of >= 0)
    valid = (key_i32 >= 0) & (key_i32 <= max_valid_key)
    valid_keys = key_i32[valid]
    valid[valid] &= (px_block_of[valid_keys] >= 0)  # Only keys with actual price data

    if not np.any(valid):
        out = np.full(total_days, np.nan, dtype=np.float64)
        t1 = time.perf_counter()
        print(f": {1e3*(t1-t0):0.3f}ms (all missing)")
        return out

    # Extract only valid rows (filter-first = sort only filtered subset)
    k = key_i32[valid].astype(np.int32)
    s = month_start_epoch[valid].astype(np.int32)
    ln = month_len[valid].astype(np.int32)
    out0 = month_out0[valid].astype(np.int32)

    # Sort the filtered subset
    order = np.lexsort((s, k))
    k, s, ln, out0 = k[order], s[order], ln[order], out0[order]

    # Group boundaries
    chg = np.flatnonzero(k[1:] != k[:-1]) + 1
    g_starts = np.r_[0, chg].astype(np.int32)
    g_ends = np.r_[chg, len(k)].astype(np.int32)
    g_keys = k[g_starts]

    out = np.full(total_days, np.nan, dtype=np.float64)
    out = run_sweep_merge(USE_PARALLEL_MERGE, g_keys, g_starts, g_ends,
                          s, ln, out0,
                          px_block_of, px_starts, px_ends,
                          px_date, px_value, out)
    t1 = time.perf_counter()

    if DEBUG:
        found = np.sum(~np.isnan(out))
        print(f": {1e3*(t1-t0):0.3f}ms ({found:,} found)")
    else:
        print(f": {1e3*(t1-t0):0.3f}ms")
    return out


# ============================================================================
# Main pipeline
# ============================================================================

script_start_time = time.perf_counter()

# ========================================================================
# Load YAML configuration
# ========================================================================
print("loading yaml".ljust(20, "."), end="")
start_time = time.perf_counter()
amt_resolved = str(Path("data/amt.yml").resolve())

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

with open(amt_resolved, "r") as f:
    run_options = yaml.load(f, Loader=Loader)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Processing amt: asset-level arrays
# ========================================================================
# asset-level arrays: length = n_assets (~189)
print("processing amt".ljust(20, "."), end="")
start_time = time.perf_counter()

amt = run_options.get("amt", {})
expiry_schedules = run_options.get("expiry_schedules")

amap: dict[str, dict[str, Any]] = {}
for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap") > 0:
            amap[underlying] = asset_data

anames = np.array(list(amap.keys()), dtype=np.dtypes.StringDType())
idx_map = dict(zip(list(amap.keys()), range(len(anames))))

# cache all straddle-related data into numpy arrays for vectorized access
nps = np.dtypes.StringDType()

# Extract asset attributes (separate list(map) calls are faster than single-pass for N=189)
hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_ticker = np.array(list(map(lambda a: amap[a]["Hedge"].get("Ticker", ""), anames)), dtype=nps)
hedge_field = np.array(list(map(lambda a: amap[a]["Hedge"].get("Field", ""), anames)), dtype=nps)
hedge_hedge = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge", ""), anames)), dtype=nps)
hedge_hedge1 = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge1", ""), anames)), dtype=nps)
hedge_ccy = np.array(list(map(lambda a: amap[a]["Hedge"].get("ccy", ""), anames)), dtype=nps)
hedge_tenor = np.array(list(map(lambda a: amap[a]["Hedge"].get("tenor", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)
vol_source = np.array(list(map(lambda a: amap[a]["Vol"].get("Source", ""), anames)), dtype=nps)
vol_ticker = np.array(list(map(lambda a: amap[a]["Vol"].get("Ticker", ""), anames)), dtype=nps)
vol_near = np.array(list(map(lambda a: amap[a]["Vol"].get("Near", ""), anames)), dtype=nps)
vol_far = np.array(list(map(lambda a: amap[a]["Vol"].get("Far", ""), anames)), dtype=nps)

# Convert hedge_source to integer codes for speed
hedge_sources, hedge_source_id = np.unique(hedge_source, return_inverse=True)
hs2id_map = dict(zip(hedge_sources, range(len(hedge_sources))))
HEDGE_FUT = hs2id_map["fut"]
hedge_source_id_fut = hedge_source_id == HEDGE_FUT
HEDGE_NONFUT = hs2id_map["nonfut"]
hedge_source_id_nonfut = hedge_source_id == HEDGE_NONFUT
HEDGE_CDS = hs2id_map["cds"]
hedge_source_id_cds = hedge_source_id == HEDGE_CDS
HEDGE_CALC = hs2id_map["calc"]
hedge_source_id_calc = hedge_source_id == HEDGE_CALC

# calc-type hedges are fixed concats
calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor
calc_hedge2 = hedge_ccy + "_fsw6m_" + hedge_tenor
calc_hedge3 = hedge_ccy + "_pva0m_" + hedge_tenor
calc_hedge4 = hedge_ccy + "_pva6m_" + hedge_tenor

# matrix of assets x months -> month_codes
hedge_fut_month_mtrx = hedge_fut_month_map.astype('S12').view('S1').reshape(-1, FUT_MONTH_MAP_LEN).astype('U1')
hedge_min_year_offset_int = hedge_min_year_offset.astype(np.int64)

# Convert vol_source to integer codes
vol_sources, vol_source_id = np.unique(vol_source, return_inverse=True)
vs2id_map = dict(zip(vol_sources, range(len(vol_sources))))
VOL_BBG_LMEVOL = vs2id_map["BBG_LMEVOL"]
vol_source_id_bbg_lmevol = vol_source_id == VOL_BBG_LMEVOL
VOL_BBG = vs2id_map["BBG"]
vol_source_id_bbg = vol_source_id == VOL_BBG
VOL_CV = vs2id_map["CV"]
vol_source_id_cv = vol_source_id == VOL_CV

# checksum of asset names
achk = np.array([np.sum(np.frombuffer(x.encode('ascii'), dtype=np.uint8)) for x in anames], dtype=np.int64)

# schedule count
aschcnt = np.array(list(map(
    lambda a: len(expiry_schedules[amap[a]["Options"]]),
    anames
)), dtype=np.int64)

# expanded count
easchcnt = np.repeat(aschcnt, aschcnt)
# expanded schedules
eastmp = np.concatenate(list(map(
    lambda a: np.array(expiry_schedules[amap[a]["Options"]], dtype="|U20"),
    anames
)), dtype="|U20")

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Schedule parsing: tokens -> per-field ids -> schedule_id_matrix
# ========================================================================
print("parse schedules".ljust(20, "."), end="")
start_time = time.perf_counter()

easchj = np.arange(np.sum(aschcnt)) - np.repeat(np.cumsum(aschcnt) - aschcnt, aschcnt)
easchi = np.repeat(np.arange(len(anames)), aschcnt)

# Parse string fields vectorized
easntrcv, _, rest = np.strings.partition(eastmp, '_')
ntrc_flat = np.strings.slice(easntrcv, 1)
ntrv_flat = np.strings.slice(easntrcv, 1, 20)

easxprcv, _, rest = np.strings.partition(rest, '_')
conds = [
    easxprcv == "OVERRIDE",
    (np.strings.slice(easxprcv, 2) == "BD") & np.isin(np.strings.slice(easxprcv, 2, 20), ["a", "b", "c", "d"]),
    np.strings.slice(easxprcv, 2) == "BD"
]
choices_xprc = [easxprcv, "BD", "BD"]
choices_xprv = [
    "",
    (easchj * (20 // easchcnt + 1) + achk[easchi] % 5 + 1).astype("U"),
    np.strings.slice(easxprcv, 2, 20)
]
xprc_flat = np.select(conds, choices_xprc, default=np.strings.slice(easxprcv, 1))
xprv_flat = np.select(conds, choices_xprv, default=np.strings.slice(easxprcv, 1, 20))
wgt_flat, _, _ = np.strings.partition(rest, '_')

# Build ID mappings per-field (5 small 1D uniques instead of one big 3D unique)
ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrv_uniq, ntrv_ids_flat = np.unique(ntrv_flat, return_inverse=True)
xprc_uniq, xprc_ids_flat = np.unique(xprc_flat, return_inverse=True)
xprv_uniq, xprv_ids_flat = np.unique(xprv_flat, return_inverse=True)
wgt_uniq, wgt_ids_flat = np.unique(wgt_flat, return_inverse=True)

# Build lookup dicts for constants
ntrc2id = dict(zip(ntrc_uniq, range(len(ntrc_uniq))))
xprc2id = dict(zip(xprc_uniq, range(len(xprc_uniq))))
STR_OVERRIDE = xprc2id.get("OVERRIDE", -1)
STR_N = ntrc2id.get("N", -1)
STR_F = ntrc2id.get("F", -1)
STR_R = ntrc2id.get("R", -1)
STR_W = ntrc2id.get("W", -1)
STR_BD = xprc2id.get("BD", -1)

# Build ID matrix only (decode strings on-demand from unique arrays)
max_schedules = int(np.max(easchcnt))
schedule_id_matrix = np.full((len(amap), max_schedules, 5), -1, dtype=np.int32)

schedule_id_matrix[easchi, easchj, 0] = ntrc_ids_flat
schedule_id_matrix[easchi, easchj, 1] = ntrv_ids_flat
schedule_id_matrix[easchi, easchj, 2] = xprc_ids_flat
schedule_id_matrix[easchi, easchj, 3] = xprv_ids_flat
schedule_id_matrix[easchi, easchj, 4] = wgt_ids_flat

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Load ID dictionaries (NumPy format - no Arrow overhead)
# ========================================================================
print("load id dicts".ljust(20, "."), end="")
start_time = time.perf_counter()

ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
# ========================================================================
# build ID dictionaries
# ========================================================================
print("build id dicts".ljust(20, "."), end="")
start_time = time.perf_counter()

n_fields = len(field_arr)

# Build dicts for asset-level mapping (faster than searchsorted for N=189)
ticker_to_id = {s: i for i, s in enumerate(ticker_arr)}
field_to_id = {s: i for i, s in enumerate(field_arr)}

# Build sorted ticker array once for futures mapping (used later in map monthly ids)
# ticker_arr already loaded as U100 from .npy file
ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

# Asset-level ticker/field IDs (dict lookup faster than searchsorted for 189 items)
hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker], dtype=np.int32)
hedge_hedge_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge], dtype=np.int32)
calc_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge1], dtype=np.int32)
hedge_field_fid = np.array([field_to_id.get(str(s), -1) for s in hedge_field], dtype=np.int32)

PX_LAST_FID = field_to_id.get("PX_LAST", -1)
EMPTY_FID = field_to_id.get("", -1)

# Hedge IDs for legs 2-4
hedge_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge1], dtype=np.int32)
calc_hedge2_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge2], dtype=np.int32)
calc_hedge3_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge3], dtype=np.int32)
calc_hedge4_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge4], dtype=np.int32)

# Vol IDs - dual mapping (BBG: near/far as fields, CV: near as ticker)
vol_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_ticker], dtype=np.int32)
vol_near_fid   = np.array([field_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)
vol_far_fid    = np.array([field_to_id.get(str(s), -1) for s in vol_far], dtype=np.int32)
vol_near_tid   = np.array([ticker_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)

NONE_FID = field_to_id.get("none", -1)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Compute straddles: straddle-level arrays
# ========================================================================
# straddle-level arrays (monthly rows): length = smlen (~222K)
print("compute straddles".ljust(20, "."), end="")
start_time = time.perf_counter()

# months
ym = np.arange(2001*12+1-1, 2026*12+1-1, dtype=np.int64)
ym_len = len(ym)

# inas: input asset
inas = anames
inalen = len(inas)
# inidx : asset-count-length numpy array of index into numpy matrices with info
inidx = np.array([idx_map[a] for a in inas], dtype=np.uint64)
# inasc : input asset straddle count. # straddles by asset
inasc = aschcnt[inidx]  # lengths
sidx = np.repeat(inidx, inasc)  # by straddle asset index
sc = np.repeat(inasc, inasc)    # by straddle strad count for this index
si1 = np.arange(np.sum(inasc))
si2 = np.repeat((np.cumsum(inasc)-inasc), inasc)
si = si1 - si2                  # straddle id within this asset

# fancy indexing for the whole loop, the straddle-month-loop
smidx = np.repeat(sidx, ym_len)
smym = np.tile(ym, len(sidx))
smlen = len(smidx)

# asset, year, month
asset_vec = inas[smidx]
year_vec = smym // 12
month_vec = smym % 12 + 1

# straddle days
dpm = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int8)
leap_feb = np.full(len(smidx), 29, dtype=np.int8)

year0, month0 = year_vec, month_vec
leap0 = (((year0 % 4 == 0) & ((year0 % 100 != 0) | (year0 % 400 == 0)))) & (month0 == 2)
days0_vec = np.where(leap0, leap_feb, dpm[month0 - 1])

year1, month1 = (year_vec*12+(month_vec-1)-1) // 12, (year_vec*12+(month_vec-1)-1) % 12 + 1
leap1 = (((year1 % 4 == 0) & ((year1 % 100 != 0) | (year1 % 400 == 0)))) & (month1 == 2)
days1_vec = np.where(leap1, leap_feb, dpm[month1 - 1])

year2, month2 = (year_vec*12+(month_vec-1)-2) // 12, (year_vec*12+(month_vec-1)-2) % 12 + 1
leap2 = (((year2 % 4 == 0) & ((year2 % 100 != 0) | (year2 % 400 == 0)))) & (month2 == 2)
days2_vec = np.where(leap2, leap_feb, dpm[month2 - 1])

# straddle
schcnt_vec = np.repeat(sc, ym_len)
schid_vec = np.repeat(si, ym_len)

# Decode strings on-demand from ID matrix + unique arrays (avoids 3D string allocation)
schedule_id_matrix_smidx = schedule_id_matrix[smidx, schid_vec, :]
ntrc_id_vec = schedule_id_matrix_smidx[:, 0]
ntrv_id_vec = schedule_id_matrix_smidx[:, 1]
xprc_id_vec = schedule_id_matrix_smidx[:, 2]
xprv_id_vec = schedule_id_matrix_smidx[:, 3]
wgt_id_vec = schedule_id_matrix_smidx[:, 4]

# total day-count (use IDs for logic, defer string decoding to output assembly)
day_count_vec = days0_vec + days1_vec + np.where(ntrc_id_vec == STR_F, days2_vec, 0)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Straddle hedge/vol keys (ID-only)
# ========================================================================
print("hedge/vol keys (ids)".ljust(20, "."), end="")
start_time = time.perf_counter()

# Type masks/indices
hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
hedge_source_id_fut_smidx    = hedge_source_id_fut[smidx]
hedge_source_id_cds_smidx    = hedge_source_id_cds[smidx]
hedge_source_id_calc_smidx   = hedge_source_id_calc[smidx]

nonfut_idx = np.flatnonzero(hedge_source_id_nonfut_smidx)
fut_idx    = np.flatnonzero(hedge_source_id_fut_smidx)
cds_idx    = np.flatnonzero(hedge_source_id_cds_smidx)
calc_idx   = np.flatnonzero(hedge_source_id_calc_smidx)

# Expand asset-level IDs to straddle-month
hedge_ticker_tid_smidx  = hedge_ticker_tid[smidx]
hedge_field_fid_smidx   = hedge_field_fid[smidx]
hedge_hedge_tid_smidx   = hedge_hedge_tid[smidx]
hedge_hedge1_tid_smidx  = hedge_hedge1_tid[smidx]
calc1_tid_smidx = calc_hedge1_tid[smidx]
calc2_tid_smidx = calc_hedge2_tid[smidx]
calc3_tid_smidx = calc_hedge3_tid[smidx]
calc4_tid_smidx = calc_hedge4_tid[smidx]

# Futures hedge1 tid (temporary strings for fut_idx only)
hedge_fut_tid_smidx = np.full(smlen, -1, dtype=np.int32)
if len(fut_idx) > 0:
    hedge_fut_code_m       = hedge_fut_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m  = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)

    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
    hedge_fut_ticker_u = (hedge_fut_code_m + hedge_fut_tail_m).astype(TICKER_U)

    hedge_fut_tid_smidx[fut_idx] = map_to_id_searchsorted(hedge_fut_ticker_u, ticker_sorted, ticker_order)

# Build hedge1-4 keys (int32)
hedge1_key = np.full(smlen, -1, dtype=np.int32)
hedge1_key[nonfut_idx] = hedge_ticker_tid_smidx[nonfut_idx] * np.int32(n_fields) + hedge_field_fid_smidx[nonfut_idx]
hedge1_key[fut_idx]    = hedge_fut_tid_smidx[fut_idx]       * np.int32(n_fields) + np.int32(PX_LAST_FID)
hedge1_key[cds_idx]    = hedge_hedge_tid_smidx[cds_idx]     * np.int32(n_fields) + np.int32(PX_LAST_FID)
hedge1_key[calc_idx]   = calc1_tid_smidx[calc_idx]          * np.int32(n_fields) + np.int32(EMPTY_FID)

hedge2_key = np.full(smlen, -1, dtype=np.int32)
hedge2_key[cds_idx]  = hedge_hedge1_tid_smidx[cds_idx] * np.int32(n_fields) + np.int32(PX_LAST_FID)
hedge2_key[calc_idx] = calc2_tid_smidx[calc_idx]       * np.int32(n_fields) + np.int32(EMPTY_FID)

hedge3_key = np.full(smlen, -1, dtype=np.int32)
hedge3_key[calc_idx] = calc3_tid_smidx[calc_idx] * np.int32(n_fields) + np.int32(EMPTY_FID)

hedge4_key = np.full(smlen, -1, dtype=np.int32)
hedge4_key[calc_idx] = calc4_tid_smidx[calc_idx] * np.int32(n_fields) + np.int32(EMPTY_FID)

# Vol key (ID-only)
vol_source_id_bbg_smidx        = vol_source_id_bbg[smidx]
vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
vol_source_id_cv_smidx         = vol_source_id_cv[smidx]

bbgN_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_N))
bbgF_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_F))
lme_idx  = np.flatnonzero(vol_source_id_bbg_lmevol_smidx)
cv_idx   = np.flatnonzero(vol_source_id_cv_smidx)

vol_ticker_tid_smidx = vol_ticker_tid[smidx]
vol_near_fid_smidx   = vol_near_fid[smidx]
vol_far_fid_smidx    = vol_far_fid[smidx]
vol_near_tid_smidx   = vol_near_tid[smidx]

# LMEVOL R-ticker (reuse fut components via position mapping)
r_tid_smidx = np.full(smlen, -1, dtype=np.int32)
if len(lme_idx) > 0:
    # LMEVOL ⊆ fut_idx, find positions in fut_idx
    lme_fut_pos = np.searchsorted(fut_idx, lme_idx)

    # Debug assertion: verify LMEVOL ⊆ FUT invariant
    if DEBUG:
        ok = (lme_fut_pos < len(fut_idx)) & (fut_idx[lme_fut_pos] == lme_idx)
        assert np.all(ok), f"LMEVOL ⊆ FUT invariant broken: {np.sum(~ok)} mismatches"

    # R-ticker format: FUT_CODE + "R" + MONTH_CODE + YEAR + " " + MARKET_CODE
    r_ticker_lme = hedge_fut_code_m[lme_fut_pos] + "R" + hedge_fut_tail_m[lme_fut_pos]
    r_ticker_lme_u = r_ticker_lme.astype(TICKER_U)
    r_tid_smidx[lme_idx] = map_to_id_searchsorted(r_ticker_lme_u, ticker_sorted, ticker_order)

vol_key = np.full(smlen, -1, dtype=np.int32)
vol_key[bbgN_idx] = vol_ticker_tid_smidx[bbgN_idx] * np.int32(n_fields) + vol_near_fid_smidx[bbgN_idx]
vol_key[bbgF_idx] = vol_ticker_tid_smidx[bbgF_idx] * np.int32(n_fields) + vol_far_fid_smidx[bbgF_idx]
vol_key[lme_idx]  = r_tid_smidx[lme_idx]           * np.int32(n_fields) + np.int32(PX_LAST_FID)
vol_key[cv_idx]   = vol_near_tid_smidx[cv_idx]     * np.int32(n_fields) + np.int32(NONE_FID)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Precompute days-since-epoch lookup
# ========================================================================
print("precompute days".ljust(20, "."), end="")
start_time = time.perf_counter()

_ym_base = 2000 * 12  # base year-month
_ym_range = np.arange(2000*12, 2027*12)  # year-months as integers
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)  # days since 1970-01-01

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Precompute anchor LUTs (once, ~324 months)
# ========================================================================
print("anchor LUTs".ljust(20, "."), end="")
start_time = time.perf_counter()

import calendar

MONTH_RANGE = range(2000*12, 2027*12)  # 324 months
N_MONTHS = len(MONTH_RANGE)
YM_OFFSET = 2000*12  # base for indexing
INVALID = np.iinfo(np.int32).max

# Reuse _ym_epoch for first day epochs (no date object creation)
ym_arr = np.arange(2000*12, 2027*12, dtype=np.int32)
years = ym_arr // 12
months = ym_arr % 12 + 1
first_day_epochs = _ym_epoch[ym_arr - _ym_base].astype(np.int32)  # reuse existing LUT
days_in_month = np.array([calendar.monthrange(y, m)[1] for y, m in zip(years, months)], dtype=np.int32)
first_day_wd = np.array([calendar.weekday(y, m, 1) for y, m in zip(years, months)], dtype=np.int32)  # no date()

# --- Vectorized weekday anchors (F/R/W) - 0.04ms ---
fri_anchor = np.full((N_MONTHS, 6), INVALID, dtype=np.int32)
thu_anchor = np.full((N_MONTHS, 6), INVALID, dtype=np.int32)
wed_anchor = np.full((N_MONTHS, 6), INVALID, dtype=np.int32)

# First occurrence offset: (target_weekday - first_day_wd) % 7
first_fri_off = (4 - first_day_wd) % 7  # Friday = weekday 4
first_thu_off = (3 - first_day_wd) % 7  # Thursday = weekday 3
first_wed_off = (2 - first_day_wd) % 7  # Wednesday = weekday 2

for n in range(1, 6):
    fri_day = first_fri_off + 7 * (n - 1)
    valid = fri_day < days_in_month
    fri_anchor[valid, n] = first_day_epochs[valid] + fri_day[valid]

    thu_day = first_thu_off + 7 * (n - 1)
    valid = thu_day < days_in_month
    thu_anchor[valid, n] = first_day_epochs[valid] + thu_day[valid]

    wed_day = first_wed_off + 7 * (n - 1)
    valid = wed_day < days_in_month
    wed_anchor[valid, n] = first_day_epochs[valid] + wed_day[valid]

# --- BD anchors (loop required for cumulative counting) - ~3.5ms ---
bd_anchor = np.full((N_MONTHS, 24), INVALID, dtype=np.int32)

for mi in range(N_MONTHS):
    fd_epoch = first_day_epochs[mi]
    fd_wd = first_day_wd[mi]
    num_days = days_in_month[mi]

    bd_count = 0
    for d in range(num_days):
        wd = (fd_wd + d) % 7
        if wd < 5:  # Mon-Fri
            bd_count += 1
            if bd_count <= 23:
                bd_anchor[mi, bd_count] = fd_epoch + d

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Build OVERRIDE epoch matrix (once per run)
# ========================================================================
print("override matrix".ljust(20, "."), end="")
start_time = time.perf_counter()

import csv
from datetime import date

n_assets = len(anames)
override_epoch = np.full((n_assets, N_MONTHS), INVALID, dtype=np.int32)

epoch_base = date(1970, 1, 1)
try:
    with open("data/overrides.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"]
            expiry = row["expiry"]  # "YYYY-MM-DD"

            # Map ticker to asset_id via idx_map
            if ticker not in idx_map:
                continue
            aid = idx_map[ticker]

            # Parse date and compute month index
            y, m, d_day = map(int, expiry.split("-"))
            mi = (y * 12 + m - 1) - YM_OFFSET
            if mi < 0 or mi >= N_MONTHS:
                continue

            # Compute epoch and store
            epoch = (date(y, m, d_day) - epoch_base).days
            override_epoch[aid, mi] = epoch
except FileNotFoundError:
    pass  # Leave all as INVALID

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Expand to daily calendar
# ========================================================================
# daily output arrays: length = total_days (~15.4M)
d_start_ym = np.where(ntrc_id_vec == STR_F, year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)
total_days = int(np.sum(day_count_vec))

# Sweep mode only needs d_start_ym and total_days (computed above)

# ========================================================================
# Load price data (NumPy format with pre-built block metadata)
# ========================================================================
# price arrays: length = n_prices (~8.5M)
print("load prices".ljust(20, "."), end="")
start_time = time.perf_counter()

pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(px_date):,} prices, {len(px_starts):,} blocks)")

# ========================================================================
# Sweep merge (5 legs with filter-first)
# ========================================================================
month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)
month_len = day_count_vec.astype(np.int32)

# DEBUG sanity check
if DEBUG:
    end_max = int((month_out0.astype(np.int64) + day_count_vec.astype(np.int64)).max())
    assert end_max == total_days, f"out0+len mismatch: {end_max} != {total_days}"

# Build straddle index mapping (daily row -> straddle)
print("straddle idx".ljust(20, "."), end="")
start_time = time.perf_counter()

# np.repeat is faster than Numba kernel for this simple pattern
straddle_idx = np.repeat(np.arange(smlen, dtype=np.int32), month_len)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Compute anchor dates (vectorized LUT lookup)
# ========================================================================
print("anchor dates".ljust(20, "."), end="")
start_time = time.perf_counter()

# Entry/expiry month indices (ensure int32 to avoid float upcasting)
ntry_year = np.where(ntrc_id_vec == STR_F, year2, year1).astype(np.int32)
ntry_month = np.where(ntrc_id_vec == STR_F, month2, month1).astype(np.int32)
ntry_mi = (ntry_year * 12 + ntry_month - 1) - YM_OFFSET  # month index

xpry_mi = (year_vec.astype(np.int32) * 12 + month_vec.astype(np.int32) - 1) - YM_OFFSET

# Parse xprv as integers (Nth occurrence) via LUT
xprv_lut = np.zeros(len(xprv_uniq), dtype=np.int32)
for i, v in enumerate(xprv_uniq):
    try:
        xprv_lut[i] = int(v.strip())
    except (ValueError, AttributeError):
        xprv_lut[i] = 0
xprv_n = xprv_lut[xprv_id_vec]  # S-length array

# Parse ntrv as integers (entry offset) via LUT
ntrv_lut = np.zeros(len(ntrv_uniq), dtype=np.int32)
for i, v in enumerate(ntrv_uniq):
    try:
        ntrv_lut[i] = int(v.strip())
    except (ValueError, AttributeError):
        ntrv_lut[i] = 0
ntrv_offsets = ntrv_lut[ntrv_id_vec]  # S-length array

# Build xprc_code lookup (string -> int)
XPRC_BD, XPRC_F, XPRC_R, XPRC_W, XPRC_OVERRIDE = 1, 2, 3, 4, 5
xprc_code_lut = np.zeros(len(xprc_uniq), dtype=np.int8)
for i, v in enumerate(xprc_uniq):
    vs = v.strip() if hasattr(v, 'strip') else str(v)
    if vs == "BD":
        xprc_code_lut[i] = XPRC_BD
    elif vs == "F":
        xprc_code_lut[i] = XPRC_F
    elif vs == "R":
        xprc_code_lut[i] = XPRC_R
    elif vs == "W":
        xprc_code_lut[i] = XPRC_W
    elif vs == "OVERRIDE":
        xprc_code_lut[i] = XPRC_OVERRIDE
xprc_code = xprc_code_lut[xprc_id_vec]  # S-length array

# Initialize anchor arrays
ntry_anchor = np.full(smlen, INVALID, dtype=np.int32)
xpry_anchor = np.full(smlen, INVALID, dtype=np.int32)

# Vectorized lookup for each xprc type using np.flatnonzero for cleaner indexing
# BD case
idx_bd = np.flatnonzero(xprc_code == XPRC_BD)
mi_ntry = ntry_mi[idx_bd]
mi_xpry = xpry_mi[idx_bd]
n_bd = xprv_n[idx_bd]
valid_ntry = (mi_ntry >= 0) & (mi_ntry < N_MONTHS) & (n_bd >= 1) & (n_bd <= 23)
valid_xpry = (mi_xpry >= 0) & (mi_xpry < N_MONTHS) & (n_bd >= 1) & (n_bd <= 23)
ntry_anchor[idx_bd[valid_ntry]] = bd_anchor[mi_ntry[valid_ntry], n_bd[valid_ntry]]
xpry_anchor[idx_bd[valid_xpry]] = bd_anchor[mi_xpry[valid_xpry], n_bd[valid_xpry]]

# F (Friday) case
idx_f = np.flatnonzero(xprc_code == XPRC_F)
mi_ntry = ntry_mi[idx_f]
mi_xpry = xpry_mi[idx_f]
n_f = xprv_n[idx_f]
valid_ntry = (mi_ntry >= 0) & (mi_ntry < N_MONTHS) & (n_f >= 1) & (n_f <= 5)
valid_xpry = (mi_xpry >= 0) & (mi_xpry < N_MONTHS) & (n_f >= 1) & (n_f <= 5)
ntry_anchor[idx_f[valid_ntry]] = fri_anchor[mi_ntry[valid_ntry], n_f[valid_ntry]]
xpry_anchor[idx_f[valid_xpry]] = fri_anchor[mi_xpry[valid_xpry], n_f[valid_xpry]]

# R (Thursday) case
idx_r = np.flatnonzero(xprc_code == XPRC_R)
mi_ntry = ntry_mi[idx_r]
mi_xpry = xpry_mi[idx_r]
n_r = xprv_n[idx_r]
valid_ntry = (mi_ntry >= 0) & (mi_ntry < N_MONTHS) & (n_r >= 1) & (n_r <= 5)
valid_xpry = (mi_xpry >= 0) & (mi_xpry < N_MONTHS) & (n_r >= 1) & (n_r <= 5)
ntry_anchor[idx_r[valid_ntry]] = thu_anchor[mi_ntry[valid_ntry], n_r[valid_ntry]]
xpry_anchor[idx_r[valid_xpry]] = thu_anchor[mi_xpry[valid_xpry], n_r[valid_xpry]]

# W (Wednesday) case
idx_w = np.flatnonzero(xprc_code == XPRC_W)
mi_ntry = ntry_mi[idx_w]
mi_xpry = xpry_mi[idx_w]
n_w = xprv_n[idx_w]
valid_ntry = (mi_ntry >= 0) & (mi_ntry < N_MONTHS) & (n_w >= 1) & (n_w <= 5)
valid_xpry = (mi_xpry >= 0) & (mi_xpry < N_MONTHS) & (n_w >= 1) & (n_w <= 5)
ntry_anchor[idx_w[valid_ntry]] = wed_anchor[mi_ntry[valid_ntry], n_w[valid_ntry]]
xpry_anchor[idx_w[valid_xpry]] = wed_anchor[mi_xpry[valid_xpry], n_w[valid_xpry]]

# --- OVERRIDE handling (12% of schedules) - VECTORIZED ---
idx_ovr = np.flatnonzero(xprc_code == XPRC_OVERRIDE)
if len(idx_ovr) > 0:
    # Use smidx (asset index) not asset_vec (strings) - avoids string mismatch
    aid = smidx[idx_ovr].astype(np.int32)          # asset id for each OVERRIDE straddle
    mi_entry = ntry_mi[idx_ovr]                    # entry month index
    mi_xpry_ovr = xpry_mi[idx_ovr]                 # expiry month index

    # Bounds check for safety
    valid_entry = (mi_entry >= 0) & (mi_entry < N_MONTHS)
    valid_xpry = (mi_xpry_ovr >= 0) & (mi_xpry_ovr < N_MONTHS)

    # Entry anchor: lookup by (asset_id, entry_month)
    e_entry = np.where(valid_entry, override_epoch[aid, np.clip(mi_entry, 0, N_MONTHS-1)], INVALID)
    ok_entry = (e_entry != INVALID)
    ntry_anchor[idx_ovr[ok_entry]] = e_entry[ok_entry]

    # Expiry anchor: lookup by (asset_id, expiry_month)
    e_xpry = np.where(valid_xpry, override_epoch[aid, np.clip(mi_xpry_ovr, 0, N_MONTHS-1)], INVALID)
    ok_xpry = (e_xpry != INVALID)
    xpry_anchor[idx_ovr[ok_xpry]] = e_xpry[ok_xpry]

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Compute month ends (vectorized, no calendar.monthrange)
# ========================================================================
# Entry month: use days1_vec (N) or days2_vec (F)
entry_days = np.where(ntrc_id_vec == STR_F, days2_vec, days1_vec).astype(np.int32)

# Entry month end: month_start_epoch IS already the entry month start
# (no need to recompute ntry_month_start from _ym_epoch)
ntry_month_end = month_start_epoch + entry_days - 1

# Expiry month end
xpry_ym = year_vec * 12 + month_vec - 1
xpry_month_start = _ym_epoch[xpry_ym - _ym_base].astype(np.int32)
xpry_month_end = xpry_month_start + days0_vec.astype(np.int32) - 1

# === All legs use filter-first (sparse path is best for all) ===
d_hedge1_value = sweep_leg_sparse("sweep hedge1", hedge1_key, month_start_epoch, month_len, month_out0,
                                  px_block_of, px_starts, px_ends, px_date, px_value, total_days)
d_hedge2_value = sweep_leg_sparse("sweep hedge2", hedge2_key, month_start_epoch, month_len, month_out0,
                                  px_block_of, px_starts, px_ends, px_date, px_value, total_days)
d_hedge3_value = sweep_leg_sparse("sweep hedge3", hedge3_key, month_start_epoch, month_len, month_out0,
                                  px_block_of, px_starts, px_ends, px_date, px_value, total_days)
d_hedge4_value = sweep_leg_sparse("sweep hedge4", hedge4_key, month_start_epoch, month_len, month_out0,
                                  px_block_of, px_starts, px_ends, px_date, px_value, total_days)
d_vol_value    = sweep_leg_sparse("sweep vol",    vol_key,    month_start_epoch, month_len, month_out0,
                                  px_block_of, px_starts, px_ends, px_date, px_value, total_days)

# ========================================================================
# Compute actions (entry/expiry detection)
# ========================================================================
# Required flags from key existence
req_vol = (vol_key >= 0)
req_hedge1 = (hedge1_key >= 0)
req_hedge2 = (hedge2_key >= 0)
req_hedge3 = (hedge3_key >= 0)
req_hedge4 = (hedge4_key >= 0)

print("compute actions".ljust(20, "."), end="")
start_time = time.perf_counter()

action, ntry_offsets, xpry_offsets = compute_actions(
    month_out0, month_len, month_start_epoch,
    ntry_anchor, xpry_anchor, ntrv_offsets,
    ntry_month_end, xpry_month_end,
    req_vol, req_hedge1, req_hedge2, req_hedge3, req_hedge4,
    d_vol_value, d_hedge1_value, d_hedge2_value, d_hedge3_value, d_hedge4_value,
)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# Summary
valid_straddles = np.sum(ntry_offsets >= 0)
print(f"  ntry found: {np.sum(action == 1):,}")
print(f"  xpry found: {np.sum(action == 2):,}")
print(f"  valid straddles: {valid_straddles:,} / {smlen:,}")

# Breakdown by hedge source
valid_mask_s = (ntry_offsets >= 0)
valid_fut    = np.sum(valid_mask_s[fut_idx])
valid_nonfut = np.sum(valid_mask_s[nonfut_idx])
valid_cds    = np.sum(valid_mask_s[cds_idx])
valid_calc   = np.sum(valid_mask_s[calc_idx])
print(f"  by hedge source: fut={valid_fut:,} nonfut={valid_nonfut:,} cds={valid_cds:,} calc={valid_calc:,}")

# ========================================================================
# Phase 4A: Roll-forward (in-place)
# ========================================================================
print("roll forward".ljust(20, "."), end="")
start_time = time.perf_counter()

roll_forward_inplace(d_vol_value, month_out0, ntry_offsets, xpry_offsets)
roll_forward_inplace(d_hedge1_value, month_out0, ntry_offsets, xpry_offsets)
# hedge2-4 only if needed for your use case

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Phase 4B: Strike capture + days-to-expiry
# ========================================================================
strike = np.full(total_days, np.nan, dtype=np.float64)
days_to_expiry = np.full(total_days, -1, dtype=np.int32)

print("fill strikes/dte".ljust(20, "."), end="")
start_time = time.perf_counter()

fill_strikes_and_dte(month_out0, month_start_epoch, ntry_offsets, xpry_offsets,
                     d_hedge1_value, strike, days_to_expiry)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Phase 5: Model (mv, delta)
# ========================================================================
print("build mask".ljust(20, "."), end="")
start_time = time.perf_counter()

valid_mask = build_active_mask(month_out0, ntry_offsets, xpry_offsets, total_days)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

print("model ES".ljust(20, "."), end="")
start_time = time.perf_counter()

mv, delta = model_ES_vectorized(d_hedge1_value, strike, d_vol_value, days_to_expiry, valid_mask)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# ========================================================================
# Phase 6: PnL
# ========================================================================
print("compute pnl".ljust(20, "."), end="")
start_time = time.perf_counter()

opnl, hpnl, pnl = compute_pnl_batch(mv, delta, d_hedge1_value, strike,
                                     month_out0, ntry_offsets, xpry_offsets)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

# PnL summary
print(f"  pnl non-nan: {np.sum(~np.isnan(pnl)):,}")
print(f"  avg daily pnl: {np.nanmean(pnl):.6f}")

# ========================================================================
# Compute pnl_valid_days per straddle (dte >= 0 and pnl not NaN)
# ========================================================================
print("pnl valid days".ljust(20, "."), end="")
start_time = time.perf_counter()

pnl_valid_days = np.zeros(smlen, dtype=np.int32)
for s in range(smlen):
    out0 = month_out0[s]
    length = month_len[s]
    dte_slice = days_to_expiry[out0:out0+length]
    pnl_slice = pnl[out0:out0+length]
    mask = (dte_slice >= 0) & ~np.isnan(pnl_slice)
    pnl_valid_days[s] = int(mask.sum())

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")
print(f"  total valid days: {pnl_valid_days.sum():,}")

# ========================================================================
# Output assembly (strings decoded for reporting)
# ========================================================================
# NOTE: For max speed, gate string decoding behind a flag and only decode
# columns needed downstream.

# Decode schedule strings from IDs (deferred from compute straddles)
ntrc_vec = ntrc_uniq[ntrc_id_vec]
ntrv_vec = ntrv_uniq[ntrv_id_vec]
xprc_vec = xprc_uniq[xprc_id_vec]
xprv_vec = xprv_uniq[xprv_id_vec]
wgt_vec = wgt_uniq[wgt_id_vec]

print("-"*40)
print(f"total: {1e3*(time.perf_counter()-script_start_time):0.3f}ms")

result = {
    "orientation": "numpy",
    "columns": [
        "year",
        "month",
        "asset",
        "schcnt", "schid",
        "ntrc", "ntrv", "xprc", "xprv", "wgt",
        "days0", "days1", "days2",
        "day_count",
    ],
    "rows": [
        year_vec,
        month_vec,
        asset_vec,
        schcnt_vec,
        schid_vec,
        ntrc_vec,
        ntrv_vec,
        xprc_vec,
        xprv_vec,
        wgt_vec,
        days0_vec, days1_vec, days2_vec,
        day_count_vec,
    ]
}

# Daily-level value arrays (total_days length, separate from straddle-month result)
daily_values = {
    "straddle_idx": straddle_idx,  # maps daily row -> straddle index
    "vol": d_vol_value,
    "hedge1": d_hedge1_value,
    "hedge2": d_hedge2_value,
    "hedge3": d_hedge3_value,
    "hedge4": d_hedge4_value,
    "strike": strike,
    "days_to_expiry": days_to_expiry,
    "mv": mv,
    "delta": delta,
    "opnl": opnl,
    "hpnl": hpnl,
    "pnl": pnl,
}

# ========================================================================
# Save to .npz files
# ========================================================================
print("saving npz".ljust(20, "."), end="")
start_time = time.perf_counter()

# Straddle metadata (222K straddles)
np.savez_compressed(
    "data/straddles.npz",
    year=year_vec,
    month=month_vec,
    asset=asset_vec,
    schcnt=schcnt_vec,
    schid=schid_vec,
    ntrc=ntrc_vec,
    ntrv=ntrv_vec,
    xprc=xprc_vec,
    xprv=xprv_vec,
    wgt=wgt_vec,
    days0=days0_vec,
    days1=days1_vec,
    days2=days2_vec,
    day_count=day_count_vec,
    # Slicing arrays for daily values
    out0=month_out0,
    length=month_len,
    month_start_epoch=month_start_epoch,
    pnl_valid_days=pnl_valid_days,
)

# Daily valuations (15.4M rows)
np.savez_compressed(
    "data/valuations.npz",
    straddle_idx=straddle_idx,
    vol=d_vol_value,
    hedge1=d_hedge1_value,
    hedge2=d_hedge2_value,
    hedge3=d_hedge3_value,
    hedge4=d_hedge4_value,
    strike=strike,
    days_to_expiry=days_to_expiry,
    mv=mv,
    delta=delta,
    opnl=opnl,
    hpnl=hpnl,
    pnl=pnl,
)

print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

    