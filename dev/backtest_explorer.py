"""Backtest Explorer v2 - Interactive straddle analysis."""
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numba import njit, prange
from streamlit_lightweight_charts import renderLightweightCharts
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from sklearn.manifold import TSNE, MDS
import umap
import streamlit.components.v1 as components

st.set_page_config(page_title="Backtest Explorer", layout="wide")

# ============================================================================
# Timing Utilities
# ============================================================================
def _fmt_dt_s(dt: float) -> str:
    """Format duration as ms or seconds."""
    ms = dt * 1000.0
    return f"{ms:.0f} ms" if ms < 1000.0 else f"{dt:.3f} s"

def _ts() -> str:
    """Current timestamp HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")

def tlog(msg: str) -> None:
    """Append message to timing log (respects enabled flag and keep limit)."""
    if not st.session_state.get("timing_enabled", True):
        return
    keep = int(st.session_state.get("timing_keep", 50))
    log = st.session_state.setdefault("timing_log", [])
    log.append(msg)
    if len(log) > keep:
        del log[:-keep]

@contextmanager
def timed_collect(name: str, store: dict, meta: str = ""):
    """Context manager: times block, logs, and stores in dict."""
    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt = time.perf_counter() - t0
        store[name] = dt
        if st.session_state.get("timing_enabled", True):
            status = "" if ok else " (error)"
            m = f" {meta}" if meta else ""
            tlog(f"{_ts()} {name}: {_fmt_dt_s(dt)}{status}{m}")

def record_timings(store: dict) -> None:
    """Record timings to session state for summary table."""
    st.session_state["timing_last"] = store
    hist = st.session_state.setdefault("timing_hist", {})
    for k, v in store.items():
        arr = hist.setdefault(k, [])
        arr.append(float(v))
        if len(arr) > 30:
            del arr[0]

def render_timings_panel():
    """Render timing panel in sidebar."""
    with st.sidebar.expander("Timings", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.checkbox("Enable", value=st.session_state.get("timing_enabled", True), key="timing_enabled")
        with c2:
            st.number_input("Keep", 10, 500, st.session_state.get("timing_keep", 50), 10, key="timing_keep")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear log"):
                st.session_state["timing_log"] = []
                st.rerun()
        with col_b:
            if st.button("Clear cache"):
                st.cache_data.clear()
                st.rerun()

        log = st.session_state.get("timing_log", [])
        st.text_area("log", "\n".join(log), height=200, label_visibility="collapsed")

        # Summary table
        last = st.session_state.get("timing_last", {})
        hist = st.session_state.get("timing_hist", {})
        if last:
            rows = []
            for k, v in sorted(last.items(), key=lambda kv: kv[1], reverse=True):
                h = hist.get(k, [])
                avg = (sum(h) / len(h)) if h else float("nan")
                rows.append((k, v*1000.0, avg*1000.0, len(h)))
            df = pd.DataFrame(rows, columns=["block", "last_ms", "avg_ms", "n"])
            st.dataframe(df, width='stretch', hide_index=True)

# Global CSS for slicker look
st.markdown("""
<style>
/* Black table headers */
.stDataFrame thead th {
    background-color: #1a1a2e !important;
    color: white !important;
    font-weight: 600 !important;
}
/* Slightly bolder tab labels */
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
}
/* Tighter metric spacing */
[data-testid="stMetricValue"] {
    font-size: 1.3rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Data Loading & Schema Validation
# ============================================================================
REQUIRED_STRADDLE_KEYS = ["year", "month", "asset", "schid", "out0", "length",
                          "ntrc", "ntrv", "xprc", "xprv", "month_start_epoch"]
REQUIRED_VALUATION_KEYS = ["vol", "hedge1", "strike", "mv", "delta", "pnl"]
OPTIONAL_VALUATION_KEYS = ["hedge2", "hedge3", "hedge4", "opnl", "hpnl", "days_to_expiry"]

@st.cache_data
def load_data():
    with np.load("data/straddles.npz", allow_pickle=True) as z:
        straddles = {k: z[k] for k in z.files}
    with np.load("data/valuations.npz", allow_pickle=True) as z:
        valuations = {k: z[k] for k in z.files}

    # Schema validation
    missing_s = [k for k in REQUIRED_STRADDLE_KEYS if k not in straddles]
    missing_v = [k for k in REQUIRED_VALUATION_KEYS if k not in valuations]
    available_optional = [k for k in OPTIONAL_VALUATION_KEYS if k in valuations]

    return straddles, valuations, missing_s, missing_v, available_optional

@st.cache_data
def get_unique_values(straddles):
    # Convert StringDType to list of Python strings (astype(str) fails with StringDType)
    assets = [str(x) for x in np.unique(straddles["asset"])]
    return {
        "assets": assets,
        "years": np.unique(straddles["year"]),
        "months": np.unique(straddles["month"]),
    }

@st.cache_data
def get_asset_strings(straddles):
    """Precompute asset strings for fast filtering."""
    asset_str = np.asarray([str(x) for x in straddles["asset"]], dtype=object)
    unique_assets = sorted(set(asset_str))
    return asset_str, unique_assets

@st.cache_data
def get_straddle_ym(straddles):
    """Precompute year-month index (year*12 + month-1) for range filtering."""
    return (straddles["year"].astype(np.int32) * 12 +
            straddles["month"].astype(np.int32) - 1)

def ym_to_date(ym: int) -> date:
    """Convert year-month index to date (first of month)."""
    return date(ym // 12, (ym % 12) + 1, 1)

def date_to_ym(d: date) -> int:
    """Convert date to year-month index."""
    return d.year * 12 + (d.month - 1)


def make_filter_key(asset_mode: str, asset_value, ym_lo: int, ym_hi: int, selected_schids):
    """Build a small, hashable cache key from filter parameters."""
    if asset_mode == "dropdown":
        asset_part = ("dropdown", asset_value)
    elif asset_mode == "similar":
        # asset_value is tuple: (anchor, n, rank, corr_method, min_overlap, partial_fill, partial_estimator)
        asset_part = ("similar",) + tuple(asset_value)
    else:
        pattern, flags = asset_value
        asset_part = ("regex", pattern, int(flags))

    sch_part = tuple(sorted(int(s) for s in selected_schids))
    return (asset_part, int(ym_lo), int(ym_hi), sch_part)


@st.cache_data
def get_filtered_indices_cached(filter_key):
    """Cached filter_key → indices. Computes asset_str/straddle_ym locally."""
    straddles, _, _, _, _ = load_data()

    # Compute locally to avoid nested cache hashing of straddles dict
    asset_str_local = np.asarray([str(x) for x in straddles["asset"]], dtype=object)
    straddle_ym_local = (straddles["year"].astype(np.int32) * 12 +
                         straddles["month"].astype(np.int32) - 1)

    asset_part, ym_lo, ym_hi, sch_part = filter_key
    if asset_part[0] == "dropdown":
        asset_mode, asset_value = "dropdown", asset_part[1]
    elif asset_part[0] == "similar":
        # Similar mode: find N most correlated assets to anchor
        # asset_part = ("similar", anchor, n, rank, corr_method, min_overlap, partial_fill, partial_estimator)
        _, anchor, n, rank, corr_method, min_overlap, p_fill, p_est = asset_part

        # Build a temporary filter_key with "All" assets for correlation computation
        temp_key = (("dropdown", "All"), ym_lo, ym_hi, sch_part)
        candidates, _ = top_correlated_assets(
            temp_key, anchor, n, corr_method, min_overlap, rank,
            partial_fill=p_fill, partial_estimator=p_est
        )
        asset_mode, asset_value = "candidates", candidates
    else:
        asset_mode = "regex"
        asset_value = (asset_part[1], asset_part[2])

    idx, _ = get_filtered_indices(
        straddles, asset_str_local, straddle_ym_local,
        asset_mode, asset_value, ym_lo, ym_hi, set(sch_part)
    )
    return idx.astype(np.int64)


@st.cache_data
def build_label_map(straddles, filtered_indices, limit=1000):
    """Build labels for selector, sorted by year-month descending (latest first)."""
    # Sort by year desc, month desc
    years = straddles["year"][filtered_indices]
    months = straddles["month"][filtered_indices]
    sort_keys = years * 100 + months  # YYYYMM as sortable int
    sorted_order = np.argsort(-sort_keys)  # descending
    sorted_indices = filtered_indices[sorted_order][:limit]

    labels = [
        f"{straddles['asset'][i]} {straddles['year'][i]}-{straddles['month'][i]:02d} sch{straddles['schid'][i]}"
        for i in sorted_indices
    ]
    return sorted_indices, labels


# ============================================================================
# Numba Kernels for Performance
# ============================================================================
@njit(parallel=True, cache=True)
def _summarize_all(out0s, lens, pnl, vol, mv, dte, have_dte):
    """Numba kernel: compute ALL per-straddle metrics in ONE pass.

    Returns: (pnl_sum, pnl_days, vol_sum, vol_days, mv_sum) arrays
    """
    n = out0s.shape[0]
    pnl_sum = np.zeros(n, np.float64)
    pnl_days = np.zeros(n, np.int32)
    vol_sum = np.zeros(n, np.float64)
    vol_days = np.zeros(n, np.int32)
    mv_sum = np.zeros(n, np.float64)

    for k in prange(n):
        o = out0s[k]
        L = lens[k]
        ps, pc = 0.0, 0
        vs, vc = 0.0, 0
        ms = 0.0

        for j in range(L):
            idx = o + j
            # DTE check (skip if have_dte and dte < 0)
            if have_dte and dte[idx] < 0:
                continue

            # PnL
            p = pnl[idx]
            if not np.isnan(p):
                ps += p
                pc += 1

            # Vol
            v = vol[idx]
            if not np.isnan(v):
                vs += v
                vc += 1

            # MV
            m = mv[idx]
            if not np.isnan(m):
                ms += m

        pnl_sum[k] = ps
        pnl_days[k] = pc
        vol_sum[k] = vs
        vol_days[k] = vc
        mv_sum[k] = ms

    return pnl_sum, pnl_days, vol_sum, vol_days, mv_sum


@njit(cache=True)
def _aggregate_daily(out0s, lens, starts_epoch, d0, pnl, vol, mv, opnl, hpnl,
                     dte, have_dte, have_opnl, have_hpnl, grid_size):
    """Numba kernel: aggregate daily totals in ONE pass over all straddles.

    Single-threaded for safe shared writes to output arrays.

    Returns: (pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum,
              norm_pnl_sum, norm_opnl_sum, norm_hpnl_sum) daily arrays
    """
    pnl_sum = np.zeros(grid_size, np.float64)
    pnl_cnt = np.zeros(grid_size, np.int32)
    vol_sum = np.zeros(grid_size, np.float64)
    vol_cnt = np.zeros(grid_size, np.int32)
    mv_sum = np.zeros(grid_size, np.float64)
    opnl_sum = np.zeros(grid_size, np.float64)
    hpnl_sum = np.zeros(grid_size, np.float64)
    norm_pnl_sum = np.zeros(grid_size, np.float64)
    norm_opnl_sum = np.zeros(grid_size, np.float64)
    norm_hpnl_sum = np.zeros(grid_size, np.float64)

    n = out0s.shape[0]
    for k in range(n):  # Single-threaded for safe writes
        o = out0s[k]
        L = lens[k]
        start = starts_epoch[k]

        for j in range(L):
            idx = o + j
            day_idx = start + j - d0

            # Skip invalid day indices
            if day_idx < 0 or day_idx >= grid_size:
                continue

            # DTE check
            if have_dte and dte[idx] < 0:
                continue

            # PnL
            p = pnl[idx]
            v = vol[idx]
            if not np.isnan(p):
                pnl_sum[day_idx] += p
                pnl_cnt[day_idx] += 1
                # Normalized P&L: pnl / (vol / 16) = pnl * 16 / vol
                if not np.isnan(v) and v > 0:
                    norm_pnl_sum[day_idx] += p * 16.0 / v

            # Vol (v already read above)
            if not np.isnan(v):
                vol_sum[day_idx] += v
                vol_cnt[day_idx] += 1

            # MV
            m = mv[idx]
            if not np.isnan(m):
                mv_sum[day_idx] += m

            # Option PnL (if available)
            if have_opnl:
                op = opnl[idx]
                if not np.isnan(op):
                    opnl_sum[day_idx] += op
                    # Normalized option P&L
                    if not np.isnan(v) and v > 0:
                        norm_opnl_sum[day_idx] += op * 16.0 / v

            # Hedge PnL (if available)
            if have_hpnl:
                hp = hpnl[idx]
                if not np.isnan(hp):
                    hpnl_sum[day_idx] += hp
                    # Normalized hedge P&L
                    if not np.isnan(v) and v > 0:
                        norm_hpnl_sum[day_idx] += hp * 16.0 / v

    return (pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum,
            norm_pnl_sum, norm_opnl_sum, norm_hpnl_sum)


@njit(cache=True)
def _aggregate_daily_by_asset(out0s, lens, starts_epoch, asset_ids, d0,
                               pnl, vol, dte, have_dte, grid_size, n_assets):
    """Aggregate daily pnl/vol per asset into 2D grids [days, assets]."""
    pnl_sum = np.zeros((grid_size, n_assets), np.float64)
    vol_sum = np.zeros((grid_size, n_assets), np.float64)

    n = out0s.shape[0]
    for k in range(n):  # Single-threaded for safe shared writes
        o = out0s[k]
        L = lens[k]
        start = starts_epoch[k]
        a = asset_ids[k]

        for j in range(L):
            idx = o + j
            day_idx = start + j - d0

            if day_idx < 0 or day_idx >= grid_size:
                continue
            if have_dte and dte[idx] < 0:
                continue

            p = pnl[idx]
            v = vol[idx]

            if not np.isnan(p):
                pnl_sum[day_idx, a] += p
            if not np.isnan(v):
                vol_sum[day_idx, a] += v

    return pnl_sum, vol_sum


@st.cache_data
def factorize_assets(_asset_str_tuple):
    """Convert asset strings to integer codes for bincount."""
    asset_str = np.array(_asset_str_tuple)
    codes, uniques = pd.factorize(asset_str, sort=True)
    return codes.astype(np.int32), np.asarray(uniques, dtype=object)


def _compute_population_daily_from_indices(straddles, valuations, filtered_indices: np.ndarray) -> pd.DataFrame:
    """Core population daily computation (uncached)."""
    if filtered_indices.size == 0:
        return pd.DataFrame()

    # Phase 1: Array prep
    t0 = time.perf_counter()
    out0s = straddles["out0"][filtered_indices].astype(np.int32)
    lens = straddles["length"][filtered_indices].astype(np.int32)
    starts = straddles["month_start_epoch"][filtered_indices].astype(np.int32)

    d0, d1 = int(starts.min()), int((starts + lens - 1).max())
    grid_size = d1 - d0 + 1

    pnl = valuations["pnl"]
    vol = valuations["vol"]
    mv = valuations["mv"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    have_opnl = "opnl" in valuations
    have_hpnl = "hpnl" in valuations
    opnl = valuations["opnl"] if have_opnl else np.empty(1, dtype=np.float64)
    hpnl = valuations["hpnl"] if have_hpnl else np.empty(1, dtype=np.float64)
    dt_prep = time.perf_counter() - t0

    # Phase 2: Numba kernel
    t0 = time.perf_counter()
    (pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum,
     norm_pnl_sum, norm_opnl_sum, norm_hpnl_sum) = _aggregate_daily(
        out0s, lens, starts, d0, pnl, vol, mv, opnl, hpnl,
        dte, have_dte, have_opnl, have_hpnl, grid_size
    )
    dt_kernel = time.perf_counter() - t0

    # Phase 3: DataFrame build
    t0 = time.perf_counter()
    base = np.datetime64('1970-01-01', 'D')
    dates = base + np.arange(d0, d1 + 1)

    has_data = pnl_cnt > 0

    result = {
        "n_straddles": pnl_cnt[has_data],
        "pnl_sum": pnl_sum[has_data],
        "mv_sum": mv_sum[has_data],
        "avg_vol": vol_sum[has_data] / np.maximum(vol_cnt[has_data], 1),
        "norm_pnl_sum": norm_pnl_sum[has_data],
    }

    if have_opnl:
        result["opnl_sum"] = opnl_sum[has_data]
        result["norm_opnl_sum"] = norm_opnl_sum[has_data]
    if have_hpnl:
        result["hpnl_sum"] = hpnl_sum[has_data]
        result["norm_hpnl_sum"] = norm_hpnl_sum[has_data]

    df = pd.DataFrame(result, index=pd.DatetimeIndex(dates[has_data]))
    df.index.name = "date"
    dt_df = time.perf_counter() - t0

    # Log sub-timings
    tlog(f"       pop_daily.prep: {_fmt_dt_s(dt_prep)}")
    tlog(f"       pop_daily.kernel: {_fmt_dt_s(dt_kernel)}")
    tlog(f"       pop_daily.df: {_fmt_dt_s(dt_df)}")

    return df


@st.cache_data
def compute_population_daily(filter_key):
    """Aggregate daily pnl/mv. Cached by filter params."""
    straddles, valuations, _, _, _ = load_data()
    idx = get_filtered_indices_cached(filter_key)
    return _compute_population_daily_from_indices(straddles, valuations, idx)


def make_single_asset_filter_key(filter_key, asset_name: str):
    """Create filter key for a single asset."""
    _, ym_lo, ym_hi, sch_part = filter_key
    return (("dropdown", asset_name), ym_lo, ym_hi, sch_part)


@st.cache_data
def compute_population_daily_for_asset(filter_key, asset_name: str) -> pd.DataFrame:
    """Compute daily aggregates for a single asset (cached)."""
    asset_filter_key = make_single_asset_filter_key(filter_key, asset_name)
    return compute_population_daily(asset_filter_key)


@st.cache_data
def compute_asset_daily_matrix(filter_key, series_source: str = "norm_pnl"):
    """Build daily matrix [dates × assets] for correlation.

    series_source: "norm_pnl" (pnl/daily_vol), "pnl", or "daily_vol"
    """
    straddles, valuations, _, _, _ = load_data()
    idx = get_filtered_indices_cached(filter_key)

    if idx.size == 0:
        return pd.DataFrame(), np.array([])

    # Prep arrays
    out0s = straddles["out0"][idx].astype(np.int32)
    lens = straddles["length"][idx].astype(np.int32)
    starts = straddles["month_start_epoch"][idx].astype(np.int32)

    # Factorize assets for selected straddles only (use precomputed asset_str)
    asset_str_all, _ = get_asset_strings(straddles)
    asset_str_sel = asset_str_all[idx]
    asset_codes, asset_names = pd.factorize(asset_str_sel, sort=True)
    asset_ids = asset_codes.astype(np.int32)
    n_assets = len(asset_names)

    # Grid bounds
    d0, d1 = int(starts.min()), int((starts + lens - 1).max())
    grid_size = d1 - d0 + 1

    # Valuations
    pnl = valuations["pnl"]
    vol = valuations["vol"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    # Run kernel
    pnl_sum, vol_sum = _aggregate_daily_by_asset(
        out0s, lens, starts, asset_ids, d0,
        pnl, vol, dte, have_dte, grid_size, n_assets
    )

    # Build X based on series_source
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_vol = vol_sum / 16.0
        if series_source == "pnl":
            X = pnl_sum.copy()
        elif series_source == "daily_vol":
            X = daily_vol.copy()
        else:  # "norm_pnl" (default)
            X = pnl_sum / daily_vol
        X[~np.isfinite(X)] = np.nan

    # Build DataFrame with date index
    base = np.datetime64('1970-01-01', 'D')
    dates = pd.DatetimeIndex(base + np.arange(d0, d1 + 1))

    # Filter rows where at least one asset has data
    row_mask = np.any(np.isfinite(X), axis=1)
    X = X[row_mask]
    dates = dates[row_mask]

    df = pd.DataFrame(X, index=dates, columns=asset_names)
    return df, np.asarray(asset_names)


@st.cache_data
def compute_asset_daily_matrices(filter_key):
    """Build all three daily matrices [dates × assets] aligned on same index.

    Returns:
        df_pnl: daily pnl_sum
        df_dvol: daily vol (vol_sum / 16)
        df_norm: normalized pnl (pnl_sum / daily_vol)
        asset_names: array of asset names
    """
    straddles, valuations, _, _, _ = load_data()
    idx = get_filtered_indices_cached(filter_key)

    if idx.size == 0:
        empty = pd.DataFrame()
        return empty, empty, empty, np.array([])

    # Prep arrays
    out0s = straddles["out0"][idx].astype(np.int32)
    lens = straddles["length"][idx].astype(np.int32)
    starts = straddles["month_start_epoch"][idx].astype(np.int32)

    # Factorize assets
    asset_str_all, _ = get_asset_strings(straddles)
    asset_str_sel = asset_str_all[idx]
    asset_codes, asset_names = pd.factorize(asset_str_sel, sort=True)
    asset_ids = asset_codes.astype(np.int32)
    n_assets = len(asset_names)

    # Grid bounds
    d0, d1 = int(starts.min()), int((starts + lens - 1).max())
    grid_size = d1 - d0 + 1

    # Valuations
    pnl = valuations["pnl"]
    vol = valuations["vol"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    # Run kernel
    pnl_sum, vol_sum = _aggregate_daily_by_asset(
        out0s, lens, starts, asset_ids, d0,
        pnl, vol, dte, have_dte, grid_size, n_assets
    )

    # Build all three matrices
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_vol = vol_sum / 16.0
        norm_pnl = pnl_sum / daily_vol

    # Set non-finite to NaN
    pnl_sum[~np.isfinite(pnl_sum)] = np.nan
    daily_vol[~np.isfinite(daily_vol)] = np.nan
    norm_pnl[~np.isfinite(norm_pnl)] = np.nan

    # Build date index
    base = np.datetime64('1970-01-01', 'D')
    dates = pd.DatetimeIndex(base + np.arange(d0, d1 + 1))

    # Filter rows where at least one asset has data (using norm_pnl as reference)
    row_mask = np.any(np.isfinite(norm_pnl), axis=1)
    pnl_sum = pnl_sum[row_mask]
    daily_vol = daily_vol[row_mask]
    norm_pnl = norm_pnl[row_mask]
    dates = dates[row_mask]

    df_pnl = pd.DataFrame(pnl_sum, index=dates, columns=asset_names)
    df_dvol = pd.DataFrame(daily_vol, index=dates, columns=asset_names)
    df_norm = pd.DataFrame(norm_pnl, index=dates, columns=asset_names)

    return df_pnl, df_dvol, df_norm, np.asarray(asset_names)


@st.cache_data
def compute_corr_and_overlap(filter_key, corr_method: str, min_overlap: int,
                             partial_fill: str = "median", partial_estimator: str = "ledoitwolf",
                             partial_standardize: bool = True, partial_mask: bool = True,
                             series_source: str = "norm_pnl"):
    """Compute correlation matrix, overlap counts, and coverage (cached for reuse).

    Supports pearson, spearman, sign, and partial correlation methods.
    """
    dfX, _ = compute_asset_daily_matrix(filter_key, series_source=series_source)
    if dfX.empty:
        return pd.DataFrame(), np.zeros((0, 0), dtype=np.int32), np.array([]), pd.Series(dtype=np.int32), {}

    if corr_method in ("pearson", "spearman"):
        corr = dfX.corr(method=corr_method, min_periods=min_overlap)
        # Convert to int32 BEFORE matrix multiply (bool @ bool returns bool, not counts!)
        notna = np.asarray(dfX.notna(), dtype=np.int32)
        overlap = notna.T @ notna
    elif corr_method == "sign":
        corr_mat, overlap, _ = compute_sign_corr(dfX, min_overlap)
        corr = pd.DataFrame(corr_mat, index=dfX.columns, columns=dfX.columns)
    else:  # partial
        corr_mat, overlap, _ = compute_partial_corr(
            dfX, min_overlap,
            fill_mode=partial_fill,
            estimator=partial_estimator,
            standardize=partial_standardize,
            mask_low_overlap=partial_mask,
        )
        corr = pd.DataFrame(corr_mat, index=dfX.columns, columns=dfX.columns)

    coverage = dfX.notna().sum().astype(np.int32)  # per-asset day count (for Top N selection)

    # Debug info
    debug_info = {
        "dfX_shape": dfX.shape,
        "method": corr_method,
    }

    return corr, overlap, np.array(dfX.columns, dtype=object), coverage, debug_info


@st.cache_data
def compute_corr_frames(
    filter_key,
    assets: tuple,  # tuple for hashability
    method: str,
    min_overlap: int,
    window_days: int = 60,
    step_days: int = 5,
    top_pct: int = 10,
    sign: str = "both",
    max_edges_per_node: int = 10,
    partial_fill: str = "median",
    partial_estimator: str = "ledoitwolf",
    partial_standardize: bool = True,
    partial_mask: bool = True,
):
    """Compute correlation frames for animation over rolling windows.

    Returns list of frame dicts with: t0, t1, threshold, edges, node_meta
    """
    dfX, _ = compute_asset_daily_matrix(filter_key)
    assets_list = list(assets)
    dfX = dfX[[a for a in assets_list if a in dfX.columns]]
    if dfX.empty or len(dfX.columns) < 2:
        return []

    assets_list = list(dfX.columns)
    dates = dfX.index
    X = dfX.to_numpy(np.float64)
    M = np.isfinite(X).astype(np.int32)
    n = len(assets_list)

    def threshold_from_C(C, top_pct):
        mask = np.triu(np.ones_like(C, dtype=bool), k=1)
        vals = np.abs(C[mask & np.isfinite(C)])
        if vals.size == 0:
            return 1.0
        return float(np.percentile(vals, 100 - top_pct))

    frames = []
    w = window_days
    s = step_days

    for start in range(0, len(dates) - w + 1, s):
        end = start + w
        Xw = X[start:end]
        Mw = M[start:end]

        dfW = pd.DataFrame(Xw, columns=assets_list)

        # Compute correlation based on method
        if method in ("pearson", "spearman"):
            C = dfW.corr(method=method, min_periods=min_overlap).to_numpy()
            overlap = Mw.T @ Mw
        elif method == "sign":
            C, overlap, _ = compute_sign_corr(dfW, min_overlap)
        else:  # partial
            C, overlap, _ = compute_partial_corr(
                dfW, min_overlap, partial_fill, partial_estimator,
                partial_standardize, partial_mask
            )

        thr = threshold_from_C(C, top_pct)

        # Build edges
        corr_df = pd.DataFrame(C, index=assets_list, columns=assets_list)
        _, edges = corr_to_edges(
            corr_df, overlap,
            min_abs_corr=thr,
            min_overlap=min_overlap,
            sign=sign,
            max_edges_per_node=max_edges_per_node,
        )

        # Node pulse: rolling std of normalized pnl
        node_vol = np.nanstd(Xw, axis=0)
        node_meta = {assets_list[i]: {"pulse": float(node_vol[i]) if np.isfinite(node_vol[i]) else 0.0}
                     for i in range(n)}

        frames.append({
            "t0": dates[start].strftime("%Y-%m-%d"),
            "t1": dates[end-1].strftime("%Y-%m-%d"),
            "threshold": float(thr),
            "edges": [
                {"source": assets_list[e[0]], "target": assets_list[e[1]],
                 "corr": float(e[2]), "strength": abs(float(e[2])), "overlap": int(e[3])}
                for e in edges
            ],
            "node_meta": node_meta,
        })

    return frames


@st.cache_data
def compute_corr_frames_v2(
    filter_key,
    assets: tuple,
    edge_list: tuple,  # Fixed edges: ((i,j), ...)
    method: str,
    min_overlap: int,
    window_days: int = 60,
    step_days: int = 5,
    partial_fill: str = "median",
    partial_estimator: str = "ledoitwolf",
    partial_standardize: bool = True,
    partial_mask: bool = True,
):
    """Compute correlation weights for fixed edges over rolling windows.

    For living force-directed network: returns weights for a fixed edge universe
    (edges don't appear/disappear, they fade based on correlation strength).

    Returns list of frame dicts with:
      t0, t1: date range strings
      edge_corr: list of (corr, overlap) aligned with edge_list
      node_pulse: dict of asset -> volatility pulse
    """
    dfX, _ = compute_asset_daily_matrix(filter_key)
    assets_list = list(assets)
    dfX = dfX[[a for a in assets_list if a in dfX.columns]]
    if dfX.empty or len(dfX.columns) < 2:
        return []

    assets_list = list(dfX.columns)
    dates = dfX.index
    X = dfX.to_numpy(np.float64)
    M = np.isfinite(X).astype(np.int32)
    n = len(assets_list)
    edge_list = list(edge_list)

    frames = []
    w = window_days
    s = step_days

    for start in range(0, len(dates) - w + 1, s):
        end = start + w
        Xw = X[start:end]
        Mw = M[start:end]

        dfW = pd.DataFrame(Xw, columns=assets_list)

        # Compute correlation
        if method in ("pearson", "spearman"):
            C = dfW.corr(method=method, min_periods=min_overlap).to_numpy()
            overlap = Mw.T @ Mw
        elif method == "sign":
            C, overlap, _ = compute_sign_corr(dfW, min_overlap)
        else:  # partial
            C, overlap, _ = compute_partial_corr(
                dfW, min_overlap, partial_fill, partial_estimator,
                partial_standardize, partial_mask
            )

        # Extract correlations for fixed edges
        edge_corr = []
        for i, j in edge_list:
            c = C[i, j]
            o = int(overlap[i, j])
            if np.isfinite(c) and o >= min_overlap:
                edge_corr.append((float(c), o))
            else:
                edge_corr.append((0.0, 0))  # Weight 0 = faded out

        # Node pulse: rolling std
        node_vol = np.nanstd(Xw, axis=0)
        node_pulse = {assets_list[i]: float(node_vol[i]) if np.isfinite(node_vol[i]) else 0.0
                      for i in range(n)}

        frames.append({
            "t0": dates[start].strftime("%Y-%m-%d"),
            "t1": dates[end-1].strftime("%Y-%m-%d"),
            "edge_corr": edge_corr,
            "node_pulse": node_pulse,
        })

    return frames


def compute_sign_corr(dfX: pd.DataFrame, min_overlap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sign correlation matrix.

    Sign correlation: s_{t,i} = sign(x_{t,i} - median_i), then corr = mean(s_i * s_j) over overlap.
    Robust to outliers and scale differences.

    Returns: (corr_mat, overlap_mat, asset_names)
    """
    X = dfX.values.astype(np.float64)
    M = np.isfinite(X)

    # Per-asset median (ignoring NaNs)
    med = np.nanmedian(X, axis=0)

    # Sign matrix: +1 if >= median, -1 if < median, 0 if NaN
    S = np.where(X >= med, 1, -1).astype(np.int32)
    S[~M] = 0

    # Overlap matrix
    Mi = M.astype(np.int32)
    overlap = Mi.T @ Mi

    # Sign product sum
    prod_sum = S.T @ S

    # Sign correlation = prod_sum / overlap (with safe division)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = prod_sum.astype(np.float64) / overlap

    # Set diagonal to 1, mask low overlap to NaN
    np.fill_diagonal(corr, 1.0)
    corr[overlap < min_overlap] = np.nan

    return corr, overlap, dfX.columns.to_numpy()


def compute_partial_corr(
    dfX: pd.DataFrame,
    min_overlap: int,
    fill_mode: str = "median",
    estimator: str = "ledoitwolf",
    standardize: bool = True,
    mask_low_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute partial correlation matrix via precision matrix inversion.

    Partial correlation: rho_{ij|rest} = -Omega_{ij} / sqrt(Omega_ii * Omega_jj)
    where Omega = Sigma^{-1} (precision matrix).

    Returns: (pcorr_mat, overlap_mat, asset_names)
    """
    from sklearn.covariance import LedoitWolf, OAS

    X = dfX.values.astype(np.float64)
    n_assets = X.shape[1]

    # Compute overlap from raw missingness (for hover and optional masking)
    M = np.isfinite(X).astype(np.int32)
    overlap = M.T @ M

    # Fill NaNs
    if fill_mode == "median":
        fill_vals = np.nanmedian(X, axis=0)
    elif fill_mode == "mean":
        fill_vals = np.nanmean(X, axis=0)
    else:  # zero
        fill_vals = np.zeros(n_assets)

    # Handle all-NaN columns: fill_vals will be NaN, replace with 0
    fill_vals = np.nan_to_num(fill_vals, nan=0.0)

    # Replace NaNs with fill values (vectorized)
    nan_mask = ~np.isfinite(X)
    X_filled = np.where(nan_mask, fill_vals[np.newaxis, :], X)

    # Final safety check: ensure no NaNs remain
    X_filled = np.nan_to_num(X_filled, nan=0.0)

    # Standardize (recommended for partial correlation)
    if standardize:
        mu = X_filled.mean(axis=0)
        std = X_filled.std(axis=0)
        std[std < 1e-12] = 1.0  # Prevent div by zero
        X_filled = (X_filled - mu) / std

    # Estimate covariance with shrinkage
    if estimator == "ledoitwolf":
        cov = LedoitWolf().fit(X_filled).covariance_
    elif estimator == "oas":
        cov = OAS().fit(X_filled).covariance_
    else:  # ridge
        cov = np.cov(X_filled, rowvar=False, bias=False)
        cov += 1e-3 * np.eye(n_assets)  # Ridge regularization

    # Invert to precision matrix (robust with double fallback)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Regularize more and try again
        cov2 = cov + 1e-2 * np.eye(n_assets)
        try:
            precision = np.linalg.inv(cov2)
        except np.linalg.LinAlgError:
            # Last resort: pseudo-inverse (always works)
            precision = np.linalg.pinv(cov2)

    # Convert to partial correlation
    # pcorr_{ij} = -Omega_{ij} / sqrt(Omega_{ii} * Omega_{jj})
    diag = np.diag(precision)
    diag_sqrt = np.sqrt(np.abs(diag))
    diag_sqrt[diag_sqrt < 1e-12] = 1e-12

    pcorr = -precision / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(pcorr, 1.0)

    # Clip to [-1, 1] for numerical safety
    pcorr = np.clip(pcorr, -1.0, 1.0)

    # Optionally mask low overlap pairs
    if mask_low_overlap:
        pcorr[overlap < min_overlap] = np.nan

    return pcorr, overlap, dfX.columns.to_numpy()


def top_correlated_assets(filter_key, anchor: str, n: int,
                          corr_method: str, min_overlap: int,
                          rank_mode: str = "abs(corr)",
                          partial_fill: str = "median",
                          partial_estimator: str = "ledoitwolf",
                          partial_standardize: bool = True,
                          partial_mask: bool = True,
                          series_source: str = "norm_pnl") -> tuple[list[str], pd.DataFrame]:
    """Find top N assets most correlated with anchor asset.

    Returns: (list of asset names including anchor, DataFrame with corr/overlap)
    """
    corr, overlap, assets, coverage, _ = compute_corr_and_overlap(
        filter_key, corr_method, min_overlap,
        partial_fill=partial_fill,
        partial_estimator=partial_estimator,
        partial_standardize=partial_standardize,
        partial_mask=partial_mask,
        series_source=series_source,
    )
    if corr.empty or anchor not in corr.columns:
        return [anchor], pd.DataFrame()

    a = corr[anchor].copy()
    a.loc[anchor] = np.nan

    # Enforce overlap for anchor pairs
    j = list(corr.columns).index(anchor)
    ov_anchor = overlap[:, j].astype(np.int32)
    ok = ov_anchor >= min_overlap
    a[~ok] = np.nan

    if rank_mode == "abs(corr)":
        score = a.abs()
    else:  # "corr" - signed, positive first
        score = a

    top_idx = score.dropna().sort_values(ascending=False).head(n).index.tolist()

    # Build summary dataframe
    rows = []
    for asset in top_idx:
        rows.append({
            "asset": asset,
            "corr": corr.loc[asset, anchor],
            "overlap": ov_anchor[list(corr.columns).index(asset)],
        })
    summary_df = pd.DataFrame(rows)

    return [anchor] + top_idx, summary_df


# ============================================================================
# Correlation Network (CorGraph) Helpers
# ============================================================================
MAX_EDGES_HARD_CAP = 3000  # Plotly performance guardrail


def corr_to_edges(corr: pd.DataFrame, overlap: np.ndarray,
                  min_abs_corr: float, min_overlap: int,
                  sign: str = "both", max_edges_per_node: int | None = None):
    """Convert correlation matrix to edge list with thresholding."""
    assets = list(corr.columns)
    n = len(assets)
    C = corr.values

    # Collect candidate edges from upper triangle
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            c = C[i, j]
            if not np.isfinite(c):
                continue
            if abs(c) < min_abs_corr:
                continue
            o = int(overlap[i, j])
            if o < min_overlap:
                continue
            if sign == "pos" and c < 0:
                continue
            if sign == "neg" and c > 0:
                continue
            edges.append((i, j, float(c), o))

    # Prune to max edges per node (keeps graph readable)
    if max_edges_per_node is not None:
        edges.sort(key=lambda e: abs(e[2]), reverse=True)
        deg = np.zeros(n, dtype=np.int32)
        kept = []
        for i, j, c, o in edges:
            if deg[i] >= max_edges_per_node or deg[j] >= max_edges_per_node:
                continue
            kept.append((i, j, c, o))
            deg[i] += 1
            deg[j] += 1
        edges = kept

    return assets, edges


def build_graph_positions(assets: list, edges: list, seed: int = 42):
    """Build NetworkX graph and compute spring layout positions."""
    G = nx.Graph()
    for i, a in enumerate(assets):
        G.add_node(i, name=a)
    for i, j, c, o in edges:
        G.add_edge(i, j, weight=abs(c), corr=c, overlap=o)

    # Spring layout: stronger correlation => closer nodes
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=None, iterations=200)
    return G, pos


def mst_edges_from_corr(corr_mat: np.ndarray, overlap_mat: np.ndarray, mode: str = "1-abs(corr)"):
    """Compute MST edges to ensure graph connectivity."""
    C = corr_mat.copy()
    C[~np.isfinite(C)] = 0.0  # NaN -> 0 corr -> distance 1

    if mode == "1-abs(corr)":
        D = 1.0 - np.abs(C)
    else:  # "1-corr"
        D = 1.0 - C

    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    mst = minimum_spanning_tree(D).tocoo()
    edges = []
    for i, j, w in zip(mst.row, mst.col, mst.data):
        c = float(C[i, j])
        o = int(overlap_mat[i, j])
        edges.append((int(i), int(j), c, o))
    return edges


def build_edge_universe(corr_df: pd.DataFrame, overlap: np.ndarray,
                        min_overlap: int, top_m: int = 10, use_mst: bool = True) -> tuple[list, list]:
    """Build fixed edge universe from full-period correlation.

    For animated network: precompute candidate edges that stay constant across all frames.
    Per node: keep top M neighbors by |corr|, plus MST backbone for connectivity.

    Returns: (assets, edge_list) where edge_list is [(i,j), ...] sorted pairs
    """
    assets = list(corr_df.columns)
    C = corr_df.values
    n = len(assets)

    # Per-node top M by abs corr
    keep = set()
    for i in range(n):
        candidates = []
        for j in range(n):
            if i == j:
                continue
            if overlap[i, j] < min_overlap:
                continue
            c = C[i, j]
            if not np.isfinite(c):
                continue
            candidates.append((abs(c), i, j))
        candidates.sort(reverse=True)
        for _, a, b in candidates[:top_m]:
            keep.add((min(a, b), max(a, b)))

    # Add MST edges for connectivity
    if use_mst:
        mst = mst_edges_from_corr(C, overlap)
        for i, j, *_ in mst:
            keep.add((min(i, j), max(i, j)))

    edge_list = sorted(list(keep))
    return assets, edge_list


# ============================================================================
# Coverage Tradeoff Helper
# ============================================================================
@st.cache_data(show_spinner=False)
def compute_common_coverage_tradeoff(dfX: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy curve: at each step drop the asset whose removal increases
    common-row coverage the most.

    Returns a DataFrame with columns:
      n_assets, common_rows, coverage_pct, dropped
    """
    if dfX is None or dfX.empty or dfX.shape[1] < 1:
        return pd.DataFrame(columns=["n_assets", "common_rows", "coverage_pct", "dropped"])

    assets = list(dfX.columns)
    missing = (~np.asarray(dfX.notna(), dtype=np.bool_)).astype(np.uint8)  # (T, N) 0/1
    T, N = missing.shape
    if N == 0 or T == 0:
        return pd.DataFrame(columns=["n_assets", "common_rows", "coverage_pct", "dropped"])

    active = np.ones(N, dtype=np.bool_)
    miss_count = missing.sum(axis=1).astype(np.int32)  # (T,)
    common_rows = int((miss_count == 0).sum())

    rows = []
    rows.append({
        "n_assets": int(active.sum()),
        "common_rows": common_rows,
        "coverage_pct": 100.0 * common_rows / max(T, 1),
        "dropped": "",
    })

    # Greedy removals until 1 asset left (or already 100% coverage)
    while active.sum() > 1:
        if common_rows == T:
            break  # Already dense for all remaining assets

        rows_miss1 = (miss_count == 1)
        if not np.any(rows_miss1):
            # No rows with exactly one missing - drop asset with most missing overall
            miss_per_asset = missing[:, active].sum(axis=0).astype(np.int32)
            idx_active = np.flatnonzero(active)
            drop_local = int(np.argmax(miss_per_asset))
            drop_idx = int(idx_active[drop_local])
        else:
            # Count which asset is responsible for most single-miss rows
            miss1_missing_counts = missing[rows_miss1][:, active].sum(axis=0).astype(np.int32)
            idx_active = np.flatnonzero(active)
            drop_local = int(np.argmax(miss1_missing_counts))
            drop_idx = int(idx_active[drop_local])

        # Apply drop
        active[drop_idx] = False
        miss_count = miss_count - missing[:, drop_idx].astype(np.int32)
        common_rows = int((miss_count == 0).sum())

        rows.append({
            "n_assets": int(active.sum()),
            "common_rows": common_rows,
            "coverage_pct": 100.0 * common_rows / max(T, 1),
            "dropped": assets[drop_idx],
        })

    return pd.DataFrame(rows)


# Single source of truth for link strength exponent — used by both NetworkX and D3
WEIGHT_POWER = 1.6


# ============================================================================
# CorGraph Dense Matrix Helpers
# ============================================================================
def build_common_matrix(dfX: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Build dense matrix keeping only rows where ALL assets have data.

    Used by static CorGraph (compute_corgraph_state) for a single correlation snapshot.
    Uses strict dropna(how="any") - all assets must have data for a row to be included.

    Note: The animation path (compute_corr_frames_dense) uses adaptive 80% completeness
    per window to avoid losing too many observations in volatile windows. This is an
    intentional design divergence - static graph wants consistency, animation wants
    responsiveness to local data availability.

    Returns empty DataFrame if < 2 columns available.
    """
    cols = [a for a in assets if a in dfX.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return dfX[cols].dropna(axis=0, how="any")


def compute_sign_corr_dense(common_df: pd.DataFrame) -> np.ndarray:
    """Compute sign correlation on dense matrix (no NaNs).

    Sign correlation: s_i = sign(x_i - median_i), then corr = mean(s_i * s_j).
    """
    X = common_df.to_numpy(np.float64)
    med = np.median(X, axis=0)
    S = np.where(X >= med, 1, -1).astype(np.int32)
    C = (S.T @ S).astype(np.float64) / X.shape[0]
    np.fill_diagonal(C, 1.0)
    return C


def compute_partial_corr_dense(common_df: pd.DataFrame,
                               estimator: str = "ledoitwolf",
                               standardize: bool = True) -> np.ndarray:
    """Compute partial correlation on dense matrix (no NaNs).

    Partial correlation: rho_{ij|rest} = -Omega_{ij} / sqrt(Omega_ii * Omega_jj)
    where Omega = Sigma^{-1} (precision matrix).
    """
    from sklearn.covariance import LedoitWolf, OAS

    X = common_df.to_numpy(np.float64)  # no NaNs
    n_assets = X.shape[1]

    if standardize:
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        X = (X - mu) / std

    if estimator == "ledoitwolf":
        cov = LedoitWolf().fit(X).covariance_
    elif estimator == "oas":
        cov = OAS().fit(X).covariance_
    else:  # ridge
        cov = np.cov(X, rowvar=False, bias=False)
        cov += 1e-3 * np.eye(n_assets)

    # Robust inversion: always add ridge regularization, use pinv as fallback
    cov_reg = cov + 1e-2 * np.eye(n_assets)
    try:
        precision = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov_reg)

    diag = np.diag(precision)
    d = np.sqrt(np.abs(diag))
    d[d < 1e-12] = 1e-12
    pcorr = -precision / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return np.clip(pcorr, -1.0, 1.0)


def corr_to_edges_dense(corr: pd.DataFrame,
                        min_abs_corr: float,
                        sign: str = "both",
                        max_edges_per_node: int | None = None) -> tuple[list, list]:
    """Convert dense correlation matrix to edge list (no overlap filtering).

    Returns: (assets, edges) where edges are 3-tuples (i, j, corr).
    """
    assets = list(corr.columns)
    n = len(assets)
    C = corr.values

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            c = C[i, j]
            if not np.isfinite(c):
                continue
            if abs(c) < min_abs_corr:
                continue
            if sign == "pos" and c < 0:
                continue
            if sign == "neg" and c > 0:
                continue
            edges.append((i, j, float(c)))

    # Prune to max edges per node
    if max_edges_per_node is not None:
        edges.sort(key=lambda e: abs(e[2]), reverse=True)
        deg = np.zeros(n, dtype=np.int32)
        kept = []
        for i, j, c in edges:
            if deg[i] >= max_edges_per_node or deg[j] >= max_edges_per_node:
                continue
            kept.append((i, j, c))
            deg[i] += 1
            deg[j] += 1
        edges = kept

    return assets, edges


def mst_edges_from_corr_dense(corr_mat: np.ndarray) -> list[tuple]:
    """Compute MST edges from dense correlation matrix.

    Returns: [(i, j, corr), ...] - no overlap in tuples.
    """
    C = corr_mat.copy()
    C[~np.isfinite(C)] = 0.0
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    mst = minimum_spanning_tree(D).tocoo()
    edges = []
    for i, j, _w in zip(mst.row, mst.col, mst.data):
        c = float(C[i, j])  # Use original correlation, not distance weight
        edges.append((int(i), int(j), c))
    return edges


def build_edge_universe_dense(corr_df: pd.DataFrame, top_m: int = 10, use_mst: bool = True) -> tuple[list, list]:
    """Build fixed edge universe from dense correlation matrix.

    Returns: (assets, edge_list) where edge_list is [(i,j), ...] sorted pairs.
    """
    assets = list(corr_df.columns)
    C = corr_df.values
    n = len(assets)

    # Per-node top M by abs corr
    keep = set()
    for i in range(n):
        candidates = []
        for j in range(n):
            if i == j:
                continue
            c = C[i, j]
            if not np.isfinite(c):
                continue
            candidates.append((abs(c), i, j))
        candidates.sort(reverse=True)
        for _, a, b in candidates[:top_m]:
            keep.add((min(a, b), max(a, b)))

    # Add MST edges
    if use_mst:
        mst = mst_edges_from_corr_dense(C)
        for i, j, _ in mst:
            keep.add((min(i, j), max(i, j)))

    return assets, sorted(list(keep))


@st.cache_data
def compute_corr_frames_dense(
    filter_key,
    assets: tuple,
    edge_list: tuple,
    method: str,
    window_days: int = 60,
    step_days: int = 5,
    partial_estimator: str = "ledoitwolf",
    partial_standardize: bool = True,
    series_source: str = "norm_pnl",
):
    """Compute correlation weights for fixed edges over rolling windows (dense).

    Each window uses adaptive row selection: keeps top 80% of rows by data completeness.
    This differs from static CorGraph (build_common_matrix) which uses strict dropna.
    The adaptive approach avoids losing too many observations in volatile windows.
    Returns list of frame dicts with: t0, t1, edge_corr (list[float]), node_stats.

    Note: Node XY positions come from keyframes (compute_layout_keyframes), not here.
    """
    # Get all three matrices for node stats
    df_pnl, df_dvol, df_norm, _ = compute_asset_daily_matrices(filter_key)

    # Select which matrix to use for correlation
    if series_source == "pnl":
        dfX = df_pnl
    elif series_source == "daily_vol":
        dfX = df_dvol
    else:
        dfX = df_norm

    assets_list = list(assets)

    # CRITICAL: Enforce exact asset set + order. Do NOT silently drop assets.
    # If any assets are missing, return empty to trigger loud Python-side error.
    missing = [a for a in assets_list if a not in dfX.columns]
    if missing:
        return []
    if len(assets_list) < 2:
        return []

    # Align all three matrices to the same assets (exact order)
    dfX = dfX[assets_list]
    df_pnl_aligned = df_pnl[assets_list] if not df_pnl.empty else dfX
    df_dvol_aligned = df_dvol[assets_list] if not df_dvol.empty else dfX
    df_norm_aligned = df_norm[assets_list] if not df_norm.empty else dfX
    dates = dfX.index
    X = dfX.to_numpy(np.float64)
    n = len(assets_list)
    edge_list = list(edge_list)

    frames = []
    w = window_days
    s = step_days

    for start in range(0, len(dates) - w + 1, s):
        end = start + w
        Xw = X[start:end]

        # Adaptive row selection: keep top 80% of rows by completeness
        # (instead of strict dropna which selects same stable rows each window)
        window_df = pd.DataFrame(Xw, columns=assets_list)
        mask = np.isfinite(Xw)
        cnt = mask.sum(axis=1)
        q = 0.80
        k = max(5, int(q * len(cnt)))
        thr = np.partition(cnt, len(cnt) - k)[len(cnt) - k] if len(cnt) > k else 0
        row_ok = cnt >= thr
        common_window = window_df.loc[row_ok]

        # Get window slices for all three series (using same date range)
        window_dates = dates[start:end]
        pnl_window = df_pnl_aligned.loc[df_pnl_aligned.index.isin(window_dates)]
        dvol_window = df_dvol_aligned.loc[df_dvol_aligned.index.isin(window_dates)]
        norm_window = df_norm_aligned.loc[df_norm_aligned.index.isin(window_dates)]

        if common_window.shape[0] < 2:
            # Not enough common data in this window - use zeros
            edge_corr = [0.0 for _ in edge_list]
            node_stats = {a: {"pnl": 0.0, "dvol": 0.0, "norm": 0.0} for a in assets_list}
        else:
            # Critical: ensure column order matches assets_list (node order)
            assert list(common_window.columns) == assets_list

            # Compute correlation on dense window
            if method in ("pearson", "spearman"):
                C = common_window.corr(method=method).to_numpy()
            elif method == "sign":
                C = compute_sign_corr_dense(common_window)
            else:  # partial
                # LedoitWolf requires strictly dense data (no NaNs)
                partial_window = common_window.dropna(axis=0, how="any")
                if partial_window.shape[0] < 2:
                    # Not enough data after dropna
                    C = np.zeros((len(assets_list), len(assets_list)))
                else:
                    C = compute_partial_corr_dense(partial_window, partial_estimator, partial_standardize)

            # Extract correlations for fixed edges - flat list of floats
            edge_corr = []
            for i, j in edge_list:
                c = C[i, j]
                edge_corr.append(float(c) if np.isfinite(c) else 0.0)

            # Node stats: window means for all three series
            node_stats = {}
            for i, asset in enumerate(assets_list):
                pnl_mean = float(pnl_window[asset].mean()) if asset in pnl_window.columns else 0.0
                dvol_mean = float(dvol_window[asset].mean()) if asset in dvol_window.columns else 0.0
                norm_mean = float(norm_window[asset].mean()) if asset in norm_window.columns else 0.0
                node_stats[asset] = {
                    "pnl": pnl_mean if np.isfinite(pnl_mean) else 0.0,
                    "dvol": dvol_mean if np.isfinite(dvol_mean) else 0.0,
                    "norm": norm_mean if np.isfinite(norm_mean) else 0.0,
                }

        frames.append({
            "t0": dates[start].strftime("%Y-%m-%d"),
            "t1": dates[end-1].strftime("%Y-%m-%d"),
            "edge_corr": edge_corr,
            "node_stats": node_stats,
        })

    return frames


# ============================================================================
# CorGraph State and UI Helpers (for two-column layout with subtabs)
# ============================================================================
@dataclass
class CorGraphState:
    """Computed state for CorGraph visualization (dense matrix approach)."""
    assets: list[str]           # Display asset list
    asset_index: dict[str, int] # {asset_name: index} for O(1) lookup
    corr_sub: pd.DataFrame      # Correlation matrix (dense)
    n_common: int               # Number of common rows (replaces overlap)
    threshold: float            # Computed |corr| threshold
    edges: list[tuple]          # [(i, j, corr), ...] - no overlap in tuple
    G: nx.Graph                 # NetworkX graph
    pos: dict                   # {node_id: (x, y)}
    node_meta: dict             # {node_id: {degree, asset, ...}}
    mst_edge_set: set | None    # MST edges: {(min(i,j), max(i,j)), ...}
    node2comm: dict | None      # {node_id: community_id}
    comms: list | None          # List of community sets
    n_display_assets: int       # Total assets available for slider range


def compute_corgraph_state(filter_key: Any, cfg: dict, timings: dict) -> CorGraphState | None:
    """Compute all CorGraph state from config using dense common-rows matrix.

    Returns None if no data or < 2 common rows.
    """
    # 1. Get all three daily matrices for node stats
    series_source = cfg.get("series_source", "norm_pnl")
    with timed_collect("corgraph_dfX", timings):
        df_pnl, df_dvol, df_norm, _ = compute_asset_daily_matrices(filter_key)

    # Select which matrix to use for correlation based on series_source
    if series_source == "pnl":
        dfX = df_pnl
    elif series_source == "daily_vol":
        dfX = df_dvol
    else:
        dfX = df_norm

    if dfX.empty:
        return None

    # 2. Select display assets (Top N by non-NaN counts, or All)
    col_counts = dfX.notna().sum().sort_values(ascending=False)
    n_display_assets = len(col_counts)

    if cfg["asset_mode"] == "All":
        display_assets = list(col_counts.index)
    else:
        display_assets = col_counts.head(cfg["top_n"]).index.tolist()

    # 3. Build dense common-rows matrix (for the selected series_source)
    with timed_collect("corgraph_common", timings):
        common_df = build_common_matrix(dfX, display_assets)

    n_common = common_df.shape[0]
    if n_common < 2:
        return None  # Error case - not enough common data

    # 3b. Rebuild asset list from common_df.columns (may be subset if some assets missing from dfX)
    display_assets = list(common_df.columns)
    asset_index = {a: i for i, a in enumerate(display_assets)}

    # 3c. Build common matrices for all three series (for node stats)
    # Use the same common rows as the main correlation matrix
    common_rows = common_df.index
    common_pnl = df_pnl.loc[common_rows, display_assets] if not df_pnl.empty else common_df
    common_dvol = df_dvol.loc[common_rows, display_assets] if not df_dvol.empty else common_df
    common_norm = df_norm.loc[common_rows, display_assets] if not df_norm.empty else common_df

    # 4. Compute correlation on dense matrix
    with timed_collect("corgraph_corr", timings):
        if cfg["method"] in ("pearson", "spearman"):
            corr_sub = common_df.corr(method=cfg["method"])
        elif cfg["method"] == "sign":
            C = compute_sign_corr_dense(common_df)
            corr_sub = pd.DataFrame(C, index=common_df.columns, columns=common_df.columns)
        else:  # partial
            C = compute_partial_corr_dense(
                common_df,
                estimator=cfg["partial_estimator"],
                standardize=cfg["partial_standardize"],
            )
            corr_sub = pd.DataFrame(C, index=common_df.columns, columns=common_df.columns)

    # 5. Compute threshold + edges (conditional on edge_mode)
    use_threshold = cfg["edge_mode"] in ("Threshold", "Threshold + MST")
    use_mst = cfg["edge_mode"] in ("Threshold + MST", "MST only")

    # Compute threshold only if needed
    if use_threshold:
        C = corr_sub.values
        mask = np.triu(np.ones_like(C, dtype=bool), k=1)
        finite = np.isfinite(C) & mask
        absvals = np.abs(C[finite])
        threshold = float(np.percentile(absvals, 100 - cfg["top_pct"])) if absvals.size else 0.0
    else:
        threshold = float("nan")  # MST-only mode

    # 6. Compute threshold edges only if needed
    edges = []
    if use_threshold:
        with timed_collect("corgraph_edges", timings):
            _, edges = corr_to_edges_dense(
                corr_sub,
                min_abs_corr=threshold,
                sign=cfg["sign"],
                max_edges_per_node=cfg["max_edges"],
            )

    # 7. Build MST if requested
    mst_edge_set = None
    mst_edges_list = []
    if use_mst:
        with timed_collect("corgraph_mst", timings):
            mst_edges_list = mst_edges_from_corr_dense(corr_sub.values)
            mst_edge_set = {(min(i, j), max(i, j)) for i, j, _ in mst_edges_list}

    # 8. Build final edge set based on mode
    if cfg["edge_mode"] == "Threshold":
        all_edges = list(edges)
        mst_edge_set = None  # No MST styling in threshold-only mode
    elif cfg["edge_mode"] == "MST only":
        all_edges = [(i, j, c) for (i, j, c) in mst_edges_list]
    else:  # "Threshold + MST"
        all_edges = list(edges)
        edge_set = {(min(e[0], e[1]), max(e[0], e[1])) for e in edges}
        for i, j, c in mst_edges_list:
            if (min(i, j), max(i, j)) not in edge_set:
                all_edges.append((i, j, c))

    # 9. Build graph + layout
    with timed_collect("corgraph_graph", timings):

        # Build graph with name attribute for Plotly hover
        G = nx.Graph()
        for i, asset in enumerate(display_assets):
            G.add_node(i, name=asset)
        for edge in all_edges:
            i, j, c = edge[:3]
            G.add_edge(i, j, weight=float(abs(c)), corr=float(c))

        # Node metadata - compute stats from all three series
        node_meta = {}
        for n in G.nodes():
            asset = display_assets[n]
            # Get means from each of the three series
            mean_pnl = float(common_pnl[asset].mean()) if asset in common_pnl.columns else 0.0
            mean_dvol = float(common_dvol[asset].mean()) if asset in common_dvol.columns else 0.0
            mean_norm = float(common_norm[asset].mean()) if asset in common_norm.columns else 0.0
            node_meta[n] = {
                "degree": G.degree(n),
                "asset": asset,
                "mean_daily_pnl": mean_pnl,
                "mean_daily_vol": mean_dvol,
                "mean_norm_pnl": mean_norm,
            }

    # 9. Layout
    with timed_collect("corgraph_layout", timings):
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=cfg["seed"], k=1.5/np.sqrt(max(G.number_of_nodes(), 1)))
        else:
            pos = {}

    # 10. Communities (only on connected nodes to avoid weird isolate "communities")
    node2comm, comms = None, None
    if cfg["detect_communities"] and G.number_of_nodes() > 2:
        with timed_collect("corgraph_communities", timings):
            try:
                H = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()
                if H.number_of_nodes() >= 3 and H.number_of_edges() > 0:
                    comms = list(nx.community.greedy_modularity_communities(H))
                    node2comm = {n: i for i, comm in enumerate(comms) for n in comm}
            except Exception:
                pass

    return CorGraphState(
        assets=display_assets,
        asset_index=asset_index,
        corr_sub=corr_sub,
        n_common=n_common,
        threshold=threshold,
        edges=all_edges,
        G=G,
        pos=pos,
        node_meta=node_meta,
        mst_edge_set=mst_edge_set,
        node2comm=node2comm,
        comms=comms,
        n_display_assets=n_display_assets,
    )


def render_corgraph_controls() -> dict:
    """Render CorGraph controls (simplified - no overlap options).

    Returns dict with all control values.
    """
    cfg = {}

    # Series source (applies to all CorGraph views)
    cfg["series_source"] = st.selectbox(
        "Series",
        ["norm_pnl", "pnl", "daily_vol"],
        format_func=lambda x: {"norm_pnl": "Normalized PnL", "pnl": "PnL", "daily_vol": "Daily Vol"}[x],
        key="cg_series_source",
        help="Time series used for correlations"
    )

    # Row 1: Method + Top %
    c1, c2 = st.columns(2)
    cfg["method"] = c1.selectbox("Method", ["pearson", "spearman", "sign", "partial"], key="cg_method")
    cfg["top_pct"] = c2.slider("Top %", 1, 50, 10, step=1, key="cg_top_pct")

    # Row 2: Edges/node (moved up since min_overlap is gone)
    cfg["max_edges"] = st.slider("Edges/node", 3, 30, 10, key="cg_max_edges")

    # Row 3: Sign
    cfg["sign"] = st.radio("Sign", ["both", "pos", "neg"], horizontal=True, key="cg_sign")

    # Row 4: Checkboxes + Seed
    c1, c2, c3 = st.columns(3)
    cfg["show_labels"] = c1.checkbox("Labels", key="cg_labels")
    cfg["edge_mode"] = c2.selectbox(
        "Edges",
        ["Threshold", "Threshold + MST", "MST only"],
        index=1,  # default to "Threshold + MST"
        key="cg_edge_mode",
    )
    cfg["seed"] = c3.number_input("Seed", 1, 999, 42, key="cg_seed")

    # Row 5: Focus asset - uses session_state for last-known assets
    # Include current focus in options even if stale (avoids session_state modification crash)
    connected_assets = st.session_state.get("cg_connected_assets", [])
    cur = st.session_state.get("cg_focus", "None")

    opts = ["None"] + list(connected_assets)
    if cur not in opts:
        opts.append(cur)  # Keep stale selection so Streamlit doesn't crash

    cfg["focus_asset"] = st.selectbox("Focus", opts, key="cg_focus", disabled=len(connected_assets) == 0)

    # Asset Selection expander
    with st.expander("Asset Selection", expanded=False):
        # Get total asset count from session_state (last-known, like focus selector)
        total_assets = st.session_state.get("cg_total_assets", 100)  # Default 100 until computed

        cfg["asset_mode"] = st.radio("Mode", ["All", "Top N"], horizontal=True, key="cg_asset_mode")

        # Slider always visible, "All" just sets to max
        if cfg["asset_mode"] == "All":
            cfg["top_n"] = total_assets
            st.slider("N", 2, max(2, total_assets), total_assets, step=1, key="cg_top_n_display", disabled=True)
        else:
            default_n = min(60, total_assets) if total_assets > 2 else 2
            cfg["top_n"] = st.slider("N", 2, max(2, total_assets), default_n, step=1, key="cg_top_n")

    # Community Detection expander
    with st.expander("Community Detection", expanded=False):
        cfg["detect_communities"] = st.checkbox("Detect communities", key="cg_communities")
        cfg["color_by_community"] = st.checkbox(
            "Color by community", key="cg_color_comm",
            disabled=not cfg["detect_communities"]
        )

    # Partial Options expander (simplified - no mask option, no NaN fill needed for dense matrix)
    if cfg["method"] == "partial":
        with st.expander("Partial Options", expanded=True):
            cfg["partial_estimator"] = st.selectbox("Estimator", ["ledoitwolf", "oas", "ridge"], key="cg_partial_est")
            cfg["partial_standardize"] = st.checkbox("Standardize", value=True, key="cg_partial_std")
    else:
        cfg["partial_estimator"] = "ledoitwolf"
        cfg["partial_standardize"] = True

    # Animation expander (always visible, note it only affects D3 Living)
    with st.expander("Animation (D3 Living only)", expanded=False):
        st.caption("These settings only affect the D3 Living tab")
        cfg["anim_window"] = st.slider("Window (days)", 30, 120, 60, step=10, key="anim_window")
        cfg["anim_step"] = st.slider("Step (days)", 1, 20, 5, key="anim_step")
        cfg["anim_speed"] = st.slider("Speed (ms)", 200, 2000, 500, step=100, key="anim_speed")
        cfg["anim_max_assets"] = st.slider("Max assets", 40, 120, 80, step=10, key="anim_max_assets")
        cfg["anim_universe_m"] = st.slider("Edges/node", 4, 20, 10, step=2, key="anim_universe_m")

        # D3 physics controls
        st.divider()
        st.caption("D3 physics")
        cfg["anim_charge"] = st.slider("Charge", -500, 0, -180, step=20, key="anim_charge",
            help="More negative = more repulsion between nodes.")
        cfg["anim_collide"] = st.slider("Collide radius", 0, 30, 10, step=1, key="anim_collide",
            help="Node collision radius.")
        cfg["anim_reheat_alpha"] = st.slider("Reheat alpha", 0.3, 0.9, 0.7, step=0.05, key="anim_reheat_alpha",
            help="How much to reheat simulation on frame change. Higher = more responsive, but can overshoot.")

    return cfg


def render_corgraph_metrics_strip(state: CorGraphState):
    """Render compact metrics strip above visualization."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes", f"{state.G.number_of_nodes():,}")
    c2.metric("Edges", f"{state.G.number_of_edges():,}")
    thr_text = "n/a" if not np.isfinite(state.threshold) else f"{state.threshold:.3f}"
    c3.metric("Threshold", thr_text)
    c4.metric("Common rows", f"{state.n_common:,}")
    avg_deg = sum(d for _, d in state.G.degree()) / max(state.G.number_of_nodes(), 1)
    c5.metric("Avg deg", f"{avg_deg:.1f}")


def render_corgraph_plotly_view(state: CorGraphState, cfg: dict):
    """Render Plotly correlation graph."""
    focus_node = state.asset_index.get(cfg["focus_asset"]) if cfg["focus_asset"] != "None" else None

    fig = render_corr_graph_plotly(
        state.G, state.pos, state.assets,
        node_meta=state.node_meta,
        focus_node=focus_node,
        show_labels=cfg["show_labels"],
        node2comm=state.node2comm,
        color_by_community=cfg["color_by_community"]
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_corgraph_d3_interactive_view(state: CorGraphState, cfg: dict):
    """Render D3 Interactive force graph."""
    d3_nodes, d3_links = build_d3_graph_data(
        state.G, state.assets, state.node_meta, state.mst_edge_set,
        node2comm=state.node2comm if cfg["detect_communities"] else None
    )

    if len(d3_links) > 4000:
        st.error(f"Too many edges ({len(d3_links):,}). Reduce edges/node.")
        return

    html = make_d3_force_html(
        d3_nodes, d3_links,
        height=650,
        focus_asset=cfg["focus_asset"] if cfg["focus_asset"] != "None" else None,
        show_labels=cfg["show_labels"],
        color_by_community=cfg["color_by_community"]
    )
    components.html(html, height=670, scrolling=False)
    st.caption("Drag nodes • Scroll to zoom • Drag background to pan")


def render_corgraph_d3_animated_view(filter_key: Any, state: CorGraphState, cfg: dict, timings: dict):
    """Render D3 Living animated network (pure D3 force-simulation, slider-driven).

    Builds edge universe from state.corr_sub (NOT from pruned graph edges).
    Layout is entirely D3-driven - no Python keyframe computation.
    """
    import zlib  # For stable deterministic hash

    # Use animation-specific asset limit
    anim_assets = state.assets[:cfg["anim_max_assets"]]
    n_anim = len(anim_assets)

    if n_anim < 3:
        st.warning("Not enough assets for animation.")
        return

    # Build edge universe from dense correlation
    anim_corr = state.corr_sub.loc[anim_assets, anim_assets]

    with timed_collect("build_edge_universe", timings):
        if cfg["edge_mode"] == "MST only":
            # MST-only: just compute MST on anim_corr (ignore top_m)
            mst_edges = mst_edges_from_corr_dense(anim_corr.values)
            edge_universe = sorted({(min(i, j), max(i, j)) for (i, j, _) in mst_edges})
        else:
            # Threshold or Threshold+MST: use build_edge_universe_dense
            use_mst = cfg["edge_mode"] == "Threshold + MST"
            _, edge_universe = build_edge_universe_dense(
                anim_corr,
                top_m=cfg["anim_universe_m"],
                use_mst=use_mst,
            )

    # Guard for empty edge universe
    if len(edge_universe) == 0:
        st.warning("Edge universe is empty. Try increasing 'Edges/node' in Animation settings.")
        return

    st.caption(f"{n_anim} assets, {len(edge_universe)} edges in universe, {state.n_common} common rows")

    # Compute frame correlations (no keyframes needed - D3 is the only layout engine)
    try:
        with timed_collect("compute_corr_frames_dense", timings):
            anim_frames = compute_corr_frames_dense(
                filter_key=filter_key,
                assets=tuple(anim_assets),
                edge_list=tuple(edge_universe),
                method=cfg["method"],
                window_days=cfg["anim_window"],
                step_days=cfg["anim_step"],
                partial_estimator=cfg["partial_estimator"],
                partial_standardize=cfg["partial_standardize"],
                series_source=cfg.get("series_source", "norm_pnl"),
            )
    except Exception as e:
        st.error(f"Animation frame computation failed: {type(e).__name__}")
        st.exception(e)
        return

    if not anim_frames:
        st.warning("No frames generated.")
        return

    st.caption(f"frames={len(anim_frames)}")

    # Build nodes with DETERMINISTIC initial positions from state.pos
    # (Critical: iframe reloads on any Streamlit rerun)
    pos_by_asset = {state.assets[i]: state.pos[i] for i in state.pos.keys() if i < len(state.assets)}

    # Scale positions to pixel coordinates
    if pos_by_asset:
        xs = [p[0] for p in pos_by_asset.values()]
        ys = [p[1] for p in pos_by_asset.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1

    margin = 80
    anim_width = 900
    anim_height = 650

    def scale_pos(p):
        x = margin + (p[0] - x_min) / (x_max - x_min + 1e-9) * (anim_width - 2*margin)
        y = margin + (p[1] - y_min) / (y_max - y_min + 1e-9) * (anim_height - 2*margin)
        return (x, y)

    def stable_hash_pos(asset: str) -> tuple[float, float]:
        """Deterministic fallback position using stable hash (not Python's salted hash)."""
        h = zlib.adler32(asset.encode("utf-8"))
        px = anim_width / 2 + (h % 200 - 100)
        py = anim_height / 2 + ((h // 200) % 200 - 100)
        return px, py

    anim_nodes = []
    for i, asset in enumerate(anim_assets):
        deg = sum(1 for a, b in edge_universe if a == i or b == i)

        # Initial position from full-period layout (if available), else stable fallback
        if asset in pos_by_asset:
            px, py = scale_pos(pos_by_asset[asset])
        else:
            px, py = stable_hash_pos(asset)

        anim_nodes.append({
            "id": asset,
            "x": px,
            "y": py,
            "degree": deg,
        })

    # Build edge list for D3 (CRITICAL: maintain index alignment with frames)
    d3_edges = [{"source": anim_assets[i], "target": anim_assets[j]} for i, j in edge_universe]

    # Render living force-directed network (pure D3, no keyframes)
    with timed_collect("render_d3_animated", timings):
        html = make_d3_animated_html(
            nodes=anim_nodes,
            edge_list=d3_edges,
            frames=anim_frames,
            height=650,
            show_labels=cfg["show_labels"],
            play_speed_ms=cfg["anim_speed"],
            charge=cfg["anim_charge"],
            collide=cfg["anim_collide"],
            reheat_alpha=cfg["anim_reheat_alpha"],
            weight_power=WEIGHT_POWER,
        )
        components.html(html, height=720, scrolling=False)

    st.caption(f"Frames: {len(anim_frames)} | Edges: {len(edge_universe)}")


def detect_communities(G: nx.Graph, seed: int = 42) -> tuple[dict, list]:
    """Detect communities using Louvain or greedy modularity.

    Returns: (node2community_dict, list_of_community_sets)
    """
    from networkx.algorithms import community as nx_comm

    if G.number_of_nodes() == 0:
        return {}, []

    # Prefer Louvain if available (nx >= 2.7)
    if hasattr(nx_comm, "louvain_communities"):
        comms = list(nx_comm.louvain_communities(G, weight="weight", seed=seed))
    else:
        comms = list(nx_comm.greedy_modularity_communities(G, weight="weight"))

    # Map node -> community id
    node2c = {}
    for cid, nodes in enumerate(comms):
        for u in nodes:
            node2c[u] = cid
    return node2c, comms


def community_summary(comms: list, assets: list, node_meta: dict) -> pd.DataFrame:
    """Build summary table for each community."""
    rows = []
    for cid, nodes in enumerate(comms):
        node_list = list(nodes)
        size = len(node_list)
        names = [assets[n] for n in node_list if n < len(assets)]

        # Top 3 names as label
        top3 = names[:3]
        label = ", ".join(top3) + ("..." if size > 3 else "")

        # Mean metrics from node_meta
        pnls = [node_meta.get(n, {}).get("mean_daily_pnl", np.nan) for n in node_list]
        ratios = [node_meta.get(n, {}).get("pnl_over_daily_vol", np.nan) for n in node_list]

        mean_pnl = np.nanmean(pnls) if pnls else np.nan
        mean_ratio = np.nanmean(ratios) if ratios else np.nan

        rows.append({
            "id": cid,
            "size": size,
            "members": label,
            "mean_daily_pnl": mean_pnl,
            "mean_pnl_ratio": mean_ratio,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Embedding Helpers (UMAP, t-SNE, MDS)
# ============================================================================
@st.cache_data
def compute_embedding(filter_key, method: str, corr_method: str, corr_min_periods: int,
                      distance_mode: str, n_assets: int | None, min_coverage_days: int,
                      perplexity: int = 30, n_neighbors: int = 15, min_dist: float = 0.1,
                      seed: int = 42,
                      partial_fill: str = "median", partial_estimator: str = "ledoitwolf",
                      partial_standardize: bool = True, partial_mask: bool = True,
                      series_source: str = "norm_pnl"):
    """Compute 2D embedding from correlation distance matrix.

    Args:
        corr_min_periods: min_periods for .corr() - keep low (5-10) to avoid NaN-heavy matrix
        min_coverage_days: exclude assets with fewer than this many days of data
        partial_*: parameters for partial correlation method
    """
    # Get correlation with small min_periods to avoid excessive NaNs
    # Note: compute_corr_and_overlap returns 5 values; ignore debug_info
    corr_full, overlap_full, all_assets, coverage, _ = compute_corr_and_overlap(
        filter_key, corr_method, corr_min_periods,
        partial_fill=partial_fill,
        partial_estimator=partial_estimator,
        partial_standardize=partial_standardize,
        partial_mask=partial_mask,
        series_source=series_source,
    )

    if corr_full.empty:
        return np.empty((0, 2)), [], pd.Series(dtype=np.int32)

    # Filter assets by minimum coverage days (use coverage.index as canonical source)
    if min_coverage_days > 0:
        valid_assets = coverage[coverage >= min_coverage_days].index.tolist()
    else:
        valid_assets = list(coverage.index)

    # Select top N assets by coverage if requested
    if n_assets is not None and n_assets < len(valid_assets):
        selected = coverage.loc[valid_assets].sort_values(ascending=False).head(n_assets).index.tolist()
    else:
        selected = valid_assets

    if len(selected) < 3:
        return np.empty((0, 2)), selected, coverage.loc[selected] if selected else pd.Series(dtype=np.int32)

    corr_sub = corr_full.loc[selected, selected]
    C = corr_sub.values.copy()

    # Handle NaN: set to 0 correlation (distance = 1)
    C[~np.isfinite(C)] = 0.0

    # Convert correlation to distance
    if distance_mode == "1-abs(corr)":
        D = 1.0 - np.abs(C)
    else:  # "1-corr"
        D = 1.0 - C

    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    # Symmetrize for algorithm stability
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    n = len(selected)

    # Run embedding algorithm
    if method == "UMAP":
        nn = max(2, min(n_neighbors, n - 1))  # UMAP needs n_neighbors >= 2
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=nn,
            min_dist=min_dist,
            random_state=seed,
        )
        coords = reducer.fit_transform(D)
    elif method == "t-SNE":
        # Proper perplexity clamping: perplexity <= (n-1)/3 for stability
        pmax = max(2, (n - 1) // 3)
        perp = min(perplexity, pmax)
        reducer = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perp,
            random_state=seed,
            init="random",
            learning_rate="auto",
            n_iter=1000,
        )
        coords = reducer.fit_transform(D)
    else:  # MDS
        reducer = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            n_init=4,
            max_iter=300,
        )
        coords = reducer.fit_transform(D)

    # Normalize coordinates for consistent scale across methods
    coords = coords - coords.mean(axis=0, keepdims=True)
    s = coords.std(axis=0, keepdims=True) + 1e-9
    coords = coords / s

    # Return coverage for selected assets (for coloring)
    return coords, selected, coverage.loc[selected]


def render_embedding_plotly(coords: np.ndarray, assets: list,
                            color_by: str = "none",
                            node_meta: dict = None,
                            show_labels: bool = False):
    """Render 2D embedding as Plotly scatter."""
    n = len(assets)

    # Build color array
    if color_by == "none" or node_meta is None:
        colors = ["#2962FF"] * n
        colorbar = None
    else:
        # Color by metric from node_meta
        vals = []
        for i, a in enumerate(assets):
            if node_meta and a in node_meta:
                v = node_meta[a].get(color_by, 0.0)
                vals.append(v if np.isfinite(v) else 0.0)
            else:
                vals.append(0.0)
        # Suppress colorbar if all values are identical (avoids pointless gradient)
        if len(set(vals)) <= 1:
            colors = ["#2962FF"] * n
            colorbar = None
        else:
            colors = vals
            colorbar = dict(title=color_by.replace("_", " "))

    # Build hover text
    hover_texts = []
    for i, a in enumerate(assets):
        parts = [f"<b>{a}</b>"]
        if node_meta and a in node_meta:
            m = node_meta[a]
            for k, v in m.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
        hover_texts.append("<br>".join(parts))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=10,
            color=colors,
            colorscale="Viridis" if color_by != "none" else None,
            colorbar=colorbar,
            line=dict(width=1, color="white"),
        ),
        text=assets if show_labels else None,
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>",
        textposition="top center",
        textfont=dict(size=8),
        showlegend=False,
    ))

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Asset Embedding",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        hovermode="closest",
    )

    return fig


def render_corr_graph_plotly(G: nx.Graph, pos: dict, assets: list,
                             node_meta: dict = None,
                             focus_node: int = None,
                             show_labels: bool = False,
                             node2comm: dict = None,
                             color_by_community: bool = False):
    """Render correlation network with Plotly (optimized edge rendering)."""
    import plotly.express as px

    # Build edge coordinates grouped by type (pos/neg/dimmed)
    pos_x, pos_y = [], []
    neg_x, neg_y = [], []
    dim_x, dim_y = [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        corr = data.get("corr", 0)

        # Determine which trace this edge belongs to
        if focus_node is not None and u != focus_node and v != focus_node:
            dim_x.extend([x0, x1, None])
            dim_y.extend([y0, y1, None])
        elif corr > 0:
            pos_x.extend([x0, x1, None])
            pos_y.extend([y0, y1, None])
        else:
            neg_x.extend([x0, x1, None])
            neg_y.extend([y0, y1, None])

    # Build edge traces (max 3 traces instead of 1 per edge)
    edge_traces = []
    if pos_x:
        edge_traces.append(go.Scatter(
            x=pos_x, y=pos_y, mode="lines",
            line=dict(width=1.5, color="#26a69a"),  # green
            hoverinfo="skip", showlegend=False,
        ))
    if neg_x:
        edge_traces.append(go.Scatter(
            x=neg_x, y=neg_y, mode="lines",
            line=dict(width=1.5, color="#ef5350"),  # red
            hoverinfo="skip", showlegend=False,
        ))
    if dim_x:
        edge_traces.append(go.Scatter(
            x=dim_x, y=dim_y, mode="lines",
            line=dict(width=0.5, color="rgba(200,200,200,0.3)"),
            hoverinfo="skip", showlegend=False,
        ))

    # Build node trace
    node_x, node_y, node_labels, node_hover, node_color = [], [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        name = data.get("name", str(n))
        node_labels.append(name)  # Short label for visible text

        # Build detailed hover text
        hover_parts = [f"<b>{name}</b>", f"neighbors: {G.degree(n)}"]
        if node_meta and n in node_meta:
            m = node_meta[n]
            hover_parts.append(f"mean daily pnl: {m.get('mean_daily_pnl', float('nan')):.6f}")
            hover_parts.append(f"mean daily vol: {m.get('mean_daily_vol', float('nan')):.6f}")
            hover_parts.append(f"mean pnl/dvol: {m.get('mean_norm_pnl', float('nan')):.3f}")
        node_hover.append("<br>".join(hover_parts))

        # Color: highlight focus node and neighbors, or by community
        if focus_node is not None:
            if n == focus_node:
                node_color.append("#2962FF")  # blue
            elif G.has_edge(n, focus_node):
                node_color.append("#FF6D00")  # orange (neighbor)
            else:
                node_color.append("rgba(200,200,200,0.5)")
        elif color_by_community and node2comm:
            # Use community as color
            cmap = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
            node_color.append(cmap[node2comm.get(n, 0) % len(cmap)])
        else:
            node_color.append("#2962FF")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=12, color=node_color, line=dict(width=1, color="white")),
        text=node_labels if show_labels else None,  # Visible labels (short)
        hovertext=node_hover,  # Detailed hover (long)
        hovertemplate="%{hovertext}<extra></extra>",
        textposition="top center",
        textfont=dict(size=8),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Correlation Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
    )

    return fig


def build_d3_graph_data(G: nx.Graph, assets: list, node_meta: dict,
                        mst_edge_set: set | None = None,
                        node2comm: dict | None = None) -> tuple[list, list]:
    """Build nodes and links JSON for D3 force graph.

    Args:
        G: NetworkX graph with nodes (indexed by int) and edges
        assets: List of asset names (index i -> assets[i])
        node_meta: Dict of node_index -> metrics dict
        mst_edge_set: Set of (i, j) tuples marking MST edges (normalized i < j)
        node2comm: Dict of node_index -> community_id

    Returns:
        (nodes, links) suitable for JSON serialization
    """
    nodes = []
    for n, data in G.nodes(data=True):
        name = assets[n]
        node = {
            "id": name,
            "degree": G.degree(n),
        }
        if node_meta and n in node_meta:
            node.update(node_meta[n])
        if node2comm:
            node["community"] = node2comm.get(n, 0)
        nodes.append(node)

    links = []
    for u, v, data in G.edges(data=True):
        corr = data.get("corr", 0)
        # Normalize edge key for MST lookup
        edge_key = (min(u, v), max(u, v))
        is_mst = mst_edge_set is not None and edge_key in mst_edge_set
        links.append({
            "source": assets[u],
            "target": assets[v],
            "corr": float(corr),
            "strength": abs(float(corr)),
            "mst": is_mst,
        })

    return nodes, links


def make_d3_force_html(nodes: list, links: list, height: int = 700,
                       focus_asset: str | None = None,
                       show_labels: bool = False,
                       color_by_community: bool = False) -> str:
    """Generate self-contained HTML with D3 force-directed graph.

    Features:
    - Drag nodes (simulation continues)
    - Zoom/pan (scroll wheel + drag background)
    - Hover tooltips (asset name + metrics)
    - Focus mode (dim non-adjacent nodes/edges)
    - Edge colors: green (positive), red (negative)
    - MST edges: dashed style
    - Community coloring (optional)
    """
    import json

    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    focus_json = json.dumps(focus_asset) if focus_asset else "null"
    show_labels_js = "true" if show_labels else "false"
    color_by_community_js = "true" if color_by_community else "false"

    html = f'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}
svg {{ width: 100%; height: {height}px; }}
.node {{ cursor: grab; }}
.node:active {{ cursor: grabbing; }}
.node-label {{ font-size: 8px; pointer-events: none; }}
.tooltip {{
    position: absolute;
    background: rgba(0,0,0,0.85);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    max-width: 250px;
    z-index: 1000;
}}
.link {{ stroke-opacity: 0.6; }}
.link.positive {{ stroke: #26a69a; }}
.link.negative {{ stroke: #ef5350; }}
.link.mst {{ stroke-dasharray: 4,2; }}
.link.dimmed {{ stroke: #ccc; stroke-opacity: 0.15; }}
.node.dimmed {{ opacity: 0.2; }}
.node.focus {{ stroke: #2962FF; stroke-width: 3px; }}
.node.neighbor {{ stroke: #FF6D00; stroke-width: 2px; }}
</style>
</head>
<body>
<div id="tooltip" class="tooltip" style="display:none;"></div>
<svg></svg>
<script>
const nodes = {nodes_json};
const links = {links_json};
const focusAsset = {focus_json};
const showLabels = {show_labels_js};
const colorByCommunity = {color_by_community_js};
const communityColors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf","#999999","#66c2a5","#fc8d62","#8da0cb"];

// Helper: D3 mutates link source/target from strings to objects after forceLink runs
// Use these to safely get the id regardless of current state
function sid(s) {{ return (typeof s === "object") ? s.id : s; }}
function tid(t) {{ return (typeof t === "object") ? t.id : t; }}

// Build neighbor lookup for focus mode (using helpers for robustness)
const neighbors = new Map();
nodes.forEach(n => neighbors.set(n.id, new Set()));
links.forEach(l => {{
    const s = sid(l.source);
    const t = tid(l.target);
    neighbors.get(s).add(t);
    neighbors.get(t).add(s);
}});

const svg = d3.select("svg");

// Fixed height from Python; bbox width with sanity clamp (can be 0 on first tick in iframe)
const fixedHeight = {height};
const bbox = svg.node().getBoundingClientRect();
const width = (bbox.width > 10) ? bbox.width : 1100;
const height = fixedHeight;

const g = svg.append("g");

// Zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.1, 4])
    .on("zoom", (event) => g.attr("transform", event.transform));
svg.call(zoom);

// Force simulation
// Note: clamp strength to [0,1] defensively (correlation can drift slightly above 1)
// Distance has floor of 30 to prevent nodes from collapsing when strength ~1
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id)
        .distance(d => {{ const s = Math.max(0, Math.min(1, d.strength)); return 30 + 120 * (1 - s); }})
        .strength(d => {{ const s = Math.max(0, Math.min(1, d.strength)); return 0.3 + 0.7 * s; }}))
    .force("charge", d3.forceManyBody().strength(-150))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(20));

// Draw links
const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", d => {{
        const s = sid(d.source);
        const t = tid(d.target);
        let cls = "link";
        cls += d.corr > 0 ? " positive" : " negative";
        if (d.mst) cls += " mst";
        if (focusAsset && s !== focusAsset && t !== focusAsset) cls += " dimmed";
        return cls;
    }})
    .attr("stroke-width", d => 1 + 2 * d.strength);

// Draw nodes
const node = g.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", d => {{
        let cls = "node";
        if (focusAsset) {{
            if (d.id === focusAsset) cls += " focus";
            else if (neighbors.get(focusAsset)?.has(d.id)) cls += " neighbor";
            else cls += " dimmed";
        }}
        return cls;
    }})
    .attr("r", d => 6 + Math.min(d.degree, 20) * 0.4)
    .attr("fill", d => {{
        if (focusAsset) {{
            if (d.id === focusAsset) return "#2962FF";
            if (neighbors.get(focusAsset)?.has(d.id)) return "#FF6D00";
            return "#ccc";
        }}
        if (colorByCommunity && d.community !== undefined) {{
            return communityColors[d.community % communityColors.length];
        }}
        return "#2962FF";
    }})
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .call(drag(simulation));

// Labels (optional)
let labels;
if (showLabels) {{
    labels = g.append("g")
        .selectAll("text")
        .data(nodes)
        .join("text")
        .attr("class", "node-label")
        .attr("dy", -10)
        .attr("text-anchor", "middle")
        .text(d => d.id);
}}

// Tooltip (using <br> for reliable line breaks with .html())
// Use clientX/clientY instead of pageX/pageY for better iframe behavior
const tooltip = d3.select("#tooltip");
node.on("mouseover", (event, d) => {{
    let html = `<b>${{d.id}}</b><br>Degree: ${{d.degree}}`;
    if (d.coverage !== undefined) html += `<br>Coverage: ${{d.coverage}}`;
    if (d.mean_daily_pnl !== undefined) html += `<br>Mean daily PnL: ${{d.mean_daily_pnl.toFixed(6)}}`;
    if (d.mean_daily_vol !== undefined) html += `<br>Mean daily vol: ${{d.mean_daily_vol.toFixed(6)}}`;
    if (d.mean_norm_pnl !== undefined) html += `<br>Mean PnL/dVol: ${{d.mean_norm_pnl.toFixed(3)}}`;
    tooltip.style("display", "block").html(html);
}})
.on("mousemove", (event) => {{
    tooltip.style("left", (event.clientX + 15) + "px")
           .style("top", (event.clientY - 10) + "px");
}})
.on("mouseout", () => tooltip.style("display", "none"));

// Simulation tick
simulation.on("tick", () => {{
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    if (showLabels) {{
        labels
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    }}
}});

// Drag behavior
function drag(simulation) {{
    return d3.drag()
        .on("start", (event, d) => {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }})
        .on("drag", (event, d) => {{
            d.fx = event.x;
            d.fy = event.y;
        }})
        .on("end", (event, d) => {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }});
}}
</script>
</body>
</html>
'''
    return html


def make_d3_animated_html(
    nodes: list,           # [{"id": str, "x": float, "y": float, "degree": int}, ...]
    edge_list: list,       # [{"source": str, "target": str}, ...]
    frames: list,          # [{edge_corr: [...], node_stats: {asset: {pnl, dvol, norm}}, t0: str, t1: str}, ...]
    height: int = 700,
    show_labels: bool = False,
    play_speed_ms: int = 500,
    charge: int = -180,
    collide: int = 10,
    reheat_alpha: float = 0.7,
    weight_power: float = 1.6,
) -> str:
    """Generate D3 HTML for living force-directed correlation network.

    Pure slider-driven: D3 is the ONLY layout engine. No keyframes, no target positions.
    - Slider changes edge weights (correlations)
    - Simulation reheats and settles naturally
    - Edge strength/distance drive layout entirely
    """
    import json

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edge_list)
    frames_json = json.dumps(frames)

    html = f'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, sans-serif; }}
  .controls {{ padding: 10px; background: #f5f5f5; display: flex; flex-direction: column; gap: 8px; }}
  .controls button {{ padding: 8px 16px; font-size: 14px; cursor: pointer; }}
  .controls input[type="range"] {{ width: 300px; }}
  .frame-info {{ font-size: 14px; color: #333; min-width: 200px; }}
  .link {{ fill: none; }}
  .node {{ stroke: #fff; stroke-width: 1.5px; cursor: grab; }}
  .node:active {{ cursor: grabbing; }}
  .node-label {{ font-size: 9px; text-anchor: middle; pointer-events: none; }}
  .tooltip {{ position: absolute; background: rgba(0,0,0,0.8); color: white;
              padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; }}
  .wrap {{ display:flex; width:100%; height: {height}px; }}
  #graph {{ flex: 1 1 auto; min-width: 0; }}
  #side {{ width: 280px; border-left: 1px solid #ddd; background:#fff; padding:10px; overflow-y:auto; flex-shrink:0; }}
  .side-title {{ font-size:12px; font-weight:600; color:#444; margin:6px 0; }}
  .side-meta {{ font-size:12px; color:#333; margin-bottom:8px; }}
  .side-list {{ font-size:12px; }}
  .side-row {{ display:flex; gap:8px; padding:3px 0; border-bottom:1px solid #f0f0f0; }}
  .side-row .lhs {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .side-row .rhs {{ width:50px; text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }}
  .side-row.pos {{ color: #26a69a; }}
  .side-row.neg {{ color: #ef5350; }}
  .side-row.side-header {{ color:#666; border-bottom:2px solid #ddd; }}
</style>
</head>
<body>
<div class="controls">
  <!-- Line 1: Play and slider -->
  <div style="display:flex;align-items:center;gap:15px;width:100%;">
    <button id="playPause">Play</button>
    <input type="range" id="frameSlider" min="0" max="0" value="0" style="flex:1;max-width:400px;">
    <label style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="frameRelink" checked>
      FrameRelink
    </label>
    <span class="frame-info" id="frameInfo">Loading...</span>
    <label style="display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="freeze">
      Freeze
    </label>
  </div>
  <!-- Line 2: Buttons -->
  <div style="display:flex;align-items:center;gap:10px;width:100%;">
    <button id="relayout">Re-layout</button>
    <button id="kickBtn">Kick</button>
    <button id="noSpeed">NoSpeed</button>
    <button id="heatUp">Heat up</button>
    <button id="relink">Re-link</button>
  </div>
  <!-- Line 3: All sliders -->
  <div style="display:flex;align-items:center;gap:15px;width:100%;flex-wrap:wrap;">
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">Kick</span>
      <input type="range" id="kickStrength" min="0" max="500" step="1" value="2" style="width:140px;">
      <span id="kickVal" style="font-size:12px;color:#555;min-width:40px;text-align:right;">2</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">Smooth</span>
      <input type="range" id="corrSmooth" min="0.0" max="1.0" step="0.01" value="0.08" style="width:100px;">
      <span id="corrSmoothVal" style="font-size:12px;color:#555;min-width:36px;text-align:right;">0.08</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">αDecay</span>
      <input type="range" id="alphaDecay" min="0" max="0.1" step="0.001" value="0.02" style="width:100px;">
      <span id="alphaDecayVal" style="font-size:12px;color:#555;min-width:44px;text-align:right;">0.020</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">αMin</span>
      <input type="range" id="alphaMin" min="0.0001" max="0.02" step="0.0001" value="0.001" style="width:100px;">
      <span id="alphaMinVal" style="font-size:12px;color:#555;min-width:50px;text-align:right;">0.0010</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">αTarget</span>
      <input type="range" id="alphaTarget" min="0.0" max="1.0" step="0.01" value="0.0" style="width:100px;">
      <span id="alphaTargetVal" style="font-size:12px;color:#555;min-width:36px;text-align:right;">0.00</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">lenMin</span>
      <input type="range" id="lengthMin" min="5" max="200" step="1" value="40" style="width:80px;">
      <span id="lengthMinVal" style="font-size:12px;color:#555;min-width:30px;text-align:right;">40</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">lenSlope</span>
      <input type="range" id="lengthSlope" min="0" max="800" step="5" value="260" style="width:80px;">
      <span id="lengthSlopeVal" style="font-size:12px;color:#555;min-width:30px;text-align:right;">260</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">strMin</span>
      <input type="range" id="strengthMin" min="0" max="0.2" step="0.005" value="0.02" style="width:80px;">
      <span id="strengthMinVal" style="font-size:12px;color:#555;min-width:36px;text-align:right;">0.020</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">strSlope</span>
      <input type="range" id="strengthSlope" min="0" max="1.0" step="0.01" value="0.23" style="width:80px;">
      <span id="strengthSlopeVal" style="font-size:12px;color:#555;min-width:36px;text-align:right;">0.23</span>
    </label>
    <label style="display:flex;align-items:center;gap:6px;">
      <span style="font-size:12px;color:#555;">strPow</span>
      <input type="range" id="strengthPower" min="0.2" max="4.0" step="0.1" value="1.6" style="width:80px;">
      <span id="strengthPowerVal" style="font-size:12px;color:#555;min-width:30px;text-align:right;">1.6</span>
    </label>
  </div>
</div>
<div class="wrap">
  <div id="graph"></div>
  <div id="side">
    <div class="side-title">Selection</div>
    <div id="selMeta" class="side-meta">Click a node…</div>
    <div class="side-title">Correlations (neighbors)</div>
    <div id="corrList" class="side-list"></div>
  </div>
</div>
<div class="tooltip" id="tooltip" style="display:none;"></div>

<script>
const nodes = {nodes_json};
const edges = {edges_json};
const frames = {frames_json};
const playSpeedMs = {play_speed_ms};
const reheatAlpha = {reheat_alpha};
const showLabels = {"true" if show_labels else "false"};

// Physics params (injected from Python — single source of truth)
const WEIGHT_POWER = {weight_power};
const CHARGE = {charge};
const COLLIDE = {collide};

// Link physics parameters (live-adjustable via sliders)
let lengthMin = 40;
let lengthSlope = 260;
let strengthMin = 0.02;
let strengthSlope = 0.23;
let strengthPower = WEIGHT_POWER;

// Safe correlation clamping: handles NaN, out-of-range
function abs01(x) {{
  if (!Number.isFinite(x)) return 0;
  const a = Math.abs(x);
  return Math.max(0, Math.min(1, a));
}}

// Build links with index alignment (CRITICAL: frames[idx].edge_corr[i] corresponds to links[i])
const links = edges.map((e, i) => ({{
  source: e.source,
  target: e.target,
  index: i,
  corr: 0,
  targetCorr: 0,
}}));

// Build adjacency: id -> array of link indices (for side panel)
const adj = new Map();
for (const n of nodes) adj.set(n.id, []);
for (let i = 0; i < links.length; i++) {{
  const s = links[i].source;
  const t = links[i].target;
  adj.get(s).push(i);
  adj.get(t).push(i);
}}

// Helper: get node id from source/target (handles D3 object mutation)
function nid(x) {{ return (typeof x === "object") ? x.id : x; }}

// Sanity check: verify frame/edge alignment at startup
if (frames.length > 0 && frames[0].edge_corr.length !== edges.length) {{
  document.getElementById("frameInfo").innerHTML =
    `<span style="color:red;font-weight:bold">ERROR: edge_corr.length (${{frames[0].edge_corr.length}}) ≠ edges.length (${{edges.length}})</span>`;
}}

if (frames.length === 0) {{
  document.getElementById('frameInfo').textContent = 'No frames available';
}} else {{
  // Use container bbox width instead of window.innerWidth (iframe-safe)
  const bbox = document.getElementById("graph").getBoundingClientRect();
  const width = (bbox.width > 10) ? bbox.width : window.innerWidth;
  const height = {height};

  const svg = d3.select("#graph")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g");

  // Zoom
  svg.call(d3.zoom()
    .scaleExtent([0.1, 4])
    .on("zoom", (event) => g.attr("transform", event.transform)));

  // Links group (below nodes)
  const linkGroup = g.append("g").attr("class", "links");
  const nodeGroup = g.append("g").attr("class", "nodes");

  // Draw links (persistent - never removed)
  const linkSelection = linkGroup.selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "link")
    .attr("stroke", "#ccc")
    .attr("stroke-width", 1)
    .attr("opacity", 0);

  // Base radius function
  function baseRadius(d) {{
    return 6 + Math.sqrt(d.degree || 1) * 1.5;
  }}

  // Draw nodes
  const nodeSelection = nodeGroup.selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "node")
    .attr("r", baseRadius)
    .attr("fill", "#1f77b4")
    .on("mouseover", showTooltip)
    .on("mouseout", hideTooltip)
    .on("click", (event, d) => selectNode(d.id));

  // Labels (controlled by sidebar setting)
  const labelSelection = nodeGroup.selectAll("text")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .attr("dy", -12)
    .text(d => d.id)
    .style("display", showLabels ? "block" : "none");

  // Link physics helper functions (use live slider parameters)
  function linkRestLen(c) {{
    const a = abs01(c);
    return lengthMin + lengthSlope * (1 - a);
  }}

  function linkStrength(c) {{
    const a = abs01(c);
    return strengthMin + strengthSlope * Math.pow(a, strengthPower);
  }}

  // Force simulation - NO forceX/forceY targets (pure D3)
  const linkForce = d3.forceLink(links)
    .id(d => d.id)
    .distance(d => linkRestLen(d.corr))
    .strength(d => linkStrength(d.corr));

  const simulation = d3.forceSimulation(nodes)
    .force("link", linkForce)
    .force("charge", d3.forceManyBody().strength(CHARGE))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(COLLIDE))
    .alphaDecay(0.02)
    .on("tick", tick);

  // Attach drag
  nodeSelection.call(drag(simulation));

  // One-shot render for frozen mode (updates visuals without physics)
  function renderEdgesAndNodes() {{
    linkSelection
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y)
      .attr("stroke", d => d.corr >= 0 ? "#26a69a" : "#ef5350")
      .attr("stroke-width", d => 1 + 3 * abs01(d.corr))
      .attr("opacity", d => {{
        const a = abs01(d.corr);
        return a < 0.05 ? 0 : Math.pow((a - 0.05) / 0.95, 1.3) * 0.8;
      }});
  }}

  function tick() {{
    // Per-tick smoothing of edge weights toward target (controlled by Smooth slider)
    links.forEach(d => {{
      d.corr += corrSmooth * (d.targetCorr - d.corr);
      // Snap when close enough to ensure exact convergence
      if (Math.abs(d.targetCorr - d.corr) < 1e-4) d.corr = d.targetCorr;
    }});

    // Update link visuals
    linkSelection
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y)
      .attr("stroke", d => d.corr >= 0 ? "#26a69a" : "#ef5350")
      .attr("stroke-width", d => 1 + 3 * abs01(d.corr))
      .attr("opacity", d => {{
        const a = abs01(d.corr);
        return a < 0.05 ? 0 : Math.pow((a - 0.05) / 0.95, 1.3) * 0.8;
      }});

    nodeSelection
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    labelSelection
      .attr("x", d => d.x)
      .attr("y", d => d.y);

    // Refresh side panel with live position/velocity data
    refreshSidePanel();
  }}

  // Animation state
  let currentFrame = 0;
  let playing = false;
  let playInterval = null;
  let selectedId = null;
  let corrSmooth = 0.08;  // correlation smoothing factor (controlled by Smooth slider)

  // Throttled side panel refresh (max 10x/sec during ticks)
  let lastSideRefresh = 0;
  function refreshSidePanel() {{
    if (!selectedId) return;
    const now = performance.now();
    if (now - lastSideRefresh < 100) return;
    lastSideRefresh = now;
    selectNode(selectedId);
  }}

  const slider = document.getElementById("frameSlider");
  slider.max = frames.length - 1;

  function updateFrame(idx) {{
    currentFrame = idx;
    const frame = frames[idx];
    slider.value = idx;

    // Update info display
    document.getElementById("frameInfo").textContent = `${{frame.t0}} → ${{frame.t1}}`;

    // Update target correlations (index-aligned!)
    frame.edge_corr.forEach((c, i) => {{
      links[i].targetCorr = c;
    }});

    // If FrameRelink is checked, immediately snap corr and reinitialize link force
    if (document.getElementById("frameRelink").checked) {{
      links.forEach(d => {{ d.corr = d.targetCorr; }});
      linkForce.links(links);
      nodes.forEach(n => {{ n.vx = 0; n.vy = 0; }});
    }}

    // Update node sizes based on abs(norm) from node_stats
    const normVals = Object.values(frame.node_stats).map(s => Math.abs(s.norm || 0));
    const maxNorm = Math.max(...normVals, 0.001);
    nodeSelection.attr("r", d => {{
      const stats = frame.node_stats[d.id] || {{norm: 0}};
      return baseRadius(d) * (1 + 0.4 * Math.abs(stats.norm) / maxNorm);
    }});

    // Handle frozen vs normal mode
    const frozen = document.getElementById("freeze").checked;
    if (frozen) {{
      // Immediately apply corr, render once (no physics)
      links.forEach(d => {{ d.corr = d.targetCorr; }});
      renderEdgesAndNodes();
    }} else {{
      // Normal: reheat simulation (alphaDecay handles cooling)
      simulation.alpha(Math.min(reheatAlpha, 0.9)).restart();
    }}

    // Update side panel if a node is selected
    if (selectedId) selectNode(selectedId);
  }}

  function showTooltip(event, d) {{
    const frame = frames[currentFrame];
    const stats = frame.node_stats[d.id] || {{pnl: 0, dvol: 0, norm: 0}};
    document.getElementById("tooltip").innerHTML =
      `<strong>${{d.id}}</strong><br>Degree: ${{d.degree || 0}}<br>` +
      `PnL: ${{stats.pnl.toFixed(6)}}<br>` +
      `Vol: ${{stats.dvol.toFixed(6)}}<br>` +
      `PnL/Vol: ${{stats.norm.toFixed(3)}}`;
    document.getElementById("tooltip").style.display = "block";
    document.getElementById("tooltip").style.left = (event.clientX + 10) + "px";
    document.getElementById("tooltip").style.top = (event.clientY + 10) + "px";
  }}

  function hideTooltip() {{
    document.getElementById("tooltip").style.display = "none";
  }}

  // Side panel: show node info and neighbor correlations
  function selectNode(id) {{
    selectedId = id;

    // Highlight selected node in red, others default blue
    nodeSelection.attr("fill", d => d.id === id ? "#e53935" : "#1f77b4");

    // Find the node object for position/velocity
    const node = nodes.find(n => n.id === id);

    // Meta info
    const frame = frames[currentFrame];
    const stats = frame.node_stats[id] || {{pnl: 0, dvol: 0, norm: 0}};
    const deg = adj.get(id)?.length ?? 0;

    document.getElementById("selMeta").innerHTML =
      `<div><b>${{id}}</b></div>` +
      `<div>degree: ${{deg}}</div>` +
      `<div>PnL: ${{stats.pnl.toFixed(6)}}, Vol: ${{stats.dvol.toFixed(6)}}</div>` +
      `<div>PnL/Vol: ${{stats.norm.toFixed(3)}}</div>` +
      `<div>frame: ${{frame.t0}} → ${{frame.t1}}</div>` +
      `<div>X: ${{(node?.x ?? 0).toFixed(1)}}, Y: ${{(node?.y ?? 0).toFixed(1)}}</div>` +
      `<div>dX: ${{(node?.vx ?? 0).toFixed(2)}}, dY: ${{(node?.vy ?? 0).toFixed(2)}}</div>`;

    // Correlations to neighbors
    const idxs = adj.get(id) || [];
    const rows = [];

    for (const li of idxs) {{
      const L = links[li];
      const s = nid(L.source);
      const t = nid(L.target);
      const other = (s === id) ? t : s;
      const dx = L.source.x - L.target.x;
      const dy = L.source.y - L.target.y;
      rows.push({{ other, cor: L.corr, tcor: L.targetCorr, dist: Math.hypot(dx, dy), abs: Math.abs(L.targetCorr) }});
    }}

    rows.sort((a, b) => b.abs - a.abs);

    const maxShow = 50;
    // Header row
    let html = `<div class="side-row side-header">
      <div class="lhs"><b>asset</b></div>
      <div class="rhs"><b>cor</b></div>
      <div class="rhs"><b>tcor</b></div>
      <div class="rhs"><b>dist</b></div>
    </div>`;
    html += rows.slice(0, maxShow).map(r => {{
      const cls = r.tcor >= 0 ? "pos" : "neg";
      return `<div class="side-row ${{cls}}">
        <div class="lhs">${{r.other}}</div>
        <div class="rhs">${{r.cor.toFixed(3)}}</div>
        <div class="rhs">${{r.tcor.toFixed(3)}}</div>
        <div class="rhs">${{r.dist.toFixed(1)}}</div>
      </div>`;
    }}).join("");

    document.getElementById("corrList").innerHTML =
      html || `<div class="side-meta">No neighbor edges.</div>`;
  }}

  // Drag behavior
  function drag(sim) {{
    return d3.drag()
      .on("start", (event, d) => {{
        if (!event.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      }})
      .on("drag", (event, d) => {{
        d.fx = event.x; d.fy = event.y;
      }})
      .on("end", (event, d) => {{
        if (!event.active) sim.alphaTarget(0);
        d.fx = null; d.fy = null;
      }});
  }}

  // Freeze toggle
  document.getElementById("freeze").addEventListener("change", (e) => {{
    if (e.target.checked) {{
      simulation.alpha(0).stop();
    }} else {{
      simulation.alpha(reheatAlpha).restart();
    }}
  }});

  // Re-layout button (randomize + restart)
  document.getElementById("relayout").addEventListener("click", () => {{
    nodes.forEach(d => {{
      d.x = width/2 + (Math.random() - 0.5) * 400;
      d.y = height/2 + (Math.random() - 0.5) * 300;
    }});
    document.getElementById("freeze").checked = false;
    simulation.alpha(1).restart();
  }});

  // Heat up button (inject energy without changing positions)
  document.getElementById("heatUp").addEventListener("click", () => {{
    document.getElementById("freeze").checked = false;
    simulation.alpha(1).restart();
  }});

  // Kick strength UI
  const kickStrengthEl = document.getElementById("kickStrength");
  const kickValEl = document.getElementById("kickVal");

  // Keep the numeric label in sync
  kickValEl.textContent = Number(kickStrengthEl.value).toFixed(0);
  kickStrengthEl.addEventListener("input", () => {{
    kickValEl.textContent = Number(kickStrengthEl.value).toFixed(0);
  }});

  // Kick (velocity-only, strength from slider, scaled by canvas + node count)
  document.getElementById("kickBtn").addEventListener("click", () => {{
    document.getElementById("freeze").checked = false;

    const kUser = Number(kickStrengthEl.value) || 0; // 0..500
    // Scale so it feels similar across sizes / N
    const amp = kUser * 0.02 * Math.min(width, height) / Math.sqrt(nodes.length);

    for (const n of nodes) {{
      n.vx = (n.vx || 0) + (Math.random() - 0.5) * amp;
      n.vy = (n.vy || 0) + (Math.random() - 0.5) * amp;
    }}

    simulation.alpha(1).restart();
  }});

  // Re-link (reinitialize link force internals for debugging)
  document.getElementById("relink").addEventListener("click", () => {{
    // Force immediate corr application
    links.forEach(d => {{ d.corr = d.targetCorr; }});

    // Recompute link force internals
    linkForce.links(links);

    // Remove inertia from earlier states
    nodes.forEach(n => {{ n.vx = 0; n.vy = 0; }});

    // Reheat
    document.getElementById("freeze").checked = false;
    simulation.alpha(0.9).restart();
  }});

  // NoSpeed (zero all velocities)
  document.getElementById("noSpeed").addEventListener("click", () => {{
    nodes.forEach(n => {{ n.vx = 0; n.vy = 0; }});
  }});

  // --- Alpha controls ---
  const alphaDecayEl = document.getElementById("alphaDecay");
  const alphaMinEl   = document.getElementById("alphaMin");
  const alphaTargetEl= document.getElementById("alphaTarget");

  const alphaDecayValEl  = document.getElementById("alphaDecayVal");
  const alphaMinValEl    = document.getElementById("alphaMinVal");
  const alphaTargetValEl = document.getElementById("alphaTargetVal");

  function syncAlphaUI() {{
    alphaDecayValEl.textContent  = Number(alphaDecayEl.value).toFixed(3);
    alphaMinValEl.textContent    = Number(alphaMinEl.value).toFixed(4);
    alphaTargetValEl.textContent = Number(alphaTargetEl.value).toFixed(2);
  }}
  syncAlphaUI();

  // Apply initial values
  simulation.alphaDecay(Number(alphaDecayEl.value));
  simulation.alphaMin(Number(alphaMinEl.value));
  simulation.alphaTarget(Number(alphaTargetEl.value));

  alphaDecayEl.addEventListener("input", () => {{
    syncAlphaUI();
    simulation.alphaDecay(Number(alphaDecayEl.value));
    if (!document.getElementById("freeze").checked) simulation.alpha(Math.max(simulation.alpha(), 0.05)).restart();
  }});

  alphaMinEl.addEventListener("input", () => {{
    syncAlphaUI();
    simulation.alphaMin(Number(alphaMinEl.value));
    if (!document.getElementById("freeze").checked) simulation.alpha(Math.max(simulation.alpha(), 0.05)).restart();
  }});

  alphaTargetEl.addEventListener("input", () => {{
    syncAlphaUI();
    simulation.alphaTarget(Number(alphaTargetEl.value));
    if (!document.getElementById("freeze").checked) simulation.alpha(Math.max(simulation.alpha(), 0.05)).restart();
  }});

  // Corr smoothing UI (JS-only, no Streamlit reruns)
  const corrSmoothEl = document.getElementById("corrSmooth");
  const corrSmoothValEl = document.getElementById("corrSmoothVal");

  function syncCorrSmoothUI() {{
    corrSmooth = Number(corrSmoothEl.value);
    if (!Number.isFinite(corrSmooth)) corrSmooth = 0.08;
    corrSmoothValEl.textContent = corrSmooth.toFixed(2);
  }}
  syncCorrSmoothUI();

  corrSmoothEl.addEventListener("input", () => {{
    syncCorrSmoothUI();
    if (!document.getElementById("freeze").checked) {{
      simulation.alpha(Math.max(simulation.alpha(), 0.05)).restart();
    }}
  }});

  // Link physics sliders (lenMin, lenSlope, strMin, strSlope, strPow)
  const lengthMinEl = document.getElementById("lengthMin");
  const lengthSlopeEl = document.getElementById("lengthSlope");
  const strengthMinEl = document.getElementById("strengthMin");
  const strengthSlopeEl = document.getElementById("strengthSlope");
  const strengthPowerEl = document.getElementById("strengthPower");

  function updateLinkParams() {{
    lengthMin = Number(lengthMinEl.value);
    lengthSlope = Number(lengthSlopeEl.value);
    strengthMin = Number(strengthMinEl.value);
    strengthSlope = Number(strengthSlopeEl.value);
    strengthPower = Number(strengthPowerEl.value);

    document.getElementById("lengthMinVal").textContent = lengthMin.toFixed(0);
    document.getElementById("lengthSlopeVal").textContent = lengthSlope.toFixed(0);
    document.getElementById("strengthMinVal").textContent = strengthMin.toFixed(3);
    document.getElementById("strengthSlopeVal").textContent = strengthSlope.toFixed(2);
    document.getElementById("strengthPowerVal").textContent = strengthPower.toFixed(1);

    // Reinitialize link force and reheat
    linkForce.links(links);
    const frozen = document.getElementById("freeze").checked;
    if (frozen) {{
      renderEdgesAndNodes();
    }} else {{
      simulation.alpha(0.3).restart();
    }}
  }}

  [lengthMinEl, lengthSlopeEl, strengthMinEl, strengthSlopeEl, strengthPowerEl].forEach(el => {{
    el.addEventListener("input", updateLinkParams);
  }});

  // Play/Pause
  document.getElementById("playPause").addEventListener("click", () => {{
    playing = !playing;
    document.getElementById("playPause").textContent = playing ? "Pause" : "Play";
    if (playing) {{
      playInterval = setInterval(() => {{
        const next = (currentFrame + 1) % frames.length;
        updateFrame(next);
      }}, playSpeedMs);
    }} else {{
      clearInterval(playInterval);
    }}
  }});

  // Slider with debounce
  let debounceTimer = null;
  slider.addEventListener("input", (e) => {{
    if (playing) {{
      playing = false;
      document.getElementById("playPause").textContent = "Play";
      clearInterval(playInterval);
    }}
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {{
      updateFrame(parseInt(e.target.value));
    }}, 50);
  }});

  // Initial render
  updateFrame(0);
}}
</script>
</body>
</html>
'''
    return html


def _compute_all_straddle_metrics_from_indices(straddles, valuations, idx: np.ndarray):
    """Core straddle metrics computation (uncached)."""
    if idx.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty, idx

    # Phase 1: Array prep
    t0 = time.perf_counter()
    out0s = straddles["out0"][idx].astype(np.int32)
    lens = straddles["length"][idx].astype(np.int32)
    pnl = valuations["pnl"]
    vol = valuations["vol"]
    mv = valuations["mv"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)
    dt_prep = time.perf_counter() - t0

    # Phase 2: Numba kernel
    t0 = time.perf_counter()
    pnl_sum, pnl_days, vol_sum, vol_days, mv_sum = _summarize_all(
        out0s, lens, pnl, vol, mv, dte, have_dte
    )
    dt_kernel = time.perf_counter() - t0

    # Log sub-timings
    tlog(f"       straddle_metrics.prep: {_fmt_dt_s(dt_prep)}")
    tlog(f"       straddle_metrics.kernel: {_fmt_dt_s(dt_kernel)}")

    return pnl_sum, pnl_days, vol_sum, vol_days, mv_sum, idx


@st.cache_data
def compute_all_straddle_metrics(filter_key):
    """Compute per-straddle metrics. Cached by filter params."""
    straddles, valuations, _, _, _ = load_data()
    idx = get_filtered_indices_cached(filter_key)
    return _compute_all_straddle_metrics_from_indices(straddles, valuations, idx)


def compute_ym_matrix_from_cache(filtered_indices_tuple, weights, value_type="pnl"):
    """Compute year×month matrix from CACHED per-straddle arrays (no kernel calls).

    NOTE: Not cached - computation is fast (bincount), caching overhead was 7+ seconds.

    Args:
        weights: Pre-computed per-straddle values (pnl_sum, pnl_days, or mv_sum)
        value_type: "pnl" | "live_days" | "mv" (for formatting hints)

    Returns: (matrix_df, year_range)
    """
    idx = np.array(filtered_indices_tuple, dtype=np.int64)
    if len(idx) == 0:
        return pd.DataFrame(), (0, 0)

    straddles, _, _, _, _ = load_data()
    years = straddles["year"][idx].astype(np.int32)
    months = straddles["month"][idx].astype(np.int32)

    y0, y1 = int(years.min()), int(years.max())
    ny = y1 - y0 + 1

    # Map to bin id: (year - y0) * 12 + (month - 1)
    bin_ids = (years - y0) * 12 + (months - 1)

    # Use CACHED arrays (no kernel call!)
    if value_type == "live_days":
        values = np.bincount(bin_ids, weights=weights.astype(np.float64), minlength=ny * 12)
    else:
        values = np.bincount(bin_ids, weights=weights, minlength=ny * 12)

    # Reshape to ny × 12
    mat = values.reshape(ny, 12)

    # Build DataFrame
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = pd.DataFrame(mat, index=np.arange(y0, y1 + 1), columns=month_names)
    df.index.name = "Year"

    # Add Total column as first column (after Year index)
    df.insert(0, "Total", df.sum(axis=1))

    return df, (y0, y1)


def render_heatmap(df, title, fmt=".4f", cmap="RdYlGn"):
    """Render year x month matrix as heatmap with Total column separator."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.4)))

    # Data without index
    data = df.values

    # Create heatmap (skip Total column for color normalization)
    data_no_total = data[:, 1:]  # Exclude Total column
    vmin, vmax = np.nanmin(data_no_total), np.nanmax(data_no_total)

    # If centered around zero, use symmetric limits
    if vmin < 0 < vmax:
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=9)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)

    # Annotate cells
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = data[i, j]
            if not np.isnan(val):
                # Use white text on dark backgrounds, black on light
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                color = "white" if norm_val < 0.3 or norm_val > 0.7 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=7, color=color)

    # Separator line after Total column
    ax.axvline(x=0.5, color="black", linewidth=2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


@st.fragment
def render_matrix_view(mat_df, year_range, title, fmt, cmap, radio_key):
    """Fragment for matrix view toggle - only reruns this section on view change."""
    y0, y1 = year_range
    view_mode = st.radio("View", ["Heatmap", "Table"], horizontal=True, key=radio_key)
    if view_mode == "Heatmap":
        fig = render_heatmap(mat_df, title, fmt=fmt, cmap=cmap)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.dataframe(mat_df.style.format(f"{{:{fmt}}}").background_gradient(cmap=cmap, axis=None), width="content")
    st.caption(f"{title} | Years: {y0}-{y1}")


@st.fragment
def render_low_coverage_table(coverage: pd.Series):
    """Fragment for low coverage table - only reruns this section on threshold change."""
    st.write("**Low Coverage Assets**")
    cov_threshold = st.slider("Coverage threshold", 0, int(coverage.max()), 30, step=5, key="cov_thresh")
    sparse = coverage[coverage < cov_threshold]
    if len(sparse) > 0:
        sparse_df = pd.DataFrame({"asset": sparse.index, "days": sparse.values})
        st.dataframe(sparse_df, use_container_width=True, height=200)
        st.caption(f"{len(sparse)} assets below {cov_threshold} days")
    else:
        st.success(f"All assets have >= {cov_threshold} days coverage")


# ============================================================================
# Core Functions
# ============================================================================
def get_filtered_indices(straddles, asset_str, straddle_ym,
                          asset_mode, asset_value, ym_lo, ym_hi, schid_set=None):
    """Filter straddles by asset (dropdown/regex), month range, and schid.

    Returns: (indices, error_msg)
        - indices: np.ndarray of matching straddle indices
        - error_msg: str if regex error, else None
    """
    mask = np.ones(len(straddles["year"]), dtype=bool)

    # Month range filter
    mask &= (straddle_ym >= ym_lo) & (straddle_ym <= ym_hi)

    # Schedule ID filter
    if schid_set:
        mask &= np.isin(straddles["schid"], list(schid_set))

    # Asset filter
    if asset_mode == "dropdown":
        if asset_value != "All":
            mask &= (asset_str == asset_value)
    elif asset_mode == "candidates":
        # Filter to specific list of asset names (used by "Similar to" mode)
        if len(asset_value) > 0:
            mask &= np.isin(asset_str, asset_value)
        else:
            return np.array([], dtype=np.int64), None
    else:  # regex mode
        pattern, flags = asset_value
        if pattern.strip():
            try:
                rx = re.compile(pattern, flags)
            except re.error as e:
                return np.array([], dtype=np.int64), str(e)

            # Fast: match unique assets first, then isin
            unique_assets = np.unique(asset_str)
            matched = [a for a in unique_assets if rx.search(str(a))]
            if len(matched) == 0:
                return np.array([], dtype=np.int64), None
            mask &= np.isin(asset_str, matched)

    return np.flatnonzero(mask), None

def get_dates_for_slice(straddles, idx):
    """Compute date array for straddle slice."""
    start_epoch = int(straddles["month_start_epoch"][idx])
    length = int(straddles["length"][idx])
    base = np.datetime64('1970-01-01', 'D')
    return base + start_epoch + np.arange(length)

def get_slice_mask(valuations, out0, length):
    """Build mask for valid rows (days_to_expiry >= 0)."""
    if "days_to_expiry" not in valuations:
        return np.ones(length, dtype=bool)
    dte = valuations["days_to_expiry"][out0:out0+length]
    return (dte >= 0)

def get_daily_slice(straddles, valuations, idx):
    """Extract daily values for selected straddle, filtered by dte >= 0."""
    out0 = int(straddles["out0"][idx])
    length = int(straddles["length"][idx])
    raw_dates = get_dates_for_slice(straddles, idx)

    mask = get_slice_mask(valuations, out0, length)

    data = {"date": raw_dates[mask]}
    for key in ["vol", "hedge1", "hedge2", "hedge3", "hedge4",
                "strike", "mv", "delta", "pnl", "opnl", "hpnl", "days_to_expiry"]:
        if key in valuations:
            data[key] = valuations[key][out0:out0+length][mask]

    return data, length, int(mask.sum())

def plot_valid_only(dates, series, title):
    """Plot series with valid-only (non-NaN) filtering."""
    valid = ~np.isnan(series)
    if not np.any(valid):
        st.warning(f"No valid data for {title}")
        return

    df = pd.DataFrame({"date": dates[valid], title: series[valid]})
    df = df.set_index("date")
    st.line_chart(df)

    # Stats
    vals = series[valid]
    st.caption(f"Valid: {len(vals):,} | Min: {vals.min():.4f} | Med: {np.median(vals):.4f} | Max: {vals.max():.4f}")

def plot_cumsum_valid(dates, series, title):
    """Plot cumulative sum with valid-only filtering."""
    valid = ~np.isnan(series)
    if not np.any(valid):
        st.warning(f"No valid data for {title}")
        return

    valid_dates = dates[valid]
    cumsum = np.cumsum(series[valid])

    df = pd.DataFrame({"date": valid_dates, title: cumsum})
    df = df.set_index("date")
    st.line_chart(df)
    st.caption(f"Final: {cumsum[-1]:.6f} | Points: {len(cumsum):,}")


def build_cum_df(pop_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative series dataframe."""
    cum_df = pd.DataFrame(index=pop_df.index)
    cum_df["pnl_cum"] = pop_df["pnl_sum"].cumsum()

    if "norm_pnl_sum" in pop_df.columns:
        cum_df["norm_pnl_cum"] = pop_df["norm_pnl_sum"].cumsum()

    if "opnl_sum" in pop_df.columns:
        cum_df["opnl_cum"] = pop_df["opnl_sum"].cumsum()
    if "hpnl_sum" in pop_df.columns:
        cum_df["hpnl_cum"] = pop_df["hpnl_sum"].cumsum()

    if "norm_opnl_sum" in pop_df.columns:
        cum_df["norm_opnl_cum"] = pop_df["norm_opnl_sum"].cumsum()
    if "norm_hpnl_sum" in pop_df.columns:
        cum_df["norm_hpnl_cum"] = pop_df["norm_hpnl_sum"].cumsum()

    return cum_df


def build_asset_daily_df(asset_pop: pd.DataFrame) -> pd.DataFrame:
    """Build daily + cumulative series for asset drilldown."""
    out = pd.DataFrame(index=asset_pop.index)
    out["pnl_sum"] = asset_pop["pnl_sum"]
    out["pnl_cum"] = asset_pop["pnl_sum"].cumsum()

    if "opnl_sum" in asset_pop.columns:
        out["opnl_sum"] = asset_pop["opnl_sum"]
        out["opnl_cum"] = asset_pop["opnl_sum"].cumsum()
    if "hpnl_sum" in asset_pop.columns:
        out["hpnl_sum"] = asset_pop["hpnl_sum"]
        out["hpnl_cum"] = asset_pop["hpnl_sum"].cumsum()

    return out


def render_cum_plot_plotly(cum_df: pd.DataFrame):
    """Render cumulative PnL with Plotly (range slider, buttons)."""
    fig = go.Figure()

    # Main cumulative lines
    fig.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["pnl_cum"],
        mode="lines", name="PnL cum", line=dict(width=2)
    ))

    if "opnl_cum" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["opnl_cum"],
            mode="lines", name="Option PnL cum", line=dict(width=1)
        ))
    if "hpnl_cum" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["hpnl_cum"],
            mode="lines", name="Hedge PnL cum", line=dict(width=1)
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=90),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        yaxis=dict(title="Cumulative PnL"),
    )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def render_norm_pnl_plot_plotly(cum_df: pd.DataFrame):
    """Render normalized cumulative PnL with Plotly."""
    if "norm_pnl_cum" not in cum_df.columns:
        st.warning("Normalized P&L data not available.")
        return

    fig = go.Figure()

    # Main normalized cumulative P&L
    fig.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df["norm_pnl_cum"],
        mode="lines", name="Norm PnL cum", line=dict(width=2)
    ))

    # Normalized option P&L (if available)
    if "norm_opnl_cum" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["norm_opnl_cum"],
            mode="lines", name="Norm Option PnL cum", line=dict(width=1)
        ))

    # Normalized hedge P&L (if available)
    if "norm_hpnl_cum" in cum_df.columns:
        fig.add_trace(go.Scatter(
            x=cum_df.index, y=cum_df["norm_hpnl_cum"],
            mode="lines", name="Norm Hedge PnL cum", line=dict(width=1)
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=90),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        yaxis=dict(title="Normalized P&L"),
    )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _tv_series_from_pd(s: pd.Series):
    """Convert pandas Series to TradingView time/value format (YYYY-MM-DD strings)."""
    s = s.astype(float)
    mask = np.isfinite(s.values)
    s = s.iloc[mask]
    # Use business-day strings for daily series (wrapper expects this format)
    t = s.index.strftime("%Y-%m-%d")
    v = s.values
    return [{"time": str(t[i]), "value": float(v[i])} for i in range(len(v))]


def render_cum_plot_tradingview(cum_df: pd.DataFrame):
    """Render cumulative PnL with TradingView Lightweight Charts."""
    chart_options = {
        "height": 520,
        "layout": {
            "background": {"type": "solid", "color": "white"},
            "textColor": "black",
        },
        "grid": {
            "vertLines": {"color": "rgba(197, 203, 206, 0.5)"},
            "horzLines": {"color": "rgba(197, 203, 206, 0.5)"},
        },
        "timeScale": {"timeVisible": True, "secondsVisible": False},
        "rightPriceScale": {"borderVisible": False},
        "crosshair": {"mode": 1},
    }

    series = [
        {
            "type": "Line",
            "data": _tv_series_from_pd(cum_df["pnl_cum"]),
            "options": {"lineWidth": 2, "color": "#2962FF", "title": "PnL cum"},
        }
    ]

    if "opnl_cum" in cum_df.columns:
        series.append({
            "type": "Line",
            "data": _tv_series_from_pd(cum_df["opnl_cum"]),
            "options": {"lineWidth": 1, "color": "#26a69a", "title": "Option cum"},
        })
    if "hpnl_cum" in cum_df.columns:
        series.append({
            "type": "Line",
            "data": _tv_series_from_pd(cum_df["hpnl_cum"]),
            "options": {"lineWidth": 1, "color": "#ef5350", "title": "Hedge cum"},
        })

    renderLightweightCharts([{"chart": chart_options, "series": series}], key="tv_cum")


def render_timeseries_plotly(df: pd.DataFrame, cols: list, title: str, height: int = 420):
    """Render time-series with Plotly (range slider, unified hover)."""
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=70),
        hovermode="x unified",
        title=dict(text=title, x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def render_hist_plotly(x: np.ndarray, title: str, xlabel: str, bins: int = 80, clip_pct=(1, 99)):
    """Render histogram with Plotly, mean/median lines, clipped to percentiles."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        st.warning("No valid data.")
        return

    lo, hi = np.percentile(x, clip_pct)
    clipped = x[(x >= lo) & (x <= hi)]

    mean_x = float(np.mean(x))
    med_x = float(np.median(x))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clipped, nbinsx=bins, name=""))

    # Mean/median vertical lines
    fig.add_vline(x=mean_x, line_width=2, line_dash="dash", annotation_text=f"Mean {mean_x:.4f}")
    fig.add_vline(x=med_x, line_width=2, line_dash="solid", annotation_text=f"Median {med_x:.4f}")

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        bargap=0.05,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mean_x:.4f}")
    c2.metric("Median", f"{med_x:.4f}")
    c3.metric("Std Dev", f"{float(np.std(x)):.4f}")
    c4.metric("Days", f"{x.size:,}")


def render_asset_bar_ratio(df: pd.DataFrame, top_n: int = 80):
    """Bar chart of mean daily PnL / mean daily vol by asset."""
    d = df.head(top_n).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["asset"],
        y=d["pnl_over_daily_vol"],
        customdata=np.stack([d["mean_daily_pnl"], d["mean_daily_vol"]], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "PnL / daily vol: %{y:.3f}<br>"
            "mean daily pnl: %{customdata[0]:.6f}<br>"
            "mean daily vol: %{customdata[1]:.6f}<extra></extra>"
        ),
        name="",
    ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Mean Daily PnL / Daily Vol by Asset (top {top_n})",
        xaxis=dict(showticklabels=False),
        yaxis_title="PnL / daily vol",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("X labels hidden; use hover and zoom/pan.")


def render_asset_bar_vol(df: pd.DataFrame, top_n: int = 80, which: str = "annual"):
    """Bar chart of vol by asset (annual or daily)."""
    col = "avg_vol" if which == "annual" else "mean_daily_vol"
    title = "Mean Annual Vol by Asset" if which == "annual" else "Mean Daily Vol by Asset"
    d = df.sort_values(col, ascending=False).head(top_n).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["asset"],
        y=d[col],
        customdata=np.stack([d["mean_daily_pnl"]], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{col}: %{{y:.6f}}<br>"
            "mean daily pnl: %{customdata[0]:.6f}<extra></extra>"
        ),
        name="",
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{title} (top {top_n})",
        xaxis=dict(showticklabels=False),
        yaxis_title=("Annual vol" if which == "annual" else "Daily vol"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("X labels hidden; use hover and zoom/pan.")


def render_asset_pnl_plotly(asset_name: str, df: pd.DataFrame):
    """Render asset drilldown with Plotly (daily/cumulative toggle)."""
    view = st.radio("View", ["Cumulative", "Daily"], horizontal=True, key=f"asset_view_{asset_name}")

    if view == "Cumulative":
        cols = ["pnl_cum"]
        labels = {"pnl_cum": "PnL cum", "opnl_cum": "Option PnL cum", "hpnl_cum": "Hedge PnL cum"}
        if "opnl_cum" in df.columns:
            cols.append("opnl_cum")
        if "hpnl_cum" in df.columns:
            cols.append("hpnl_cum")
        title = f"{asset_name} — Cumulative PnL"
    else:
        cols = ["pnl_sum"]
        labels = {"pnl_sum": "PnL daily", "opnl_sum": "Option PnL daily", "hpnl_sum": "Hedge PnL daily"}
        if "opnl_sum" in df.columns:
            cols.append("opnl_sum")
        if "hpnl_sum" in df.columns:
            cols.append("hpnl_sum")
        title = f"{asset_name} — Daily PnL"

    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=labels.get(c, c)))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=90),
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
        ),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def render_correlation_heatmap(
    corr_mat: np.ndarray,
    overlap: np.ndarray,
    asset_names: np.ndarray,
    boundaries: list = None,
    title_prefix: str = "Asset Correlation Heatmap"
):
    """Render interactive Plotly correlation heatmap with optional cluster separators."""
    n = len(asset_names)
    boundaries = boundaries or []

    # Build hover text matrix
    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            c = corr_mat[i, j]
            o = int(overlap[i, j])
            if np.isfinite(c):
                row.append(f"{asset_names[i]} x {asset_names[j]}<br>corr: {c:.3f}<br>overlap: {o} days")
            else:
                row.append(f"{asset_names[i]} x {asset_names[j]}<br>corr: NaN<br>overlap: {o} days")
        hover_text.append(row)

    # Use numeric axes for precise separator placement
    fig = go.Figure(data=go.Heatmap(
        z=corr_mat,
        x=np.arange(n),
        y=np.arange(n),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        hoverinfo="text",
        text=hover_text,
        colorbar=dict(title="Correlation"),
    ))

    # Dynamic height: scale with assets, cap at 1200
    height = min(1200, max(520, 8 * n))

    # Tick label policy: hide if too many assets
    show_ticks = n <= 60
    tick_font_size = 8 if n <= 40 else 6

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{title_prefix} ({n} assets)",
        xaxis=dict(
            showticklabels=show_ticks,
            tickvals=np.arange(n) if show_ticks else None,
            ticktext=asset_names if show_ticks else None,
            tickfont=dict(size=tick_font_size) if show_ticks else None,
            title="",
        ),
        yaxis=dict(
            showticklabels=show_ticks,
            tickvals=np.arange(n) if show_ticks else None,
            ticktext=asset_names if show_ticks else None,
            tickfont=dict(size=tick_font_size) if show_ticks else None,
            title="",
            autorange="reversed",
        ),
    )

    # Draw cluster separators
    for b in boundaries:
        pos = b - 0.5
        # Vertical line
        fig.add_shape(
            type="line",
            x0=pos, x1=pos, y0=-0.5, y1=n - 0.5,
            xref="x", yref="y",
            line=dict(width=2, color="black"),
        )
        # Horizontal line
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n - 0.5, y0=pos, y1=pos,
            xref="x", yref="y",
            line=dict(width=2, color="black"),
        )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # Show note about hidden labels
    if not show_ticks:
        st.caption("Axis labels hidden (>60 assets). Hover over cells to see asset names.")


# ============================================================================
# Clustering Helpers
# ============================================================================
def corr_to_distance(corr_mat: np.ndarray, mode: str) -> np.ndarray:
    """Convert correlation matrix to distance matrix for clustering.

    mode:
      - "1-corr": D = 1 - corr (negative correlations are far)
      - "1-abs(corr)": D = 1 - abs(corr) (opposite movers cluster together)
    """
    C = corr_mat.copy()
    C[~np.isfinite(C)] = 0.0  # NaN -> 0 corr -> distance 1

    if mode == "1-corr":
        D = 1.0 - C
    elif mode == "1-abs(corr)":
        D = 1.0 - np.abs(C)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")

    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    return D


def cluster_order_from_corr(corr_mat: np.ndarray, dist_mode: str, linkage_method: str) -> np.ndarray:
    """Return permutation (leaf order) from hierarchical clustering."""
    n = corr_mat.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int64)

    D = corr_to_distance(corr_mat, dist_mode)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return leaves_list(Z).astype(np.int64)


def cluster_boundaries_from_corr(
    corr_mat: np.ndarray, dist_mode: str, linkage_method: str, k: int
):
    """Return (order, boundaries) where boundaries mark cluster changes.

    boundaries: list of indices where cluster membership changes in leaf order
    """
    n = corr_mat.shape[0]
    if n <= 2 or k <= 1:
        return np.arange(n, dtype=np.int64), []

    D = corr_to_distance(corr_mat, dist_mode)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)

    order = leaves_list(Z).astype(np.int64)
    labels = fcluster(Z, t=k, criterion="maxclust").astype(np.int64)
    labels_ord = labels[order]

    boundaries = []
    for i in range(1, n):
        if labels_ord[i] != labels_ord[i - 1]:
            boundaries.append(i)
    return order, boundaries


# ============================================================================
# Main App
# ============================================================================
timings = {}
tlog(f"{_ts()} ---- run ----")

with timed_collect("load_data", timings):
    straddles, valuations, missing_s, missing_v, avail_opt = load_data()

# Schema warnings
if missing_s or missing_v:
    st.error(f"Missing required fields - Straddles: {missing_s}, Valuations: {missing_v}")
    st.stop()

st.title("Backtest Explorer")

# ============================================================================
# Sidebar with Tabs
# ============================================================================
# Precompute for filtering
with timed_collect("get_asset_strings", timings):
    asset_str, unique_assets = get_asset_strings(straddles)

with timed_collect("get_straddle_ym", timings):
    straddle_ym = get_straddle_ym(straddles)

# Precompute asset IDs once for fast slicing in Assets tab
with timed_collect("factorize_assets", timings):
    asset_ids_all, asset_names_all = factorize_assets(tuple(asset_str))

with st.sidebar:
    st.header("Filters")

    # --- Asset Filter ---
    st.subheader("Asset")
    asset_mode = st.radio("Mode", ["Dropdown", "Regex", "Similar to"], horizontal=True,
                          label_visibility="collapsed")

    # Initialize sim_neighbors_df for use in expander later
    sim_neighbors_df = pd.DataFrame()

    if asset_mode == "Dropdown":
        selected_asset = st.selectbox("Asset", ["All"] + unique_assets,
                                      label_visibility="collapsed")
        asset_value = selected_asset
    elif asset_mode == "Similar to":
        sim_anchor = st.selectbox("Anchor asset", unique_assets, key="sim_anchor")
        sim_n = st.slider("N neighbors", 5, 200, 25, step=5, key="sim_n")
        sim_rank = st.radio("Rank by", ["abs(corr)", "corr"], horizontal=True, key="sim_rank")
        sim_corr_method = st.selectbox("Corr method", ["pearson", "spearman", "sign", "partial"], key="sim_corr_method")
        sim_min_overlap = st.slider("Min overlap", 5, 250, 30, step=5, key="sim_min_overlap")

        # Partial controls if needed
        sim_partial_fill = "median"
        sim_partial_estimator = "ledoitwolf"
        if sim_corr_method == "partial":
            sim_partial_row = st.columns(2)
            with sim_partial_row[0]:
                sim_partial_fill = st.selectbox("NaN fill", ["median", "mean", "zero"], key="sim_partial_fill")
            with sim_partial_row[1]:
                sim_partial_estimator = st.selectbox("Estimator", ["ledoitwolf", "oas", "ridge"], key="sim_partial_estimator")

        # asset_value = (anchor, n, rank, corr_method, min_overlap, partial_fill, partial_estimator)
        asset_value = (sim_anchor, sim_n, sim_rank, sim_corr_method, sim_min_overlap, sim_partial_fill, sim_partial_estimator)
        asset_mode = "similar"  # Internal mode name
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            regex_pattern = st.text_input("Pattern", value="",
                                          placeholder="e.g. ^CL|^HO",
                                          label_visibility="collapsed")
        with col2:
            case_insensitive = st.checkbox("i", help="Case insensitive")

        flags = re.IGNORECASE if case_insensitive else 0
        asset_value = (regex_pattern, flags)

        # Show matching assets count
        if regex_pattern.strip():
            try:
                rx = re.compile(regex_pattern, flags)
                matched = [a for a in unique_assets if rx.search(a)]
                st.caption(f"Matches {len(matched)} of {len(unique_assets)} assets")
            except re.error as e:
                st.error(f"Invalid regex: {e}")
        else:
            st.caption("Empty pattern = All assets")

    st.divider()

    # --- Month Range Filter (dropdowns, no popup) ---
    st.subheader("Month Range")

    # Build YYYY-MM options from data bounds
    ym_min = int(straddle_ym.min())
    ym_max = int(straddle_ym.max())
    ym_options = [f"{ym // 12}-{(ym % 12) + 1:02d}" for ym in range(ym_min, ym_max + 1)]

    c1, c2 = st.columns(2)
    with c1:
        start_ym_str = st.selectbox("From", ym_options, index=0)
    with c2:
        end_ym_str = st.selectbox("To", ym_options, index=len(ym_options) - 1)

    # Parse back to ym index
    def parse_ym(s):
        y, m = s.split("-")
        return int(y) * 12 + int(m) - 1

    ym_lo = parse_ym(start_ym_str)
    ym_hi = parse_ym(end_ym_str)

    # Swap if reversed
    if ym_lo > ym_hi:
        ym_lo, ym_hi = ym_hi, ym_lo

    st.divider()

    # --- Schedule ID Filter ---
    st.subheader("Schedule ID")
    unique_schids = sorted(set(int(x) for x in np.unique(straddles["schid"])))
    selected_schids = st.multiselect(
        "Select schedules",
        options=unique_schids,
        default=unique_schids,
        label_visibility="collapsed"
    )
    if not selected_schids:
        st.warning("Select at least one schedule ID")

    st.divider()

    # --- Apply filters ---
    with timed_collect("get_filtered_indices", timings, meta=f"(schids={len(selected_schids)})"):
        if asset_mode.lower() == "similar":
            # Similar mode requires correlation computation - use cached version
            temp_key = make_filter_key(asset_mode.lower(), asset_value, ym_lo, ym_hi, selected_schids)
            filtered_indices = get_filtered_indices_cached(temp_key)
            regex_error = None
        else:
            filtered_indices, regex_error = get_filtered_indices(
                straddles, asset_str, straddle_ym,
                asset_mode.lower(), asset_value, ym_lo, ym_hi, set(selected_schids)
            )
    tlog(f"       -> matched {len(filtered_indices):,} straddles")

    if regex_error:
        st.error(f"Regex error: {regex_error}")

    st.write(f"**Matching:** {len(filtered_indices):,} straddles")

# ============================================================================
# Main Content - Population Aggregate View
# ============================================================================
if len(filtered_indices) == 0:
    st.warning("No straddles match the current filters.")
else:
    # Build filter key once (small, hashable)
    filter_key = make_filter_key(asset_mode.lower(), asset_value, ym_lo, ym_hi, selected_schids)

    # Show selected neighbors if in "Similar to" mode
    if asset_mode.lower() == "similar" and isinstance(asset_value, tuple) and len(asset_value) >= 5:
        anchor, n, rank, corr_method, min_overlap = asset_value[:5]
        p_fill = asset_value[5] if len(asset_value) > 5 else "median"
        p_est = asset_value[6] if len(asset_value) > 6 else "ledoitwolf"

        # Build temp filter_key for All assets to compute neighbors
        temp_key = (("dropdown", "All"), ym_lo, ym_hi, tuple(sorted(int(s) for s in selected_schids)))
        neighbors, neighbors_df = top_correlated_assets(
            temp_key, anchor, n, corr_method, min_overlap, rank,
            partial_fill=p_fill, partial_estimator=p_est
        )
        with st.expander(f"Selected neighbors ({len(neighbors)} assets)", expanded=False):
            st.caption(f"Anchor: **{anchor}** + {len(neighbors)-1} neighbors")
            if not neighbors_df.empty:
                st.dataframe(neighbors_df.style.format({"corr": "{:.3f}", "overlap": "{:d}"}),
                            width="stretch", hide_index=True, height=200)

    # Compute population daily aggregates (Numba kernel)
    with timed_collect("compute_population_daily", timings, meta=f"(n={len(filtered_indices):,})"):
        pop_df = compute_population_daily(filter_key)

    # Compute ALL per-straddle metrics ONCE (Numba kernel) - reuse everywhere
    with timed_collect("compute_all_straddle_metrics", timings, meta=f"(n={len(filtered_indices):,})"):
        pnl_sum_sel, pnl_days_sel, vol_sum_sel, vol_days_sel, mv_sum_sel, idx_sel = \
            compute_all_straddle_metrics(filter_key)

    if pop_df.empty:
        st.warning("No valid data for selected population.")
    else:
        # Summary strip
        cols = st.columns(4)
        cols[0].metric("Straddles", f"{len(filtered_indices):,}")
        cols[1].metric("Total PnL", f"{pop_df['pnl_sum'].sum():.4f}")
        cols[2].metric("Avg Contributors/Day", f"{pop_df['n_straddles'].mean():.1f}")
        cols[3].metric("P&L Days", f"{pop_df['n_straddles'].sum():,}")

        st.divider()

        # Mini-metrics dashboard
        pnl_arr = pop_df["pnl_sum"].values
        cum_pnl = np.cumsum(pnl_arr)
        max_dd = np.min(cum_pnl - np.maximum.accumulate(cum_pnl))
        win_rate = (pnl_arr > 0).sum() / len(pnl_arr) * 100
        sharpe_ish = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Mean Daily", f"{np.mean(pnl_arr):.4f}")
        m2.metric("Median Daily", f"{np.median(pnl_arr):.4f}")
        m3.metric("Std Dev", f"{np.std(pnl_arr):.4f}")
        m4.metric("Win Rate", f"{win_rate:.1f}%")
        m5.metric("Max DD", f"{max_dd:.4f}")
        m6.metric("Sharpe-ish", f"{sharpe_ish:.2f}")

        st.divider()

        # Tabs: Days, Cumulative PnL, Norm PnL, MV, Contributors, Straddles, Assets, Matrices
        tab_names = ["Days", "Density", "Norm Density", "Cumulative PnL", "Norm PnL", "MV", "Contributors", "Avg Vol", "Straddles", "Assets", "P&L Matrix", "Live Days", "MV Matrix", "Correlation", "CorGraph", "Embedding", "Coverage", "Attribution"]
        tabs = dict(zip(tab_names, st.tabs(tab_names)))

        # --- Days Tab ---
        with tabs["Days"]:
            st.subheader("Daily Aggregates")

            # Display columns
            display_cols = ["n_straddles", "pnl_sum"]
            if "opnl_sum" in pop_df.columns:
                display_cols.insert(2, "opnl_sum")
            if "hpnl_sum" in pop_df.columns:
                display_cols.insert(3, "hpnl_sum")

            with timed_collect("render_days_table", timings, meta=f"(rows={len(pop_df)})"):
                st.dataframe(pop_df[display_cols], width='stretch', height=400)
            st.caption("n_straddles = straddles with finite pnl on that date (dte >= 0)")

        # --- Density Tab ---
        with tabs["Density"]:
            st.subheader("Daily P&L Distribution")

            pnl_values = pop_df["pnl_sum"].values
            mean_pnl = np.mean(pnl_values)
            median_pnl = np.median(pnl_values)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(pnl_values, bins=80, color="#4a90d9", edgecolor="white", alpha=0.8)

            # Mean and median lines
            ax.axvline(mean_pnl, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_pnl:.4f}")
            ax.axvline(median_pnl, color="#2ecc71", linestyle="-", linewidth=2, label=f"Median: {median_pnl:.4f}")

            ax.set_xlabel("Daily P&L", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title("Distribution of Daily P&L", fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{mean_pnl:.4f}")
            c2.metric("Median", f"{median_pnl:.4f}")
            c3.metric("Std Dev", f"{np.std(pnl_values):.4f}")
            c4.metric("Days", f"{len(pnl_values):,}")

        # --- Normalized Density Tab ---
        with tabs["Norm Density"]:
            st.subheader("Daily P&L / Daily Vol Distribution")

            # Filter valid avg_vol > 0
            valid_mask = pop_df["avg_vol"] > 0
            if valid_mask.sum() > 0:
                pnl_vals = pop_df.loc[valid_mask, "pnl_sum"].values
                vol_vals = pop_df.loc[valid_mask, "avg_vol"].values  # annual vol
                daily_vol = vol_vals / 16.0  # convert to daily vol
                norm_pnl = pnl_vals / daily_vol

                with timed_collect("render_norm_density_plotly", timings):
                    render_hist_plotly(
                        norm_pnl,
                        title="Daily P&L / Daily Vol Distribution (1%-99% shown)",
                        xlabel="Daily P&L / Daily Vol"
                    )
            else:
                st.warning("No valid avg_vol data for normalization.")

        # --- Cumulative PnL Tab ---
        with tabs["Cumulative PnL"]:
            st.subheader("Cumulative PnL")

            cum_df = build_cum_df(pop_df)

            chart_mode = st.radio(
                "Chart", ["Plotly", "TradingView"],
                horizontal=True, key="cum_chart_mode"
            )

            if chart_mode == "Plotly":
                with timed_collect("render_cum_plotly", timings):
                    render_cum_plot_plotly(cum_df)
            else:
                with timed_collect("render_cum_tradingview", timings):
                    render_cum_plot_tradingview(cum_df)

            st.caption(f"Final PnL: {cum_df['pnl_cum'].iloc[-1]:.6f}")

        # --- Normalized P&L Tab ---
        with tabs["Norm PnL"]:
            st.subheader("Cumulative Normalized P&L")
            st.caption("P&L normalized by daily expected move: pnl / (vol / 16)")

            cum_df = build_cum_df(pop_df)

            with timed_collect("render_norm_pnl_plotly", timings):
                render_norm_pnl_plot_plotly(cum_df)

            if "norm_pnl_cum" in cum_df.columns:
                st.caption(f"Final Normalized PnL: {cum_df['norm_pnl_cum'].iloc[-1]:.2f}")

        # --- MV Tab ---
        with tabs["MV"]:
            st.subheader("Population MV (sum)")

            with timed_collect("render_mv_plotly", timings):
                render_timeseries_plotly(pop_df, ["mv_sum"], "MV Sum")
            st.caption(f"MV sum across {pop_df['n_straddles'].mean():.1f} avg contributors/day")

        # --- Contributors Tab ---
        with tabs["Contributors"]:
            st.subheader("Daily P&L Contributors")

            with timed_collect("render_contributors_plotly", timings):
                render_timeseries_plotly(pop_df, ["n_straddles"], "Contributors")
            st.caption(f"Straddles with valid P&L per day | Avg: {pop_df['n_straddles'].mean():.1f} | Max: {pop_df['n_straddles'].max():,}")

        # --- Avg Vol Tab ---
        with tabs["Avg Vol"]:
            st.subheader("Daily Average Vol")

            with timed_collect("render_avgvol_plotly", timings):
                render_timeseries_plotly(pop_df, ["avg_vol"], "Avg Vol")
            st.caption(f"Average vol across contributors | Mean: {pop_df['avg_vol'].mean():.4f} | Max: {pop_df['avg_vol'].max():.4f}")

        # --- Straddles Tab ---
        with tabs["Straddles"]:
            st.subheader("Per-Straddle Summary")

            # Use pre-computed arrays (no kernel call!)
            if len(idx_sel) > 0:
                # Summary metrics (computed from arrays, no dataframe needed)
                c1, c2, c3 = st.columns(3)
                c1.metric("Straddles", f"{len(idx_sel):,}")
                c2.metric("Total PnL", f"{pnl_sum_sel.sum():.4f}")
                c3.metric("Total P&L Days", f"{pnl_days_sel.sum():,}")

                # Build dataframe from arrays (fast) - use vectorized slicing
                str_df = pd.DataFrame({
                    "asset": asset_str[idx_sel],  # Vectorized slice, no str() calls
                    "year": straddles["year"][idx_sel].astype(np.int32),
                    "month": straddles["month"][idx_sel].astype(np.int16),
                    "schid": straddles["schid"][idx_sel].astype(np.int16),
                    "ntrc": np.asarray([str(straddles["ntrc"][i]) for i in idx_sel], dtype=object),
                    "pnl_sum": pnl_sum_sel,
                    "pnl_days": pnl_days_sel,
                })

                # Sort and limit display
                sort_col = st.selectbox("Sort by", ["pnl_sum", "pnl_days", "year"], key="str_sort")
                ascending = sort_col == "year"
                str_df = str_df.sort_values(sort_col, ascending=ascending)

                top_n = st.slider("Rows to show", 100, min(20000, len(str_df)), min(5000, len(str_df)), step=100, key="str_topn")
                st.dataframe(str_df.head(top_n), width='stretch', height=500)

                if len(str_df) > top_n:
                    st.caption(f"Showing top {top_n:,} of {len(str_df):,} straddles")

        # --- Assets Tab ---
        with tabs["Assets"]:
            st.subheader("Per-Asset Summary")

            # Use pre-computed arrays (no kernel call!)
            if len(idx_sel) > 0:
                # Slice precomputed asset IDs (same length as idx_sel, no re-factorization)
                asset_ids = asset_ids_all[idx_sel]
                n_assets = len(asset_names_all)

                # Fast aggregation with bincount
                with timed_collect("assets_bincount", timings, meta=f"(n={len(idx_sel):,})"):
                    asset_pnl_sum = np.bincount(asset_ids, weights=pnl_sum_sel, minlength=n_assets)
                    asset_pnl_days = np.bincount(asset_ids, weights=pnl_days_sel.astype(np.float64), minlength=n_assets)
                    asset_n_straddles = np.bincount(asset_ids, minlength=n_assets)
                    asset_vol_sum = np.bincount(asset_ids, weights=vol_sum_sel, minlength=n_assets)
                    asset_vol_days = np.bincount(asset_ids, weights=vol_days_sel.astype(np.float64), minlength=n_assets)

                # Compute avg_vol (avoid div by zero)
                avg_vol = asset_vol_sum / np.maximum(asset_vol_days, 1.0)

                # Compute daily metrics for bar charts
                mean_daily_pnl = asset_pnl_sum / np.maximum(asset_pnl_days, 1.0)
                mean_daily_vol = avg_vol / 16.0  # avg_vol is annual, divide by 16 for daily
                pnl_over_daily_vol = mean_daily_pnl / np.maximum(mean_daily_vol, 1e-12)

                # Build dataframe with pct columns
                total_pnl = asset_pnl_sum.sum()
                asset_df = pd.DataFrame({
                    "asset": asset_names_all,
                    "pnl_sum": asset_pnl_sum,
                    "pnl_pct": asset_pnl_sum / (total_pnl if total_pnl != 0 else 1) * 100,
                    "pnl_days": asset_pnl_days.astype(np.int64),
                    "n_straddles": asset_n_straddles,
                    "avg_vol": avg_vol,
                    "mean_daily_pnl": mean_daily_pnl,
                    "mean_daily_vol": mean_daily_vol,
                    "pnl_over_daily_vol": pnl_over_daily_vol,
                })
                asset_df = asset_df[asset_df["n_straddles"] > 0].sort_values("pnl_sum", ascending=False).reset_index(drop=True)

                # Add cumulative pct
                asset_df["pnl_pct_cum"] = asset_df["pnl_pct"].cumsum()

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Assets", f"{len(asset_df):,}")
                c2.metric("Total PnL", f"{asset_df['pnl_sum'].sum():.4f}")
                c3.metric("Total P&L Days", f"{asset_df['pnl_days'].sum():,}")
                c4.metric("Avg Vol", f"{asset_df['avg_vol'].mean():.4f}")

                # Display with conditional formatting
                styled_df = asset_df.style.format({
                    "pnl_sum": "{:.4f}",
                    "pnl_pct": "{:.2f}%",
                    "pnl_pct_cum": "{:.1f}%",
                    "pnl_days": "{:,}",
                    "n_straddles": "{:,}",
                    "avg_vol": "{:.4f}",
                }).background_gradient(subset=["pnl_sum"], cmap="RdYlGn"
                ).background_gradient(subset=["avg_vol"], cmap="Blues")

                with timed_collect("render_assets_table", timings, meta=f"(rows={len(asset_df)})"):
                    st.dataframe(styled_df, width='stretch', height=500)

                st.divider()
                st.subheader("Asset Drilldown")

                asset_choice = st.selectbox(
                    "Select asset",
                    options=asset_df["asset"].tolist(),
                    index=0,
                    key="asset_drill_select"
                )

                with timed_collect("asset_drilldown_pop_daily", timings, meta=f"({asset_choice})"):
                    asset_pop = compute_population_daily_for_asset(filter_key, asset_choice)

                if asset_pop.empty:
                    st.warning("No data for this asset in the current filter range.")
                else:
                    asset_daily = build_asset_daily_df(asset_pop)
                    with timed_collect("asset_drilldown_plot", timings, meta=f"({asset_choice})"):
                        render_asset_pnl_plotly(asset_choice, asset_daily)

                st.divider()
                st.subheader("Asset Bar Charts")

                if len(asset_df) > 10:
                    bar_top_n = st.slider("Assets in bar charts", 10, len(asset_df), len(asset_df), step=10, key="asset_bar_topn")
                else:
                    bar_top_n = len(asset_df)

                with timed_collect("render_asset_bar_ratio", timings):
                    # Sort by pnl/vol ratio for this chart
                    ratio_df = asset_df.sort_values("pnl_over_daily_vol", ascending=False)
                    render_asset_bar_ratio(ratio_df, top_n=bar_top_n)

                vol_kind = st.radio("Vol scale", ["annual", "daily"], horizontal=True, key="vol_scale")
                with timed_collect("render_asset_bar_vol", timings):
                    render_asset_bar_vol(asset_df, top_n=bar_top_n, which=vol_kind)

        # --- P&L Matrix Tab ---
        with tabs["P&L Matrix"]:
            st.subheader("P&L by Year x Month")

            # Use pre-computed pnl_sum_sel (no kernel call!)
            with timed_collect("compute_pnl_matrix", timings):
                pnl_mat, year_range = compute_ym_matrix_from_cache(
                    tuple(filtered_indices), pnl_sum_sel, "pnl"
                )

            if not pnl_mat.empty:
                with timed_collect("render_pnl_heatmap", timings):
                    render_matrix_view(pnl_mat, year_range, "P&L by Year x Month", ".4f", "RdYlGn", "pnl_mat_view")

        # --- Live Days Matrix Tab ---
        with tabs["Live Days"]:
            st.subheader("Live Days by Year x Month")

            # Use pre-computed pnl_days_sel (no kernel call!)
            with timed_collect("compute_days_matrix", timings):
                days_mat, year_range = compute_ym_matrix_from_cache(
                    tuple(filtered_indices), pnl_days_sel, "live_days"
                )

            if not days_mat.empty:
                with timed_collect("render_days_heatmap", timings):
                    render_matrix_view(days_mat, year_range, "Live Days by Year x Month", ",.0f", "Blues", "days_mat_view")

        # --- MV Matrix Tab ---
        with tabs["MV Matrix"]:
            st.subheader("MV by Year x Month")

            # Use pre-computed mv_sum_sel (no kernel call!)
            with timed_collect("compute_mv_matrix", timings):
                mv_mat, year_range = compute_ym_matrix_from_cache(
                    tuple(filtered_indices), mv_sum_sel, "mv"
                )

            if not mv_mat.empty:
                with timed_collect("render_mv_heatmap", timings):
                    render_matrix_view(mv_mat, year_range, "MV by Year x Month", ".2f", "Purples", "mv_mat_view")

        # --- Correlation Tab ---
        with tabs["Correlation"]:
            st.subheader("Asset Correlation Heatmap")

            # Row 1: Basic controls + ordering
            row1 = st.columns(5)
            with row1[0]:
                corr_method = st.selectbox("Method", ["pearson", "spearman", "sign", "partial"], key="corr_method")
            with row1[1]:
                min_overlap = st.slider("Min overlap days", 10, 250, 30, step=10, key="corr_min_overlap")
            with row1[2]:
                display_mode = st.radio("Assets", ["All", "Top N"], horizontal=True, key="corr_display_mode")
            with row1[3]:
                if display_mode == "Top N":
                    top_n_assets = st.slider("N", 20, 500, 60, step=10, key="corr_top_n")
                else:
                    top_n_assets = None
                    st.caption("All assets")
            with row1[4]:
                order_mode = st.radio("Order", ["Original", "Clustered"], horizontal=True, key="corr_order_mode")

            # Row 2: Clustering controls (only if Clustered)
            dist_mode = "1-corr"
            linkage_method = "average"
            cluster_k = 8
            show_separators = True

            if order_mode == "Clustered":
                row2 = st.columns(4)
                with row2[0]:
                    dist_mode = st.selectbox("Distance", ["1-corr", "1-abs(corr)"], index=0, key="corr_dist_mode")
                with row2[1]:
                    linkage_method = st.selectbox("Linkage", ["average", "complete", "single", "ward"], index=0, key="corr_linkage")
                with row2[2]:
                    cluster_k = st.slider("Clusters (k)", 2, 20, 8, step=1, key="corr_k")
                with row2[3]:
                    show_separators = st.checkbox("Show separators", value=True, key="corr_sep")

            # Partial correlation controls
            partial_fill = "median"
            partial_estimator = "ledoitwolf"
            partial_standardize = True
            partial_mask_low_overlap = True

            if corr_method == "partial":
                partial_row = st.columns(4)
                with partial_row[0]:
                    partial_fill = st.selectbox("NaN fill", ["median", "mean", "zero"], key="partial_fill")
                with partial_row[1]:
                    partial_estimator = st.selectbox("Estimator", ["ledoitwolf", "oas", "ridge"], key="partial_estimator")
                with partial_row[2]:
                    partial_standardize = st.checkbox("Standardize", value=True, key="partial_std")
                with partial_row[3]:
                    partial_mask_low_overlap = st.checkbox("Mask low overlap", value=True, key="partial_mask")

            # Method explanation captions
            if corr_method == "sign":
                st.caption("Sign correlation: how often two assets are on the same side of their median (robust to outliers)")
            elif corr_method == "partial":
                st.caption("Partial correlation: correlation after controlling for all other assets (reveals direct relationships)")

            with timed_collect("compute_asset_daily_matrix", timings):
                dfX, all_assets = compute_asset_daily_matrix(filter_key)

            if dfX.empty:
                st.warning("No data for correlation.")
            else:
                # Compute correlation on FULL asset set
                with timed_collect("compute_correlation_full", timings):
                    if corr_method in ("pearson", "spearman"):
                        corr_full = dfX.corr(method=corr_method, min_periods=min_overlap)
                        notna_full = np.asarray(dfX.notna(), dtype=np.int32)
                        overlap_full = notna_full.T @ notna_full
                    elif corr_method == "sign":
                        corr_mat_full, overlap_full, _ = compute_sign_corr(dfX, min_overlap)
                        corr_full = pd.DataFrame(corr_mat_full, index=dfX.columns, columns=dfX.columns)
                    else:  # partial
                        corr_mat_full, overlap_full, _ = compute_partial_corr(
                            dfX, min_overlap,
                            fill_mode=partial_fill,
                            estimator=partial_estimator,
                            standardize=partial_standardize,
                            mask_low_overlap=partial_mask_low_overlap,
                        )
                        corr_full = pd.DataFrame(corr_mat_full, index=dfX.columns, columns=dfX.columns)

                # Determine display subset
                if display_mode == "Top N" and top_n_assets is not None:
                    valid_counts = dfX.notna().sum().sort_values(ascending=False)
                    display_assets = valid_counts.head(top_n_assets).index.tolist()
                    st.caption(f"Showing top {len(display_assets)} assets (of {len(all_assets)}) by data coverage")
                else:
                    display_assets = list(dfX.columns)
                    st.caption(f"Showing all {len(display_assets)} assets")

                # Subset correlation/overlap for display (optimized lookup)
                col_index = {a: i for i, a in enumerate(dfX.columns)}
                asset_idx = [col_index[a] for a in display_assets]
                corr_mat = corr_full.loc[display_assets, display_assets].values
                overlap_mat = overlap_full[np.ix_(asset_idx, asset_idx)]
                asset_names = np.array(display_assets, dtype=object)

                # Apply clustering if requested
                boundaries = []
                if order_mode == "Clustered":
                    with timed_collect("compute_clustering", timings):
                        if show_separators:
                            perm, boundaries = cluster_boundaries_from_corr(
                                corr_mat, dist_mode=dist_mode,
                                linkage_method=linkage_method, k=cluster_k
                            )
                        else:
                            perm = cluster_order_from_corr(
                                corr_mat, dist_mode=dist_mode,
                                linkage_method=linkage_method
                            )
                        # Reorder everything
                        asset_names = asset_names[perm]
                        corr_mat = corr_mat[np.ix_(perm, perm)]
                        overlap_mat = overlap_mat[np.ix_(perm, perm)]

                # Render heatmap
                with timed_collect("render_correlation_heatmap", timings):
                    method_name = {"pearson": "Pearson", "spearman": "Spearman", "sign": "Sign", "partial": "Partial"}[corr_method]
                    if order_mode == "Clustered":
                        title_prefix = f"Clustered {method_name} Correlation"
                    else:
                        title_prefix = f"{method_name} Correlation"
                    render_correlation_heatmap(
                        corr_mat=corr_mat,
                        overlap=overlap_mat,
                        asset_names=asset_names,
                        boundaries=boundaries if show_separators else None,
                        title_prefix=title_prefix
                    )

                # Summary stats
                mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
                upper_vals = corr_mat[mask]
                upper_vals = upper_vals[np.isfinite(upper_vals)]

                if len(upper_vals) > 0:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Assets", f"{len(display_assets):,}")
                    c2.metric("Mean corr", f"{np.mean(upper_vals):.3f}")
                    c3.metric("Median corr", f"{np.median(upper_vals):.3f}")
                    c4.metric("Min corr", f"{np.min(upper_vals):.3f}")
                    c5.metric("Max corr", f"{np.max(upper_vals):.3f}")

        # --- CorGraph Tab ---
        with tabs["CorGraph"]:
            st.subheader("Correlation Network")

            # Two columns: viz (75%) | controls (25%)
            viz_col, ctrl_col = st.columns([3, 1], gap="large")

            # ========== RIGHT COLUMN: Controls ==========
            with ctrl_col:
                cfg = render_corgraph_controls()

            # ========== LEFT COLUMN: Tabs + Visualization ==========
            with viz_col:
                # Compute state once (shared across all views)
                state = compute_corgraph_state(filter_key, cfg, timings)

                if state is None:
                    st.warning("No correlation data available.")
                else:
                    # Update session_state with connected assets for focus selector (set for dedup, sorted for stability)
                    connected = sorted({state.assets[n] for n in state.G.nodes() if state.G.degree(n) > 0})
                    st.session_state["cg_connected_assets"] = connected

                    # Store total asset count for slider range
                    st.session_state["cg_total_assets"] = state.n_display_assets

                    # Warn user if focus is stale (asset no longer in graph)
                    if cfg["focus_asset"] not in (["None"] + connected):
                        st.caption(f"Focus '{cfg['focus_asset']}' not in current graph; choose again.")

                    # Edge count warnings
                    if len(state.edges) > MAX_EDGES_HARD_CAP:
                        st.warning(f"Too many edges ({len(state.edges):,}). Raise threshold or lower max edges/node.")
                    elif len(state.edges) == 0:
                        st.warning("No edges passed filters. Try lowering Top % threshold.")
                    else:
                        if len(state.edges) > 1500:
                            st.info(f"Many edges ({len(state.edges):,}). Plot may be slow.")

                        # View tabs
                        view_tabs = st.tabs(["Plotly", "D3 Interactive", "D3 Living"])

                        # ----- Plotly Tab -----
                        with view_tabs[0]:
                            render_corgraph_metrics_strip(state)
                            with timed_collect("corgraph_render", timings):
                                render_corgraph_plotly_view(state, cfg)

                            with st.expander("Edge Stats", expanded=False):
                                # Edge statistics as dataframe
                                # Note: state.edges includes MST edges merged in, so mean |corr| may be lower than threshold
                                corrs = [e[2] for e in state.edges]
                                n_mst = len(state.mst_edge_set) if state.mst_edge_set else 0
                                n_thresh = len(state.edges) - n_mst  # Approximate (MST may overlap)
                                edge_stats = pd.DataFrame({
                                    "Metric": ["Final edges", "Thresholded", "MST added", "Mean |corr|", "Common rows"],
                                    "Value": [
                                        len(state.edges),
                                        f"~{n_thresh}",
                                        n_mst,
                                        f"{np.mean(np.abs(corrs)):.3f}" if corrs else "N/A",
                                        state.n_common,
                                    ]
                                })
                                st.dataframe(edge_stats, hide_index=True, use_container_width=True)

                            if cfg["detect_communities"] and state.comms:
                                with st.expander(f"Communities ({len(state.comms)})", expanded=False):
                                    comm_df = community_summary(state.comms, state.assets, state.node_meta)
                                    st.dataframe(comm_df, hide_index=True)

                        # ----- D3 Interactive Tab -----
                        with view_tabs[1]:
                            render_corgraph_metrics_strip(state)
                            with timed_collect("corgraph_d3_render", timings):
                                render_corgraph_d3_interactive_view(state, cfg)

                        # ----- D3 Living Tab -----
                        with view_tabs[2]:
                            st.info("This view animates a force-directed layout — may be CPU intensive.")
                            if st.checkbox("Enable animation", value=False, key="cg_anim_enable"):
                                render_corgraph_metrics_strip(state)
                                render_corgraph_d3_animated_view(filter_key, state, cfg, timings)
                            else:
                                st.caption("Animation disabled. Enable above to start.")
                                render_corgraph_metrics_strip(state)
                                # Show static snapshot instead
                                render_corgraph_d3_interactive_view(state, cfg)

        # --- Embedding Tab ---
        with tabs["Embedding"]:
            st.subheader("Asset Embedding (Dimensionality Reduction)")

            # Controls row 1
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                emb_method = st.selectbox("Method", ["UMAP", "t-SNE", "MDS"], key="emb_method")
            with col2:
                emb_corr_method = st.selectbox("Correlation", ["pearson", "spearman", "sign", "partial"], key="emb_corr_method")
            with col3:
                emb_distance = st.radio("Distance", ["1-abs(corr)", "1-corr"], horizontal=True, key="emb_distance")
                st.caption("1-abs: anti-correlated cluster together; 1-corr: they separate")
            with col4:
                emb_corr_min_periods = st.slider("Corr min periods", 2, 30, 5, step=1, key="emb_corr_min_periods")

            # Controls row 2
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                emb_asset_mode = st.radio("Assets", ["All", "Top N"], horizontal=True, key="emb_asset_mode")
            with col2:
                if emb_asset_mode == "Top N":
                    emb_top_n = st.slider("N", 20, 300, 100, step=10, key="emb_top_n")
                else:
                    emb_top_n = None
                    st.caption("All assets")
            with col3:
                emb_min_coverage = st.slider("Min coverage days", 0, 100, 0, step=5, key="emb_min_coverage")
            with col4:
                emb_seed = st.number_input("Seed", 1, 999, 42, key="emb_seed")

            # Controls row 3
            col1, col2 = st.columns(2)
            with col1:
                emb_show_labels = st.checkbox("Show labels", value=False, key="emb_show_labels")
            with col2:
                color_options = ["none", "coverage", "mean_daily_pnl", "mean_daily_vol", "pnl_over_daily_vol"]
                emb_color_by = st.selectbox("Color by", color_options, key="emb_color_by")

            # Method-specific parameters
            if emb_method == "UMAP":
                col1, col2 = st.columns(2)
                with col1:
                    emb_n_neighbors = st.slider("n_neighbors", 2, 50, 15, key="emb_n_neighbors")
                with col2:
                    emb_min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.05, key="emb_min_dist")
                emb_perplexity = 30  # unused
            elif emb_method == "t-SNE":
                emb_perplexity = st.slider("perplexity", 2, 50, 30, key="emb_perplexity")
                st.caption("Auto-clamped to (n-1)/3 if needed")
                emb_n_neighbors = 15  # unused
                emb_min_dist = 0.1  # unused
            else:  # MDS
                st.caption("MDS has no tunable parameters")
                emb_perplexity = 30
                emb_n_neighbors = 15
                emb_min_dist = 0.1

            # Partial correlation controls (Embedding)
            emb_partial_fill = "median"
            emb_partial_estimator = "ledoitwolf"
            emb_partial_standardize = True
            emb_partial_mask = True

            if emb_corr_method == "partial":
                emb_partial_row = st.columns(4)
                with emb_partial_row[0]:
                    emb_partial_fill = st.selectbox("NaN fill", ["median", "mean", "zero"], key="emb_partial_fill")
                with emb_partial_row[1]:
                    emb_partial_estimator = st.selectbox("Estimator", ["ledoitwolf", "oas", "ridge"], key="emb_partial_estimator")
                with emb_partial_row[2]:
                    emb_partial_standardize = st.checkbox("Standardize", value=True, key="emb_partial_std")
                with emb_partial_row[3]:
                    emb_partial_mask = st.checkbox("Mask low overlap", value=True, key="emb_partial_mask")

            # Method caption
            if emb_corr_method == "sign":
                st.caption("Sign correlation: robust to outliers")
            elif emb_corr_method == "partial":
                st.caption("Partial correlation: reveals direct relationships")

            # Compute embedding
            with timed_collect("embedding_compute", timings):
                coords, emb_assets, emb_coverage = compute_embedding(
                    filter_key,
                    method=emb_method,
                    corr_method=emb_corr_method,
                    corr_min_periods=emb_corr_min_periods,
                    distance_mode=emb_distance,
                    n_assets=emb_top_n,
                    min_coverage_days=emb_min_coverage,
                    perplexity=emb_perplexity,
                    n_neighbors=emb_n_neighbors,
                    min_dist=emb_min_dist,
                    seed=emb_seed,
                    partial_fill=emb_partial_fill,
                    partial_estimator=emb_partial_estimator,
                    partial_standardize=emb_partial_standardize,
                    partial_mask=emb_partial_mask,
                )

            if len(coords) == 0:
                st.warning("Not enough assets for embedding (need at least 3).")
            else:
                # Build node metadata for coloring/hover
                node_meta = {}
                for a in emb_assets:
                    node_meta[a] = {"coverage": int(emb_coverage.get(a, 0))}

                # Add asset metrics if available (use locals().get to avoid NameError)
                asset_df_local = locals().get("asset_df", None)
                if asset_df_local is not None:
                    meta_lookup = asset_df_local.set_index("asset")[["mean_daily_pnl", "mean_daily_vol", "pnl_over_daily_vol"]].to_dict("index")
                    for a in emb_assets:
                        if a in meta_lookup:
                            node_meta[a].update(meta_lookup[a])

                st.caption(f"Embedding {len(emb_assets)} assets using {emb_method}")

                # Render
                with timed_collect("embedding_render", timings):
                    fig = render_embedding_plotly(
                        coords, emb_assets,
                        color_by=emb_color_by,
                        node_meta=node_meta,
                        show_labels=emb_show_labels,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                # Summary metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Assets", f"{len(emb_assets):,}")
                spread_x = float(np.std(coords[:, 0]))
                spread_y = float(np.std(coords[:, 1]))
                c2.metric("Spread (x)", f"{spread_x:.3f}")
                c3.metric("Spread (y)", f"{spread_y:.3f}")

        # --- Coverage Tab ---
        with tabs["Coverage"]:
            st.subheader("Coverage & Overlap Explorer")

            with timed_collect("coverage_compute", timings):
                cov_dfX, cov_all_assets = compute_asset_daily_matrix(filter_key)

            if cov_dfX.empty:
                st.warning("No data for coverage analysis.")
            else:
                coverage = cov_dfX.notna().sum().sort_values(ascending=False)

                # Coverage histogram
                st.write("**Coverage Distribution**")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=coverage.values, nbinsx=50, marker_color="#2962FF"))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Days with data", yaxis_title="Assets")
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                # Stats
                cov_c1, cov_c2, cov_c3, cov_c4 = st.columns(4)
                cov_c1.metric("Assets", f"{len(coverage):,}")
                cov_c2.metric("Median coverage", f"{int(coverage.median())}")
                cov_c3.metric("Min", f"{int(coverage.min())}")
                cov_c4.metric("Max", f"{int(coverage.max())}")

                # Sparse assets table (isolated in fragment - slider won't trigger overlap recompute)
                render_low_coverage_table(coverage)

                # Overlap matrix for all selected assets
                st.divider()
                st.write("**Pairwise Overlap (Selected Assets)**")

                # Use all assets from current filter (cov_dfX.columns)
                top_assets = list(cov_dfX.columns)
                notna_sub = np.asarray(cov_dfX.notna(), dtype=np.int32)
                overlap_sub = notna_sub.T @ notna_sub

                # Render as heatmap
                n = len(top_assets)
                show_ticks = n <= 40

                fig = go.Figure(data=go.Heatmap(
                    z=overlap_sub,
                    x=top_assets,
                    y=top_assets,
                    colorscale="Blues",
                    colorbar=dict(title="Overlap days"),
                ))
                fig.update_layout(
                    height=min(1200, max(450, 6 * n)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=show_ticks, tickfont=dict(size=7)),
                    yaxis=dict(showticklabels=show_ticks, tickfont=dict(size=7), autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                # Overlap stats
                mask = np.triu(np.ones_like(overlap_sub, dtype=bool), k=1)
                ov_vals = overlap_sub[mask]
                if ov_vals.size == 0:
                    st.caption("Need at least 2 assets to compute pairwise overlap stats.")
                else:
                    ov_c1, ov_c2, ov_c3 = st.columns(3)
                    ov_c1.metric("Min overlap", f"{int(ov_vals.min())}")
                    ov_c2.metric("Median overlap", f"{int(np.median(ov_vals))}")
                    ov_c3.metric("Max overlap", f"{int(ov_vals.max())}")

                # Common coverage tradeoff plot
                st.divider()
                st.write("**Common Coverage vs Assets Removed**")

                trade = compute_common_coverage_tradeoff(cov_dfX)

                if trade.empty:
                    st.caption("No data available for coverage tradeoff plot.")
                else:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=trade["n_assets"],
                        y=trade["coverage_pct"],
                        mode="lines+markers",
                        name="Greedy best common coverage",
                        hovertemplate="Assets: %{x}<br>Coverage: %{y:.2f}%<br>Common rows: %{customdata[0]}<br>Dropped: %{customdata[1]}<extra></extra>",
                        customdata=np.stack([trade["common_rows"].to_numpy(), trade["dropped"].fillna("").to_numpy()], axis=1),
                    ))

                    # Show single-best-asset ceiling (max individual coverage)
                    col_cov = cov_dfX.notna().sum(axis=0)
                    if len(col_cov) > 0:
                        best_one = 100.0 * float(col_cov.max()) / max(len(cov_dfX), 1)
                        fig.add_hline(y=best_one, line_dash="dot", annotation_text="Best single asset")

                    fig.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Number of assets remaining",
                        yaxis_title="Common coverage (%)",
                        xaxis=dict(autorange="reversed"),  # left-to-right = kick out more assets
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                    st.caption(
                        "Greedy curve: at each step, drops the single asset that most increases the number of dates "
                        "where *all remaining* assets have data."
                    )

        # --- Attribution Tab ---
        with tabs["Attribution"]:
            st.subheader("P&L Attribution by Asset")

            # Date range selector
            dmin, dmax = pop_df.index.min().date(), pop_df.index.max().date()

            attr_col1, attr_col2 = st.columns(2)
            with attr_col1:
                attr_start = st.date_input("Start", value=dmin, min_value=dmin, max_value=dmax, key="attr_start")
            with attr_col2:
                attr_end = st.date_input("End", value=dmax, min_value=dmin, max_value=dmax, key="attr_end")

            if attr_start > attr_end:
                attr_start, attr_end = attr_end, attr_start

            # Options
            attr_opt1, attr_opt2, attr_opt3 = st.columns(3)
            with attr_opt1:
                attr_top_k = st.slider("Top K contributors", 5, 50, 15, key="attr_top_k")
            with attr_opt2:
                attr_show_normalized = st.checkbox("Show normalized", value=False, key="attr_norm")
            with attr_opt3:
                attr_show_negative = st.checkbox("Show losers", value=True, key="attr_neg")

            with timed_collect("attribution_compute", timings):
                attr_dfX, attr_assets = compute_asset_daily_matrix(filter_key)

            if attr_dfX.empty:
                st.warning("No data for attribution.")
            else:
                # Build date mask
                attr_dates = attr_dfX.index
                date_mask = (attr_dates.date >= attr_start) & (attr_dates.date <= attr_end)

                if not date_mask.any():
                    st.warning("No data in selected date range.")
                else:
                    # Sum P&L per asset in window (using normalized daily pnl)
                    attr_window = attr_dfX.loc[date_mask]
                    window_pnl = attr_window.sum(axis=0, skipna=True)

                    # Build attribution dataframe
                    attr_df = pd.DataFrame({
                        "asset": window_pnl.index,
                        "pnl": window_pnl.values,
                    })

                    attr_df = attr_df.sort_values("pnl", ascending=False)

                    # Separate winners and losers
                    winners = attr_df[attr_df["pnl"] > 0].head(attr_top_k)
                    losers = attr_df[attr_df["pnl"] < 0].tail(attr_top_k).iloc[::-1]

                    # Waterfall data
                    wf_labels = []
                    wf_values = []
                    wf_measures = []

                    for _, row in winners.iterrows():
                        wf_labels.append(row["asset"])
                        wf_values.append(row["pnl"])
                        wf_measures.append("relative")

                    if attr_show_negative:
                        if len(winners) > 0 and len(losers) > 0:
                            wf_labels.append("")
                            wf_values.append(0)
                            wf_measures.append("relative")

                        for _, row in losers.iterrows():
                            wf_labels.append(row["asset"])
                            wf_values.append(row["pnl"])
                            wf_measures.append("relative")

                    # Total
                    total_pnl = window_pnl.sum()
                    wf_labels.append("Total")
                    wf_values.append(total_pnl)
                    wf_measures.append("total")

                    # Render waterfall
                    st.write("**P&L Waterfall**")
                    fig = go.Figure(go.Waterfall(
                        name="",
                        orientation="v",
                        measure=wf_measures,
                        x=wf_labels,
                        y=wf_values,
                        connector={"line": {"width": 1}},
                        increasing={"marker": {"color": "#26a69a"}},
                        decreasing={"marker": {"color": "#ef5350"}},
                        totals={"marker": {"color": "#2962FF"}},
                    ))
                    fig.update_layout(
                        height=500,
                        margin=dict(l=10, r=10, t=40, b=100),
                        title=f"P&L Attribution: {attr_start} to {attr_end}",
                        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                        yaxis_title="Normalized P&L",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                    # Summary metrics
                    winner_pnl = winners["pnl"].sum() if len(winners) > 0 else 0
                    loser_pnl = losers["pnl"].sum() if len(losers) > 0 else 0

                    attr_m1, attr_m2, attr_m3, attr_m4 = st.columns(4)
                    attr_m1.metric("Total P&L", f"{total_pnl:.4f}")
                    attr_m2.metric(f"Top {len(winners)} winners", f"{winner_pnl:.4f}")
                    attr_m3.metric(f"Top {len(losers)} losers", f"{loser_pnl:.4f}")
                    attr_m4.metric("Other", f"{total_pnl - winner_pnl - loser_pnl:.4f}")

                    # Table
                    st.divider()
                    st.write("**Top Contributors**")
                    display_df = attr_df.head(attr_top_k * 2).copy()
                    display_df["pnl"] = display_df["pnl"].map(lambda x: f"{x:.4f}")
                    st.dataframe(display_df, use_container_width=True, height=300)

        # Record and render timings
        record_timings(timings)
        render_timings_panel()
