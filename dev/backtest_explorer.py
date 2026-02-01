"""Backtest Explorer v2 - Interactive straddle analysis."""
import re
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from numba import njit, prange

st.set_page_config(page_title="Backtest Explorer", layout="wide")

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

    Returns: (pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum) daily arrays
    """
    pnl_sum = np.zeros(grid_size, np.float64)
    pnl_cnt = np.zeros(grid_size, np.int32)
    vol_sum = np.zeros(grid_size, np.float64)
    vol_cnt = np.zeros(grid_size, np.int32)
    mv_sum = np.zeros(grid_size, np.float64)
    opnl_sum = np.zeros(grid_size, np.float64)
    hpnl_sum = np.zeros(grid_size, np.float64)

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
            if not np.isnan(p):
                pnl_sum[day_idx] += p
                pnl_cnt[day_idx] += 1

            # Vol
            v = vol[idx]
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

            # Hedge PnL (if available)
            if have_hpnl:
                hp = hpnl[idx]
                if not np.isnan(hp):
                    hpnl_sum[day_idx] += hp

    return pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum


@st.cache_data
def factorize_assets(_asset_str_tuple):
    """Convert asset strings to integer codes for bincount."""
    asset_str = np.array(_asset_str_tuple)
    codes, uniques = pd.factorize(asset_str, sort=True)
    return codes.astype(np.int32), np.asarray(uniques, dtype=object)


@st.cache_data
def compute_population_daily(_straddles, _valuations, filtered_indices_tuple):
    """Aggregate daily pnl/mv across all straddles using Numba kernel.

    Returns DataFrame indexed by date with columns:
        n_straddles, pnl_sum, mv_sum, avg_vol, [opnl_sum, hpnl_sum if available]
    """
    filtered_indices = np.array(filtered_indices_tuple, dtype=np.int64)
    if len(filtered_indices) == 0:
        return pd.DataFrame()

    straddles, valuations = _straddles, _valuations

    # Extract arrays for Numba kernel
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

    # Check for optional opnl/hpnl columns
    have_opnl = "opnl" in valuations
    have_hpnl = "hpnl" in valuations
    opnl = valuations["opnl"] if have_opnl else np.empty(1, dtype=np.float64)
    hpnl = valuations["hpnl"] if have_hpnl else np.empty(1, dtype=np.float64)

    # Single Numba pass for daily aggregation
    pnl_sum, pnl_cnt, vol_sum, vol_cnt, mv_sum, opnl_sum, hpnl_sum = _aggregate_daily(
        out0s, lens, starts, d0, pnl, vol, mv, opnl, hpnl,
        dte, have_dte, have_opnl, have_hpnl, grid_size
    )

    # Build result DataFrame
    base = np.datetime64('1970-01-01', 'D')
    dates = base + np.arange(d0, d1 + 1)

    # Only keep dates with at least one contributor
    has_data = pnl_cnt > 0

    result = {
        "n_straddles": pnl_cnt[has_data],
        "pnl_sum": pnl_sum[has_data],
        "mv_sum": mv_sum[has_data],
        "avg_vol": vol_sum[has_data] / np.maximum(vol_cnt[has_data], 1),
    }

    # Add optional columns if they exist
    if have_opnl:
        result["opnl_sum"] = opnl_sum[has_data]
    if have_hpnl:
        result["hpnl_sum"] = hpnl_sum[has_data]

    df = pd.DataFrame(result, index=pd.DatetimeIndex(dates[has_data]))
    df.index.name = "date"
    return df


@st.cache_data
def compute_all_straddle_metrics(_straddles, _valuations, filtered_indices_tuple):
    """Compute ALL per-straddle metrics in ONE Numba pass.

    Returns: (pnl_sum, pnl_days, vol_sum, vol_days, mv_sum, idx) arrays
    """
    idx = np.array(filtered_indices_tuple, dtype=np.int64)
    if len(idx) == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty, idx

    straddles, valuations = _straddles, _valuations
    out0s = straddles["out0"][idx].astype(np.int32)
    lens = straddles["length"][idx].astype(np.int32)
    pnl = valuations["pnl"]
    vol = valuations["vol"]
    mv = valuations["mv"]
    have_dte = "days_to_expiry" in valuations
    dte = valuations["days_to_expiry"] if have_dte else np.empty(1, dtype=np.int32)

    pnl_sum, pnl_days, vol_sum, vol_days, mv_sum = _summarize_all(
        out0s, lens, pnl, vol, mv, dte, have_dte
    )
    return pnl_sum, pnl_days, vol_sum, vol_days, mv_sum, idx


@st.cache_data
def compute_ym_matrix_from_cache(_straddles, filtered_indices_tuple, weights, value_type="pnl"):
    """Compute year×month matrix from CACHED per-straddle arrays (no kernel calls).

    Args:
        weights: Pre-computed per-straddle values (pnl_sum, pnl_days, or mv_sum)
        value_type: "pnl" | "live_days" | "mv" (for formatting hints)

    Returns: (matrix_df, year_range)
    """
    idx = np.array(filtered_indices_tuple, dtype=np.int64)
    if len(idx) == 0:
        return pd.DataFrame(), (0, 0)

    straddles = _straddles
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

# ============================================================================
# Main App
# ============================================================================
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
asset_str, unique_assets = get_asset_strings(straddles)
straddle_ym = get_straddle_ym(straddles)

# Precompute asset IDs once for fast slicing in Assets tab
asset_ids_all, asset_names_all = factorize_assets(tuple(asset_str))

with st.sidebar:
    st.header("Filters")

    # --- Asset Filter ---
    st.subheader("Asset")
    asset_mode = st.radio("Mode", ["Dropdown", "Regex"], horizontal=True,
                          label_visibility="collapsed")

    if asset_mode == "Dropdown":
        selected_asset = st.selectbox("Asset", ["All"] + unique_assets,
                                      label_visibility="collapsed")
        asset_value = selected_asset
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
    filtered_indices, regex_error = get_filtered_indices(
        straddles, asset_str, straddle_ym,
        asset_mode.lower(), asset_value, ym_lo, ym_hi, set(selected_schids)
    )

    if regex_error:
        st.error(f"Regex error: {regex_error}")

    st.write(f"**Matching:** {len(filtered_indices):,} straddles")

# ============================================================================
# Main Content - Population Aggregate View
# ============================================================================
if len(filtered_indices) == 0:
    st.warning("No straddles match the current filters.")
else:
    # Compute population daily aggregates (Numba kernel)
    pop_df = compute_population_daily(straddles, valuations, tuple(filtered_indices))

    # Compute ALL per-straddle metrics ONCE (Numba kernel) - reuse everywhere
    pnl_sum_sel, pnl_days_sel, vol_sum_sel, vol_days_sel, mv_sum_sel, idx_sel = \
        compute_all_straddle_metrics(straddles, valuations, tuple(filtered_indices))

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

        # Tabs: Days, Cumulative PnL, MV, Contributors, Straddles, Assets, Matrices
        tab_names = ["Days", "Density", "Norm Density", "Cumulative PnL", "MV", "Contributors", "Avg Vol", "Straddles", "Assets", "P&L Matrix", "Live Days", "MV Matrix"]
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
            st.subheader("Daily P&L / Vol Distribution")

            # Filter valid avg_vol > 0
            valid_mask = pop_df["avg_vol"] > 0
            if valid_mask.sum() > 0:
                pnl_vals = pop_df.loc[valid_mask, "pnl_sum"].values
                vol_vals = pop_df.loc[valid_mask, "avg_vol"].values
                norm_pnl = pnl_vals / vol_vals

                # Clip extreme tails for better visualization
                q01, q99 = np.percentile(norm_pnl, [1, 99])
                clipped = norm_pnl[(norm_pnl >= q01) & (norm_pnl <= q99)]

                mean_norm = np.mean(norm_pnl)
                median_norm = np.median(norm_pnl)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(clipped, bins=80, color="#9b59b6", edgecolor="white", alpha=0.8)

                # Mean and median lines (use full data stats)
                ax.axvline(mean_norm, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_norm:.4f}")
                ax.axvline(median_norm, color="#2ecc71", linestyle="-", linewidth=2, label=f"Median: {median_norm:.4f}")

                ax.set_xlabel("Daily P&L / Avg Vol", fontsize=11)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.set_title("Normalized Daily P&L Distribution (1%-99% shown)", fontsize=13, fontweight="bold")
                ax.legend(loc="upper right", fontsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.3)

                st.pyplot(fig)
                plt.close(fig)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{mean_norm:.4f}")
                c2.metric("Median", f"{median_norm:.4f}")
                c3.metric("Std Dev", f"{np.std(norm_pnl):.4f}")
                c4.metric("Valid Days", f"{len(norm_pnl):,}")
            else:
                st.warning("No valid avg_vol data for normalization.")

        # --- Cumulative PnL Tab ---
        with tabs["Cumulative PnL"]:
            st.subheader("Cumulative PnL")

            cum_df = pd.DataFrame(index=pop_df.index)
            cum_df["pnl_cum"] = pop_df["pnl_sum"].cumsum()
            if "opnl_sum" in pop_df.columns:
                cum_df["opnl_cum"] = pop_df["opnl_sum"].cumsum()
            if "hpnl_sum" in pop_df.columns:
                cum_df["hpnl_cum"] = pop_df["hpnl_sum"].cumsum()

            st.line_chart(cum_df)
            st.caption(f"Final PnL: {cum_df['pnl_cum'].iloc[-1]:.6f}")

        # --- MV Tab ---
        with tabs["MV"]:
            st.subheader("Population MV (sum)")

            st.line_chart(pop_df[["mv_sum"]])
            st.caption(f"MV sum across {pop_df['n_straddles'].mean():.1f} avg contributors/day")

        # --- Contributors Tab ---
        with tabs["Contributors"]:
            st.subheader("Daily P&L Contributors")

            st.line_chart(pop_df[["n_straddles"]])
            st.caption(f"Straddles with valid P&L per day | Avg: {pop_df['n_straddles'].mean():.1f} | Max: {pop_df['n_straddles'].max():,}")

        # --- Avg Vol Tab ---
        with tabs["Avg Vol"]:
            st.subheader("Daily Average Vol")

            st.line_chart(pop_df[["avg_vol"]])
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
                asset_pnl_sum = np.bincount(asset_ids, weights=pnl_sum_sel, minlength=n_assets)
                asset_pnl_days = np.bincount(asset_ids, weights=pnl_days_sel.astype(np.float64), minlength=n_assets)
                asset_n_straddles = np.bincount(asset_ids, minlength=n_assets)
                asset_vol_sum = np.bincount(asset_ids, weights=vol_sum_sel, minlength=n_assets)
                asset_vol_days = np.bincount(asset_ids, weights=vol_days_sel.astype(np.float64), minlength=n_assets)

                # Compute avg_vol (avoid div by zero)
                avg_vol = asset_vol_sum / np.maximum(asset_vol_days, 1.0)

                # Build dataframe with pct columns
                total_pnl = asset_pnl_sum.sum()
                asset_df = pd.DataFrame({
                    "asset": asset_names_all,
                    "pnl_sum": asset_pnl_sum,
                    "pnl_pct": asset_pnl_sum / (total_pnl if total_pnl != 0 else 1) * 100,
                    "pnl_days": asset_pnl_days.astype(np.int64),
                    "n_straddles": asset_n_straddles,
                    "avg_vol": avg_vol,
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

                st.dataframe(styled_df, width='stretch', height=500)

                # Sparklines for top assets
                show_sparklines = st.checkbox("Show sparklines (top 10 assets)", value=False, key="spark_check")

                if show_sparklines and len(asset_df) > 0:
                    top_assets = asset_df.head(10)["asset"].tolist()

                    # Get indices for each top asset
                    spark_cols = st.columns(5)
                    for i, asset_name in enumerate(top_assets):
                        asset_mask = (asset_str == asset_name)
                        asset_indices = np.where(asset_mask)[0]

                        # Filter to current selection
                        asset_sel = np.intersect1d(asset_indices, filtered_indices)

                        if len(asset_sel) > 0:
                            # Compute daily cumulative pnl for this asset
                            asset_pop = compute_population_daily(straddles, valuations, tuple(asset_sel.tolist()))
                            cum_pnl = asset_pop["pnl_sum"].cumsum()

                            with spark_cols[i % 5]:
                                st.caption(f"{asset_name}")
                                st.line_chart(cum_pnl, height=80)

        # --- P&L Matrix Tab ---
        with tabs["P&L Matrix"]:
            st.subheader("P&L by Year x Month")

            # Use pre-computed pnl_sum_sel (no kernel call!)
            pnl_mat, year_range = compute_ym_matrix_from_cache(
                straddles, tuple(filtered_indices), pnl_sum_sel, "pnl"
            )

            if not pnl_mat.empty:
                render_matrix_view(pnl_mat, year_range, "P&L by Year x Month", ".4f", "RdYlGn", "pnl_mat_view")

        # --- Live Days Matrix Tab ---
        with tabs["Live Days"]:
            st.subheader("Live Days by Year x Month")

            # Use pre-computed pnl_days_sel (no kernel call!)
            days_mat, year_range = compute_ym_matrix_from_cache(
                straddles, tuple(filtered_indices), pnl_days_sel, "live_days"
            )

            if not days_mat.empty:
                render_matrix_view(days_mat, year_range, "Live Days by Year x Month", ",.0f", "Blues", "days_mat_view")

        # --- MV Matrix Tab ---
        with tabs["MV Matrix"]:
            st.subheader("MV by Year x Month")

            # Use pre-computed mv_sum_sel (no kernel call!)
            mv_mat, year_range = compute_ym_matrix_from_cache(
                straddles, tuple(filtered_indices), mv_sum_sel, "mv"
            )

            if not mv_mat.empty:
                render_matrix_view(mv_mat, year_range, "MV by Year x Month", ".2f", "Purples", "mv_mat_view")
