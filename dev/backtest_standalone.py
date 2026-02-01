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
USE_SWEEP_MERGE = True  # Toggle: True=new 3.55x faster algorithm, False=old binary search
USE_PARALLEL_MERGE = False  # Toggle: True=parallel, False=serial (test OpenMP overhead)
DEBUG = False  # Enable sanity assertions

# String dtype constants
TICKER_U = "U100"
FIELD_U = "U100"
FUT_MONTH_MAP_LEN = 12

print(f"USE_SWEEP_MERGE {USE_SWEEP_MERGE}")
print(f"USE_PARALLEL_MERGE {USE_PARALLEL_MERGE}")

# ============================================================================
# Numba kernels
# ============================================================================
@njit(cache=True)
def _binsearch(a, lo, hi, x):
    """Binary search for exact match in sorted array slice a[lo:hi]."""
    while lo < hi:
        mid = (lo + hi) // 2
        v = a[mid]
        if v < x:
            lo = mid + 1
        elif v > x:
            hi = mid
        else:
            return mid
    return -1

@njit(parallel=True, cache=True)
def _lookup_parallel(q_key, q_date, block_of, starts, ends, px_date, px_value):
    """Parallel price lookup using binary search per key block."""
    out = np.empty(len(q_key), dtype=np.float64)
    out[:] = np.nan
    n_block = len(block_of)
    for i in prange(len(q_key)):
        k = q_key[i]
        if k < 0 or k >= n_block:
            continue
        b = block_of[k]
        if b < 0:
            continue
        j = _binsearch(px_date, starts[b], ends[b], q_date[i])
        if j >= 0:
            out[i] = px_value[j]
    return out

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
# Expand to daily calendar
# ========================================================================
# daily output arrays: length = total_days (~15.4M)
d_start_ym = np.where(ntrc_id_vec == STR_F, year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)
total_days = int(np.sum(day_count_vec))

if USE_SWEEP_MERGE:
    # Sweep mode only needs d_start_ym and total_days (computed above)
    pass
else:
    # Full expand for binary search mode
    print("expand days".ljust(20, "."), end="")
    start_time = time.perf_counter()

    cumsum_vec = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)
    di = np.arange(total_days, dtype=np.int32) - np.repeat(cumsum_vec, day_count_vec)
    d_stridx = np.repeat(np.arange(len(day_count_vec), dtype=np.int32), day_count_vec)
    d_smidx = np.repeat(smidx, day_count_vec)
    d_schid = np.repeat(schid_vec, day_count_vec)
    d_epoch = _ym_epoch[d_start_ym[d_stridx] - _ym_base] + di

    print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(list(amap.keys()))} assets)")

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

if USE_SWEEP_MERGE:
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

else:
    # === Original approach (expand + binary search) - deprecated, use USE_SWEEP_MERGE=True ===
    raise NotImplementedError("Binary search fallback not updated for ID-only pipeline. Use USE_SWEEP_MERGE=True")

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
    "hedge1": d_hedge1_value,
    "hedge2": d_hedge2_value,
    "hedge3": d_hedge3_value,
    "hedge4": d_hedge4_value,
    "vol": d_vol_value,
}

    