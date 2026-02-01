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
USE_PARALLEL_MERGE = True  # Toggle: True=parallel, False=serial (test OpenMP overhead)
DEBUG = False  # Enable sanity assertions

# String dtype constants
TICKER_U = "U100"
FIELD_U = "U100"
FUT_MONTH_MAP_LEN = 12

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


# ============================================================================
# Main pipeline
# ============================================================================
def main() -> dict:
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
    # Straddle hedge tickers
    # ========================================================================
    print("hedge tickers".ljust(20, "."), end="")
    start_time = time.perf_counter()

    hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
    hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
    hedge_source_id_cds_smidx = hedge_source_id_cds[smidx]
    hedge_source_id_calc_smidx = hedge_source_id_calc[smidx]

    cond_hedge = [
        hedge_source_id_nonfut_smidx,
        hedge_source_id_fut_smidx,
        hedge_source_id_cds_smidx,
        hedge_source_id_calc_smidx
    ]

    # hedge ticker for nonfut
    hedge_ticker_smidx = hedge_ticker[smidx]

    # future calc for fut - only compute for futures (11.3% of straddles)
    fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)

    # Compute expensive string operations only for futures indices
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)

    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
    hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m
    hedge_fut_ticker_m_u = hedge_fut_ticker_m.astype(TICKER_U)  # Pre-convert once for searchsorted

    # Create full arrays with empty strings, fill only futures rows
    hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_ticker = np.full(smlen, "", dtype=nps)

    hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
    hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m
    hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m

    # hedge from cds, hedge1 from calc
    hedge_hedge_smidx = hedge_hedge[smidx]
    calc_hedge1_smidx = calc_hedge1[smidx]

    choices_hedge1t = [hedge_ticker_smidx, hedge_fut_ticker, hedge_hedge_smidx, calc_hedge1_smidx]
    choices_hedge1f = [hedge_field[smidx], "PX_LAST", "PX_LAST", ""]
    hedge1t_vec = np.select(cond_hedge, choices_hedge1t, default="")
    hedge1f_vec = np.select(cond_hedge, choices_hedge1f, default="")

    choices_hedge2t = ["", "", hedge_hedge1[smidx], calc_hedge2[smidx]]
    choices_hedge2f = ["", "", "PX_LAST", ""]
    hedge2t_vec = np.select(cond_hedge, choices_hedge2t, default="")
    hedge2f_vec = np.select(cond_hedge, choices_hedge2f, default="")

    choices_hedge3t = ["", "", "", calc_hedge3[smidx]]
    hedge3t_vec = np.select(cond_hedge, choices_hedge3t, default="")
    hedge3f_vec = np.full(len(hedge3t_vec), "", dtype="U")

    hedge4t_vec = np.where(cond_hedge[3], calc_hedge4[smidx], "")
    hedge4f_vec = hedge3f_vec

    print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

    # ========================================================================
    # Straddle vol tickers
    # ========================================================================
    print("vol tickers (select)".ljust(20, "."), end="")
    start_time = time.perf_counter()

    vol_source_id_smidx = vol_source_id[smidx]
    vol_source_id_bbg_smidx = vol_source_id_bbg[smidx]
    vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
    vol_source_id_bbg_cv_smidx = vol_source_id_cv[smidx]

    cond_vol = [
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_N),
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_F),
        (vol_source_id_bbg_lmevol_smidx),
        (vol_source_id_bbg_cv_smidx)
    ]

    vol_ticker_smidx = vol_ticker[smidx]
    vol_near_smidx = vol_near[smidx]
    vol_far_smidx = vol_far[smidx]
    vol_tkrz_smidx = hedge_fut_code_smidx + "R" + hedge_fut_tail_smidx

    choices_volt = [vol_ticker_smidx, vol_ticker_smidx, vol_tkrz_smidx, vol_near_smidx]
    volt_vec = np.select(cond_vol, choices_volt, default="")

    choices_volf = [vol_near_smidx, vol_far_smidx, "PX_LAST", "none"]
    volf_vec = np.select(cond_vol, choices_volf, default="")

    print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

    print("vol tickers (cond)".ljust(20, "."), end="")
    start_time = time.perf_counter()
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
    # Map monthly hedge tickers to price dictionary IDs
    # ========================================================================
    print("map monthly ids".ljust(20, "."), end="")
    start_time = time.perf_counter()

    # Map futures tickers with searchsorted (ticker_sorted/ticker_order precomputed in processing amt)
    hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

    # Expand asset-level IDs to straddle level (fast integer indexing)
    hedge_ticker_tid_smidx = hedge_ticker_tid[smidx]
    hedge_hedge_tid_smidx = hedge_hedge_tid[smidx]
    calc_hedge1_tid_smidx = calc_hedge1_tid[smidx]

    # Futures: full array with -1 for non-futures
    hedge_fut_ticker_tid = np.full(smlen, -1, dtype=np.int32)
    hedge_fut_ticker_tid[fut_idx] = hedge_fut_ticker_tid_m

    hedge_field_fid_smidx = hedge_field_fid[smidx]

    # np.select with integer arrays (no np.unique needed - 38x faster!)
    choices_tid = [hedge_ticker_tid_smidx, hedge_fut_ticker_tid, hedge_hedge_tid_smidx, calc_hedge1_tid_smidx]
    choices_fid = [hedge_field_fid_smidx, PX_LAST_FID, PX_LAST_FID, EMPTY_FID]

    month_tid = np.select(cond_hedge, choices_tid, default=-1).astype(np.int32)
    month_fid = np.select(cond_hedge, choices_fid, default=-1).astype(np.int32)

    print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms")

    # ========================================================================
    # Sweep merge (or binary search fallback)
    # ========================================================================
    month_key = (month_tid * np.int32(n_fields) + month_fid).astype(np.int32)
    month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
    month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)

    # DEBUG sanity check
    if DEBUG:
        end_max = int((month_out0.astype(np.int64) + day_count_vec.astype(np.int64)).max())
        assert end_max == total_days, f"out0+len mismatch: {end_max} != {total_days}"

    if USE_SWEEP_MERGE:
        # === Key-grouped merge (3.55x faster) ===
        print("sweep merge prep".ljust(20, "."), end="")
        start_time = time.perf_counter()

        order = np.lexsort((month_start_epoch, month_key))
        m_key_s = month_key[order]
        m_start_s = month_start_epoch[order]
        m_len_s = day_count_vec[order].astype(np.int32)
        m_out0_s = month_out0[order]

        # Filter out-of-range keys using boolean mask
        max_valid_key = len(px_block_of) - 1
        valid_mask = (m_key_s >= 0) & (m_key_s <= max_valid_key)
        m_key_valid = m_key_s[valid_mask].astype(np.int32)
        m_start_valid = m_start_s[valid_mask].astype(np.int32)
        m_len_valid = m_len_s[valid_mask].astype(np.int32)
        m_out0_valid = m_out0_s[valid_mask].astype(np.int32)

        chg = np.flatnonzero(m_key_valid[1:] != m_key_valid[:-1]) + 1
        g_starts = np.r_[0, chg].astype(np.int32)
        g_ends = np.r_[chg, len(m_key_valid)].astype(np.int32)
        g_keys = m_key_valid[g_starts]

        print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({len(g_keys):,} groups)")

        merge_label = "sweep merge" if USE_PARALLEL_MERGE else "sweep merge (serial)"
        print(merge_label.ljust(20, "."), end="")
        start_time = time.perf_counter()

        d_hedge1_value = np.full(total_days, np.nan, dtype=np.float64)
        d_hedge1_value = run_sweep_merge(
            USE_PARALLEL_MERGE, g_keys, g_starts, g_ends,
            m_start_valid, m_len_valid, m_out0_valid,
            px_block_of, px_starts, px_ends, px_date, px_value, d_hedge1_value
        )

        found = np.sum(~np.isnan(d_hedge1_value))
        print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({found:,} found, {total_days-found:,} missing)")

    else:
        # === Original approach (expand + binary search) ===
        print("expand to daily".ljust(20, "."), end="")
        start_time = time.perf_counter()

        d_tid = month_tid[d_stridx]
        d_fid = month_fid[d_stridx]
        d_key = (d_tid * np.int32(n_fields) + d_fid).astype(np.int32)
        d_epoch_i32 = d_epoch.astype(np.int32)

        print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms (total day-straddles: {total_days:,})")

        print("numba lookup".ljust(20, "."), end="")
        start_time = time.perf_counter()

        d_hedge1_value = _lookup_parallel(d_key, d_epoch_i32,
                                          px_block_of, px_starts, px_ends, px_date, px_value)

        found = np.sum(~np.isnan(d_hedge1_value))
        print(f": {1e3*(time.perf_counter()-start_time):0.3f}ms ({found:,} found, {total_days-found:,} missing)")

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
            "hedge1t", "hedge1f",
            "hedge2t", "hedge2f",
            "hedge3t", "hedge3f",
            "hedge4t", "hedge4f",
            "volt", "volf",
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
            hedge1t_vec,
            hedge1f_vec,
            hedge2t_vec,
            hedge2f_vec,
            hedge3t_vec,
            hedge3f_vec,
            hedge4t_vec,
            hedge4f_vec,
            volt_vec,
            volf_vec,
            days0_vec, days1_vec, days2_vec,
            day_count_vec,
        ]
    }

    return result


if __name__ == "__main__":
    result = main()
