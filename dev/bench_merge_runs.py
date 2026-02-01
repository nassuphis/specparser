"""Experiment: Run-based sweep merge vs current implementation.

Tests:
1. Current: _merge_per_key (per-straddle forward scan)
2. With lower_bound: Add binary search at group start
3. Run-based: Buffer once per group, copy slices
"""
import numpy as np
from numba import njit, prange
import time
from pathlib import Path
from typing import Any
import yaml

# ============================================================================
# Numba kernels: Original
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

@njit(cache=True)
def lower_bound(a, lo, hi, x):
    """Lower bound: first position where a[pos] >= x."""
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(parallel=True, cache=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Original: per-straddle forward scan."""
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


@njit(parallel=True, cache=True)
def _merge_per_key_lb(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                      px_block_of, px_starts, px_ends, px_date, px_value, out):
    """With lower_bound: binary search at group start instead of linear scan."""
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

        # Binary search to first relevant price (instead of linear scan)
        first_start = m_start_s[g_starts[gi]]
        pi0 = lower_bound(px_date, ps, pe, first_start)

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


@njit(parallel=True, cache=True)
def _merge_per_key_runs(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                        px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Run-based: buffer once per group, copy slices.

    Since all intervals within a group overlap into 1 run (per diagnostics),
    we allocate one buffer per group, fill it once, then copy slices.
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

        s0 = g_starts[gi]
        s1 = g_ends[gi]
        if s0 >= s1:
            continue

        # Compute run bounds (union of all intervals in group)
        run_start = m_start_s[s0]
        run_end = run_start + m_len_s[s0]
        for si in range(s0 + 1, s1):
            st = m_start_s[si]
            en = st + m_len_s[si]
            if st < run_start:
                run_start = st
            if en > run_end:
                run_end = en

        run_len = run_end - run_start
        if run_len <= 0:
            continue

        # Allocate and fill buffer once
        buf = np.empty(run_len, dtype=np.float64)
        buf[:] = np.nan

        # Binary search to first relevant price
        pi = lower_bound(px_date, ps, pe, run_start)
        while pi < pe:
            d = px_date[pi]
            if d >= run_end:
                break
            buf[d - run_start] = px_value[pi]
            pi += 1

        # Copy slices for each straddle
        for si in range(s0, s1):
            start = m_start_s[si]
            length = m_len_s[si]
            if length <= 0:
                continue
            out0 = m_out0_s[si]
            off = start - run_start
            for i in range(length):
                out[out0 + i] = buf[off + i]

    return out


@njit(parallel=True, cache=True)
def _merge_per_key_runs_v2(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                           px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Run-based v2: direct fill without buffer allocation.

    Instead of allocating a buffer, we fill each straddle directly from prices,
    but only scan prices once using a smarter approach.
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

        s0 = g_starts[gi]
        s1 = g_ends[gi]
        if s0 >= s1:
            continue

        # Get first straddle's start for initial lower_bound
        first_start = m_start_s[s0]
        pi = lower_bound(px_date, ps, pe, first_start)

        # Scan prices once, writing to all applicable straddles
        while pi < pe:
            d = px_date[pi]
            v = px_value[pi]

            # For each straddle, check if date d falls within its interval
            for si in range(s0, s1):
                start = m_start_s[si]
                length = m_len_s[si]
                if d >= start and d < start + length:
                    out0 = m_out0_s[si]
                    out[out0 + (d - start)] = v

            pi += 1

    return out


@njit(parallel=True, cache=True)
def _merge_per_key_runs_v3(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                           px_block_of, px_starts, px_ends, px_date, px_value, out,
                           max_buf_size):
    """Run-based v3: hybrid with capped buffer size.

    For small runs, use buffer. For large runs, fall back to per-straddle scan.
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

        s0 = g_starts[gi]
        s1 = g_ends[gi]
        if s0 >= s1:
            continue

        # Compute run bounds
        run_start = m_start_s[s0]
        run_end = run_start + m_len_s[s0]
        for si in range(s0 + 1, s1):
            st = m_start_s[si]
            en = st + m_len_s[si]
            if st < run_start:
                run_start = st
            if en > run_end:
                run_end = en

        run_len = run_end - run_start

        if run_len <= max_buf_size:
            # Small run: use buffer
            buf = np.empty(run_len, dtype=np.float64)
            buf[:] = np.nan

            pi = lower_bound(px_date, ps, pe, run_start)
            while pi < pe:
                d = px_date[pi]
                if d >= run_end:
                    break
                buf[d - run_start] = px_value[pi]
                pi += 1

            for si in range(s0, s1):
                start = m_start_s[si]
                length = m_len_s[si]
                if length <= 0:
                    continue
                out0 = m_out0_s[si]
                off = start - run_start
                for i in range(length):
                    out[out0 + i] = buf[off + i]
        else:
            # Large run: per-straddle scan (original approach with lower_bound)
            pi0 = lower_bound(px_date, ps, pe, m_start_s[s0])

            for si in range(s0, s1):
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
# Setup: Load data
# ============================================================================
print("Loading data...")

# String dtype constants
TICKER_U = "U100"
FUT_MONTH_MAP_LEN = 12

def map_to_id_searchsorted(query_u, sorted_u, order_arr):
    pos = np.searchsorted(sorted_u, query_u)
    pos_clipped = np.minimum(pos, len(sorted_u) - 1)
    valid = (pos < len(sorted_u)) & (sorted_u[pos_clipped] == query_u)
    return np.where(valid, order_arr[pos_clipped], -1).astype(np.int32)

# Load YAML
amt_resolved = str(Path("data/amt.yml").resolve())
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

with open(amt_resolved, "r") as f:
    run_options = yaml.load(f, Loader=Loader)

# Processing amt (abbreviated version)
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
nps = np.dtypes.StringDType()

hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_ticker = np.array(list(map(lambda a: amap[a]["Hedge"].get("Ticker", ""), anames)), dtype=nps)
hedge_field = np.array(list(map(lambda a: amap[a]["Hedge"].get("Field", ""), anames)), dtype=nps)
hedge_hedge = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge", ""), anames)), dtype=nps)
hedge_ccy = np.array(list(map(lambda a: amap[a]["Hedge"].get("ccy", ""), anames)), dtype=nps)
hedge_tenor = np.array(list(map(lambda a: amap[a]["Hedge"].get("tenor", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)

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

calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor
hedge_fut_month_mtrx = hedge_fut_month_map.astype('S12').view('S1').reshape(-1, FUT_MONTH_MAP_LEN).astype('U1')
hedge_min_year_offset_int = hedge_min_year_offset.astype(np.int64)

achk = np.array([np.sum(np.frombuffer(x.encode('ascii'), dtype=np.uint8)) for x in anames], dtype=np.int64)
aschcnt = np.array(list(map(lambda a: len(expiry_schedules[amap[a]["Options"]]), anames)), dtype=np.int64)

easchcnt = np.repeat(aschcnt, aschcnt)
eastmp = np.concatenate(list(map(
    lambda a: np.array(expiry_schedules[amap[a]["Options"]], dtype="|U20"), anames
)), dtype="|U20")

easchj = np.arange(np.sum(aschcnt)) - np.repeat(np.cumsum(aschcnt) - aschcnt, aschcnt)
easchi = np.repeat(np.arange(len(anames)), aschcnt)

easntrcv, _, rest = np.strings.partition(eastmp, '_')
ntrc_flat = np.strings.slice(easntrcv, 1)

ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrc2id = dict(zip(ntrc_uniq, range(len(ntrc_uniq))))
STR_F = ntrc2id.get("F", -1)

max_schedules = int(np.max(easchcnt))
schedule_id_matrix = np.full((len(amap), max_schedules, 5), -1, dtype=np.int32)
schedule_id_matrix[easchi, easchj, 0] = ntrc_ids_flat

ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)
n_fields = len(field_arr)

ticker_to_id = {s: i for i, s in enumerate(ticker_arr)}
field_to_id = {s: i for i, s in enumerate(field_arr)}
ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker], dtype=np.int32)
hedge_hedge_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge], dtype=np.int32)
calc_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge1], dtype=np.int32)
hedge_field_fid = np.array([field_to_id.get(str(s), -1) for s in hedge_field], dtype=np.int32)

PX_LAST_FID = field_to_id.get("PX_LAST", -1)
EMPTY_FID = field_to_id.get("", -1)

ym = np.arange(2001*12+1-1, 2026*12+1-1, dtype=np.int64)
ym_len = len(ym)

inas = anames
inidx = np.array([idx_map[a] for a in inas], dtype=np.uint64)
inasc = aschcnt[inidx]
sidx = np.repeat(inidx, inasc)
sc = np.repeat(inasc, inasc)
si1 = np.arange(np.sum(inasc))
si2 = np.repeat((np.cumsum(inasc)-inasc), inasc)
si = si1 - si2

smidx = np.repeat(sidx, ym_len)
smym = np.tile(ym, len(sidx))
smlen = len(smidx)

year_vec = smym // 12
month_vec = smym % 12 + 1

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

schcnt_vec = np.repeat(sc, ym_len)
schid_vec = np.repeat(si, ym_len)

schedule_id_matrix_smidx = schedule_id_matrix[smidx, schid_vec, :]
ntrc_id_vec = schedule_id_matrix_smidx[:, 0]

day_count_vec = days0_vec + days1_vec + np.where(ntrc_id_vec == STR_F, days2_vec, 0)

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

fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)

hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)

hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m
hedge_fut_ticker_m_u = hedge_fut_ticker_m.astype(TICKER_U)

hedge_fut_ticker = np.full(smlen, "", dtype=nps)
hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m

_ym_base = 2000 * 12
_ym_range = np.arange(2000*12, 2027*12)
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)

d_start_ym = np.where(ntrc_id_vec == STR_F, year2 * 12 + month2 - 1, year1 * 12 + month1 - 1)
total_days = int(np.sum(day_count_vec))

pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])

hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

hedge_ticker_tid_smidx = hedge_ticker_tid[smidx]
hedge_hedge_tid_smidx = hedge_hedge_tid[smidx]
calc_hedge1_tid_smidx = calc_hedge1_tid[smidx]

hedge_fut_ticker_tid = np.full(smlen, -1, dtype=np.int32)
hedge_fut_ticker_tid[fut_idx] = hedge_fut_ticker_tid_m

hedge_field_fid_smidx = hedge_field_fid[smidx]

choices_tid = [hedge_ticker_tid_smidx, hedge_fut_ticker_tid, hedge_hedge_tid_smidx, calc_hedge1_tid_smidx]
choices_fid = [hedge_field_fid_smidx, PX_LAST_FID, PX_LAST_FID, EMPTY_FID]

month_tid = np.select(cond_hedge, choices_tid, default=-1).astype(np.int32)
month_fid = np.select(cond_hedge, choices_fid, default=-1).astype(np.int32)

month_key = (month_tid * np.int32(n_fields) + month_fid).astype(np.int32)
month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)

order = np.lexsort((month_start_epoch, month_key))
m_key_s = month_key[order]
m_start_s = month_start_epoch[order]
m_len_s = day_count_vec[order].astype(np.int32)
m_out0_s = month_out0[order]

max_valid_key = len(px_block_of) - 1
valid_mask = (m_key_s >= 0) & (m_key_s <= max_valid_key)
m_key_valid = np.ascontiguousarray(m_key_s[valid_mask].astype(np.int32))
m_start_valid = np.ascontiguousarray(m_start_s[valid_mask].astype(np.int32))
m_len_valid = np.ascontiguousarray(m_len_s[valid_mask].astype(np.int32))
m_out0_valid = np.ascontiguousarray(m_out0_s[valid_mask].astype(np.int32))

chg = np.flatnonzero(m_key_valid[1:] != m_key_valid[:-1]) + 1
g_starts = np.ascontiguousarray(np.r_[0, chg].astype(np.int32))
g_ends = np.ascontiguousarray(np.r_[chg, len(m_key_valid)].astype(np.int32))
g_keys = np.ascontiguousarray(m_key_valid[g_starts])

print(f"Data loaded: {len(g_keys):,} groups, {len(m_key_valid):,} straddles, {total_days:,} output cells")
print()

# ============================================================================
# Warmup JIT compilation
# ============================================================================
print("Warming up JIT compilation...")
out_warmup = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key(g_keys[:10], g_starts[:10], g_ends[:10],
               m_start_valid, m_len_valid, m_out0_valid,
               px_block_of, px_starts, px_ends, px_date, px_value, out_warmup)

out_warmup = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key_lb(g_keys[:10], g_starts[:10], g_ends[:10],
                  m_start_valid, m_len_valid, m_out0_valid,
                  px_block_of, px_starts, px_ends, px_date, px_value, out_warmup)

out_warmup = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key_runs(g_keys[:10], g_starts[:10], g_ends[:10],
                    m_start_valid, m_len_valid, m_out0_valid,
                    px_block_of, px_starts, px_ends, px_date, px_value, out_warmup)

out_warmup = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key_runs_v2(g_keys[:10], g_starts[:10], g_ends[:10],
                       m_start_valid, m_len_valid, m_out0_valid,
                       px_block_of, px_starts, px_ends, px_date, px_value, out_warmup)

out_warmup = np.full(total_days, np.nan, dtype=np.float64)
_merge_per_key_runs_v3(g_keys[:10], g_starts[:10], g_ends[:10],
                       m_start_valid, m_len_valid, m_out0_valid,
                       px_block_of, px_starts, px_ends, px_date, px_value, out_warmup, 512)

print("JIT warmup complete.")
print()

# ============================================================================
# Benchmark
# ============================================================================
N_RUNS = 5

print("=" * 60)
print("BENCHMARK: Sweep Merge Variants")
print("=" * 60)
print()

results = {}

# 1. Original
print("1. Original (_merge_per_key)...")
times = []
for i in range(N_RUNS):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key(g_keys, g_starts, g_ends,
                         m_start_valid, m_len_valid, m_out0_valid,
                         px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)
found_orig = np.sum(~np.isnan(out))
out_orig = out.copy()
results["original"] = (np.median(times) * 1000, found_orig)
print(f"   {np.median(times)*1000:.1f}ms (median), found={found_orig:,}")

# 2. With lower_bound
print("2. With lower_bound (_merge_per_key_lb)...")
times = []
for i in range(N_RUNS):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key_lb(g_keys, g_starts, g_ends,
                            m_start_valid, m_len_valid, m_out0_valid,
                            px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)
found_lb = np.sum(~np.isnan(out))
results["lower_bound"] = (np.median(times) * 1000, found_lb)
print(f"   {np.median(times)*1000:.1f}ms (median), found={found_lb:,}")
assert np.allclose(out, out_orig, equal_nan=True), "lower_bound mismatch!"

# 3. Run-based buffering
print("3. Run-based buffering (_merge_per_key_runs)...")
times = []
for i in range(N_RUNS):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key_runs(g_keys, g_starts, g_ends,
                              m_start_valid, m_len_valid, m_out0_valid,
                              px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)
found_runs = np.sum(~np.isnan(out))
results["runs"] = (np.median(times) * 1000, found_runs)
print(f"   {np.median(times)*1000:.1f}ms (median), found={found_runs:,}")
assert np.allclose(out, out_orig, equal_nan=True), "runs mismatch!"

# 4. Run-based v2 (no buffer)
print("4. Run-based v2 (no buffer, scan once)...")
times = []
for i in range(N_RUNS):
    out = np.full(total_days, np.nan, dtype=np.float64)
    t0 = time.perf_counter()
    out = _merge_per_key_runs_v2(g_keys, g_starts, g_ends,
                                  m_start_valid, m_len_valid, m_out0_valid,
                                  px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    times.append(t1 - t0)
found_runs_v2 = np.sum(~np.isnan(out))
results["runs_v2"] = (np.median(times) * 1000, found_runs_v2)
print(f"   {np.median(times)*1000:.1f}ms (median), found={found_runs_v2:,}")
assert np.allclose(out, out_orig, equal_nan=True), "runs_v2 mismatch!"

# 5. Run-based v3 (hybrid with cap)
for cap in [256, 512, 1024, 2048]:
    print(f"5. Run-based v3 (cap={cap})...")
    times = []
    for i in range(N_RUNS):
        out = np.full(total_days, np.nan, dtype=np.float64)
        t0 = time.perf_counter()
        out = _merge_per_key_runs_v3(g_keys, g_starts, g_ends,
                                      m_start_valid, m_len_valid, m_out0_valid,
                                      px_block_of, px_starts, px_ends, px_date, px_value, out, cap)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    found_runs_v3 = np.sum(~np.isnan(out))
    results[f"runs_v3_{cap}"] = (np.median(times) * 1000, found_runs_v3)
    print(f"   {np.median(times)*1000:.1f}ms (median), found={found_runs_v3:,}")
    assert np.allclose(out, out_orig, equal_nan=True), f"runs_v3_{cap} mismatch!"

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

baseline = results["original"][0]
print(f"{'Variant':<25} {'Time (ms)':>10} {'Speedup':>10} {'Found':>15}")
print("-" * 60)
for name, (time_ms, found) in sorted(results.items(), key=lambda x: x[1][0]):
    speedup = baseline / time_ms
    print(f"{name:<25} {time_ms:>10.1f} {speedup:>9.2f}x {found:>15,}")

print()
print("Notes:")
print("- 'original' is current _merge_per_key")
print("- 'lower_bound' adds binary search at group start")
print("- 'runs' allocates buffer per group, fills once, copies slices")
print("- 'runs_v2' scans prices once, writes to all applicable straddles")
print("- 'runs_v3_{cap}' uses buffer for small runs, per-straddle for large")
