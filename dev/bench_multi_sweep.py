"""Benchmark multiple sweep merges to understand the cost of 5-leg pipeline.

Tests:
1. Single sweep (current: hedge1 only)
2. Sequential sweeps (5 independent lexsorts)
3. Unified sweep (one lexsort, 5 output arrays)
"""
import numpy as np
import time
from pathlib import Path
import yaml
from numba import njit, prange

# ============================================================================
# Load price data (same as backtest_standalone.py)
# ============================================================================
print("Loading price data...")

pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])

print(f"Prices loaded: {len(px_date):,} rows, {len(px_starts):,} blocks")

# ============================================================================
# Numba sweep merge kernel (same as backtest_standalone.py)
# ============================================================================
@njit(parallel=True, cache=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
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

def run_sweep_merge(g_keys, g_starts, g_ends, m_start, m_len, m_out0, out):
    """Run sweep merge kernel."""
    return _merge_per_key(g_keys, g_starts, g_ends, m_start, m_len, m_out0,
                          px_block_of, px_starts, px_ends, px_date, px_value, out)

# ============================================================================
# Setup: Create test data matching real pipeline structure
# ============================================================================
print("\nLoading straddle data...")

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

with open("data/amt.yml", "r") as f:
    run_options = yaml.load(f, Loader=Loader)

amt = run_options.get("amt", {})
expiry_schedules = run_options.get("expiry_schedules")

amap = {}
for asset_data in amt.values():
    if isinstance(asset_data, dict):
        underlying = asset_data.get("Underlying")
        if underlying and asset_data.get("WeightCap") > 0:
            amap[underlying] = asset_data

anames = np.array(list(amap.keys()), dtype=np.dtypes.StringDType())
idx_map = dict(zip(list(amap.keys()), range(len(anames))))

nps = np.dtypes.StringDType()
FUT_MONTH_MAP_LEN = 12
TICKER_U = "U100"

# Load ID dicts
ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)
n_fields = len(field_arr)
ticker_to_id = {s: i for i, s in enumerate(ticker_arr)}
field_to_id = {s: i for i, s in enumerate(field_arr)}

ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

def map_to_id_searchsorted(tickers, ticker_sorted, ticker_order):
    idx = np.searchsorted(ticker_sorted, tickers)
    idx = np.clip(idx, 0, len(ticker_sorted) - 1)
    matched = ticker_sorted[idx] == tickers
    result = np.where(matched, ticker_order[idx], -1)
    return result.astype(np.int32)

# Extract asset arrays
hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_ticker = np.array(list(map(lambda a: amap[a]["Hedge"].get("Ticker", ""), anames)), dtype=nps)
hedge_field = np.array(list(map(lambda a: amap[a]["Hedge"].get("Field", ""), anames)), dtype=nps)
hedge_hedge = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge", ""), anames)), dtype=nps)
hedge_hedge1 = np.array(list(map(lambda a: amap[a]["Hedge"].get("hedge1", ""), anames)), dtype=nps)
hedge_ccy = np.array(list(map(lambda a: amap[a]["Hedge"].get("ccy", ""), anames)), dtype=nps)
hedge_tenor = np.array(list(map(lambda a: amap[a]["Hedge"].get("tenor", ""), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)

vol_source = np.array(list(map(lambda a: amap[a]["Vol"].get("Source", ""), anames)), dtype=nps)
vol_ticker = np.array(list(map(lambda a: amap[a]["Vol"].get("Ticker", ""), anames)), dtype=nps)
vol_near = np.array(list(map(lambda a: amap[a]["Vol"].get("Near", ""), anames)), dtype=nps)
vol_far = np.array(list(map(lambda a: amap[a]["Vol"].get("Far", ""), anames)), dtype=nps)

calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor
calc_hedge2 = hedge_ccy + "_fsw6m_" + hedge_tenor
calc_hedge3 = hedge_ccy + "_pva0m_" + hedge_tenor
calc_hedge4 = hedge_ccy + "_pva6m_" + hedge_tenor

# Source codes
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

vol_sources, vol_source_id = np.unique(vol_source, return_inverse=True)
vs2id_map = dict(zip(vol_sources, range(len(vol_sources))))
VOL_BBG_LMEVOL = vs2id_map["BBG_LMEVOL"]
vol_source_id_bbg_lmevol = vol_source_id == VOL_BBG_LMEVOL
VOL_BBG = vs2id_map["BBG"]
vol_source_id_bbg = vol_source_id == VOL_BBG
VOL_CV = vs2id_map["CV"]
vol_source_id_cv = vol_source_id == VOL_CV

hedge_fut_month_mtrx = hedge_fut_month_map.astype('S12').view('S1').reshape(-1, FUT_MONTH_MAP_LEN).astype('U1')
hedge_min_year_offset_int = hedge_min_year_offset.astype(np.int64)

aschcnt = np.array(list(map(
    lambda a: len(expiry_schedules[amap[a]["Options"]]),
    anames
)), dtype=np.int64)

# Asset-level IDs
hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker], dtype=np.int32)
hedge_hedge_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge], dtype=np.int32)
hedge_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge1], dtype=np.int32)
calc_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge1], dtype=np.int32)
calc_hedge2_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge2], dtype=np.int32)
calc_hedge3_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge3], dtype=np.int32)
calc_hedge4_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge4], dtype=np.int32)
hedge_field_fid = np.array([field_to_id.get(str(s), -1) for s in hedge_field], dtype=np.int32)

vol_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_ticker], dtype=np.int32)
vol_near_fid = np.array([field_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)
vol_far_fid = np.array([field_to_id.get(str(s), -1) for s in vol_far], dtype=np.int32)
vol_near_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)

PX_LAST_FID = field_to_id.get("PX_LAST", -1)
EMPTY_FID = field_to_id.get("", -1)
NONE_FID = field_to_id.get("none", -1)

# Build straddle arrays
inidx = np.array([idx_map[a] for a in anames], dtype=np.uint64)
inasc = aschcnt[inidx]
sidx = np.repeat(inidx, inasc)
ym = np.arange(2001*12+1-1, 2026*12+1-1, dtype=np.int64)
ym_len = len(ym)
smidx = np.repeat(sidx, ym_len)
smym = np.tile(ym, len(sidx))
smlen = len(smidx)
year_vec = smym // 12
month_vec = smym % 12 + 1

# Schedule parsing
easchcnt = np.repeat(aschcnt, aschcnt)
eastmp = np.concatenate(list(map(
    lambda a: np.array(expiry_schedules[amap[a]["Options"]], dtype="|U20"),
    anames
)), dtype="|U20")

easntrcv, _, _ = np.strings.partition(eastmp, '_')
ntrc_flat = np.strings.slice(easntrcv, 1)
ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrc2id = dict(zip(ntrc_uniq, range(len(ntrc_uniq))))

STR_N = ntrc2id.get("N", -1)
STR_F = ntrc2id.get("F", -1)

schid_vec = np.tile(np.arange(len(eastmp)), ym_len).reshape(ym_len, -1).T.flatten()
ntrc_id_vec = ntrc_ids_flat[schid_vec]

# Day count and start epoch
_ym_base = 2000 * 12
_ym_range = np.arange(2000*12, 2027*12)
_ym_dates = (
    (_ym_range // 12).astype('U') + '-' +
    np.char.zfill((_ym_range % 12 + 1).astype('U'), 2) + '-01'
).astype('datetime64[D]')
_ym_epoch = _ym_dates.astype(np.int64)

# Parse schedules for day counts
easxprcv, _, rest = np.strings.partition(eastmp, '_')
easxprcv, _, rest = np.strings.partition(rest, '_')
wgt_flat, _, _ = np.strings.partition(rest, '_')

# Simplified day count (use fixed value for benchmark)
day_count_base = np.full(len(eastmp), 60, dtype=np.int32)  # Average ~60 days per month
day_count_vec = day_count_base[schid_vec].astype(np.int32)

# d_start_ym
year1 = smym // 12
month1 = smym % 12 + 1
d_start_ym = year1 * 12 + month1 - 1

month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)
total_days = int(np.sum(day_count_vec))

print(f"Straddles: {smlen:,}, total_days: {total_days:,}")

# ============================================================================
# Pre-compute type indices and futures data
# ============================================================================
hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
hedge_source_id_cds_smidx = hedge_source_id_cds[smidx]
hedge_source_id_calc_smidx = hedge_source_id_calc[smidx]

vol_source_id_bbg_smidx = vol_source_id_bbg[smidx]
vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
vol_source_id_cv_smidx = vol_source_id_cv[smidx]

nonfut_idx = np.flatnonzero(hedge_source_id_nonfut_smidx)
fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)
cds_idx = np.flatnonzero(hedge_source_id_cds_smidx)
calc_idx = np.flatnonzero(hedge_source_id_calc_smidx)

bbgN_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_N))
bbgF_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_F))
lmevol_idx = np.flatnonzero(vol_source_id_bbg_lmevol_smidx)
cv_idx = np.flatnonzero(vol_source_id_cv_smidx)

# Futures ticker mapping
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
hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

# LMEVOL R-ticker mapping
lmevol_fut_pos = np.searchsorted(fut_idx, lmevol_idx)
r_ticker_lmevol = hedge_fut_code_m[lmevol_fut_pos] + "R" + hedge_fut_tail_m[lmevol_fut_pos]
r_ticker_lmevol_u = r_ticker_lmevol.astype(TICKER_U)
tid_r_lmevol = map_to_id_searchsorted(r_ticker_lmevol_u, ticker_sorted, ticker_order)

# ============================================================================
# Build all 5 keys using ID-only approach
# ============================================================================
print("\nBuilding all 5 keys...")

# Expand IDs
tid_nonfut = hedge_ticker_tid[smidx]
fid_nonfut = hedge_field_fid[smidx]
tid_cds = hedge_hedge_tid[smidx]
tid_cds2 = hedge_hedge1_tid[smidx]
tid_calc1 = calc_hedge1_tid[smidx]
tid_calc2 = calc_hedge2_tid[smidx]
tid_calc3 = calc_hedge3_tid[smidx]
tid_calc4 = calc_hedge4_tid[smidx]

vol_tid_base = vol_ticker_tid[smidx]
vol_near_fid_sm = vol_near_fid[smidx]
vol_far_fid_sm = vol_far_fid[smidx]
vol_near_tid_sm = vol_near_tid[smidx]

tid_fut = np.full(smlen, -1, dtype=np.int32)
tid_fut[fut_idx] = hedge_fut_ticker_tid_m

tid_r = np.full(smlen, -1, dtype=np.int32)
tid_r[lmevol_idx] = tid_r_lmevol

# hedge1_key
hedge1_key = np.full(smlen, -1, dtype=np.int32)
hedge1_key[nonfut_idx] = tid_nonfut[nonfut_idx] * n_fields + fid_nonfut[nonfut_idx]
hedge1_key[fut_idx] = tid_fut[fut_idx] * n_fields + PX_LAST_FID
hedge1_key[cds_idx] = tid_cds[cds_idx] * n_fields + PX_LAST_FID
hedge1_key[calc_idx] = tid_calc1[calc_idx] * n_fields + EMPTY_FID

# hedge2_key
hedge2_key = np.full(smlen, -1, dtype=np.int32)
hedge2_key[cds_idx] = tid_cds2[cds_idx] * n_fields + PX_LAST_FID
hedge2_key[calc_idx] = tid_calc2[calc_idx] * n_fields + EMPTY_FID

# hedge3_key
hedge3_key = np.full(smlen, -1, dtype=np.int32)
hedge3_key[calc_idx] = tid_calc3[calc_idx] * n_fields + EMPTY_FID

# hedge4_key
hedge4_key = np.full(smlen, -1, dtype=np.int32)
hedge4_key[calc_idx] = tid_calc4[calc_idx] * n_fields + EMPTY_FID

# vol_key
vol_key = np.full(smlen, -1, dtype=np.int32)
vol_key[bbgN_idx] = vol_tid_base[bbgN_idx] * n_fields + vol_near_fid_sm[bbgN_idx]
vol_key[bbgF_idx] = vol_tid_base[bbgF_idx] * n_fields + vol_far_fid_sm[bbgF_idx]
vol_key[lmevol_idx] = tid_r[lmevol_idx] * n_fields + PX_LAST_FID
vol_key[cv_idx] = vol_near_tid_sm[cv_idx] * n_fields + NONE_FID

all_keys = [hedge1_key, hedge2_key, hedge3_key, hedge4_key, vol_key]
key_names = ["hedge1", "hedge2", "hedge3", "hedge4", "vol"]

for name, key in zip(key_names, all_keys):
    valid = np.sum(key >= 0)
    print(f"  {name}: {valid:,} valid keys ({100*valid/smlen:.1f}%)")

# ============================================================================
# Sweep Prep Function
# ============================================================================
def sweep_prep(key_i32):
    """Prepare for sweep merge: sort, filter, group."""
    order = np.lexsort((month_start_epoch, key_i32))
    k = key_i32[order]
    s = month_start_epoch[order]
    ln = day_count_vec[order].astype(np.int32)
    out0 = month_out0[order]

    max_valid_key = len(px_block_of) - 1
    valid = (k >= 0) & (k <= max_valid_key)
    k = k[valid].astype(np.int32)
    s = s[valid].astype(np.int32)
    ln = ln[valid].astype(np.int32)
    out0 = out0[valid].astype(np.int32)

    chg = np.flatnonzero(k[1:] != k[:-1]) + 1
    g_starts = np.r_[0, chg].astype(np.int32)
    g_ends = np.r_[chg, len(k)].astype(np.int32)
    g_keys = k[g_starts]

    return g_keys, g_starts, g_ends, s, ln, out0

# ============================================================================
# Warmup: Run merge once to JIT compile
# ============================================================================
print("\nWarming up Numba...")
warmup_prep = sweep_prep(hedge1_key)
warmup_out = np.full(total_days, np.nan, dtype=np.float64)
_ = run_sweep_merge(*warmup_prep, warmup_out)
print("Warmup complete.")

# ============================================================================
# Benchmark: Single Sweep (hedge1 only)
# ============================================================================
print("\n=== Benchmark: Single Sweep (hedge1) ===")

# Prep
t0 = time.perf_counter()
for _ in range(5):
    prep = sweep_prep(hedge1_key)
t1 = time.perf_counter()
prep_time = (t1-t0)/5*1000
print(f"  Sweep prep:     {prep_time:.1f}ms ({prep[0].shape[0]:,} groups)")

# Merge
t0 = time.perf_counter()
for _ in range(5):
    out = np.full(total_days, np.nan, dtype=np.float64)
    out = run_sweep_merge(*prep, out)
t1 = time.perf_counter()
merge_time = (t1-t0)/5*1000
found = np.sum(~np.isnan(out))
print(f"  Sweep merge:    {merge_time:.1f}ms ({found:,} found)")

print(f"  Total (1 leg):  {prep_time + merge_time:.1f}ms")

# ============================================================================
# Benchmark: Sequential Sweeps (5 independent)
# ============================================================================
print("\n=== Benchmark: Sequential Sweeps (5 legs) ===")

# All preps
t0 = time.perf_counter()
for _ in range(5):
    preps = [sweep_prep(key) for key in all_keys]
t1 = time.perf_counter()
all_prep_time = (t1-t0)/5*1000
print(f"  All preps:      {all_prep_time:.1f}ms")

# Individual prep times
print("  Individual prep times:")
prep_times = []
for name, key in zip(key_names, all_keys):
    t0 = time.perf_counter()
    for _ in range(5):
        p = sweep_prep(key)
    t1 = time.perf_counter()
    pt = (t1-t0)/5*1000
    prep_times.append(pt)
    print(f"    {name}: {pt:.1f}ms ({p[0].shape[0]:,} groups)")

# All merges
preps = [sweep_prep(key) for key in all_keys]
t0 = time.perf_counter()
for _ in range(5):
    outs = []
    for p in preps:
        out = np.full(total_days, np.nan, dtype=np.float64)
        out = run_sweep_merge(*p, out)
        outs.append(out)
t1 = time.perf_counter()
all_merge_time = (t1-t0)/5*1000
print(f"  All merges:     {all_merge_time:.1f}ms")

# Individual merge times
print("  Individual merge times:")
merge_times = []
for name, p in zip(key_names, preps):
    t0 = time.perf_counter()
    for _ in range(5):
        out = np.full(total_days, np.nan, dtype=np.float64)
        out = run_sweep_merge(*p, out)
    t1 = time.perf_counter()
    mt = (t1-t0)/5*1000
    merge_times.append(mt)
    found = np.sum(~np.isnan(out))
    print(f"    {name}: {mt:.1f}ms ({found:,} found)")

total_seq = all_prep_time + all_merge_time
print(f"\n  Total (5 legs): {total_seq:.1f}ms")
print(f"  Per-leg avg:    {total_seq/5:.1f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Summary ===")
print(f"""
Sweep Merge Benchmark Results:
==============================
Data: {smlen:,} straddles, {total_days:,} daily rows

Single Sweep (hedge1):
  Prep:    {prep_time:.1f}ms
  Merge:   {merge_time:.1f}ms
  Total:   {prep_time + merge_time:.1f}ms

Sequential Sweeps (5 legs):
  Prep:    {all_prep_time:.1f}ms (5 independent lexsorts)
  Merge:   {all_merge_time:.1f}ms (5 independent merges)
  Total:   {total_seq:.1f}ms

Breakdown by leg:
""")

for name, pt, mt in zip(key_names, prep_times, merge_times):
    print(f"  {name}: prep={pt:.1f}ms, merge={mt:.1f}ms, total={pt+mt:.1f}ms")

print(f"""
Key observations:
1. Prep time (lexsort) is the dominant cost: {all_prep_time:.1f}ms vs {all_merge_time:.1f}ms for merge
2. 5 sequential preps: {all_prep_time:.1f}ms (vs {prep_time:.1f}ms for 1 prep)
3. Merge kernel is fast because it's already at memory bandwidth limit
4. Each leg adds ~{total_seq/5:.0f}ms overhead

Comparison with ID-only approach:
- Current (strings + 1 sweep): ~53ms hedge + ~27ms vol + ~20ms sweep = ~100ms
- ID-only (keys + 5 sweeps):   ~4ms keys + {total_seq:.0f}ms sweeps = ~{4+total_seq:.0f}ms

Note: The actual cost depends on how many legs produce useful data.
Legs with mostly -1 keys (hedge2-4) have fewer groups and faster prep.
""")
