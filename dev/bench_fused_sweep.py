"""Benchmark fused sweep vs sequential sweeps.

Compares:
1. Sequential sweeps: 5 independent lexsorts + 5 merge calls
2. Fused sweep: 1 lexsort on concatenated intervals + 1 merge call with leg_id
"""
import numpy as np
import time
from numba import njit, prange

# ============================================================================
# Load price data
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
# Numba kernels
# ============================================================================

@njit(parallel=True, cache=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
    """Standard merge kernel - writes to 1D output."""
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
def _merge_per_key_fused(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s, m_leg_s,
                         px_block_of, px_starts, px_ends, px_date, px_value, out2d):
    """Fused merge kernel - writes to 2D output based on leg_id."""
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
            leg = m_leg_s[si]  # Which output column (0-4)
            end = start + length
            while pi0 < pe and px_date[pi0] < start:
                pi0 += 1
            pi = pi0
            while pi < pe:
                d = px_date[pi]
                if d >= end:
                    break
                out2d[leg, out0 + (d - start)] = px_value[pi]
                pi += 1
    return out2d


# ============================================================================
# Load straddle data (simplified from bench_multi_sweep.py)
# ============================================================================
print("\nLoading straddle data...")

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

import yaml
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

day_count_base = np.full(len(eastmp), 60, dtype=np.int32)
day_count_vec = day_count_base[schid_vec].astype(np.int32)

year1 = smym // 12
month1 = smym % 12 + 1
d_start_ym = year1 * 12 + month1 - 1

month_start_epoch = _ym_epoch[d_start_ym - _ym_base].astype(np.int32)
month_out0 = (np.cumsum(day_count_vec) - day_count_vec).astype(np.int32)
month_len = day_count_vec.astype(np.int32)
total_days = int(np.sum(day_count_vec))

print(f"Straddles: {smlen:,}, total_days: {total_days:,}")

# ============================================================================
# Pre-compute type indices and build keys
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

# Debug assertion: verify LMEVOL ⊆ FUT invariant
DEBUG = True
if DEBUG:
    ok = (lmevol_fut_pos < len(fut_idx)) & (fut_idx[lmevol_fut_pos] == lmevol_idx)
    assert np.all(ok), f"LMEVOL ⊆ FUT invariant broken: {np.sum(~ok)} mismatches"

r_ticker_lmevol = hedge_fut_code_m[lmevol_fut_pos] + "R" + hedge_fut_tail_m[lmevol_fut_pos]
r_ticker_lmevol_u = r_ticker_lmevol.astype(TICKER_U)
tid_r_lmevol = map_to_id_searchsorted(r_ticker_lmevol_u, ticker_sorted, ticker_order)

# Build all 5 keys
print("\nBuilding all 5 keys...")

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
# Sweep Prep Functions
# ============================================================================
max_valid_key = len(px_block_of) - 1

def sweep_prep_sequential(key_i32):
    """Prepare for sequential sweep: FILTER FIRST, then sort (correct for sparse keys)."""
    # Filter first - this is the key optimization for sparse legs like hedge2-4
    v = (key_i32 >= 0) & (key_i32 <= max_valid_key)
    k = key_i32[v].astype(np.int32)
    s = month_start_epoch[v].astype(np.int32)
    ln = month_len[v].astype(np.int32)
    out0 = month_out0[v].astype(np.int32)

    # Sort only the filtered subset
    order = np.lexsort((s, k))
    k, s, ln, out0 = k[order], s[order], ln[order], out0[order]

    chg = np.flatnonzero(k[1:] != k[:-1]) + 1
    g_starts = np.r_[0, chg].astype(np.int32)
    g_ends = np.r_[chg, len(k)].astype(np.int32)
    g_keys = k[g_starts]

    return g_keys, g_starts, g_ends, s, ln, out0


def sweep_prep_fused(keys_list):
    """Prepare for fused sweep: concatenate all keys with leg_id, one sort."""
    # Filter and concatenate all keys
    Ks, Ss, Ls, Os, legs = [], [], [], [], []
    for leg_id, key in enumerate(keys_list):
        v = (key >= 0) & (key <= max_valid_key)
        Ks.append(key[v])
        Ss.append(month_start_epoch[v])
        Ls.append(month_len[v])
        Os.append(month_out0[v])
        legs.append(np.full(np.sum(v), leg_id, dtype=np.int8))

    K = np.concatenate(Ks).astype(np.int32)
    S = np.concatenate(Ss).astype(np.int32)
    L = np.concatenate(Ls).astype(np.int32)
    O = np.concatenate(Os).astype(np.int32)
    G = np.concatenate(legs).astype(np.int8)

    # Single lexsort on combined data
    order = np.lexsort((S, K))
    K, S, L, O, G = K[order], S[order], L[order], O[order], G[order]

    # Group by K
    chg = np.flatnonzero(K[1:] != K[:-1]) + 1
    g_starts = np.r_[0, chg].astype(np.int32)
    g_ends = np.r_[chg, len(K)].astype(np.int32)
    g_keys = K[g_starts]

    return g_keys, g_starts, g_ends, S, L, O, G


# ============================================================================
# Warmup: Run both kernels once to JIT compile
# ============================================================================
print("\nWarming up Numba kernels...")

# Warmup sequential kernel
prep_seq = sweep_prep_sequential(hedge1_key)
out1d = np.full(total_days, np.nan, dtype=np.float64)
_ = _merge_per_key(prep_seq[0], prep_seq[1], prep_seq[2], prep_seq[3], prep_seq[4], prep_seq[5],
                   px_block_of, px_starts, px_ends, px_date, px_value, out1d)

# Warmup fused kernel
prep_fused = sweep_prep_fused(all_keys)
out2d = np.full((5, total_days), np.nan, dtype=np.float64)
_ = _merge_per_key_fused(prep_fused[0], prep_fused[1], prep_fused[2],
                         prep_fused[3], prep_fused[4], prep_fused[5], prep_fused[6],
                         px_block_of, px_starts, px_ends, px_date, px_value, out2d)

print("Warmup complete.")

# ============================================================================
# Benchmark: Sequential Sweeps (5 independent)
# ============================================================================
print("\n=== Benchmark: Sequential Sweeps (5 independent) ===")

# Prep timing
t0 = time.perf_counter()
for _ in range(5):
    preps = [sweep_prep_sequential(key) for key in all_keys]
t1 = time.perf_counter()
seq_prep_time = (t1-t0)/5*1000

# Individual prep breakdown
prep_times = []
for name, key in zip(key_names, all_keys):
    t0 = time.perf_counter()
    for _ in range(5):
        p = sweep_prep_sequential(key)
    t1 = time.perf_counter()
    prep_times.append((t1-t0)/5*1000)

print(f"  Prep time (5 lexsorts): {seq_prep_time:.1f}ms")
for name, pt in zip(key_names, prep_times):
    print(f"    {name}: {pt:.1f}ms")

# Merge timing
preps = [sweep_prep_sequential(key) for key in all_keys]
t0 = time.perf_counter()
for _ in range(5):
    outs = []
    for p in preps:
        out = np.full(total_days, np.nan, dtype=np.float64)
        out = _merge_per_key(p[0], p[1], p[2], p[3], p[4], p[5],
                             px_block_of, px_starts, px_ends, px_date, px_value, out)
        outs.append(out)
t1 = time.perf_counter()
seq_merge_time = (t1-t0)/5*1000

# Individual merge breakdown
merge_times = []
seq_founds = []
for name, p in zip(key_names, preps):
    t0 = time.perf_counter()
    for _ in range(5):
        out = np.full(total_days, np.nan, dtype=np.float64)
        out = _merge_per_key(p[0], p[1], p[2], p[3], p[4], p[5],
                             px_block_of, px_starts, px_ends, px_date, px_value, out)
    t1 = time.perf_counter()
    merge_times.append((t1-t0)/5*1000)
    seq_founds.append(np.sum(~np.isnan(out)))

print(f"  Merge time (5 merges): {seq_merge_time:.1f}ms")
for name, mt, found in zip(key_names, merge_times, seq_founds):
    print(f"    {name}: {mt:.1f}ms ({found:,} found)")

seq_total = seq_prep_time + seq_merge_time
print(f"  TOTAL (sequential): {seq_total:.1f}ms")

# ============================================================================
# Benchmark: Fused Sweep (1 unified)
# ============================================================================
print("\n=== Benchmark: Fused Sweep (1 unified) ===")

# Prep timing
t0 = time.perf_counter()
for _ in range(5):
    prep = sweep_prep_fused(all_keys)
t1 = time.perf_counter()
fused_prep_time = (t1-t0)/5*1000

n_intervals = len(prep[3])
n_groups = len(prep[0])
print(f"  Prep time (1 lexsort): {fused_prep_time:.1f}ms")
print(f"    Total intervals: {n_intervals:,}")
print(f"    Total groups: {n_groups:,}")

# Merge timing
prep = sweep_prep_fused(all_keys)
t0 = time.perf_counter()
for _ in range(5):
    out2d = np.full((5, total_days), np.nan, dtype=np.float64)
    out2d = _merge_per_key_fused(prep[0], prep[1], prep[2],
                                  prep[3], prep[4], prep[5], prep[6],
                                  px_block_of, px_starts, px_ends, px_date, px_value, out2d)
t1 = time.perf_counter()
fused_merge_time = (t1-t0)/5*1000

# Count found per leg
fused_founds = []
for leg in range(5):
    fused_founds.append(np.sum(~np.isnan(out2d[leg])))

print(f"  Merge time (1 merge): {fused_merge_time:.1f}ms")
for name, found in zip(key_names, fused_founds):
    print(f"    {name}: {found:,} found")

fused_total = fused_prep_time + fused_merge_time
print(f"  TOTAL (fused): {fused_total:.1f}ms")

# ============================================================================
# Verify correctness
# ============================================================================
print("\n=== Verify Correctness ===")

# Run both approaches
preps = [sweep_prep_sequential(key) for key in all_keys]
seq_outs = []
for p in preps:
    out = np.full(total_days, np.nan, dtype=np.float64)
    out = _merge_per_key(p[0], p[1], p[2], p[3], p[4], p[5],
                         px_block_of, px_starts, px_ends, px_date, px_value, out)
    seq_outs.append(out)

prep = sweep_prep_fused(all_keys)
out2d = np.full((5, total_days), np.nan, dtype=np.float64)
out2d = _merge_per_key_fused(prep[0], prep[1], prep[2],
                              prep[3], prep[4], prep[5], prep[6],
                              px_block_of, px_starts, px_ends, px_date, px_value, out2d)

# Compare
all_match = True
for leg, name in enumerate(key_names):
    seq_out = seq_outs[leg]
    fused_out = out2d[leg]

    # Compare non-NaN values
    seq_valid = ~np.isnan(seq_out)
    fused_valid = ~np.isnan(fused_out)

    if not np.array_equal(seq_valid, fused_valid):
        print(f"  {name}: MISMATCH in valid positions!")
        all_match = False
    elif not np.allclose(seq_out[seq_valid], fused_out[fused_valid], equal_nan=True):
        print(f"  {name}: MISMATCH in values!")
        all_match = False
    else:
        print(f"  {name}: ✓ matches")

if all_match:
    print("  All legs match between sequential and fused approaches!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("SUMMARY: Fused vs Sequential Sweeps")
print("="*60)

speedup = seq_total / fused_total
savings = seq_total - fused_total

print(f"""
Data: {smlen:,} straddles, {total_days:,} daily rows

Sequential (5 independent):
  Prep (5 lexsorts):  {seq_prep_time:.1f}ms
  Merge (5 kernels):  {seq_merge_time:.1f}ms
  TOTAL:              {seq_total:.1f}ms

Fused (1 unified):
  Prep (1 lexsort):   {fused_prep_time:.1f}ms
  Merge (1 kernel):   {fused_merge_time:.1f}ms
  TOTAL:              {fused_total:.1f}ms

Speedup: {speedup:.2f}x
Savings: {savings:.1f}ms

Breakdown:
  Lexsort savings:    {seq_prep_time - fused_prep_time:.1f}ms ({seq_prep_time:.1f}ms -> {fused_prep_time:.1f}ms)
  Merge savings:      {seq_merge_time - fused_merge_time:.1f}ms ({seq_merge_time:.1f}ms -> {fused_merge_time:.1f}ms)

Memory:
  Sequential: 5 x {total_days*8/1e6:.0f}MB = {5*total_days*8/1e6:.0f}MB output
  Fused:      1 x {5*total_days*8/1e6:.0f}MB = {5*total_days*8/1e6:.0f}MB output (2D array)
""")

print("Conclusion:")
if savings > 20:
    print(f"  Fused approach saves {savings:.0f}ms - WORTH implementing.")
elif savings > 5:
    print(f"  Fused approach saves {savings:.0f}ms - Marginal benefit, complexity trade-off.")
else:
    print(f"  Fused approach saves only {savings:.0f}ms - NOT worth the added complexity.")
