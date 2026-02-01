"""Benchmark ID-only pipeline approach for hedge and vol keys.

Tests the hypothesis that computing keys directly (tid*n_fields+fid) is faster
than building string arrays first.
"""
import numpy as np
import time
from pathlib import Path
import yaml

# ============================================================================
# Setup: Load data exactly as in backtest_standalone.py
# ============================================================================
print("Loading data...")

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

# ============================================================================
# Extract all asset-level arrays
# ============================================================================
# Hedge arrays
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

# Vol arrays
vol_source = np.array(list(map(lambda a: amap[a]["Vol"].get("Source", ""), anames)), dtype=nps)
vol_ticker = np.array(list(map(lambda a: amap[a]["Vol"].get("Ticker", ""), anames)), dtype=nps)
vol_near = np.array(list(map(lambda a: amap[a]["Vol"].get("Near", ""), anames)), dtype=nps)
vol_far = np.array(list(map(lambda a: amap[a]["Vol"].get("Far", ""), anames)), dtype=nps)

# Calc-type hedges
calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor
calc_hedge2 = hedge_ccy + "_fsw6m_" + hedge_tenor
calc_hedge3 = hedge_ccy + "_pva0m_" + hedge_tenor
calc_hedge4 = hedge_ccy + "_pva6m_" + hedge_tenor

# Integer codes for hedge source
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

# Integer codes for vol source
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

# ============================================================================
# Load ID dictionaries (same as backtest_standalone.py)
# ============================================================================
ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)

n_fields = len(field_arr)
ticker_to_id = {s: i for i, s in enumerate(ticker_arr)}
field_to_id = {s: i for i, s in enumerate(field_arr)}

# Sorted ticker array for searchsorted mapping
ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

# Helper for searchsorted mapping
def map_to_id_searchsorted(tickers, ticker_sorted, ticker_order):
    """Map string tickers to IDs using searchsorted."""
    idx = np.searchsorted(ticker_sorted, tickers)
    idx = np.clip(idx, 0, len(ticker_sorted) - 1)
    matched = ticker_sorted[idx] == tickers
    result = np.where(matched, ticker_order[idx], -1)
    return result.astype(np.int32)

# ============================================================================
# Asset-level ID precomputation (CURRENT)
# ============================================================================
hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker], dtype=np.int32)
hedge_hedge_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge], dtype=np.int32)
calc_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge1], dtype=np.int32)
hedge_field_fid = np.array([field_to_id.get(str(s), -1) for s in hedge_field], dtype=np.int32)

PX_LAST_FID = field_to_id.get("PX_LAST", -1)
EMPTY_FID = field_to_id.get("", -1)

# ============================================================================
# Asset-level ID precomputation (NEW - for hedge2-4 and vol)
# ============================================================================
# Hedge IDs for hedge2-4
hedge_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge1], dtype=np.int32)
calc_hedge2_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge2], dtype=np.int32)
calc_hedge3_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge3], dtype=np.int32)
calc_hedge4_tid = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge4], dtype=np.int32)

# Vol IDs - dual mapping for near/far
vol_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_ticker], dtype=np.int32)
vol_near_fid = np.array([field_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)  # BBG field
vol_far_fid = np.array([field_to_id.get(str(s), -1) for s in vol_far], dtype=np.int32)    # BBG field
vol_near_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32) # CV ticker

NONE_FID = field_to_id.get("none", -1)

# ============================================================================
# Build straddle indices
# ============================================================================
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

# Schedule parsing for ntrc_id_vec
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

print(f"Data loaded: smlen={smlen:,}, n_assets={len(anames)}, n_fields={n_fields}")

# ============================================================================
# Pre-compute type masks
# ============================================================================
hedge_source_id_nonfut_smidx = hedge_source_id_nonfut[smidx]
hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
hedge_source_id_cds_smidx = hedge_source_id_cds[smidx]
hedge_source_id_calc_smidx = hedge_source_id_calc[smidx]

vol_source_id_bbg_smidx = vol_source_id_bbg[smidx]
vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
vol_source_id_cv_smidx = vol_source_id_cv[smidx]

# Type indices
nonfut_idx = np.flatnonzero(hedge_source_id_nonfut_smidx)
fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)
cds_idx = np.flatnonzero(hedge_source_id_cds_smidx)
calc_idx = np.flatnonzero(hedge_source_id_calc_smidx)

bbgN_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_N))
bbgF_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_F))
lmevol_idx = np.flatnonzero(vol_source_id_bbg_lmevol_smidx)
cv_idx = np.flatnonzero(vol_source_id_cv_smidx)

# ============================================================================
# Pre-compute futures ticker components (needed for both hedge1 and LMEVOL)
# ============================================================================
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

# Map futures tickers to IDs
hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

print(f"\nType indices: nonfut={len(nonfut_idx):,}, fut={len(fut_idx):,}, cds={len(cds_idx):,}, calc={len(calc_idx):,}")
print(f"Vol indices: bbgN={len(bbgN_idx):,}, bbgF={len(bbgF_idx):,}, lmevol={len(lmevol_idx):,}, cv={len(cv_idx):,}")

# ============================================================================
# Benchmark: Current approach (build strings, then map to IDs)
# ============================================================================
print("\n=== Current Approach: Build Strings, Then Map ===")

# Full-size arrays (for LMEVOL)
hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m

def current_approach():
    """Current approach: build string arrays, then map to IDs."""
    # Hedge strings (type-dispatch)
    hedge1t_vec = np.empty(smlen, dtype=nps)
    hedge1t_vec.fill("")
    hedge1t_vec[nonfut_idx] = hedge_ticker[smidx[nonfut_idx]]
    hedge1t_vec[fut_idx] = hedge_fut_ticker_m
    hedge1t_vec[cds_idx] = hedge_hedge[smidx[cds_idx]]
    hedge1t_vec[calc_idx] = calc_hedge1[smidx[calc_idx]]

    hedge1f_vec = np.empty(smlen, dtype=nps)
    hedge1f_vec.fill("")
    hedge1f_vec[nonfut_idx] = hedge_field[smidx[nonfut_idx]]
    hedge1f_vec[fut_idx] = "PX_LAST"
    hedge1f_vec[cds_idx] = "PX_LAST"

    # Map to IDs (simplified - actual code uses cond_hedge)
    month_tid = np.full(smlen, -1, dtype=np.int32)
    month_tid[nonfut_idx] = hedge_ticker_tid[smidx[nonfut_idx]]
    month_tid[fut_idx] = hedge_fut_ticker_tid_m
    month_tid[cds_idx] = hedge_hedge_tid[smidx[cds_idx]]
    month_tid[calc_idx] = calc_hedge1_tid[smidx[calc_idx]]

    month_fid = np.full(smlen, -1, dtype=np.int32)
    month_fid[nonfut_idx] = hedge_field_fid[smidx[nonfut_idx]]
    month_fid[fut_idx] = PX_LAST_FID
    month_fid[cds_idx] = PX_LAST_FID
    month_fid[calc_idx] = EMPTY_FID

    month_key = (month_tid * np.int32(n_fields) + month_fid).astype(np.int32)
    return month_key, hedge1t_vec, hedge1f_vec

t0 = time.perf_counter()
for _ in range(5):
    month_key_cur, h1t, h1f = current_approach()
t1 = time.perf_counter()
print(f"  Hedge1 (strings + ID mapping): {(t1-t0)/5*1000:.1f}ms")

# ============================================================================
# Benchmark: ID-only approach (compute keys directly)
# ============================================================================
print("\n=== ID-Only Approach: Compute Keys Directly ===")

def id_only_hedge1():
    """ID-only approach for hedge1: compute key directly."""
    # Expand IDs to straddle level
    tid_nonfut = hedge_ticker_tid[smidx]
    fid_nonfut = hedge_field_fid[smidx]
    tid_cds = hedge_hedge_tid[smidx]
    tid_calc1 = calc_hedge1_tid[smidx]

    # Futures tid (pre-mapped)
    tid_fut = np.full(smlen, -1, dtype=np.int32)
    tid_fut[fut_idx] = hedge_fut_ticker_tid_m

    # Build key directly
    hedge1_key = np.full(smlen, -1, dtype=np.int32)
    hedge1_key[nonfut_idx] = tid_nonfut[nonfut_idx] * n_fields + fid_nonfut[nonfut_idx]
    hedge1_key[fut_idx] = tid_fut[fut_idx] * n_fields + PX_LAST_FID
    hedge1_key[cds_idx] = tid_cds[cds_idx] * n_fields + PX_LAST_FID
    hedge1_key[calc_idx] = tid_calc1[calc_idx] * n_fields + EMPTY_FID

    return hedge1_key

t0 = time.perf_counter()
for _ in range(5):
    hedge1_key_idonly = id_only_hedge1()
t1 = time.perf_counter()
print(f"  Hedge1 (ID-only):              {(t1-t0)/5*1000:.1f}ms")

# Verify
if np.array_equal(month_key_cur, hedge1_key_idonly):
    print("  ✓ hedge1_key matches")
else:
    mismatch = np.sum(month_key_cur != hedge1_key_idonly)
    print(f"  ✗ hedge1_key: {mismatch:,} mismatches")

# ============================================================================
# Benchmark: All 5 hedge keys (ID-only)
# ============================================================================
print("\n=== All 5 Hedge Keys (ID-Only) ===")

def id_only_all_hedge_keys():
    """Compute all 5 hedge keys directly."""
    # Expand IDs to straddle level
    tid_nonfut = hedge_ticker_tid[smidx]
    fid_nonfut = hedge_field_fid[smidx]
    tid_cds = hedge_hedge_tid[smidx]
    tid_cds2 = hedge_hedge1_tid[smidx]
    tid_calc1 = calc_hedge1_tid[smidx]
    tid_calc2 = calc_hedge2_tid[smidx]
    tid_calc3 = calc_hedge3_tid[smidx]
    tid_calc4 = calc_hedge4_tid[smidx]

    # Futures tid
    tid_fut = np.full(smlen, -1, dtype=np.int32)
    tid_fut[fut_idx] = hedge_fut_ticker_tid_m

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

    return hedge1_key, hedge2_key, hedge3_key, hedge4_key

t0 = time.perf_counter()
for _ in range(5):
    h1k, h2k, h3k, h4k = id_only_all_hedge_keys()
t1 = time.perf_counter()
print(f"  All 4 hedge keys:              {(t1-t0)/5*1000:.1f}ms")

# ============================================================================
# Benchmark: Vol key (ID-only)
# ============================================================================
print("\n=== Vol Key (ID-Only) ===")

# Build LMEVOL R-ticker IDs
# lmevol_idx rows need mapping to fut_idx positions
lmevol_fut_pos = np.searchsorted(fut_idx, lmevol_idx)
r_ticker_lmevol = hedge_fut_code_m[lmevol_fut_pos] + "R" + hedge_fut_tail_m[lmevol_fut_pos]
r_ticker_lmevol_u = r_ticker_lmevol.astype(TICKER_U)
tid_r_lmevol = map_to_id_searchsorted(r_ticker_lmevol_u, ticker_sorted, ticker_order)

print(f"  LMEVOL R-ticker mapping: {len(lmevol_idx):,} rows -> {np.sum(tid_r_lmevol >= 0):,} found")

def id_only_vol_key():
    """Compute vol key directly."""
    # Expand vol IDs to straddle level
    vol_tid_base = vol_ticker_tid[smidx]
    vol_near_fid_sm = vol_near_fid[smidx]
    vol_far_fid_sm = vol_far_fid[smidx]
    vol_near_tid_sm = vol_near_tid[smidx]

    # LMEVOL R-ticker tid
    tid_r = np.full(smlen, -1, dtype=np.int32)
    tid_r[lmevol_idx] = tid_r_lmevol

    # Build vol_key
    vol_key = np.full(smlen, -1, dtype=np.int32)
    vol_key[bbgN_idx] = vol_tid_base[bbgN_idx] * n_fields + vol_near_fid_sm[bbgN_idx]
    vol_key[bbgF_idx] = vol_tid_base[bbgF_idx] * n_fields + vol_far_fid_sm[bbgF_idx]
    vol_key[lmevol_idx] = tid_r[lmevol_idx] * n_fields + PX_LAST_FID
    vol_key[cv_idx] = vol_near_tid_sm[cv_idx] * n_fields + NONE_FID

    return vol_key

t0 = time.perf_counter()
for _ in range(5):
    vol_key = id_only_vol_key()
t1 = time.perf_counter()
print(f"  Vol key:                       {(t1-t0)/5*1000:.1f}ms")

# Check vol key validity
valid_vol_keys = np.sum(vol_key >= 0)
print(f"  Valid vol keys: {valid_vol_keys:,} / {smlen:,} ({100*valid_vol_keys/smlen:.1f}%)")

# ============================================================================
# Benchmark: All 5 keys combined
# ============================================================================
print("\n=== All 5 Keys Combined ===")

def id_only_all_keys():
    """Compute all 5 keys (hedge1-4 + vol) directly."""
    # Expand IDs to straddle level
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

    # Futures tid
    tid_fut = np.full(smlen, -1, dtype=np.int32)
    tid_fut[fut_idx] = hedge_fut_ticker_tid_m

    # LMEVOL R-ticker tid
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

    return hedge1_key, hedge2_key, hedge3_key, hedge4_key, vol_key

t0 = time.perf_counter()
for _ in range(5):
    keys = id_only_all_keys()
t1 = time.perf_counter()
print(f"  All 5 keys (hedge1-4 + vol):   {(t1-t0)/5*1000:.1f}ms")

# ============================================================================
# Compare: Current strings vs ID-only
# ============================================================================
print("\n=== Comparison ===")

# Current: hedge strings + vol strings
def current_all_strings():
    """Current approach: build all strings (hedge1-4 + vol)."""
    # Hedge strings
    hedge1t_vec = np.empty(smlen, dtype=nps)
    hedge1t_vec.fill("")
    hedge1t_vec[nonfut_idx] = hedge_ticker[smidx[nonfut_idx]]
    hedge1t_vec[fut_idx] = hedge_fut_ticker_m
    hedge1t_vec[cds_idx] = hedge_hedge[smidx[cds_idx]]
    hedge1t_vec[calc_idx] = calc_hedge1[smidx[calc_idx]]

    hedge1f_vec = np.empty(smlen, dtype=nps)
    hedge1f_vec.fill("")
    hedge1f_vec[nonfut_idx] = hedge_field[smidx[nonfut_idx]]
    hedge1f_vec[fut_idx] = "PX_LAST"
    hedge1f_vec[cds_idx] = "PX_LAST"

    hedge2t_vec = np.empty(smlen, dtype=nps)
    hedge2t_vec.fill("")
    hedge2t_vec[cds_idx] = hedge_hedge1[smidx[cds_idx]]
    hedge2t_vec[calc_idx] = calc_hedge2[smidx[calc_idx]]

    hedge2f_vec = np.empty(smlen, dtype=nps)
    hedge2f_vec.fill("")
    hedge2f_vec[cds_idx] = "PX_LAST"

    hedge3t_vec = np.empty(smlen, dtype=nps)
    hedge3t_vec.fill("")
    hedge3t_vec[calc_idx] = calc_hedge3[smidx[calc_idx]]

    hedge3f_vec = np.full(smlen, "", dtype="U")

    hedge4t_vec = np.empty(smlen, dtype=nps)
    hedge4t_vec.fill("")
    hedge4t_vec[calc_idx] = calc_hedge4[smidx[calc_idx]]

    hedge4f_vec = hedge3f_vec

    # Vol strings (type-dispatch)
    volt_vec = np.empty(smlen, dtype=nps)
    volt_vec.fill("")
    volt_vec[bbgN_idx] = vol_ticker[smidx[bbgN_idx]]
    volt_vec[bbgF_idx] = vol_ticker[smidx[bbgF_idx]]
    volt_vec[lmevol_idx] = hedge_fut_code[smidx[lmevol_idx]] + "R" + hedge_fut_tail_smidx[lmevol_idx]
    volt_vec[cv_idx] = vol_near[smidx[cv_idx]]

    volf_vec = np.empty(smlen, dtype=nps)
    volf_vec.fill("")
    volf_vec[bbgN_idx] = vol_near[smidx[bbgN_idx]]
    volf_vec[bbgF_idx] = vol_far[smidx[bbgF_idx]]
    volf_vec[lmevol_idx] = "PX_LAST"
    volf_vec[cv_idx] = "none"

    return (hedge1t_vec, hedge1f_vec, hedge2t_vec, hedge2f_vec,
            hedge3t_vec, hedge3f_vec, hedge4t_vec, hedge4f_vec,
            volt_vec, volf_vec)

t0 = time.perf_counter()
for _ in range(5):
    strings = current_all_strings()
t1 = time.perf_counter()
time_strings = (t1-t0)/5*1000
print(f"  Current (all strings):         {time_strings:.1f}ms")

t0 = time.perf_counter()
for _ in range(5):
    keys = id_only_all_keys()
t1 = time.perf_counter()
time_idonly = (t1-t0)/5*1000
print(f"  ID-only (all keys):            {time_idonly:.1f}ms")

print(f"\n  Speedup: {time_strings/time_idonly:.2f}x")
print(f"  Savings: {time_strings - time_idonly:.1f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Summary ===")
print(f"""
ID-Only Pipeline Benchmark Results:
===================================
Data: {smlen:,} straddles, {len(anames)} assets

Key Computation Times:
- All 4 hedge keys:        ~{(t1-t0)/5*1000:.1f}ms
- Vol key:                 ~{time_idonly:.1f}ms (includes {len(lmevol_idx):,} LMEVOL mappings)

Comparison:
- Current (strings):       {time_strings:.1f}ms (builds 10 string arrays)
- ID-only (keys):          {time_idonly:.1f}ms (builds 5 int32 arrays)
- Savings:                 {time_strings - time_idonly:.1f}ms ({100*(time_strings-time_idonly)/time_strings:.0f}% reduction)

This savings is from eliminating string materialization.
Additional sweep merge costs must be added for hedge2-4 and vol.
""")
