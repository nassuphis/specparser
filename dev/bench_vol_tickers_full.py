"""Benchmark full vol tickers section to understand where ~27ms goes."""
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

# Extract hedge asset-level arrays (needed for LMEVOL R-ticker construction)
hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)

# Extract vol asset-level arrays
vol_source = np.array(list(map(lambda a: amap[a]["Vol"].get("Source", ""), anames)), dtype=nps)
vol_ticker = np.array(list(map(lambda a: amap[a]["Vol"].get("Ticker", ""), anames)), dtype=nps)
vol_near = np.array(list(map(lambda a: amap[a]["Vol"].get("Near", ""), anames)), dtype=nps)
vol_far = np.array(list(map(lambda a: amap[a]["Vol"].get("Far", ""), anames)), dtype=nps)

# Integer codes for hedge source (needed for futures components)
hedge_sources, hedge_source_id = np.unique(hedge_source, return_inverse=True)
hs2id_map = dict(zip(hedge_sources, range(len(hedge_sources))))
HEDGE_FUT = hs2id_map["fut"]
hedge_source_id_fut = hedge_source_id == HEDGE_FUT

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

# Build straddle indices
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

# Parse schedules to get ntrc_id_vec (needed for STR_N/STR_F conditions)
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

# Expand to straddle-month level
schid_vec = np.tile(np.arange(len(eastmp)), ym_len).reshape(ym_len, -1).T.flatten()
ntrc_id_vec = ntrc_ids_flat[schid_vec]

print(f"Data loaded: smlen={smlen:,}, n_assets={len(anames)}")

# ============================================================================
# Data Distribution Analysis
# ============================================================================
print("\n=== Data Distribution ===")

# Vol source distribution at asset level
print(f"Vol sources at asset level (N={len(anames)}):")
for vs in vol_sources:
    count = np.sum(vol_source == vs)
    print(f"  {vs}: {count} assets ({100*count/len(anames):.1f}%)")

# Vol source distribution at straddle level
vol_source_id_smidx = vol_source_id[smidx]
print(f"\nVol sources at straddle level (N={smlen:,}):")
for vs, vs_id in vs2id_map.items():
    mask = vol_source_id_smidx == vs_id
    count = np.sum(mask)
    print(f"  {vs}: {count:,} rows ({100*count/smlen:.1f}%)")

# Check LMEVOL / fut overlap
lmevol_assets = np.flatnonzero(vol_source_id_bbg_lmevol)
fut_assets = np.flatnonzero(hedge_source_id_fut)
lmevol_in_fut = np.isin(lmevol_assets, fut_assets)
print(f"\nLMEVOL assets: {len(lmevol_assets)}")
print(f"Fut hedge assets: {len(fut_assets)}")
print(f"LMEVOL assets with fut hedge: {np.sum(lmevol_in_fut)} ({100*np.mean(lmevol_in_fut):.1f}%)")

# Schedule type distribution for BBG assets
bbg_smidx_mask = vol_source_id_bbg[smidx]
bbg_rows = np.sum(bbg_smidx_mask)
bbg_n_rows = np.sum(bbg_smidx_mask & (ntrc_id_vec == STR_N))
bbg_f_rows = np.sum(bbg_smidx_mask & (ntrc_id_vec == STR_F))
print(f"\nBBG straddles by schedule type:")
print(f"  STR_N: {bbg_n_rows:,} ({100*bbg_n_rows/bbg_rows:.1f}%)")
print(f"  STR_F: {bbg_f_rows:,} ({100*bbg_f_rows/bbg_rows:.1f}%)")

# ============================================================================
# Pre-compute futures components (needed for LMEVOL R-ticker)
# ============================================================================
hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)

hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)

hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]

# Full-size arrays for vol section
hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m

print(f"\nFutures components computed: {len(fut_idx):,} rows")

# ============================================================================
# Benchmark: Vol tickers section breakdown
# ============================================================================
print("\n=== Vol Tickers Section Breakdown ===")

# Expand vol source conditions
vol_source_id_bbg_smidx = vol_source_id_bbg[smidx]
vol_source_id_bbg_lmevol_smidx = vol_source_id_bbg_lmevol[smidx]
vol_source_id_cv_smidx = vol_source_id_cv[smidx]

# Step 1: Build condition arrays
t0 = time.perf_counter()
for _ in range(5):
    cond_vol = [
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_N),
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_F),
        (vol_source_id_bbg_lmevol_smidx),
        (vol_source_id_cv_smidx)
    ]
t1 = time.perf_counter()
print(f"  Build condition arrays:  {(t1-t0)/5*1000:.1f}ms")

# Step 2: vol_ticker[smidx]
t0 = time.perf_counter()
for _ in range(5):
    vol_ticker_smidx = vol_ticker[smidx]
t1 = time.perf_counter()
print(f"  vol_ticker[smidx]:       {(t1-t0)/5*1000:.1f}ms")

# Step 3: vol_near[smidx]
t0 = time.perf_counter()
for _ in range(5):
    vol_near_smidx = vol_near[smidx]
t1 = time.perf_counter()
print(f"  vol_near[smidx]:         {(t1-t0)/5*1000:.1f}ms")

# Step 4: vol_far[smidx]
t0 = time.perf_counter()
for _ in range(5):
    vol_far_smidx = vol_far[smidx]
t1 = time.perf_counter()
print(f"  vol_far[smidx]:          {(t1-t0)/5*1000:.1f}ms")

# Step 5: String concat for LMEVOL R-ticker
t0 = time.perf_counter()
for _ in range(5):
    vol_tkrz_smidx = hedge_fut_code_smidx + "R" + hedge_fut_tail_smidx
t1 = time.perf_counter()
print(f"  String concat (R-ticker):{(t1-t0)/5*1000:.1f}ms")

# Step 6: np.select for volt_vec
t0 = time.perf_counter()
for _ in range(5):
    choices_volt = [vol_ticker_smidx, vol_ticker_smidx, vol_tkrz_smidx, vol_near_smidx]
    volt_vec = np.select(cond_vol, choices_volt, default="")
t1 = time.perf_counter()
print(f"  np.select volt_vec:      {(t1-t0)/5*1000:.1f}ms")

# Step 7: np.select for volf_vec
t0 = time.perf_counter()
for _ in range(5):
    choices_volf = [vol_near_smidx, vol_far_smidx, "PX_LAST", "none"]
    volf_vec = np.select(cond_vol, choices_volf, default="")
t1 = time.perf_counter()
print(f"  np.select volf_vec:      {(t1-t0)/5*1000:.1f}ms")

# Total
print("\n=== End-to-end timing ===")
times = []
for _ in range(5):
    t0 = time.perf_counter()

    # Full vol tickers section
    cond_vol = [
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_N),
        (vol_source_id_bbg_smidx) & (ntrc_id_vec == STR_F),
        (vol_source_id_bbg_lmevol_smidx),
        (vol_source_id_cv_smidx)
    ]

    vol_ticker_smidx = vol_ticker[smidx]
    vol_near_smidx = vol_near[smidx]
    vol_far_smidx = vol_far[smidx]
    vol_tkrz_smidx = hedge_fut_code_smidx + "R" + hedge_fut_tail_smidx

    choices_volt = [vol_ticker_smidx, vol_ticker_smidx, vol_tkrz_smidx, vol_near_smidx]
    volt_vec = np.select(cond_vol, choices_volt, default="")

    choices_volf = [vol_near_smidx, vol_far_smidx, "PX_LAST", "none"]
    volf_vec = np.select(cond_vol, choices_volf, default="")

    t1 = time.perf_counter()
    times.append((t1-t0)*1000)

print(f"Total vol tickers: {np.median(times):.1f}ms (median)")
print(f"Individual runs: {[f'{t:.1f}' for t in times]}")

# ============================================================================
# Compare: Type-dispatch approach (like hedge tickers optimization)
# ============================================================================
print("\n=== Type-Dispatch Approach (no np.select) ===")

# Pre-compute type indices
t0 = time.perf_counter()
for _ in range(5):
    bbgN_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_N))
    bbgF_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_F))
    lmevol_idx = np.flatnonzero(vol_source_id_bbg_lmevol_smidx)
    cv_idx = np.flatnonzero(vol_source_id_cv_smidx)
t1 = time.perf_counter()
print(f"  Compute type indices:    {(t1-t0)/5*1000:.1f}ms")
print(f"    BBG+N: {len(bbgN_idx):,}, BBG+F: {len(bbgF_idx):,}, LMEVOL: {len(lmevol_idx):,}, CV: {len(cv_idx):,}")

# Type-dispatch volt_vec
t0 = time.perf_counter()
for _ in range(5):
    volt_vec_td = np.empty(smlen, dtype=nps)
    volt_vec_td.fill("")
    volt_vec_td[bbgN_idx] = vol_ticker[smidx[bbgN_idx]]
    volt_vec_td[bbgF_idx] = vol_ticker[smidx[bbgF_idx]]
    volt_vec_td[lmevol_idx] = (hedge_fut_code[smidx[lmevol_idx]] + "R" +
                               hedge_fut_tail_smidx[lmevol_idx])  # Use pre-computed tail
    volt_vec_td[cv_idx] = vol_near[smidx[cv_idx]]
t1 = time.perf_counter()
print(f"  Type-dispatch volt_vec:  {(t1-t0)/5*1000:.1f}ms")

# Type-dispatch volf_vec
t0 = time.perf_counter()
for _ in range(5):
    volf_vec_td = np.empty(smlen, dtype=nps)
    volf_vec_td.fill("")
    volf_vec_td[bbgN_idx] = vol_near[smidx[bbgN_idx]]
    volf_vec_td[bbgF_idx] = vol_far[smidx[bbgF_idx]]
    volf_vec_td[lmevol_idx] = "PX_LAST"
    volf_vec_td[cv_idx] = "none"
t1 = time.perf_counter()
print(f"  Type-dispatch volf_vec:  {(t1-t0)/5*1000:.1f}ms")

# Verify correctness
print("\n=== Verify correctness ===")
if np.array_equal(volt_vec, volt_vec_td):
    print("✓ volt_vec matches")
else:
    mismatch = np.sum(volt_vec != volt_vec_td)
    print(f"✗ volt_vec: {mismatch:,} mismatches")

if np.array_equal(volf_vec, volf_vec_td):
    print("✓ volf_vec matches")
else:
    mismatch = np.sum(volf_vec != volf_vec_td)
    print(f"✗ volf_vec: {mismatch:,} mismatches")

# Total with type-dispatch
print("\n=== End-to-end timing (type-dispatch) ===")
times_td = []
for _ in range(5):
    t0 = time.perf_counter()

    # Type indices
    bbgN_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_N))
    bbgF_idx = np.flatnonzero(vol_source_id_bbg_smidx & (ntrc_id_vec == STR_F))
    lmevol_idx = np.flatnonzero(vol_source_id_bbg_lmevol_smidx)
    cv_idx = np.flatnonzero(vol_source_id_cv_smidx)

    # volt_vec
    volt_vec_td = np.empty(smlen, dtype=nps)
    volt_vec_td.fill("")
    volt_vec_td[bbgN_idx] = vol_ticker[smidx[bbgN_idx]]
    volt_vec_td[bbgF_idx] = vol_ticker[smidx[bbgF_idx]]
    volt_vec_td[lmevol_idx] = hedge_fut_code[smidx[lmevol_idx]] + "R" + hedge_fut_tail_smidx[lmevol_idx]
    volt_vec_td[cv_idx] = vol_near[smidx[cv_idx]]

    # volf_vec
    volf_vec_td = np.empty(smlen, dtype=nps)
    volf_vec_td.fill("")
    volf_vec_td[bbgN_idx] = vol_near[smidx[bbgN_idx]]
    volf_vec_td[bbgF_idx] = vol_far[smidx[bbgF_idx]]
    volf_vec_td[lmevol_idx] = "PX_LAST"
    volf_vec_td[cv_idx] = "none"

    t1 = time.perf_counter()
    times_td.append((t1-t0)*1000)

print(f"Total vol tickers (type-dispatch): {np.median(times_td):.1f}ms (median)")
print(f"Individual runs: {[f'{t:.1f}' for t in times_td]}")

speedup = np.median(times) / np.median(times_td)
print(f"\nSpeedup: {speedup:.2f}x ({np.median(times):.1f}ms -> {np.median(times_td):.1f}ms)")
print(f"Savings: {np.median(times) - np.median(times_td):.1f}ms")

# ============================================================================
# Summary
# ============================================================================
print("\n=== Summary ===")
print("""
Key findings:
1. Vol tickers section has similar structure to hedge tickers
2. Type-dispatch should provide similar speedup as hedge tickers optimization
3. The string concat for LMEVOL R-ticker is expensive but only affects LMEVOL rows

Recommended optimization:
- Apply type-dispatch to vol tickers section (same pattern as hedge tickers)
- Expected savings: similar to hedge tickers (~50% reduction)
""")
