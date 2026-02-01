"""Benchmark full hedge tickers section to understand where 51ms goes."""
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

# Extract all asset-level arrays (matching backtest_standalone.py)
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

# Calc-type hedges
calc_hedge1 = hedge_ccy + "_fsw0m_" + hedge_tenor
calc_hedge2 = hedge_ccy + "_fsw6m_" + hedge_tenor
calc_hedge3 = hedge_ccy + "_pva0m_" + hedge_tenor
calc_hedge4 = hedge_ccy + "_pva6m_" + hedge_tenor

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

print(f"Data loaded: smlen={smlen:,}, n_assets={len(anames)}")

# ============================================================================
# Benchmark: Full hedge tickers section (lines 467-532)
# ============================================================================
print("\n=== Full Hedge Tickers Section Breakdown ===")

# Step 1: Build condition arrays (lines 470-481)
t0 = time.perf_counter()
for _ in range(5):
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
t1 = time.perf_counter()
print(f"  Build condition arrays:  {(t1-t0)/5*1000:.1f}ms")

# Step 2: hedge_ticker_smidx (line 485)
t0 = time.perf_counter()
for _ in range(5):
    hedge_ticker_smidx = hedge_ticker[smidx]
t1 = time.perf_counter()
print(f"  hedge_ticker[smidx]:     {(t1-t0)/5*1000:.1f}ms")

# Step 3: fut_idx (line 488)
t0 = time.perf_counter()
for _ in range(5):
    fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)
t1 = time.perf_counter()
print(f"  np.flatnonzero:          {(t1-t0)/5*1000:.1f}ms")
print(f"    fut_idx size: {len(fut_idx):,}")

# Step 4: Futures component extraction (lines 491-502)
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]
    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
    hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m
t1 = time.perf_counter()
print(f"  Futures ticker build:    {(t1-t0)/5*1000:.1f}ms")

# Step 5: Full-size array allocation (lines 505-511)
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_ticker = np.full(smlen, "", dtype=nps)
    hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
    hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m
    hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m
t1 = time.perf_counter()
print(f"  Full-size array alloc:   {(t1-t0)/5*1000:.1f}ms")

# Step 6: Other hedge arrays (lines 514-516)
t0 = time.perf_counter()
for _ in range(5):
    hedge_hedge_smidx = hedge_hedge[smidx]
    calc_hedge1_smidx = calc_hedge1[smidx]
t1 = time.perf_counter()
print(f"  hedge_hedge/calc[smidx]: {(t1-t0)/5*1000:.1f}ms")

# Step 7: np.select for hedge1t/hedge1f (lines 517-520)
t0 = time.perf_counter()
for _ in range(5):
    choices_hedge1t = [hedge_ticker_smidx, hedge_fut_ticker, hedge_hedge_smidx, calc_hedge1_smidx]
    choices_hedge1f = [hedge_field[smidx], "PX_LAST", "PX_LAST", ""]
    hedge1t_vec = np.select(cond_hedge, choices_hedge1t, default="")
    hedge1f_vec = np.select(cond_hedge, choices_hedge1f, default="")
t1 = time.perf_counter()
print(f"  np.select hedge1t/f:     {(t1-t0)/5*1000:.1f}ms")

# Step 8: np.select for hedge2t/hedge2f (lines 522-525)
t0 = time.perf_counter()
for _ in range(5):
    choices_hedge2t = ["", "", hedge_hedge1[smidx], calc_hedge2[smidx]]
    choices_hedge2f = ["", "", "PX_LAST", ""]
    hedge2t_vec = np.select(cond_hedge, choices_hedge2t, default="")
    hedge2f_vec = np.select(cond_hedge, choices_hedge2f, default="")
t1 = time.perf_counter()
print(f"  np.select hedge2t/f:     {(t1-t0)/5*1000:.1f}ms")

# Step 9: np.select for hedge3t/hedge3f (lines 527-529)
t0 = time.perf_counter()
for _ in range(5):
    choices_hedge3t = ["", "", "", calc_hedge3[smidx]]
    hedge3t_vec = np.select(cond_hedge, choices_hedge3t, default="")
    hedge3f_vec = np.full(len(hedge3t_vec), "", dtype="U")
t1 = time.perf_counter()
print(f"  np.select hedge3t/f:     {(t1-t0)/5*1000:.1f}ms")

# Step 10: np.where for hedge4t (lines 531-532)
t0 = time.perf_counter()
for _ in range(5):
    hedge4t_vec = np.where(cond_hedge[3], calc_hedge4[smidx], "")
    hedge4f_vec = hedge3f_vec
t1 = time.perf_counter()
print(f"  np.where hedge4t:        {(t1-t0)/5*1000:.1f}ms")

# Total
print("\n=== End-to-end timing ===")
times = []
for _ in range(5):
    t0 = time.perf_counter()

    # Full hedge tickers section
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

    hedge_ticker_smidx = hedge_ticker[smidx]
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

    hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_ticker = np.full(smlen, "", dtype=nps)
    hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
    hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m
    hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m

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

    t1 = time.perf_counter()
    times.append((t1-t0)*1000)

print(f"Total hedge tickers: {np.median(times):.1f}ms (median)")
print(f"Individual runs: {[f'{t:.1f}' for t in times]}")

# ============================================================================
# Analysis: Where does the time go?
# ============================================================================
print("\n=== Summary ===")
print("""
Time breakdown (approximate):
  - Condition arrays:      ~1ms (boolean indexing)
  - hedge_ticker[smidx]:   ~4ms (string indexing 222K)
  - Futures ticker build:  ~6ms (string concat for 25K)
  - Full-size array alloc: ~4ms (222K empty string arrays + fill)
  - hedge_hedge/calc:      ~8ms (string indexing 222K)
  - np.select hedge1:     ~11ms (creates 222K string arrays)
  - np.select hedge2:     ~11ms (creates 222K string arrays)
  - np.select hedge3:      ~4ms
  - np.where hedge4:       ~2ms

Key insight: The expensive operations are the np.select calls that create
full 222K string arrays for hedge1t/f, hedge2t/f, etc.

The futures dedup optimization only saves ~1ms on the "Futures ticker build"
step because:
1. Pack keys takes ~6ms (two np.unique calls are slow)
2. String build is only ~6ms to begin with
3. Net savings: ~1ms

Bigger wins would require:
1. Not materializing hedge1t/f, hedge2t/f strings at all
2. Work directly with IDs instead of strings
3. Only decode strings at final output assembly
""")
