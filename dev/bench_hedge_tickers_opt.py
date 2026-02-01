"""Benchmark optimized hedge tickers: type-dispatch vs np.select."""
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

# Extract all asset-level arrays
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
# CURRENT APPROACH: Full hedge tickers section
# ============================================================================
def current_approach():
    """Current implementation from backtest_standalone.py"""
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

    # Futures ticker construction
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]
    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
    hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m

    # Full-size arrays
    hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_ticker = np.full(smlen, "", dtype=nps)
    hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
    hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m
    hedge_fut_ticker[fut_idx] = hedge_fut_ticker_m

    hedge_hedge_smidx = hedge_hedge[smidx]
    calc_hedge1_smidx = calc_hedge1[smidx]

    # np.select for all hedge vectors
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

    return (hedge1t_vec, hedge1f_vec, hedge2t_vec, hedge2f_vec,
            hedge3t_vec, hedge3f_vec, hedge4t_vec, hedge4f_vec,
            cond_hedge, fut_idx, hedge_fut_ticker_m, hedge_fut_code_smidx, hedge_fut_tail_smidx)

# ============================================================================
# OPTIMIZED APPROACH: Type-dispatch (only expand strings for rows that need them)
# ============================================================================
def optimized_approach():
    """Optimized: compute indices once, only expand needed strings."""
    # Compute indices once
    nonfut_idx = np.flatnonzero(hedge_source_id_nonfut[smidx])
    fut_idx = np.flatnonzero(hedge_source_id_fut[smidx])
    cds_idx = np.flatnonzero(hedge_source_id_cds[smidx])
    calc_idx = np.flatnonzero(hedge_source_id_calc[smidx])

    # Futures ticker construction (same as before)
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]
    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code[smidx[fut_idx]]
    hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m

    # hedge1t: type-dispatch
    hedge1t_vec = np.empty(smlen, dtype=nps)
    hedge1t_vec.fill("")
    hedge1t_vec[nonfut_idx] = hedge_ticker[smidx[nonfut_idx]]
    hedge1t_vec[fut_idx] = hedge_fut_ticker_m
    hedge1t_vec[cds_idx] = hedge_hedge[smidx[cds_idx]]
    hedge1t_vec[calc_idx] = calc_hedge1[smidx[calc_idx]]

    # hedge1f: type-dispatch
    hedge1f_vec = np.empty(smlen, dtype=nps)
    hedge1f_vec.fill("")
    hedge1f_vec[nonfut_idx] = hedge_field[smidx[nonfut_idx]]
    hedge1f_vec[fut_idx] = "PX_LAST"
    hedge1f_vec[cds_idx] = "PX_LAST"
    # calc_idx stays ""

    # hedge2t/f
    hedge2t_vec = np.empty(smlen, dtype=nps)
    hedge2t_vec.fill("")
    hedge2t_vec[cds_idx] = hedge_hedge1[smidx[cds_idx]]
    hedge2t_vec[calc_idx] = calc_hedge2[smidx[calc_idx]]

    hedge2f_vec = np.empty(smlen, dtype=nps)
    hedge2f_vec.fill("")
    hedge2f_vec[cds_idx] = "PX_LAST"
    # others stay ""

    # hedge3t/f
    hedge3t_vec = np.empty(smlen, dtype=nps)
    hedge3t_vec.fill("")
    hedge3t_vec[calc_idx] = calc_hedge3[smidx[calc_idx]]

    hedge3f_vec = np.full(len(hedge3t_vec), "", dtype="U")

    # hedge4t/f
    hedge4t_vec = np.empty(smlen, dtype=nps)
    hedge4t_vec.fill("")
    hedge4t_vec[calc_idx] = calc_hedge4[smidx[calc_idx]]

    hedge4f_vec = hedge3f_vec

    # Also return data needed for downstream (cond_hedge for ID mapping)
    cond_hedge = [
        np.zeros(smlen, dtype=bool),
        np.zeros(smlen, dtype=bool),
        np.zeros(smlen, dtype=bool),
        np.zeros(smlen, dtype=bool),
    ]
    cond_hedge[0][nonfut_idx] = True
    cond_hedge[1][fut_idx] = True
    cond_hedge[2][cds_idx] = True
    cond_hedge[3][calc_idx] = True

    # Sparse arrays for vol tickers (still needed downstream)
    hedge_fut_code_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_tail_smidx = np.full(smlen, "", dtype=nps)
    hedge_fut_code_smidx[fut_idx] = hedge_fut_code_m
    hedge_fut_tail_smidx[fut_idx] = hedge_fut_tail_m

    return (hedge1t_vec, hedge1f_vec, hedge2t_vec, hedge2f_vec,
            hedge3t_vec, hedge3f_vec, hedge4t_vec, hedge4f_vec,
            cond_hedge, fut_idx, hedge_fut_ticker_m, hedge_fut_code_smidx, hedge_fut_tail_smidx)

# ============================================================================
# Verify correctness
# ============================================================================
print("\n=== Verify correctness ===")
ref = current_approach()
opt = optimized_approach()

all_match = True
for i, name in enumerate(["hedge1t", "hedge1f", "hedge2t", "hedge2f",
                          "hedge3t", "hedge3f", "hedge4t", "hedge4f"]):
    if np.array_equal(ref[i], opt[i]):
        print(f"✓ {name} matches")
    else:
        mismatch = np.sum(ref[i] != opt[i])
        print(f"✗ {name}: {mismatch:,} mismatches")
        all_match = False
        # Debug
        idx = np.flatnonzero(ref[i] != opt[i])[0]
        print(f"  First mismatch at {idx}: ref='{ref[i][idx]}' opt='{opt[i][idx]}'")

if all_match:
    print("\nAll outputs match!")

# ============================================================================
# Benchmark
# ============================================================================
print("\n=== Benchmark (5 runs each) ===")

# Warmup
current_approach()
optimized_approach()

times_current = []
for _ in range(5):
    t0 = time.perf_counter()
    _ = current_approach()
    t1 = time.perf_counter()
    times_current.append((t1-t0)*1000)

times_optimized = []
for _ in range(5):
    t0 = time.perf_counter()
    _ = optimized_approach()
    t1 = time.perf_counter()
    times_optimized.append((t1-t0)*1000)

print(f"Current approach:   {np.median(times_current):.1f}ms (runs: {[f'{t:.1f}' for t in times_current]})")
print(f"Optimized approach: {np.median(times_optimized):.1f}ms (runs: {[f'{t:.1f}' for t in times_optimized]})")
print(f"Speedup: {np.median(times_current)/np.median(times_optimized):.2f}x")
print(f"Savings: {np.median(times_current) - np.median(times_optimized):.1f}ms")

# ============================================================================
# Breakdown of optimized approach
# ============================================================================
print("\n=== Breakdown of optimized approach ===")

# Index computation
t0 = time.perf_counter()
for _ in range(5):
    nonfut_idx = np.flatnonzero(hedge_source_id_nonfut[smidx])
    fut_idx = np.flatnonzero(hedge_source_id_fut[smidx])
    cds_idx = np.flatnonzero(hedge_source_id_cds[smidx])
    calc_idx = np.flatnonzero(hedge_source_id_calc[smidx])
t1 = time.perf_counter()
print(f"  Index computation:  {(t1-t0)/5*1000:.1f}ms")
print(f"    nonfut: {len(nonfut_idx):,}, fut: {len(fut_idx):,}, cds: {len(cds_idx):,}, calc: {len(calc_idx):,}")

# Futures ticker build
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
print(f"  Futures ticker:     {(t1-t0)/5*1000:.1f}ms")

# hedge1t type-dispatch
t0 = time.perf_counter()
for _ in range(5):
    hedge1t_vec = np.empty(smlen, dtype=nps)
    hedge1t_vec.fill("")
    hedge1t_vec[nonfut_idx] = hedge_ticker[smidx[nonfut_idx]]
    hedge1t_vec[fut_idx] = hedge_fut_ticker_m
    hedge1t_vec[cds_idx] = hedge_hedge[smidx[cds_idx]]
    hedge1t_vec[calc_idx] = calc_hedge1[smidx[calc_idx]]
t1 = time.perf_counter()
print(f"  hedge1t dispatch:   {(t1-t0)/5*1000:.1f}ms")

# hedge2t type-dispatch
t0 = time.perf_counter()
for _ in range(5):
    hedge2t_vec = np.empty(smlen, dtype=nps)
    hedge2t_vec.fill("")
    hedge2t_vec[cds_idx] = hedge_hedge1[smidx[cds_idx]]
    hedge2t_vec[calc_idx] = calc_hedge2[smidx[calc_idx]]
t1 = time.perf_counter()
print(f"  hedge2t dispatch:   {(t1-t0)/5*1000:.1f}ms")
