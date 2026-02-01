"""Benchmark hedge tickers optimization: dedupe-first vs current approach."""
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

# Extract asset-level arrays
hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_fut_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_code", ""), anames)), dtype=nps)
hedge_market_code = np.array(list(map(lambda a: amap[a]["Hedge"].get("market_code", ""), anames)), dtype=nps)
hedge_fut_month_map = np.array(list(map(lambda a: amap[a]["Hedge"].get("fut_month_map", " " * FUT_MONTH_MAP_LEN), anames)), dtype=nps)
hedge_min_year_offset = np.array(list(map(lambda a: amap[a]["Hedge"].get("min_year_offset", "0"), anames)), dtype=nps)

hedge_sources, hedge_source_id = np.unique(hedge_source, return_inverse=True)
hs2id_map = dict(zip(hedge_sources, range(len(hedge_sources))))
HEDGE_FUT = hs2id_map["fut"]
hedge_source_id_fut = hedge_source_id == HEDGE_FUT

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

# Load ticker dict for searchsorted
ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
ticker_order = np.argsort(ticker_arr)
ticker_sorted = ticker_arr[ticker_order]

def map_to_id_searchsorted(queries, sorted_arr, order):
    """Map string queries to IDs using searchsorted."""
    pos = np.searchsorted(sorted_arr, queries)
    pos = np.clip(pos, 0, len(sorted_arr) - 1)
    valid = sorted_arr[pos] == queries
    ids = np.where(valid, order[pos], -1)
    return ids.astype(np.int32)

print(f"Data loaded: smlen={smlen:,}, n_assets={len(anames)}")

# Futures indices
hedge_source_id_fut_smidx = hedge_source_id_fut[smidx]
fut_idx = np.flatnonzero(hedge_source_id_fut_smidx)
print(f"Futures rows: {len(fut_idx):,} ({100*len(fut_idx)/smlen:.1f}% of total)")

# ============================================================================
# CURRENT APPROACH: Build strings for all fut_idx rows
# ============================================================================
def current_approach():
    """Current implementation: build all 25k ticker strings."""
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

    # Map to IDs
    hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)

    return hedge_fut_ticker_tid_m

# ============================================================================
# OPTIMIZED APPROACH: Dedupe first using packed keys
# ============================================================================
def optimized_approach():
    """Optimized: pack components into int64, dedupe, build strings only for unique."""
    # Get component arrays (same as current)
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_market_code_m = hedge_market_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]

    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
    year_i = (year_vec[fut_idx] + yo_m).astype(np.int64)

    # Get categorical IDs for string components
    fut_code_u, fut_code_id = np.unique(hedge_fut_code_m, return_inverse=True)
    mkt_code_u, mkt_code_id = np.unique(hedge_market_code_m, return_inverse=True)

    # Month code to id (0..11)
    month_code_id = np.searchsorted(month_code, hedge_fut_month_code_m).astype(np.int64)

    # Pack into int64: [fut_code_id(16) | mkt_code_id(16) | month_id(4) | year(12)]
    year_pack = year_i - 1900
    packed = (
        (fut_code_id.astype(np.int64) << 32) |
        (mkt_code_id.astype(np.int64) << 16) |
        (month_code_id.astype(np.int64) << 12) |
        (year_pack.astype(np.int64))
    )

    uniq_packed, inv = np.unique(packed, return_inverse=True)

    # Decode unique components
    uf = (uniq_packed >> 32).astype(np.int64)
    um = ((uniq_packed >> 16) & 0xFFFF).astype(np.int64)
    umon = ((uniq_packed >> 12) & 0xF).astype(np.int64)
    uy = (uniq_packed & 0xFFF).astype(np.int64) + 1900

    # Build strings ONLY for unique tickers
    uniq_ticker = (
        fut_code_u[uf] +
        month_code[umon] +
        uy.astype("U") +
        " " +
        mkt_code_u[um]
    ).astype(TICKER_U)

    # Map unique strings to IDs
    uniq_tid = map_to_id_searchsorted(uniq_ticker, ticker_sorted, ticker_order)

    # Expand back to per-row IDs
    hedge_fut_ticker_tid_m = uniq_tid[inv].astype(np.int32)

    return hedge_fut_ticker_tid_m

# ============================================================================
# Verify correctness
# ============================================================================
print("\n=== Verifying correctness ===")
result_current = current_approach()
result_optimized = optimized_approach()

if np.array_equal(result_current, result_optimized):
    print("✓ Results match!")
else:
    mismatch = np.sum(result_current != result_optimized)
    print(f"✗ Results differ: {mismatch:,} mismatches out of {len(result_current):,}")
    # Debug first mismatch
    idx = np.flatnonzero(result_current != result_optimized)[0]
    print(f"  First mismatch at index {idx}: current={result_current[idx]}, optimized={result_optimized[idx]}")

# ============================================================================
# Analyze uniqueness
# ============================================================================
print("\n=== Uniqueness Analysis ===")

# Get components for analysis
hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
hedge_market_code_m = hedge_market_code[smidx[fut_idx]]
hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]
myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
year_i = (year_vec[fut_idx] + yo_m).astype(np.int64)

fut_code_u, fut_code_id = np.unique(hedge_fut_code_m, return_inverse=True)
mkt_code_u, mkt_code_id = np.unique(hedge_market_code_m, return_inverse=True)
month_code_id = np.searchsorted(month_code, hedge_fut_month_code_m).astype(np.int64)
year_pack = year_i - 1900

packed = (
    (fut_code_id.astype(np.int64) << 32) |
    (mkt_code_id.astype(np.int64) << 16) |
    (month_code_id.astype(np.int64) << 12) |
    (year_pack.astype(np.int64))
)

uniq_packed = np.unique(packed)

print(f"Total futures rows:     {len(fut_idx):,}")
print(f"Unique fut_codes:       {len(fut_code_u)}")
print(f"Unique market_codes:    {len(mkt_code_u)}")
print(f"Unique packed keys:     {len(uniq_packed):,}")
print(f"Dedup ratio:            {len(fut_idx) / len(uniq_packed):.1f}x")
print(f"String allocations saved: {len(fut_idx) - len(uniq_packed):,}")

# ============================================================================
# Benchmark
# ============================================================================
print("\n=== Benchmark (5 runs each) ===")

# Warmup
current_approach()
optimized_approach()

# Current approach
times_current = []
for i in range(5):
    t0 = time.perf_counter()
    _ = current_approach()
    t1 = time.perf_counter()
    times_current.append((t1-t0)*1000)

# Optimized approach
times_optimized = []
for i in range(5):
    t0 = time.perf_counter()
    _ = optimized_approach()
    t1 = time.perf_counter()
    times_optimized.append((t1-t0)*1000)

print(f"\nCurrent approach:   {np.median(times_current):.1f}ms (median of {times_current})")
print(f"Optimized approach: {np.median(times_optimized):.1f}ms (median of {times_optimized})")
print(f"Speedup: {np.median(times_current)/np.median(times_optimized):.2f}x")

# ============================================================================
# Breakdown: where does the time go?
# ============================================================================
print("\n=== Time Breakdown (current approach) ===")

# Component extraction
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_code_m = hedge_fut_code[smidx[fut_idx]]
    hedge_market_code_m = hedge_market_code[smidx[fut_idx]]
    hedge_fut_month_code_m = hedge_fut_month_mtrx[smidx[fut_idx], month_vec[fut_idx]-1]
    myo_m = hedge_min_year_offset_int[smidx[fut_idx]]
t1 = time.perf_counter()
print(f"  Component extraction: {(t1-t0)/5*1000:.1f}ms")

# Year offset calc
t0 = time.perf_counter()
for _ in range(5):
    month_code = np.frombuffer(b"FGHJKMNQUVXZ", dtype="S1").astype("U1")
    hedge_opt_month_code_m = month_code[month_vec[fut_idx]-1]
    yo_m = np.maximum(np.where(hedge_fut_month_code_m < hedge_opt_month_code_m, 1, 0), myo_m)
t1 = time.perf_counter()
print(f"  Year offset calc:     {(t1-t0)/5*1000:.1f}ms")

# String concat (tail)
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_yeartxt_m = (year_vec[fut_idx] + yo_m).astype("U")
    hedge_fut_tail_m = hedge_fut_month_code_m + hedge_fut_yeartxt_m + " " + hedge_market_code_m
t1 = time.perf_counter()
print(f"  String concat (tail): {(t1-t0)/5*1000:.1f}ms")

# String concat (ticker)
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_ticker_m = hedge_fut_code_m + hedge_fut_tail_m
t1 = time.perf_counter()
print(f"  String concat (ticker): {(t1-t0)/5*1000:.1f}ms")

# Type conversion
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_ticker_m_u = hedge_fut_ticker_m.astype(TICKER_U)
t1 = time.perf_counter()
print(f"  Type conversion (U100): {(t1-t0)/5*1000:.1f}ms")

# Searchsorted
t0 = time.perf_counter()
for _ in range(5):
    hedge_fut_ticker_tid_m = map_to_id_searchsorted(hedge_fut_ticker_m_u, ticker_sorted, ticker_order)
t1 = time.perf_counter()
print(f"  Searchsorted:         {(t1-t0)/5*1000:.1f}ms")

print("\n=== Time Breakdown (optimized approach) ===")

# Pack keys
t0 = time.perf_counter()
for _ in range(5):
    fut_code_u, fut_code_id = np.unique(hedge_fut_code_m, return_inverse=True)
    mkt_code_u, mkt_code_id = np.unique(hedge_market_code_m, return_inverse=True)
    month_code_id = np.searchsorted(month_code, hedge_fut_month_code_m).astype(np.int64)
    year_pack = (year_vec[fut_idx] + yo_m - 1900).astype(np.int64)
    packed = (
        (fut_code_id.astype(np.int64) << 32) |
        (mkt_code_id.astype(np.int64) << 16) |
        (month_code_id.astype(np.int64) << 12) |
        year_pack
    )
t1 = time.perf_counter()
print(f"  Pack keys:            {(t1-t0)/5*1000:.1f}ms")

# Unique
t0 = time.perf_counter()
for _ in range(5):
    uniq_packed, inv = np.unique(packed, return_inverse=True)
t1 = time.perf_counter()
print(f"  np.unique:            {(t1-t0)/5*1000:.1f}ms")

# Decode unique
t0 = time.perf_counter()
for _ in range(5):
    uf = (uniq_packed >> 32).astype(np.int64)
    um = ((uniq_packed >> 16) & 0xFFFF).astype(np.int64)
    umon = ((uniq_packed >> 12) & 0xF).astype(np.int64)
    uy = (uniq_packed & 0xFFF).astype(np.int64) + 1900
t1 = time.perf_counter()
print(f"  Decode unique:        {(t1-t0)/5*1000:.1f}ms")

# Build unique strings
t0 = time.perf_counter()
for _ in range(5):
    uniq_ticker = (
        fut_code_u[uf] +
        month_code[umon] +
        uy.astype("U") +
        " " +
        mkt_code_u[um]
    ).astype(TICKER_U)
t1 = time.perf_counter()
print(f"  Build unique strings: {(t1-t0)/5*1000:.1f}ms")

# Map unique to IDs
t0 = time.perf_counter()
for _ in range(5):
    uniq_tid = map_to_id_searchsorted(uniq_ticker, ticker_sorted, ticker_order)
t1 = time.perf_counter()
print(f"  Map unique to IDs:    {(t1-t0)/5*1000:.1f}ms")

# Expand back
t0 = time.perf_counter()
for _ in range(5):
    result = uniq_tid[inv].astype(np.int32)
t1 = time.perf_counter()
print(f"  Expand back:          {(t1-t0)/5*1000:.1f}ms")
