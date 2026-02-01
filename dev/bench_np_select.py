"""Benchmark np.select alternatives for string arrays."""
import numpy as np
import time
from numba import njit, prange

# Create test data matching hedge tickers section
smlen = 222_300
nps = np.dtypes.StringDType()

# Simulate condition arrays (boolean)
np.random.seed(42)
cond1 = np.random.rand(smlen) < 0.25  # nonfut
cond2 = np.random.rand(smlen) < 0.11  # fut
cond3 = np.random.rand(smlen) < 0.30  # cds
cond4 = ~(cond1 | cond2 | cond3)      # calc (remainder)

# Simulate choice arrays (strings)
choices = [
    np.array([f"ticker_{i % 100}" for i in range(smlen)], dtype=nps),
    np.array([f"fut_{i % 50}" for i in range(smlen)], dtype=nps),
    np.array([f"cds_{i % 30}" for i in range(smlen)], dtype=nps),
    np.array([f"calc_{i % 20}" for i in range(smlen)], dtype=nps),
]

print(f"Test data: smlen={smlen:,}")
print(f"Condition distribution: nonfut={cond1.sum():,}, fut={cond2.sum():,}, cds={cond3.sum():,}, calc={cond4.sum():,}")

# ============================================================================
# Approach 1: np.select (current)
# ============================================================================
def approach_np_select():
    return np.select([cond1, cond2, cond3, cond4], choices, default="")

# ============================================================================
# Approach 2: np.where chain
# ============================================================================
def approach_np_where():
    # Reversed order: last condition first
    result = np.where(cond4, choices[3], "")
    result = np.where(cond3, choices[2], result)
    result = np.where(cond2, choices[1], result)
    result = np.where(cond1, choices[0], result)
    return result

# ============================================================================
# Approach 3: Pre-allocate and fill
# ============================================================================
def approach_fill():
    result = np.empty(smlen, dtype=nps)
    result.fill("")  # default
    result[cond4] = choices[3][cond4]
    result[cond3] = choices[2][cond3]
    result[cond2] = choices[1][cond2]
    result[cond1] = choices[0][cond1]
    return result

# ============================================================================
# Approach 4: Integer index array + fancy indexing
# ============================================================================
def approach_int_index():
    # Build integer index: 0=nonfut, 1=fut, 2=cds, 3=calc, 4=default
    idx = np.full(smlen, 4, dtype=np.int8)
    idx[cond4] = 3
    idx[cond3] = 2
    idx[cond2] = 1
    idx[cond1] = 0

    # Stack choices with default
    all_choices = np.vstack([c for c in choices] + [np.full(smlen, "", dtype=nps)])

    # Fancy index
    return all_choices[idx, np.arange(smlen)]

# ============================================================================
# Approach 5: Use smidx-based pre-computed index
# ============================================================================
# This simulates the case where we precompute a hedge_type_id array at asset level
# and just expand it with smidx indexing

# Simulate asset-level hedge type (0-3)
n_assets = 189
asset_hedge_type = np.random.randint(0, 4, n_assets, dtype=np.int8)

# smidx maps straddles to assets
smidx = np.random.randint(0, n_assets, smlen, dtype=np.int32)

# Expand to straddle level
hedge_type_smidx = asset_hedge_type[smidx]

def approach_precomputed_type():
    # Build choices as 2D: [type, straddle] -> string
    # Actually need (5, smlen) but that's huge
    # Instead, use advanced indexing per type
    idx = hedge_type_smidx

    result = np.empty(smlen, dtype=nps)
    result.fill("")
    mask0 = idx == 0
    mask1 = idx == 1
    mask2 = idx == 2
    mask3 = idx == 3
    result[mask0] = choices[0][mask0]
    result[mask1] = choices[1][mask1]
    result[mask2] = choices[2][mask2]
    result[mask3] = choices[3][mask3]
    return result

# ============================================================================
# Verify correctness
# ============================================================================
print("\n=== Verify correctness ===")
ref = approach_np_select()
for name, fn in [
    ("np.where", approach_np_where),
    ("fill", approach_fill),
    # ("int_index", approach_int_index),  # Different structure
]:
    result = fn()
    if np.array_equal(ref, result):
        print(f"✓ {name} matches")
    else:
        mismatch = np.sum(ref != result)
        print(f"✗ {name}: {mismatch:,} mismatches")

# ============================================================================
# Benchmark
# ============================================================================
print("\n=== Benchmark (5 runs each) ===")

approaches = [
    ("np.select", approach_np_select),
    ("np.where chain", approach_np_where),
    ("fill", approach_fill),
    # ("int_index", approach_int_index),
    # ("precomputed_type", approach_precomputed_type),
]

# Warmup
for name, fn in approaches:
    fn()

results = {}
for name, fn in approaches:
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    results[name] = np.median(times)
    print(f"{name:20s}: {np.median(times):.1f}ms (runs: {[f'{t:.1f}' for t in times]})")

# ============================================================================
# Test with actual hedge_source_id approach
# ============================================================================
print("\n=== Test with smidx-expanded condition ===")
# This more closely matches the actual code structure

# Asset-level hedge source type (0=nonfut, 1=fut, 2=cds, 3=calc)
asset_hedge_type = np.random.randint(0, 4, n_assets, dtype=np.int8)

# Pre-compute boolean masks at asset level
asset_is_nonfut = asset_hedge_type == 0
asset_is_fut = asset_hedge_type == 1
asset_is_cds = asset_hedge_type == 2
asset_is_calc = asset_hedge_type == 3

# Simulate smidx (maps 222K straddles to 189 assets)
# In reality, each asset has multiple straddles across all months
smidx = np.tile(np.arange(n_assets), smlen // n_assets + 1)[:smlen].astype(np.int32)

# Expand booleans via smidx
is_nonfut = asset_is_nonfut[smidx]
is_fut = asset_is_fut[smidx]
is_cds = asset_is_cds[smidx]
is_calc = asset_is_calc[smidx]

# Asset-level string arrays (expanded via smidx)
asset_ticker = np.array([f"ticker_{i}" for i in range(n_assets)], dtype=nps)
asset_fut = np.array([f"fut_{i}" for i in range(n_assets)], dtype=nps)
asset_cds = np.array([f"cds_{i}" for i in range(n_assets)], dtype=nps)
asset_calc = np.array([f"calc_{i}" for i in range(n_assets)], dtype=nps)

def approach_smidx_select():
    """Current approach: expand strings then select."""
    ticker_smidx = asset_ticker[smidx]
    fut_smidx = asset_fut[smidx]
    cds_smidx = asset_cds[smidx]
    calc_smidx = asset_calc[smidx]

    return np.select(
        [is_nonfut, is_fut, is_cds, is_calc],
        [ticker_smidx, fut_smidx, cds_smidx, calc_smidx],
        default=""
    )

def approach_smidx_type_dispatch():
    """Alternative: dispatch by type, only expand needed strings."""
    result = np.empty(smlen, dtype=nps)
    result.fill("")

    # Only expand the strings for rows that need them
    nonfut_idx = np.flatnonzero(is_nonfut)
    fut_idx = np.flatnonzero(is_fut)
    cds_idx = np.flatnonzero(is_cds)
    calc_idx = np.flatnonzero(is_calc)

    result[nonfut_idx] = asset_ticker[smidx[nonfut_idx]]
    result[fut_idx] = asset_fut[smidx[fut_idx]]
    result[cds_idx] = asset_cds[smidx[cds_idx]]
    result[calc_idx] = asset_calc[smidx[calc_idx]]

    return result

def approach_smidx_combined():
    """Alternative: use asset-level index to choose string array."""
    # Asset-level: combined strings (one per type)
    # Straddle-level: just index by type

    # Pre-expand hedge type to straddle level
    hedge_type = asset_hedge_type[smidx]  # int8 array

    result = np.empty(smlen, dtype=nps)
    result.fill("")

    for t, arr in enumerate([asset_ticker, asset_fut, asset_cds, asset_calc]):
        mask = hedge_type == t
        result[mask] = arr[smidx[mask]]

    return result

# Verify
print("Verifying...")
ref = approach_smidx_select()
for name, fn in [
    ("type_dispatch", approach_smidx_type_dispatch),
    ("combined", approach_smidx_combined),
]:
    result = fn()
    if np.array_equal(ref, result):
        print(f"✓ {name} matches")
    else:
        mismatch = np.sum(ref != result)
        print(f"✗ {name}: {mismatch:,} mismatches")

# Benchmark
print("\nBenchmark:")
approaches2 = [
    ("smidx_select", approach_smidx_select),
    ("type_dispatch", approach_smidx_type_dispatch),
    ("combined", approach_smidx_combined),
]

for name, fn in approaches2:
    fn()  # warmup

for name, fn in approaches2:
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    print(f"{name:20s}: {np.median(times):.1f}ms")
