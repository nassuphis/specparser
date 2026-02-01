# Performance Experiments: Vectorized Backtest Pipeline

This document summarizes all performance optimization experiments conducted on the backtest pipeline, including approaches tested, benchmark results, and conclusions.

## Executive Summary

| Optimization | Speedup | Status |
|--------------|---------|--------|
| Numba parallel binary search (vs Arrow join) | 10x | Implemented |
| Asset-level ID precomputation | 38x | Implemented |
| int32 for d_stridx | ~10% | Implemented |
| Key-grouped merge sweep | 4.1x | Implemented |
| Skip expand days in sweep mode | ~95ms saved | Implemented |
| Searchsorted for ticker ID mapping | ~2ms | Implemented (marginal) |
| NumPy format (replace Parquet) | ~220ms saved | Implemented |
| Hedge tickers type-dispatch | 2x (~27ms saved) | Superseded by ID-only |
| **ID-only pipeline (5 legs)** | **5 legs for ~2x cost of 1** | **Implemented** |
| Filter-first sweep (sparse legs) | 7-8x for sparse | Implemented (in ID-only) |
| Fused sweep (5 legs in 1 sort) | 0.86x (-13ms) | Rejected (sequential faster) |
| Single-pass asset extraction + per-field unique | 0ms | Rejected (no improvement) |
| Sweep merge run-based buffering | 0ms | Rejected (already memory-bound) |
| Futures ticker dedup (packed keys) | ~1ms | Rejected (overhead too high) |

**Final pipeline performance:**
- Before optimizations: ~748ms
- After algorithmic optimizations: ~490ms (34% faster)
- After NumPy format: ~275ms (63% faster total)
- After hedge tickers type-dispatch: ~243ms (67% faster total)
- **After ID-only 5-leg pipeline: ~296ms** (all 5 legs vs 1 leg previously)

---

## 1. Numba Parallel Binary Search

### Problem
Arrow join was taking ~174ms to match 6.7M daily queries with 8.5M price rows.

### Solution
Replace Arrow join with Numba parallel binary search:
1. Pre-sort prices by (key, date) into `prices_keyed_sorted.parquet`
2. Build key→block mapping for O(1) key lookup
3. Use `@njit(parallel=True)` with `prange` for parallel binary search

### Results

```
Arrow join:           149.6ms
Numba parallel:        13.6ms
Speedup:              10x
```

### Key Code

```python
@njit(parallel=True)
def _lookup_parallel(q_key, q_date, block_of, starts, ends, px_date, px_value):
    out = np.empty(len(q_key), dtype=np.float64)
    out[:] = np.nan
    for i in prange(len(q_key)):
        k = q_key[i]
        b = block_of[k]
        if b >= 0:
            j = _binsearch(px_date, starts[b], ends[b], q_date[i])
            if j >= 0:
                out[i] = px_value[j]
    return out
```

### Why It Works
- Queries grouped by key (~40K queries per key change)
- Key's price block stays in L2/L3 cache while queried ~40K times
- Effective bandwidth: 69 GB/s (cache, not RAM)

---

## 2. Asset-level ID Precomputation

### Problem
`np.unique` on 222K monthly tickers took ~65ms per run.

### Solution
Pre-map ticker→ID at asset level (189 assets) during startup, then use integer `np.select`:

```python
# At startup (once)
hedge_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_ticker])

# At runtime (fast integer indexing)
hedge_ticker_tid_smidx = hedge_ticker_tid[smidx]
month_tid = np.select(cond_hedge, choices_tid, default=-1)
```

### Results

```
Before (np.unique):     65.0ms
After (precompute):      1.7ms
Speedup:               38x
```

---

## 3. int32 Optimization

### Problem
15M-element arrays using int64 waste cache space.

### Solution
Use int32 for indices that fit:

```python
d_stridx = np.repeat(np.arange(len(day_count_vec), dtype=np.int32), day_count_vec)
```

### Results

```
int64:   84.3ms
int32:   75.1ms
Savings: ~9ms
```

---

## 4. Packed Key Experiments (Rejected)

### Hypothesis
Pack (key, date) into single uint64 for fewer memory accesses.

### Approaches Tested

1. **Global binary search on packed keys**
   - Eliminates block lookup overhead
   - Result: 15.4ms vs 10.3ms for block-based (slower!)
   - Why: More comparisons (23 vs 11) due to larger search space

2. **32-bit packing**
   - Key: 13 bits, Date: 16 bits relative = 29 bits total
   - Fits in uint32
   - Result: No improvement, block-based already optimal

### Conclusion
Block-based approach is optimal. Fewer comparisons beat fewer memory accesses.

---

## 5. Merge Join Experiments (Rejected for current query order)

### Hypothesis
Sort queries, then merge-join with prices in O(P + Q) time.

### Results

```
Merge join kernel:     1.1ms (11x faster than binary search!)
Query sort overhead:  135.3ms
Total:               145.5ms (slower)
```

### Key Finding
Queries are grouped by key (148 changes for 6.7M queries) but dates NOT sorted within keys. The 135ms sort overhead kills the benefit.

**If queries were generated sorted by date, merge would win decisively.**

---

## 6. Sweep-wise Experiments (Implemented)

### Motivation
Current approach:
1. Expand 222K monthly straddles to 15.4M daily rows
2. For each daily row, binary search for price

This creates 15.4M random memory accesses and allocates ~216MB of temporary arrays.

### Experiment A: Key-grouped Merge

**Concept:**
1. Sort 222K monthly straddles by (key, start_epoch) - 17ms
2. For each key group, two-pointer merge straddles with price series

```python
@njit(parallel=True)
def _merge_per_key(g_keys, g_starts, g_ends, m_start_s, m_len_s, m_out0_s,
                   px_block_of, px_starts, px_ends, px_date, px_value, out):
    for gi in prange(len(g_keys)):
        key = g_keys[gi]
        b = px_block_of[key]
        ps, pe = px_starts[b], px_ends[b]

        for si in range(g_starts[gi], g_ends[gi]):
            straddle_start = m_start_s[si]
            straddle_end = straddle_start + m_len_s[si]

            # Two-pointer merge
            pi = ps
            while pi < pe and px_date[pi] < straddle_start:
                pi += 1
            while pi < pe and px_date[pi] < straddle_end:
                out[m_out0_s[si] + px_date[pi] - straddle_start] = px_value[pi]
                pi += 1
```

### Experiment B: Sweep-line

**Concept:**
- Iterate price dates (not straddle days)
- Maintain active straddle set
- Assign each price to all overlapping straddles

### Results

```
BASELINE (expand + binary search):
  Expand:     102.8ms
  Lookup:      18.6ms
  Total:      121.4ms

EXPERIMENT A (key-grouped merge):
  Sort:        17.9ms
  Merge:       16.4ms
  Total:       34.2ms
  Speedup:     3.55x      <- WINNER

EXPERIMENT B (sweep-line):
  Sort:        17.9ms
  Sweep:       20.4ms
  Total:       38.3ms
  Speedup:     3.17x
```

### Why A Beats B

Despite ~1,173 straddles per key (high overlap), A wins because:
1. A's forward-only scan is simpler (better ILP)
2. B must check ~600 straddles per price date for expiry
3. A accesses price array sequentially; B accesses straddle arrays repeatedly

### Memory Savings

Experiment A avoids allocating:
- `d_tid`: 15.4M × 4 bytes
- `d_fid`: 15.4M × 4 bytes
- `d_key`: 15.4M × 4 bytes
- `d_epoch_i32`: 15.4M × 4 bytes
- **Total: 216 MB**

### Verification

All 7,880,819 found values match exactly between baseline and Experiment A.

### Implementation Notes (Bugs Fixed)

Two bugs were identified during integration:

1. **pi reset bug**: Original code reset `pi = ps` for every straddle, rescanning prices from the start. Fixed by maintaining forward-only `pi0` pointer that never goes backward within a key group.

2. **Unnecessary expand days**: Original integration still ran full daily expansion (~95ms) even in sweep mode. Fixed by skipping expand days when `USE_SWEEP_MERGE=True`.

After fixes:
```
sweep merge prep:   ~18ms
sweep merge:        ~92ms
Total lookup:      ~110ms (was ~444ms with expand+binary search)
Speedup:           4.1x
```

---

## 7. Bandwidth Analysis

### Real Data vs Random Queries

```
Random queries (15M):   39.0ms
Real data queries:      11.8ms
Ratio:                  0.33x (real is 3x faster)
```

### Why Real Data is Fast

- Average 40,009 queries per key change
- Key's price block stays in L2/L3 cache
- Effective bandwidth: 69 GB/s (cache, not memory)

### Conclusion

**At physical memory/cache bandwidth limit.** No further algorithmic improvement possible for the current binary search approach.

---

## 8. Price-driven Approach (Rejected)

### Hypothesis
Build hash table of queries, iterate prices and assign.

### Results

```
Hash build:    ~1000ms
Assignment:    ~2400ms
Total:         ~3400ms (333x slower!)
```

### Why It Failed

Python dict overhead for 6.7M entries is catastrophic. Would need a C extension or Numba-compatible hash table.

---

## 9. Searchsorted for Ticker ID Mapping (Marginal)

### Hypothesis
Replace `str(s)` loops and dict lookups with vectorized `np.searchsorted` for ticker→ID mapping.

### Approaches Tested

1. **Asset-level mapping (189 elements)**
   - Replace: `[ticker_to_id.get(str(s), -1) for s in hedge_ticker]`
   - With: `np.searchsorted` on sorted ticker array
   - Result: No measurable improvement (array too small)

2. **Futures ticker mapping (~25K elements)**
   - Replace: `np.unique` + dict building + `str(s)` loops
   - With: Direct `np.searchsorted` call
   - Result: ~2ms savings (10ms → 8ms)

3. **Early filtering of invalid keys**
   - Add `m_key_s <= max_valid_key` check in sweep prep
   - Result: Negligible (~1ms)

### Results

```
map monthly ids (before):  ~10ms
map monthly ids (after):    ~8ms
Savings:                    ~2ms
```

### Why Limited Impact

1. **Asset-level arrays are trivial (189 elements)** - the `str(s)` conversion overhead is negligible
2. **"Processing amt" dominated by other work:**
   - YAML dict traversal with `list(map(lambda ...))` calls
   - Schedule matrix string parsing (`np.strings.partition/slice`)
   - `np.unique` on schedule_matrix (232×N×5 string array)
   - Parquet file loads
3. **Futures ticker mapping was already fast** - only ~10ms, not a major bottleneck

### Conclusion

Searchsorted is correctly implemented and works, but the expected 20-40ms savings didn't materialize. The `str(s)` overhead was a small fraction of "processing amt" time. Further optimization would require caching YAML processing results or rewriting schedule matrix parsing.

**Current total: ~490ms is at a good stopping point.**

---

## 10. Processing AMT Optimizations (Minimal Impact)

### Problem
"Processing amt" at ~145ms remained a large portion of total time. Two optimizations were attempted to reduce this.

### Optimization A: Single-pass Asset Attribute Extraction

**Before:**
```python
hedge_source = np.array(list(map(lambda a: amap[a]["Hedge"].get("Source", ""), anames)), dtype=nps)
hedge_ticker = np.array(list(map(lambda a: amap[a]["Hedge"].get("Ticker", ""), anames)), dtype=nps)
# ... 15 more separate list(map(...)) calls
```

**After:**
```python
n_assets = len(anames)
hedge_source = np.empty(n_assets, dtype=nps)
hedge_ticker = np.empty(n_assets, dtype=nps)
# ... pre-allocate all arrays

for i, a in enumerate(anames):
    h = amap[a]["Hedge"]
    v = amap[a]["Vol"]
    hedge_source[i] = h.get("Source", "")
    hedge_ticker[i] = h.get("Ticker", "")
    # ... all 15 attributes in one loop
```

### Optimization B: Per-field np.unique (Instead of 3D String Unique)

**Before:**
```python
schedule_matrix = np.empty((len(amap), np.max(easchcnt), 5), dtype=nps)
# ... fill 3D array with strings
schedule_unique, schedule_ids_flat = np.unique(schedule_matrix, return_inverse=True)
```

**After:**
```python
# Parse each field separately
ntrc_uniq, ntrc_ids_flat = np.unique(ntrc_flat, return_inverse=True)
ntrv_uniq, ntrv_ids_flat = np.unique(ntrv_flat, return_inverse=True)
xprc_uniq, xprc_ids_flat = np.unique(xprc_flat, return_inverse=True)
xprv_uniq, xprv_ids_flat = np.unique(xprv_flat, return_inverse=True)
wgt_uniq, wgt_ids_flat = np.unique(wgt_flat, return_inverse=True)
```

### Results

```
Processing amt (before): ~145ms
Processing amt (after):  ~150ms
Savings:                 None (slight regression within noise)
```

### Why It Failed

1. **N=189 is too small**: With only 189 assets, the overhead of 15 separate `list(map(...))` calls vs one loop is negligible (~1ms difference)
2. **Per-field unique processes similar data**: Five 1D uniques on ~41K elements each is not significantly faster than one 3D unique on 232×max_schedules×5 string array
3. **String parsing dominates**: The `np.strings.partition/slice` calls and YAML dict traversal are the actual bottleneck, not the unique operations

### Conclusion

These optimizations are correctly implemented but yield no measurable improvement. The "processing amt" time is dominated by:
- YAML dict access patterns (nested dicts, string keys)
- `np.strings.partition` and `np.strings.slice` operations
- Python list/dict overhead at the interpreter level

Further optimization would require either C extension code or restructuring the input data format entirely.

---

## 11. NumPy Format (Replace Parquet)

### Problem

After all algorithmic optimizations, I/O overhead remained significant:
- `load id dicts`: ~161ms (tiny Parquet files, huge Arrow overhead)
- `load prices` + `process prices`: ~84ms (Arrow decode + block building)

**Total I/O overhead: ~245ms** - nearly half of the ~490ms total.

### Solution

Replace Parquet files with NumPy `.npy`/`.npz` format. Pre-compute block metadata during preprocessing instead of at runtime.

**Preprocessing script (`dev/preprocess_to_numpy.py`):**
```python
# Convert ticker/field dictionaries
ticker = pq.read_table("data/prices_ticker_dict.parquet")["ticker"].to_pylist()
np.save("data/prices_ticker_dict.npy", np.array(ticker, dtype="U100"))

# Convert prices with pre-built block metadata
t = pq.read_table("data/prices_keyed_sorted.parquet")
px_key = t["key"].to_numpy().astype(np.int32)
px_date = t["date"].to_numpy().astype("datetime64[D]").astype(np.int32)
px_value = t["value"].to_numpy().astype(np.float64)

# Build block metadata once (instead of at runtime)
chg = np.flatnonzero(px_key[1:] != px_key[:-1]) + 1
starts = np.r_[0, chg].astype(np.int32)
ends = np.r_[chg, len(px_key)].astype(np.int32)
block_of = np.full(max_key + 1, -1, dtype=np.int32)
block_of[keys] = np.arange(len(keys), dtype=np.int32)

np.savez("data/prices_keyed_sorted_np.npz",
         date=px_date, value=px_value,
         starts=starts, ends=ends, block_of=block_of)
```

**Runtime loading:**
```python
# ID dicts: direct numpy load
ticker_arr = np.load("data/prices_ticker_dict.npy", allow_pickle=False)
field_arr = np.load("data/prices_field_dict.npy", allow_pickle=False)

# Prices: load with pre-built metadata
pz = np.load("data/prices_keyed_sorted_np.npz", allow_pickle=False)
px_date = np.ascontiguousarray(pz["date"])
px_value = np.ascontiguousarray(pz["value"])
px_starts = np.ascontiguousarray(pz["starts"])
px_ends = np.ascontiguousarray(pz["ends"])
px_block_of = np.ascontiguousarray(pz["block_of"])
```

### Results

```
BEFORE (Parquet/Arrow):
  load id dicts:     ~161ms
  load prices:        ~35ms
  process prices:     ~49ms
  Total I/O:         ~245ms

AFTER (NumPy format):
  load id dicts:       0.4ms  (400x faster)
  load prices:        ~17ms   (includes pre-built blocks)
  Total I/O:          ~17ms

Savings:             ~228ms
```

### Implementation Notes

1. **Use `np.ascontiguousarray()`**: NPZ files return memory-mapped views. Without explicit copy, Numba kernels run ~4x slower due to non-contiguous memory access.

2. **Pre-build block metadata**: Moving `np.flatnonzero` and block array construction to preprocessing saves ~49ms at runtime.

3. **Removed pyarrow dependency**: The pipeline no longer requires `pyarrow.parquet` since Arrow joins were replaced earlier.

### Files Created

| File | Size | Contents |
|------|------|----------|
| `prices_ticker_dict.npy` | ~0.5MB | 4,646 tickers as U100 |
| `prices_field_dict.npy` | ~1KB | 7 fields as U100 |
| `prices_keyed_sorted_np.npz` | ~100MB | dates, values, block metadata |

### Final Timing Breakdown

```
loading yaml........:  36.6ms
processing amt......:   1.3ms
parse schedules.....:   1.0ms
load id dicts.......:   0.4ms  <- was 161ms
build id dicts......:   1.6ms
compute straddles...:  13.8ms
hedge tickers.......:  54.5ms
vol tickers (select):  28.4ms
precompute days.....:   0.6ms
load prices.........:  17.4ms  <- was 84ms (load+process)
map monthly ids.....:   6.7ms
sweep merge prep....:  18.2ms
sweep merge.........:  85.5ms
----------------------------------------
total:                274.5ms
```

---

## 12. Sweep Merge Optimization Analysis

### Problem

The sweep merge kernel was showing ~85ms in timing output, which seemed like a candidate for optimization. Proposed improvements included:
1. Run-based buffering (scan prices once, copy slices)
2. Lower bound binary search at group start
3. Remove redundant bounds checks

### Investigation

Created diagnostics (`dev/bench_merge_diagnostics.py`) to understand the workload:

```
Groups:           3,404
Straddles:        222,300
Total runs:       3,404 (all intervals in each group overlap)
Avg overlap:      5.4x
Avg run length:   559 days
Output cells:     15,394,866
```

### Discovery: Numba First-Call Overhead

Benchmarking revealed the 85ms is NOT algorithm time but Numba initialization:

```
Process 1 (cold cache):
  Run 1: 334ms  <- JIT compilation
  Run 2: 0.1ms

Process 2 (warm cache):
  Run 1: 70ms   <- Cache loading + OpenMP init
  Run 2: 0.1ms
```

The actual kernel execution is only **~2ms**, which is at the **memory bandwidth limit**:

```
Output array: 15,394,866 float64 = 123.2 MB
At 60 GB/s memory bandwidth: 2.1 ms theoretical minimum
Actual kernel time: ~2 ms
```

### Results

| Variant | Time (ms) | Notes |
|---------|-----------|-------|
| Original | 2.1 | Current implementation |
| With lower_bound | 2.1 | Binary search at group start |
| Run-based buffering | 2.1 | Buffer once, copy slices |
| Run-based v2 (no buffer) | 181 | Scan once, write all - SLOWER |

All algorithmic variants perform identically because the kernel is **I/O bound**, not compute bound.

### Conclusion

The sweep merge kernel is **already optimal** for this workload:
- Actual execution: ~2ms (memory bandwidth limited)
- Reported 85ms is Numba first-call overhead (cache load + OpenMP init)
- No algorithmic improvement possible

**Recommendations:**
- Keep current `_merge_per_key` implementation
- The ~85ms overhead is unavoidable per Python process
- For repeated runs within same process, kernel executes at ~2ms

### Files Created

| File | Purpose |
|------|---------|
| `bench_merge_diagnostics.py` | Workload analysis |
| `bench_merge_runs.py` | Variant comparison |
| `bench_merge_debug.py` | Timing investigation |
| `bench_first_call.py` | First-call overhead test |

---

## 13. Hedge Tickers Type-Dispatch Optimization

### Problem

The "hedge tickers" section was taking ~53ms. Initial hypothesis: futures ticker string construction (25K rows with 7.8x dedup potential) was the bottleneck.

### Investigation

**Breakdown of the 53ms:**
```
Build condition arrays:   0.8ms
hedge_ticker[smidx]:      3.1ms
np.flatnonzero:           0.1ms
Futures ticker build:     5.6ms
Full-size array alloc:    5.3ms
hedge_hedge/calc[smidx]:  6.5ms
np.select hedge1t/f:      9.4ms  <- expensive
np.select hedge2t/f:     11.7ms  <- expensive
np.select hedge3t/f:      5.9ms
np.where hedge4t:         5.4ms
--------------------------------
Total:                   53.1ms
```

**Key discovery:** The `np.select` calls were the major cost (~32ms total), not futures ticker construction (~6ms).

**Why np.select is slow:** It expands ALL choice arrays to full size (222K) before selecting, even though each hedge type uses only a fraction of rows:
- nonfut: 176,400 rows (79.4%)
- fut: 25,200 rows (11.3%)
- cds: 1,500 rows (0.7%)
- calc: 19,200 rows (8.6%)

### Solution: Type-Dispatch

Instead of:
```python
choices = [ticker_smidx, fut_ticker, hedge_smidx, calc_smidx]  # all 222K each
result = np.select(conditions, choices, default="")
```

Use index-based filling:
```python
result = np.empty(smlen, dtype=nps)
result.fill("")
result[nonfut_idx] = ticker[smidx[nonfut_idx]]   # 176K
result[fut_idx] = fut_ticker_m                    # 25K
result[cds_idx] = hedge[smidx[cds_idx]]          # 1.5K
result[calc_idx] = calc[smidx[calc_idx]]         # 19K
```

This only expands strings for rows that actually need them.

### Results

```
BEFORE:
  hedge tickers: 53ms

AFTER:
  hedge tickers: 26ms

Savings: 27ms (2x speedup)
```

### Rejected: Futures Ticker Dedup

Also tested packing futures components into int64 keys, deduping, building strings only for unique combinations:
- Total futures rows: 25,200
- Unique tickers: 3,236
- Dedup ratio: 7.8x
- **Result: Only ~1ms savings**

The overhead from `np.unique` calls (~6ms) ate most of the benefit from reduced string allocations.

### Files Created

| File | Purpose |
|------|---------|
| `bench_hedge_tickers.py` | Futures dedup benchmark |
| `bench_hedge_tickers_full.py` | Full section breakdown |
| `bench_hedge_tickers_opt.py` | Type-dispatch benchmark |
| `bench_np_select.py` | np.select alternatives |

---

## 14. ID-Only Pipeline Investigation

### Problem

The pipeline currently runs only 1 sweep merge (hedge1). A request was made to support all 5 leg values (hedge1-4 + vol). This investigation explores:
1. Can we eliminate string arrays entirely by computing keys directly?
2. What is the cost of running 5 sweep merges?

### Data Distribution (from bench_vol_tickers_full.py)

Vol source types at straddle level (N=222,300):
- **BBG**: 198,000 rows (89.1%)
- **BBG_LMEVOL**: 3,600 rows (1.6%)
- **CV**: 20,700 rows (9.3%)

Key insight: LMEVOL assets always have `fut` hedge source (100%), so R-ticker construction can reuse futures components.

### Benchmark 1: Vol Tickers Breakdown

```
Vol Tickers Section (~29ms):
  Build condition arrays:   0.1ms
  vol_ticker[smidx]:        4.2ms
  vol_near[smidx]:          5.7ms
  vol_far[smidx]:           4.6ms
  String concat (R-ticker): 4.8ms
  np.select volt_vec:       3.4ms
  np.select volf_vec:       4.1ms
  --------------------------------
  Total:                   28.9ms

Type-Dispatch Alternative:  20.0ms
Speedup:                    1.44x
Savings:                    8.9ms
```

### Benchmark 2: ID-Only Key Computation

Instead of building string arrays and then mapping to IDs, compute keys directly:

```
Current (all strings):      39.0ms  (builds 10 string arrays)
ID-only (all 5 keys):        3.6ms  (builds 5 int32 arrays)
Speedup:                   10.7x
Savings:                   35.4ms
```

**Key insight:** Integer operations are 10x faster than string materialization.

### Benchmark 3: Multi-Sweep Cost

Running 5 independent sweep merges:

```
Single Sweep (hedge1 only):
  Prep:    18.0ms (lexsort, 3,404 groups)
  Merge:    5.4ms (8.5M values found)
  Total:   23.3ms

5 Sequential Sweeps:
  Prep:    68.0ms (5 independent lexsorts)
  Merge:   37.2ms (5 independent merges)
  Total:  105.2ms

Per-leg breakdown:
  hedge1: 23.1ms (3,404 groups, 8.5M found)
  hedge2: 12.9ms (17 groups, 792K found) ← very sparse
  hedge3: 10.9ms (16 groups, 754K found) ← very sparse
  hedge4: 11.0ms (16 groups, 754K found) ← very sparse
  vol:    20.2ms (278 groups, 7.4M found)
```

**Key insight:** hedge2-4 have very few groups (16-17) because only CDS (1.5K rows) and calc (19.2K rows) types populate them.

### Cost-Benefit Analysis

**Option A: Current pipeline (hedge1 only)**
- Hedge tickers (strings): ~26ms
- Vol tickers (strings): ~27ms
- Map monthly IDs: ~7ms
- 1 sweep: ~23ms
- **Total relevant: ~83ms**

**Option B: Naive 5-leg (strings + 5 sweeps)**
- Hedge strings: ~26ms
- Vol strings: ~27ms
- Map monthly IDs (5 legs): ~35ms
- 5 sweeps: ~105ms
- **Total: ~193ms**

**Option C: ID-only 5-leg (keys + 5 sweeps)**
- Key computation: ~4ms
- 5 sweeps: ~105ms
- **Total: ~109ms**

**Savings: C vs B = ~84ms (44% faster)**

### Impact on Total Pipeline

| Scenario | String/Key Section | Sweep Section | Delta vs Current |
|----------|-------------------|---------------|------------------|
| Current (hedge1 only) | 60ms | 23ms | baseline |
| Naive 5-leg | 88ms | 105ms | +110ms |
| ID-only 5-leg | 4ms | 105ms | +26ms |

**Conclusion:** If all 5 leg values are needed, the ID-only approach adds ~26ms to the pipeline (243ms → ~269ms) but saves ~84ms compared to the naive string-based approach.

### Recommended Implementation

1. **Extend asset-level ID precomputation** (add hedge_hedge1_tid, calc_hedge2-4_tid, vol IDs)
2. **Replace hedge tickers section** with direct key computation
3. **Replace vol tickers section** with direct key computation
4. **Add sweep_leg helper function** for clean per-leg sweep merge
5. **Run 5 sweep merges** in sequence

### Files Created

| File | Purpose |
|------|---------|
| `bench_vol_tickers_full.py` | Vol tickers breakdown + type-dispatch test |
| `bench_id_only.py` | ID-only key computation benchmark |
| `bench_multi_sweep.py` | Multi-sweep cost analysis |

---

## 15. Fused Sweep Investigation

### Problem

The ID-only approach (Section 14) requires 5 sequential sweeps, each with its own lexsort. Since all 5 legs share identical date intervals (month_start_epoch, month_len, month_out0), can we exploit this with a single fused sweep?

### Approach

**Sequential (filter-first):** Filter valid keys first, then 5 independent lexsorts
- Critical optimization: sort only the filtered subset (huge win for sparse legs)
- 5 separate output arrays

**Fused:** 1 lexsort + 1 merge kernel call with leg_id
- Concatenate all valid intervals with leg_id marker
- Single lexsort on combined data
- Modified kernel writes to 2D output based on leg_id

### Benchmark Results (Corrected)

Initial benchmark used sort-then-filter, which unfairly penalized sequential for sparse legs.
**Corrected benchmark uses filter-first:**

```
Data: 222,300 straddles (benchmark snapshot; production has 15.4M daily rows)

Sequential (filter-first, 5 independent):
  Prep (5 lexsorts):  39.5ms
  Merge (5 kernels):  40.0ms
  TOTAL:              79.5ms

  Per-leg breakdown (filter-first is crucial for sparse legs):
    hedge1: 20.0ms (100% valid)
    hedge2:  1.6ms (9% valid)   ← 7x faster than sort-then-filter!
    hedge3:  1.3ms (8.6% valid) ← 8x faster!
    hedge4:  1.5ms (8.6% valid) ← 7x faster!
    vol:    18.3ms (98% valid)

Fused (1 unified):
  Prep (1 lexsort):   49.5ms  (499,500 intervals, 3,731 groups)
  Merge (1 kernel):   43.1ms
  TOTAL:              92.6ms

Result: Sequential is 13ms FASTER than fused!
```

### Analysis

With filter-first sequential:
- **Sparse legs are fast:** hedge2-4 sort only ~20K rows instead of 222K
- **Fused has overhead:** Single lexsort on 500K combined rows is slower than 5 small sorts
- **Merge 2D writes are slower:** 2D array indexing adds cache pressure

### Conclusion

**Sequential filter-first is the clear winner:**
- 13ms faster than fused (79.5ms vs 92.6ms)
- Simpler implementation (no 2D output, no leg_id array)
- Naturally exploits sparsity of hedge2-4

**Recommendation:** Use sequential filter-first approach. No benefit to fused sweep.

### Files Created

| File | Purpose |
|------|---------|
| `bench_fused_sweep.py` | Fused vs sequential sweep comparison (filter-first corrected) |

---

## 16. ID-Only Pipeline Implementation (Shipped)

### Summary

The ID-only pipeline was implemented in `dev/backtest_standalone.py`, eliminating all string materialization for hedge/vol tickers by computing integer keys directly. This enables computing all 5 leg values (hedge1-4 + vol) for approximately 2x the cost of the previous single-leg implementation.

### Implementation Details

**Step 1: Asset-level IDs** (line ~395)
```python
# Hedge IDs for legs 2-4
hedge_hedge1_tid = np.array([ticker_to_id.get(str(s), -1) for s in hedge_hedge1], dtype=np.int32)
calc_hedge2_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge2], dtype=np.int32)
calc_hedge3_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge3], dtype=np.int32)
calc_hedge4_tid  = np.array([ticker_to_id.get(str(s), -1) for s in calc_hedge4], dtype=np.int32)

# Vol IDs - dual mapping (BBG: near/far as fields, CV: near as ticker)
vol_ticker_tid = np.array([ticker_to_id.get(str(s), -1) for s in vol_ticker], dtype=np.int32)
vol_near_fid   = np.array([field_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)
vol_far_fid    = np.array([field_to_id.get(str(s), -1) for s in vol_far], dtype=np.int32)
vol_near_tid   = np.array([ticker_to_id.get(str(s), -1) for s in vol_near], dtype=np.int32)
```

**Step 2: Key computation** (line ~467)

Replaced hedge/vol string sections with direct integer key computation:
```python
hedge1_key[nonfut_idx] = hedge_ticker_tid_smidx[nonfut_idx] * n_fields + hedge_field_fid_smidx[nonfut_idx]
hedge1_key[fut_idx]    = hedge_fut_tid_smidx[fut_idx]       * n_fields + PX_LAST_FID
hedge1_key[cds_idx]    = hedge_hedge_tid_smidx[cds_idx]     * n_fields + PX_LAST_FID
hedge1_key[calc_idx]   = calc1_tid_smidx[calc_idx]          * n_fields + EMPTY_FID
# ... similar for hedge2-4 and vol
```

**Step 3: Filter-first sweep_leg helper** (line ~188)

Critical optimization: filter valid keys BEFORE sorting (7-8x faster for sparse legs):
```python
def sweep_leg(label, key_i32, month_start_epoch, month_len, month_out0, ...):
    # Filter-first: key in range AND has price data
    valid = (key_i32 >= 0) & (key_i32 <= max_valid_key)
    valid_keys = key_i32[valid]
    valid[valid] &= (px_block_of[valid_keys] >= 0)  # Only keys with actual price data

    # Sort only the filtered subset (huge win for sparse legs like hedge2-4)
    k = key_i32[valid].astype(np.int32)
    order = np.lexsort((s, k))
    # ... merge
```

**Step 4: 5 sequential sweeps** (line ~653)
```python
d_hedge1_value = sweep_leg("sweep hedge1", hedge1_key, ...)
d_hedge2_value = sweep_leg("sweep hedge2", hedge2_key, ...)
d_hedge3_value = sweep_leg("sweep hedge3", hedge3_key, ...)
d_hedge4_value = sweep_leg("sweep hedge4", hedge4_key, ...)
d_vol_value    = sweep_leg("sweep vol",    vol_key,    ...)
```

### Results (Warmed Cache)

```
hedge/vol keys (ids): 19.5ms
sweep hedge1........: 111.2ms (10,043,445 found)
sweep hedge2........:  16.4ms (803,425 found)  ← sparse, filter-first fast
sweep hedge3........:  16.1ms (764,680 found)  ← sparse, filter-first fast
sweep hedge4........:  16.8ms (764,680 found)  ← sparse, filter-first fast
sweep vol...........:  33.8ms (8,416,107 found)
----------------------------------------
total: 295.6ms
```

### Key Insights

1. **Filter-first is critical:** Sparse legs (hedge2-4 are only ~9% valid) benefit enormously from sorting only the filtered subset.

2. **5 legs for ~2x cost of 1:** Previous single-leg pipeline was ~150ms for hedge1 only. New 5-leg pipeline is ~296ms for all 5 values.

3. **LMEVOL ⊆ FUT invariant:** LMEVOL assets always have `fut` hedge source, so R-ticker construction reuses futures components.

4. **String elimination:** No more string arrays for hedge1t/f through volt/f. All replaced by integer keys.

### Comparison

| Scenario | Key/String Section | Sweep Section | Total | Legs |
|----------|-------------------|---------------|-------|------|
| Old (strings, hedge1 only) | ~83ms | ~23ms | ~150ms | 1 |
| New (ID-only, all 5) | ~20ms | ~194ms | ~296ms | 5 |

**Net result:** 5 legs for ~2x cost. Per-leg cost dropped from ~150ms to ~59ms.

### Debug Features

- LMEVOL ⊆ FUT invariant check: `DEBUG = True` enables assertion
- `px_block_of[key] >= 0` filter: Skips keys with no price data

---

## Benchmark Files

Created during experimentation:

| File | Purpose |
|------|---------|
| `bench_numba_lookup.py` | Numba vs Arrow comparison |
| `bench_numba_presorted.py` | Pre-sorted price performance |
| `bench_numba_realistic.py` | Real query pattern analysis |
| `bench_bandwidth_real.py` | Cache bandwidth analysis |
| `bench_packed_keys.py` | Packed key approaches |
| `bench_merge_join.py` | Sort + merge join |
| `bench_merge_join_v2.py` | Query order analysis |
| `bench_price_driven.py` | Hash table approach |
| `bench_inverted_index.py` | Inverted index + merge |
| `bench_sweep_a.py` | Key-grouped merge |
| `bench_sweep_b.py` | Sweep-line comparison |
| `preprocess_to_numpy.py` | Parquet → NumPy conversion |
| `bench_merge_diagnostics.py` | Sweep merge workload analysis |
| `bench_merge_runs.py` | Sweep merge variant comparison |
| `bench_merge_debug.py` | Timing investigation |
| `bench_first_call.py` | Numba first-call overhead test |
| `bench_hedge_tickers.py` | Futures dedup benchmark |
| `bench_hedge_tickers_full.py` | Hedge tickers breakdown |
| `bench_hedge_tickers_opt.py` | Type-dispatch benchmark |
| `bench_np_select.py` | np.select alternatives |
| `bench_vol_tickers_full.py` | Vol tickers breakdown |
| `bench_id_only.py` | ID-only key computation benchmark |
| `bench_multi_sweep.py` | Multi-sweep cost analysis |
| `bench_fused_sweep.py` | Fused vs sequential sweep comparison |
| `backtest_standalone.py` | **Main implementation** (ID-only 5-leg pipeline) |

---

## Recommendations

### Implemented (All Major Wins)
1. Numba parallel binary search (10x faster joins)
2. Asset-level ID precomputation (38x faster ID mapping)
3. int32 for d_stridx (~10% faster expansion)
4. **Key-grouped merge sweep** (4.1x speedup for lookup pipeline)
5. Skip expand days in sweep mode (~95ms saved)
6. Searchsorted for ticker ID mapping (~2ms saved)
7. **NumPy format** (~228ms saved, removed Arrow/Parquet overhead)
8. ~~Hedge tickers type-dispatch~~ (superseded by ID-only)
9. **ID-only pipeline with 5 legs** (all hedge1-4 + vol for ~2x cost of single leg)
10. **Filter-first sweep** (7-8x faster for sparse legs like hedge2-4)

### Not Recommended
- Packed keys: No benefit over block-based search
- Merge join with sorting: Sort overhead too high
- Price-driven hash: Python overhead too high
- Sweep-line (B): Slightly slower than A
- Single-pass asset extraction: N=189 too small for measurable benefit
- Per-field np.unique: No faster than 3D unique for this data
- Further "processing amt" optimization: Bottleneck is interpreter overhead, not algorithm
- **Sweep merge optimization**: Kernel already at memory bandwidth limit (~2ms actual execution); reported ~85ms is Numba first-call overhead
- **Futures ticker dedup**: Packed int64 keys + dedupe saves only ~1ms; `np.unique` overhead too high
- **Fused sweep**: Sequential filter-first is 13ms faster than fused approach

### Current Performance (5-leg ID-only pipeline)

At **~296ms total** (warmed cache), the pipeline computes all 5 leg values:
```
loading yaml........:  37ms
processing amt......:   1ms
parse schedules.....:   1ms
load id dicts.......:   0.5ms
build id dicts......:   2ms
compute straddles...:  12ms
hedge/vol keys (ids):  19ms  ← replaced 53ms strings
precompute days.....:   1ms
load prices.........:  17ms
sweep hedge1........: 111ms (10M found)
sweep hedge2........:  16ms (803K found) ← sparse, fast
sweep hedge3........:  16ms (765K found) ← sparse, fast
sweep hedge4........:  17ms (765K found) ← sparse, fast
sweep vol...........:  34ms (8.4M found)
----------------------------------------
total:              ~296ms
```

### Performance Comparison

| Configuration | Total | Legs | Notes |
|---------------|-------|------|-------|
| Original (before optimizations) | ~748ms | 1 | Arrow joins, strings |
| After all optimizations (hedge1 only) | ~243ms | 1 | Type-dispatch, NumPy format |
| **Current (ID-only, all 5 legs)** | **~296ms** | **5** | **Filter-first sweeps** |

**Key insight:** Getting all 5 leg values costs only ~53ms more than the previous single-leg pipeline. The ID-only approach with filter-first sweeps makes multi-leg extraction practical.

### Performance Floor

The current ~312ms (with action detection) is near its practical floor:
- I/O: ~55ms (YAML + prices)
- Key computation: ~19ms
- Anchor computation: ~18ms (LUTs + override matrix + anchor dates)
- Merge kernels: ~170ms (memory bandwidth limited after Numba warmup)
- Action detection: ~11ms (parallel kernel)

**Note on Numba warmup:** First run in a new Python process takes ~1300ms due to JIT compilation. Subsequent runs are at ~312ms. Kernel execution is ~2ms per leg (memory bandwidth limited at 60 GB/s).

---

## Phase 2-3: Action Detection (Entry/Expiry)

### Goal

Compute `action` array (int8[N]) where:
- `0` = no action
- `1` = entry (ntry)
- `2` = expiry (xpry)

Also outputs `ntry_offsets[S]` and `xpry_offsets[S]` (offset within straddle, -1 if not found).

### Key Optimizations

1. **No `dates[N]` array** - Compute inline: `d = month_start_epoch[s] + i`
   - Saves 15.4M × 4 bytes = 62MB allocation
   - Avoids cache pressure from large array access
2. **Month-precomputed anchor LUTs** - ~324 months, not 222K straddles
   - BD anchors: loop (cumulative counting)
   - F/R/W anchors: vectorized (0.04ms)
3. **Validity = key exists** - `req_vol = (vol_key >= 0)`, not n_hedges

### xprc Distribution

| Type | % | Method |
|------|---|--------|
| F (Friday) | 45.3% | Vectorized LUT |
| BD (Business Day) | 41.0% | Loop LUT |
| OVERRIDE | 11.9% | CSV lookup |
| R (Thursday) | 1.1% | Vectorized LUT |
| W (Wednesday) | 0.7% | Vectorized LUT |

### Implementation

```
Step 1: Anchor LUTs (324 months)
  - bd_anchor[month_idx, n]: Nth business day (n=1..23)
  - fri/thu/wed_anchor[month_idx, n]: Nth weekday (n=1..5)

Step 2: Anchor dates per straddle
  - Vectorized LUT lookup by xprc type
  - OVERRIDE: separate entry/expiry month lookups from CSV

Step 3: Month ends
  - ntry_month_end = month_start_epoch + entry_days - 1
  - xpry_month_end = xpry_month_start + days0_vec - 1

Step 4: Numba kernel (parallel)
  - Find entry: first valid day >= target in entry month
  - Fallback: last valid day in entry month
  - Find expiry: first valid day >= target, must be >= ntry_off
```

### OVERRIDE Optimization

Initial implementation had slow Python loop for OVERRIDE lookups:
```python
# Slow: Python loop with string formatting + dict lookup
for i in idx_ovr:
    k_entry = f"{smidx[i]}_{ntry_mi[i]}"
    k_xpry = f"{smidx[i]}_{xpry_mi[i]}"
    ntry_anchor[i] = override_dates.get(k_entry, INVALID)
    xpry_anchor[i] = override_dates.get(k_xpry, INVALID)
```

**Optimization:** Build dense matrix once, vectorized lookup:
```python
# Build override_epoch[n_assets, N_MONTHS] matrix once (~8ms)
override_epoch = np.full((n_assets, N_MONTHS), INVALID, dtype=np.int32)
for key, epoch in override_dates.items():
    smid, mi = key.split("_")
    override_epoch[int(smid), int(mi)] = epoch

# Vectorized lookup using smidx (integer asset index)
idx_ovr = np.flatnonzero(xprc_code == XPRC_OVERRIDE)
aid = smidx[idx_ovr].astype(np.int32)
mi_entry = ntry_mi[idx_ovr]
e_entry = np.where(valid_entry, override_epoch[aid, np.clip(mi_entry, 0, N_MONTHS-1)], INVALID)
ntry_anchor[idx_ovr[ok_entry]] = e_entry[ok_entry]
```

**Results:**
```
anchor dates (before): 47.6ms  (Python loop)
anchor dates (after):   5.3ms  (vectorized)
override matrix:        8.6ms  (one-time build)
Net savings:           33.7ms
```

### Parallel vs Serial Kernel

Tested both parallel and serial `compute_actions` kernel:
```
Parallel (@njit(parallel=True) + prange):  11.2ms
Serial (@njit + range):                    26.2ms
```

**Conclusion:** Parallel is 2.3x faster despite short inner loops (~60-90 days per straddle). Thread overhead is amortized across 222K straddles.

### Benchmark Results (Final)

```
anchor LUTs.........:  4.1ms
override matrix.....:  8.6ms  (one-time build)
anchor dates........:  5.3ms  (vectorized OVERRIDE)
compute actions.....: 11.2ms  (parallel kernel)

Results:
  ntry found: 165,913
  xpry found: 165,913
  valid straddles: 165,913 / 222,300 (74.6%)
```

**Note:** ~25% of straddles have INVALID anchors due to missing OVERRIDE CSV data.

### Updated Pipeline Summary

| Stage | Time | Notes |
|-------|------|-------|
| YAML + prices | ~55ms | I/O |
| Key computation | ~19ms | ID-only |
| Anchor LUTs | ~4ms | 324 months |
| Override matrix | ~9ms | One-time build |
| Anchor dates | ~5ms | Vectorized OVERRIDE |
| Sweep merge (5 legs) | ~170ms | Filter-first |
| Compute actions | ~11ms | Numba parallel |
| **Total** | **~312ms** | **+16ms for actions** |

---

## Appendix: Data Characteristics

```
Assets:            189
Monthly straddles: 222,300
Daily straddles:   15,394,866
Prices:            8,489,913
Unique keys:       4,850
Straddles/key:     avg 1,173, max 1,200
Prices/key:        avg 1,750

5-Leg Output (ID-only pipeline):
  hedge1: 10,043,445 found (100% of straddles have hedge1)
  hedge2:    803,425 found (~9% - CDS + calc types only)
  hedge3:    764,680 found (~9% - calc types only)
  hedge4:    764,680 found (~9% - calc types only)
  vol:     8,416,107 found (~98% of straddles have vol)
```
