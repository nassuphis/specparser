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
| Single-pass asset extraction + per-field unique | 0ms | Rejected (no improvement) |

**Final pipeline performance:**
- Before optimizations: ~748ms
- After algorithmic optimizations: ~490ms (34% faster)
- After NumPy format: **~275ms** (63% faster total)

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
Pre-map ticker→ID at asset level (74 assets) during startup, then use integer `np.select`:

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

### Not Recommended
- Packed keys: No benefit over block-based search
- Merge join with sorting: Sort overhead too high
- Price-driven hash: Python overhead too high
- Sweep-line (B): Slightly slower than A
- Single-pass asset extraction: N=189 too small for measurable benefit
- Per-field np.unique: No faster than 3D unique for this data
- Further "processing amt" optimization: Bottleneck is interpreter overhead, not algorithm

### Performance Floor
At **~275ms total**, the pipeline is near its practical floor given:
- YAML loading: ~37ms (unavoidable)
- Processing amt + schedules: ~2ms
- Load/build ID dicts: ~2ms
- Compute straddles: ~14ms
- Hedge/vol tickers: ~83ms
- Load prices: ~17ms (NumPy format with pre-built blocks)
- Sweep merge: ~104ms (prep + merge)

The lookup pipeline (sweep merge) is 4.1x faster than the original expand+binary search approach. The NumPy format optimization removed ~228ms of Arrow/Parquet overhead. Remaining time is dominated by hedge/vol ticker string operations (~83ms) and the sweep merge kernel (~104ms).

---

## Appendix: Data Characteristics

```
Assets:           189
Monthly straddles: 222,300
Daily straddles:   13,523,250
Prices:           8,489,913
Unique keys:       4,850
Straddles/key:     avg 1,173, max 1,200
Prices/key:        avg 1,750
```
