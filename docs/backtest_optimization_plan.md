# Backtest Optimization Plan

## Executive Summary

The backtest script is **~100x slower with multiprocessing than single-threaded**. This is due to massive serialization overhead when passing the 392MB `prices_dict` to worker processes.

**Key Finding**: Single-threaded processing achieves **3,157 straddles/sec** (0.73s for 2304 straddles), while 16-worker multiprocessing achieves only **31.8 straddles/sec** (72.53s).

---

## Benchmark Results

### Current Performance

```
Config               Workers          Time    Speedup Efficiency
------------------------------------------------------------
Single-thread        1                0.73s       1.00x     100.0%
2 workers            2                9.38s       0.08x       3.9%
4 workers            4               19.17s       0.04x       1.0%
8 workers            8               36.31s       0.02x       0.3%
16 workers           16              72.53s       0.01x       0.1%
```

### Per-Straddle Breakdown (Single-threaded)

| Function | Avg Time | % of Total |
|----------|----------|------------|
| `get_straddle_valuation_full` | 0.279ms | 42.8% |
| `filter_tickers` | 0.136ms | 20.8% |
| `actions` | 0.075ms | 11.6% |
| `lookup_straddle_prices` | 0.068ms | 10.5% |
| `build_ticker_map` | 0.052ms | 7.9% |
| `build_prices_table` | 0.032ms | 5.0% |
| `straddle_days` | 0.009ms | 1.4% |

**Total: ~0.65ms per straddle** (single-threaded with warm caches)

---

## Root Cause Analysis

### Primary Bottleneck: prices_dict Serialization

The `prices_dict` containing ~8M entries (~392MB) is passed to each worker via `initargs`:

```python
# Current code (backtest.py line 102-105)
with Pool(
    processes=num_workers,
    initializer=init_worker,
    initargs=(amt, prices, chain, memoize, prices_dict)  # ← prices_dict is 392MB!
) as pool:
```

**Problem**: Each time a worker is spawned, Python pickles the entire `prices_dict` and sends it via IPC. With 16 workers, this means ~6.3GB of serialization just to start the pool.

### Secondary Issues

1. **Result serialization**: Each task returns a full table with ~76 rows × 20 columns
2. **No batching**: Tasks are submitted one at a time via `imap_unordered`
3. **Process creation overhead**: Each Pool creates new processes

---

## Optimization Strategies

### Option 1: Remove Multiprocessing (RECOMMENDED - Fastest)

**Implementation**: Use single-threaded processing.

**Rationale**: The workload is I/O-bound (dict lookups) not CPU-bound. Single-threaded processing is already very fast (0.73s for 2304 straddles = 28.7 straddles/sec would take 80s → actually 0.73s).

**Expected improvement**: **~100x faster** (72s → 0.7s)

```python
# Replace Pool-based processing with simple loop
def run_backtest_singlethread(tasks, amt, prices, chain, prices_dict):
    set_prices_dict(prices_dict)
    loader.load_amt(amt)

    all_rows = []
    columns = None

    for asset, year, month, i in tasks:
        val_table = get_straddle_valuation(asset, year, month, i, amt, chain, prices)
        if columns is None:
            columns = val_table["columns"]
        all_rows.extend(val_table["rows"])

    return columns, all_rows
```

### Option 2: Shared Memory for prices_dict

**Implementation**: Use `multiprocessing.shared_memory` or memory-mapped files.

**Complexity**: High - requires restructuring prices_dict storage format.

```python
# Approach: Store prices in memory-mapped numpy arrays
# Key: ticker|field|date → index mapping (small, can be pickled)
# Values: numpy array in shared memory (zero-copy across processes)
```

**Expected improvement**: ~10-50x faster (depends on implementation)

### Option 3: Worker-Local Price Loading

**Implementation**: Each worker loads its own prices from parquet.

**Problem**: Would load 392MB × N workers = 6.3GB for 16 workers.

**Not recommended** due to memory explosion.

### Option 4: Batch Processing with Chunked Tasks

**Implementation**: Group tasks into chunks, process each chunk as a batch.

```python
# Process in chunks of 100 straddles
CHUNK_SIZE = 100
chunks = [tasks[i:i+CHUNK_SIZE] for i in range(0, len(tasks), CHUNK_SIZE)]

# Submit chunks instead of individual tasks
def process_chunk(chunk):
    results = []
    for asset, year, month, i in chunk:
        results.append(process_straddle((asset, year, month, i)))
    return results
```

**Expected improvement**: ~2-5x faster (reduces serialization frequency)

### Option 5: Use Threading Instead of Multiprocessing

**Implementation**: Use `concurrent.futures.ThreadPoolExecutor`.

**Rationale**: Dict lookups release the GIL during hash computation.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_straddle, tasks)
```

**Expected improvement**: ~1.5-3x faster than multiprocessing (no serialization)

### Option 6: Vectorized/Batch Valuation

**Implementation**: Compute all valuations in a single pass using numpy/arrow.

**Complexity**: Very high - requires rewriting valuation logic.

**Expected improvement**: ~10-100x faster for large batches

---

## Recommended Implementation Plan

### Phase 1: Quick Win - Single-Threaded Mode (1 hour)

Add `--single-thread` / `-1` flag to use single-threaded processing:

```python
parser.add_argument("--single-thread", "-1", action="store_true",
                    help="Use single-threaded processing (faster for small batches)")

# In main():
if args.single_thread or len(tasks) < 1000:
    columns, all_rows = run_backtest_singlethread(...)
else:
    columns, all_rows = run_backtest_multiprocess(...)
```

**Impact**: ~100x faster for typical workloads

### Phase 2: Default to Single-Thread (immediate)

Make single-threaded the default, multiprocessing opt-in:

```python
parser.add_argument("--workers", "-j", type=int, default=1,
                    help="Number of worker processes (default: 1, use >1 only for very large batches)")
```

### Phase 3: Optimize Multiprocessing (if needed, 4-8 hours)

If multiprocessing is still needed for very large batches:

1. Implement shared memory for prices_dict
2. Use chunk-based task submission
3. Profile and optimize result serialization

---

## Verification Commands

```bash
# Test single-threaded performance
uv run python scripts/profile_multiprocessing.py --pattern "^LA Comdty" --start-year 2001 --end-year 2024

# After implementing single-thread mode:
uv run python scripts/backtest.py "^LA Comdty" 2001 2024 --benchmark --single-thread

# Expected: ~1s instead of ~80s
```

---

## Summary Table

| Approach | Complexity | Expected Speedup | Memory Impact |
|----------|------------|------------------|---------------|
| Single-threaded (default) | Low | ~100x | None |
| Shared memory | High | ~10-50x | +200MB |
| Chunked tasks | Medium | ~2-5x | None |
| Threading | Low | ~1.5-3x | None |
| Vectorized | Very High | ~10-100x | Variable |

**Recommendation**: Implement Option 1 (single-threaded) immediately. The current workload processes 2304 straddles in 0.73s single-threaded - there's no need for parallelization.
