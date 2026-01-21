# Memoization and Caching for Backtest Performance

This document details the caching strategies implemented to optimize backtest performance, the experiments conducted, and conclusions reached.

## Problem Statement

Running backtests across many years (e.g., 20 years with ~8,892 straddles/year = 177,840 straddles) was taking approximately 1 hour. The goal was to reduce this time significantly through caching and parallelization.

## Caching Strategies Considered

### 1. Standard Python Memoization (`functools.lru_cache`)

**Conclusion: Works for per-worker caching, but manual dicts preferred**

- macOS uses `spawn` (not `fork`) to create worker processes
- Each worker gets its own memory space
- Cached results in one process are not visible to others
- `lru_cache` decorators work fine within a single worker process
- Since each worker processes ~11,000 straddles sequentially, `lru_cache` would provide the same benefit as manual dicts

**Why we use manual dicts instead:**
- Easier to toggle on/off with `_MEMOIZE_ENABLED` flag
- Easier to clear caches explicitly (`clear_caches()` function)
- More explicit control over cache key construction
- Can inspect cache contents for debugging

### 2. Cross-Process Caching Options

Considered but rejected for this use case:
- `joblib.Memory` - disk-based caching, adds I/O overhead
- `diskcache` package - similar disk-based approach
- Redis/memcached - overkill for local backtesting

### 3. Per-Worker Manual Caches (Implemented)

**Conclusion: Best approach for this use case**

Each worker process maintains its own cache dictionaries. Since workers process multiple straddles sequentially, they benefit from cached data for repeated lookups within their assigned work.

## Implemented Caches

### In `tickers.py`

#### 1. `_TSCHEMAS_CACHE`
```python
_TSCHEMAS_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
```
- **Function**: `get_tschemas(path, underlying)`
- **Key**: `(resolved_path, underlying)`
- **Benefit**: Ticker schema lookups are expensive (parsing asset data, building ticker lists). Same underlying is queried multiple times across straddles.

#### 2. `_TICKERS_YM_CACHE`
```python
_TICKERS_YM_CACHE: dict[tuple[str, str, int, int, str | None], dict[str, Any]] = {}
```
- **Function**: `get_tickers_ym(path, asset, year, month, chain_csv)`
- **Key**: `(resolved_path, asset, year, month, resolved_chain_path)`
- **Benefit**: Ticker expansion for year/month involves date constraint parsing and futures ticker computation. Same asset/month combinations appear across multiple straddles.

#### 3. `_DUCKDB_CACHE`
```python
_DUCKDB_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}
```
- **Function**: `_get_prices_connection(prices_parquet)`
- **Key**: `resolved_path`
- **Benefit**: DuckDB connection creation is expensive. Reusing connections across queries within a worker eliminates this overhead.

#### 4. `_OVERRIDE_CACHE`
```python
_OVERRIDE_CACHE: dict[tuple[str, str], str] | None = None
```
- **Function**: `_load_overrides(path)`
- **Benefit**: Override CSV is loaded once per worker and cached.

### In `schedules.py`

#### 5. `_SCHEDULE_CACHE`
```python
_SCHEDULE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
```
- **Function**: `get_schedule(path, underlying)`
- **Key**: `(resolved_path, underlying)`
- **Benefit**: Schedule lookups involve parsing AMT data and building schedule rows. Same underlying is queried for every straddle of that asset.

#### 6. `_EXPAND_YM_CACHE`
```python
_EXPAND_YM_CACHE: dict[tuple[str, str, int, int], dict[str, Any]] = {}
```
- **Function**: `get_expand_ym(path, underlying, year, month)`
- **Key**: `(resolved_path, underlying, year, month)`
- **Benefit**: Straddle expansion for a given month is reused when multiple straddles share the same expiry month.

### In `loader.py` (Pre-existing)

#### 7. `_AMT_CACHE`
- **Function**: `load_amt(path)`
- **Benefit**: YAML parsing is expensive. File loaded once per worker.

#### 8. `_ASSET_BY_UNDERLYING`
- **Function**: `get_asset(path, underlying)`
- **Benefit**: Asset lookup by underlying is O(1) after initial indexing.

## Multiprocessing Architecture

### Worker Pool with Initialization

```python
from multiprocessing import Pool, cpu_count

def init_worker(amt, prices, chain):
    """Initialize worker process by warming up caches."""
    global _worker_amt, _worker_prices, _worker_chain
    _worker_amt = amt
    _worker_prices = prices
    _worker_chain = chain

    # Warm up caches
    loader.load_amt(amt)  # Cache AMT file
    _get_prices_connection(prices)  # Cache DuckDB connection

with Pool(
    processes=num_workers,
    initializer=init_worker,
    initargs=(args.amt, args.prices, args.chain)
) as pool:
    for result in pool.imap_unordered(process_straddle, tasks):
        # Process results...
```

### Key Design Decisions

1. **`imap_unordered`**: Returns results as they complete, not in input order. Improves throughput.

2. **Worker initialization**: Pre-loads AMT file and creates DuckDB connection before processing starts. Eliminates cold-start overhead.

3. **Global variables in workers**: File paths stored as globals, accessed by `process_straddle`. Avoids passing large data through multiprocessing serialization.

4. **Result collection and sorting**: Results collected out-of-order, then sorted by (asset, straddle, date) for consistent output.

## Performance Experiments

### Test Environment
- macOS (Darwin 24.5.0)
- 16 CPU cores available
- DuckDB for price queries against parquet files

### Experiment 1: Single Asset, Single Year (96 straddles)

```bash
time uv run python scripts/backtest.py '^LA Comdty' 2025 2025 --verbose
```

**Results:**
- Time: ~1.6 seconds
- CPU utilization: 1117%
- Rate: ~60 straddles/second

### Experiment 2: Single Asset, 6 Years (576 straddles)

```bash
time uv run python scripts/backtest.py '^LA Comdty' 2020 2025
```

**Results:**
- Time: ~5.1 seconds
- Rate: ~113 straddles/second

### Experiment 3: 3 Assets, 2 Years (384 straddles)

```bash
time uv run python scripts/backtest.py '^(LA|CL|NG) Comdty' 2024 2025
```

**Results:**
- Time: ~4.6 seconds
- Output: 11,691 rows (days of valuation data)
- Rate: ~83 straddles/second

### Extrapolation for Full Backtest

For 177,840 straddles (20 years × 8,892/year):
- At 83 straddles/second: **~36 minutes**
- Previous estimate: ~1 hour
- **Improvement: ~40% reduction in time**

## Why Per-Worker Caching Works

Even though caches aren't shared between workers, they still provide significant benefit:

1. **Work distribution**: Each worker processes many straddles. With 16 workers and 177,840 straddles, each worker handles ~11,000 straddles.

2. **Locality**: Tasks are distributed such that the same asset/month combinations tend to cluster. A worker processing "LA Comdty 2024-03" straddles will cache that data and reuse it.

3. **Hierarchical caching**: Lower-level caches (tschemas, schedules) benefit higher-level functions. `get_straddle_days` calls `asset_straddle_tickers` which calls `get_tickers_ym` which calls `get_tschemas`. Each level benefits from caching.

## Cache Key Design

All caches use resolved absolute paths to ensure consistency:

```python
path_str = str(Path(path).resolve())
cache_key = (path_str, underlying)
```

This handles:
- Relative vs absolute paths
- Symlinks
- Different working directories

## Memory Considerations

Each worker maintains its own cache. For a typical backtest:
- AMT data: ~1-5 MB per worker
- DuckDB connection: minimal (connection object, not data)
- Ticker/schedule caches: ~100KB-1MB per worker (depends on number of unique assets/months)

With 16 workers: **~100-200 MB total** - acceptable for modern systems.

## Future Optimization Opportunities

### 1. Batch DuckDB Queries

Currently, each straddle makes one DuckDB query. Could potentially batch multiple straddles' price queries into fewer, larger queries.

**Trade-off**: Increased complexity, may not help if straddles have different date ranges.

### 2. Pre-compute All Straddles

Instead of workers computing straddles on-demand, pre-expand all straddles and distribute the list.

**Status**: Already implemented - `schedules.expand()` runs once in main process, results distributed to workers.

### 3. Shared Memory for Read-Only Data

Use `multiprocessing.shared_memory` for large read-only data structures (e.g., override lookups).

**Trade-off**: Added complexity, likely minimal benefit given current cache sizes.

## Conclusion

The combination of:
1. **Multiprocessing** with worker pool
2. **Per-worker caching** for expensive operations
3. **Worker initialization** to warm up critical caches

Achieves approximately **40% reduction in backtest time** compared to the naive approach, bringing a 1-hour backtest down to ~36 minutes.

The per-worker caching strategy is simpler than cross-process caching solutions and provides sufficient benefit for this use case where:
- Each worker processes many items
- There is natural locality in the data (same assets, similar date ranges)
- The cached data is relatively small (MBs, not GBs)
