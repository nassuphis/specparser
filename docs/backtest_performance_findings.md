# Backtest Performance Findings

This document captures performance analysis findings from profiling the `full_backtest_kernel_sorted()` pipeline.

## Executive Summary

**Key Finding:** The Numba kernel itself is fast (0.25s). The bottleneck was Python preparation phases.

| Phase | Original Time | Optimized Time | Savings |
|-------|--------------|----------------|---------|
| Phase 1+2 - Find & expand straddles | 2.57s | 1.27s | **1.3s (50%)** |
| Phase 3 - Resolve tickers | 3.46s | 3.46s | - |
| Phase 3b - Prepare arrays | 11.96s | 1.41s | **10.5s (88%)** |
| Phase 4 - KERNEL | 0.25s | 0.25s | - |
| **Total** | **~19s** | **~9s** | **54%** |

The kernel is NOT the bottleneck. 98% of time was in Python preparation.

---

## Key Insight: Massive Redundancy

| Entity | Total Calls | Unique Count | Redundancy |
|--------|-------------|--------------|------------|
| Straddles | 231,192 | 231,192 | 1x (no redundancy) |
| Assets | 231,192 calls | **189 unique** | **1,223x redundant** |
| Asset+StrYM+Ntrc combos | 231,192 | ~13,894 | ~17x redundant |

---

## Finding 1: `loader.get_asset()` Function Call Overhead

Even though `get_asset()` has an internal dict cache, the function call overhead and dict.get() operations are repeated 231K times per run.

### Profile Results

```
231K get_asset calls: 3.060s
```

### Solution: Pre-compute Asset Properties

```python
# Before: 3.06s for 231K calls
for asset, straddle in straddle_list:
    asset_data = loader.get_asset(amt_path, asset)

# After: 0.03s total
# Build dict once for 189 unique assets
asset_to_data = {}
for asset_name in unique_asset_names:
    asset_to_data[asset_name] = loader.get_asset(amt_path, asset_name)

# Then O(1) dict lookups
for asset, straddle in straddle_list:
    asset_data = asset_to_data[asset]
```

**Savings: ~3s (100x faster)**

---

## Finding 2: `Path.resolve()` is Extremely Expensive

The `chain.fut_act2norm()` function calls `str(Path(csv_path).resolve())` on every lookup, even though the path is already cached.

### Profile Results

```python
# 500K Path().resolve() calls: 6.975s
# 500K dict.get() calls:       0.024s
```

`Path.resolve()` is **290x slower** than a dict lookup!

### Root Cause

```python
def fut_act2norm(csv_path: str | Path, ticker: str) -> str | None:
    csv_path = str(Path(csv_path).resolve())  # THIS LINE - called every time
    if csv_path not in _ACTUAL_CACHE:
        # Load CSV once...
    return _ACTUAL_CACHE[csv_path].get(ticker)
```

With 462K calls (231K straddles × 2 params), this accounts for ~7s.

### Solution: Cache Resolved Path

**Fixed at the source in `chain.py`:**

```python
# Cache for resolved paths (Path.resolve() is expensive - ~14μs per call)
_PATH_RESOLVE_CACHE: dict[str, str] = {}

def _resolve_path(csv_path: str | Path) -> str:
    """Resolve path with caching to avoid expensive Path.resolve() calls."""
    key = str(csv_path)
    if key not in _PATH_RESOLVE_CACHE:
        _PATH_RESOLVE_CACHE[key] = str(Path(csv_path).resolve())
    return _PATH_RESOLVE_CACHE[key]

def fut_act2norm(csv_path: str | Path, ticker: str) -> str | None:
    csv_path = _resolve_path(csv_path)  # Was: str(Path(csv_path).resolve())
    ...
```

This fix benefits **all callers** of `fut_act2norm()` and `fut_norm2act()`, not just the backtest code path.

**Additionally, in `_prepare_backtest_arrays_sorted()`**, ticker normalizations are cached locally:

```python
# Pre-cache ticker normalizations to avoid repeated lookups
ticker_norm_cache: dict[str, str | None] = {}

def _normalize_ticker(ticker: str) -> str:
    if ticker not in ticker_norm_cache:
        ticker_norm_cache[ticker] = chain._ACTUAL_CACHE[chain_csv_resolved].get(ticker)
    return ticker_norm_cache[ticker] if ticker_norm_cache[ticker] else ticker
```

**Savings: ~7s**

---

## Finding 3: Python Date Expansion Loop

The original implementation used a Python loop with `straddle_days()` and millions of `list.append()` calls.

### Profile Results

```
Old approach (Phase 2 - date expansion): 2.030s
16,010,256 days expanded via Python loops
```

### Solution: Use `find_straddle_days_u8m()`

A Numba kernel `expand_months_to_date32()` already exists that:
- Uses vectorized operations
- Returns u8m matrices + date32 array directly
- No Python string objects created
- Uses `parent_idx` array for efficient row expansion

```python
# Before: 2.03s
for asset, straddle in straddle_list:
    days = schedules.straddle_days(straddle)
    for dt in days:
        expanded_dates.append((dt - epoch).days)

# After: ~0.05s for the kernel portion
straddle_days_u8m = find_straddle_days_u8m(amt_path, start_year, end_year, pattern)
dates = straddle_days_u8m["rows"][2]  # Already int32 array
```

**Savings: ~1.3s (combined with Phase 1)**

---

## Finding 4: u8m String Padding Requires Stripping

The u8m format pads strings to fixed width:
- Asset: `'LA Comdty     '` (14 chars, space-padded)
- Original: `'LA Comdty'` (no padding)

This causes `loader.get_asset()` lookups to fail because the key doesn't match.

### Solution

```python
# Strip padding from assets (required for loader lookups)
straddle_list = [(asset.strip(), straddle) for asset, straddle in zip(...)]
```

**Note:** Straddle strings don't need stripping - the parsing functions (e.g., `schedules.xpry()`) work correctly with padding because they parse by position.

---

## Finding 5: Price Loading Performance

### PyArrow vs DuckDB

| Loader | Time | Notes |
|--------|------|-------|
| DuckDB Matrix (`load_prices_matrix`) | 6.2s | Baseline |
| PyArrow Sorted (`load_prices_numba`) | 0.42s | **15x faster** |

### Pre-sorted Parquet Benefits

| Scenario | Time | Improvement |
|----------|------|-------------|
| Unsorted parquet | 0.60s | baseline |
| Sorted parquet | 0.42s | **30% faster** |

Pre-sorted parquet allows skipping `np.argsort()` on 8.5M elements.

### Dictionary Encoding Gotcha

PyArrow's `dictionary_encode()` assigns indices based on **order of first appearance**, not alphabetical order. For composite keys to be monotonic with sorted parquet, indices must be remapped to alphabetical order:

```python
# Remap to alphabetical order
ticker_alpha_order = np.argsort(ticker_strings)
ticker_remap = np.empty(len(ticker_alpha_order), dtype=np.int32)
ticker_remap[ticker_alpha_order] = np.arange(len(ticker_alpha_order), dtype=np.int32)
ticker_idx = ticker_remap[ticker_idx]
```

---

## Optimization Summary

### Implemented Optimizations

| Optimization | Savings | Implementation |
|--------------|---------|----------------|
| `find_straddle_days_u8m()` | ~1.3s | Replace Python loop with Numba kernel |
| Pre-compute asset properties | ~3s | Build dict for 189 unique assets |
| Cache resolved chain path | ~7s | Avoid 462K `Path.resolve()` calls |
| Cache ticker normalizations | (included above) | Dict lookup instead of function call |

### Total Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Backtest time (231K straddles) | 19.4s | 8.9s | **54% faster** |
| Straddles/sec | 11,900 | 26,100 | **2.2x faster** |

---

## Remaining Bottlenecks

| Phase | Time | % of Total | Notes |
|-------|------|------------|-------|
| Phase 1+2 - Straddle expansion | 1.27s | 14% | Numba kernel |
| Phase 3 - `_batch_resolve_tickers()` | 3.46s | 39% | **Largest remaining** |
| Phase 3b - Prepare arrays | 1.41s | 16% | Optimized |
| Phase 4 - Kernel | 0.25s | 3% | Fast |
| Other (output, etc.) | ~2.5s | 28% | Arrow conversion, pandas |

### Potential Future Optimizations

1. **`_batch_resolve_tickers()`** - Could pre-compute ticker mappings per (asset, strym, ntrc) combination
2. **Output conversion** - Could return Arrow table directly instead of converting to pandas
3. **Parallel processing** - The kernel supports `parallel=True` for very large datasets

---

## Profiling Code

To reproduce these measurements:

```python
import time
from specparser.amt import schedules, loader
from specparser.amt.prices import load_prices_numba, get_prices_numba
from specparser.amt.valuation import _batch_resolve_tickers, _prepare_backtest_arrays_sorted

# Load prices
load_prices_numba('data/prices_sorted.parquet')
prices_numba = get_prices_numba()

# Phase 1+2
t0 = time.perf_counter()
straddle_days_table = schedules.find_straddle_days_u8m('data/amt.yml', 2001, 2026, '.')
print(f'Phase 1+2: {time.perf_counter() - t0:.3f}s')

# Phase 3
t0 = time.perf_counter()
ticker_map = _batch_resolve_tickers(assets, stryms, ntrcs, 'data/amt.yml')
print(f'Phase 3: {time.perf_counter() - t0:.3f}s')

# Phase 3b
t0 = time.perf_counter()
backtest_arrays = _prepare_backtest_arrays_sorted(...)
print(f'Phase 3b: {time.perf_counter() - t0:.3f}s')
```

---

## Files Modified

| File | Changes |
|------|---------|
| [schedules.py](../src/specparser/amt/schedules.py) | Added `parent_idx` return from `find_straddle_days_u8m()` |
| [valuation.py](../src/specparser/amt/valuation.py) | Pre-compute asset properties, cache ticker normalization, use `find_straddle_days_u8m()` |
| [chain.py](../src/specparser/amt/chain.py) | Added `_resolve_path()` cache to avoid expensive `Path.resolve()` calls |
