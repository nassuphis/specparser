# Plan: Sorted Parquet Support for PyArrow Loader

## Architecture Overview

There are **two completely separate code paths** for price loading and backtest computation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Code Path 1: BASELINE                               │
│                         (backtest.py, price_lookup='numba')                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  load_prices_matrix()          full_backtest_kernel()                       │
│  ┌─────────────────────┐       ┌─────────────────────────────────────┐     │
│  │ DuckDB query        │       │ Matrix indexing: price_matrix[r,c]  │     │
│  │ Python dict/loops   │  ───► │ Direct array access                 │     │
│  │ Build 2D matrix     │       │ O(1) per lookup                     │     │
│  └─────────────────────┘       └─────────────────────────────────────┘     │
│                                                                             │
│  Time: 6.2s load                Time: fast (Numba JIT)                      │
│  THIS IS THE BASELINE - DO NOT OPTIMIZE                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Code Path 2: FAST PATH                              │
│                         (backtest_new.py, price_lookup='numba_sorted_kernel')│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  load_prices_numba()           full_backtest_kernel_sorted()                │
│  ┌─────────────────────┐       ┌─────────────────────────────────────┐     │
│  │ PyArrow read        │       │ Binary search on sorted arrays      │     │
│  │ Dictionary encode   │  ───► │ lookup_price_sorted() per lookup    │     │
│  │ Build sorted arrays │       │ O(log n) per lookup                 │     │
│  │ np.argsort()        │       └─────────────────────────────────────┘     │
│  └─────────────────────┘                                                    │
│                                                                             │
│  Time: 0.45s load               Time: fast (Numba JIT)                      │
│  THIS IS WHAT WE OPTIMIZE                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Point:** The matrix loader (`load_prices_matrix`) is NOT used by `numba_sorted_kernel`.
They are independent implementations. We keep the DuckDB path as the 6.2s baseline for comparison.

---

## Executive Summary

This plan addresses one goal:
**Support pre-sorted `prices_sorted.parquet` to skip sorting during PyArrow load**

**Current Performance:**
| Script | Loader | Time | Notes |
|--------|--------|------|-------|
| backtest.py | DuckDB Matrix | 6.2s | Baseline - unchanged |
| backtest_new.py | PyArrow Sorted | 0.45s | Target for optimization |

**Target:** PyArrow loader under 0.35s with pre-sorted parquet (skip np.argsort).

---

## Pre-Sorted Parquet Support

### Problem

The sorted array loader (`load_prices_numba`) currently performs these steps (full 8.5M row file):

| Step | Operation | Time | Notes |
|------|-----------|------|-------|
| 1 | `pq.read_table()` | ~0.23s | Load parquet into Arrow table |
| 2 | Date filter (optional) | ~0.10s | Filter rows by date range |
| 3 | `pc.dictionary_encode()` | ~0.14s | Convert ticker/field strings to integer indices |
| 4 | `.to_numpy()` extraction | ~0.05s | Extract ticker_idx, field_idx, date, value arrays |
| 5 | Alphabetical remap | ~0.02s | Remap dict indices to alphabetical order |
| 6 | Date conversion | ~0.01s | Convert dates to int32 offsets |
| 7 | Build composite keys | ~0.02s | `(ticker_idx * n_fields + field_idx) * n_dates + date_offset` |
| 8 | Check if sorted | ~0.002s | O(n) monotonicity check |
| 9 | **`np.argsort()` + reorder** | **~0.17s** | Sort 8.5M composite keys (skipped if pre-sorted) |
| **Total (unsorted)** | | **~0.74s** | |
| **Total (pre-sorted)** | | **~0.57s** | Skips step 9 |

If parquet is pre-sorted by `(ticker, field, date)`, step 9 is skipped (saving ~0.17s, ~23%).

### Challenge: Dictionary Encoding

PyArrow's `dictionary_encode()` assigns indices based on **order of first appearance**, not alphabetical order. So even with sorted parquet:
- `AAPL US Equity` (first ticker) gets index 0
- `ABBV US Equity` (second ticker) gets index 1
- etc.

This means ticker indices ARE monotonically non-decreasing, but the composite key calculation assumes indices correspond to sorted string order for the key to be monotonic.

### Solution: Detect and Use Physical Sort Order

When parquet is sorted by `(ticker, field, date)`:
1. Ticker indices are monotonically non-decreasing ✓
2. Within each ticker, field indices are monotonically non-decreasing ✓
3. Within each ticker+field, dates are strictly increasing ✓

**Approach A: Check if composite key is already monotonic**
```python
composite = (ticker_idx * n_fields + field_idx) * n_dates + date_offset
is_sorted = np.all(composite[:-1] <= composite[1:])  # 0.002s check
if is_sorted:
    sorted_keys = composite
    sorted_values = value_arr
else:
    sort_idx = np.argsort(composite)
    sorted_keys = composite[sort_idx]
    sorted_values = value_arr[sort_idx]
```

**Approach B: Re-sort dictionary indices alphabetically**
```python
# Get alphabetically sorted order for ticker dictionary
ticker_strings = ticker_dict.dictionary.to_pylist()
alpha_order = np.argsort(ticker_strings)
ticker_remap = np.empty(len(alpha_order), dtype=np.int32)
ticker_remap[alpha_order] = np.arange(len(alpha_order))
ticker_idx_sorted = ticker_remap[ticker_idx]
# Now composite key will be monotonic if parquet was sorted
```

**Recommendation:** Use Approach A (simpler, 0.002s overhead for check).

### Implementation Steps

1. **Create `prices_sorted.parquet`**
   ```python
   import pyarrow.parquet as pq
   import pyarrow.compute as pc

   table = pq.read_table('data/prices.parquet')
   sort_indices = pc.sort_indices(table, sort_keys=[
       ('ticker', 'ascending'),
       ('field', 'ascending'),
       ('date', 'ascending')
   ])
   sorted_table = table.take(sort_indices)
   pq.write_table(sorted_table, 'data/prices_sorted.parquet')
   ```

2. **Update `load_prices_numba()` to detect sorted data**
   - Add quick monotonic check on composite key
   - Skip argsort/reorder if already sorted
   - Expected savings: 0.135s (23% of load time)

3. **Add script to generate sorted parquet**
   - `scripts/sort_prices.py` - one-time conversion tool

---

## Implementation Plan

### Phase 1: Create Sorted Parquet Tool

**File:** `scripts/sort_prices.py`

```python
#!/usr/bin/env python
"""Sort prices.parquet by (ticker, field, date) for faster loading."""

import argparse
import pyarrow.parquet as pq
import pyarrow.compute as pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input parquet file')
    parser.add_argument('output', help='Output sorted parquet file')
    args = parser.parse_args()

    print(f'Loading {args.input}...')
    table = pq.read_table(args.input)
    print(f'  Rows: {table.num_rows:,}')

    print('Sorting by (ticker, field, date)...')
    sort_indices = pc.sort_indices(table, sort_keys=[
        ('ticker', 'ascending'),
        ('field', 'ascending'),
        ('date', 'ascending')
    ])
    sorted_table = table.take(sort_indices)

    print(f'Writing {args.output}...')
    pq.write_table(sorted_table, args.output)
    print('Done.')

if __name__ == '__main__':
    main()
```

### Phase 2: Update `load_prices_numba()` for Sorted Detection

**File:** `src/specparser/amt/prices.py`

Add sorted detection after building composite keys:

```python
# Check if already sorted (cheap check)
is_sorted = len(composite_key) <= 1 or np.all(composite_key[:-1] <= composite_key[1:])

if is_sorted:
    sorted_keys = composite_key
    sorted_values = value_arr
else:
    sort_idx = np.argsort(composite_key)
    sorted_keys = composite_key[sort_idx]
    sorted_values = value_arr[sort_idx]
```

---

## Actual Results (Implemented)

### PyArrow Sorted Loader (`load_prices_numba`)

| Scenario | Date Filter | Time | Improvement |
|----------|-------------|------|-------------|
| Unsorted parquet | 2021-2025 | 0.20s | baseline |
| Sorted parquet | 2021-2025 | 0.14s | **30% faster** |
| Unsorted parquet | Full file | 0.60s | baseline |
| Sorted parquet | Full file | ~0.43s | **28% faster** |

Note: Full file is 8.5M rows. Date-filtered (2001-2026) loads most rows.

### DuckDB Matrix Loader (`load_prices_matrix`)

| Scenario | Time | Notes |
|----------|------|-------|
| Current (DuckDB + Python loops) | 6.2s | **BASELINE - NO CHANGES** |

The DuckDB matrix loader remains as the baseline for performance comparison.

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/sort_prices.py` | NEW - tool to create sorted parquet |
| `src/specparser/amt/prices.py` | Update `load_prices_numba()` with sorted detection |

---

## Verification

```bash
# 1. Create sorted parquet
uv run python scripts/sort_prices.py data/prices.parquet data/prices_sorted.parquet

# 2. Benchmark sorted vs unsorted
uv run python scripts/backtest_new.py '^LA Comdty' 2022 2024 --benchmark --prices data/prices.parquet
uv run python scripts/backtest_new.py '^LA Comdty' 2022 2024 --benchmark --prices data/prices_sorted.parquet

# 3. Benchmark optimized matrix loader
uv run python scripts/backtest.py '^LA Comdty' 2022 2024 --benchmark

# 4. Verify identical output
uv run python scripts/backtest.py '.' 2022 2024 > /tmp/old.tsv
uv run python scripts/backtest_new.py '.' 2022 2024 > /tmp/new.tsv
diff /tmp/old.tsv /tmp/new.tsv

# 5. Run tests
uv run pytest tests/ -v
```

---

## Summary

| Optimization | Effort | Impact |
|--------------|--------|--------|
| Sorted parquet detection | Low | ~0.06s savings (30% of PyArrow load) |

**Actual improvement (implemented):**
- PyArrow sorted loader: 0.20s → 0.14s with pre-sorted parquet (30% faster)
- DuckDB matrix loader: 6.2s (unchanged - baseline)

The PyArrow loader is already 14x faster than the DuckDB baseline. Pre-sorted parquet
provides an additional 30% improvement by skipping `np.argsort()` and array reordering.

### Implementation Details

The optimization required remapping dictionary indices to alphabetical order because
PyArrow's `dictionary_encode()` assigns indices by order of first appearance, not
alphabetically. With alphabetical remapping (~0.02s overhead), the composite key
becomes monotonic for sorted parquet, allowing us to skip:
- `np.argsort()` on 8.5M elements (~0.11s)
- Array reordering (~0.02s)

Net savings: ~0.11s for full file, ~0.06s for date-filtered load.

---

## Why Not Optimize the Matrix Loader?

The matrix loader (`load_prices_matrix` + `full_backtest_kernel`) serves as the **baseline**
for performance comparison. It uses DuckDB and Python loops, which are slower than the
PyArrow + Numba approach.

While we *could* optimize it (PyArrow + array lookup would get it to ~0.5s), that would:
1. Blur the comparison between the two approaches
2. Add maintenance burden for code that's not the primary path
3. Remove the reference point for measuring the benefit of the sorted kernel approach

The recommended path forward is `price_lookup='numba_sorted_kernel'` with `load_prices_numba()`.
